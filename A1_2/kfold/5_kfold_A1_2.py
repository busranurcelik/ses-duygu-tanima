"""
A1.2 / adim 5: A0 ile ayni 10-fold group-aware CV + OOF (RF ve MLP).
Girdi: 10 metadata + 184 A0 ozelligi + 8 F0 = 192 ozellik (23268 satir).

Kurallar (A0/A3 ile birebir):
  * Fold atamalari manifest_v2_dedup_draft.csv'den; REVIEW_LABEL kayitlari cikarilir.
  * Test = fold k & aug_id == 'orig'; train = diger 9 fold'un TUM satirlari
    (orig + o fold'un mevcut aug varyantlari). Test fold'undan aug girmez.
  * Scaler her fold'da yalnizca o fold'un train satirlarina fit edilir; RF ve MLP'ye uygulanir.
  * Seed'ler 43..52 (SEED=42+k); RF 200 agac/class_weight balanced;
    MLP 25 epoch, batch 32, 256-128-64 BN+Dropout, lr=0.001, label_smoothing=0.03,
    class_weight balanced + happy*1.25 + sad*1.15.
  * MODEL AGIRLIGI KAYDEDILMEZ. McNemar / ROC-AUC / OOF olasiligi CALISTIRILMAZ.

A1.2'ye ozgu tek kural:
  * Fallback satirlarinda ilk 7 F0 kolonu (184..190) 0.0 sentinel tasir.
    Bu 7 kolon icin imputasyon DEGERI her fold'da yalnizca o fold'un train
    satirlari icindki fallback'siz orneklerden (median) ogrenilir;
    train ve test'teki sentinel'ler bu medyanla degistirilir.
    191 (voiced_frame_ratio) gercek 0 degeridir, impute edilmez.
  * Tum imputasyon medyanlari ve sayimlari f0_imputasyon_degerleri.csv'ye yazilir.

Ciktilar YALNIZCA A1_2/kfold/ altinda. A0/A1.1/A3/sonuclar_v2_dedup/ yazilmaz.
70/15/15 ve commit/push YOK.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

SCRIPT_DIR = Path(__file__).resolve().parent            # A1_2/kfold
A12_DIR = SCRIPT_DIR.parent                             # A1_2
ROOT = A12_DIR.parent                                   # ses_duygu_projesi
FEATURE_FILE = A12_DIR / "final_oznitelik_seti_A1_2.csv"
DETAIL_FILE = A12_DIR / "f0_istatistikleri.csv"
MANIFEST_FILE = ROOT / "manifest_v2_dedup_draft.csv"
A0_OOF_FILE = ROOT / "sonuclar_v2_dedup" / "kfold" / "oof_tahminler.csv"
OUT_DIR = SCRIPT_DIR

FEATURE_COLS = [str(i) for i in range(192)]
A0_FEATURE_COLS = [str(i) for i in range(184)]
F0_SENTINEL_COLS = [str(i) for i in range(184, 191)]  # 7 kolon: mean..slope
F0_VOICED_COL = "191"                                 # gercek oran; impute edilmez
METADATA_COLS = ["source_id", "group_id", "source", "label", "aug_id",
                 "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]
CLASSES = ["angry", "fear", "happy", "sad"]
FOLDS = list(range(1, 11))
SEED = 42
EPOCHS = 25
BATCH_SIZE = 32
RF_TREES = 200
EXPECTED_CANONICAL = 7780
EXPECTED_TOTAL_COLS = 202
SENTINEL_IDX = np.arange(184, 191)

GUARD_FILES = ["oof_tahminler.csv", "kfold_per_fold.csv", "karsilastirma_A0_A1_2.md"]
OUT_DIR.mkdir(parents=True, exist_ok=True)
for name in GUARD_FILES:
    if (OUT_DIR / name).exists():
        raise SystemExit(f"REFUSE OVERWRITE: {OUT_DIR / name}")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    lines = ["| " + " | ".join(str(c) for c in d.columns) + " |",
             "|" + "|".join(["---"] * len(d.columns)) + "|"]
    for row in d.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def check_inputs(man: pd.DataFrame, feat: pd.DataFrame) -> None:
    expected = set(METADATA_COLS + FEATURE_COLS)
    if set(feat.columns) != expected:
        missing = sorted(expected - set(feat.columns), key=str)
        extra = sorted(set(feat.columns) - expected, key=str)
        raise RuntimeError(f"Beklenen 202 kolonluk shema bozuldu. Eksik={missing}, fazla={extra}")
    if feat.shape[1] != EXPECTED_TOTAL_COLS:
        raise AssertionError(f"{EXPECTED_TOTAL_COLS} kolon beklenirken {feat.shape[1]} bulundu.")
    if len(man) != EXPECTED_CANONICAL or man.source_id.nunique() != EXPECTED_CANONICAL:
        raise AssertionError("Manifest 7780 benzersiz kayit bekliyor.")
    if not man.fold.isin(FOLDS).all():
        raise AssertionError("Manifest fold degerleri 1..10 disinda.")
    bad = man[~man.apply(lambda r: r.group_id.startswith(r["source"] + "_"), axis=1)]
    if len(bad):
        raise AssertionError(f"{len(bad)} group_id kaynak on ekini tasimiyor.")
    if (man.groupby("group_id")["source"].nunique() > 1).any():
        raise AssertionError("Bir group_id birden fazla kaynaga ait.")
    if (man.groupby("group_id").fold.nunique() > 1).any():
        raise AssertionError("Bir group_id birden fazla fold'a dagildi.")
    if (man.groupby("source_id").fold.nunique() > 1).any():
        raise AssertionError("Bir source_id birden fazla fold'a dagildi.")
    if set(man.source_id) != set(feat.source_id):
        raise AssertionError("Manifest ve feature CSV source_id kumeleri farkli.")
    fold_map = dict(zip(man.source_id, man.fold.astype(int)))
    mapped = feat.source_id.map(fold_map)
    if mapped.isna().any() or not (mapped.to_numpy() == pd.to_numeric(feat.fold).to_numpy()).all():
        raise AssertionError("Feature/manifest fold degerleri birebir eslesmiyor.")
    feat["fold"] = mapped.astype(int)   # A3 ile ayni: fold int, OOF/A0 karsilastirmasi icin
    if not np.isfinite(feat[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise AssertionError("Feature matrisinde NaN/Inf var.")
    print(f"Girdi dogrulandi: {len(man)} kayit, {man.group_id.nunique()} grup, "
          f"{len(FEATURE_COLS)} ozellik, {feat.shape[0]} feature satiri", flush=True)


def attach_fallback(feat: pd.DataFrame) -> pd.DataFrame:
    det = pd.read_csv(DETAIL_FILE, dtype={"source_id": str, "aug_id": str})
    det["fallback"] = det.fallback.astype(int)
    if det.duplicated(["source_id", "aug_id"]).any():
        raise AssertionError("f0_istatistikleri.csv'de tekrarlayan source_id/aug_id.")
    fmap = {(sid, aug): int(fb) for sid, aug, fb in
            zip(det.source_id, det.aug_id, det.fallback)}
    feat = feat.copy()
    feat["f0_fallback"] = [fmap.get((sid, aug), -1)
                           for sid, aug in zip(feat.source_id, feat.aug_id)]
    if (feat.f0_fallback < 0).any():
        raise AssertionError("Baz feature satirlari icin F0 fallback bayragi yok.")
    x = feat[F0_SENTINEL_COLS].to_numpy(dtype=float)
    fb = feat.f0_fallback.to_numpy(dtype=int)
    if not np.allclose(x[fb == 1], 0.0, atol=0.0):
        raise AssertionError("Fallback satirlarinda 184..190 sentinel 0.0 degil.")
    # Not: fallback'siz satirlarda f0_std/f0_range 0.0 ve f0_slope negatif
    # olabilir (sabit F0 / dusen egilim); bunlar mesrudur, sentinel degildir.
    # Sentinel ayirimi yalnizca f0_istatistikleri.csv fallback bayragiyla yapilir.
    print(f"F0 fallback bayragi eklendi: {int(fb.sum())} satir sentinel 0.0", flush=True)
    return feat


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    f1c = f1_score(y_true, y_pred, labels=range(len(CLASSES)),
                   average=None, zero_division=0)
    for i, c in enumerate(CLASSES):
        out[f"f1_{c}"] = float(f1c[i])
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fi, hi = CLASSES.index("fear"), CLASSES.index("happy")
    out["fear_to_happy"] = int(cm[fi, hi])
    out["happy_to_fear"] = int(cm[hi, fi])
    out["fear_to_happy_rate"] = float(cm[fi, hi] / max(int(cm[fi].sum()), 1))
    out["happy_to_fear_rate"] = float(cm[hi, fi] / max(int(cm[hi].sum()), 1))
    return out


def build_fold_data(feat: pd.DataFrame, le: LabelEncoder, k: int) -> dict:
    test = feat[(feat.fold == k) & (feat.aug_id == "orig")].reset_index(drop=True)
    train = feat[feat.fold != k].reset_index(drop=True)
    if len(test) == 0 or len(train) == 0:
        raise AssertionError(f"fold {k}: train/test bos.")
    if (test.aug_id != "orig").any():
        raise AssertionError(f"fold {k}: test icinde augmentation var.")
    if set(train.source_id) & set(test.source_id):
        raise AssertionError(f"fold {k}: source_id sizintisi.")
    if set(train.group_id) & set(test.group_id):
        raise AssertionError(f"fold {k}: grup sizintisi.")

    Xtr = train[FEATURE_COLS].to_numpy(dtype=float)
    Xte = test[FEATURE_COLS].to_numpy(dtype=float)
    fb_tr = train.f0_fallback.to_numpy(dtype=int)
    fb_te = test.f0_fallback.to_numpy(dtype=int)

    # F0 imputasyonu: medyan YALNIZCA bu fold train'inin fallback'siz satirlarindan.
    ok = fb_tr == 0
    if int(ok.sum()) < 3:
        raise AssertionError(f"fold {k}: imputasyon icin yeterli train ornegi yok.")
    med = np.median(Xtr[ok][:, SENTINEL_IDX], axis=0)
    # f0_slope medyani negatif olabilir (dusen egilim); yalnizca finite zorunludur.
    if not np.isfinite(med).all():
        raise AssertionError(f"fold {k}: imputasyon medyani gecersiz.")
    n_tr_imp, n_te_imp = int((fb_tr == 1).sum()), int((fb_te == 1).sum())
    Xtr[:, SENTINEL_IDX] = np.where(fb_tr[:, None] == 1, med, Xtr[:, SENTINEL_IDX])
    Xte[:, SENTINEL_IDX] = np.where(fb_te[:, None] == 1, med, Xte[:, SENTINEL_IDX])
    if not (np.isfinite(Xtr[:, SENTINEL_IDX]).all() and
            np.isfinite(Xte[:, SENTINEL_IDX]).all()):
        raise AssertionError(f"fold {k}: imputasyon sonrasi NaN/Inf var.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(Xtr)
    X_test = scaler.transform(Xte)
    y_train = le.transform(train.labels.to_numpy())
    y_test = le.transform(test.labels.to_numpy())
    if set(np.unique(y_train)) != set(range(len(CLASSES))):
        raise AssertionError(f"fold {k}: train icinde eksik sinif.")

    test_groups = sorted(test.group_id.unique())
    tess_test = [g.replace("TESS_", "") for g in test_groups if g.startswith("TESS_")] or ["-"]
    return {
        "k": k, "test": test, "train": train,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "n_train_rows": len(train), "n_train_orig": int((train.aug_id == "orig").sum()),
        "n_train_aug": int((train.aug_id != "orig").sum()), "n_test_rows": len(test),
        "train_aug_ids": ",".join(sorted(set(train.loc[train.aug_id != "orig", "aug_id"]))),
        "test_sources": ",".join(sorted(test.source.unique())),
        "tess_test_speakers": ",".join(sorted(tess_test)),
        "bnc_in_test": "yes" if "BNC_SPEAKER_01" in test_groups else "no",
        "imputation": {"medians": med, "n_train_nonfallback": int(ok.sum()),
                       "n_train_imputed": n_tr_imp, "n_test_imputed": n_te_imp},
    }


def build_mlp(seed: int) -> Sequential:
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    model = Sequential([
        Input(shape=(len(FEATURE_COLS),)),
        Dense(256, kernel_initializer="he_normal"), BatchNormalization(),
        tf.keras.layers.Activation("relu"), Dropout(0.40),
        Dense(128, kernel_initializer="he_normal"), BatchNormalization(),
        tf.keras.layers.Activation("relu"), Dropout(0.35),
        Dense(64, kernel_initializer="he_normal"), BatchNormalization(),
        tf.keras.layers.Activation("relu"), Dropout(0.30),
        Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
        metrics=["accuracy"],
    )
    return model


def class_weights(y_train: np.ndarray) -> dict:
    arr = compute_class_weight(class_weight="balanced",
                               classes=np.arange(len(CLASSES)), y=y_train)
    w = {i: float(v) for i, v in enumerate(arr)}
    w[CLASSES.index("happy")] *= 1.25
    w[CLASSES.index("sad")] *= 1.15
    return w


def save_checks(checks: list[dict]) -> None:
    df = pd.DataFrame(checks)
    df.to_csv(OUT_DIR / "sizdirma_kontrolleri.csv", index=False, encoding="utf-8")
    lines = ["# A1.2 10-fold OOF — sizdirma ve protokol kontrolleri", "",
             "| kapsam | kontrol | sonuc | detay |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['scope']} | {r['check']} | {r['result']} | {r['detail']} |")
    lines += ["",
              "- REVIEW_LABEL kayitlari manifestten cikarildi; 7780 canonical kayit dogrulandi.",
              "- Feature CSV fold degerleri manifest fold degerleriyle birebir eslesiyor.",
              "- Test = `fold == k & aug_id == 'orig'`; train = diger 9 fold (orig + aug).",
              "- Scaler her fold'da yalnizca o fold train'ine fit edildi (RF ve MLP).",
              "- F0 imputasyon medyani her fold'da yalnizca o fold train'inin "
              "fallback'siz satirlarindan ogrenildi; test'ten hicbir deger kullanilmadi.",
              "- 191 (voiced_frame_ratio) gercek 0 deger tasir; impute edilmedi.",
              "- Model agirligi kaydedilmedi; McNemar/ROC/OOF olasiliklari calistirilmadi.",
              "- 70/15/15 calistirilmadi; commit/push yapilmadi.", ""]
    (OUT_DIR / "sizdirma_kontrolleri.md").write_text("\n".join(lines), encoding="utf-8")


def build_comparison(oof: pd.DataFrame, le: LabelEncoder) -> None:
    a0 = pd.read_csv(A0_OOF_FILE, dtype={"source_id": str, "group_id": str,
                                         "source": str, "fold": int,
                                         "y_true": str, "rf_pred": str, "mlp_pred": str})
    if len(a0) != EXPECTED_CANONICAL or set(a0.source_id) != set(oof.source_id):
        raise AssertionError("A0 OOF source_id kumesi A1.2 OOF ile ayni degil.")
    if not (a0.fold.to_numpy() == oof.set_index("source_id").loc[a0.source_id, "fold"].to_numpy()).all():
        raise AssertionError("A0/A1.2 fold atamalari farkli.")

    rows: list[dict] = []

    def add(section: str, model: str, key: str, a: float, b: float) -> None:
        rows.append({"bolum": section, "model": model, "metrik": key,
                     "A0": float(a), "A1_2": float(b), "fark": float(b - a)})

    for model, col in [("RF", "rf_pred"), ("MLP", "mlp_pred")]:
        ma = metrics(le.transform(a0.y_true.to_numpy()), le.transform(a0[col].to_numpy()))
        mb = metrics(le.transform(oof.y_true.to_numpy()), le.transform(oof[col].to_numpy()))
        for key in ["accuracy", "f1_macro", "f1_fear", "f1_happy",
                    "fear_to_happy", "happy_to_fear",
                    "fear_to_happy_rate", "happy_to_fear_rate"]:
            add("genel", model, key, ma[key], mb[key])

        # Kaynak bazli
        for src in sorted(oof.source.unique()):
            sa = a0[a0.source == src]
            sb = oof[oof.source == src]
            ma_s = metrics(le.transform(sa.y_true.to_numpy()), le.transform(sa[col].to_numpy()))
            mb_s = metrics(le.transform(sb.y_true.to_numpy()), le.transform(sb[col].to_numpy()))
            add("kaynak", f"{model}:{src}", "accuracy", ma_s["accuracy"], mb_s["accuracy"])
            add("kaynak", f"{model}:{src}", "f1_macro", ma_s["f1_macro"], mb_s["f1_macro"])

        # Fold bazli tutarlik
        for k in FOLDS:
            fa = a0[a0.fold == k]
            fb = oof[oof.fold == k]
            ma_f = metrics(le.transform(fa.y_true.to_numpy()), le.transform(fa[col].to_numpy()))
            mb_f = metrics(le.transform(fb.y_true.to_numpy()), le.transform(fb[col].to_numpy()))
            add("fold", f"{model}:fold_{k}", "accuracy", ma_f["accuracy"], mb_f["accuracy"])
            add("fold", f"{model}:fold_{k}", "f1_macro", ma_f["f1_macro"], mb_f["f1_macro"])
        # mean/std icin fold serilerini dogrudan yeniden hesapla
        fa_acc, fb_acc, fa_f1, fb_f1 = [], [], [], []
        for k in FOLDS:
            ma_f = metrics(le.transform(a0[a0.fold == k].y_true.to_numpy()),
                           le.transform(a0[a0.fold == k][col].to_numpy()))
            mb_f = metrics(le.transform(oof[oof.fold == k].y_true.to_numpy()),
                           le.transform(oof[oof.fold == k][col].to_numpy()))
            fa_acc.append(ma_f["accuracy"]); fb_acc.append(mb_f["accuracy"])
            fa_f1.append(ma_f["f1_macro"]); fb_f1.append(mb_f["f1_macro"])
        add("fold_ozet", model, "accuracy_mean", np.mean(fa_acc), np.mean(fb_acc))
        add("fold_ozet", model, "accuracy_std", np.std(fa_acc), np.std(fb_acc))
        add("fold_ozet", model, "f1_macro_mean", np.mean(fa_f1), np.mean(fb_f1))
        add("fold_ozet", model, "f1_macro_std", np.std(fa_f1), np.std(fb_f1))

    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT_DIR / "karsilastirma_A0_A1_2.csv", index=False, encoding="utf-8")

    def pick(section: str, model: str, metric: str) -> pd.DataFrame:
        return cmp[cmp.bolum.eq(section) & cmp.model.eq(model) & cmp.metrik.eq(metric)]

    lines = [
        "# A1.2 10-fold OOF — A0 ile karsilastirma",
        "",
        "A0 ve A1.2 ayni fold atamalari, ayni seed'ler (43..52), ayni RF/MLP ayarlari "
        "ve ayni degerlendirme protokolunu kullanir (test = ilgili fold'un `orig` satirlari). "
        "A1.2'de 184 A0 ozelligine 8 F0 ozelligi eklendi; fallback sentinel'leri her fold'da "
        "yalnizca o fold train'inden ogrenilen medyanla impute edildi.",
        "McNemar, ROC/AUC, olasilik kaydi, 70/15/15 ve model kaydi YAPILMADI.",
        "",
        "## 1. Genel performans ve fear/happy",
        "",
        "| model | metrik | A0 | A1.2 | fark (A1.2-A0) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in ["RF", "MLP"]:
        for key in ["accuracy", "f1_macro", "f1_fear", "f1_happy",
                    "fear_to_happy", "happy_to_fear",
                    "fear_to_happy_rate", "happy_to_fear_rate"]:
            r = pick("genel", model, key).iloc[0]
            lines.append(f"| {model} | {key} | {r['A0']:.4f} | {r['A1_2']:.4f} | {r['fark']:+.4f} |")
    lines += ["", "## 2. Kaynak bazli performans", "",
              "| model | kaynak | metrik | A0 | A1.2 | fark |", "|---|---|---|---:|---:|---:|"]
    for model in ["RF", "MLP"]:
        for src in ["BNC", "CREMA", "RAVDESS", "SAVEE", "TESS"]:
            for key in ["accuracy", "f1_macro"]:
                r = pick("kaynak", f"{model}:{src}", key).iloc[0]
                lines.append(f"| {model} | {src} | {key} | {r['A0']:.4f} | {r['A1_2']:.4f} | {r['fark']:+.4f} |")
    lines += ["", "## 3. Fold tutarligi", "",
              "| model | metrik | A0 | A1.2 | fark |", "|---|---|---:|---:|---:|"]
    for model in ["RF", "MLP"]:
        for key in ["accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std"]:
            r = pick("fold_ozet", model, key).iloc[0]
            lines.append(f"| {model} | {key} | {r['A0']:.4f} | {r['A1_2']:.4f} | {r['fark']:+.4f} |")
    lines += ["", "| model | fold | accuracy A0 | accuracy A1.2 | fark | F1 A0 | F1 A1.2 | fark |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in ["RF", "MLP"]:
        for k in FOLDS:
            ra = pick("fold", f"{model}:fold_{k}", "accuracy").iloc[0]
            rf1 = pick("fold", f"{model}:fold_{k}", "f1_macro").iloc[0]
            lines.append(f"| {model} | {k} | {ra['A0']:.4f} | {ra['A1_2']:.4f} | {ra['fark']:+.4f} | "
                         f"{rf1['A0']:.4f} | {rf1['A1_2']:.4f} | {rf1['fark']:+.4f} |")
    lines += ["",
              "Not: `fear_to_happy` / `happy_to_fear` sayilari OOF karisiklik matrisinden alinmistir; "
              "oranlar ilgili gercek sinif satir sayisina bolunmustur. "
              "Farklar betimseldir; istatistiksel anlamlilik testi yapilmadi.", ""]
    (OUT_DIR / "karsilastirma_A0_A1_2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=== A1.2 / 10-fold group-aware CV + OOF (RF + MLP, 192 ozellik) ===", flush=True)
    man = pd.read_csv(MANIFEST_FILE, dtype=str)
    if "dedup_status" not in man.columns:
        raise RuntimeError("Dedup manifesti dedup_status kolonunu icermiyor.")
    review_n = int(man.dedup_status.eq("REVIEW_LABEL").sum())
    if review_n != 2:
        raise AssertionError(f"REVIEW_LABEL beklenen 2, bulunan {review_n}.")
    man = man[man.dedup_status.ne("REVIEW_LABEL")].copy()
    man["fold"] = man.fold.astype(int)
    feat = pd.read_csv(FEATURE_FILE, dtype={c: str for c in METADATA_COLS})
    check_inputs(man, feat)
    feat = attach_fallback(feat)

    le = LabelEncoder()
    le.fit(CLASSES)
    checks = [
        {"scope": "global", "check": "REVIEW_LABEL dislama", "result": "PASS",
         "detail": f"{review_n} kayit cikarildi, {len(man)} canonical"},
        {"scope": "global", "check": "202 kolon / 192 ozellik", "result": "PASS",
         "detail": f"{feat.shape[1]-1} kolon + f0_fallback flag"},
        {"scope": "global", "check": "F0 sentinel dogrulama", "result": "PASS",
         "detail": f"{int(feat.f0_fallback.sum())} fallback satiri, 184..190 = 0.0"},
        {"scope": "global", "check": "fold/split/manifest birebir", "result": "PASS",
         "detail": f"{len(man)} kayit"},
    ]

    per_fold_rows: list[dict] = []
    oof_rows: list[dict] = []
    imp_rows: list[dict] = []

    for k in FOLDS:
        t0 = time.time()
        d = build_fold_data(feat, le, k)
        fold_seed = SEED + k
        imp = d["imputation"]
        for j, col in enumerate(F0_SENTINEL_COLS):
            imp_rows.append({"fold": k, "kolon": col,
                             "ozellik": ["f0_mean", "f0_median", "f0_std", "f0_min",
                                         "f0_max", "f0_range", "f0_slope"][j],
                             "train_medyan": float(imp["medians"][j]),
                             "n_train_fallbacksiz": imp["n_train_nonfallback"],
                             "n_train_impute": imp["n_train_imputed"],
                             "n_test_impute": imp["n_test_imputed"],
                             "kaynak": "yalnizca bu fold train'i"})

        rf = RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced",
                                    random_state=fold_seed, n_jobs=-1)
        t_rf = time.time()
        rf.fit(d["X_train"], d["y_train"])
        rf_seconds = round(time.time() - t_rf, 2)
        rf_pred = rf.predict(d["X_test"])

        tf.keras.backend.clear_session()
        mlp = build_mlp(fold_seed)
        ytr_cat = to_categorical(d["y_train"], num_classes=len(CLASSES))
        t_mlp = time.time()
        mlp.fit(d["X_train"], ytr_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                verbose=0, class_weight=class_weights(d["y_train"]))
        mlp_pred = np.argmax(mlp.predict(d["X_test"], verbose=0), axis=1)
        mlp_seconds = round(time.time() - t_mlp, 2)

        for model_name, pred, secs in [("RF", rf_pred, rf_seconds), ("MLP", mlp_pred, mlp_seconds)]:
            row = {"model": model_name, "fold": k}
            row.update(metrics(d["y_test"], pred))
            row.update({
                "n_train_rows": d["n_train_rows"], "n_train_orig": d["n_train_orig"],
                "n_train_aug": d["n_train_aug"], "n_test_rows": d["n_test_rows"],
                "train_aug_ids": d["train_aug_ids"], "test_sources": d["test_sources"],
                "tess_test_speakers": d["tess_test_speakers"],
                "bnc_in_test": d["bnc_in_test"], "seed": fold_seed, "seconds": secs,
                "n_train_imputed": imp["n_train_imputed"],
                "n_test_imputed": imp["n_test_imputed"],
            })
            per_fold_rows.append(row)

        test = d["test"]
        for i, sid in enumerate(test.source_id.to_numpy()):
            oof_rows.append({
                "source_id": sid, "group_id": test.group_id.iloc[i],
                "source": test.source.iloc[i], "fold": k,
                "y_true": test.labels.iloc[i],
                "rf_pred": CLASSES[rf_pred[i]], "mlp_pred": CLASSES[mlp_pred[i]],
                "rf_correct": bool(rf_pred[i] == d["y_test"][i]),
                "mlp_correct": bool(mlp_pred[i] == d["y_test"][i]),
                "f0_fallback": int(test.f0_fallback.iloc[i]),
            })

        checks += [
            {"scope": f"fold_{k}", "check": "test aug_id == orig", "result": "PASS",
             "detail": f"{d['n_test_rows']} satir, 0 aug"},
            {"scope": f"fold_{k}", "check": "train/test grup ve source kesismesi",
             "result": "PASS", "detail": "0 / 0"},
            {"scope": f"fold_{k}", "check": "aug satirlari yalnizca train'de",
             "result": "PASS", "detail": f"train_aug={d['n_train_aug']}, test_aug=0"},
            {"scope": f"fold_{k}", "check": "F0 imputasyon medyani train'den",
             "result": "PASS",
             "detail": f"n_train_fallbacksiz={imp['n_train_nonfallback']}, "
                       f"train_impute={imp['n_train_imputed']}, test_impute={imp['n_test_imputed']}"},
        ]
        rm = metrics(d["y_test"], rf_pred)
        mm = metrics(d["y_test"], mlp_pred)
        print(f"fold {k:>2}: n_test={d['n_test_rows']:>4} | "
              f"RF acc={rm['accuracy']:.4f} f1={rm['f1_macro']:.4f} | "
              f"MLP acc={mm['accuracy']:.4f} f1={mm['f1_macro']:.4f} | "
              f"impute train/test={imp['n_train_imputed']}/{imp['n_test_imputed']} | "
              f"{time.time()-t0:.1f}s", flush=True)

    per_fold = pd.DataFrame(per_fold_rows)
    oof = pd.DataFrame(oof_rows)

    if len(oof) != len(man) or not oof.source_id.is_unique:
        raise AssertionError("OOF satir sayisi/benzersizligi hatali.")
    if set(oof.source_id) != set(man.source_id):
        raise AssertionError("OOF source_id kumesi manifest ile ayni degil.")
    merged = oof.merge(man[["source_id", "fold"]], on="source_id", suffixes=("", "_manifest"))
    if not (merged.fold == merged.fold_manifest).all():
        raise AssertionError("OOF fold atamasi manifest ile uyusmuyor.")
    checks.append({"scope": "OOF", "check": "her kayit icin tek tahmin", "result": "PASS",
                   "detail": f"{len(oof)} satir, {oof.source_id.nunique()} benzersiz kayit"})

    # ---------------- Ciktilar ----------------
    per_fold.to_csv(OUT_DIR / "kfold_per_fold.csv", index=False, encoding="utf-8")
    pd.DataFrame(imp_rows).to_csv(OUT_DIR / "f0_imputasyon_degerleri.csv", index=False, encoding="utf-8")
    value_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                  "precision_weighted", "recall_weighted", "f1_weighted",
                  "f1_angry", "f1_fear", "f1_happy", "f1_sad"]
    ozet = per_fold.groupby("model")[value_cols].agg(["mean", "std"]).round(6)
    ozet.columns = [f"{a}_{b}" for a, b in ozet.columns]
    ozet = ozet.reset_index()
    ozet["n_folds"] = 10
    ozet.to_csv(OUT_DIR / "kfold_ozet.csv", index=False, encoding="utf-8")

    oof_metric_rows, source_rows, class_rows, cm_rows = [], [], [], []
    for model_name, pred_col in [("RF", "rf_pred"), ("MLP", "mlp_pred")]:
        yt = le.transform(oof.y_true.to_numpy())
        yp = le.transform(oof[pred_col].to_numpy())
        m = metrics(yt, yp)
        m.update({"model": model_name, "n_records": len(oof), "evaluation": "OOF_10fold"})
        oof_metric_rows.append(m)
        for src in sorted(oof.source.unique()):
            sub = oof[oof.source == src]
            sm = metrics(le.transform(sub.y_true.to_numpy()),
                         le.transform(sub[pred_col].to_numpy()))
            sm.update({"model": model_name, "source": src, "n_records": len(sub)})
            source_rows.append(sm)
        pr = precision_score(yt, yp, labels=range(4), average=None, zero_division=0)
        rc = recall_score(yt, yp, labels=range(4), average=None, zero_division=0)
        f1 = f1_score(yt, yp, labels=range(4), average=None, zero_division=0)
        for i, c in enumerate(CLASSES):
            class_rows.append({"model": model_name, "sinif": c,
                               "precision": float(pr[i]), "recall": float(rc[i]),
                               "f1": float(f1[i]),
                               "destek": int((oof.y_true == c).sum())})
        cm = confusion_matrix(yt, yp, labels=range(4))
        for i, t in enumerate(CLASSES):
            for j, p in enumerate(CLASSES):
                cm_rows.append({"model": model_name, "gercek": t, "tahmin": p,
                                "n": int(cm[i, j])})
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"OOF 10-fold - {model_name}")
        plt.xticks(range(4), CLASSES, rotation=45)
        plt.yticks(range(4), CLASSES)
        for i in range(4):
            for j in range(4):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"oof_cm_{model_name}.png", dpi=150)
        plt.close()

    pd.DataFrame(oof_metric_rows).to_csv(OUT_DIR / "oof_metrikler.csv", index=False, encoding="utf-8")
    pd.DataFrame(source_rows).to_csv(OUT_DIR / "oof_kaynak_bazli.csv", index=False, encoding="utf-8")
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "oof_sinif_f1.csv", index=False, encoding="utf-8")
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "oof_karisiklik.csv", index=False, encoding="utf-8")
    oof.to_csv(OUT_DIR / "oof_tahminler.csv", index=False, encoding="utf-8")
    save_checks(checks)

    config = {
        "evaluation": "A1_2_10fold_group_aware_OOF_RF_MLP",
        "experiment": "A1.2: A0 184 ozellik + 8 F0/pitch ozellik",
        "n_features": len(FEATURE_COLS),
        "n_a0_features": len(A0_FEATURE_COLS),
        "n_f0_features": 8,
        "manifest": str(MANIFEST_FILE.relative_to(ROOT)),
        "feature_file": str(FEATURE_FILE.relative_to(ROOT)),
        "detail_file": str(DETAIL_FILE.relative_to(ROOT)),
        "a0_oof_reference": str(A0_OOF_FILE.relative_to(ROOT)),
        "feature_sha256": sha256(FEATURE_FILE),
        "fold_source": "manifest_v2_dedup_draft.csv (REVIEW_LABEL excluded)",
        "seed_base": SEED,
        "fold_seeds": {str(k): SEED + k for k in FOLDS},
        "rf": {"n_estimators": RF_TREES, "class_weight": "balanced"},
        "mlp": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "early_stopping": False,
                "class_weight": "balanced + happy*1.25 + sad*1.15",
                "architecture": "256-128-64-4, BN + Dropout"},
        "augmentation": "yalnizca fold train kayitlarinin mevcut deterministic aug satirlari",
        "scaler": "her fold'da yalnizca o fold train'ine fit; RF ve MLP'ye uygulandi",
        "f0_imputation": {
            "columns": F0_SENTINEL_COLS,
            "sentinel": 0.0,
            "method": "fold train fallback'siz satirlarin median'i",
            "learned_from": "train only (her fold ayri)",
            "not_imputed": {"191": "voiced_frame_ratio gercek 0 degeridir"},
            "log": "f0_imputasyon_degerleri.csv",
        },
        "models_saved": False,
        "not_run": ["McNemar", "ROC/AUC", "70_15_15", "commit/push"],
    }
    with open(OUT_DIR / "cv_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    build_comparison(oof, le)

    print("\n=== Fold ortalamasi +/- std ===")
    print(ozet[["model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std",
                "f1_fear_mean", "f1_happy_mean"]].to_string(index=False))
    print("\n=== OOF metrikler ===")
    mdf = pd.DataFrame(oof_metric_rows)
    print(mdf[["model", "n_records", "accuracy", "f1_macro", "f1_fear", "f1_happy",
               "fear_to_happy", "happy_to_fear"]].to_string(index=False))
    print("\n=== OOF kaynak bazli ===")
    print(pd.DataFrame(source_rows)[["model", "source", "n_records", "accuracy", "f1_macro"]]
          .to_string(index=False))
    print("\nTum sizdirma kontrolleri PASS; McNemar/ROC/70-15-15/model kaydi CALISTIRILMADI.")


if __name__ == "__main__":
    main()
