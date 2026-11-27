"""
v2 / adim 5: 10-fold group-aware CV + OOF (RF ve MLP).

Kurallar:
  * Fold atamalari manifest_v2_dedup_draft.csv'den gelir; REVIEW_LABEL kayitlari cikarilir.
  * 7780 benzersiz source_id dogrulanir; CSV ve manifest fold degerleri birebir eslesir.
  * Augmentation satirlari YALNIZCA kaydın kendi fold'u train tarafindaysa kullanilir.
    Test fold'undan gelen hicbir aug satiri egitime girmez.
  * Scaler her fold'da yalnizca o fold'un train satirlarina fit edilir ve
    hem RF hem MLP icin kullanilir.
  * RF ve MLP ayni fold bolumleri uzerinde ayri ayri egitilir.
  * Fold bazli metrikler, fold ortalamasi +/- std ve birlesik OOF metrikleri ayri yazilir.
  * Kaynak bazli OOF accuracy / macro-F1 uretilir.
  * McNemar ve ROC/AUC bu scriptte CALISTIRILMAZ; OOF olasiliklari kaydedilmez.
  * Hicbir model agirligi kaydedilmez; ciktilar sonuclar_v2_dedup/kfold/ altindadir.
    Eski sonuclar_v2/ ASLA yazilmaz.
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

ROOT = Path(__file__).resolve().parent
FEATURE_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"
MANIFEST_FILE = ROOT / "manifest_v2_dedup_draft.csv"
OUT_DIR = ROOT / "sonuclar_v2_dedup" / "kfold"

FEATURE_COLS = [str(i) for i in range(184)]
METADATA_COLS = ["source_id", "group_id", "source", "label", "aug_id",
                 "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]
CLASSES = ["angry", "fear", "happy", "sad"]
FOLDS = list(range(1, 11))
SEED = 42
EPOCHS = 25          # sabit epoch; ic validation / early stopping kullanilmaz
BATCH_SIZE = 32
RF_TREES = 200
EXPECTED_CANONICAL = 7780
EXPECTED_TOTAL_COLS = 194

OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def check_schema(feat: pd.DataFrame) -> None:
    if len(FEATURE_COLS) != 184 or len(METADATA_COLS) != 10:
        raise AssertionError("184 ozellik + 10 metadata kolonu bekleniyor.")
    expected = set(FEATURE_COLS + METADATA_COLS)
    actual = set(feat.columns)
    if actual != expected:
        missing = sorted(expected - actual, key=str)
        extra = sorted(actual - expected, key=str)
        raise RuntimeError(
            f"Beklenen 194 kolonluk shema bozuldu. Eksik={missing}, fazla={extra}"
        )
    if feat.shape[1] != EXPECTED_TOTAL_COLS:
        raise AssertionError(
            f"{EXPECTED_TOTAL_COLS} kolon beklenirken {feat.shape[1]} kolon bulundu."
        )


def check_inputs(man: pd.DataFrame, feat: pd.DataFrame) -> None:
    check_schema(feat)
    required_m = {"source_id", "group_id", "source", "fold"}
    required_f = set(METADATA_COLS + FEATURE_COLS)
    if not required_m.issubset(man.columns):
        raise RuntimeError(f"Manifest eksik kolon: {required_m - set(man.columns)}")
    if not required_f.issubset(feat.columns):
        raise RuntimeError(f"Feature CSV eksik kolon: {required_f - set(feat.columns)}")

    if len(man) != EXPECTED_CANONICAL or man.source_id.nunique() != EXPECTED_CANONICAL:
        raise AssertionError(
            f"Manifest {EXPECTED_CANONICAL} benzersiz kayit bekliyor."
        )
    if not man.fold.isin(FOLDS).all():
        raise AssertionError("Manifest fold degerleri 1..10 disinda.")

    bad = man[~man.apply(lambda r: r.group_id.startswith(r["source"] + "_"), axis=1)]
    if len(bad):
        raise AssertionError(f"{len(bad)} group_id kaynak on ekini tasimiyor.")
    shared = man.groupby("group_id")["source"].nunique()
    if (shared > 1).any():
        raise AssertionError("Bir group_id birden fazla kaynaga ait.")
    if (man.groupby("group_id").fold.nunique() > 1).any():
        raise AssertionError("Bir group_id birden fazla fold'a dagildi.")
    if (man.groupby("source_id").fold.nunique() > 1).any():
        raise AssertionError("Bir source_id birden fazla fold'a dagildi.")

    if set(man.source_id) != set(feat.source_id):
        raise AssertionError("Manifest ve feature CSV source_id kumeleri farkli.")

    fold_map = dict(zip(man.source_id, man.fold.astype(int)))
    mapped_fold = feat.source_id.map(fold_map)
    if mapped_fold.isna().any():
        raise AssertionError("Bazı feature satirlarinin foldu bulunamadi.")
    csv_fold = pd.to_numeric(feat["fold"], errors="raise").astype(int)
    if not (csv_fold.to_numpy() == mapped_fold.to_numpy()).all():
        raise AssertionError(
            "Manifest ve feature CSV fold degerleri birebir eslesmiyor."
        )
    feat["fold"] = mapped_fold
    if (feat.groupby("source_id").fold.nunique() > 1).any():
        raise AssertionError("Aug satirlari source_id icin farkli fold tasimiyor.")

    if not np.isfinite(feat[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise AssertionError("Feature matrisinde NaN/Inf var.")

    print(f"Girdi dogrulandi: {len(man)} kayit, {man.group_id.nunique()} grup, "
          f"{len(FEATURE_COLS)} ozellik, fold={man.fold.nunique()}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


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
    if (train.groupby("source_id").fold.nunique() > 1).any():
        raise AssertionError(f"fold {k}: train icinde karisik fold.")

    Xtr_raw = train[FEATURE_COLS].to_numpy(dtype=float)
    Xte_raw = test[FEATURE_COLS].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(Xtr_raw)     # yalnizca bu fold train
    X_test = scaler.transform(Xte_raw)
    y_train = le.transform(train.labels.to_numpy())
    y_test = le.transform(test.labels.to_numpy())

    if set(np.unique(y_train)) != set(range(len(CLASSES))):
        raise AssertionError(f"fold {k}:train icinde eksik sinif.")

    test_groups = sorted(test.group_id.unique())
    tess_test = [g.replace("TESS_", "") for g in test_groups if g.startswith("TESS_")] or ["-"]
    bnc_in_test = "yes" if "BNC_SPEAKER_01" in test_groups else "no"

    return {
        "k": k,
        "test": test,
        "train": train,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_train_rows": len(train),
        "n_train_orig": int((train.aug_id == "orig").sum()),
        "n_train_aug": int((train.aug_id != "orig").sum()),
        "n_test_rows": len(test),
        "train_aug_ids": ",".join(sorted(set(train.loc[train.aug_id != "orig", "aug_id"]))),
        "test_sources": ",".join(sorted(test.source.unique())),
        "tess_test_speakers": ",".join(sorted(tess_test)),
        "bnc_in_test": bnc_in_test,
    }


def build_mlp(seed: int) -> Sequential:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
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
    lines = ["# v2 10-fold OOF sizdirma ve veri ayiri kontrolleri", "",
             "| kapsam | kontrol | sonuc | detay |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['scope']} | {r['check']} | {r['result']} | {r['detail']} |")
    lines += ["",
              "- REVIEW_LABEL kayitlari manifestten cikarildi; "
              f"{EXPECTED_CANONICAL} canonical kayit dogrulandi.",
              "- Feature CSV fold degerleri manifest fold degerleriyle birebir eslesiyor.",
              "- TESS'in iki atomik grup (OAF, YAF) fold icinde birlikte tutuldu.",
              "- BNC'nin 100 kaydi tek grup (`BNC_SPEAKER_01`) ve tek fold'da.",
              "- Augmentation satirlari yalnizca kaydın kendi fold'u train tarafindaysa egitime girdi.",
              "- Scaler her fold'da yalnizca o fold train'ine fit edildi.",
              "- Model agirligi kaydedilmedi; McNemar/ROC/OOF olasiliklari calistirilmadi.", ""]
    (OUT_DIR / "sizdirma_kontrolleri.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=== v2 / 10-fold group-aware CV + OOF (RF + MLP) ===")
    man = pd.read_csv(MANIFEST_FILE, dtype=str)
    if "dedup_status" not in man.columns:
        raise RuntimeError("Dedup manifesti dedup_status kolonunu icermiyor.")
    review_label_n = int((man.dedup_status == "REVIEW_LABEL").sum())
    if review_label_n != 2:
        raise AssertionError(f"REVIEW_LABEL beklenen 2, bulunan {review_label_n}.")
    man = man[man.dedup_status != "REVIEW_LABEL"].copy()
    man["fold"] = man.fold.astype(int)
    feat = pd.read_csv(FEATURE_FILE, dtype={c: str for c in METADATA_COLS})
    check_inputs(man, feat)

    checks: list[dict] = []
    checks.append({"scope": "global", "check": "group_id kaynak on eki ve benzersizligi",
                   "result": "PASS", "detail": f"{man.group_id.nunique()} grup, 0 cakisma"})
    checks.append({"scope": "global", "check": "source_id -> tek fold",
                   "result": "PASS",
                   "detail": f"{len(man)}/{len(man)} kayit tek fold'da"})
    checks.append({"scope": "global", "check": "feature/manifest source_id kumesi",
                   "result": "PASS", "detail": f"{len(man)} ortak kayit"})

    le = LabelEncoder()
    le.fit(CLASSES)

    per_fold_rows: list[dict] = []
    oof_rows: list[dict] = []

    for k in FOLDS:
        t0 = time.time()
        d = build_fold_data(feat, le, k)
        fold_seed = SEED + k

        # ---------------- RF ----------------
        rf = RandomForestClassifier(
            n_estimators=RF_TREES, class_weight="balanced",
            random_state=fold_seed, n_jobs=-1)
        t_rf = time.time()
        rf.fit(d["X_train"], d["y_train"])
        rf_seconds = round(time.time() - t_rf, 2)
        rf_pred = rf.predict(d["X_test"])

        # ---------------- MLP ----------------
        tf.keras.backend.clear_session()
        mlp = build_mlp(fold_seed)
        ytr_cat = to_categorical(d["y_train"], num_classes=len(CLASSES))
        t_mlp = time.time()
        mlp.fit(d["X_train"], ytr_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                verbose=0, class_weight=class_weights(d["y_train"]))
        mlp_pred = np.argmax(mlp.predict(d["X_test"], verbose=0), axis=1)
        mlp_seconds = round(time.time() - t_mlp, 2)

        for model_name, pred in [("RF", rf_pred), ("MLP", mlp_pred)]:
            row = {"model": model_name, "fold": k}
            row.update(metrics(d["y_test"], pred))
            row.update({
                "n_train_rows": d["n_train_rows"],
                "n_train_orig": d["n_train_orig"],
                "n_train_aug": d["n_train_aug"],
                "n_test_rows": d["n_test_rows"],
                "train_aug_ids": d["train_aug_ids"],
                "test_sources": d["test_sources"],
                "tess_test_speakers": d["tess_test_speakers"],
                "bnc_in_test": d["bnc_in_test"],
                "seed": fold_seed,
                "seconds": rf_seconds if model_name == "RF" else mlp_seconds,
            })
            per_fold_rows.append(row)

        # OOF satirlari: her kayit yalnizca kendi fold'unda bir kez
        test_ids = d["test"].source_id.to_numpy()
        y_true_lbl = d["test"].labels.to_numpy()
        for i, sid in enumerate(test_ids):
            oof_rows.append({
                "source_id": sid,
                "group_id": d["test"].group_id.iloc[i],
                "source": d["test"].source.iloc[i],
                "fold": k,
                "y_true": y_true_lbl[i],
                "rf_pred": CLASSES[rf_pred[i]],
                "mlp_pred": CLASSES[mlp_pred[i]],
                "rf_correct": bool(rf_pred[i] == d["y_test"][i]),
                "mlp_correct": bool(mlp_pred[i] == d["y_test"][i]),
            })

        checks.append({"scope": f"fold_{k}", "check": "test aug_id == orig",
                       "result": "PASS", "detail": f"{d['n_test_rows']} satir, 0 aug"})
        checks.append({"scope": f"fold_{k}", "check": "train/test grup ve source kesismesi",
                       "result": "PASS", "detail": "0 / 0"})
        checks.append({"scope": f"fold_{k}", "check": "aug satirlari yalnizca train'de",
                       "result": "PASS",
                       "detail": f"train_aug={d['n_train_aug']}, test_aug=0"})

        rf_m = metrics(d["y_test"], rf_pred)
        mlp_m = metrics(d["y_test"], mlp_pred)
        print(f"fold {k:>2}: n_test={d['n_test_rows']:>4} | "
              f"RF acc={rf_m['accuracy']:.4f} f1={rf_m['f1_macro']:.4f} | "
              f"MLP acc={mlp_m['accuracy']:.4f} f1={mlp_m['f1_macro']:.4f} | "
              f"{time.time() - t0:.1f}s")

    per_fold = pd.DataFrame(per_fold_rows)
    oof = pd.DataFrame(oof_rows)

    # ---------------- OOF tekillik dogrulamasi ----------------
    if len(oof) != len(man):
        raise AssertionError(f"OOF satir sayisi {len(oof)}, beklenen {len(man)}.")
    if not oof.source_id.is_unique:
        raise AssertionError("OOF icinde tekrarlanan source_id var.")
    if set(oof.source_id) != set(man.source_id):
        raise AssertionError("OOF source_id kumesi manifest ile ayni degil.")
    merged = oof.merge(man[["source_id", "fold"]], on="source_id", suffixes=("", "_manifest"))
    if not (merged.fold == merged.fold_manifest).all():
        raise AssertionError("OOF fold atamasi manifest ile uyusmuyor.")
    checks.append({"scope": "OOF", "check": "her kayit icin tek tahmin",
                   "result": "PASS", "detail": f"{len(oof)} satir, {oof.source_id.nunique()} benzersiz kayit"})

    # ---------------- Ciktilar ----------------
    per_fold.to_csv(OUT_DIR / "kfold_per_fold.csv", index=False, encoding="utf-8")

    value_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                  "precision_weighted", "recall_weighted", "f1_weighted"]
    ozet = per_fold.groupby("model")[value_cols].agg(["mean", "std"]).round(6)
    ozet.columns = [f"{a}_{b}" for a, b in ozet.columns]
    ozet = ozet.reset_index()
    ozet["n_folds"] = 10
    ozet.to_csv(OUT_DIR / "kfold_ozet.csv", index=False, encoding="utf-8")

    oof_metrics_rows = []
    source_rows = []
    for model_name, pred_col in [("RF", "rf_pred"), ("MLP", "mlp_pred")]:
        y_true = le.transform(oof.y_true.to_numpy())
        y_pred = le.transform(oof[pred_col].to_numpy())
        m = metrics(y_true, y_pred)
        m.update({"model": model_name, "n_records": len(oof), "evaluation": "OOF_10fold"})
        oof_metrics_rows.append(m)
        for src in sorted(oof.source.unique()):
            sub = oof[oof.source == src]
            sm = metrics(le.transform(sub.y_true.to_numpy()),
                         le.transform(sub[pred_col].to_numpy()))
            sm.update({"model": model_name, "source": src, "n_records": len(sub)})
            source_rows.append(sm)

    oof_metrics = pd.DataFrame(oof_metrics_rows)
    oof_metrics.to_csv(OUT_DIR / "oof_metrikler.csv", index=False, encoding="utf-8")
    src_df = pd.DataFrame(source_rows)
    src_df.to_csv(OUT_DIR / "oof_kaynak_bazli.csv", index=False, encoding="utf-8")
    oof.to_csv(OUT_DIR / "oof_tahminler.csv", index=False, encoding="utf-8")
    save_checks(checks)

    # OOF confusion matrix grafikleri (kucuk, yol/kimlik icermez)
    for model_name, pred_col in [("RF", "rf_pred"), ("MLP", "mlp_pred")]:
        cm = confusion_matrix(le.transform(oof.y_true.to_numpy()),
                              le.transform(oof[pred_col].to_numpy()),
                              labels=range(len(CLASSES)))
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

    config = {
        "evaluation": "10fold_group_aware_OOF_RF_MLP",
        "manifest": str(MANIFEST_FILE.relative_to(ROOT)),
        "feature_file": str(FEATURE_FILE.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST_FILE),
        "feature_sha256": sha256(FEATURE_FILE),
        "fold_source": "manifest_v2_dedup_draft.csv (REVIEW_LABEL excluded, source_id -> fold)",
        "expected_canonical": EXPECTED_CANONICAL,
        "review_label_excluded": review_label_n,
        "fold_match": "feature CSV fold == manifest fold (birebir)",
        "seed_base": SEED,
        "fold_seeds": {str(k): SEED + k for k in FOLDS},
        "rf": {"n_estimators": RF_TREES, "class_weight": "balanced"},
        "mlp": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "early_stopping": False,
                "class_weight": "balanced + happy*1.25 + sad*1.15",
                "architecture": "256-128-64-4, BN + Dropout"},
        "augmentation": "yalnizca fold train kayitlarinin mevcut deterministic aug satirlari",
        "scaler": "her fold'da yalnizca o fold train'ine fit; RF ve MLP'ye uygulandi",
        "models_saved": False,
        "not_run": ["McNemar", "ROC/AUC"],
    }
    with open(OUT_DIR / "cv_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---------------- Konsol ozeti ----------------
    print("\n=== Fold ortalamasi +/- std ===")
    print(ozet[[c for c in ozet.columns if c in ("model", "accuracy_mean", "accuracy_std",
                                                 "f1_macro_mean", "f1_macro_std",
                                                 "f1_weighted_mean", "f1_weighted_std")]].to_string(index=False))
    print("\n=== OOF metrikler ===")
    print(oof_metrics[["model", "n_records", "accuracy", "precision_macro",
                       "recall_macro", "f1_macro", "f1_weighted"]].to_string(index=False))
    print("\n=== OOF kaynak bazli ===")
    print(src_df[["model", "source", "n_records", "accuracy", "f1_macro"]].to_string(index=False))
    print("\nTum sizdirma kontrolleri PASS; McNemar/ROC/CALISTIRILMADI.")


if __name__ == "__main__":
    main()
