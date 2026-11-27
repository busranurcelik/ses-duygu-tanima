"""
A1.2 / 70/15/15 MLP + MLP ile eslesmis RF-B.

Bu script mevcut 4_ysa_modeli_v2.py protokolunu birebir kullanir:
  * split_70_15_15 == 'train'/'val'/'test' disindaki satirlar kullanilmaz.
  * A0 ile ayni split atamalari, ayni kayitlar, ayni seed ve ayni model ayarlari.
  * Scaler yalnizca 70/15/15 train'e fit edilir; RF-B de ayni scaleri kullanir.
  * Test kumesi tamamiyla TESS YAF'tir; genel basari iddiasi tasimaz.
  * McNemar / ROC / OOF / 10-fold CV bu scriptte CALISTIRILMAZ.

A1.2 farki:
  * Giris 192 ozelliktir: 184 A0 ozelligi + 8 F0 ozelligi.
  * Ilk 184 ozellik A0 CSV'siyle dogrulanir.
  * 184..190 sentinel degerleri yalnizca train'in fallback'siz satirlarindan
    ogrenilen medyanla doldurulur; 191 (voiced_frame_ratio) impute edilmez.
  * Butun ciktilar yalnizca A1_2/70_15_15/ altindadir.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import hashlib
import json
import random
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

ROOT = HERE.parents[1]
INPUT_FILE = ROOT / "A1_2" / "final_oznitelik_seti_A1_2.csv"
A0_INPUT_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"
DETAIL_FILE = ROOT / "A1_2" / "f0_istatistikleri.csv"
A0_METRIC_FILE = ROOT / "sonuclar_v2_dedup" / "ysa" / "70_15_15_karsilastirma_metrikleri_dedup.csv"
A0_PRED_DIR = ROOT / "sonuclar_v2_dedup" / "karsilastirma"

OUT_DIR = HERE / "ysa"
MODEL_DIR = HERE / "modeller"
CMP_DIR = HERE / "karsilastirma"
AUDIT_DIR = HERE
for d in (OUT_DIR, MODEL_DIR, CMP_DIR, AUDIT_DIR, HERE / ".mplconfig"):
    d.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [str(i) for i in range(192)]
A0_FEATURE_COLS = [str(i) for i in range(184)]
F0_SENTINEL_COLS = [str(i) for i in range(184, 191)]
F0_VOICED_COL = "191"
SENTINEL_IDX = np.arange(184, 191)
METADATA_COLS = ["source_id", "group_id", "source", "label", "aug_id",
                 "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]
CLASSES = ["angry", "fear", "happy", "sad"]
SPLIT_COL = "split_70_15_15"
SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
EXPECTED_TEST_ROWS = 800
EXPECTED_TOTAL_COLS = 202

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

CHECKS: list[dict] = []


def add_check(scope: str, check: str, expected, actual, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    CHECKS.append({"scope": scope, "check": check, "beklenen": str(expected),
                   "gercek": str(actual), "durum": status, "detay": detail})
    if not ok:
        raise AssertionError(f"{scope}/{check}: beklenen={expected}, gercek={actual}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_schema(df: pd.DataFrame) -> None:
    expected = set(FEATURE_COLS + METADATA_COLS)
    actual = set(df.columns)
    add_check("global", "202 kolon / 192 ozellik", 202, df.shape[1],
              actual == expected and df.shape[1] == EXPECTED_TOTAL_COLS,
              f"metadata={len(METADATA_COLS)}, ozellik={len(FEATURE_COLS)}")
    add_check("global", "ilk_184_A0_ozelligi", 184, len(A0_FEATURE_COLS),
              len(A0_FEATURE_COLS) == 184)


def check_group_identity(df: pd.DataFrame) -> None:
    bad = df[~df.apply(lambda r: r["group_id"].startswith(r["source"] + "_"), axis=1)]
    add_check("global", "group_id_kaynak_on_eki", 0, len(bad), len(bad) == 0)
    shared = df.groupby("group_id")["source"].nunique()
    add_check("global", "bir_group_tek_kaynak", 0, int((shared > 1).sum()),
              not (shared > 1).any())


def load_inputs() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(INPUT_FILE, dtype={c: str for c in METADATA_COLS})
    a0 = pd.read_csv(A0_INPUT_FILE, dtype={c: str for c in METADATA_COLS})
    check_schema(raw)
    check_group_identity(raw)

    add_check("global", "satir_sayisi", 23268, len(raw), len(raw) == 23268)
    add_check("global", "A0_satir_sayisi", 23268, len(a0), len(a0) == 23268)

    keys = ["source_id", "aug_id"]
    merged = raw.merge(a0, on=keys, suffixes=("_a12", "_a0"), validate="one_to_one")
    add_check("global", "A0_A1_2_satir_eslesmesi", 23268, len(merged), len(merged) == 23268)
    meta_ne = 0
    for c in METADATA_COLS:
        if c in keys:
            continue
        meta_ne += int((merged[c + "_a12"] != merged[c + "_a0"]).sum())
    add_check("global", "metadata_A0_ile_birebir", 0, meta_ne, meta_ne == 0,
              "source_id+aug_id ile eslesildi")
    x0 = merged[[c + "_a0" for c in A0_FEATURE_COLS]].to_numpy(dtype=float)
    x12 = merged[[c + "_a12" for c in A0_FEATURE_COLS]].to_numpy(dtype=float)
    maxdiff = float(np.abs(x0 - x12).max())
    add_check("global", "ilk_184_ozellik_max_diff", "<=1e-10", f"{maxdiff:.3e}",
              maxdiff <= 1e-10)

    det = pd.read_csv(DETAIL_FILE, dtype={"source_id": str, "aug_id": str})
    add_check("global", "f0_detay_satir_sayisi", 23268, len(det), len(det) == 23268)
    add_check("global", "f0_detay_tekrar", 0, int(det.duplicated(keys).sum()),
              not det.duplicated(keys).any())
    fmap = {(sid, aug): int(fb) for sid, aug, fb in
            zip(det.source_id, det.aug_id, det.fallback)}
    raw = raw.copy()
    raw["f0_fallback"] = [fmap.get((sid, aug), -1)
                          for sid, aug in zip(raw.source_id, raw.aug_id)]
    add_check("global", "fallback_bayragi_var", 0, int((raw.f0_fallback < 0).sum()),
              not (raw.f0_fallback < 0).any())
    fb = raw.f0_fallback.to_numpy(dtype=int)
    sent = raw[F0_SENTINEL_COLS].to_numpy(dtype=float)
    add_check("global", "fallback_sayisi", 1606, int(fb.sum()), int(fb.sum()) == 1606)
    add_check("global", "fallback_satir_sentinel_sifir", 0,
              int((np.abs(sent[fb == 1]) > 0).sum()), np.allclose(sent[fb == 1], 0.0, atol=0.0))
    feat = raw[FEATURE_COLS].to_numpy(dtype=float)
    add_check("global", "ozellik_matrisi_finite", True, bool(np.isfinite(feat).all()),
              bool(np.isfinite(feat).all()))
    return raw, {"a0_path": str(A0_INPUT_FILE), "a0_sha256": sha256(A0_INPUT_FILE),
                 "a12_path": str(INPUT_FILE), "a12_sha256": sha256(INPUT_FILE),
                 "detail_path": str(DETAIL_FILE), "detail_sha256": sha256(DETAIL_FILE),
                 "first184_maxdiff": maxdiff}


def audit_filter(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw[raw[SPLIT_COL].isin(["train", "val", "test"])].reset_index(drop=True)
    train = df[df[SPLIT_COL] == "train"]
    val = df[df[SPLIT_COL] == "val"]
    test = df[df[SPLIT_COL] == "test"]

    add_check("global", "EXCLUDED_filtrelendi", 0, int((raw[SPLIT_COL] == "EXCLUDED").sum()),
              not (df[SPLIT_COL] == "EXCLUDED").any(),
              f"{int((raw[SPLIT_COL] == 'EXCLUDED').sum())} satir atildi")
    add_check("global", "train_satir", 17536, len(train), len(train) == 17536)
    add_check("global", "val_satir", 1268, len(val), len(val) == 1268)
    add_check("global", "test_satir", 800, len(test), len(test) == 800)
    add_check("global", "val_test_yalnizca_orig", 0,
              int(((val.aug_id != "orig") | (test.aug_id != "orig")).sum()),
              bool((val.aug_id == "orig").all() and (test.aug_id == "orig").all()))
    for a, b, label in [(train, val, "train-val"), (train, test, "train-test"), (val, test, "val-test")]:
        go = len(set(a.group_id) & set(b.group_id))
        so = len(set(a.source_id) & set(b.source_id))
        add_check("global", label + "_grup_kesisimi", 0, go, go == 0)
        add_check("global", label + "_source_kesisimi", 0, so, so == 0)
    roles = df.groupby("source_id")[SPLIT_COL].nunique()
    add_check("global", "source_tek_split", 0, int((roles > 1).sum()), not (roles > 1).any())

    test_sources = sorted(test.source.unique())
    test_groups = sorted(test.group_id.unique())
    add_check("global", "test_kaynagi", "TESS", ",".join(test_sources),
              test_sources == ["TESS"])
    add_check("global", "test_grup", "TESS_YAF", ",".join(test_groups),
              test_groups == ["TESS_YAF"], "Test yalnizca TESS_YAF konusmacisidir")
    add_check("global", "train_aug_varyantlari", True, sorted(set(train.aug_id.unique())) == sorted(
              ["noise", "noise2", "orig", "pitch_down", "pitch_up", "stretch_0.9", "stretch_1.1"]),
              sorted(set(train.aug_id.unique())) == sorted(
                  ["noise", "noise2", "orig", "pitch_down", "pitch_up", "stretch_0.9", "stretch_1.1"]))

    audit = {
        "script": "4_ysa_A1_2.py",
        "split_column": SPLIT_COL,
        "input_rows": len(raw),
        "excluded_dropped": int((raw[SPLIT_COL] == "EXCLUDED").sum()),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "n_features": len(FEATURE_COLS),
        "train_aug_ids": ",".join(sorted(set(train.aug_id))),
        "eval_aug_ids": ",".join(sorted(set(val.aug_id) | set(test.aug_id))),
        "eval_nonorig_rows": int(((val.aug_id != "orig") | (test.aug_id != "orig")).sum()),
        "group_overlap": len(set(train.group_id) & (set(val.group_id) | set(test.group_id))),
        "source_overlap": len(set(train.source_id) & (set(val.source_id) | set(test.source_id))),
        "test_sources": ",".join(test_sources),
        "test_groups": ",".join(test_groups),
        "status": "PASS",
    }
    pd.DataFrame([audit]).to_csv(AUDIT_DIR / "filtre_dogrulama.csv", index=False, encoding="utf-8")
    return df, audit


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
    pc = precision_score(y_true, y_pred, labels=range(4), average=None, zero_division=0)
    rc = recall_score(y_true, y_pred, labels=range(4), average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=range(4), average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(4))
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"] = float(pc[i])
        out[f"recall_{c}"] = float(rc[i])
        out[f"f1_{c}"] = float(f1c[i])
        out[f"support_{c}"] = int(cm[i].sum())
    fi, hi = CLASSES.index("fear"), CLASSES.index("happy")
    out["fear_to_happy"] = int(cm[fi, hi])
    out["happy_to_fear"] = int(cm[hi, fi])
    out["fear_to_happy_rate"] = float(cm[fi, hi] / max(int(cm[fi].sum()), 1))
    out["happy_to_fear_rate"] = float(cm[hi, fi] / max(int(cm[hi].sum()), 1))
    return out


def class_weights(y_train: np.ndarray) -> dict:
    arr = compute_class_weight(class_weight="balanced", classes=np.arange(4), y=y_train)
    w = {i: float(v) for i, v in enumerate(arr)}
    w[CLASSES.index("happy")] *= 1.25
    w[CLASSES.index("sad")] *= 1.15
    return w


def build_mlp() -> Sequential:
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


def plot_cm(cm: np.ndarray, name: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"A1.2 {name} - TESS YAF (unseen speaker)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"70_15_15_{name}_cm_A1_2.png", dpi=200)
    plt.close()


def save_checks() -> dict:
    df = pd.DataFrame(CHECKS)
    df.to_csv(AUDIT_DIR / "dogrulama_kontrolleri.csv", index=False, encoding="utf-8")
    lines = ["# A1.2 70/15/15 — veri sızıntısı ve doğrulama kontrolleri", "",
             f"- Kontrol: **{len(df)}**, PASS: **{int(df.durum.eq('PASS').sum())}**, "
             f"FAIL: **{int(df.durum.eq('FAIL').sum())}**", "",
             "| kapsam | kontrol | beklenen | gerçek | durum | detay |",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['scope']} | {r['check']} | {r['beklenen']} | {r['gercek']} | "
                     f"{r['durum']} | {r['detay']} |")
    lines += ["",
              "- A0 ile aynı `split_70_15_15` atamaları, aynı train/val/test kayıtları kullanıldı.",
              "- İlk 184 özellik A0 CSV'siyle birebir doğrulandı.",
              "- F0 medyanları yalnızca train'in fallback'siz satırlarından öğrenildi.",
              "- `voiced_frame_ratio` (191) impute edilmedi; gerçek sıfırlar korundu.",
              "- Scaler yalnızca 70/15/15 train'e fit edildi.",
              "- 10-fold OOF, McNemar, ROC ve mevcut A1.2 raporu değiştirilmedi.", ""]
    (AUDIT_DIR / "dogrulama_kontrolleri.md").write_text("\n".join(lines), encoding="utf-8")
    return {"n": len(df), "pass": int(df.durum.eq("PASS").sum()),
            "fail": int(df.durum.eq("FAIL").sum())}


def fmt(v: float, nd: int = 4) -> str:
    return f"{float(v):.{nd}f}".replace(".", ",")


def sfmt(v: float, nd: int = 4) -> str:
    return f"{float(v):+.{nd}f}".replace(".", ",")


def write_report(overall: pd.DataFrame, class_df: pd.DataFrame, cmp_df: pd.DataFrame,
                 conf_df: pd.DataFrame, gaps: pd.DataFrame, manifest: dict,
                 imp_rows: list[dict], validation: dict, audit: dict) -> None:
    L: list[str] = []
    A = "## "
    L += ["# A1.2 — 70/15/15 Değerlendirme Raporu", ""]
    L += ["Bu çalışma, mevcut A0 70/15/15 protokolü birebir korunarak A1.2'nin 192 özellikli "
          "veri seti üzerinde çalıştırılmıştır. 10-fold OOF sonuçları ve mevcut A1.2 raporu "
          "değiştirilmemiştir.", ""]
    L += [A + "1. Kritik kapsam uyarısı", "",
          "**Mevcut 70/15/15 test kümesinin tamamı `TESS_YAF` konuşmacısındandır "
          "(800 özgün kayıt, kaynak=TESS). Bu sonuç konuşmacıdan bağımsız genelleme başarısı "
          "olarak sunulamaz; yalnızca görülmemiş tek bir TESS konuşmacısındaki performansı "
          "gösterir.**", ""]
    L += [A + "2. Protokol ve veri", "",
          f"- A0 ile aynı split kolonu: `{SPLIT_COL}`.",
          f"- Satırlar: girdi {audit['input_rows']:,}, EXCLUDED atılan "
          f"{audit['excluded_dropped']:,}, train {audit['train_rows']:,}, "
          f"val {audit['val_rows']:,}, test {audit['test_rows']:,}.".replace(",", "."),
          f"- Train aug_id'leri: {audit['train_aug_ids']}; val/test: {audit['eval_aug_ids']}.",
          f"- Test kaynağı/grubu: {audit['test_sources']} / {audit['test_groups']}.",
          "- Seed 42, MLP 200 epoch (erken durma ile), batch 32; RF-B 200 ağaç, "
          "`class_weight='balanced'`, `random_state=42`.",
          "- Scaler yalnızca train'e fit edildi; RF-B ve MLP aynı scaleri kullandı.", ""]
    L += [A + "3. F0 imputasyonu", "",
          "7 sentinel sütun (184..190) yalnızca train'in fallback'siz satırlarından hesaplanan "
          "medyanlarla dolduruldu; val ve test için aynı medyanlar kullanıldı. "
          "`voiced_frame_ratio` (191) impute edilmedi.", "",
          "| özellik | train medyanı | train fallback'siz | train impute | val impute | test impute |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in imp_rows:
        L.append(f"| {r['ozellik']} | {fmt(r['train_medyan'], 6)} | {r['n_train_fallbacksiz']:,} | "
                 f"{r['n_train_impute']:,} | {r['n_val_impute']:,} | {r['n_test_impute']:,} |".replace(",", "."))
    L += ["", f"Train'de {imp_rows[0]['n_train_fallbacksiz']:,} fallback'siz satır, "
          f"{imp_rows[0]['n_train_impute']:,} impute edilen satır; val'de "
          f"{imp_rows[0]['n_val_impute']:,}, test'te {imp_rows[0]['n_test_impute']:,} impute uygulandı."
          .replace(",", "."), ""]
    L += [A + "4. Test performansı: A0 vs A1.2", "",
          "A0 değerleri kaydedilmiş test tahminlerinden, A1.2 değerleri bu çalışmadaki "
          "tahminlerden hesaplanmıştır. Test kümesi iki deney için birebir aynıdır.", "",
          "| Model | Metrik | A0 | A1.2 | Fark |", "|---|---|---:|---:|---:|"]
    main_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "f1_angry", "f1_fear", "f1_happy", "f1_sad"]
    for model in ["RF-B", "MLP"]:
        for metric in main_metrics:
            r = cmp_df[(cmp_df.model == model) & (cmp_df.metrik == metric)].iloc[0]
            L.append(f"| {model} | {metric} | {fmt(r.A0)} | {fmt(r.A1_2)} | {sfmt(r.fark)} |")
    L += [""]
    L += [A + "5. Sınıf bazlı precision / recall / F1 (A1.2 test)", "",
          "| Model | Sınıf | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|"]
    for _, r in class_df[class_df.split == "test"].iterrows():
        L.append(f"| {r['model']} | {r['sinif']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                 f"{fmt(r['f1'])} | {int(r['support'])} |")
    L += [""]
    L += [A + "6. Confusion matrix (A1.2 test)", ""]
    for model in ["RF-B", "MLP"]:
        L += [f"**{model}**", "", "| gerçek \\ tahmin | " + " | ".join(CLASSES) + " |",
              "|---|" + "---:|" * 4]
        sub = conf_df[(conf_df.model == model) & (conf_df.split == "test")]
        for c in CLASSES:
            row = sub[sub.gercek_sinif == c].set_index("tahmin_sinif").n
            L.append(f"| {c} | " + " | ".join(str(int(row[x])) for x in CLASSES) + " |")
        L += [""]
    L += [A + "7. Fear ↔ happy karışıklığı", "",
          "| Model | Yön | A0 | A1.2 | Fark |", "|---|---|---:|---:|---:|"]
    for model in ["RF-B", "MLP"]:
        for metric in ["fear_to_happy", "happy_to_fear", "fear_to_happy_rate", "happy_to_fear_rate"]:
            r = cmp_df[(cmp_df.model == model) & (cmp_df.metrik == metric)].iloc[0]
            nd = 0 if "rate" not in metric else 4
            L.append(f"| {model} | {metric} | {fmt(r.A0, nd)} | {fmt(r.A1_2, nd)} | {sfmt(r.fark, nd)} |")
    L += ["", "F0 eklenmesine rağmen fear/happy karışıklığı tamamen çözülmemiştir; "
          "yön ve büyüklük modele göre değişmektedir.", ""]
    L += [A + "8. Eğitim / validation performans farkı", "",
          "Train metrikleri augmentation'lı eğitim satırlarını içerir; val ve test yalnızca "
          "özgün kayıtlardır.", "",
          "| Model | Metrik | Train | Val | Train−Val | Test |", "|---|---|---:|---:|---:|---:|"]
    for _, r in gaps.iterrows():
        L.append(f"| {r['model']} | {r['metrik']} | {fmt(r['train'])} | {fmt(r['val'])} | "
                 f"{sfmt(r['train_val_farki'])} | {fmt(r['test'])} |")
    L += [""]
    L += [A + "9. Eğitim süreleri ve epoch", "",
          f"- MLP: **{manifest['models']['mlp']['seconds']} sn**, "
          f"çalıştırılan epoch **{manifest['epochs_run']}** / {manifest['epochs_requested']}, "
          f"en iyi val_accuracy epoch **{manifest['best_epoch']}**.",
          f"- RF-B: **{manifest['models']['rf_b']['seconds']} sn**.", ""]
    L += [A + "10. Doğrulama ve sızıntı kontrolleri", "",
          f"- Kontrol: **{validation['n']}**, PASS: **{validation['pass']}**, "
          f"FAIL: **{validation['fail']}**.",
          "- İlk 184 özellik A0 ile birebir doğrulandı; grup/source sızıntısı yok; "
          "val/test yalnızca orig; scaler ve F0 medyanları yalnızca train'den.",
          "- Ayrıntılar: `dogrulama_kontrolleri.csv` / `.md`.", ""]
    L += [A + "11. Sınırlılıklar", "",
          "1. Test yalnızca `TESS_YAF` konuşmacısından oluşur; konuşmacıdan bağımsız genelleme iddiası yoktur.",
          "2. TESS'te test için yalnızca 4 grup/konuşmacı vardır ve bu split'te test tek konuşmacıdır.",
          "3. Train metrikleri augmentation nedeniyle bağımsız değildir.",
          "4. McNemar/ROC/10-fold CV bu deneyde çalıştırılmamıştır.",
          "5. A0 70/15/15 modeli yeniden eğitilmemiş; kaydedilmiş test tahminleri kullanılmıştır.", ""]
    L += ["## Çıktılar", "",
          "- `ysa/`: metrikler, sınıf metrikleri, confusion matrix, eğitim grafikleri, label map.",
          "- `modeller/`: scaler, RF-B, en iyi ve son MLP.",
          "- `karsilastirma/`: A1.2 test tahminleri ve A0 karşılaştırma tabloları.",
          "- `dogrulama_kontrolleri.*`, `filtre_dogrulama.csv`, `f0_imputasyon_degerleri.csv`, "
          "`artifact_manifest_A1_2.json`, `egitim_log.txt`.", ""]
    (AUDIT_DIR / "son_rapor.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    print("=== A1.2 / 70_15_15 MLP + eslesmis RF-B ===", flush=True)
    raw, input_info = load_inputs()
    df, audit = audit_filter(raw)

    X_raw = df[FEATURE_COLS].to_numpy(dtype=float)
    voiced_before = X_raw[:, int(F0_VOICED_COL)].copy()
    mask_train = (df[SPLIT_COL] == "train").to_numpy()
    mask_val = (df[SPLIT_COL] == "val").to_numpy()
    mask_test = (df[SPLIT_COL] == "test").to_numpy()
    fb = df.f0_fallback.to_numpy(dtype=int)

    ok = fb[mask_train] == 0
    med = np.median(X_raw[mask_train][ok][:, SENTINEL_IDX], axis=0)
    add_check("global", "imputasyon_medyan_finite", True, bool(np.isfinite(med).all()),
              bool(np.isfinite(med).all()), f"n_train_fallbacksiz={int(ok.sum())}")
    X = X_raw.copy()
    for m in (mask_train, mask_val, mask_test):
        X[m][:, SENTINEL_IDX] = np.where(fb[m][:, None] == 1, med, X[m][:, SENTINEL_IDX])
    add_check("global", "voiced_frame_ratio_degismedi", True,
              bool(np.array_equal(X[:, int(F0_VOICED_COL)], voiced_before)),
              bool(np.array_equal(X[:, int(F0_VOICED_COL)], voiced_before)),
              "191 impute edilmedi")
    add_check("global", "imputasyon_sonrasi_finite", True, bool(np.isfinite(X).all()),
              bool(np.isfinite(X).all()))

    imp_rows = []
    f0_names = ["f0_mean", "f0_median", "f0_std", "f0_min", "f0_max", "f0_range", "f0_slope"]
    for j, col in enumerate(F0_SENTINEL_COLS):
        imp_rows.append({
            "kolon": col, "ozellik": f0_names[j], "train_medyan": float(med[j]),
            "n_train_fallbacksiz": int(ok.sum()),
            "n_train_impute": int((fb[mask_train] == 1).sum()),
            "n_val_impute": int((fb[mask_val] == 1).sum()),
            "n_test_impute": int((fb[mask_test] == 1).sum()),
            "kaynak": "yalnizca 70/15/15 train fallback'siz satirlari",
        })
    pd.DataFrame(imp_rows).to_csv(AUDIT_DIR / "f0_imputasyon_degerleri.csv",
                                  index=False, encoding="utf-8")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[mask_train])
    X_val = scaler.transform(X[mask_val])
    X_test = scaler.transform(X[mask_test])
    add_check("global", "scaler_train_only", 17536, int(scaler.n_samples_seen_),
              int(scaler.n_samples_seen_) == 17536)
    joblib.dump(scaler, MODEL_DIR / "scaler_A1_2.joblib")

    le = LabelEncoder()
    y_enc = le.fit_transform(df["labels"].to_numpy())
    add_check("global", "sinif_sirasi", str(CLASSES), str(list(le.classes_)),
              list(le.classes_) == CLASSES)
    y_cat = to_categorical(y_enc)
    y_train, y_val, y_test = y_enc[mask_train], y_enc[mask_val], y_enc[mask_test]
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}", flush=True)

    # ---------------- MLP (A0 ayarlari) ----------------
    model = build_mlp()
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    cw_map = {i: float(w) for i, w in enumerate(cw)}
    cw_map[CLASSES.index("happy")] *= 1.25
    cw_map[CLASSES.index("sad")] *= 1.15
    best_path = MODEL_DIR / "mlp_best_70_15_15_A1_2.keras"
    callbacks = [
        ModelCheckpoint(str(best_path), monitor="val_accuracy", save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5, verbose=1),
    ]
    t0 = time.time()
    history = model.fit(X_train, y_train_cat := y_cat[mask_train],
                        validation_data=(X_val, y_cat[mask_val]),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
                        class_weight=cw_map, verbose=1)
    mlp_seconds = round(time.time() - t0, 2)
    model.save(str(MODEL_DIR / "mlp_final_70_15_15_A1_2.keras"))
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("A1.2 YSA 70/15/15 - Loss")
    plt.savefig(OUT_DIR / "70_15_15_loss_A1_2.png", dpi=200); plt.close()
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title("A1.2 YSA 70/15/15 - Accuracy")
    plt.savefig(OUT_DIR / "70_15_15_accuracy_A1_2.png", dpi=200); plt.close()

    best = tf.keras.models.load_model(str(best_path))
    probs_mlp_test = best.predict(X_test, verbose=0)
    pred_mlp = {
        "train": np.argmax(best.predict(X_train, verbose=0), axis=1),
        "val": np.argmax(best.predict(X_val, verbose=0), axis=1),
        "test": np.argmax(probs_mlp_test, axis=1),
    }

    # ---------------- RF-B (MLP ile esles, A0 ayarlari) ----------------
    rf_b = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)
    t1 = time.time()
    rf_b.fit(X_train, y_train)
    rf_seconds = round(time.time() - t1, 2)
    joblib.dump(rf_b, MODEL_DIR / "rf_70_15_15_matched_A1_2.joblib")
    probs_rf = rf_b.predict_proba(X_test)
    pred_rf = {
        "train": rf_b.predict(X_train),
        "val": rf_b.predict(X_val),
        "test": rf_b.predict(X_test),
    }

    test_ids = df.loc[mask_test, "source_id"].to_numpy()
    add_check("global", "test_satir_sayisi_karsilastirma", 800, len(test_ids),
              len(test_ids) == EXPECTED_TEST_ROWS)

    models = {"RF-B": pred_rf, "MLP": pred_mlp}
    y_by_split = {"train": y_train, "val": y_val, "test": y_test}

    overall_rows, class_rows, conf_rows = [], [], []
    for model_name, preds in models.items():
        for split in ("train", "val", "test"):
            m = metrics(y_by_split[split], preds[split])
            row = {"model": model_name, "split": split, "n": len(y_by_split[split])}
            row.update(m)
            overall_rows.append(row)
            for i, cls in enumerate(CLASSES):
                class_rows.append({
                    "model": model_name, "split": split, "sinif": cls,
                    "precision": m[f"precision_{cls}"], "recall": m[f"recall_{cls}"],
                    "f1": m[f"f1_{cls}"], "support": m[f"support_{cls}"],
                })
            cm = confusion_matrix(y_by_split[split], preds[split], labels=range(4))
            for i, true_c in enumerate(CLASSES):
                for j, pred_c in enumerate(CLASSES):
                    conf_rows.append({"model": model_name, "split": split,
                                      "gercek_sinif": true_c, "tahmin_sinif": pred_c,
                                      "n": int(cm[i, j])})

    overall = pd.DataFrame(overall_rows)
    class_df = pd.DataFrame(class_rows)
    conf_df = pd.DataFrame(conf_rows)
    overall.to_csv(OUT_DIR / "70_15_15_train_val_test_metrikleri_A1_2.csv", index=False, encoding="utf-8")
    class_df.to_csv(OUT_DIR / "70_15_15_sinif_metrikleri_A1_2.csv", index=False, encoding="utf-8")
    conf_df.to_csv(OUT_DIR / "70_15_15_confusion_matrix_A1_2.csv", index=False, encoding="utf-8")
    overall[overall.split == "test"].drop(columns=["split"]).to_csv(
        OUT_DIR / "70_15_15_karsilastirma_metrikleri_A1_2.csv", index=False, encoding="utf-8")

    gaps = []
    for model_name in models:
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            tr = overall[(overall.model == model_name) & (overall.split == "train")][metric].iloc[0]
            va = overall[(overall.model == model_name) & (overall.split == "val")][metric].iloc[0]
            te = overall[(overall.model == model_name) & (overall.split == "test")][metric].iloc[0]
            gaps.append({"model": model_name, "metrik": metric, "train": tr, "val": va,
                         "train_val_farki": tr - va, "test": te})
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(OUT_DIR / "70_15_15_egitim_val_farki_A1_2.csv", index=False, encoding="utf-8")

    with open(OUT_DIR / "70_15_15_label_map_A1_2.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)
    for model_name in ("RF-B", "MLP"):
        cm = confusion_matrix(y_test, models[model_name]["test"], labels=range(4))
        plot_cm(cm, "rf_matched" if model_name == "RF-B" else "mlp")

    # ---------------- Tahmin dosyalari ----------------
    for name, pred, prob in [("rf_matched", pred_rf["test"], probs_rf),
                             ("mlp", pred_mlp["test"], probs_mlp_test)]:
        out = pd.DataFrame({
            "source_id": test_ids,
            "group_id": df.loc[mask_test, "group_id"].to_numpy(),
            "source": df.loc[mask_test, "source"].to_numpy(),
            "y_true": y_test,
            "y_pred": pred,
        })
        for i, cls in enumerate(CLASSES):
            out[f"prob_{cls}"] = prob[:, i]
        out.to_csv(CMP_DIR / f"70_15_15_{name}_test_predictions_A1_2.csv", index=False, encoding="utf-8")

    # ---------------- A0 ile ayni test kumesi karsilastirmasi ----------------
    a0_metric_ref = pd.read_csv(A0_METRIC_FILE)
    a0_files = {"RF-B": A0_PRED_DIR / "70_15_15_rf_matched_test_predictions_dedup.csv",
                "MLP": A0_PRED_DIR / "70_15_15_mlp_test_predictions_dedup.csv"}
    a0_ref_names = {"RF-B": "RF-B (70/15/15 matched)", "MLP": "MLP"}
    a0_preds = {}
    cmp_rows = []
    for model_name, path in a0_files.items():
        ap = pd.read_csv(path, dtype={"source_id": str, "group_id": str, "source": str})
        add_check("A0_karsilastirma", model_name + "_test_kumesi", EXPECTED_TEST_ROWS,
                  len(ap), set(ap.source_id) == set(test_ids) and len(ap) == EXPECTED_TEST_ROWS)
        ap = ap.set_index("source_id").loc[test_ids].reset_index()
        add_check("A0_karsilastirma", model_name + "_y_true_ayni", True,
                  bool(np.array_equal(ap.y_true.to_numpy(), y_test)),
                  bool(np.array_equal(ap.y_true.to_numpy(), y_test)))
        a0_preds[model_name] = ap.y_pred.to_numpy()
        a0m = metrics(y_test, a0_preds[model_name])
        ref = a0_metric_ref[a0_metric_ref.model == a0_ref_names[model_name]].iloc[0]
        max_ref_diff = max(abs(float(a0m[k]) - float(ref[k])) for k in
                           ["accuracy", "precision_macro", "recall_macro", "f1_macro"])
        add_check("A0_karsilastirma", model_name + "_A0_metrik_reproduksiyon", "<=1e-6",
                  f"{max_ref_diff:.2e}", max_ref_diff <= 1e-6)
        a12m = metrics(y_test, models[model_name]["test"])
        for metric in a0m:
            cmp_rows.append({"model": model_name, "metrik": metric,
                             "A0": float(a0m[metric]), "A1_2": float(a12m[metric]),
                             "fark": float(a12m[metric] - a0m[metric])})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(CMP_DIR / "A0_vs_A1_2_test_metrikleri.csv", index=False, encoding="utf-8")

    class_cmp_rows = []
    for model_name in models:
        for cls in CLASSES:
            for metric in ["precision", "recall", "f1"]:
                a = cmp_df[(cmp_df.model == model_name) &
                           (cmp_df.metrik == f"{metric}_{cls}")].iloc[0]
                class_cmp_rows.append(a.to_dict())
    pd.DataFrame(class_cmp_rows).to_csv(CMP_DIR / "A0_vs_A1_2_sinif_metrikleri.csv",
                                        index=False, encoding="utf-8")
    conf_cmp = cmp_df[cmp_df.metrik.isin(["fear_to_happy", "happy_to_fear",
                                         "fear_to_happy_rate", "happy_to_fear_rate"])].copy()
    conf_cmp.to_csv(CMP_DIR / "A0_vs_A1_2_karisiklik.csv", index=False, encoding="utf-8")

    validation = save_checks()
    manifest = {
        "evaluation": "A1_2_70_15_15_TESS_YAF_unseen_speaker",
        "test_note": "Test kumesinin tamami TESS YAF konusmacisindan; genel basari degildir.",
        "seed": SEED,
        "features": len(FEATURE_COLS),
        "n_a0_features": len(A0_FEATURE_COLS),
        "epochs_requested": EPOCHS,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "batch_size": BATCH_SIZE,
        "train_rows": int(mask_train.sum()),
        "val_rows": int(mask_val.sum()),
        "test_rows": int(mask_test.sum()),
        "scaler_fit_on": "70/15/15 train only",
        "imputation": {
            "method": "median of fallback-free 70/15/15 train rows",
            "sentinel_columns": F0_SENTINEL_COLS,
            "not_imputed": {"191": "voiced_frame_ratio gercek 0 degeridir"},
        },
        "models": {
            "mlp": {"file": "modeller/mlp_best_70_15_15_A1_2.keras",
                    "sha256": sha256(best_path), "seconds": mlp_seconds},
            "rf_b": {"file": "modeller/rf_70_15_15_matched_A1_2.joblib",
                     "sha256": sha256(MODEL_DIR / "rf_70_15_15_matched_A1_2.joblib"),
                     "seconds": rf_seconds, "note": "A0 RF-B ile ayni ayarlar"},
            "scaler": {"file": "modeller/scaler_A1_2.joblib",
                       "sha256": sha256(MODEL_DIR / "scaler_A1_2.joblib")},
        },
        "inputs": input_info,
        "validation": validation,
        "versions": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "tensorflow": tf.__version__,
        },
    }
    with open(AUDIT_DIR / "artifact_manifest_A1_2.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    write_report(overall, class_df, cmp_df, conf_df, gaps_df, manifest, imp_rows,
                 validation, audit)

    print("\n=== A1.2 70/15/15 test sonuclari (TESS YAF unseen speaker) ===", flush=True)
    print(overall[overall.split == "test"][
        ["model", "n", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False),
        flush=True)
    print(f"\nMLP sure: {mlp_seconds} sn | RF-B sure: {rf_seconds} sn | "
          f"epoch: {manifest['epochs_run']} | dogrulama: {validation['pass']}/{validation['n']}",
          flush=True)
    print("Ciktilar yalnizca A1_2/70_15_15/ altinda. McNemar/ROC/CV/OOF CALISTIRILMADI.",
          flush=True)


if __name__ == "__main__":
    main()
