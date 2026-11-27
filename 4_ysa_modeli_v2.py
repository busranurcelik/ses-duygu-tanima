"""
v2 / adim 3b: 70/15/15 MLP + MLP ile ESLES MIS RF-B.

Kurallar:
  * split_70_15_15 == 'train'/'val'/'test' disindaki satirlar (EXCLUDED) kullanilmaz.
  * Ozellikler ISIMLE secilir: '0'..'183'. Giris 194 kolonludur (10 metadata + 184 ozellik).
  * Scaler yalnizca 70/15/15 train'e fit edilir; RF-B de ayni scaleri kullanir.
  * Test kumesi tamamiyla TESS YAF ve 800 satir -> "unseen speaker" olarak etiketlenir.
  * RF-B ayri bir modeldir (RF-A'dan farkli): rf_70_15_15_matched_dedup.joblib.
  * McNemar / ROC / OOF / CV BU SCRIPTTE CALISTIRILMAZ; sadece tahminler kaydedilir.
  * Butun ciktilar sonuclar_v2_dedup/ altinadir; eski sonuclar_v2/ ASLA yazilmaz.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"
OUT_DIR = ROOT / "sonuclar_v2_dedup" / "ysa"
MODEL_DIR = ROOT / "sonuclar_v2_dedup" / "modeller"
CMP_DIR = ROOT / "sonuclar_v2_dedup" / "karsilastirma"
AUDIT_DIR = ROOT / "sonuclar_v2_dedup"
FEATURE_COLS = [str(i) for i in range(184)]
METADATA_COLS = ["source_id", "group_id", "source", "label", "aug_id",
                 "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]
CLASSES = ["angry", "fear", "happy", "sad"]
SPLIT_COL = "split_70_15_15"
SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
EXPECTED_TEST_ROWS = 800
EXPECTED_TOTAL_COLS = 194

for d in (OUT_DIR, MODEL_DIR, CMP_DIR, AUDIT_DIR):
    d.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def check_schema(df: pd.DataFrame) -> None:
    if len(FEATURE_COLS) != 184 or len(METADATA_COLS) != 10:
        raise AssertionError("184 ozellik + 10 metadata kolonu bekleniyor.")
    expected = set(FEATURE_COLS + METADATA_COLS)
    actual = set(df.columns)
    if actual != expected:
        missing = sorted(expected - actual, key=str)
        extra = sorted(actual - expected, key=str)
        raise RuntimeError(
            f"Beklenen 194 kolonluk shema bozuldu. Eksik={missing}, fazla={extra}"
        )
    if df.shape[1] != EXPECTED_TOTAL_COLS:
        raise AssertionError(
            f"{EXPECTED_TOTAL_COLS} kolon beklenirken {df.shape[1]} kolon bulundu."
        )


def check_group_identity(df: pd.DataFrame) -> None:
    bad = df[~df.apply(lambda r: r["group_id"].startswith(r["source"] + "_"), axis=1)]
    if len(bad):
        raise AssertionError(f"{len(bad)} group_id kaynak on ekini tasimiyor.")
    shared = df.groupby("group_id")["source"].nunique()
    if (shared > 1).any():
        raise AssertionError("Bir group_id birden fazla kaynaga ait.")


def audit_filter(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    check_group_identity(df)
    if (df[SPLIT_COL] == "EXCLUDED").any():
        raise AssertionError("EXCLUDED satirlari filtrelenmeden kaldi.")
    if len(df) != len(raw[raw[SPLIT_COL].isin(["train", "val", "test"])]):
        raise AssertionError("70/15/15 filtresi beklenen satir sayisini vermedi.")

    train = df[df[SPLIT_COL] == "train"]
    val = df[df[SPLIT_COL] == "val"]
    test = df[df[SPLIT_COL] == "test"]
    if min(len(train), len(val), len(test)) == 0:
        raise AssertionError("train/val/test bolumlerinden biri bos.")

    for part in (val, test):
        if (part["aug_id"] != "orig").any():
            raise AssertionError("val/test icinde augmentation var.")
    for a, b, label in [(train, val, "train-val"), (train, test, "train-test"), (val, test, "val-test")]:
        if set(a.group_id) & set(b.group_id):
            raise AssertionError(f"{label} grup sizintisi.")
        if set(a.source_id) & set(b.source_id):
            raise AssertionError(f"{label} source_id sizintisi.")
    roles = df[df[SPLIT_COL] != "EXCLUDED"].groupby("source_id")[SPLIT_COL].nunique()
    if (roles > 1).any():
        raise AssertionError("Ayni source_id birden fazla split'te.")

    test_sources = sorted(test.source.unique())
    test_groups = sorted(test.group_id.unique())
    if test_sources != ["TESS"] or test_groups != ["TESS_YAF"]:
        raise AssertionError(
            f"Beklenen test yalnizca TESS_YAF iken: sources={test_sources}, groups={test_groups}"
        )

    return {
        "script": "4_ysa_modeli_v2.py",
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
        "status": "PASS",
    }


def save_audit(audit: dict) -> None:
    path = AUDIT_DIR / "filtre_dogrulama.csv"
    cols = ["script", "split_column", "input_rows", "excluded_dropped",
            "train_rows", "val_rows", "test_rows", "n_features",
            "train_aug_ids", "eval_aug_ids", "eval_nonorig_rows",
            "group_overlap", "source_overlap", "test_sources", "status"]
    old = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=cols)
    old = old[old.script != audit["script"]]
    out = pd.concat([old, pd.DataFrame([audit], columns=cols)], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8")

    lines = [
        "# v2 filtre ve veri ayirimi audit raporu",
        "",
        "| script | split kolonu | girdi | EXCLUDED atilan | train | val | test | "
        "ozellik | eval aug | grup kesisimi | source kesisimi | durum |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {r['script']} | {r['split_column']} | {r['input_rows']} | "
            f"{r['excluded_dropped']} | {r['train_rows']} | {r['val_rows']} | "
            f"{r['test_rows']} | {r['n_features']} | {r['eval_aug_ids']} | "
            f"{r['group_overlap']} | {r['source_overlap']} | {r['status']} |"
        )
    lines += [
        "",
        "- `3_modelleri_egit_v2.py`: 80/20 test kaynaklari CREMA, RAVDESS, SAVEE, BNC (TESS yok).",
        "- `4_ysa_modeli_v2.py`: 70/15/15 test **yalnizca TESS YAF** -> "
        "sonuc 'TESS YAF unseen speaker' olarak etiketlenir; genel basari iddiasi tasimaz.",
        "- Ozellikler 184 kolon `0..183` olarak isimle secildi; metadata X'e girmedi.",
        "- Scaler yalnizca ilgili train kumesine fit edildi.",
        "- RF-A (80/20) ve RF-B (70/15/15, MLP ile eslesmis) AYRI modellerdir.",
        "",
    ]
    (AUDIT_DIR / "filtre_dogrulama.md").write_text("\n".join(lines), encoding="utf-8")


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


def plot_cm(cm: np.ndarray, name: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"{name} - TESS YAF (unseen speaker)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"70_15_15_{name}_cm_dedup.png", dpi=200)
    plt.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=== v2 / 70_15_15 MLP + eslesmis RF-B ===")
    raw = pd.read_csv(INPUT_FILE, dtype={c: str for c in METADATA_COLS})
    check_schema(raw)
    df = raw[raw[SPLIT_COL].isin(["train", "val", "test"])].reset_index(drop=True)
    audit = audit_filter(raw, df)
    save_audit(audit)
    print("Audit:", audit)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    le = LabelEncoder()
    y_enc = le.fit_transform(df["labels"].to_numpy())
    if list(le.classes_) != CLASSES:
        raise AssertionError(f"Beklenen sinif sirasi {CLASSES}, bulunan {list(le.classes_)}")

    mask_train = (df[SPLIT_COL] == "train").to_numpy()
    mask_val = (df[SPLIT_COL] == "val").to_numpy()
    mask_test = (df[SPLIT_COL] == "test").to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[mask_train])       # yalnizca train
    X_val = scaler.transform(X[mask_val])
    X_test = scaler.transform(X[mask_test])
    joblib.dump(scaler, MODEL_DIR / "scaler_70_15_15_dedup.joblib")

    y_cat = to_categorical(y_enc)
    y_train_cat, y_val_cat, y_test_cat = y_cat[mask_train], y_cat[mask_val], y_cat[mask_test]
    y_train, y_test = y_enc[mask_train], y_enc[mask_test]
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    # ---------------- MLP ----------------
    cw = compute_class_weight(class_weight="balanced",
                              classes=np.unique(y_train), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    if "happy" in CLASSES:
        class_weight[CLASSES.index("happy")] *= 1.25
    if "sad" in CLASSES:
        class_weight[CLASSES.index("sad")] *= 1.15

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

    best_path = MODEL_DIR / "mlp_best_70_15_15_dedup.keras"
    callbacks = [
        ModelCheckpoint(str(best_path), monitor="val_accuracy", save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5, verbose=1),
    ]
    t0 = time.time()
    history = model.fit(
        X_train, y_train_cat, validation_data=(X_val, y_val_cat),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
        class_weight=class_weight, verbose=1,
    )
    mlp_seconds = round(time.time() - t0, 2)
    model.save(str(MODEL_DIR / "mlp_final_70_15_15_dedup.keras"))

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("YSA 70/15/15 - Loss")
    plt.savefig(OUT_DIR / "70_15_15_loss_dedup.png", dpi=200); plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title("YSA 70/15/15 - Accuracy")
    plt.savefig(OUT_DIR / "70_15_15_accuracy_dedup.png", dpi=200); plt.close()

    best = tf.keras.models.load_model(str(best_path))
    probs_mlp = best.predict(X_test, verbose=0)
    pred_mlp = np.argmax(probs_mlp, axis=1)

    # ---------------- RF-B (MLP ile ESLES) ----------------
    rf_b = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
    t1 = time.time()
    rf_b.fit(X_train, y_train)
    rf_seconds = round(time.time() - t1, 2)
    joblib.dump(rf_b, MODEL_DIR / "rf_70_15_15_matched_dedup.joblib")
    probs_rf = rf_b.predict_proba(X_test)
    pred_rf = rf_b.predict(X_test)

    # Eslik dogrulamasi: ayni satirlar, ayni sira
    ids_mlp = df.loc[mask_test, "source_id"].to_numpy()
    ids_rf = df.loc[mask_test, "source_id"].to_numpy()
    if not np.array_equal(ids_mlp, ids_rf):
        raise AssertionError("RF-B ve MLP test source_id dizileri farkli!")
    if len(ids_mlp) != EXPECTED_TEST_ROWS:
        raise AssertionError(
            f"Test satir sayisi {EXPECTED_TEST_ROWS} beklenirken {len(ids_mlp)}"
        )

    # ---------------- METRIKLER ----------------
    rows = []
    for name, pred in [("MLP", pred_mlp), ("RF-B (70/15/15 matched)", pred_rf)]:
        m = metrics(y_test, pred)
        m.update({"model": name, "evaluation": "70_15_15_TESS_YAF_unseen_speaker",
                  "n_test": len(y_test), "test_source": "TESS", "test_group": "TESS_YAF"})
        rows.append(m)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "70_15_15_karsilastirma_metrikleri_dedup.csv", index=False)

    with open(OUT_DIR / "70_15_15_label_map_dedup.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)

    plot_cm(confusion_matrix(y_test, pred_mlp, labels=range(4)), "mlp")
    plot_cm(confusion_matrix(y_test, pred_rf, labels=range(4)), "rf_matched")

    # Tahminler (McNemar/ROC icin girdi; simdi analiz edilmez)
    for name, pred, prob in [("mlp", pred_mlp, probs_mlp), ("rf_matched", pred_rf, probs_rf)]:
        out = pd.DataFrame({
            "source_id": ids_mlp,
            "group_id": df.loc[mask_test, "group_id"].to_numpy(),
            "source": df.loc[mask_test, "source"].to_numpy(),
            "y_true": y_test,
            "y_pred": pred,
        })
        for i, cls in enumerate(CLASSES):
            out[f"prob_{cls}"] = prob[:, i]
        out.to_csv(CMP_DIR / f"70_15_15_{name}_test_predictions_dedup.csv", index=False)

    artifact = {
        "evaluation": "70_15_15_TESS_YAF_unseen_speaker",
        "test_note": "Test kumesinin tamami TESS YAF konusmacisindan; genel basari degildir.",
        "seed": SEED,
        "features": len(FEATURE_COLS),
        "epochs_requested": EPOCHS,
        "epochs_run": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "train_rows": int(mask_train.sum()),
        "val_rows": int(mask_val.sum()),
        "test_rows": int(mask_test.sum()),
        "scaler_fit_on": "70/15/15 train only",
        "models": {
            "mlp": {"file": "sonuclar_v2_dedup/modeller/mlp_best_70_15_15_dedup.keras",
                    "sha256": sha256(best_path), "seconds": mlp_seconds},
            "rf_b": {"file": "sonuclar_v2_dedup/modeller/rf_70_15_15_matched_dedup.joblib",
                     "sha256": sha256(MODEL_DIR / "rf_70_15_15_matched_dedup.joblib"),
                     "seconds": rf_seconds, "note": "RF-A'dan ayri model; MLP ile ayni train/test"},
            "scaler": {"file": "sonuclar_v2_dedup/modeller/scaler_70_15_15_dedup.joblib",
                       "sha256": sha256(MODEL_DIR / "scaler_70_15_15_dedup.joblib")},
        },
        "versions": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "tensorflow": tf.__version__,
        },
    }
    with open(AUDIT_DIR / "artifact_manifest_dedup.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print("\n=== 70/15/15 sonuclar (TESS YAF unseen speaker) ===")
    print(metrics_df[["model", "n_test", "accuracy", "f1_macro", "precision_macro", "recall_macro"]].to_string(index=False))
    print(f"\nMLP sure: {mlp_seconds} sn | RF-B sure: {rf_seconds} sn | epoch: {artifact['epochs_run']}")
    print("Tahmin dosyalari kaydedildi; McNemar/ROC/CV CALISTIRILMADI.")


if __name__ == "__main__":
    main()
