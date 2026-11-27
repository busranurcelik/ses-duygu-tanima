"""
v2 / adim 3a: 80/20 split ile klasik modeller (RF-A dahil).

Kurallar:
  * split_80_20 == 'train' / 'test' disindaki hicbir satir kullanilmaz (EXCLUDED).
  * Ozellikler ISIMLE secilir: '0'..'183'. Metadata asla X'e girmez.
  * Giris: final_oznitelik_seti_v2_dedup.csv (194 kolon: 10 metadata + 184 ozellik).
  * RF-A (80/20) -> sonuclar_v2_dedup/modeller/rf_80_20_dedup.joblib.
  * 70/15/15 icin kullanilacak RF-B bu scriptte EGITILMEZ; 4_ysa_modeli_v2.py'de
    MLP ile ayni train uzerinde ayri bir model olarak egitilir.
  * Butun ciktilar sonuclar_v2_dedup/ altinadir; eski sonuclar_v2/ ASLA yazilmaz.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"
RESULT_DIR = ROOT / "sonuclar_v2_dedup" / "klasik"
MODEL_DIR = ROOT / "sonuclar_v2_dedup" / "modeller"
AUDIT_DIR = ROOT / "sonuclar_v2_dedup"
FEATURE_COLS = [str(i) for i in range(184)]
METADATA_COLS = ["source_id", "group_id", "source", "label", "aug_id",
                 "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]
CLASSES = ["angry", "fear", "happy", "sad"]
SPLIT_COL = "split_80_20"
RANDOM_STATE = 42
EXPECTED_TOTAL_COLS = 194

for d in (RESULT_DIR, MODEL_DIR, AUDIT_DIR):
    d.mkdir(parents=True, exist_ok=True)


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
    bad = df[~df.apply(
        lambda r: r["group_id"].startswith(r["source"] + "_"), axis=1
    )]
    if len(bad):
        raise AssertionError(f"{len(bad)} group_id kaynak on ekini tasimiyor.")
    shared = df.groupby("group_id")["source"].nunique()
    if (shared > 1).any():
        raise AssertionError("Bir group_id birden fazla kaynaga ait.")
    print(f"group_id dogrulandı: {df.group_id.nunique()} grup, hepsi kaynak on ekli ve benzersiz")


def audit_filter(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    check_group_identity(df)

    if (df[SPLIT_COL] == "EXCLUDED").any():
        raise AssertionError("EXCLUDED satirlari filtrelenmeden kaldi.")
    if len(df) != len(raw[raw[SPLIT_COL].isin(["train", "test"])]):
        raise AssertionError("80/20 filtresi beklenen satir sayisini vermedi.")

    train = df[df[SPLIT_COL] == "train"]
    test = df[df[SPLIT_COL] == "test"]
    if len(test) == 0 or len(train) == 0:
        raise AssertionError("Train veya test bos.")

    if (test["aug_id"] != "orig").any():
        raise AssertionError("80/20 test icinde augmentation var.")
    if set(train["group_id"]) & set(test["group_id"]):
        raise AssertionError("80/20 grup sizintisi.")
    if set(train["source_id"]) & set(test["source_id"]):
        raise AssertionError("80/20 source_id sizintisi.")
    if (df.groupby("source_id")[SPLIT_COL].nunique() > 1).any():
        raise AssertionError("Ayni source_id 80/20'de iki split'e dagildi.")

    roles = df[df[SPLIT_COL] != "EXCLUDED"].groupby("source_id")[SPLIT_COL].nunique()
    if (roles > 1).any():
        raise AssertionError("Ayni kayit hem train hem test gorunuyor.")

    return {
        "script": "3_modelleri_egit_v2.py",
        "split_column": SPLIT_COL,
        "input_rows": len(raw),
        "excluded_dropped": int((raw[SPLIT_COL] == "EXCLUDED").sum()),
        "train_rows": len(train),
        "val_rows": 0,
        "test_rows": len(test),
        "n_features": len(FEATURE_COLS),
        "train_aug_ids": ",".join(sorted(set(train.aug_id))),
        "eval_aug_ids": ",".join(sorted(set(test.aug_id))),
        "eval_nonorig_rows": int((test.aug_id != "orig").sum()),
        "group_overlap": len(set(train.group_id) & set(test.group_id)),
        "source_overlap": len(set(train.source_id) & set(test.source_id)),
        "test_sources": ",".join(sorted(test.source.unique())),
        "status": "PASS",
    }


def save_audit(audit: dict) -> None:
    cols = ["script", "split_column", "input_rows", "excluded_dropped",
            "train_rows", "val_rows", "test_rows", "n_features",
            "train_aug_ids", "eval_aug_ids", "eval_nonorig_rows",
            "group_overlap", "source_overlap", "test_sources", "status"]
    path = AUDIT_DIR / "filtre_dogrulama_80_20.csv"
    pd.DataFrame([audit], columns=cols).to_csv(path, index=False, mode="w", encoding="utf-8")

    lines = [
        "# v2 dedup / 80_20 filtre ve veri ayirimi audit raporu",
        "",
        "| script | split kolonu | girdi | EXCLUDED atilan | train | val | test | "
        "ozellik | eval aug | grup kesisimi | source kesisimi | durum |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
        f"| {audit['script']} | {audit['split_column']} | {audit['input_rows']} | "
        f"{audit['excluded_dropped']} | {audit['train_rows']} | {audit['val_rows']} | "
        f"{audit['test_rows']} | {audit['n_features']} | {audit['eval_aug_ids']} | "
        f"{audit['group_overlap']} | {audit['source_overlap']} | {audit['status']} |",
        "",
        f"- 80/20 test kaynaklari: {audit['test_sources']} (TESS bu split'te yok).",
        "- Ozellikler 184 kolon `0..183` olarak isimle secildi; metadata X'e girmedi.",
        "- Scaler kullanan modeller pipeline icinde yalnizca train'e fit edildi.",
        "- RF-A (80/20) ile RF-B (70/15/15, MLP ile eslesmis) AYRI modellerdir.",
        "",
    ]
    (AUDIT_DIR / "filtre_dogrulama_80_20.md").write_text("\n".join(lines), encoding="utf-8")


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


def plot_cm(cm: np.ndarray, name: str, normalized: bool) -> None:
    plt.figure(figsize=(7, 6))
    if normalized:
        data = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fmt, cmap, suffix = ".2f", "Greens", "normalized"
    else:
        data, fmt, cmap, suffix = cm, "d", "Blues", "raw"
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"{name} - 80/20 Confusion Matrix ({suffix})")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    plt.savefig(RESULT_DIR / f"80_20_{slug}_cm_{suffix}.png", dpi=200)
    plt.close()


def main() -> None:
    print("=== v2 / 80_20 klasik modeller ===")
    raw = pd.read_csv(INPUT_FILE, dtype={c: str for c in METADATA_COLS})
    check_schema(raw)
    missing = [c for c in FEATURE_COLS + METADATA_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Eksik kolonlar: {missing[:10]}")

    df = raw[raw[SPLIT_COL].isin(["train", "test"])].reset_index(drop=True)
    audit = audit_filter(raw, df)
    save_audit(audit)
    print("Audit:", audit)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(df["labels"].to_numpy())
    if list(le.classes_) != CLASSES:
        raise AssertionError(f"Beklenen sinif sirasi {CLASSES}, bulunan {list(le.classes_)}")
    test_mask = (df[SPLIT_COL] == "test").to_numpy()
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    classifiers = {
        "1-Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "2-SVM": make_pipeline(StandardScaler(), SVC_RBF()),
        "3-KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "4-Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1500, solver="lbfgs", random_state=RANDOM_STATE)),
        "5-Random Forest (80/20)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    ensembles = {
        "Ensemble-Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    voting = VotingClassifier(
        estimators=[("svm", classifiers["2-SVM"]),
                    ("rf", classifiers["5-Random Forest (80/20)"]),
                    ("lr", classifiers["4-Logistic Regression"])],
        voting="soft",
    )
    ensembles["Ensemble-Voting Classifier"] = voting
    all_models = {**classifiers, **ensembles}

    results, per_source = [], []
    for name, model in all_models.items():
        start = time.time()
        print(f"--> {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = metrics(y_test, y_pred)
        m["model"] = name
        m["fit_predict_seconds"] = round(time.time() - start, 2)
        results.append(m)

        for src in sorted(df.loc[test_mask, "source"].unique()):
            mask_src = (df.loc[test_mask, "source"] == src).to_numpy()
            sm = metrics(y_test[mask_src], y_pred[mask_src])
            sm.update({"model": name, "source": src, "n_test": int(mask_src.sum())})
            per_source.append(sm)

        cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASSES)))
        plot_cm(cm, name, False)
        plot_cm(cm, name, True)

        if name == "5-Random Forest (80/20)":
            joblib.dump(model, MODEL_DIR / "rf_80_20_dedup.joblib")
            np.savez_compressed(
                RESULT_DIR / "80_20_rf_test_predictions_dedup.npz",
                source_id=df.loc[test_mask, "source_id"].to_numpy(),
                y_true=y_test, y_pred=y_pred,
            )

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(RESULT_DIR / "80_20_genel_sonuc_tablosu_dedup.csv", index=False)
    pd.DataFrame(per_source).to_csv(RESULT_DIR / "80_20_kaynak_bazli_sonuclar_dedup.csv", index=False)

    with open(AUDIT_DIR / "model_kimliklari_80_20.json", "w", encoding="utf-8") as f:
        json.dump({
            "rf_80_20": {
                "file": "sonuclar_v2_dedup/modeller/rf_80_20_dedup.joblib",
                "split_column": "split_80_20",
                "train_rows": int((~test_mask).sum()),
                "test_rows": int(test_mask.sum()),
                "features": len(FEATURE_COLS),
                "random_state": RANDOM_STATE,
            },
            "not": "RF-B (70/15/15, MLP ile eslesmis) 4_ysa_modeli_v2.py'de ayri egitilir."
        }, f, indent=2, ensure_ascii=False)

    print("\n=== 80/20 sonuclar ===")
    print(results_df[["model", "accuracy", "f1_macro", "precision_macro", "recall_macro"]].to_string(index=False))
    print("\n=== RF 80/20 kaynak bazli ===")
    rf_src = pd.DataFrame(per_source)
    rf_src = rf_src[rf_src.model == "5-Random Forest (80/20)"]
    print(rf_src[["source", "n_test", "accuracy", "f1_macro"]].to_string(index=False))
    print(f"\nRF-A kaydedildi: {MODEL_DIR / 'rf_80_20_dedup.joblib'}")
    print("Audit PASS; 70/15/15 ve McNemar asamasi CALISTIRILMADI.")


def SVC_RBF():
    from sklearn.svm import SVC
    return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)


if __name__ == "__main__":
    main()
