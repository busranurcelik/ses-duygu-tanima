"""
v2 / adim 2: train-only augmentation ve 184 ozellik cikarimi.

Kurallar:
  * Augmentation YALNIZCA ilgili degerlendirmede `train` olan kayitlara uygulanir.
    (80/20 veya 70/15/15 icinde en az birinde train ise uretilir.)
  * val/test satirlari yalnizca `orig` kalir; aug_id != 'orig' olamaz.
  * Bir aug satiri diger degerlendirmede train degilse o degerlendirme icin
    `split_* = "EXCLUDED"` isaretlenir; boylece ikisi de temiz filtrelenir.
  * Seed: sha256(source_id + aug_id + 'v2') -> deterministik, tekrarlanabilir.
  * Scaler bu scriptte uretilmez; egitim scripti yalnizca train'e fit edecek.
Cikti: final_oznitelik_seti_v2.csv (Git-ignored, buyuk dosya)
       sonuclar_v2/augmentation_dagilim.csv (kucuk, yolsuz ozet)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "manifest_v2_local.csv"
OUTPUT_FILE = ROOT / "final_oznitelik_seti_v2.csv"
REPORT_FILE = ROOT / "sonuclar_v2" / "augmentation_dagilim.csv"

SAMPLE_RATE = 22050
DURATION = 2.5

NON_BNC_AUG = ["noise", "pitch_up"]
BNC_AUG = ["noise", "pitch_up", "pitch_down", "stretch_0.9", "stretch_1.1", "noise2"]
EVAL_SPLITS = [("80_20", "split_80_20"), ("70_15_15", "split_70_15_15")]


# ---------------------------------------------------------------------------
# Ozellik cikarimi (v1 ile birebir ayni)
# ---------------------------------------------------------------------------
def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    y = librosa.util.normalize(y)
    return librosa.util.fix_length(y, size=int(DURATION * sr))


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = preprocess_audio(y, sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return np.hstack([mfcc, chroma, mel,
                      np.mean(rms), np.std(rms),
                      np.mean(zcr), np.std(zcr)])


# ---------------------------------------------------------------------------
# Seedli augmentation
# ---------------------------------------------------------------------------
def rng_for(source_id: str, aug_id: str) -> np.random.Generator:
    key = f"{source_id}|{aug_id}|v2".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)


def add_noise(y: np.ndarray, rng: np.random.Generator, level: float = 0.015) -> np.ndarray:
    amp = level * rng.uniform() * np.max(np.abs(y))
    return y + amp * rng.normal(size=len(y))


def augment(y: np.ndarray, sr: int, aug_id: str,
            source_id: str) -> np.ndarray:
    rng = rng_for(source_id, aug_id)
    if aug_id == "orig":
        return y
    if aug_id == "noise":
        return add_noise(y, rng, 0.015)
    if aug_id == "pitch_up":
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1)
    if aug_id == "pitch_down":
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)
    if aug_id == "stretch_0.9":
        return librosa.effects.time_stretch(y=y, rate=0.9)
    if aug_id == "stretch_1.1":
        return librosa.effects.time_stretch(y=y, rate=1.1)
    if aug_id == "noise2":
        # v1'deki "cift noise" davranisi; ayni rng dizisi kullanilir.
        y2 = add_noise(y, rng, 0.02)
        return add_noise(y2, rng, 0.02)
    raise ValueError(f"Bilinmeyen aug_id: {aug_id}")


# ---------------------------------------------------------------------------
# Dogrulama
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c.isdigit()]
    if len(feature_cols) != 184:
        raise AssertionError(f"184 ozellik beklenirken {len(feature_cols)} bulundu.")

    values = df[feature_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("Ozelliklerde NaN/Inf var.")

    checks = {}
    for eval_name, col in EVAL_SPLITS:
        train = df[df[col] == "train"]
        eval_rows = df[df[col].isin(["val", "test"])]
        excluded = df[df[col] == "EXCLUDED"]

        # val/test kumelerinde augmentation OLAMAZ
        if (eval_rows["aug_id"] != "orig").any():
            raise AssertionError(f"{eval_name}: val/test icinde augmentation var.")
        # EXCLUDED satirlari yalnizca augmentation varyantlari olabilir
        if (excluded["aug_id"] == "orig").any():
            raise AssertionError(f"{eval_name}: orig satir EXCLUDED olarak isaretlendi.")
        if set(train["group_id"]) & set(eval_rows["group_id"]):
            raise AssertionError(f"{eval_name}: grup sizintisi.")
        if set(train["source_id"]) & set(eval_rows["source_id"]):
            raise AssertionError(f"{eval_name}: source_id sizintisi.")

        # Her source_id icin tek bir "gercek" rol (train/val/test) olmali
        real_roles = df[df[col] != "EXCLUDED"].groupby("source_id")[col].nunique()
        if (real_roles > 1).any():
            raise AssertionError(f"{eval_name}: source_id farkli split'lere dagildi.")

        aug_in_train = sorted(set(train["aug_id"]) - {"orig"})
        checks[eval_name] = {
            "n_train_rows": len(train),
            "n_eval_rows": len(eval_rows),
            "n_excluded_rows": len(excluded),
            "train_aug_ids": aug_in_train,
            "eval_all_orig": True,
            "group_leak": False,
            "source_leak": False,
        }
    return checks


def main() -> None:
    if not INPUT_FILE.is_file():
        print("HATA: manifest_v2_local.csv yok. Once 1_veri_hazirlama_v2.py calistirin.")
        sys.exit(1)

    manifest = pd.read_csv(INPUT_FILE, dtype={"source_id": str, "group_id": str})
    print(f"Manifest kaydi: {len(manifest)}")

    # Bir kayit, iki degerlendirmenin en az birinde train ise augmentation alir.
    is_train = np.zeros(len(manifest), dtype=bool)
    for _, col in EVAL_SPLITS:
        is_train |= (manifest[col] == "train").to_numpy()
    print(f"Augmentation alacak kayit: {int(is_train.sum())}")

    rows = []
    errors = []
    for idx, rec in enumerate(manifest.itertuples(index=False), start=1):
        path = rec.path
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE)
        except Exception as exc:
            errors.append((path, str(exc)))
            if len(errors) <= 5:
                print(f"HATA ({path}): {exc}")
            continue

        aug_ids = ["orig"]
        if is_train[idx - 1]:
            aug_ids += BNC_AUG if rec.source == "BNC" else NON_BNC_AUG

        for aug_id in aug_ids:
            y_aug = augment(y, sr, aug_id, rec.source_id)
            feats = extract_features(y_aug, sr)
            row = {
                "source_id": rec.source_id,
                "group_id": rec.group_id,
                "source": rec.source,
                "label": rec.label,
                "aug_id": aug_id,
                "split_80_20": rec.split_80_20,
                "split_70_15_15": rec.split_70_15_15,
                "labels": rec.label,
            }
            # Bir satir, aug_id != 'orig' oldugu degerlendirmede train degilse
            # o degerlendirme icin tamamen dislanir. Boylece hicbir val/test
            # kumesi augmented satir icermez.
            if aug_id != "orig":
                for _, col in EVAL_SPLITS:
                    if row[col] != "train":
                        row[col] = "EXCLUDED"
            # 0..183 ozellik kolonlari
            row.update({str(i): float(v) for i, v in enumerate(feats)})
            rows.append(row)

        if idx % 500 == 0:
            print(f"Islenen: {idx}/{len(manifest)} | uretilen satir: {len(rows)}", end="\r")

    if errors:
        raise RuntimeError(f"{len(errors)} WAV okunamadi; cikti uretilmedi.")

    out = pd.DataFrame(rows)
    checks = validate(out)

    out.to_csv(OUTPUT_FILE, index=False)

    summary = []
    for (source, aug_id), n in out.groupby(["source", "aug_id"]).size().items():
        summary.append({"source": source, "aug_id": aug_id, "n_rows": int(n)})
    pd.DataFrame(summary).sort_values(["source", "aug_id"]).to_csv(REPORT_FILE, index=False)

    print("\n\n=== v2 augmentation tamamlandi ===")
    print(f"Cikti: {OUTPUT_FILE.name} | satir: {len(out)} | kolon: {out.shape[1]}")
    print(f"Ozellik: {out.shape[1] - 8} (184 beklenir)")
    for eval_name, info in checks.items():
        print(f"{eval_name}: train_rows={info['n_train_rows']} "
              f"eval_rows={info['n_eval_rows']} "
              f"excluded_rows={info['n_excluded_rows']} "
              f"train_aug={info['train_aug_ids']} "
              f"eval_all_orig={info['eval_all_orig']}")
    print("Tum augmentation/sizdirma kontrolleri PASS")


if __name__ == "__main__":
    main()
