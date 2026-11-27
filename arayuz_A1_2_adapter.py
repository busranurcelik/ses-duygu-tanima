"""A1.2 RF-B modeli için GUI çıkarım adaptörü.

Bu modül eğitimde kullanılan 192 özellik zincirini birebir uygular:
A0'ın 184 özelliği + 8 F0/pitch özelliği + 70/15/15 train medyanlarıyla
sentinel imputasyonu. Modeli yeniden eğitmez; yalnızca kaydedilmiş
model/scaler/label map/medyan dosyalarını okur.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import librosa
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "A1_2" / "70_15_15"

MODEL_PATH = ASSET_DIR / "modeller" / "rf_70_15_15_matched_A1_2.joblib"
SCALER_PATH = ASSET_DIR / "modeller" / "scaler_A1_2.joblib"
MEDIAN_PATH = ASSET_DIR / "f0_imputasyon_degerleri.csv"
LABEL_MAP_PATH = ASSET_DIR / "ysa" / "70_15_15_label_map_A1_2.json"

# Eğitimle aynı ses/pencere ayarları.
SAMPLE_RATE = 22050
DURATION = 2.5
N_SAMPLES = int(DURATION * SAMPLE_RATE)       # 55125

# A0 özellik blokları: 40 MFCC + 12 chroma + 128 mel + 4 RMS/ZCR.
N_A0_FEATURES = 184
N_F0_FEATURES = 8
N_FEATURES = N_A0_FEATURES + N_F0_FEATURES
SENTINEL_IDX = np.arange(184, 191)            # f0_mean .. f0_slope

# A1.2 pYIN ayarları.
FMIN = 60.0
FMAX = 400.0
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MIN_VOICED = 3


def _load_medians() -> np.ndarray:
    """70/15/15 train'den kaydedilmiş 7 medyanı 184..190 sırasıyla döndürür."""
    try:
        with MEDIAN_PATH.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:
        raise RuntimeError(f"F0 medyan dosyası okunamadı ({MEDIAN_PATH.name}): {exc}") from exc

    try:
        med = {int(r["kolon"]): float(r["train_medyan"]) for r in rows}
        values = np.array([med[i] for i in range(184, 191)], dtype=float)
    except Exception as exc:
        raise RuntimeError(f"F0 medyan dosyası beklenen biçimde değil: {exc}") from exc

    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError("F0 imputasyon medyanları eksik veya sonlu değil.")
    return values


def _load_label_map() -> list[str]:
    try:
        with LABEL_MAP_PATH.open("r", encoding="utf-8") as fh:
            labels = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Label map okunamadı ({LABEL_MAP_PATH.name}): {exc}") from exc
    if not isinstance(labels, list) or len(labels) != 4 or not all(isinstance(x, str) for x in labels):
        raise RuntimeError("Label map 4 string etiket içermeli.")
    return labels


def _validate_assets(model, scaler, labels: list[str]) -> None:
    model_n = getattr(model, "n_features_in_", None)
    scaler_n = getattr(scaler, "n_features_in_", None)
    if model_n != N_FEATURES:
        raise RuntimeError(f"Model {model_n} özellik bekliyor; beklenen {N_FEATURES}.")
    if scaler_n != N_FEATURES:
        raise RuntimeError(f"Scaler {scaler_n} özellik bekliyor; beklenen {N_FEATURES}.")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Yüklenen RF modelinde predict_proba yok.")
    classes = getattr(model, "classes_", None)
    if classes is None or list(classes) != list(range(len(labels))):
        raise RuntimeError(f"Model sınıf sırası label map ile uyuşmuyor: {classes} / {labels}")


def load_assets():
    """RF-B, scaler, label map ve F0 medyanlarını yükleyip doğrular."""
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"RF-B model dosyası yüklenemedi ({MODEL_PATH.name}): {exc}") from exc

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as exc:
        raise RuntimeError(f"Scaler dosyası yüklenemedi ({SCALER_PATH.name}): {exc}") from exc

    labels = _load_label_map()
    medians = _load_medians()
    _validate_assets(model, scaler, labels)
    return model, scaler, labels, medians


def preprocess_audio(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A0/A1.2 ile aynı: 22050 Hz, normalize, ilk 2,5 s (55125 örnek)."""
    # Eğitimdeki librosa.load çıktısı float32'dir; dtype'ı değiştirmeyerek
    # normalize/FFT sonuçlarının birebir aynı olmasını koruruz.
    y = np.asarray(y)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    return librosa.util.fix_length(y, size=N_SAMPLES)


def extract_a0_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A0'ın 184 özelliği; sıra eğitim/GUI zinciriyle birebir aynıdır."""
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)          # 40
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)             # 12
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1) # 128
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats = np.hstack([
        mfcc,
        chroma,
        mel,
        np.mean(rms), np.std(rms),
        np.mean(zcr), np.std(zcr),
    ])
    if feats.shape != (N_A0_FEATURES,):
        raise RuntimeError(f"A0 özellik sayısı hatalı: {feats.shape}")
    return feats


def extract_f0_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """A1.2'deki pYIN/F0 çıkarımının aynısı (imputasyon uygulanmadan)."""
    pyin_error = ""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=FMIN, fmax=FMAX, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
        )
    except Exception as exc:
        pyin_error = f"{type(exc).__name__}: {exc}"[:300]
        f0 = np.full(1, np.nan, dtype=float)
        voiced_flag = np.zeros(1, dtype=bool)

    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    n_frames = int(len(f0))
    finite = np.isfinite(f0)
    in_range = (f0 >= FMIN) & (f0 <= FMAX)
    valid = voiced_flag & finite & in_range & (f0 > 0)
    n_valid = int(valid.sum())
    voiced_ratio = float(n_valid / n_frames) if n_frames else 0.0

    fallback = False
    slope = 0.0
    if pyin_error:
        fallback = True
        stats = [0.0] * 7
    elif n_valid >= MIN_VOICED:
        vals = f0[valid]
        times = np.nonzero(valid)[0] * (HOP_LENGTH / sr)
        try:
            slope = float(np.polyfit(times, vals, 1)[0])
            stats = [
                float(vals.mean()), float(np.median(vals)), float(vals.std()),
                float(vals.min()), float(vals.max()), float(vals.max() - vals.min()),
                slope,
            ]
        except Exception:
            fallback = True
            stats = [0.0] * 7
    else:
        fallback = True
        stats = [0.0] * 7

    vector = np.asarray(stats + [voiced_ratio], dtype=float)
    if vector.shape != (N_F0_FEATURES,):
        raise RuntimeError(f"F0 özellik sayısı hatalı: {vector.shape}")
    return {
        "vector": vector,
        "fallback": fallback,
        "n_frames": n_frames,
        "n_valid": n_valid,
        "voiced_ratio": voiced_ratio,
        "pyin_error": pyin_error,
    }


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE, medians: np.ndarray | None = None,
                     return_debug: bool = False):
    """192 özellik üretir; sentinel F0 alanlarını eğitim medyanlarıyla doldurur."""
    y_win = preprocess_audio(y, sr)
    a0 = extract_a0_features(y_win, sr)
    f0_info = extract_f0_features(y_win, sr)
    f0 = f0_info["vector"].copy()

    fallback = bool(f0_info["fallback"])
    if fallback:
        if medians is None:
            raise ValueError("F0 fallback tespit edildi ama imputasyon medyanları yüklenmedi.")
        f0[:7] = medians

    features = np.concatenate([a0, f0])
    if features.shape != (N_FEATURES,):
        raise RuntimeError(f"Toplam özellik sayısı hatalı: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("Özellik vektöründe NaN/Inf var.")

    if return_debug:
        return features, fallback, {
            "a0": a0,
            "f0_raw": f0_info["vector"],
            "f0_final": f0,
            "n_frames": f0_info["n_frames"],
            "n_valid": f0_info["n_valid"],
            "voiced_ratio": f0_info["voiced_ratio"],
            "pyin_error": f0_info["pyin_error"],
        }
    return features, fallback


def predict_audio(model, scaler, labels: list[str], medians: np.ndarray,
                  y: np.ndarray, sr: int = SAMPLE_RATE):
    """Ham ses dizisinden olasılık, 192 özellik ve fallback bilgisi döndürür."""
    features, fallback = extract_features(y, sr, medians)
    X = scaler.transform(features.reshape(1, -1))
    probs = np.asarray(model.predict_proba(X)[0], dtype=float)
    if probs.shape != (len(labels),) or not np.isfinite(probs).all():
        raise RuntimeError("Model olasılık çıktısı beklenen biçimde değil.")
    return probs, features, fallback
