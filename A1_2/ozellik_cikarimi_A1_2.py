"""
A1.2 / adim 2: A0'in 184 ozelligine YALNIZCA 8 F0/Pitch ozelligi ekler.
HAZIRLIK ASAMASI — egittim/CV/tahmin/model/PCA YOK.

Kurallar:
  * Ilk 184 ozellik A0 CSV'sinden BIREBIR kopyalanir (yeniden tanimlanmaz).
  * 8 yeni ozellik (kolon 184..191) her (source_id, aug_id) icin GERCEK sestEN,
    A0 ile ayni zincirle uretilir: librosa.load(tam dosya, sr=22050)
    -> augment() -> normalize() -> ilk 2,5 s / 55.125 ornege sabitleme -> pyin.
  * pYIN parametreleri sabit: sr=22050, fmin=60, fmax=400,
    frame_length=2048, hop_length=512. 2,5 s pencerede 108 analiz karesi.
  * Istatistikler yalnizca voiced_flag=1 VE sonlu VE 0<f0 VE [60,400] araligindaki
    karelerden hesaplanir. Sessizlik/voiceless/NaN/Inf asla 0 Hz sayilmaz.
  * Yetersiz voiced (<3) veya pyin hatasi: 7 F0 istatistigi icin 0.0 sentinel;
    voiced_ratio gercek payda uzerinden uretilir. Her olay kaydedilir.
  * Etiket, tahmin, fold, split veya test sonucu F0 cikariminda KULLANILMAZ.
  * Scaler degisikligi YOK; augmentation train-only; val/test yalnizca orig.
  * Butun ciktilar yalnizca A1_2/ altinda. A0/A1.1/A3/GUI ASLA yazilmaz.

Ciktilar: final_oznitelik_seti_A1_2.csv, f0_istatistikleri.csv,
f0_kalite_kontrol.csv, f0_ozellik_tanimlari.csv, f0_sinif_kaynak_istatistikleri.csv,
f0_augmentasyon_kontrolu.csv, A0_karsilastirma_A1_2.csv, leakage_kontrolleri.csv,
augmentation_plani.csv, augmentation_dagilim.csv, kaynak_sayilari.csv,
sinif_sayilari.csv, split_sayilari.csv, fold_sayilari.csv,
sha256_source_id_kontrolleri.csv, veri_hazirlama_raporu.md,
ozellik_cikarimi_raporu.md, f0_dagilim_raporu.md.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import librosa
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
WORK_DIR = SCRIPT_DIR

DRAFT_FILE = ROOT / "manifest_v2_dedup_draft.csv"
QC_FILE = ROOT / "manifest_v2_dedup_review_qc.csv"
A0_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"

OUTPUT_FILE = WORK_DIR / "final_oznitelik_seti_A1_2.csv"
F0_DETAIL_FILE = WORK_DIR / "f0_istatistikleri.csv"
CACHE_FILE = WORK_DIR / "f0_raw_results.pkl"
QUALITY_FILE = WORK_DIR / "f0_kalite_kontrol.csv"
DEF_FILE = WORK_DIR / "f0_ozellik_tanimlari.csv"
CLASS_SRC_FILE = WORK_DIR / "f0_sinif_kaynak_istatistikleri.csv"
AUG_F0_FILE = WORK_DIR / "f0_augmentasyon_kontrolu.csv"
CMP_FILE = WORK_DIR / "A0_karsilastirma_A1_2.csv"
LEAK_FILE = WORK_DIR / "leakage_kontrolleri.csv"
PLAN_FILE = WORK_DIR / "augmentation_plani.csv"
AUG_DIST_FILE = WORK_DIR / "augmentation_dagilim.csv"
SOURCE_FILE = WORK_DIR / "kaynak_sayilari.csv"
CLASS_FILE = WORK_DIR / "sinif_sayilari.csv"
SPLIT_FILE = WORK_DIR / "split_sayilari.csv"
FOLD_FILE = WORK_DIR / "fold_sayilari.csv"
SHA_FILE = WORK_DIR / "sha256_source_id_kontrolleri.csv"
DATA_REPORT = WORK_DIR / "veri_hazirlama_raporu.md"
FEAT_REPORT = WORK_DIR / "ozellik_cikarimi_raporu.md"
DIST_REPORT = WORK_DIR / "f0_dagilim_raporu.md"

SAMPLE_RATE = 22050
DURATION = 2.5
N_SAMPLES = int(DURATION * SAMPLE_RATE)          # 55125
FMIN, FMAX = 60.0, 400.0
FRAME_LENGTH, HOP_LENGTH = 2048, 512
MIN_VOICED = 3
N_WORKERS = 12

NON_BNC_AUG = ["noise", "pitch_up"]
BNC_AUG = ["noise", "pitch_up", "pitch_down", "stretch_0.9", "stretch_1.1", "noise2"]
EVAL_SPLITS = [("80_20", "split_80_20"), ("70_15_15", "split_70_15_15")]

A0_FEATURE_COLS = [str(i) for i in range(184)]
F0_COLS = [str(i) for i in range(184, 192)]
FEATURE_COLS = A0_FEATURE_COLS + F0_COLS
META_COLS = [
    "source_id", "group_id", "source", "label", "aug_id",
    "split_80_20", "split_70_15_15", "fold", "labels", "sha256",
]

F0_NAMES = [
    "f0_mean", "f0_median", "f0_std", "f0_min", "f0_max",
    "f0_range", "f0_slope", "voiced_frame_ratio",
]
F0_UNITS = ["Hz", "Hz", "Hz (ddof=0)", "Hz", "Hz", "Hz",
            "Hz/s (gercek zaman)", "oran"]

F0_DETAIL_COLS = [
    "source_id", "group_id", "source", "label", "aug_id", "fold",
    "split_80_20", "split_70_15_15", "n_frames", "n_voiced_raw",
    "n_f0_finite", "n_voiced_valid", "n_voiced_invalid", "voiced_ratio",
    "f0_mean", "f0_median", "f0_std", "f0_min", "f0_max", "f0_range",
    "f0_slope", "fallback", "fallback_reason", "pyin_error",
]

ALL_TARGETS = [
    OUTPUT_FILE, F0_DETAIL_FILE, QUALITY_FILE, DEF_FILE, CLASS_SRC_FILE,
    AUG_F0_FILE, CMP_FILE, LEAK_FILE, PLAN_FILE, AUG_DIST_FILE, SOURCE_FILE,
    CLASS_FILE, SPLIT_FILE, FOLD_FILE, SHA_FILE, DATA_REPORT, FEAT_REPORT,
    DIST_REPORT,
]


# ---------------------------------------------------------------------------
# Yardimci (A0 ile birebir)
# ---------------------------------------------------------------------------
def guard_targets() -> None:
    for p in ALL_TARGETS:
        if p.exists():
            raise SystemExit(f"REFUSE OVERWRITE: {p}")
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{digits}f}" if pd.notna(v) else "")
    lines = ["| " + " | ".join(str(c) for c in d.columns) + " |",
             "|" + "|".join(["---"] * len(d.columns)) + "|"]
    for row in d.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """A0'daki birebir on isleme: normalize + 2,5 s / 55125 ornek."""
    y = librosa.util.normalize(y)
    return librosa.util.fix_length(y, size=N_SAMPLES)


def rng_for(source_id: str, aug_id: str) -> np.random.Generator:
    key = f"{source_id}|{aug_id}|v2".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)


def add_noise(y: np.ndarray, rng: np.random.Generator, level: float = 0.015) -> np.ndarray:
    amp = level * rng.uniform() * np.max(np.abs(y))
    return y + amp * rng.normal(size=len(y))


def augment(y: np.ndarray, sr: int, aug_id: str, source_id: str) -> np.ndarray:
    """A0 augmentation fonksiyonu — degistirilmedi."""
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
        y2 = add_noise(y, rng, 0.02)
        return add_noise(y2, rng, 0.02)
    raise ValueError(f"Bilinmeyen aug_id: {aug_id}")


# ---------------------------------------------------------------------------
# F0 cikarimi
# ---------------------------------------------------------------------------
def extract_f0(y: np.ndarray, sr: int, aug_id: str) -> dict:
    """Bir ses varyantinin 2,5 s penceresinden 8 F0 ozelligi uretir."""
    y_win = preprocess_audio(y, sr)
    pyin_error = ""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_win, fmin=FMIN, fmax=FMAX, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
        )
    except Exception as exc:                       # sessizce gizlenmez
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
    n_voiced_raw = int(voiced_flag.sum())
    n_finite = int(finite.sum())
    n_invalid = int((voiced_flag & ~valid).sum())
    voiced_ratio = float(n_valid / n_frames) if n_frames else 0.0

    fallback = 0
    reason = ""
    slope = 0.0
    if n_valid >= MIN_VOICED:
        vals = f0[valid]
        # Gercek zaman ekseni: kare i -> i * hop_length / sr saniye.
        times = np.nonzero(valid)[0] * (HOP_LENGTH / sr)
        try:
            slope = float(np.polyfit(times, vals, 1)[0])
            stats = [float(vals.mean()), float(np.median(vals)), float(vals.std()),
                     float(vals.min()), float(vals.max()),
                     float(vals.max() - vals.min()), slope]
        except Exception as exc:
            fallback = 1
            reason = f"slope_error: {type(exc).__name__}"[:120]
            stats = [0.0] * 7
    else:
        fallback = 1
        reason = ("pyin_error" if pyin_error else
                  "no_valid_voiced" if n_valid == 0 else
                  f"insufficient_voiced_{n_valid}_lt_{MIN_VOICED}")
        stats = [0.0] * 7

    return {
        "aug_id": aug_id, "n_frames": n_frames, "n_voiced_raw": n_voiced_raw,
        "n_f0_finite": n_finite, "n_voiced_valid": n_valid,
        "n_voiced_invalid": n_invalid, "voiced_ratio": voiced_ratio,
        "f0_mean": stats[0], "f0_median": stats[1], "f0_std": stats[2],
        "f0_min": stats[3], "f0_max": stats[4], "f0_range": stats[5],
        "f0_slope": slope if not fallback else 0.0,
        "voiced_frame_ratio": voiced_ratio,
        "fallback": fallback, "fallback_reason": reason, "pyin_error": pyin_error,
    }


def worker(task: tuple) -> dict:
    """Bir source_id ve tum varyantlari; Pool icinde calisir."""
    source_id, path_str, sha, aug_ids = task
    out: dict = {"source_id": source_id, "sha_ok": False, "load_error": "", "per_aug": {}}
    path = Path(path_str)
    try:
        out["sha_ok"] = sha256_file(path) == sha
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
    except Exception as exc:
        out["load_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out
    for aug_id in aug_ids:
        try:
            y_aug = augment(y, sr, aug_id, source_id)
            out["per_aug"][aug_id] = extract_f0(y_aug, sr, aug_id)
        except Exception as exc:
            # augmentation hatasi da kaydedilir; sessiz gecilmez.
            r = {c: 0.0 for c in F0_NAMES}
            r.update({"aug_id": aug_id, "n_frames": 0, "n_voiced_raw": 0,
                      "n_f0_finite": 0, "n_voiced_valid": 0, "n_voiced_invalid": 0,
                      "voiced_ratio": 0.0, "fallback": 1,
                      "fallback_reason": f"augment_error: {type(exc).__name__}"[:120],
                      "pyin_error": f"{type(exc).__name__}: {exc}"[:300]})
            out["per_aug"][aug_id] = r
    return out


# ---------------------------------------------------------------------------
# Girdi hazirlama
# ---------------------------------------------------------------------------
def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not A0_FILE.is_file():
        raise SystemExit(f"A0 referans dosyasi yok: {A0_FILE}")
    draft = pd.read_csv(DRAFT_FILE, dtype=str)
    qc = pd.read_csv(QC_FILE, dtype=str)
    if not draft.source_id.is_unique:
        raise AssertionError("Manifest source_id tekrari var.")
    review = draft[draft.dedup_status.eq("REVIEW_LABEL")].copy()
    base = draft[~draft.dedup_status.eq("REVIEW_LABEL")].copy()
    if len(review) != 2 or review.sha256.nunique() != 1:
        raise AssertionError("REVIEW_LABEL beklenen 2 kayit / 1 hash degil.")
    if len(set(base.source_id) & set(qc.source_id)):
        raise AssertionError("REVIEW_QC taslakta var.")
    if not base.sha256.is_unique:
        raise AssertionError("Canonical SHA256 tekrari var.")

    a0 = pd.read_csv(A0_FILE, dtype={c: str for c in META_COLS})
    if len(a0) != 23268 or list(a0.columns) != META_COLS + A0_FEATURE_COLS:
        raise AssertionError(f"A0 beklenen 23268 x 194, bulunan {a0.shape}")
    if a0.duplicated(["source_id", "aug_id"]).any():
        raise AssertionError("A0'da tekrarlayan source_id/aug_id.")
    if set(a0.source_id) != set(base.source_id):
        raise AssertionError("A0 source_id kumesi manifest ile ayni degil.")
    if len(set(a0.source_id) & set(review.source_id)):
        raise AssertionError("A0'da REVIEW_LABEL kaydi var.")
    if len(set(a0.source_id) & set(qc.source_id)):
        raise AssertionError("A0'da REVIEW_QC kaydi var.")
    sha_map = dict(zip(base.source_id, base.sha256))
    if not (a0.source_id.map(sha_map) == a0.sha256).all():
        raise AssertionError("A0 sha256 degerleri manifest ile farkli.")

    # Augmentation plani: A0'daki varyant kumeleri manifest kuraliyla birebir mi?
    plan = base[["source_id", "group_id", "source", "label", "split_80_20",
                 "split_70_15_15", "fold", "sha256"]].copy()
    is_train = (plan.split_80_20.eq("train") | plan.split_70_15_15.eq("train")).to_numpy()
    plan["is_train"] = is_train
    plan["aug_ids"] = [
        "orig;" + (";".join(BNC_AUG if s == "BNC" else NON_BNC_AUG) if t else "")
        for s, t in zip(plan.source, is_train)
    ]
    a0_augs = a0.groupby("source_id").aug_id.apply(lambda s: ";".join(sorted(s))).to_dict()
    plan_augs = {r.source_id: ";".join(sorted(x for x in r.aug_ids.split(";") if x))
                 for r in plan.itertuples(index=False)}
    if a0_augs != plan_augs:
        raise AssertionError("A0 augmentation varyantlari manifest planiyla farkli.")
    plan["expected_rows"] = [7 if (s == "BNC" and t) else (3 if t else 1)
                             for s, t in zip(plan.source, is_train)]
    if int(plan.expected_rows.sum()) != len(a0):
        raise AssertionError("Plan satir sayisi A0 satir sayisiyla farkli.")
    return base, review, qc, a0


def make_tasks(a0: pd.DataFrame, base: pd.DataFrame) -> list[tuple]:
    path_map = dict(zip(base.source_id, base.path))
    sha_map = dict(zip(base.source_id, base.sha256))
    tasks = []
    for sid, grp in a0.groupby("source_id", sort=True):
        tasks.append((sid, str(path_map[sid]), sha_map[sid], list(grp.aug_id)))
    return tasks


# ---------------------------------------------------------------------------
# Dogrulama
# ---------------------------------------------------------------------------
def leakage_checks(out: pd.DataFrame) -> list[dict]:
    rows = []
    for eval_name, col in EVAL_SPLITS:
        train = out[out[col].eq("train")]
        ev = out[out[col].isin(["val", "test"])]
        ex = out[out[col].eq("EXCLUDED")]
        ev_aug = int(ev.aug_id.ne("orig").sum())
        ex_orig = int(ex.aug_id.eq("orig").sum())
        grp_leak = len(set(train.group_id) & set(ev.group_id))
        src_leak = len(set(train.source_id) & set(ev.source_id))
        roles = out[out[col].ne("EXCLUDED")].groupby("source_id")[col].nunique()
        role_bad = int((roles > 1).sum())
        ok = not (ev_aug or ex_orig or grp_leak or src_leak or role_bad)
        rows.append({"kapsam": eval_name, "train_rows": len(train), "eval_rows": len(ev),
                     "excluded_rows": len(ex), "eval_aug_rows": ev_aug,
                     "excluded_orig_rows": ex_orig, "group_leak": grp_leak,
                     "source_leak": src_leak, "multi_role_source": role_bad,
                     "sonuc": "PASS" if ok else "FAIL"})
    for k in sorted(out.fold.astype(int).unique()):
        fold_all = out[out.fold.astype(int).eq(k)]
        test = fold_all[fold_all.aug_id.eq("orig")]
        train = out[out.fold.astype(int).ne(k)]
        grp_leak = len(set(train.group_id) & set(test.group_id))
        src_leak = len(set(train.source_id) & set(test.source_id))
        ok = len(test) > 0 and len(train) > 0 and not (grp_leak or src_leak)
        rows.append({"kapsam": f"fold_{k}", "train_rows": len(train), "eval_rows": len(test),
                     "excluded_rows": int(fold_all.aug_id.ne("orig").sum()),
                     "eval_aug_rows": 0, "excluded_orig_rows": 0,
                     "group_leak": grp_leak, "source_leak": src_leak,
                     "multi_role_source": 0, "sonuc": "PASS" if ok else "FAIL"})
    if any(r["sonuc"] != "PASS" for r in rows):
        raise AssertionError("Leakage kontrolu FAIL: " + str([r for r in rows if r["sonuc"] != "PASS"]))
    return rows


def compare_a0(a0: pd.DataFrame, out: pd.DataFrame) -> list[dict]:
    rows = []
    for col in META_COLS:
        ok = bool((a0[col].to_numpy() == out[col].to_numpy()).all())
        rows.append({"kontrol": f"meta_{col}_birebir", "sonuc": ok})
        if not ok:
            raise AssertionError(f"A0/A1.2 metadata farki: {col}")
    a0x = a0[A0_FEATURE_COLS].to_numpy(dtype=float)
    outx = out[A0_FEATURE_COLS].to_numpy(dtype=float)
    md = float(np.abs(a0x - outx).max())
    rows.append({"kontrol": "ilk_184_max_abs_diff", "sonuc": md})
    rows.append({"kontrol": "ilk_184_allclose_atol_1e-10",
                 "sonuc": bool(np.allclose(a0x, outx, rtol=0, atol=1e-10))})
    if md > 1e-10:
        raise AssertionError(f"Ilk 184 ozellik A0'dan sapti: {md}")
    f = out[F0_COLS].to_numpy(dtype=float)
    rows.append({"kontrol": "f0_8_kolon_finite", "sonuc": bool(np.isfinite(f).all())})
    rows.append({"kontrol": "f0_8_kolon_sabit_olmayan",
                 "sonuc": int((np.ptp(f, axis=0) > 0).sum())})
    if not np.isfinite(f).all():
        raise AssertionError("F0 kolonlarinda NaN/Inf.")
    return rows


# ---------------------------------------------------------------------------
# Betimsel F0 analizi (model/test sonucu kullanilmaz)
# ---------------------------------------------------------------------------
def descriptive(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orig = detail[detail.aug_id.eq("orig")].copy()
    orig_valid = orig[orig.fallback.eq(0)].copy()
    # TUM varyantlar: augmentation karsilastirmasi origin ve varyant satirlarini birlikte ister.
    valid = detail[detail.fallback.eq(0)].copy()

    def q(g: pd.DataFrame) -> dict:
        gv = g[g.fallback.eq(0)]
        v = gv.f0_mean.to_numpy(dtype=float)
        if len(v) == 0:
            return {"n": len(g), "n_valid": 0,
                    "f0_mean_ort": float("nan"), "f0_mean_std": float("nan"),
                    "f0_mean_medyan": float("nan"), "f0_mean_p10": float("nan"),
                    "f0_mean_p90": float("nan"),
                    "voiced_oran_ort": float(g.voiced_ratio.mean()) if len(g) else 0.0,
                    "fallback_oran": float(g.fallback.mean()) if len(g) else 0.0}
        return {"n": len(g), "n_valid": len(gv),
                "f0_mean_ort": float(v.mean()), "f0_mean_std": float(v.std()),
                "f0_mean_medyan": float(np.median(v)),
                "f0_mean_p10": float(np.percentile(v, 10)),
                "f0_mean_p90": float(np.percentile(v, 90)),
                "voiced_oran_ort": float(gv.voiced_ratio.mean()),
                "fallback_oran": float(g.fallback.mean())}

    cs = []
    for (src, cls), g in orig.groupby(["source", "label"], sort=True):
        r = q(g); r.update({"source": src, "label": cls}); cs.append(r)
    for cls, g in orig.groupby("label", sort=True):
        r = q(g); r.update({"source": "TUMU", "label": cls}); cs.append(r)
    class_src = pd.DataFrame(cs)[["source", "label"] + list(q(orig).keys())]

    # Augmentation: ayni source_id icin orig -> varyant F0 orani
    aug_rows = []
    piv = valid.pivot_table(index="source_id", columns="aug_id", values="f0_mean", aggfunc="first")
    med = valid.pivot_table(index="source_id", columns="aug_id", values="f0_median", aggfunc="first")
    for aug in ["noise", "pitch_up", "pitch_down", "stretch_0.9", "stretch_1.1", "noise2"]:
        if aug not in piv.columns or "orig" not in piv.columns:
            continue
        sub = piv[["orig", aug]].dropna()
        if sub.empty:
            continue
        ratio = (sub[aug] / sub["orig"]).to_numpy()
        f0o = valid[valid.aug_id.eq("orig")].set_index("source_id").loc[sub.index]
        f0a = valid[valid.aug_id.eq(aug)].set_index("source_id").loc[sub.index]
        aug_rows.append({
            "aug_id": aug, "n_cift": len(sub),
            "orig_f0_mean_ort": float(f0o.f0_mean.mean()),
            "aug_f0_mean_ort": float(f0a.f0_mean.mean()),
            "ortani_arac": float(np.median(ratio)),
            "oran_p10": float(np.percentile(ratio, 10)),
            "oran_p90": float(np.percentile(ratio, 90)),
            "beklenen_oran": {"noise": 1.0, "noise2": 1.0, "pitch_up": 2 ** (1 / 12),
                              "pitch_down": 2 ** (-2 / 12),
                              "stretch_0.9": 1.0, "stretch_1.1": 1.0}[aug],
            "beklenen_fark": "pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; "
                             "noise/stretch F0'yu degistirmemeli",
        })
    aug_f0 = pd.DataFrame(aug_rows)

    fh_rows = []
    for cls in ["fear", "happy"]:
        g = orig_valid[orig_valid.label.eq(cls)]
        v = g.f0_mean.to_numpy(dtype=float)
        fh_rows.append({"label": cls, "n_valid": len(v),
                        "mean": float(v.mean()), "std": float(v.std()),
                        "median": float(np.median(v)),
                        "p10": float(np.percentile(v, 10)),
                        "p25": float(np.percentile(v, 25)),
                        "p75": float(np.percentile(v, 75)),
                        "p90": float(np.percentile(v, 90))})
    fh = pd.DataFrame(fh_rows)
    return class_src, aug_f0, fh


def write_dist_report(detail: pd.DataFrame, class_src: pd.DataFrame,
                      aug_f0: pd.DataFrame, fh: pd.DataFrame) -> None:
    orig = detail[detail.aug_id.eq("orig")]
    lines = [
        "# A1.2 — F0 betimsel analiz (yalnızca hazırlık)",
        "",
        "Yalnızca betimsel istatistik kullanıldı. Model eğitimi, CV, tahmin veya test "
        "sonucu kullanılmadı. F0 istatistikleri geçerli voiced karelerden, "
        "voiced paydası 108 analiz karesinden hesaplandı.",
        "",
        "## 1. Kaynak ve sınıf bazında F0 (yalnızca `orig`, fallback'siz kayıtlar)",
        "",
        md_table(class_src),
        "",
        "## 2. Fear vs happy F0 dağılımı (`orig`, fallback'siz)",
        "",
        md_table(fh),
        "",
    ]
    if len(fh) == 2:
        a, b = fh.iloc[0], fh.iloc[1]
        pooled = np.sqrt((a["std"] ** 2 + b["std"] ** 2) / 2)
        d = (a["mean"] - b["mean"]) / pooled if pooled > 0 else 0.0
        lo, hi = max(a["p10"], b["p10"]), min(a["p90"], b["p90"])
        ov = max(0.0, hi - lo) / max(a["p90"] - a["p10"], b["p90"] - b["p10"], 1e-9)
        lines += [
            f"- Ortalama F0 farkı (fear − happy): **{a['mean'] - b['mean']:+.2f} Hz**; "
            f"havuzlanmış standart sapma cinsinden etki büyüklüğü (deskriptif d): **{d:+.3f}**.",
            f"- p10–p90 aralıklarının kesişimi: **[{lo:.2f}, {hi:.2f}] Hz**; "
            f"kaba örtüşme oranı: **{ov:.2f}** (1.0 = tam örtüşme).",
            "- Bu değerler sınıflandırma performansı veya istatistiksel anlamlılık kanıtı değildir.",
            "",
        ]
    lines += [
        "## 3. Pitch augmentation'ın F0'ya etkisi",
        "",
        md_table(aug_f0),
        "",
        "- Beklenen oranlar: `pitch_up` için 2^(1/12) ≈ 1.0595 (+1 semiton), "
        "`pitch_down` için 2^(-2/12) ≈ 0.8909 (−2 semiton). "
        "Noise ve time-stretch varyantlarında F0 değişmemeli (oran ≈ 1).",
        "- Oranlar aynı `source_id` için eşleştirilmiş `aug_f0_mean / orig_f0_mean` "
        "değerlerinin medyanıdır; fallback uygulanan kayıtlar hariç tutuldu.",
        "",
        "## 4. Voiced frame oranı ve F0 çıkılamama",
        "",
        md_table(orig.groupby("source").agg(
            n_kayit=("source_id", "size"),
            voiced_oran_ort=("voiced_ratio", "mean"),
            voiced_oran_medyan=("voiced_ratio", "median"),
            voiced_sifir_sayisi=("n_voiced_valid", lambda s: int((s == 0).sum())),
            fallback_sayisi=("fallback", "sum"),
        ).reset_index()),
        "",
        md_table(orig.groupby("label").agg(
            n_kayit=("source_id", "size"),
            voiced_oran_ort=("voiced_ratio", "mean"),
            f0_ort=("f0_mean", "mean"),
            f0_std=("f0_mean", "std"),
            fallback_sayisi=("fallback", "sum"),
        ).reset_index()),
        "",
        "Not: BNC yalnızca 100 kayıt/1 kaynak grubu içerir; bu kaynakta F0 güvenilirliği "
        "örneklem büyüklüğü nedeniyle sınırlıdır. TESS 2, SAVEE 4 grup içerir.",
    ]
    DIST_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def main() -> None:
    guard_targets()
    base, review, qc, a0 = prepare_inputs()
    draft = pd.read_csv(DRAFT_FILE, dtype=str)
    plan_rows = []
    for r in base.itertuples(index=False):
        train = r.split_80_20 == "train" or r.split_70_15_15 == "train"
        aug_ids = "orig;" + (";".join(BNC_AUG if r.source == "BNC" else NON_BNC_AUG)
                             if train else "")
        plan_rows.append({"source_id": r.source_id, "group_id": r.group_id,
                          "source": r.source, "label": r.label,
                          "split_80_20": r.split_80_20,
                          "split_70_15_15": r.split_70_15_15, "fold": r.fold,
                          "sha256": r.sha256, "is_train": bool(train),
                          "aug_ids": aug_ids,
                          "expected_rows": 7 if (r.source == "BNC" and train) else (3 if train else 1)})
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(PLAN_FILE, index=False, encoding="utf-8")
    expected = int(plan.expected_rows.sum())
    print(f"Canonical: {len(base)} | REVIEW_LABEL: {len(review)} | REVIEW_QC: {len(qc)}")
    print(f"A0 satir: {len(a0)} | beklenen: {expected} | gorev: {len(base)}", flush=True)

    tasks = make_tasks(a0, base)
    t0 = time.perf_counter()
    if os.environ.get("A1_2_USE_CACHE") == "1" and CACHE_FILE.is_file():
        with open(CACHE_FILE, "rb") as fh:
            results = pickle.load(fh)
        print(f"pYIN sonuclari cache'den okundu: {len(results)} kayit", flush=True)
    else:
        results = {}
        with Pool(processes=N_WORKERS) as pool:
            for i, res in enumerate(pool.imap(worker, tasks, chunksize=4), start=1):
                results[res["source_id"]] = res
                if i % 500 == 0:
                    print(f"pYIN gorev: {i}/{len(tasks)} | {time.perf_counter()-t0:.0f}s", flush=True)
        with open(CACHE_FILE, "wb") as fh:
            pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"pYIN tamamlandi: {time.perf_counter()-t0:.0f}s", flush=True)

    # --- satir bazli F0 tablosu (A0 satir sirasini korur) ---
    detail_rows: list[dict] = []
    f0_matrix = np.empty((len(a0), 8), dtype=float)
    load_errors, sha_mismatch, aug_errors = [], [], []
    for pos, row in enumerate(a0.itertuples(index=False)):
        res = results[row.source_id]
        if res["load_error"]:
            load_errors.append({"source_id": row.source_id, "error": res["load_error"]})
        if not res["sha_ok"]:
            sha_mismatch.append(row.source_id)
        r = res["per_aug"].get(row.aug_id)
        if r is None:
            aug_errors.append(row.source_id)
            r = {k: (0.0 if k in F0_NAMES else 0) for k in F0_NAMES} | {
                "n_frames": 0, "n_voiced_raw": 0, "n_f0_finite": 0,
                "n_voiced_valid": 0, "n_voiced_invalid": 0, "voiced_ratio": 0.0,
                "fallback": 1, "fallback_reason": "missing_result",
                "pyin_error": "missing_result"}
        f0_values = [float(r[n]) for n in F0_NAMES]
        f0_matrix[pos] = f0_values
        d = {"source_id": row.source_id, "group_id": row.group_id,
             "source": row.source, "label": row.label, "aug_id": row.aug_id,
             "fold": row.fold, "split_80_20": row.split_80_20,
             "split_70_15_15": row.split_70_15_15}
        d.update({k: r[k] for k in F0_DETAIL_COLS if k not in d})
        detail_rows.append(d)

    if load_errors or sha_mismatch or aug_errors:
        raise AssertionError(
            f"Eksik WAV={len(load_errors)} | SHA uyusmazligi={len(sha_mismatch)} | "
            f"eksik varyant={len(aug_errors)}; cikti uretilmedi.")
    detail = pd.DataFrame(detail_rows)[F0_DETAIL_COLS]

    out = a0.copy()
    for i, col in enumerate(F0_COLS):
        out[col] = f0_matrix[:, i]
    out = out[META_COLS + FEATURE_COLS]

    # --- dogrulamalar ---
    leak_rows = leakage_checks(out)
    cmp_rows = compare_a0(a0, out)
    if len(out) != 23268 or out.shape[1] != 202:
        raise AssertionError(f"Beklenen 23268x202, bulunan {out.shape}")
    if out.groupby(["source_id", "aug_id"]).size().max() != 1:
        raise AssertionError("Tekrarlayan source_id/aug_id.")
    if out.groupby("source_id").fold.nunique().max() != 1:
        raise AssertionError("Bir source_id birden fazla fold'da.")
    if out.groupby("sha256").source_id.nunique().max() != 1:
        raise AssertionError("Bir SHA256 birden fazla source_id'ye bagli.")

    # --- ciktilar ---
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    detail.to_csv(F0_DETAIL_FILE, index=False, encoding="utf-8")
    pd.DataFrame(leak_rows).to_csv(LEAK_FILE, index=False, encoding="utf-8")
    pd.DataFrame(cmp_rows).to_csv(CMP_FILE, index=False, encoding="utf-8")
    pd.DataFrame({"kolon": F0_COLS, "ozellik": F0_NAMES, "birim": F0_UNITS,
                  "tanim": [
                      "Gecerli voiced karelerin F0 ortalamasi",
                      "Gecerli voiced karelerin F0 medyani",
                      "Gecerli voiced karelerin F0 std (ddof=0)",
                      "Gecerli voiced karelerin minimum F0",
                      "Gecerli voiced karelerin maksimum F0",
                      "maksimum - minimum F0",
                      "np.polyfit(zaman[voiced], F0[voiced], 1) egitimi; zaman=i*512/22050 s",
                      "n_voiced_valid / 108 (2,5 s pencerenin toplam pYIN kare sayisi)",
                  ],
                  "fallback": "Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; "
                              "voiced_ratio gercek paydadan uretilir"}).to_csv(
        DEF_FILE, index=False, encoding="utf-8")

    sha_rows = [
        {"kontrol": "output_rows", "deger": len(out)},
        {"kontrol": "unique_source_id", "deger": int(out.source_id.nunique())},
        {"kontrol": "unique_sha256", "deger": int(out.sha256.nunique())},
        {"kontrol": "source_aug_pairs", "deger": len(out.groupby(["source_id", "aug_id"]))},
        {"kontrol": "duplicate_source_aug", "deger": int(out.duplicated(["source_id", "aug_id"]).sum())},
        {"kontrol": "sha256_to_multi_source", "deger": int((out.groupby("sha256").source_id.nunique() > 1).sum())},
        {"kontrol": "finite_all_192_features", "deger": bool(np.isfinite(out[FEATURE_COLS].to_numpy(dtype=float)).all())},
        {"kontrol": "missing_metadata", "deger": int(out[META_COLS].isna().sum().sum())},
    ]
    pd.DataFrame(sha_rows).to_csv(SHA_FILE, index=False, encoding="utf-8")

    quality = []
    for label, value in [
        ("pyin_cagrisi", len(detail)), ("wav_okuma_hatasi", len(load_errors)),
        ("sha256_uyusmazligi", len(sha_mismatch)), ("eksik_varyant", len(aug_errors)),
        ("pyin_hatali_kayit", int((detail.pyin_error.ne("")).sum())),
        ("fallback_uygulanan_kayit", int(detail.fallback.sum())),
        ("voiced_kare_sifir_kayit", int((detail.n_voiced_valid.eq(0)).sum())),
        ("voiced_flag_var_f0_gecersiz_kare", int(detail.n_voiced_invalid.sum())),
        ("f0_sonlu_olmayan_kare", int((detail.n_frames - detail.n_f0_finite).sum())),
        ("beklenen_kare_sayisi", 108),
        ("kare_sayisi_108_olmayan_kayit", int(detail.n_frames.ne(108).sum())),
    ]:
        quality.append({"kontrol": label, "deger": value})
    for reason, n in detail[detail.fallback.eq(1)].fallback_reason.value_counts().items():
        quality.append({"kontrol": f"fallback_sebebi:{reason}", "deger": int(n)})
    for src, g in detail.groupby("source", sort=True):
        quality.append({"kontrol": f"kaynak:{src}:fallback", "deger": int(g.fallback.sum())})
    for aug, g in detail.groupby("aug_id", sort=True):
        quality.append({"kontrol": f"aug:{aug}:fallback", "deger": int(g.fallback.sum())})
    for r in detail[detail.pyin_error.ne("")].head(50).itertuples(index=False):
        quality.append({"kontrol": f"hata:{r.source_id}:{r.aug_id}", "deger": r.pyin_error})
    pd.DataFrame(quality).to_csv(QUALITY_FILE, index=False, encoding="utf-8")

    aug_dist = out.groupby(["source", "aug_id"]).size().reset_index(name="n_rows")
    aug_dist.to_csv(AUG_DIST_FILE, index=False, encoding="utf-8")
    base.groupby("source").size().reset_index(name="canonical_kayit").to_csv(SOURCE_FILE, index=False, encoding="utf-8")
    base.groupby("label").size().reset_index(name="canonical_kayit").to_csv(CLASS_FILE, index=False, encoding="utf-8")
    split_out = []
    for eval_name, col in EVAL_SPLITS:
        for role, n in out[col].value_counts().items():
            split_out.append({"eval": eval_name, "split": role, "feature_rows": int(n)})
    pd.DataFrame(split_out).to_csv(SPLIT_FILE, index=False, encoding="utf-8")
    out.groupby("fold").size().reset_index(name="feature_rows").to_csv(FOLD_FILE, index=False, encoding="utf-8")

    class_src, aug_f0, fh = descriptive(detail)
    class_src.to_csv(CLASS_SRC_FILE, index=False, encoding="utf-8")
    aug_f0.to_csv(AUG_F0_FILE, index=False, encoding="utf-8")
    write_dist_report(detail, class_src, aug_f0, fh)

    # --- raporlar ---
    src_rows = out.groupby("source").size().reset_index(name="feature_rows")
    cls_rows = out.groupby("label").size().reset_index(name="feature_rows")
    fold_rows = out.groupby("fold").size().reset_index(name="feature_rows")
    DATA_REPORT.write_text("\n".join([
        "# Veri Hazırlama Raporu — A1.2 (A0 girdisi, F0 hazırlığı)",
        "",
        "Yalnızca veri hazırlama ve F0 çıkarımı yapıldı. Eğitim, CV, 70/15/15, "
        "tahmin, model/scaler kaydı veya GUI değişikliği yapılmadı.",
        "",
        "## Girdiler",
        "",
        f"- `{DRAFT_FILE.name}`: {len(draft)} satır (draft)",
        f"- `{QC_FILE.name}`: {len(qc)} satır (REVIEW_QC)",
        f"- `{A0_FILE.name}`: {len(a0)} satır × 194 kolon (ilk 184 özellik kaynağı)",
        "",
        "## Dışlananlar",
        "",
        f"- REVIEW_LABEL: {len(review)} kayıt / {review.sha256.nunique()} SHA256 — etiket değiştirilmedi.",
        f"- REVIEW_QC: {len(qc)} kayıt — canonical kümeyle kesişim "
        f"{len(set(base.source_id) & set(qc.source_id))}.",
        "",
        "## Canonical küme",
        "",
        f"- Kayıt: **{len(base)}**, tekil source_id: **{base.source_id.nunique()}**, "
        f"tekil SHA256: **{base.sha256.nunique()}**, group_id: **{base.group_id.nunique()}**",
        "",
        md_table(base.groupby("source").size().reset_index(name="canonical_kayit")),
        "",
        md_table(base.groupby("label").size().reset_index(name="canonical_kayit")),
        "",
        "## Augmentation planı (A0 ile aynı)",
        "",
        "Bir kayıt iki değerlendirmenin en az birinde train ise augmentation alır; "
        "BNC 6, diğer kaynaklar 2 varyant. Train olmayan kayıtlar yalnızca `orig`.",
        "",
        f"- Beklenen feature satırı: **{expected}**, üretilen: **{len(out)}**",
        "",
        "## Ön leakage",
        "",
        f"- Tekrarlayan SHA256: **{int(base.sha256.duplicated().sum())}**",
        f"- Çoklu fold/source: **{int((base.groupby('source_id').fold.nunique() > 1).sum())}**",
        "",
    ]) + "\n", encoding="utf-8")

    FEAT_REPORT.write_text("\n".join([
        "# Özellik Çıkarım Raporu — A1.2",
        "",
        "İlk 184 özellik A0 CSV'sinden birebir kopyalandı; 8 yeni F0 özelliği "
        "her varyantın gerçek sinyalinden A0 zinciriyle çıkarıldı. "
        "Scaler ve model değiştirilmedi.",
        "",
        "## Çıktı",
        "",
        f"- `{OUTPUT_FILE.name}`: **{len(out)}** satır × **{out.shape[1]}** kolon "
        "(10 metadata + 184 A0 + 8 F0)",
        f"- Boyut: **{OUTPUT_FILE.stat().st_size}** byte",
        "",
        "## 8 F0 özelliği",
        "",
        md_table(pd.read_csv(DEF_FILE)),
        "",
        "## pYIN / fallback",
        "",
        md_table(pd.DataFrame(quality)),
        "",
        "## A0 eşleşmesi",
        "",
        md_table(pd.DataFrame(cmp_rows)),
        "",
        "## Leakage",
        "",
        md_table(pd.DataFrame(leak_rows)),
        "",
        "## Satır/kaynak/sınıf/fold",
        "",
        md_table(src_rows), "", md_table(cls_rows), "", md_table(fold_rows), "",
        md_table(aug_dist), "",
        "Not: Fold OOF test satırları `fold == k & aug_id == 'orig'` ile seçilir.",
        "",
        "Tüm doğrulamalar PASS; eğitim/CV/tahmin yapılmadı.",
    ]) + "\n", encoding="utf-8")

    print(f"\nCikti: {OUTPUT_FILE.name} | {out.shape[0]}x{out.shape[1]}")
    print(f"F0: pyin={len(detail)} | fallback={int(detail.fallback.sum())} | "
          f"pyin_hata={int(detail.pyin_error.ne('').sum())}")
    print(f"Leakage: {len(leak_rows)}/{len(leak_rows)} PASS")
    print("Tum dogrulamalar PASS; egittim/CV YOK.")


if __name__ == "__main__":
    main()
