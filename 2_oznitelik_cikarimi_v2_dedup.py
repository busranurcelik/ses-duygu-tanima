"""
v2 dedup / adim 2: yalnizca dedup etkisini olcmek icin 184 ozellik cikarimi.

Kurallar:
  * Girdiler: manifest_v2_dedup_draft.csv + manifest_v2_dedup_review_qc.csv
  * REVIEW_LABEL kayitlari tamamen dislanir (egitim ve degerlendirme).
  * REVIEW_QC kayitlari tamamen dislanir (taslakta zaten yok; ayrica dogrulanir).
  * Ayni SHA256 icerik icin tek canonical source_id kullanilir.
  * Augmentation yalnizca ilgili degerlendirmede train olan kayitlarda uretilir.
    val/test satirlari yalnizca 'orig' kalir; diger degerlendirmede train
    degilse ilgili split kolonu 'EXCLUDED' yapilir.
  * 184 ozellik ve augmentation fonksiyonlari mevcut v2 pipeline ile birebir aynidir.
  * VAD / log-mel / yeni ozellik YOK.
  * Eski final_oznitelik_seti_v2.csv ve eski modeller degistirilmez.

Ciktilar:
  final_oznitelik_seti_v2_dedup.csv
  sonuclar_v2_dedup/veri_hazirlama_raporu.md
  sonuclar_v2_dedup/ozellik_cikarimi_raporu.md
  sonuclar_v2_dedup/augmentation_plani.csv
  sonuclar_v2_dedup/augmentation_dagilim.csv
  sonuclar_v2_dedup/kaynak_sayilari.csv
  sonuclar_v2_dedup/sinif_sayilari.csv
  sonuclar_v2_dedup/split_sayilari.csv
  sonuclar_v2_dedup/fold_sayilari.csv
  sonuclar_v2_dedup/sha256_source_id_kontrolleri.csv
  sonuclar_v2_dedup/leakage_kontrolleri.csv
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DRAFT_FILE = ROOT / "manifest_v2_dedup_draft.csv"
QC_FILE = ROOT / "manifest_v2_dedup_review_qc.csv"
OUTPUT_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"
OUT_DIR = ROOT / "sonuclar_v2_dedup"

SAMPLE_RATE = 22050
DURATION = 2.5

NON_BNC_AUG = ["noise", "pitch_up"]
BNC_AUG = ["noise", "pitch_up", "pitch_down", "stretch_0.9", "stretch_1.1", "noise2"]
EVAL_SPLITS = [("80_20", "split_80_20"), ("70_15_15", "split_70_15_15")]

FEATURE_COLS = [str(i) for i in range(184)]
META_COLS = [
    "source_id", "group_id", "source", "label", "aug_id",
    "split_80_20", "split_70_15_15", "fold", "labels", "sha256",
]

DATA_REPORT = OUT_DIR / "veri_hazirlama_raporu.md"
FEAT_REPORT = OUT_DIR / "ozellik_cikarimi_raporu.md"
PLAN_FILE = OUT_DIR / "augmentation_plani.csv"
AUG_DIST_FILE = OUT_DIR / "augmentation_dagilim.csv"
SOURCE_FILE = OUT_DIR / "kaynak_sayilari.csv"
CLASS_FILE = OUT_DIR / "sinif_sayilari.csv"
SPLIT_FILE = OUT_DIR / "split_sayilari.csv"
FOLD_FILE = OUT_DIR / "fold_sayilari.csv"
SHA_FILE = OUT_DIR / "sha256_source_id_kontrolleri.csv"
LEAK_FILE = OUT_DIR / "leakage_kontrolleri.csv"

ALL_TARGETS = [
    OUTPUT_FILE, DATA_REPORT, FEAT_REPORT, PLAN_FILE, AUG_DIST_FILE,
    SOURCE_FILE, CLASS_FILE, SPLIT_FILE, FOLD_FILE, SHA_FILE, LEAK_FILE,
]


# ---------------------------------------------------------------------------
# Yardimci
# ---------------------------------------------------------------------------
def guard_targets() -> None:
    for p in ALL_TARGETS:
        if p.exists():
            raise SystemExit(f"REFUSE OVERWRITE: {p}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 184 ozellik — mevcut v2 pipeline ile birebir ayni
# ---------------------------------------------------------------------------
def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    y = librosa.util.normalize(y)
    return librosa.util.fix_length(y, size=int(DURATION * sr))


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = preprocess_audio(y, sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return np.hstack([mfcc, chroma, mel,
                      np.mean(rms), np.std(rms),
                      np.mean(zcr), np.std(zcr)])


def rng_for(source_id: str, aug_id: str) -> np.random.Generator:
    key = f"{source_id}|{aug_id}|v2".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(seed)


def add_noise(y: np.ndarray, rng: np.random.Generator, level: float = 0.015) -> np.ndarray:
    amp = level * rng.uniform() * np.max(np.abs(y))
    return y + amp * rng.normal(size=len(y))


def augment(y: np.ndarray, sr: int, aug_id: str, source_id: str) -> np.ndarray:
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
# Veri hazirlama
# ---------------------------------------------------------------------------
def prepare_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [DRAFT_FILE, QC_FILE]:
        if not p.is_file():
            raise SystemExit(f"Girdi dosyasi yok: {p}")

    draft = pd.read_csv(DRAFT_FILE, dtype=str)
    qc = pd.read_csv(QC_FILE, dtype=str)

    required = {"source_id", "group_id", "source", "label", "split_80_20",
                "split_70_15_15", "fold", "path", "sha256", "dedup_status"}
    missing = required - set(draft.columns)
    if missing:
        raise AssertionError(f"Draft manifest eksik kolon: {sorted(missing)}")

    review_label = draft[draft.dedup_status.eq("REVIEW_LABEL")].copy()
    base = draft[~draft.dedup_status.eq("REVIEW_LABEL")].copy()

    if len(review_label) != 2:
        raise AssertionError(f"REVIEW_LABEL beklenen 2, bulunan {len(review_label)}")
    if review_label.sha256.nunique() != 1:
        raise AssertionError("REVIEW_LABEL kayitlari tek hash grubunda degil.")
    if len(set(base.source_id) & set(qc.source_id)):
        raise AssertionError("REVIEW_QC kayidlari taslakta bulunuyor.")
    if (base.dedup_status.ne("KEEP_UNIQUE") & base.dedup_status.ne("KEEP_CANONICAL")).any():
        raise AssertionError("Beklenmeyen dedup_status degeri var.")
    if not base.source_id.is_unique:
        raise AssertionError("Taslakta source_id tekrar ediyor.")
    if not base.sha256.is_unique:
        raise AssertionError("REVIEW_LABEL disinda SHA256 tekrar ediyor.")

    base["fold"] = base.fold.astype(int)
    for col in ["split_80_20", "split_70_15_15"]:
        allowed = {"train", "test", "val"}
        bad = set(base[col]) - allowed
        if bad:
            raise AssertionError(f"{col} beklenmeyen deger: {bad}")

    return base, review_label, qc


def build_plan(base: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    plan = base[["source_id", "group_id", "source", "label", "split_80_20",
                 "split_70_15_15", "fold", "sha256"]].copy()
    is_train = (plan.split_80_20.eq("train") | plan.split_70_15_15.eq("train")).to_numpy()
    plan["is_train"] = is_train
    plan["aug_ids"] = [
        "orig;" + (";".join(BNC_AUG if s == "BNC" else NON_BNC_AUG) if t else "")
        for s, t in zip(plan.source, is_train)
    ]
    plan["expected_rows"] = [
        7 if (s == "BNC" and t) else (3 if t else 1)
        for s, t in zip(plan.source, is_train)
    ]
    expected = int(plan.expected_rows.sum())
    return plan, expected


def write_data_report(base: pd.DataFrame, review_label: pd.DataFrame,
                      qc: pd.DataFrame, plan: pd.DataFrame, expected: int) -> None:
    source_counts = base.groupby("source").size().reset_index(name="canonical_kayit")
    class_counts = base.groupby("label").size().reset_index(name="canonical_kayit")
    split_rows = []
    for eval_name, col in EVAL_SPLITS:
        for role, n in base[col].value_counts().items():
            split_rows.append({"eval": eval_name, "split": role, "canonical_kayit": int(n)})
    split_counts = pd.DataFrame(split_rows)
    fold_counts = base.groupby("fold").size().reset_index(name="canonical_kayit")
    combo_counts = base.groupby(["split_80_20", "split_70_15_15", "fold"]).size().reset_index(name="canonical_kayit")

    plan.to_csv(PLAN_FILE, index=False, encoding="utf-8")

    lines = [
        "# Veri Hazırlama Raporu — dedup pipeline",
        "",
        "Bu rapor yalnızca yeni özellik çıkarım pipeline'ının girdi hazırlığını açıklar. "
        "Model eğitimi, CV ve istatistik yapılmadı.",
        "",
        "## Girdiler",
        "",
        f"- `{DRAFT_FILE.name}`: {7782} satır (dedup taslağı)",
        f"- `{QC_FILE.name}`: {len(qc)} satır (`REVIEW_QC`)",
        "",
        "## Dışlanan kayıtlar",
        "",
        f"- `REVIEW_LABEL`: {len(review_label)} kayıt, {review_label.sha256.nunique()} SHA256 grubu. "
        "İkisi de eğitim ve değerlendirmeden tamamen çıkarıldı; etiket değiştirilmedi.",
        f"- `REVIEW_QC`: {len(qc)} kayıt. Taslakla kesişim: "
        f"{len(set(base.source_id) & set(qc.source_id))}.",
        "",
        md_table(review_label[["source_id", "source", "label", "split_80_20",
                               "split_70_15_15", "fold", "dedup_status"]]),
        "",
        "## Kalan canonical kayıtlar",
        "",
        f"- Toplam: **{len(base)}**",
        f"- Tekil `source_id`: **{base.source_id.nunique()}**",
        f"- Tekil SHA256: **{base.sha256.nunique()}**",
        f"- `group_id`: **{base.group_id.nunique()}**",
        f"- Fold: **{base.fold.nunique()}**",
        "",
        "### Kaynak bazında",
        "",
        md_table(source_counts),
        "",
        "### Sınıf bazında",
        "",
        md_table(class_counts),
        "",
        "### Split bazında",
        "",
        md_table(split_counts),
        "",
        "### Fold bazında",
        "",
        md_table(fold_counts),
        "",
        "### Split × split × fold kombinasyonları",
        "",
        md_table(combo_counts),
        "",
        "## Augmentation planı",
        "",
        "Mevcut kural korundu: bir kayıt iki değerlendirmenin en az birinde `train` ise "
        "augmentation alır. BNC 6 varyant, diğer kaynaklar 2 varyant üretir. "
        "Train olmayan kayıtlar yalnızca `orig` üretir.",
        "",
        f"- 80/20 train: **{int(base.split_80_20.eq('train').sum())}**",
        f"- 70/15/15 train: **{int(base.split_70_15_15.eq('train').sum())}**",
        f"- Birleşik train (augmentation alan): **{int(plan.is_train.sum())}**",
        f"  - BNC: **{int((plan.is_train & plan.source.eq('BNC')).sum())}**",
        f"  - non-BNC: **{int((plan.is_train & plan.source.ne('BNC')).sum())}**",
        f"- Train olmayan: **{int((~plan.is_train).sum())}**",
        f"- Beklenen özellik satırı: **{expected}**",
        "",
        "Satır bazlı plan `sonuclar_v2_dedup/augmentation_plani.csv` dosyasında.",
        "",
        "## Ön leakage kontrolü",
        "",
        f"- Tekrar eden SHA256: **{int(base.sha256.duplicated().sum())}**",
        f"- Tekrar eden source_id: **{int(base.source_id.duplicated().sum())}**",
        f"- Birden fazla split'e giden source_id: "
        f"**{int((base.groupby('source_id')[['split_80_20','split_70_15_15']].nunique() > 1).any(axis=1).sum())}**",
        f"- Birden fazla fold'a giden source_id: **{int((base.groupby('source_id').fold.nunique() > 1).sum())}**",
        "",
        "Not: 23.274 sayısı varsayılmadı. REVIEW_LABEL çıkarımı sonrası gerçek beklenti "
        f"**{expected}** olarak hesaplandı.",
    ]
    DATA_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Doğrulama
# ---------------------------------------------------------------------------
def validate_output(out: pd.DataFrame, base: pd.DataFrame,
                    review_label: pd.DataFrame, qc: pd.DataFrame,
                    expected: int) -> tuple[list[dict], list[dict], list[dict]]:
    cols = [c for c in out.columns if c.isdigit()]
    if sorted(cols, key=int) != FEATURE_COLS:
        raise AssertionError(f"184 özellik kolonu beklenirken {len(cols)} bulundu.")
    if len(out) != expected:
        raise AssertionError(f"Satır beklentisi {expected}, bulunan {len(out)}.")
    if not np.isfinite(out[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise AssertionError("Özelliklerde NaN/Inf var.")
    if int(out[META_COLS].isna().sum().sum()) != 0:
        raise AssertionError("Metadata kolonlarında eksik değer var.")
    if set(out.source_id) != set(base.source_id):
        raise AssertionError("Feature source_id kümesi manifest ile aynı değil.")
    if len(set(out.source_id) & set(review_label.source_id)):
        raise AssertionError("REVIEW_LABEL feature CSV'ye girdi.")
    if len(set(out.source_id) & set(qc.source_id)):
        raise AssertionError("REVIEW_QC feature CSV'ye girdi.")
    if out.groupby(["source_id", "aug_id"]).size().max() != 1:
        raise AssertionError("Aynı source_id/aug_id için birden fazla satır var.")
    if out.sha256.nunique() != base.sha256.nunique():
        raise AssertionError("SHA256 tekliği bozuldu.")
    if out.groupby("sha256").source_id.nunique().max() != 1:
        raise AssertionError("Bir SHA256 birden fazla source_id'ye bağlı.")
    if out.groupby("source_id").fold.nunique().max() != 1:
        raise AssertionError("Bir source_id birden fazla fold taşıyor.")
    fold_map = dict(zip(base.source_id, base.fold))
    if not (out.source_id.map(fold_map) == out.fold).all():
        raise AssertionError("Fold bilgisi manifestten korunmadı.")

    leak_rows: list[dict] = []
    for eval_name, col in EVAL_SPLITS:
        train = out[out[col].eq("train")]
        ev = out[out[col].isin(["val", "test"])]
        excluded = out[out[col].eq("EXCLUDED")]
        ev_aug = int((ev.aug_id.ne("orig")).sum())
        ex_orig = int((excluded.aug_id.eq("orig")).sum())
        grp_leak = len(set(train.group_id) & set(ev.group_id))
        src_leak = len(set(train.source_id) & set(ev.source_id))
        roles = out[out[col].ne("EXCLUDED")].groupby("source_id")[col].nunique()
        role_bad = int((roles > 1).sum())
        if ev_aug or ex_orig or grp_leak or src_leak or role_bad:
            raise AssertionError(f"{eval_name}: leakage/audit kontrolü başarısız.")
        leak_rows.append({
            "kapsam": eval_name, "train_rows": len(train), "eval_rows": len(ev),
            "excluded_rows": len(excluded), "eval_aug_rows": ev_aug,
            "excluded_orig_rows": ex_orig, "group_leak": grp_leak,
            "source_leak": src_leak, "multi_role_source": role_bad, "sonuc": "PASS",
        })

    for k in sorted(base.fold.unique()):
        fold_all = out[out.fold.eq(k)]
        test = fold_all[fold_all.aug_id.eq("orig")]
        train = out[out.fold.ne(k)]
        grp_leak = len(set(train.group_id) & set(test.group_id))
        src_leak = len(set(train.source_id) & set(test.source_id))
        if len(test) == 0 or len(train) == 0 or grp_leak or src_leak:
            raise AssertionError(f"fold {k}: OOF leakage kontrolü başarısız.")
        leak_rows.append({
            "kapsam": f"fold_{k}", "train_rows": len(train), "eval_rows": len(test),
            "excluded_rows": int((fold_all.aug_id.ne("orig")).sum()),
            "eval_aug_rows": 0, "excluded_orig_rows": 0,
            "group_leak": grp_leak, "source_leak": src_leak,
            "multi_role_source": 0, "sonuc": "PASS",
        })

    sha_rows = [
        {"kontrol": "output_rows", "deger": len(out)},
        {"kontrol": "unique_source_id", "deger": int(out.source_id.nunique())},
        {"kontrol": "unique_sha256", "deger": int(out.sha256.nunique())},
        {"kontrol": "source_aug_pairs", "deger": len(out.groupby(["source_id", "aug_id"]))},
        {"kontrol": "duplicate_source_aug", "deger": int(out.duplicated(["source_id", "aug_id"]).sum())},
        {"kontrol": "sha256_to_multi_source", "deger": int((out.groupby("sha256").source_id.nunique() > 1).sum())},
        {"kontrol": "finite_features", "deger": bool(np.isfinite(out[FEATURE_COLS].to_numpy(dtype=float)).all())},
        {"kontrol": "missing_metadata", "deger": int(out[META_COLS].isna().sum().sum())},
    ]
    return leak_rows, sha_rows, cols


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def main() -> None:
    guard_targets()
    base, review_label, qc = prepare_base()
    plan, expected = build_plan(base)
    write_data_report(base, review_label, qc, plan, expected)

    print(f"Kalan canonical kayit: {len(base)}")
    print(f"REVIEW_LABEL dislanan: {len(review_label)} | REVIEW_QC dislanan: {len(qc)}")
    print(f"Aug alan: {int(plan.is_train.sum())} | beklenen satir: {expected}", flush=True)

    rows: list[dict] = []
    missing_files: list[str] = []
    read_errors: list[tuple[str, str]] = []
    hash_mismatch: list[str] = []

    for idx, rec in enumerate(base.itertuples(index=False), start=1):
        path = Path(rec.path)
        if not path.is_file():
            missing_files.append(rec.source_id)
            continue
        try:
            file_hash = sha256_file(path)
            if file_hash != rec.sha256:
                hash_mismatch.append(rec.source_id)
            y, sr = librosa.load(path, sr=SAMPLE_RATE)
        except Exception as exc:
            read_errors.append((rec.source_id, str(exc)))
            if len(read_errors) <= 5:
                print(f"HATA ({rec.source_id}): {exc}", flush=True)
            continue

        is_train = rec.split_80_20 == "train" or rec.split_70_15_15 == "train"
        aug_ids = ["orig"]
        if is_train:
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
                "fold": rec.fold,
                "labels": rec.label,
                "sha256": rec.sha256,
            }
            if aug_id != "orig":
                for _, col in EVAL_SPLITS:
                    if row[col] != "train":
                        row[col] = "EXCLUDED"
            row.update({str(i): float(v) for i, v in enumerate(feats)})
            rows.append(row)

        if idx % 1000 == 0:
            print(f"Islenen: {idx}/{len(base)} | uretilen satir: {len(rows)}", flush=True)

    if missing_files:
        raise RuntimeError(f"{len(missing_files)} eksik dosya; cikti uretilmedi.")
    if read_errors:
        raise RuntimeError(f"{len(read_errors)} WAV okunamadi; cikti uretilmedi.")
    if hash_mismatch:
        raise RuntimeError(f"{len(hash_mismatch)} SHA256 uyusmazligi; cikti uretilmedi.")

    out = pd.DataFrame(rows)[META_COLS + FEATURE_COLS]
    leak_rows, sha_rows, cols = validate_output(out, base, review_label, qc, expected)

    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    pd.DataFrame(leak_rows).to_csv(LEAK_FILE, index=False, encoding="utf-8")
    pd.DataFrame(sha_rows).to_csv(SHA_FILE, index=False, encoding="utf-8")

    aug_dist = out.groupby(["source", "aug_id"]).size().reset_index(name="n_rows")
    aug_dist.to_csv(AUG_DIST_FILE, index=False, encoding="utf-8")

    base.groupby("source").size().reset_index(name="canonical_kayit").to_csv(SOURCE_FILE, index=False, encoding="utf-8")
    base.groupby("label").size().reset_index(name="canonical_kayit").to_csv(CLASS_FILE, index=False, encoding="utf-8")
    out.groupby("fold").size().reset_index(name="feature_rows").to_csv(FOLD_FILE, index=False, encoding="utf-8")
    split_out = []
    for eval_name, col in EVAL_SPLITS:
        for role, n in out[col].value_counts().items():
            split_out.append({"eval": eval_name, "split": role, "feature_rows": int(n)})
    pd.DataFrame(split_out).to_csv(SPLIT_FILE, index=False, encoding="utf-8")

    src_rows = out.groupby("source").size().reset_index(name="feature_rows")
    cls_rows = out.groupby("label").size().reset_index(name="feature_rows")
    fold_rows = out.groupby("fold").size().reset_index(name="feature_rows")

    lines = [
        "# Özellik Çıkarım Raporu — dedup",
        "",
        "Mevcut 184 özellikli v2 pipeline birebir kullanıldı. VAD, log-mel veya yeni özellik eklenmedi.",
        "",
        "## Çıktı",
        "",
        f"- `{OUTPUT_FILE.name}`: **{len(out)}** satır, **{len(out.columns)}** kolon "
        f"({len(META_COLS)} metadata + {len(cols)} özellik)",
        f"- Dosya boyutu: **{OUTPUT_FILE.stat().st_size}** byte",
        "",
        "## Satır sayısı",
        "",
        f"- Beklenen: **{expected}**",
        f"- Üretilen: **{len(out)}**",
        f"- Eşitlik: **{len(out) == expected}**",
        "",
        "23.274 sayısı varsayılmadı; REVIEW_LABEL çıkarımı sonrası gerçek beklenti hesaplandı.",
        "",
        "## Kaynak bazında feature satırı",
        "",
        md_table(src_rows),
        "",
        "## Sınıf bazında feature satırı",
        "",
        md_table(cls_rows),
        "",
        "## Split bazında feature satırı",
        "",
        md_table(pd.DataFrame(split_out)),
        "",
        "## Fold bazında feature satırı",
        "",
        md_table(fold_rows),
        "",
        "## Augmentation dağılımı",
        "",
        md_table(aug_dist),
        "",
        "## SHA256 / source_id kontrolleri",
        "",
        md_table(pd.DataFrame(sha_rows)),
        "",
        "## NaN / Inf / eksik dosya",
        "",
        f"- Eksik WAV: **{len(missing_files)}**",
        f"- Okunamayan WAV: **{len(read_errors)}**",
        f"- SHA256 uyuşmazlığı: **{len(hash_mismatch)}**",
        f"- NaN/Inf özellik: **{bool(np.isfinite(out[FEATURE_COLS].to_numpy(dtype=float)).all()) is False}**",
        f"- Eksik metadata hücresi: **{int(out[META_COLS].isna().sum().sum())}**",
        "",
        "## Leakage kontrolü",
        "",
        md_table(pd.DataFrame(leak_rows)),
        "",
        "Not: Fold bazlı OOF test satırları `fold == k & aug_id == 'orig'` ile seçilir; "
        "aynı fold'a ait augmentation satırları CV train tarafında kullanılmaz.",
        "",
        "## Sonuç",
        "",
        "Tüm doğrulamalar **PASS**. Model eğitimi, CV, istatistik ve GUI değişikliği yapılmadı.",
    ]
    FEAT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== tamamlandı ===")
    print(f"Çıktı: {OUTPUT_FILE.name} | satır: {len(out)} | kolon: {out.shape[1]}")
    print(f"Raporlar: {DATA_REPORT.name}, {FEAT_REPORT.name}")
    print(f"Leakage kontrolü: {len(leak_rows)}/{len(leak_rows)} PASS")
    print("Tüm doğrulamalar PASS")


if __name__ == "__main__":
    main()
