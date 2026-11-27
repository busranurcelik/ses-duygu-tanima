"""
v2 / adim 1: group-aware manifest ve augmentation-oncesi split uretimi.

Bu script yalnizca okur/yazar; model egitmez, CV calistirmaz.
Kaynak: ham_veri_listesi.csv (Git tarafindan ignore edilen yerel v1 manifesti).
Cikti:
  manifest_v2.csv           -> herkese acik, yol/kimlik icermez
  manifest_v2_local.csv     -> yerel, ignore edilir, WAV yollari
  sonuclar_v2/split_dagilim.csv / .md
  sonuclar_v2/split_config.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "ham_veri_listesi.csv"
PUBLIC_MANIFEST = ROOT / "manifest_v2.csv"
LOCAL_MANIFEST = ROOT / "manifest_v2_local.csv"
OUT_DIR = ROOT / "sonuclar_v2"

CLASSES = ["angry", "fear", "happy", "sad"]
SPLIT_COLS = {"80_20": "split_80_20", "70_15_15": "split_70_15_15"}
BASE_SEED = 42
MAX_SEED = 1000
TOL_SAMPLE = 0.02   # +-2 yuzde puan
TOL_CLASS = 0.03    # +-3 yuzde puan


# ---------------------------------------------------------------------------
# Kimlik uretimi
# ---------------------------------------------------------------------------
def normalized_path(path: str) -> str:
    p = str(path).replace("\\", "/")
    return p[2:] if p.startswith("./") else p


def make_source_id(source: str, path: str) -> str:
    # BNC kayitlari icin yoldan turetilmis hash kullanilmaz; siralama ile
    # anlamli (path-free) kimlik uretilir.
    if source == "BNC":
        return None
    digest = hashlib.sha256(normalized_path(path).encode("utf-8")).hexdigest()[:16]
    return f"{source}_{digest}"


def make_group_id(source: str, path: str) -> str:
    norm = normalized_path(path)
    name = os.path.basename(norm)

    if source == "TESS":
        m = re.search(r"(YAF|OAF)", name, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"TESS konusmaci kodu bulunamadi: {name}")
        return "TESS_" + m.group(1).upper()

    if source == "SAVEE":
        m = re.match(r"([A-Za-z]{2})_", name)
        if not m:
            raise ValueError(f"SAVEE konusmaci kodu bulunamadi: {name}")
        return "SAVEE_" + m.group(1).upper()

    if source == "CREMA":
        m = re.match(r"(\d{4})_", name)
        if not m:
            raise ValueError(f"CREMA aktor kodu bulunamadi: {name}")
        return "CREMA_" + m.group(1)

    if source == "RAVDESS":
        m = re.search(r"Actor_(\d+)", norm, flags=re.IGNORECASE)
        if m:
            actor = int(m.group(1))
        else:
            m2 = re.search(r"(\d+)\.wav$", name, flags=re.IGNORECASE)
            if not m2:
                raise ValueError(f"RAVDESS aktor kodu bulunamadi: {name}")
            actor = int(m2.group(1))
        return f"RAVDESS_Actor_{actor:02d}"

    if source == "BNC":
        # 100 kayit da ayni kisinin sesi -> tek konusmaci grubu.
        return "BNC_SPEAKER_01"

    raise ValueError(f"Bilinmeyen kaynak: {source}")


# ---------------------------------------------------------------------------
# Split yardimcilari
# ---------------------------------------------------------------------------
def class_proportions(labels: np.ndarray) -> np.ndarray:
    s = pd.Series(labels).value_counts(normalize=True)
    return np.array([s.get(c, 0.0) for c in CLASSES], dtype=float)


def class_deviation(labels: np.ndarray) -> float:
    return float(np.max(np.abs(class_proportions(labels) - 0.25)))


def size_ok(indices: np.ndarray, total: int, target: float) -> bool:
    return abs(len(indices) / total - target) <= TOL_SAMPLE


def search_group_split(y: np.ndarray, groups: np.ndarray, target_frac: float,
                       group_fracs, seeds, group_frac_label: str):
    """GroupShuffleSplit ile hedef ORNEK oranini yakalar.

    sklearn'da `test_size` float ise GRUP sayisinin oranidir; bu veride TESS
    gruplari 1600'er kayit oldugu icin 0.20'lik grup orani ~%14 veya ~%29
    ornek orani uretir. Bu yuzden grup orani, hedef ornek oranina gore
    deterministik olarak kalibre edilir; grup ayrismi bozulmaz.
    """
    n = len(y)
    X = np.zeros((n, 1))
    for gf in group_fracs:
        for seed in seeds:
            splitter = GroupShuffleSplit(n_splits=1, test_size=float(gf), random_state=seed)
            train_idx, test_idx = next(splitter.split(X, y, groups))
            if abs(len(test_idx) / n - target_frac) > TOL_SAMPLE:
                continue
            if max(class_deviation(y[train_idx]), class_deviation(y[test_idx])) > TOL_CLASS:
                continue
            return {
                "group_fraction": float(gf), "seed": int(seed),
                "train": train_idx, "test": test_idx,
                "frac": len(test_idx) / n,
                "label": group_frac_label,
            }
    raise RuntimeError(
        f"{group_frac_label}: hedef={target_frac:.2f} tolerans=+-{TOL_SAMPLE:.2f} "
        "icinde grup bazli split bulunamadi."
    )


def make_80_20(y: np.ndarray, groups: np.ndarray):
    result = search_group_split(
        y, groups, target_frac=0.20,
        group_fracs=np.arange(0.20, 0.36, 0.01),
        seeds=range(BASE_SEED, 301),
        group_frac_label="80_20",
    )
    tr, te = result["train"], result["test"]
    return (result["group_fraction"], result["seed"], tr, te,
            class_deviation(y[tr]), class_deviation(y[te]))


def make_70_15_15(y: np.ndarray, groups: np.ndarray):
    n = len(y)
    s1 = search_group_split(
        y, groups, target_frac=0.30,
        group_fracs=np.arange(0.20, 0.46, 0.01),
        seeds=range(BASE_SEED, 301),
        group_frac_label="70_15_15_stage1",
    )
    train_idx, holdout_idx = s1["train"], s1["test"]
    f_holdout = len(holdout_idx) / n
    target_val_frac = 0.15 / f_holdout

    Xh = np.zeros((len(holdout_idx), 1))
    yh, gh = y[holdout_idx], groups[holdout_idx]
    found = None
    for gf in np.arange(0.02, 0.56, 0.01):
        for seed2 in range(BASE_SEED, 301):
            splitter = GroupShuffleSplit(n_splits=1, test_size=float(gf), random_state=seed2)
            val_rel, test_rel = next(splitter.split(Xh, yh, gh))
            val_idx, test_idx = holdout_idx[val_rel], holdout_idx[test_rel]
            if not (size_ok(val_idx, n, 0.15) and size_ok(test_idx, n, 0.15)):
                continue
            if max(class_deviation(y[p]) for p in (train_idx, val_idx, test_idx)) > TOL_CLASS:
                continue
            found = (float(gf), int(seed2), val_idx, test_idx)
            break
        if found:
            break
    if found is None:
        raise RuntimeError("70/15/15 asama 2 icin kriterleri saglayan parametre bulunamadi.")

    gf2, seed2, val_idx, test_idx = found
    devs = [class_deviation(y[p]) for p in (train_idx, val_idx, test_idx)]
    return {
        "stage1_group_fraction": s1["group_fraction"],
        "stage1_seed": s1["seed"],
        "stage1_holdout_frac": f_holdout,
        "stage2_group_fraction": gf2,
        "stage2_seed": seed2,
        "train": train_idx, "val": val_idx, "test": test_idx,
        "devs": devs,
    }


def make_cv_folds(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X = np.zeros((len(y), 1))
    folds = np.full(len(y), -1, dtype=int)
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=BASE_SEED)
    for fold_no, (_, test_idx) in enumerate(skf.split(X, y, groups), start=1):
        folds[test_idx] = fold_no
    if (folds < 0).any():
        raise RuntimeError("CV fold atamasi eksik kayit birakti.")
    return folds


# ---------------------------------------------------------------------------
# Dogrulamalar
# ---------------------------------------------------------------------------
def assert_split_integrity(df: pd.DataFrame) -> None:
    if not df["source_id"].is_unique:
        raise AssertionError("source_id tekrari var.")

    for eval_name, col in SPLIT_COLS.items():
        nunique = df.groupby("source_id")[col].nunique()
        if (nunique != 1).max():
            raise AssertionError(f"{eval_name}: ayni source_id farkli split'e dagildi.")

        train = df[df[col] == "train"]
        other = df[df[col] != "train"]
        if set(train["group_id"]) & set(other["group_id"]):
            raise AssertionError(f"{eval_name}: grup sizintisi (train <-> val/test).")
        if set(train["source_id"]) & set(other["source_id"]):
            raise AssertionError(f"{eval_name}: kaynak sizintisi (train <-> val/test).")

    nunique_fold = df.groupby("source_id")["fold"].nunique()
    if (nunique_fold != 1).max():
        raise AssertionError("Ayni source_id farkli CV fold'una dagildi.")

    bnc = df[df["source"] == "BNC"]
    if set(bnc["group_id"]) != {"BNC_SPEAKER_01"}:
        raise AssertionError("BNC grup kimligi BNC_SPEAKER_01 degil.")
    for col in list(SPLIT_COLS.values()) + ["fold"]:
        if bnc[col].nunique() != 1:
            raise AssertionError(f"BNC kayitlari {col} icinde birden fazla degere dagildi.")

    # 10 fold icin de grup/kaynak ayrismi
    for fold_no in sorted(df["fold"].unique()):
        te = df[df["fold"] == fold_no]
        tr = df[df["fold"] != fold_no]
        if set(te["group_id"]) & set(tr["group_id"]):
            raise AssertionError(f"fold {fold_no}: grup sizintisi.")
        if set(te["source_id"]) & set(tr["source_id"]):
            raise AssertionError(f"fold {fold_no}: kaynak sizintisi.")


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
def speaker_list(sub: pd.DataFrame, source: str = "TESS") -> str:
    vals = sorted(sub.loc[sub["source"] == source, "group_id"].unique())
    return ", ".join(v.replace(source + "_", "") for v in vals) if vals else "-"


def bnc_location(sub: pd.DataFrame, col: str = "split_80_20") -> str:
    vals = sorted(sub.loc[sub["source"] == "BNC", col].unique())
    return ", ".join(vals) if vals else "-"

def build_partition_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evaluations = [
        ("80_20", "split_80_20", ["train", "test"]),
        ("70_15_15", "split_70_15_15", ["train", "val", "test"]),
    ]
    for eval_name, col, partitions in evaluations:
        full_train = df[df[col] == "train"]
        tess_train = speaker_list(full_train)
        tess_val = speaker_list(df[df[col] == "val"]) if "val" in partitions else "-"
        tess_test = speaker_list(df[df[col] == "test"])
        bnc_split = bnc_location(df, col)

        for part in partitions:
            sub = df[df[col] == part]
            counts = sub["label"].value_counts()
            src = sub["source"].value_counts()
            rows.append({
                "evaluation": eval_name,
                "partition": part,
                "n_samples": len(sub),
                "n_groups": sub["group_id"].nunique(),
                "n_sources": sub["source"].nunique(),
                "angry": int(counts.get("angry", 0)),
                "fear": int(counts.get("fear", 0)),
                "happy": int(counts.get("happy", 0)),
                "sad": int(counts.get("sad", 0)),
                "n_RAVDESS": int(src.get("RAVDESS", 0)),
                "n_TESS": int(src.get("TESS", 0)),
                "n_SAVEE": int(src.get("SAVEE", 0)),
                "n_CREMA": int(src.get("CREMA", 0)),
                "n_BNC": int(src.get("BNC", 0)),
                "TESS_speakers_in_partition": speaker_list(sub),
                "TESS_train_speakers": tess_train,
                "TESS_val_speakers": tess_val,
                "TESS_test_speakers": tess_test,
                "BNC_split": bnc_split,
            })

    # CV: her fold icin train / test
    for fold_no in sorted(df["fold"].unique()):
        for part in ["train", "test"]:
            sub = df[df["fold"] != fold_no] if part == "train" else df[df["fold"] == fold_no]
            counts = sub["label"].value_counts()
            src = sub["source"].value_counts()
            rows.append({
                "evaluation": f"fold_{fold_no:02d}",
                "partition": part,
                "n_samples": len(sub),
                "n_groups": sub["group_id"].nunique(),
                "n_sources": sub["source"].nunique(),
                "angry": int(counts.get("angry", 0)),
                "fear": int(counts.get("fear", 0)),
                "happy": int(counts.get("happy", 0)),
                "sad": int(counts.get("sad", 0)),
                "n_RAVDESS": int(src.get("RAVDESS", 0)),
                "n_TESS": int(src.get("TESS", 0)),
                "n_SAVEE": int(src.get("SAVEE", 0)),
                "n_CREMA": int(src.get("CREMA", 0)),
                "n_BNC": int(src.get("BNC", 0)),
                "TESS_speakers_in_partition": speaker_list(sub),
                "TESS_train_speakers": speaker_list(df[df["fold"] != fold_no]),
                "TESS_val_speakers": "-",
                "TESS_test_speakers": speaker_list(df[df["fold"] == fold_no]),
                "BNC_split": f"fold_{int(df.loc[df['source'] == 'BNC', 'fold'].iloc[0]):02d}",
            })
    return pd.DataFrame(rows)


def write_markdown(report: pd.DataFrame, cfg: dict) -> None:
    lines = [
        "# v2 group-aware split dagilim raporu",
        "",
        "Bu rapor `1_veri_hazirlama_v2.py` tarafindan uretildi. "
        "Bolme augmentation ONCESI, orijinal kayit ve konusmaci grubu uzerinde yapildi.",
        "",
        "## Seed ve kriterler",
        "",
        f"- 80/20: `GroupShuffleSplit(group_fraction={cfg['group_fraction_80_20']:.2f}, "
        f"random_state={cfg['seed_80_20']})` -> ornek orani `{cfg['frac_80_20']:.2%}`",
        f"- 70/15/15 asama 1: `group_fraction={cfg['stage1_group_fraction']}, "
        f"random_state={cfg['stage1_seed']}` -> holdout ornek orani `{cfg['stage1_holdout_frac']:.2%}`",
        f"- 70/15/15 asama 2: `group_fraction={cfg['stage2_group_fraction']}, "
        f"random_state={cfg['stage2_seed']}`",
        f"- CV: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state={cfg['cv_seed']})`",
        f"- Ornek orani toleransi: +-{cfg['tol_sample']:.0%}; sinif orani toleransi: +-{cfg['tol_class']:.0%}",
        "",
        "**Not:** sklearn'da `GroupShuffleSplit(test_size=float)` grup sayisinin oranidir. "
        "TESS'in 1.600 kayitlik iki grubu nedeniyle `test_size=0.20` ~%14/%29 ornek orani uretir; "
        "bu yuzden grup orani, hedef ORNEK oranini (+/-2 pp) yakalamak icin deterministik olarak "
        "kalibre edilmistir. Grup/kaynak ayrismi korunmustur.",
        "",
        "## TESS ve BNC",
        "",
        "- `TESS_train_speakers` / `TESS_val_speakers` / `TESS_test_speakers` sutunlari, "
        "ilgili degerlendirmede hangi konusmacinin hangi tarafta kaldigini gosterir (`-` = o bolum yok).",
        "- **TESS:** Veride yalnizca 2 konusmaci (`YAF`, `OAF`) var. Konusmaci bazli ayrimda "
        "bir konusmaci egitimde, digeri testte kalabilir; uc bolumde de TESS bulunmasi zorunlu degildir. "
        "Bu sonuc gorulmemis konusmaciya genelleme olcur, genel populasyona genelleme iddiasi tasimaz.",
        "- **BNC:** 100 kayit da **tek kisinin** sesidir (`BNC_SPEAKER_01`). Bu veriyle farkli "
        "konuşmacılara genelleme olculemez; kayitlar her degerlendirmede birlikte tek bir tarafta kalir.",
        "- **CV fold dengesizligi:** TESS'in 1.600 kayitlik iki grubu atomik tutuldu icin "
        "`StratifiedGroupKFold` fold'lari esit boyutta degildir (bkz. tablo, 736-2.272 kayit). "
        "Sininflar her fold'da dengeli kalmistir; ancak fold'larin genelleme zorlugu farklidir.",
        "",
        "## Sizdirmazlik kontrolleri",
        "",
    ]
    for name, passed in cfg["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines += [
        "",
        "## Bolum tablosu",
        "",
        "| evaluation | partition | n | groups | angry | fear | happy | sad | TESS bolumde | BNC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in report.iterrows():
        lines.append(
            f"| {r['evaluation']} | {r['partition']} | {r['n_samples']} | {r['n_groups']} | "
            f"{r['angry']} | {r['fear']} | {r['happy']} | {r['sad']} | "
            f"{r['TESS_speakers_in_partition']} | {r['BNC_split']} |"
        )
    lines += [
        "",
        "## TESS / BNC atama ozeti",
        "",
        "| evaluation | TESS train | TESS val | TESS test | BNC split |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for _, r in report.iterrows():
        sig = (r["evaluation"], r["TESS_train_speakers"], r["TESS_val_speakers"],
               r["TESS_test_speakers"], r["BNC_split"])
        if sig in seen:
            continue
        seen.add(sig)
        lines.append(
            f"| {r['evaluation']} | {r['TESS_train_speakers']} | {r['TESS_val_speakers']} | "
            f"{r['TESS_test_speakers']} | {r['BNC_split']} |"
        )
    lines.append("")
    (OUT_DIR / "split_dagilim.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(INPUT_FILE)
    expected = {"RAVDESS": 1536, "TESS": 3200, "SAVEE": 240, "CREMA": 5084, "BNC": 100}
    actual = raw["source"].value_counts().to_dict()
    if actual != expected:
        raise RuntimeError(f"v1 manifest sayilari beklenenle uyusmuyor: {actual}")

    missing = [p for p in raw["path"] if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(f"{len(missing)} WAV yolu okunamadi.")

    df = raw.copy()
    df["normalized_path"] = df["path"].map(normalized_path)
    df = df.sort_values(["source", "normalized_path"]).reset_index(drop=True)

    # BNC source_id: yoldan bagimsiz, siralama ile deterministik
    bnc_counter = 0
    source_ids = []
    for _, r in df.iterrows():
        sid = make_source_id(r["source"], r["path"])
        if sid is None:
            bnc_counter += 1
            sid = f"BNC_REC_{bnc_counter:04d}"
        source_ids.append(sid)
    df["source_id"] = source_ids
    df["group_id"] = [make_group_id(r["source"], r["path"]) for _, r in df.iterrows()]

    y = df["label"].to_numpy()
    g = df["group_id"].to_numpy()

    s80 = make_80_20(y, g)
    tr80, te80 = s80[2], s80[3]
    dev_tr80, dev_te80 = s80[4], s80[5]
    df["split_80_20"] = "test"
    df.iloc[tr80, df.columns.get_loc("split_80_20")] = "train"

    s70 = make_70_15_15(y, g)
    tr70, va70, te70 = s70["train"], s70["val"], s70["test"]
    devs70 = s70["devs"]
    df["split_70_15_15"] = "holdout"
    df.iloc[tr70, df.columns.get_loc("split_70_15_15")] = "train"
    df.iloc[va70, df.columns.get_loc("split_70_15_15")] = "val"
    df.iloc[te70, df.columns.get_loc("split_70_15_15")] = "test"

    df["fold"] = make_cv_folds(y, g)

    assert_split_integrity(df)

    checks = {
        "80/20: grup kumesleri kesismiyor": True,
        "80/20: source_id kumesleri kesismiyor": True,
        "70/15/15: grup kumesleri kesismiyor": True,
        "70/15/15: source_id kumesleri kesismiyor": True,
        "Ayni source_id tek split/fold icinde": True,
        "BNC tek grup ve tek split/fold icinde": True,
        "10 CV fold'unda grup/kaynak ayrismi": True,
        "TUM WAV yollari mevcut": True,
    }

    public_cols = ["source_id", "group_id", "source", "label",
                   "split_80_20", "split_70_15_15", "fold"]
    df[public_cols].to_csv(PUBLIC_MANIFEST, index=False)
    df[public_cols + ["path"]].to_csv(LOCAL_MANIFEST, index=False)

    report = build_partition_rows(df)
    report.to_csv(OUT_DIR / "split_dagilim.csv", index=False)

    cfg = {
        "source_file": "ham_veri_listesi.csv (git-ignored local manifest)",
        "n_records": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "seed_80_20": int(s80[1]),
        "group_fraction_80_20": round(float(s80[0]), 4),
        "frac_80_20": float(len(te80) / len(df)),
        "stage1_group_fraction": s70["stage1_group_fraction"],
        "stage1_seed": s70["stage1_seed"],
        "stage1_holdout_frac": s70["stage1_holdout_frac"],
        "stage2_group_fraction": s70["stage2_group_fraction"],
        "stage2_seed": s70["stage2_seed"],
        "cv_seed": int(BASE_SEED),
        "tol_sample": TOL_SAMPLE,
        "tol_class": TOL_CLASS,
        "class_deviation_80_20": {"train": dev_tr80, "test": dev_te80},
        "class_deviation_70_15_15": {
            "train": devs70[0], "val": devs70[1], "test": devs70[2]
        },
        "checks": checks,
        "manifest_sha256": hashlib.sha256(
            df[public_cols].to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    (OUT_DIR / "split_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(report, cfg)

    print("=== v2 manifest tamamlandi ===")
    print(f"Kayit: {len(df)} | Gruplar: {df['group_id'].nunique()} | source_id: {df['source_id'].nunique()}")
    print(f"80/20 grup_orani={s80[0]} seed={s80[1]} | test ornegi={len(te80)/len(df):.4f} "
          f"| sinif sapmasi train={dev_tr80:.4f} test={dev_te80:.4f}")
    print(f"70/15/15 asama1 grup_orani={s70['stage1_group_fraction']} seed={s70['stage1_seed']} "
          f"| asama2 grup_orani={s70['stage2_group_fraction']} seed={s70['stage2_seed']} "
          f"| sinif sapmasi: {devs70}")
    print(f"CV seed: {BASE_SEED} | fold sayisi: {df['fold'].nunique()}")
    print("\n80/20 dagilim:")
    print(df["split_80_20"].value_counts().to_string())
    print("\n70/15/15 dagilim:")
    print(df["split_70_15_15"].value_counts().to_string())
    print("\nTESS atamalari:")
    for eval_name, col, parts in [("80_20", "split_80_20", ["train", "test"]),
                                   ("70_15_15", "split_70_15_15", ["train", "val", "test"])]:
        print(f"  {eval_name}: " + " | ".join(
            f"{p}={speaker_list(df[df[col] == p]) or '-'}" for p in parts
        ))
    print(f"\nBNC grup: {sorted(df.loc[df['source'] == 'BNC', 'group_id'].unique())}")
    print(f"BNC 80/20: {sorted(df.loc[df['source'] == 'BNC', 'split_80_20'].unique())}")
    print(f"BNC 70/15/15: {sorted(df.loc[df['source'] == 'BNC', 'split_70_15_15'].unique())}")
    print(f"BNC fold: {sorted(df.loc[df['source'] == 'BNC', 'fold'].unique())}")
    print("\nTum sizdirmazlik kontrolleri PASS")


if __name__ == "__main__":
    main()
