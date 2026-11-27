"""A1.2 10-fold sonrasi SALT OKUNUR dogrulama. Cikti yalnizca A1_2/kfold/.

Egitim/CV/70-15-15/model kaydi yapmaz; A0/A1.1/A3 dosyalarina yazmaz.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KF = Path(__file__).resolve().parent
A12 = KF.parent
ROOT = A12.parent
MANIFEST = ROOT / "manifest_v2_dedup_draft.csv"
A0_OOF = ROOT / "sonuclar_v2_dedup" / "kfold" / "oof_tahminler.csv"
FEATURES = A12 / "final_oznitelik_seti_A1_2.csv"
DETAIL = A12 / "f0_istatistikleri.csv"
F0_COLS = [str(i) for i in range(184, 191)]
META = ["source_id", "group_id", "source", "label", "aug_id",
        "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]


def main() -> None:
    rows: list[dict] = []

    def ck(name: str, ok: bool, detail: str) -> None:
        rows.append({"kontrol": name, "durum": "PASS" if ok else "FAIL", "detay": detail})

    need = ["oof_tahminler.csv", "kfold_per_fold.csv", "kfold_ozet.csv",
            "oof_metrikler.csv", "oof_kaynak_bazli.csv", "oof_sinif_f1.csv",
            "oof_karisiklik.csv", "f0_imputasyon_degerleri.csv",
            "sizdirma_kontrolleri.csv", "cv_config.json",
            "karsilastirma_A0_A1_2.csv", "karsilastirma_A0_A1_2.md",
            "oof_cm_RF.png", "oof_cm_MLP.png"]
    missing = [n for n in need if not (KF / n).exists()]
    ck("tum_cikti_dosyalari", not missing, f"eksik={missing}")

    oof = pd.read_csv(KF / "oof_tahminler.csv", dtype={"source_id": str, "group_id": str,
                                                        "source": str, "y_true": str,
                                                        "rf_pred": str, "mlp_pred": str})
    man = pd.read_csv(MANIFEST, dtype=str)
    man = man[man.dedup_status.ne("REVIEW_LABEL")].copy()
    man["fold"] = man.fold.astype(int)
    ck("oof_satir_sayisi", len(oof) == 7780, f"{len(oof)}")
    ck("oof_source_tekillik", oof.source_id.is_unique, f"{oof.source_id.nunique()} tekil")
    mp = dict(zip(man.source_id, man.fold))
    ck("oof_fold_manifest", bool((oof.source_id.map(mp).to_numpy() == oof.fold.to_numpy()).all()),
       "birebir")

    a0 = pd.read_csv(A0_OOF, dtype={"source_id": str})
    ck("A0_kumesi_ayni", set(a0.source_id) == set(oof.source_id),
       f"A0={len(a0)}, A1_2={len(oof)}")

    pf = pd.read_csv(KF / "kfold_per_fold.csv")
    ck("per_fold_20_satir", len(pf) == 20, f"{len(pf)}")
    ck("seed_43_52", sorted(pf.seed.unique().tolist()) == list(range(43, 53)),
       str(sorted(pf.seed.unique().tolist())))
    ck("her_model_10_fold", bool((pf.groupby("model").fold.nunique() == 10).all()),
       str(pf.groupby("model").size().to_dict()))

    for name, n in [("oof_metrikler.csv", 2), ("oof_kaynak_bazli.csv", 10),
                    ("oof_sinif_f1.csv", 8), ("oof_karisiklik.csv", 32),
                    ("f0_imputasyon_degerleri.csv", 70)]:
        d = pd.read_csv(KF / name)
        ck(f"{name}_satir", len(d) == n, f"{len(d)} / {n}")

    sz = pd.read_csv(KF / "sizdirma_kontrolleri.csv")
    ck("sizdirma_tum_PASS", bool((sz.result == "PASS").all()),
       f"{int((sz.result == 'PASS').sum())}/{len(sz)}")

    # F0 imputasyon medyanlarini bagimsiz yeniden hesapla
    feat = pd.read_csv(FEATURES, dtype={c: str for c in META})
    det = pd.read_csv(DETAIL, dtype={"source_id": str, "aug_id": str})
    fmap = {(s, a): int(f) for s, a, f in zip(det.source_id, det.aug_id, det.fallback)}
    feat["fb"] = [fmap[(s, a)] for s, a in zip(feat.source_id, feat.aug_id)]
    imp = pd.read_csv(KF / "f0_imputasyon_degerleri.csv")
    max_diff, count_bad = 0.0, 0
    for k in range(1, 11):
        tr = feat[feat.fold.astype(int) != k]
        te = feat[(feat.fold.astype(int) == k) & (feat.aug_id == "orig")]
        med = np.median(tr.loc[tr.fb == 0, F0_COLS].to_numpy(dtype=float), axis=0)
        sub = imp[imp.fold == k].sort_values("kolon")
        got = sub.train_medyan.to_numpy(dtype=float)
        max_diff = max(max_diff, float(np.abs(med - got).max()))
        exp_tr, exp_te = int((tr.fb == 1).sum()), int((te.fb == 1).sum())
        if not (bool((sub.n_train_impute == exp_tr).all()) and
                bool((sub.n_test_impute == exp_te).all())):
            count_bad += 1
    ck("imputasyon_medyan_train_den", max_diff <= 1e-9, f"maks-diff={max_diff:.3g}")
    ck("imputasyon_sayimlari", count_bad == 0, f"bozuk fold={count_bad}")

    model_files = [p.name for p in KF.rglob("*") if p.is_file() and
                   p.suffix.lower() in {".joblib", ".h5", ".keras", ".pkl", ".pickle"}]
    ck("model_veya_scaler_kaydi_yok", not model_files, str(model_files))
    split_files = [p.name for p in KF.rglob("*") if "70_15_15" in p.name]
    ck("70_15_15_ciktisi_yok", not split_files, str(split_files))

    cmp = pd.read_csv(KF / "karsilastirma_A0_A1_2.csv")
    ck("karsilastirma_bolumleri", set(cmp.bolum.unique()) ==
       {"genel", "kaynak", "fold", "fold_ozet"}, str(sorted(cmp.bolum.unique())))

    h = hashlib.sha256()
    with open(A0_OOF, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    ck("A0_OOF_salt_okundu", A0_OOF.is_file(), f"sha256={h.hexdigest()[:16]}")

    out = pd.DataFrame(rows)
    out.to_csv(KF / "dogrulama_sonrasi.csv", index=False, encoding="utf-8")
    fails = out[out.durum.eq("FAIL")]
    md = ["# A1.2 10-fold sonrasi dogrulama", "",
          f"- Kontrol: **{len(out)}**, PASS: **{len(out)-len(fails)}**, FAIL: **{len(fails)}**",
          "", "| kontrol | durum | detay |", "|---|---|---|"]
    for r in out.itertuples(index=False):
        md.append(f"| {r.kontrol} | {r.durum} | {r.detay} |")
    md += ["", "SALT OKUNUR; egitim/CV/70-15-15/model kaydi yapilmadi.", ""]
    (KF / "dogrulama_sonrasi.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Kontrol: {len(out)} | PASS {len(out)-len(fails)} | FAIL {len(fails)}")
    if len(fails):
        print(fails.to_string(index=False))


if __name__ == "__main__":
    main()
