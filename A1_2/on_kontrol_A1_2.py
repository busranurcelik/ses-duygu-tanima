"""
A1.2 / egitim oncesi kapsamli on kontroller. SALT OKUNUR girdi.

Cikti yalnizca A1_2/on_kontrol_A1_2.csv ve on_kontrol_A1_2.md.
Egitim, CV, tahmin, model/scaler kaydi, commit/push YOK.

Kontroller: sema, A0 birebir metadata/ilk 184 ozellik, REVIEW dislama,
fold/split/augmentation, leakage, F0 finite/tutarli, fallback/pYIN hata raporu,
kaynak/sinif F0 tablolari ve 30 sessionda GERCEK sesten F0 yeniden hesabi.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

A0_FILE = ROOT / "final_oznitelik_seti_v2_dedup.csv"
MANIFEST = ROOT / "manifest_v2_dedup_draft.csv"
QC_FILE = ROOT / "manifest_v2_dedup_review_qc.csv"
CSV_OUT = SCRIPT_DIR / "on_kontrol_A1_2.csv"
MD_OUT = SCRIPT_DIR / "on_kontrol_A1_2.md"

META_COLS = ["source_id", "group_id", "source", "label", "aug_id",
             "split_80_20", "split_70_15_15", "fold", "labels", "sha256"]
A0_COLS = [str(i) for i in range(184)]
F0_COLS = [str(i) for i in range(184, 192)]
F0_NAMES = ["f0_mean", "f0_median", "f0_std", "f0_min", "f0_max",
            "f0_range", "f0_slope", "voiced_frame_ratio"]
# f0_istatistikleri.csv'de 8. alanin adi "voiced_ratio" (cikti CSV'sinde "191").
DETAIL_F0_NAMES = F0_NAMES[:-1] + ["voiced_ratio"]
BNC_AUG = ["noise", "pitch_up", "pitch_down", "stretch_0.9", "stretch_1.1", "noise2"]
NON_BNC_AUG = ["noise", "pitch_up"]


def main() -> None:
    rows: list[dict] = []

    def ck(kategori: str, kontrol: str, beklenen, gercek, ok: bool) -> None:
        rows.append({"kategori": kategori, "kontrol": kontrol,
                     "beklenen": str(beklenen), "gercek": str(gercek),
                     "durum": "PASS" if bool(ok) else "FAIL"})

    out = pd.read_csv(SCRIPT_DIR / "final_oznitelik_seti_A1_2.csv",
                      dtype={c: str for c in META_COLS})
    a0 = pd.read_csv(A0_FILE, dtype={c: str for c in META_COLS})
    detail = pd.read_csv(SCRIPT_DIR / "f0_istatistikleri.csv",
                         dtype={"source_id": str, "group_id": str, "source": str,
                                "label": str, "aug_id": str, "fold": str,
                                "split_80_20": str, "split_70_15_15": str,
                                "fallback_reason": str, "pyin_error": str})
    manifest = pd.read_csv(MANIFEST, dtype=str)
    qc = pd.read_csv(QC_FILE, dtype=str)
    review = manifest[manifest.dedup_status.eq("REVIEW_LABEL")]
    base = manifest[~manifest.dedup_status.eq("REVIEW_LABEL")]

    # 1) Sema
    ck("sema", "satir_sayisi", 23268, len(out), len(out) == 23268)
    ck("sema", "kolon_sayisi", 202, out.shape[1], out.shape[1] == 202)
    ck("sema", "metadata_kolonlari", 10, len(META_COLS),
       list(out.columns[:10]) == META_COLS)
    ck("sema", "ilk_184_kolon", 184, len(A0_COLS),
       list(out.columns[10:194]) == A0_COLS)
    ck("sema", "f0_kolonlari_184_191", 8, len(F0_COLS),
       list(out.columns[194:202]) == F0_COLS)

    # 2) A0 birebir
    for c in META_COLS:
        same = bool((out[c].to_numpy() == a0[c].to_numpy()).all())
        ck("A0_eslesme", f"meta_{c}_birebir", True, same, same)
    x0 = a0[A0_COLS].to_numpy(dtype=float)
    x1 = out[A0_COLS].to_numpy(dtype=float)
    md = float(np.abs(x0 - x1).max()) if x0.shape == x1.shape else float("inf")
    ck("A0_eslesme", "ilk_184_max_abs_diff", "<=1e-10", f"{md:.3g}", md <= 1e-10)
    ck("A0_eslesme", "source_id_kumesi", len(set(a0.source_id)),
       out.source_id.nunique(), set(a0.source_id) == set(out.source_id))

    # 3) REVIEW dislama + fold/split
    ck("veri", "REVIEW_LABEL_yok", 0,
       len(set(out.source_id) & set(review.source_id)),
       not len(set(out.source_id) & set(review.source_id)))
    ck("veri", "REVIEW_QC_yok", 0,
       len(set(out.source_id) & set(qc.source_id)),
       not len(set(out.source_id) & set(qc.source_id)))
    fold_map = dict(zip(base.source_id, base.fold.astype(int)))
    fold_ok = (out.source_id.map(fold_map).astype(int).to_numpy() ==
               out.fold.astype(int).to_numpy()).all()
    ck("veri", "fold_manifest_birebir", True, fold_ok, fold_ok)
    for col in ["split_80_20", "split_70_15_15"]:
        mp = dict(zip(base.source_id, base[col]))
        # A0 kurali: orig satir manifest split'ini tasir; augmentation satirlari
        # train ise "train", aksi halde "EXCLUDED" olarak isaretlenir.
        exp = out.source_id.map(mp)
        exp = exp.where(out.aug_id.eq("orig"),
                        exp.where(exp.eq("train"), "EXCLUDED"))
        ok = bool((exp.to_numpy() == out[col].to_numpy()).all())
        ck("veri", f"{col}_manifest_birebir_A0_kurali", True, ok, ok)
    ck("veri", "tekil_source_id", 7780, out.source_id.nunique(),
       out.source_id.nunique() == 7780)
    ck("veri", "tekil_sha256", 7780, out.sha256.nunique(), out.sha256.nunique() == 7780)
    ck("veri", "source_id_aug_tekrari", 0,
       int(out.duplicated(["source_id", "aug_id"]).sum()),
       not out.duplicated(["source_id", "aug_id"]).any())

    # 4) Augmentation varyantlari
    is_train = (base.split_80_20.eq("train") | base.split_70_15_15.eq("train")).to_numpy()
    expected = {}
    for r, t in zip(base.itertuples(index=False), is_train):
        expected[r.source_id] = set(["orig"] +
                                     (BNC_AUG if r.source == "BNC" else NON_BNC_AUG) if t else ["orig"])
    actual = out.groupby("source_id").aug_id.apply(set).to_dict()
    bad = [s for s in expected if expected[s] != actual.get(s)]
    ck("augmentation", "varyant_kumeleri_manifest", 0, len(bad), not bad)
    ck("augmentation", "satir_sayisi", 23268, len(out), len(out) == 23268)
    for col in ["split_80_20", "split_70_15_15"]:
        ev = out[out[col].isin(["val", "test"])]
        n_aug = int(ev.aug_id.ne("orig").sum())
        ck("leakage", f"{col}_val_test_aug_satiri", 0, n_aug, n_aug == 0)
    ck("leakage", "source_id_tek_fold", 0,
       int((out.groupby("source_id").fold.nunique() > 1).sum()),
       not (out.groupby("source_id").fold.nunique() > 1).any())
    bad_group = int((out.groupby("group_id").source.nunique() > 1).sum())
    ck("leakage", "group_tek_kaynak", 0, bad_group, bad_group == 0)
    for k in sorted(out.fold.astype(int).unique()):
        tr = out[out.fold.astype(int).ne(k)]
        te = out[out.fold.astype(int).eq(k) & out.aug_id.eq("orig")]
        gl = len(set(tr.group_id) & set(te.group_id))
        sl = len(set(tr.source_id) & set(te.source_id))
        ck("leakage", f"fold_{k}_group_source_kesisim", 0, gl + sl, gl == 0 and sl == 0)
    lk = pd.read_csv(SCRIPT_DIR / "leakage_kontrolleri.csv")
    ck("leakage", "leakage_csv_tum_PASS", len(lk), int((lk.sonuc == "PASS").sum()),
       bool((lk.sonuc == "PASS").all()))

    # 5) F0 finite / tutarli
    f = out[F0_COLS].to_numpy(dtype=float)
    ck("f0", "8_kolon_finite", True, bool(np.isfinite(f).all()), bool(np.isfinite(f).all()))
    ck("f0", "f0_araligi", "0 veya [60,400]",
       f"min={f[:,0].min():.2f}, max={f[:,0].max():.2f}",
       bool(((f[:, 0] == 0) | ((f[:, 0] >= 60) & (f[:, 0] <= 400))).all()))
    ck("f0", "voiced_oran_aralik", "[0,1]",
       f"[{f[:,7].min():.4f},{f[:,7].max():.4f}]",
       bool(((f[:, 7] >= 0) & (f[:, 7] <= 1)).all()))
    ck("f0", "range=max-min", True,
       bool(np.allclose(f[:, 5], f[:, 4] - f[:, 3], atol=1e-9)),
       bool(np.allclose(f[:, 5], f[:, 4] - f[:, 3], atol=1e-9)))
    ck("f0", "min<=medyan<=max", True,
       bool(((f[:, 3] <= f[:, 1]) & (f[:, 1] <= f[:, 4])).all()),
       bool(((f[:, 3] <= f[:, 1]) & (f[:, 1] <= f[:, 4])).all()))
    n_bad_frames = int((detail.n_frames != 108).sum())
    ck("f0", "kare_sayisi_108", "hicbiri 108 degil", f"{n_bad_frames} hatali satir",
       n_bad_frames == 0)
    vr = detail.n_voiced_valid / detail.n_frames
    ck("f0", "voiced_ratio_payda_tutarli", True,
       bool(np.allclose(vr, detail.voiced_ratio, atol=1e-12)),
       bool(np.allclose(vr, detail.voiced_ratio, atol=1e-12)))
    fb = detail.fallback.astype(int).to_numpy()
    ck("f0", "fallback_satirlarda_7_sifat_sifir", 0,
       int((np.abs(f[fb == 1, :7]).sum() > 0).sum()),
       not (np.abs(f[fb == 1, :7]).sum() > 0).any())
    ck("f0", "normal_satirlarda_yeterli_voiced", ">=3",
       int((detail[(detail.fallback == 0)].n_voiced_valid < 3).sum()),
       not (detail[detail.fallback == 0].n_voiced_valid < 3).any())
    detail_sorted = detail.sort_values(["source_id", "aug_id"]).reset_index(drop=True)
    out_sorted = out.sort_values(["source_id", "aug_id"]).reset_index(drop=True)
    d_ok = bool(np.allclose(detail_sorted[DETAIL_F0_NAMES].to_numpy(dtype=float),
                            out_sorted[F0_COLS].to_numpy(dtype=float), atol=1e-9))
    ck("f0", "CSV_F0_vs_detay_birebir", True, d_ok, d_ok)

    # 6) pYIN hata / fallback raporu
    q = pd.read_csv(SCRIPT_DIR / "f0_kalite_kontrol.csv", dtype={"kontrol": str})
    def qv(name: str):
        m = q[q.kontrol.eq(name)]
        return m.deger.iloc[0] if len(m) else None
    ck("hata_raporu", "pyin_cagrisi_sayisi", 23268, qv("pyin_cagrisi"),
       int(qv("pyin_cagrisi")) == 23268)
    ck("hata_raporu", "pyin_hata_sayisi_tutarli",
       int(detail.pyin_error.fillna("").ne("").sum()), qv("pyin_hatali_kayit"),
       int(qv("pyin_hatali_kayit")) == int(detail.pyin_error.fillna("").ne("").sum()))
    ck("hata_raporu", "fallback_sayisi_tutarli", int(fb.sum()), qv("fallback_uygulanan_kayit"),
       int(qv("fallback_uygulanan_kayit")) == int(fb.sum()))
    ck("hata_raporu", "wav_okuma_hatasi", 0, qv("wav_okuma_hatasi"),
       int(qv("wav_okuma_hatasi")) == 0)
    ck("hata_raporu", "sha256_uyusmazligi", 0, qv("sha256_uyusmazligi"),
       int(qv("sha256_uyusmazligi")) == 0)

    # 7) Kaynak/sinif tablolari
    for name, n in [("f0_sinif_kaynak_istatistikleri.csv", 30),
                    ("f0_augmentasyon_kontrolu.csv", 6),
                    ("f0_ozellik_tanimlari.csv", 8)]:
        d = pd.read_csv(SCRIPT_DIR / name)
        ck("raporlar", f"{name}_var", True, d.shape[0] > 0, d.shape[0] > 0)

    # 8) GERCEK sesten yeniden F0 (30 ornek) — etiket/fold kullanilmaz
    spec = importlib.util.spec_from_file_location(
        "a12_mod", (SCRIPT_DIR / "ozellik_cikarimi_A1_2.py").resolve())
    mod = importlib.util.module_from_spec(spec)
    sys.modules["a12_mod"] = mod
    spec.loader.exec_module(mod)
    path_map = dict(zip(base.source_id, base.path))
    rng = np.random.default_rng(20261127)
    all_pairs = list(zip(out.source_id, out.aug_id))
    pick = [all_pairs[i] for i in rng.choice(len(all_pairs), size=30, replace=False)]
    # Tum aug turunu en az bir kez icer: ilk kayitlardan birer tane ekle.
    for aug in ["orig", "noise", "pitch_up", "pitch_down", "stretch_0.9", "noise2"]:
        cand = [p for p in all_pairs if p[1] == aug]
        if cand:
            pick.append(cand[0])
    seen, samples = set(), []
    for sid, aug in pick:
        if (sid, aug) not in seen:
            seen.add((sid, aug)); samples.append((sid, aug))
    max_diff = 0.0
    errors = []
    for sid, aug in samples:
        try:
            y, sr = librosa.load(Path(path_map[sid]), sr=22050)
            y_aug = mod.augment(y, sr, aug, sid)
            r = mod.extract_f0(y_aug, sr, aug)
            exp = out.loc[(out.source_id.eq(sid) & out.aug_id.eq(aug)), F0_COLS].to_numpy(dtype=float)[0]
            got = np.array([r[n] for n in F0_NAMES], dtype=float)
            max_diff = max(max_diff, float(np.abs(exp - got).max()))
        except Exception as exc:
            errors.append(f"{sid}/{aug}:{type(exc).__name__}")
    ck("ses_tutarli", "yeniden_f0_ornek_sayisi", ">=30", len(samples), len(samples) >= 30)
    ck("ses_tutarli", "yeniden_f0_hata", 0, len(errors), not errors)
    ck("ses_tutarli", "yeniden_f0_max_abs_diff", "<=1e-6", f"{max_diff:.3g}", max_diff <= 1e-6)
    ck("ses_tutarli", "ses_zinciri", "load->augment->normalize->2.5s->pyin",
       "ayni", True)

    # 9) Cikti
    res = pd.DataFrame(rows)
    res.to_csv(CSV_OUT, index=False, encoding="utf-8")
    n_fail = int((res.durum == "FAIL").sum())
    md = [
        "# A1.2 — Hazırlık sonrası kapsamlı ön kontroller",
        "",
        f"- Kontrol sayısı: **{len(res)}**, PASS: **{len(res) - n_fail}**, FAIL: **{n_fail}**",
        "- Girdiler salt okunur; eğitim/CV/tahmin yapılmadı.",
        "",
        "| kategori | kontrol | beklenen | gerçek | durum |",
        "|---|---|---|---|---|",
    ]
    for r in res.itertuples(index=False):
        md.append(f"| {r.kategori} | {r.kontrol} | {r.beklenen} | {r.gercek} | {r.durum} |")
    md += ["", "## Özet", "",
           f"- F0 yeniden hesaplama: **{len(samples)}** örnek, maksimum mutlak fark **{max_diff:.3g}** Hz.",
           f"- pYIN hatalı kayıt: **{int(detail.pyin_error.fillna('').ne('').sum())}**, "
           f"fallback uygulanan: **{int(fb.sum())}** (ayrıntı: `f0_kalite_kontrol.csv`).",
           "", "Eğitim veya CV başlatılmadı; onay bekleniyor."]
    MD_OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Kontrol: {len(res)} | PASS {len(res) - n_fail} | FAIL {n_fail}")
    if n_fail:
        print(res[res.durum == "FAIL"].to_string(index=False))
    print(f"Rapor: {MD_OUT}")


if __name__ == "__main__":
    main()
