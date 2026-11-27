"""
A0 vs A1.2 — salt okunur istatistiksel karsilastirma (10-fold OOF).

Kurallar:
  * Mevcut A0 ve A1.2 OOF tahminleri kullanilir; EGITIM/CV TEKRAR CALISTIRILMAZ.
  * Eslesme (source_id, y_true) anahtariyla yapilir; group/source/fold birebir dogrulanir.
  * Paired cluster bootstrap: küme = group_id, tabaka = source.
    Her source icin o source'a ait group_id'lerden ayni sayida küme
    "with replacement" çekilir; ayni örneklenen satirlar A0 ve A12 icin kullanilir.
  * B = 10000, seed = 42 (sabit), yöntem = percentile %95 CI.
  * Accuracy ve macro-F1 farklari + her sinifin F1 farki raporlanir.
  * McNemar yalnizca yardimci analizdir (exact binomial, two-sided);
    grup bagimliligini modellemedigi icin birincil kanit degildir.
  * Ciktilar YALNIZCA A1_2/kfold/ altinda; mevcut dosyalar degistirilmez.
    70/15/15, model kaydi ve commit/push YOK.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

KF = Path(__file__).resolve().parent              # A1_2/kfold
ROOT = KF.parent.parent                           # ses_duygu_projesi
A0_OOF = ROOT / "sonuclar_v2_dedup" / "kfold" / "oof_tahminler.csv"
A12_OOF = KF / "oof_tahminler.csv"
A12_METRICS = KF / "oof_metrikler.csv"

MODELS = ["RF", "MLP"]
CLASSES = ["angry", "fear", "happy", "sad"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
N_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05
EXPECTED_N = 7780
METRIC_KEYS = ["accuracy", "f1_macro"] + [f"f1_{c}" for c in CLASSES]
GENERAL_KEYS = ["accuracy", "f1_macro"]
CLASS_KEYS = [f"f1_{c}" for c in CLASSES]

OUTPUTS = [
    KF / "istatistiksel_karsilastirma_raporu.md",
    KF / "istatistik_genel.csv",
    KF / "istatistik_sinif_f1.csv",
    KF / "istatistik_mcnemar.csv",
    KF / "istatistik_bootstrap_farklari.csv",
    KF / "istatistik_kontrolleri.csv",
    KF / "istatistik_kontrolleri.md",
    KF / "istatistik_konfig.json",
]
for p in OUTPUTS:
    if p.resolve().parent != KF.resolve():
        raise SystemExit(f"REFUSE OUTSIDE A1_2/kfold: {p}")
    if p.exists():
        raise SystemExit(f"REFUSE OVERWRITE: {p}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    lines = ["| " + " | ".join(str(c) for c in d.columns) + " |",
             "|" + "|".join(["---"] * len(d.columns)) + "|"]
    for row in d.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def f1_per_class(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """sklearn f1_score(average=None, zero_division=0) ile ayni degerler."""
    tp = np.bincount(yt[yt == yp], minlength=len(CLASSES)).astype(float)
    true_cnt = np.bincount(yt, minlength=len(CLASSES)).astype(float)
    pred_cnt = np.bincount(yp, minlength=len(CLASSES)).astype(float)
    denom = true_cnt + pred_cnt
    out = np.zeros(len(CLASSES), dtype=float)
    np.divide(2.0 * tp, denom, out=out, where=denom > 0)
    return out


def metric_values(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    f1c = f1_per_class(yt, yp)
    out = {"accuracy": float((yt == yp).mean()), "f1_macro": float(f1c.mean())}
    for c, v in zip(CLASSES, f1c):
        out[f"f1_{c}"] = float(v)
    return out


def check_frame(name: str, d: pd.DataFrame, checks: list[dict]) -> None:
    need = {"source_id", "group_id", "source", "fold", "y_true", "rf_pred", "mlp_pred"}
    missing = sorted(need - set(d.columns))
    checks.append({"kontrol": f"{name}_kolonlar", "durum": "PASS" if not missing else "FAIL",
                   "detay": "eksik=" + str(missing)})
    checks.append({"kontrol": f"{name}_satir_ve_tekillik",
                   "durum": "PASS" if len(d) == EXPECTED_N and d.source_id.is_unique else "FAIL",
                   "detay": f"n={len(d)}, tekil_source={d.source_id.nunique()}"})
    valid = bool(d.y_true.isin(CLASSES).all() and
                 d.rf_pred.isin(CLASSES).all() and d.mlp_pred.isin(CLASSES).all())
    bad = int((~d.y_true.isin(CLASSES)).sum() + (~d.rf_pred.isin(CLASSES)).sum() +
              (~d.mlp_pred.isin(CLASSES)).sum())
    checks.append({"kontrol": f"{name}_etiket_ve_tahmin_degerleri",
                   "durum": "PASS" if valid else "FAIL", "detay": f"gecersiz={bad}"})


def main() -> None:
    print("=== A0 vs A1.2 istatistiksel karsilastirma (salt okunur) ===", flush=True)
    checks: list[dict] = []

    a0 = pd.read_csv(A0_OOF, dtype={"source_id": str, "group_id": str, "source": str,
                                    "fold": int, "y_true": str, "rf_pred": str, "mlp_pred": str})
    a12 = pd.read_csv(A12_OOF, dtype={"source_id": str, "group_id": str, "source": str,
                                      "fold": int, "y_true": str, "rf_pred": str, "mlp_pred": str})
    check_frame("A0", a0, checks)
    check_frame("A1_2", a12, checks)
    if any(c["durum"] == "FAIL" for c in checks):
        raise AssertionError("Girdi OOF dosyalari beklenen sartlari saglamiyor: " +
                             str([c for c in checks if c["durum"] == "FAIL"]))

    a0r = a0.rename(columns={"group_id": "group_id_a0", "source": "source_a0",
                             "fold": "fold_a0", "rf_pred": "rf_pred_a0",
                             "mlp_pred": "mlp_pred_a0"})
    a12r = a12.rename(columns={"group_id": "group_id_a12", "source": "source_a12",
                               "fold": "fold_a12", "rf_pred": "rf_pred_a12",
                               "mlp_pred": "mlp_pred_a12"})
    m = pd.merge(a0r, a12r, on=["source_id", "y_true"], how="outer",
                 validate="one_to_one", indicator=True)
    matched = bool((m._merge == "both").all() and len(m) == EXPECTED_N)
    checks.append({"kontrol": "source_id_y_true_eslesmesi",
                   "durum": "PASS" if matched else "FAIL",
                   "detay": f"both={int((m._merge == 'both').sum())}, toplam={len(m)}"})
    same = {}
    for ca, cb, label in [("group_id_a0", "group_id_a12", "group_id"),
                          ("source_a0", "source_a12", "source"),
                          ("fold_a0", "fold_a12", "fold")]:
        same[label] = bool((m[ca] == m[cb]).all())
        checks.append({"kontrol": f"{label}_birebir", "durum": "PASS" if same[label] else "FAIL",
                       "detay": "eslesmeyen=" + str(int((m[ca] != m[cb]).sum()))})
    if not matched or not all(same.values()):
        raise AssertionError("Eslesme dogrulamasi basarisiz; tahmin uretilmeyecek.")

    paired = pd.DataFrame({
        "source_id": m.source_id.to_numpy(),
        "group_id": m.group_id_a12.to_numpy(),
        "source": m.source_a12.to_numpy(),
        "fold": m.fold_a12.to_numpy(),
        "y_true": m.y_true.to_numpy(),
        "rf_pred_a0": m.rf_pred_a0.to_numpy(),
        "rf_pred_a12": m.rf_pred_a12.to_numpy(),
        "mlp_pred_a0": m.mlp_pred_a0.to_numpy(),
        "mlp_pred_a12": m.mlp_pred_a12.to_numpy(),
    }).reset_index(drop=True)

    yt = np.array([LABEL2ID[v] for v in paired.y_true.to_numpy()], dtype=np.int64)
    preds = {
        "RF": {"A0": np.array([LABEL2ID[v] for v in paired.rf_pred_a0], dtype=np.int64),
               "A1_2": np.array([LABEL2ID[v] for v in paired.rf_pred_a12], dtype=np.int64)},
        "MLP": {"A0": np.array([LABEL2ID[v] for v in paired.mlp_pred_a0], dtype=np.int64),
                "A1_2": np.array([LABEL2ID[v] for v in paired.mlp_pred_a12], dtype=np.int64)},
    }
    corr = {mo: {"A0": yt == preds[mo]["A0"], "A1_2": yt == preds[mo]["A1_2"]} for mo in MODELS}

    observed = {mo: {"A0": metric_values(yt, preds[mo]["A0"]),
                     "A1_2": metric_values(yt, preds[mo]["A1_2"])} for mo in MODELS}

    # sklean referans kontrolu ve mevcut A1.2 metrik dosyasiyla cakisma kontrolu
    existing = pd.read_csv(A12_METRICS)
    max_sklearn_diff, max_file_diff = 0.0, 0.0
    for mo in MODELS:
        sk_acc = accuracy_score(yt, preds[mo]["A1_2"])
        sk_f1 = f1_score(yt, preds[mo]["A1_2"], labels=range(len(CLASSES)),
                         average=None, zero_division=0)
        max_sklearn_diff = max(max_sklearn_diff, abs(sk_acc - observed[mo]["A1_2"]["accuracy"]),
                               max(abs(sk_f1[i] - observed[mo]["A1_2"][f"f1_{c}"])
                                   for i, c in enumerate(CLASSES)))
        row = existing[existing.model == mo].iloc[0]
        for key in ["accuracy", "f1_macro"] + [f"f1_{c}" for c in CLASSES]:
            max_file_diff = max(max_file_diff, abs(float(row[key]) - observed[mo]["A1_2"][key]))
    checks.append({"kontrol": "sklearn_formul_cakismasi", "durum": "PASS" if max_sklearn_diff < 1e-12 else "FAIL",
                   "detay": f"maks_diff={max_sklearn_diff:.3g}"})
    checks.append({"kontrol": "mevcut_oof_metrikleriyle_cakisma",
                   "durum": "PASS" if max_file_diff < 1e-9 else "FAIL",
                   "detay": f"maks_diff={max_file_diff:.3g}"})

    # ---------------- Paired cluster bootstrap (source tabakali) ----------------
    sources = sorted(paired.source.unique())
    group_rows: dict[str, list[np.ndarray]] = {}
    group_counts = {}
    record_counts = {}
    for src in sources:
        sub = paired[paired.source == src]
        groups = sorted(sub.group_id.unique())
        group_rows[src] = [sub.index[sub.group_id == g].to_numpy(dtype=np.int64) for g in groups]
        group_counts[src] = len(groups)
        record_counts[src] = len(sub)

    boot = {mo: {k: np.empty(N_BOOT, dtype=float) for k in METRIC_KEYS} for mo in MODELS}
    row_counts = np.empty(N_BOOT, dtype=np.int64)
    rng = np.random.default_rng(BOOT_SEED)
    print(f"Bootstrap: B={N_BOOT}, seed={BOOT_SEED}, tabaka={len(sources)}, "
          f"kume={sum(group_counts.values())}", flush=True)

    for b in range(N_BOOT):
        parts: list[np.ndarray] = []
        for src in sources:
            clusters = group_rows[src]
            draw = rng.integers(0, len(clusters), size=len(clusters))
            for j in draw:
                parts.append(clusters[int(j)])
        idx = np.concatenate(parts)
        row_counts[b] = idx.size
        ys = yt[idx]
        for mo in MODELS:
            p0 = preds[mo]["A0"][idx]
            p1 = preds[mo]["A1_2"][idx]
            f0 = f1_per_class(ys, p0)
            f12 = f1_per_class(ys, p1)
            boot[mo]["accuracy"][b] = (ys == p1).mean() - (ys == p0).mean()
            boot[mo]["f1_macro"][b] = f12.mean() - f0.mean()
            for i, c in enumerate(CLASSES):
                boot[mo][f"f1_{c}"][b] = f12[i] - f0[i]
        if (b + 1) % 2500 == 0:
            print(f"  {b + 1}/{N_BOOT} tekrar tamamlandi", flush=True)

    def summarize(mo: str, key: str) -> dict:
        d = boot[mo][key]
        lo, hi = np.percentile(d, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        return {
            "model": mo, "metrik": key,
            "A0": observed[mo]["A0"][key],
            "A1_2": observed[mo]["A1_2"][key],
            "gozlenen_fark": observed[mo]["A1_2"][key] - observed[mo]["A0"][key],
            "bootstrap_ortalama": float(d.mean()),
            "bootstrap_medyan": float(np.median(d)),
            "bootstrap_std": float(d.std(ddof=1)),
            "ci95_alt": float(lo), "ci95_ust": float(hi),
            "ci95_sifir_iceriyor": bool(lo <= 0.0 <= hi),
            "oran_fark_pozitif": float((d > 0).mean()),
            "oran_fark_negatif": float((d < 0).mean()),
            "n_tekrar": N_BOOT, "seed": BOOT_SEED,
            "yontem": "source-tabakali paired cluster bootstrap (group_id), percentile CI",
        }

    genel = pd.DataFrame([summarize(mo, k) for mo in MODELS for k in GENERAL_KEYS])
    sinif = pd.DataFrame([summarize(mo, k) for mo in MODELS for k in CLASS_KEYS])
    sinif.insert(2, "sinif", [k.replace("f1_", "") for k in sinif.metrik])
    sinif = sinif.drop(columns=["metrik"])

    # McNemar (yardimci)
    mc_rows = []
    for mo in MODELS:
        c0, c12 = corr[mo]["A0"], corr[mo]["A1_2"]
        n01 = int((c0 & ~c12).sum())     # A0 dogru, A1.2 yanlis -> A0 lehine
        n10 = int((~c0 & c12).sum())     # A0 yanlis, A1.2 dogru -> A1.2 lehine
        n = n01 + n10
        pval = float(binomtest(n10, n, p=0.5, alternative="two-sided").pvalue) if n else float("nan")
        mc_rows.append({"model": mo, "A0_dogru_A12_yanlis_n01": n01,
                        "A0_yanlis_A12_dogru_n10": n10, "uyusmayan_toplam": n,
                        "exact_binomial_p": pval,
                        "not": "yardimci analiz; grup bagimliligini modellemez"})
    mcnemar = pd.DataFrame(mc_rows)

    # Ham bootstrap farklari (denetim icin)
    raw_rows = []
    for mo in MODELS:
        for b in range(N_BOOT):
            row = {"model": mo, "tekrar": b + 1}
            for k in METRIC_KEYS:
                row[k] = float(boot[mo][k][b])
            raw_rows.append(row)
    raw = pd.DataFrame(raw_rows)

    checks.append({"kontrol": "bootstrap_tekrar_sayisi",
                   "durum": "PASS" if len(raw) == 2 * N_BOOT else "FAIL",
                   "detay": f"{len(raw)} satir (2 model x {N_BOOT})"})
    checks.append({"kontrol": "bootstrap_seed_sabit",
                   "durum": "PASS", "detay": str(BOOT_SEED)})
    checks.append({"kontrol": "tabakali_kume_yapisi",
                   "durum": "PASS", "detay": "source=" + str(group_counts) +
                   " | kayit=" + str(record_counts)})
    checks.append({"kontrol": "orneklenen_satir_sayisi_araligi",
                   "durum": "PASS", "detay": f"min={row_counts.min()}, maks={row_counts.max()}, "
                   f"ort={row_counts.mean():.1f} (orijinal={EXPECTED_N})"})
    checks.append({"kontrol": "CI_ve_mcnemar_uretildi",
                   "durum": "PASS" if len(genel) == 4 and len(sinif) == 8 and len(mcnemar) == 2 else "FAIL",
                   "detay": f"genel={len(genel)}, sinif={len(sinif)}, mcnemar={len(mcnemar)}"})
    checks.append({"kontrol": "ciktilar_yalnizca_A1_2_kfold",
                   "durum": "PASS", "detay": str(KF)})
    checks.append({"kontrol": "egitim_tekrar_calistirilmadi",
                   "durum": "PASS", "detay": "yalnizca OOF CSVleri okundu"})

    ctrl = pd.DataFrame(checks)
    ctrl.to_csv(KF / "istatistik_kontrolleri.csv", index=False, encoding="utf-8")
    genel.to_csv(KF / "istatistik_genel.csv", index=False, encoding="utf-8")
    sinif.to_csv(KF / "istatistik_sinif_f1.csv", index=False, encoding="utf-8")
    mcnemar.to_csv(KF / "istatistik_mcnemar.csv", index=False, encoding="utf-8")
    raw.to_csv(KF / "istatistik_bootstrap_farklari.csv", index=False, encoding="utf-8")

    cfg = {
        "analysis": "A0_vs_A1_2_OOF_statistical_comparison",
        "data": {
            "A0_oof": str(A0_OOF.relative_to(ROOT)),
            "A1_2_oof": str(A12_OOF.relative_to(ROOT)),
            "A1_2_metrics_reference": str(A12_METRICS.relative_to(ROOT)),
            "A0_sha256": sha256(A0_OOF),
            "A1_2_sha256": sha256(A12_OOF),
            "n_paired": len(paired),
        },
        "matching": {"keys": ["source_id", "y_true"],
                     "verified_equal": ["group_id", "source", "fold"]},
        "bootstrap": {
            "type": "paired cluster bootstrap, stratified by source",
            "cluster": "group_id",
            "strata": "source",
            "n_repeats": N_BOOT,
            "seed": BOOT_SEED,
            "ci": "percentile 95% (2.5, 97.5)",
            "draw": "her source icin ayni sayida grup, with replacement; ayni satirlar iki modelde",
            "groups_by_source": group_counts,
            "records_by_source": record_counts,
        },
        "mcnemar": "exact binomial two-sided, auxiliary only (group dependence ignored)",
        "models": MODELS,
        "classes": CLASSES,
        "not_run": ["training", "CV", "70_15_15", "model_save", "commit/push"],
    }
    with open(KF / "istatistik_konfig.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    # ---------------- Rapor ----------------
    n_pass = int((ctrl.durum == "PASS").sum())
    md = [
        "# A0 vs A1.2 — istatistiksel karşılaştırma (10-fold OOF)",
        "",
        "Mevcut A0 ve A1.2 OOF tahminleri kullanıldı; **eğitim/CV tekrar çalıştırılmadı**, "
        "yeni tahmin üretilmedi. Eşleştirme `(source_id, y_true)` anahtarıyla yapıldı; "
        "`group_id`, `source` ve `fold` değerlerinin birebir aynı olduğu doğrulandı.",
        "",
        "## Yöntem",
        "",
        f"- **Paired cluster bootstrap:** küme `group_id`, tabaka `source`. Her kaynak için "
        f"o kaynağa ait {sum(group_counts.values())} kümenin tamamı kendi içinde "
        "`with replacement` örneklenip birleştirildi; aynı örneklenen satırlar A0 ve A1.2 "
        "için birlikte kullanıldı. Böylece grup/konuşmacı bağımlılığı ve kaynak tabakası korundu.",
        f"- **Tekrar sayısı:** {N_BOOT:,}; **sabit seed:** {BOOT_SEED}; "
        "**güven aralığı:** yüzdelik (percentile) %95 [2,5; 97,5].",
        "- **Metrikler:** OOF accuracy, macro-F1 ve dört sınıfın F1 farkı (A1.2 − A0).",
        "- **McNemar:** yalnızca yardımcı analiz (exact binomial, two-sided); "
        "grup bağımlılığını modellemediği için birincil kanıt değildir.",
        "",
        f"- Kontroller: **{n_pass}/{len(ctrl)} PASS**. "
        f"Örneklenen satır sayısı aralığı: {row_counts.min()}–{row_counts.max()} "
        f"(orijinal {EXPECTED_N}; küme boyutları değiştiği için sabit değil).",
        "- Kaynak başına küme sayısı: " + ", ".join(f"{k}={v}" for k, v in group_counts.items()) + ".",
        "- BNC tek küme olduğundan o tabakada bootstrap çeşitliliği yoktur; "
        "SAVEE (4) ve TESS (2) kümeleri de azdır, kaynak bazlı yorumlar sınırlıdır.",
        "",
        "## 1. Accuracy ve macro-F1 farkları",
        "",
        "| model | metrik | A0 | A1.2 | gözlenen fark | bootstrap ort. | %95 CI | CI sıfır içeriyor | fark>0 oranı |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for _, r in genel.iterrows():
        md.append(f"| {r['model']} | {r['metrik']} | {r['A0']:.4f} | {r['A1_2']:.4f} | "
                  f"{r['gozlenen_fark']:+.4f} | {r['bootstrap_ortalama']:+.4f} | "
                  f"[{r['ci95_alt']:+.4f}, {r['ci95_ust']:+.4f}] | "
                  f"{'evet' if r['ci95_sifir_iceriyor'] else 'hayır'} | "
                  f"{r['oran_fark_pozitif']:.3f} |")
    md += ["", "## 2. Sınıf bazlı F1 farkları", "",
           "| model | sınıf | A0 F1 | A1.2 F1 | fark | %95 CI | CI sıfır içeriyor |",
           "|---|---|---:|---:|---:|---|---|"]
    for _, r in sinif.iterrows():
        md.append(f"| {r['model']} | {r['sinif']} | {r['A0']:.4f} | {r['A1_2']:.4f} | "
                  f"{r['gozlenen_fark']:+.4f} | [{r['ci95_alt']:+.4f}, {r['ci95_ust']:+.4f}] | "
                  f"{'evet' if r['ci95_sifir_iceriyor'] else 'hayır'} |")
    md += ["", "## 3. McNemar (yardımcı)", "",
           "| model | A0 doğru / A1.2 yanlış (n01) | A0 yanlış / A1.2 doğru (n10) | uyuşmayan | exact p |",
           "|---|---:|---:|---:|---:|"]
    for _, r in mcnemar.iterrows():
        md.append(f"| {r['model']} | {r['A0_dogru_A12_yanlis_n01']} | "
                  f"{r['A0_yanlis_A12_dogru_n10']} | {r['uyusmayan_toplam']} | "
                  f"{r['exact_binomial_p']:.6f} |")
    md += ["", "## 4. Yorum", ""]
    for _, r in genel.iterrows():
        durum = ("CI sıfırı içeriyor; gözlenen artışın belirsizliği 'fark yok' olasılığını dışlamıyor."
                 if r["ci95_sifir_iceriyor"] else
                 "CI sıfırı içermiyor; gözlenen fark bu bootstrap dağılımında sıfırdan ayrışıyor.")
        md.append(f"- **{r['model']} {r['metrik']}:** fark {r['gozlenen_fark']:+.4f}, "
                  f"%95 CI [{r['ci95_alt']:+.4f}, {r['ci95_ust']:+.4f}]. {durum}")
    md += [
        "",
        "Sınıf bazlı F1 farklarının tamamı aynı yöntemle raporlanmıştır; "
        "fear/happy değişimleri de bu tablodadır. Farklar betimseldir; "
        "anlamlılık iddiası yalnızca bootstrap CI ve yardımcı McNemar sonuçlarıyla sınırlıdır.",
        "",
        "## 5. Kontroller",
        "",
        "| kontrol | durum | detay |",
        "|---|---|---|",
    ]
    for _, r in ctrl.iterrows():
        md.append(f"| {r['kontrol']} | {r['durum']} | {r['detay']} |")
    md += [
        "",
        "Çıktıların tamamı `A1_2/kfold/` altındadır; mevcut OOF/rapor dosyaları "
        "değiştirilmedi. 70/15/15, model kaydı ve commit/push yapılmadı.",
        "",
    ]
    (KF / "istatistiksel_karsilastirma_raporu.md").write_text("\n".join(md), encoding="utf-8")

    ck_md = ["# A1.2 istatistiksel karşılaştırma kontrolleri", "",
             f"- PASS: **{n_pass}/{len(ctrl)}**, FAIL: **{len(ctrl)-n_pass}**", "",
             "| kontrol | durum | detay |", "|---|---|---|"]
    for _, r in ctrl.iterrows():
        ck_md.append(f"| {r['kontrol']} | {r['durum']} | {r['detay']} |")
    ck_md += ["", "Salt okunur analiz; eğitim/CV/70-15-15/model kaydi/commit/push yok.", ""]
    (KF / "istatistik_kontrolleri.md").write_text("\n".join(ck_md), encoding="utf-8")

    print("\n=== Genel ===")
    print(genel[["model", "metrik", "gozlenen_fark", "ci95_alt", "ci95_ust",
                 "ci95_sifir_iceriyor"]].to_string(index=False))
    print("\n=== Sinif F1 ===")
    print(sinif[["model", "sinif", "gozlenen_fark", "ci95_alt", "ci95_ust",
                 "ci95_sifir_iceriyor"]].to_string(index=False))
    print("\n=== McNemar (yardimci) ===")
    print(mcnemar.to_string(index=False))
    print(f"\nKontroller: {n_pass}/{len(ctrl)} PASS. Rapor: istatistiksel_karsilastirma_raporu.md")


if __name__ == "__main__":
    main()
