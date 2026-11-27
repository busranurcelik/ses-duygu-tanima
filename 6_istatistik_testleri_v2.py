"""
v2 / adim 6: McNemar, group-cluster bootstrap ve ROC/AUC (eşlesmis analiz).

Kapsam:
  * 70/15/15: RF-B ve MLP, ayni 800 TESS_YAF kaydi; McNemar + ROC/AUC.
  * OOF 10-fold: RF ve MLP, ayni 7780 kayit; McNemar + group-cluster bootstrap.
  * Sayilar manifest_v2_dedup_draft.csv'den (REVIEW_LABEL disinda) dinamik okunur.
  * OOF ROC/AUC: sonuç dosyasinda olasilik yok; yeni egitim yapilmadan uretilemez.
    Bu durum acikca BLOCKED olarak raporlanir; sert etiketlerden AUC uretilmez.
  * Hicbir model egitimi, kaydi veya Git islemi yapilmaz.
  * Tum girdiler/sonuclar sonuclar_v2_dedup/ altindadir; eski sonuclar_v2/ yazilmaz.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
MANIFEST_FILE = ROOT / "manifest_v2_dedup_draft.csv"
CMP_DIR = ROOT / "sonuclar_v2_dedup" / "karsilastirma"
KFOLD_DIR = ROOT / "sonuclar_v2_dedup" / "kfold"
OUT_DIR = ROOT / "sonuclar_v2_dedup" / "istatistik"

CLASSES = ["angry", "fear", "happy", "sad"]   # sabit sinif sirasi
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
BOOT_SEED = 42
N_BOOT = 5000
PROB_COLS = [f"prob_{c}" for c in CLASSES]
EXPECTED_REVIEW_LABEL = 2


def load_dedup_manifest() -> pd.DataFrame:
    if not MANIFEST_FILE.is_file():
        raise RuntimeError(f"Girdi manifest yok: {MANIFEST_FILE}")
    man = pd.read_csv(MANIFEST_FILE, dtype=str)
    required = {
        "source_id", "group_id", "source", "label",
        "split_70_15_15", "dedup_status",
    }
    missing = required - set(man.columns)
    if missing:
        raise RuntimeError(f"Manifest eksik kolon: {sorted(missing)}")
    n_review = int((man.dedup_status == "REVIEW_LABEL").sum())
    if n_review != EXPECTED_REVIEW_LABEL:
        raise AssertionError(
            f"REVIEW_LABEL beklenen {EXPECTED_REVIEW_LABEL}, bulunan {n_review}."
        )
    man = man[man.dedup_status != "REVIEW_LABEL"].copy()
    if not man.source_id.is_unique:
        raise AssertionError("Dedup manifest source_id tekil degil.")
    return man


OUT_DIR.mkdir(parents=True, exist_ok=True)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray, name_a: str, name_b: str) -> dict:
    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))   # A yanlis, B dogru
    n10 = int(np.sum(correct_a & ~correct_b))   # A dogru, B yanlis
    n11 = int(np.sum(correct_a & correct_b))
    discordant = n01 + n10
    p = float(binomtest(n01, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "model_a": name_a,
        "model_b": name_b,
        "n": len(correct_a),
        "correct_a": int(correct_a.sum()),
        "correct_b": int(correct_b.sum()),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "discordant": discordant,
        "p_value": p,
        "acc_a": float(correct_a.mean()),
        "acc_b": float(correct_b.mean()),
        "acc_diff_a_minus_b": float(correct_a.mean() - correct_b.mean()),
    }


def cluster_bootstrap(df: pd.DataFrame, correct_a_col: str, correct_b_col: str,
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    g = df.groupby("group_id", sort=True)
    sums_a = g[correct_a_col].sum().to_numpy(dtype=float)
    sums_b = g[correct_b_col].sum().to_numpy(dtype=float)
    counts = g.size().to_numpy(dtype=float)
    n_groups = len(counts)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        n = counts[pick].sum()
        diffs[i] = sums_a[pick].sum() / n - sums_b[pick].sum() / n

    observed = float(df[correct_a_col].mean() - df[correct_b_col].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    degenerate = n_groups < 2
    return {
        "n_groups": n_groups,
        "n_records": len(df),
        "n_boot": n_boot,
        "seed": seed,
        "observed_diff": observed,
        "boot_mean_diff": float(diffs.mean()),
        "boot_std_diff": float(diffs.std(ddof=1)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "ci_includes_zero": bool(lo <= 0 <= hi),
        "degenerate": degenerate,
        "note": ("yalnizca 1 grup; cluster bootstrap degenerate, CI yorumlanamaz"
                 if degenerate else "group_id kumeleri birlikte, replacement ile örneklenir"),
    }


def auc_for_model(y_true_idx: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.isfinite(prob).all():
        raise AssertionError("Olasiliklarda NaN/Inf var.")
    if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-4):
        raise AssertionError("Olasilik satirlari 1'e toplanmiyor.")
    ybin = np.zeros((len(y_true_idx), len(CLASSES)), dtype=int)
    ybin[np.arange(len(y_true_idx)), y_true_idx] = 1
    per_class = roc_auc_score(ybin, prob, average=None, multi_class="ovr",
                              labels=np.arange(len(CLASSES)))
    macro = roc_auc_score(ybin, prob, average="macro", multi_class="ovr",
                          labels=np.arange(len(CLASSES)))
    return np.asarray(per_class, dtype=float), float(macro)


def plot_roc(y_true_idx: np.ndarray, probs: dict[str, np.ndarray], title: str, path: Path) -> None:
    plt.figure(figsize=(9, 7))
    ybin = np.zeros((len(y_true_idx), len(CLASSES)), dtype=int)
    ybin[np.arange(len(y_true_idx)), y_true_idx] = 1
    for model_name, prob in probs.items():
        per_class, macro = auc_for_model(y_true_idx, prob)
        for i, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(ybin[:, i], prob[:, i])
            plt.plot(fpr, tpr, lw=1.4,
                     label=f"{model_name} - {cls} (AUC={per_class[i]:.3f})")
        plt.plot([], [], label=f"{model_name} macro-AUC={macro:.3f}", linestyle="--", color="gray")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def load_70(expected_n: int) -> dict:
    rf = pd.read_csv(CMP_DIR / "70_15_15_rf_matched_test_predictions_dedup.csv",
                     dtype={"source_id": str, "group_id": str, "source": str})
    mlp = pd.read_csv(CMP_DIR / "70_15_15_mlp_test_predictions_dedup.csv",
                      dtype={"source_id": str, "group_id": str, "source": str})
    if len(rf) != expected_n or len(mlp) != expected_n:
        raise AssertionError(
            f"70/15/15 tahmin dosyalari {expected_n} satir olmali."
        )
    if not np.array_equal(rf.source_id.to_numpy(), mlp.source_id.to_numpy()):
        raise AssertionError("RF-B ve MLP test source_id sirasi farkli.")
    if not np.array_equal(rf.y_true.to_numpy(), mlp.y_true.to_numpy()):
        raise AssertionError("RF-B ve MLP y_true dizileri farkli.")
    if sorted(rf.group_id.unique()) != ["TESS_YAF"] or sorted(rf.source.unique()) != ["TESS"]:
        raise AssertionError("70/15/15 test kumesi yalnizca TESS_YAF olmali.")
    for d in (rf, mlp):
        if not set(np.unique(d.y_true.to_numpy())).issubset(set(range(4))):
            raise AssertionError("70/15/15 y_true beklenen sinif araliginda degil.")

    paired = pd.DataFrame({
        "source_id": rf.source_id,
        "group_id": rf.group_id,
        "source": rf.source,
        "y_true": [CLASSES[i] for i in rf.y_true.to_numpy()],
        "rf_pred": [CLASSES[i] for i in rf.y_pred.to_numpy()],
        "mlp_pred": [CLASSES[i] for i in mlp.y_pred.to_numpy()],
    })
    for c in PROB_COLS:
        paired[f"rf_{c}"] = rf[c].to_numpy()
        paired[f"mlp_{c}"] = mlp[c].to_numpy()
    paired["rf_correct"] = rf.y_true.to_numpy() == rf.y_pred.to_numpy()
    paired["mlp_correct"] = mlp.y_true.to_numpy() == mlp.y_pred.to_numpy()
    paired.to_csv(OUT_DIR / "eslesmis_tahminler_70_15_15.csv", index=False, encoding="utf-8")

    return {
        "paired": paired,
        "y_true_idx": rf.y_true.to_numpy(dtype=int),
        "probs": {
            "RF-B (70/15/15)": rf[PROB_COLS].to_numpy(dtype=float),
            "MLP (70/15/15)": mlp[PROB_COLS].to_numpy(dtype=float),
        },
    }


def load_oof(expected_n: int) -> dict:
    oof = pd.read_csv(KFOLD_DIR / "oof_tahminler.csv",
                      dtype={"source_id": str, "group_id": str, "source": str})
    if len(oof) != expected_n or not oof.source_id.is_unique:
        raise AssertionError(
            f"OOF dosyasi {expected_n} benzersiz kayit olmali."
        )
    for c in ["y_true", "rf_pred", "mlp_pred"]:
        if not oof[c].isin(CLASSES).all():
            raise AssertionError(f"OOF {c} beklenen sinif etiketlerinde degil.")
    oof = oof.copy()
    oof["rf_correct_recheck"] = oof.y_true.eq(oof.rf_pred)
    oof["mlp_correct_recheck"] = oof.y_true.eq(oof.mlp_pred)
    if not (oof.rf_correct_recheck == oof.rf_correct.astype(bool)).all():
        raise AssertionError("OOF rf_correct tutarsiz.")
    if not (oof.mlp_correct_recheck == oof.mlp_correct.astype(bool)).all():
        raise AssertionError("OOF mlp_correct tutarsiz.")

    paired = oof[["source_id", "group_id", "source", "fold", "y_true", "rf_pred", "mlp_pred",
                  "rf_correct", "mlp_correct"]].copy()
    paired.to_csv(OUT_DIR / "eslesmis_tahminler_oof.csv", index=False, encoding="utf-8")
    return {"paired": paired}


def main() -> None:
    print("=== v2 dedup / adim 6: McNemar + cluster bootstrap + ROC/AUC ===")
    man = load_dedup_manifest()
    expected_70_test = int((man.split_70_15_15 == "test").sum())
    expected_oof = len(man)
    source_counts = man.source.value_counts().to_dict()
    d70 = load_70(expected_70_test)
    doof = load_oof(expected_oof)

    # ---------------- McNemar ----------------
    mc_rows = []
    m70 = mcnemar(d70["paired"].rf_correct.to_numpy(bool),
                  d70["paired"].mlp_correct.to_numpy(bool),
                  "RF-B (70/15/15)", "MLP (70/15/15)")
    m70.update({"evaluation": "70_15_15_TESS_YAF_unseen_speaker",
                "note": "tek konusmaci; bagimsizlik varsayimi ihlal edilebilir"})
    mc_rows.append(m70)

    moof = mcnemar(doof["paired"].rf_correct.to_numpy(bool),
                   doof["paired"].mlp_correct.to_numpy(bool),
                   "RF (OOF 10-fold)", "MLP (OOF 10-fold)")
    moof.update({"evaluation": "OOF_10fold",
                 "note": "ayni konusmacinin farkli kayitlari bagimli olabilir"})
    mc_rows.append(moof)
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(OUT_DIR / "mcnemar_sonuclari.csv", index=False, encoding="utf-8")

    # ---------------- cluster bootstrap ----------------
    b70 = cluster_bootstrap(d70["paired"], "rf_correct", "mlp_correct", seed=BOOT_SEED)
    b70.update({"evaluation": "70_15_15_TESS_YAF_unseen_speaker",
                "model_a": "RF-B", "model_b": "MLP"})
    boof = cluster_bootstrap(doof["paired"], "rf_correct", "mlp_correct", seed=BOOT_SEED)
    boof.update({"evaluation": "OOF_10fold", "model_a": "RF", "model_b": "MLP"})
    boot_df = pd.DataFrame([b70, boof])
    bcols = ["evaluation", "model_a", "model_b", "n_groups", "n_records", "n_boot", "seed",
             "observed_diff", "boot_mean_diff", "boot_std_diff", "ci95_low", "ci95_high",
             "ci_includes_zero", "degenerate", "note"]
    boot_df[bcols].to_csv(OUT_DIR / "group_bootstrap_sonuclari.csv", index=False, encoding="utf-8")

    # ---------------- ROC / AUC (yalnizca 70/15/15) ----------------
    auc_rows = []
    y70 = d70["y_true_idx"]
    for model_name, prob in d70["probs"].items():
        per_class, macro = auc_for_model(y70, prob)
        for i, cls in enumerate(CLASSES):
            auc_rows.append({"evaluation": "70_15_15_TESS_YAF_unseen_speaker",
                             "model": model_name, "scope": cls,
                             "auc": float(per_class[i]), "status": "OK"})
        auc_rows.append({"evaluation": "70_15_15_TESS_YAF_unseen_speaker",
                         "model": model_name, "scope": "macro",
                         "auc": macro, "status": "OK"})

    # OOF: olasilik kaydedilmedigi icin AUC uretilemez (yeni egitim yasak)
    for model_name in ["RF (OOF 10-fold)", "MLP (OOF 10-fold)"]:
        for scope in CLASSES + ["macro"]:
            auc_rows.append({"evaluation": "OOF_10fold", "model": model_name,
                             "scope": scope, "auc": np.nan,
                             "status": "BLOCKED: OOF olasilik kolonlari kaydedilmedi; yeni egitim gerekir"})
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(OUT_DIR / "auc_sonuclari.csv", index=False, encoding="utf-8")

    # Kaynak bazli AUC: 70/15/15 icin TESS (tek kaynak) hesaplanabilir
    src_auc_rows = []
    for model_name, prob in d70["probs"].items():
        _, macro = auc_for_model(y70, prob)
        src_auc_rows.append({"evaluation": "70_15_15_TESS_YAF_unseen_speaker",
                             "model": model_name, "source": "TESS", "n": expected_70_test,
                             "macro_auc": macro, "status": "OK"})
    for model_name in ["RF (OOF 10-fold)", "MLP (OOF 10-fold)"]:
        for src, n in sorted(source_counts.items(), key=lambda item: item[0]):
            src_auc_rows.append({"evaluation": "OOF_10fold", "model": model_name,
                                 "source": src, "n": n, "macro_auc": np.nan,
                                 "status": "BLOCKED: OOF olasilik kolonlari kaydedilmedi"})
    pd.DataFrame(src_auc_rows).to_csv(OUT_DIR / "auc_kaynak_bazli.csv", index=False, encoding="utf-8")

    plot_roc(
        y70, d70["probs"],
        "ROC - 70/15/15 TESS_YAF unseen speaker (RF-B vs MLP)",
        OUT_DIR / "roc_70_15_15_TESS_YAF.png",
    )
    (OUT_DIR / "roc_oof_BEKLEMEDE.txt").write_text(
        "OOF ROC/AUC uretilemedi: sonuclar_v2_dedup/kfold/oof_tahminler.csv yalnizca sert etiket icerir.\n"
        "RF ve MLP OOF olasiliklari 5_kfold_dogrulama_v2.py cagrisinda hesaplanmis olsa da kaydedilmemistir.\n"
        "Sert etiketlerden ROC/AUC uretmek yaniltici oldugu icin yapilmadi.\n"
        "OOF ROC/AUC icin 10-fold egitimin olasiliklarla birlikte tekrar calistirilmasi gerekir (yeni egitim; onay gerekir).\n",
        encoding="utf-8",
    )

    config = {
        "class_order": CLASSES,
        "manifest": str(MANIFEST_FILE.relative_to(ROOT)),
        "review_label_excluded": EXPECTED_REVIEW_LABEL,
        "source_counts": source_counts,
        "mcnemar": "exact binomial two-sided on n01 vs n10; A=RF, B=MLP",
        "cluster_bootstrap": {"seed": BOOT_SEED, "n_boot": N_BOOT,
                              "unit": "group_id", "sampling": "groups with replacement"},
        "roc_auc": "one-vs-rest, per-class and macro",
        "evaluations": {
            "70_15_15": f"RF-B vs MLP, ayni {expected_70_test} TESS_YAF kaydi, olasiliklar mevcut",
            "OOF_10fold": f"RF vs MLP, ayni {expected_oof} kayit, olasiliklar MEVCUT DEGIL",
        },
        "oof_probabilities_saved": False,
        "models_trained": False,
        "previous_files_modified": False,
        "old_results_dir": "sonuclar_v2/ (not written)",
        "commit_push": False,
    }
    with open(OUT_DIR / "istatistik_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---------------- Rapor ----------------
    lines = [
        "# v2 dedup istatistik analizi: McNemar, cluster bootstrap, ROC/AUC",
        "",
        "## Uyarilar",
        "",
        "- Standart McNemar, örneklerin birbirinden bağımsız olduğunu varsayar. "
        "Aynı konuşmacı/gruba ait birden fazla kayıt bu varsayımı zayıflatır; p-değerleri "
        "fazla küçük çıkabilir. Sonuç yalnızca eşleşmiş hata deseninin tanımı olarak okunmalıdır.",
        "- 70/15/15 test kümesi tek gruptur (`TESS_YAF`). `group_id` cluster bootstrap bu nedenle "
        "degenerate olur; güven aralığı yorumlanamaz.",
        "- OOF ROC/AUC hesaplanamadı: OOF tahmin dosyasında olasılık kolonları yok. "
        "Yeni model eğitimi onay kapsamı dışında olduğu için yeniden CV çalıştırılmadı.",
        "- **P-değerine dayanarak model üstünlüğü iddia edilmez.** Etki büyüklüğü (accuracy farkı), "
        "discordant sayıları ve bootstrap aralığı birlikte raporlanır.",
        f"- Sayılar `manifest_v2_dedup_draft.csv` üzerinden dinamik okundu: "
        f"70/15/15 test={expected_70_test}, OOF={expected_oof}. "
        f"Kaynak dağılımı: {source_counts}.",
        "",
        "## McNemar (A=RF, B=MLP)",
        "",
        "| Değerlendirme | n | A doğru | B doğru | n00 | n01 | n10 | n11 | discordant | p | acc farkı (A-B) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in mc_df.iterrows():
        lines.append(
            f"| {r['evaluation']} | {r['n']} | {r['correct_a']} | {r['correct_b']} | "
            f"{r['n00']} | {r['n01']} | {r['n10']} | {r['n11']} | {r['discordant']} | "
            f"{r['p_value']:.6g} | {r['acc_diff_a_minus_b']:+.4f} |"
        )
    lines += ["", "## Group-cluster bootstrap (accuracy farkı, RF - MLP)", "",
              "| Değerlendirme | Gruplar | Kayıt | Boot | Gözlenen fark | CI %95 | CI 0 içerir? | Not |",
              "|---|---:|---:|---:|---:|---|---|---|"]
    for _, r in boot_df.iterrows():
        lines.append(
            f"| {r['evaluation']} | {r['n_groups']} | {r['n_records']} | {r['n_boot']} | "
            f"{r['observed_diff']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | "
            f"{r['ci_includes_zero']} | {r['note']} |"
        )
    lines += ["", "## ROC/AUC", "",
              "- 70/15/15: sınıf bazlı ve macro one-vs-rest AUC `auc_sonuclari.csv` içinde; "
              "grafik `roc_70_15_15_TESS_YAF.png`.",
              "- OOF: `auc_sonuclari.csv` içinde BLOCKED satırları var; "
              "grafik yerine `roc_oof_BEKLEMEDE.txt` oluşturuldu.",
              "- Sınıf sırası sabit: angry, fear, happy, sad.", ""]
    (OUT_DIR / "istatistik_raporu.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== McNemar ===")
    print(mc_df[["evaluation", "n", "n00", "n01", "n10", "n11", "discordant", "p_value",
                 "acc_diff_a_minus_b"]].to_string(index=False))
    print("\n=== Group-cluster bootstrap ===")
    print(boot_df[["evaluation", "n_groups", "observed_diff", "ci95_low", "ci95_high",
                   "ci_includes_zero", "degenerate"]].to_string(index=False))
    print("\n=== AUC (70/15/15) ===")
    print(auc_df[auc_df.status.eq("OK")][["evaluation", "model", "scope", "auc"]].to_string(index=False))
    print("\nOOF ROC/AUC: BLOCKED (olasilik kaydedilmedi; yeni egitim yapilmadi).")
    print("Tum ciktilar sonuclar_v2_dedup/istatistik/ altinda.")


if __name__ == "__main__":
    main()
