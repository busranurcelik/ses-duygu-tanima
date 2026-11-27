"""A1.2 RF-B GUI entegrasyonu için doğrulama testleri.

Bu betik model eğitmez; yalnızca mevcut artefaktları ve GUI adaptörünü okur.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

import arayuz_A1_2_adapter as a12

ROOT = Path(__file__).resolve().parent
PASS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"{name}: {detail}")
    PASS.append(name)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Modül yüklenemedi: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_manifest_path(raw: str) -> Path:
    p = Path(raw.replace("\\", "/"))
    if str(p).startswith("./"):
        p = Path(str(p)[2:])
    return ROOT / p


def main() -> None:
    print("=== A1.2 RF-B GUI entegrasyon testleri ===")

    # 1) Scaler / model / label map / medyan uyumu.
    model, scaler, labels, medians = a12.load_assets()
    check("model_scaler_192_feature_uyumu",
          model.n_features_in_ == 192 and scaler.n_features_in_ == 192,
          f"model={model.n_features_in_}, scaler={scaler.n_features_in_}")
    check("label_map_ve_rf_sinif_sirasi",
          labels == ["angry", "fear", "happy", "sad"]
          and list(model.classes_) == [0, 1, 2, 3],
          f"labels={labels}, classes={list(model.classes_)}")
    check("f0_medyanlari", medians.shape == (7,) and np.isfinite(medians).all(),
          f"shape={medians.shape}")

    # Referans modüller: A0 184 özellik ve A1.2 pYIN çıkarımı.
    a0_mod = load_module("a0_ref", ROOT / "2_oznitelik_cikarimi_v2_dedup.py")
    a12_mod = load_module("a12_ref", ROOT / "A1_2" / "ozellik_cikarimi_A1_2.py")

    # 2) Gerçek test-seti kaydı: 192 özellik, sıra, finite ve uçtan uca tahmin.
    preds = pd.read_csv(
        ROOT / "A1_2" / "70_15_15" / "karsilastirma" /
        "70_15_15_rf_matched_test_predictions_A1_2.csv",
        dtype={"source_id": str},
    )
    manifest = pd.read_csv(ROOT / "manifest_v2_dedup_draft.csv", dtype=str)
    row = preds.iloc[0]
    wav_path = resolve_manifest_path(
        manifest.loc[manifest.source_id.eq(row.source_id), "path"].iloc[0]
    )
    y, sr = librosa.load(wav_path, sr=a12.SAMPLE_RATE)

    feats, fallback, debug = a12.extract_features(y, sr, medians, return_debug=True)
    check("toplam_192_ozellik", feats.shape == (192,), f"shape={feats.shape}")
    check("nan_inf_yok", bool(np.isfinite(feats).all()))

    a0_ref = a0_mod.extract_features(y, sr)
    a0_diff = float(np.max(np.abs(debug["a0"] - a0_ref)))
    check("ilk_184_ozellik_A0_ile_birebir", a0_diff <= 1e-10,
          f"max_diff={a0_diff:.3e}")

    f0_keys = ["f0_mean", "f0_median", "f0_std", "f0_min", "f0_max",
               "f0_range", "f0_slope", "voiced_frame_ratio"]
    f0_ref_dict = a12_mod.extract_f0(y, sr, "orig")
    f0_ref = np.array([f0_ref_dict[k] for k in f0_keys], dtype=float)
    f0_diff = float(np.max(np.abs(debug["f0_raw"] - f0_ref)))
    check("f0_ozellikleri_pYIN_referansiyla_birebir", f0_diff <= 1e-10,
          f"max_diff={f0_diff:.3e}, fallback={fallback}")

    order_diff = float(np.max(np.abs(feats - np.concatenate([a0_ref, debug["f0_final"]]))))
    check("ozellik_sirasi_184_then_8", order_diff <= 1e-12,
          f"max_diff={order_diff:.3e}")
    check("voiced_frame_ratio_impute_edilmedi",
          debug["f0_final"][7] == debug["f0_raw"][7],
          f"ratio={debug['f0_final'][7]:.6f}")

    probs, pred_feats, _ = a12.predict_audio(model, scaler, labels, medians, y, sr)
    saved = np.array([row.prob_angry, row.prob_fear, row.prob_happy, row.prob_sad],
                     dtype=float)
    prob_diff = float(np.max(np.abs(probs - saved)))
    pred_idx = int(np.argmax(probs))
    check("ucdan_uca_tahmin_sinifi_kayitla_ayni", pred_idx == int(row.y_pred),
          f"pred={labels[pred_idx]}, kayit={labels[int(row.y_pred)]}, "
          f"wav={wav_path.name}")
    check("ucdan_uca_olasiliklar_kayitla_uyumlu", prob_diff <= 1e-3,
          f"max_diff={prob_diff:.6f}")
    check("olasilik_toplami_1", abs(float(probs.sum()) - 1.0) <= 1e-6,
          f"sum={probs.sum():.8f}")
    check("tahmin_192_ozellik_kullandi", pred_feats.shape == (192,),
          f"shape={pred_feats.shape}")

    # 3) Bilinen fallback kaydında imputasyonun uygulandığını doğrula.
    f0_detail = pd.read_csv(ROOT / "A1_2" / "f0_istatistikleri.csv", dtype=str)
    fb_row = f0_detail.loc[
        f0_detail.fallback.eq("1") & f0_detail.aug_id.eq("orig"), "source_id"
    ].iloc[0]
    fb_path = resolve_manifest_path(
        manifest.loc[manifest.source_id.eq(fb_row), "path"].iloc[0]
    )
    y_fb, sr_fb = librosa.load(fb_path, sr=a12.SAMPLE_RATE)
    feats_fb, fb_flag, dbg_fb = a12.extract_features(
        y_fb, sr_fb, medians, return_debug=True
    )
    impute_diff = float(np.max(np.abs(dbg_fb["f0_final"][:7] - medians)))
    check("fallback_tespit_edildi", fb_flag, f"source_id={fb_row}")
    check("yedi_f0_sutunu_medyanla_dolduruldu", impute_diff <= 1e-12,
          f"max_diff={impute_diff:.3e}")
    check("fallback_sonrasi_finite", bool(np.isfinite(feats_fb).all()))

    # 4) Eski happy bias kuralı ve model yükleme hatası koruması.
    gui = (ROOT / "7_final_arayuz.py").read_text(encoding="utf-8")
    forbidden = ["HAPPY_MIN", "HAPPY_MARGIN", "happy_idx",
                 'self.classes.index("happy")', "tf.keras", "self.model.predict("]
    check("eski_happy_bias_kaldirildi",
          not any(tok in gui for tok in forbidden),
          f"arama={forbidden}")
    check("rf_adapter_kullaniyor",
          "a12.load_assets()" in gui and "a12.predict_audio(" in gui)
    check("model_yuklenemezse_tahmin_engelli",
          "if not self.ready" in gui and 'showerror("Model hatası"' in gui
          and "state=\"disabled\"" in gui)

    # 5) GUI sınıfını gerçek ses dosyasıyla, pencere açmadan çalıştır.
    gui = load_module("gui_mod", ROOT / "7_final_arayuz.py")
    dialogs = []
    gui.messagebox.showerror = lambda title, msg: dialogs.append((title, msg))

    class WidgetStub:
        def __init__(self):
            self.calls = []
            self.value = None

        def config(self, **kw):
            self.calls.append(kw)

        def configure(self, **kw):
            self.calls.append(kw)

        def update(self):
            self.calls.append({"update": True})

        def title(self, text):
            self.calls.append({"title": text})

        def geometry(self, text):
            self.calls.append({"geometry": text})

        def resizable(self, *args):
            self.calls.append({"resizable": args})

        def __setitem__(self, key, value):
            self.value = value

    app = gui.DuyguAnaliziApp.__new__(gui.DuyguAnaliziApp)
    app.ready = True
    app.model = model
    app.scaler = scaler
    app.classes = labels
    app.medians = medians
    app.file_path = str(wav_path)
    app.root = WidgetStub()
    app.btn_analyze = WidgetStub()
    app.lbl_result = WidgetStub()
    app.lbl_probs = WidgetStub()
    app.status = WidgetStub()
    app.bars = [(WidgetStub(), WidgetStub(), WidgetStub()) for _ in labels]

    dialogs.clear()
    app.analiz_et()
    result_text = app.lbl_result.calls[-1]["text"]
    prob_text = app.lbl_probs.calls[-1]["text"]
    check("gui_ucdan_uca_tahmin", labels[pred_idx] in result_text and "%" in result_text,
          result_text)
    check("gui_olasiliklari_ham_model_ciktisindan_gosteriyor",
          all(f"{c}:" in prob_text for c in labels), prob_text[:120])
    check("gui_status_f0_bilgisi", "F0 imputasyonu:" in app.status.calls[-1]["text"],
          app.status.calls[-1]["text"])
    check("gui_analiz_sonrasi_buton_aktif",
          app.btn_analyze.calls[-1].get("state") == "normal")
    check("gui_basari_parcasinda_hata_penceresi_yok", not dialogs, str(dialogs))

    app_bad = gui.DuyguAnaliziApp.__new__(gui.DuyguAnaliziApp)
    app_bad.ready = False
    app_bad.model = None
    app_bad.scaler = None
    app_bad.classes = labels
    app_bad.medians = medians
    app_bad.btn_analyze = WidgetStub()
    app_bad.status = WidgetStub()
    dialogs.clear()
    app_bad.analiz_et()
    check("gui_model_yokken_tahmin_yapmiyor",
          any(title == "Model hatası" for title, _ in dialogs)
          and app_bad.btn_analyze.calls[-1].get("state") == "disabled",
          str(dialogs[:1]))

    old_model_path = a12.MODEL_PATH
    try:
        a12.MODEL_PATH = ROOT / "yok_boyle_model.joblib"
        try:
            a12.load_assets()
            raised = False
            message = "exception alinmadi"
        except RuntimeError as exc:
            raised = True
            message = str(exc)
        check("model_yukleme_hatasi_acik_mesaj",
              raised and "yüklenemedi" in message and "RF-B" in message,
              message[:140])
    finally:
        a12.MODEL_PATH = old_model_path

    print(f"\nTOPLAM: {len(PASS)}/{len(PASS)} PASS")


if __name__ == "__main__":
    main()
