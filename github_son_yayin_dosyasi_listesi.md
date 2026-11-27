# GitHub Son Yayın Dosyası Listesi (A1.2 RF-B)

Tarih: 2026-11-27\
Dayanak: `github_yayin_oncesi_denetim.md`, `model_agirliklari_lisans_incelemesi.md` ve çalışma ağacının bu tarihteki okunmuş hâli.\
Kapsam: `.git/`, `.venv/`, `data/` ve önbellek dizinleri dışında kalan **351 dosya**. Her dosya aşağıda bir karar ve gerekçeyle listelenmiştir.

> Bu rapor yalnızca karar envanteridir. Hiçbir dosya silinmedi, taşınmadı, değiştirilmedi; `.gitignore`'a dokunulmadı; commit/push yapılmadı. Öneriler uygulanmış değildir.

## Karar ve gerekçe kodları

| Kod | Anlamı |
|---|---|
| **YAYINLA** | Nihai GitHub ağacına dahil edilecek. |
| **KOŞULLU** | Belirtilen koşul yerine gelmeden yayınlanmaz; bölüm 3'te izlenir. |
| **OPSİYONEL** | Lisans/gizlilik açısından güvenli ama A1.2 yayını için gerekli değil; nihai ağaca alınmaz. |
| **YAYINLAMA** | Kesinlikle GitHub'a yüklenmez. |

| Gerekçe | Kısa açıklama |
|---|---|
| G1 | Kaynak kod/kurulum dosyası; proje `LICENSE` dosyası seçildikten sonra yayın. Veri lisansı koda otomatik uygulanmaz. |
| G2 | Toplu metrik/doğrulama/rapor dosyası; kişisel mutlak yol, ham ses, kayıt bazlı kimlik veya ağırlık yok. |
| G3 | Toplu eşik/sayısal ajan (F0 medyanı, label map, split sayıları); eğitimden türetilmiş olsa da kayıt bazlı değil. |
| G4 | Lisans/atıf durumu belgesi; hak durumu ve erişim adresi kayıtlı, ham veri içermez. |
| G5 | Eski deney toplu çıktısı; risk yok ama nihai A1.2 yayını için gerekli değil. |
| G6 | Eğitilmiş model/scaler ağırlığı; açık dağıtım izni yok, `LICENSE_REVIEW`. |
| G7 | Ham ses veya kayıt bazlı `source_id`/SHA-256/yol/özellik satırı; SAVEE/BNC kayıtları da dahil. |
| G8 | Kişisel mutlak yol veya yazar kimliği içerir; temizlenmeden yayın yok. |
| G9 | SAVEE lisansı / BNC rızası doğrulanmadı; türev dosyalarla birlikte yayın yok. |
| G10 | Büyük türev tablo veya serileştirilmiş dosya; kayıt bazlı ya da lisansı belirsiz. |
| G11 | Geçici, önbellek, yedek veya gereksiz dosya. |
| G12 | Ekran görüntüsü piksel içeriği insan gözüyle doğrulanmadı. |
| G13 | Git geçmişinde/HEAD'de hassas dosya; geçmiş değişikliği bu çalışmanın kapsamı dışı. |
| G14 | Denetim raporu kişisel kanıt içerir; olduğu gibi yayın yok. |
| G15 | Provenance/bütünlük eksikliği (manifest hash'i yok); inceleme bekliyor. |
| G16 | Yapılandırma dosyası; mevcut hâliyle yayın, önerilen kurallar ayrıca eklenecek. |
| G17 | README lisans/atıf/yapı düzeltmeleri 2026-11-27'de tamamlandı; onayla `YAYINLA`. |

---

## 1. Yayımlanabilir dosyaların tam listesi

### 1.1 Kök dizin

| Karar | Gerekçe | Dosyalar |
|---|---|---|
| YAYINLA | G16 | `.gitignore` |
| YAYINLA | G17 (tamamlandı) | `README.md` |
| YAYINLA | G1, G4 | `LICENSE`, `LICENSES.md`, `ATIF.md`, `MODEL_LICENSE_REVIEW.md` |
| YAYINLA | G1 | `requirements.txt`, `requirements-gui.txt` |
| YAYINLA | G1 | `1_veri_hazirlama.py`, `1_veri_hazirlama_v2.py`, `2_oznitelik_cikarimi.py`, `2_oznitelik_cikarimi_v2.py`, `2_oznitelik_cikarimi_v2_dedup.py`, `3_modelleri_egit.py`, `3_modelleri_egit_v2.py`, `4_ysa_modeli.py`, `4_ysa_modeli_v2.py`, `5_kfold_dogrulama.py`, `5_kfold_dogrulama_v2.py`, `6_istatistik_testleri.py`, `6_istatistik_testleri_v2.py`, `7_final_arayuz.py`, `arayuz_A1_2_adapter.py`, `test_arayuz_A1_2.py`, `scaler_kaydet.py` |
| YAYINLA | G2 | `A1_2_RF_B_GUI_entegrasyon_raporu.md`, `model_secimi_paketleme_oncesi_inceleme.md`, `model_agirliklari_lisans_incelemesi.md`, `github_son_yayin_dosyasi_listesi.md` |

### 1.2 `A1_2/` — yayınlanacak dosyalar

| Karar | Gerekçe | Dosyalar |
|---|---|---|
| YAYINLA | G1 | `A1_2/70_15_15/4_ysa_A1_2.py`, `A1_2/kfold/5_kfold_A1_2.py`, `A1_2/kfold/dogrulama_sonrasi_A1_2.py`, `A1_2/kfold/istatistiksel_karsilastirma_A1_2.py`, `A1_2/on_kontrol_A1_2.py`, `A1_2/ozellik_cikarimi_A1_2.py`, `A1_2/dis_gercek_ses_testi/00_secim_ve_cakismalar.py` |
| YAYINLA | G2 | `A1_2/70_15_15/son_rapor.md`, `A1_2/70_15_15/dogrulama_kontrolleri.csv`, `A1_2/70_15_15/dogrulama_kontrolleri.md`, `A1_2/70_15_15/filtre_dogrulama.csv`, `A1_2/70_15_15/karsilastirma/A0_vs_A1_2_karisiklik.csv`, `A1_2/70_15_15/karsilastirma/A0_vs_A1_2_sinif_metrikleri.csv`, `A1_2/70_15_15/karsilastirma/A0_vs_A1_2_test_metrikleri.csv`, `A1_2/A0_karsilastirma_A1_2.csv`, `A1_2/augmentation_dagilim.csv`, `A1_2/f0_augmentasyon_kontrolu.csv`, `A1_2/f0_dagilim_raporu.md`, `A1_2/f0_kalite_kontrol.csv`, `A1_2/f0_sinif_kaynak_istatistikleri.csv`, `A1_2/fold_sayilari.csv`, `A1_2/kaynak_sayilari.csv`, `A1_2/sinif_sayilari.csv`, `A1_2/split_sayilari.csv`, `A1_2/leakage_kontrolleri.csv`, `A1_2/on_kontrol_A1_2.csv`, `A1_2/on_kontrol_A1_2.md`, `A1_2/sha256_source_id_kontrolleri.csv`, `A1_2/ozellik_cikarimi_raporu.md`, `A1_2/veri_hazirlama_raporu.md` |
| YAYINLA | G2 | `A1_2/kfold/cv_config.json`, `A1_2/kfold/dogrulama_sonrasi.csv`, `A1_2/kfold/dogrulama_sonrasi.md`, `A1_2/kfold/istatistik_bootstrap_farklari.csv`, `A1_2/kfold/istatistik_genel.csv`, `A1_2/kfold/istatistik_konfig.json`, `A1_2/kfold/istatistik_mcnemar.csv`, `A1_2/kfold/istatistik_sinif_f1.csv`, `A1_2/kfold/karsilastirma_A0_A1_2.csv`, `A1_2/kfold/karsilastirma_A0_A1_2.md`, `A1_2/kfold/kfold_ozet.csv`, `A1_2/kfold/kfold_per_fold.csv`, `A1_2/kfold/oof_cm_MLP.png`, `A1_2/kfold/oof_cm_RF.png`, `A1_2/kfold/oof_karisiklik.csv`, `A1_2/kfold/oof_kaynak_bazli.csv`, `A1_2/kfold/oof_metrikler.csv`, `A1_2/kfold/oof_sinif_f1.csv`, `A1_2/kfold/sizdirma_kontrolleri.csv`, `A1_2/kfold/sizdirma_kontrolleri.md`, `A1_2/kfold/son_rapor_A1_2.md` |
| YAYINLA | G3 | `A1_2/70_15_15/f0_imputasyon_degerleri.csv`, `A1_2/kfold/f0_imputasyon_degerleri.csv`, `A1_2/f0_ozellik_tanimlari.csv` |
| YAYINLA | G3 | `A1_2/70_15_15/ysa/70_15_15_label_map_A1_2.json` |
| YAYINLA | G2 | `A1_2/70_15_15/ysa/70_15_15_accuracy_A1_2.png`, `A1_2/70_15_15/ysa/70_15_15_confusion_matrix_A1_2.csv`, `A1_2/70_15_15/ysa/70_15_15_egitim_val_farki_A1_2.csv`, `A1_2/70_15_15/ysa/70_15_15_karsilastirma_metrikleri_A1_2.csv`, `A1_2/70_15_15/ysa/70_15_15_loss_A1_2.png`, `A1_2/70_15_15/ysa/70_15_15_mlp_cm_A1_2.png`, `A1_2/70_15_15/ysa/70_15_15_rf_matched_cm_A1_2.png`, `A1_2/70_15_15/ysa/70_15_15_sinif_metrikleri_A1_2.csv`, `A1_2/70_15_15/ysa/70_15_15_train_val_test_metrikleri_A1_2.csv` |
| YAYINLA | G4 | `A1_2/dis_gercek_ses_testi/cakisma_ve_dogrulama_raporu.md`, `A1_2/dis_gercek_ses_testi/lisans_durumu.csv` |

### 1.3 Opsiyonel dosyalar (güvenli ama nihai ağaca dahil değil)

Bunlar "yayınlanabilir" niteliktedir; ancak A1.2 RF-B yayını için gerekli olmadığından ve eski deney katmanını temiz tutmak için son ağaca alınmaz. Karar: **OPSİYONEL**, gerekçe **G5**.

| Dizin | Dosyalar |
|---|---|
| `A1_1/` | `4_ysa_A1_1.py`, `5_kfold_A1_1.py`, `A0_karsilastirma_A1_1.csv`, `augmentation_dagilim.csv`, `fold_sayilari.csv`, `karsilastir_A0_A1_1.py`, `kaynak_sayilari.csv`, `kfold/cv_config.json`, `kfold/karsilastirma_A0_A1_1.md`, `kfold/karsilastirma_fear_happy.csv`, `kfold/karsilastirma_fold_bazli.csv`, `kfold/karsilastirma_kaynak_bazli.csv`, `kfold/karsilastirma_oof_metrikler.csv`, `kfold/karsilastirma_sinif_bazli.csv`, `kfold/kfold_ozet.csv`, `kfold/kfold_per_fold.csv`, `kfold/oof_cm_MLP.png`, `kfold/oof_cm_RF.png`, `kfold/oof_kaynak_bazli.csv`, `kfold/oof_metrikler.csv`, `kfold/sizdirma_kontrolleri.csv`, `kfold/sizdirma_kontrolleri.md`, `leakage_kontrolleri.csv`, `on_kontrol_A1_1.csv`, `on_kontrol_A1_1.md`, `on_kontrol_A1_1.py`, `ozellik_cikarimi_A1_1.py`, `ozellik_cikarimi_raporu.md`, `sha256_source_id_kontrolleri.csv`, `sinif_sayilari.csv`, `split_sayilari.csv`, `veri_hazirlama_raporu.md` |
| `A3/` | `5_kfold_A3.py`, `GUI_uygulama_notu.md`, `augmentation_dagilim.csv`, `fold_sayilari.csv`, `istatistik/bootstrap_kaynak_bazli.csv`, `istatistik/bootstrap_sonuclari.csv`, `istatistik/gozlenen_farklar.csv`, `istatistik/istatistik_A0_A3.py`, `istatistik/istatistik_raporu.md`, `istatistik/mcnemar.csv`, `istatistik/roc_auc_durumu.json`, `karsilastir_A0_A3.py`, `kaynak_sayilari.csv`, `kfold/cv_config.json`, `kfold/karsilastirma_A0_A3.md`, `kfold/karsilastirma_fear_happy.csv`, `kfold/karsilastirma_fold_bazli.csv`, `kfold/karsilastirma_kaynak_bazli.csv`, `kfold/karsilastirma_oof_metrikler.csv`, `kfold/karsilastirma_sinif_bazli.csv`, `kfold/kfold_ozet.csv`, `kfold/kfold_per_fold.csv`, `kfold/oof_cm_MLP.png`, `kfold/oof_cm_RF.png`, `kfold/oof_kaynak_bazli.csv`, `kfold/oof_metrikler.csv`, `kfold/sizdirma_kontrolleri.csv`, `kfold/sizdirma_kontrolleri.md`, `leakage_kontrolleri.csv`, `on_kontrol_A3.csv`, `on_kontrol_A3.md`, `on_kontrol_A3.py`, `ozellik_cikarimi_A3.py`, `ozellik_cikarimi_raporu.md`, `pencere_istatistikleri.csv`, `pencere_kurali.json`, `sha256_source_id_kontrolleri.csv`, `sinif_sayilari.csv`, `split_sayilari.csv`, `uretim_log.txt`, `veri_hazirlama_raporu.md` |
| `A1_2/` | `70_15_15/egitim_log.txt`, `uretim_log.txt` |
| `sonuclar_istatistik/` | `mcnemar_sonuclari.txt`, `roc_auc_sonuclari.csv`, `ysa_roc_egrisi.png` |
| `sonuclar_klasik/` | `1-Decision Tree_cm_normalized.png`, `1-Decision Tree_cm_raw.png`, `2-SVM_cm_normalized.png`, `2-SVM_cm_raw.png`, `3-KNN_cm_normalized.png`, `3-KNN_cm_raw.png`, `4-Logistic Regression_cm_normalized.png`, `4-Logistic Regression_cm_raw.png`, `5-Random Forest_cm_normalized.png`, `5-Random Forest_cm_raw.png`, `Ensemble-Gradient Boosting_cm_normalized.png`, `Ensemble-Gradient Boosting_cm_raw.png`, `Ensemble-Voting Classifier_cm_normalized.png`, `Ensemble-Voting Classifier_cm_raw.png`, `genel_sonuc_tablosu.csv`, `k_fold_sonuclari.csv` |
| `sonuclar_v2/` | `artifact_manifest.json`, `augmentation_dagilim.csv`, `filtre_dogrulama.csv`, `filtre_dogrulama.md`, `istatistik/auc_kaynak_bazli.csv`, `istatistik/auc_sonuclari.csv`, `istatistik/group_bootstrap_sonuclari.csv`, `istatistik/istatistik_config.json`, `istatistik/istatistik_raporu.md`, `istatistik/mcnemar_sonuclari.csv`, `istatistik/roc_70_15_15_TESS_YAF.png`, `istatistik/roc_oof_BEKLEMEDE.txt`, `kfold/cv_config.json`, `kfold/kfold_ozet.csv`, `kfold/kfold_per_fold.csv`, `kfold/oof_cm_MLP.png`, `kfold/oof_cm_RF.png`, `kfold/oof_kaynak_bazli.csv`, `kfold/oof_metrikler.csv`, `kfold/sizdirma_kontrolleri.csv`, `kfold/sizdirma_kontrolleri.md`, `klasik/80_20_1-Decision_Tree_cm_normalized.png`, `klasik/80_20_1-Decision_Tree_cm_raw.png`, `klasik/80_20_2-SVM_cm_normalized.png`, `klasik/80_20_2-SVM_cm_raw.png`, `klasik/80_20_3-KNN_cm_normalized.png`, `klasik/80_20_3-KNN_cm_raw.png`, `klasik/80_20_4-Logistic_Regression_cm_normalized.png`, `klasik/80_20_4-Logistic_Regression_cm_raw.png`, `klasik/80_20_5-Random_Forest_80_20_cm_normalized.png`, `klasik/80_20_5-Random_Forest_80_20_cm_raw.png`, `klasik/80_20_Ensemble-Gradient_Boosting_cm_normalized.png`, `klasik/80_20_Ensemble-Gradient_Boosting_cm_raw.png`, `klasik/80_20_Ensemble-Voting_Classifier_cm_normalized.png`, `klasik/80_20_Ensemble-Voting_Classifier_cm_raw.png`, `klasik/80_20_genel_sonuc_tablosu.csv`, `klasik/80_20_kaynak_bazli_sonuclar.csv`, `model_kimliklari.json`, `split_config.json`, `split_dagilim.csv`, `split_dagilim.md`, `ysa/70_15_15_accuracy.png`, `ysa/70_15_15_karsilastirma_metrikleri.csv`, `ysa/70_15_15_label_map.json`, `ysa/70_15_15_loss.png`, `ysa/70_15_15_mlp_cm.png`, `ysa/70_15_15_rf_matched_cm.png` |
| `sonuclar_v2_dedup/` | `artifact_manifest_dedup.json`, `augmentation_dagilim.csv`, `filtre_dogrulama.csv`, `filtre_dogrulama.md`, `filtre_dogrulama_80_20.csv`, `filtre_dogrulama_80_20.md`, `fold_sayilari.csv`, `kaynak_sayilari.csv`, `kfold/cv_config.json`, `kfold/kfold_ozet.csv`, `kfold/kfold_per_fold.csv`, `kfold/oof_cm_MLP.png`, `kfold/oof_cm_RF.png`, `kfold/oof_kaynak_bazli.csv`, `kfold/oof_metrikler.csv`, `kfold/sizdirma_kontrolleri.csv`, `kfold/sizdirma_kontrolleri.md`, `klasik/80_20_1-Decision_Tree_cm_normalized.png`, `klasik/80_20_1-Decision_Tree_cm_raw.png`, `klasik/80_20_2-SVM_cm_normalized.png`, `klasik/80_20_2-SVM_cm_raw.png`, `klasik/80_20_3-KNN_cm_normalized.png`, `klasik/80_20_3-KNN_cm_raw.png`, `klasik/80_20_4-Logistic_Regression_cm_normalized.png`, `klasik/80_20_4-Logistic_Regression_cm_raw.png`, `klasik/80_20_5-Random_Forest_80_20_cm_normalized.png`, `klasik/80_20_5-Random_Forest_80_20_cm_raw.png`, `klasik/80_20_Ensemble-Gradient_Boosting_cm_normalized.png`, `klasik/80_20_Ensemble-Gradient_Boosting_cm_raw.png`, `klasik/80_20_Ensemble-Voting_Classifier_cm_normalized.png`, `klasik/80_20_Ensemble-Voting_Classifier_cm_raw.png`, `klasik/80_20_genel_sonuc_tablosu_dedup.csv`, `klasik/80_20_kaynak_bazli_sonuclar_dedup.csv`, `leakage_kontrolleri.csv`, `model_kimliklari_80_20.json`, `ozellik_cikarimi_raporu.md`, `sha256_source_id_kontrolleri.csv`, `sinif_sayilari.csv`, `split_sayilari.csv`, `veri_hazirlama_raporu.md`, `ysa/70_15_15_accuracy_dedup.png`, `ysa/70_15_15_karsilastirma_metrikleri_dedup.csv`, `ysa/70_15_15_label_map_dedup.json`, `ysa/70_15_15_loss_dedup.png`, `ysa/70_15_15_mlp_cm_dedup.png`, `ysa/70_15_15_rf_matched_cm_dedup.png` |
| `sonuclar_ysa/` | `label_map.json`, `ysa_accuracy.png`, `ysa_confusion_matrix.png`, `ysa_loss.png`, `ysa_sonuclari_tablo.csv` |

---

## 2. Kesinlikle yayımlanmaması gereken dosya ve klasörler

### 2.1 Model, scaler ve serileştirilmiş ağırlıklar — G6 / G9 / G15

| Karar | Gerekçe | Dosyalar |
|---|---|---|
| YAYINLAMA | G6, G9 | `A1_2/70_15_15/modeller/rf_70_15_15_matched_A1_2.joblib`, `A1_2/70_15_15/modeller/scaler_A1_2.joblib`, `A1_2/70_15_15/modeller/mlp_best_70_15_15_A1_2.keras` |
| YAYINLAMA | G6, G9, G15 | `A1_2/70_15_15/modeller/mlp_final_70_15_15_A1_2.keras` — manifestte SHA-256 kaydı yok, proverans eksik. |
| YAYINLAMA | G6, G9 | `sonuclar_v2/modeller/mlp_best_70_15_15.keras`, `sonuclar_v2/modeller/mlp_final_70_15_15.keras`, `sonuclar_v2/modeller/rf_70_15_15_matched.joblib`, `sonuclar_v2/modeller/rf_80_20.joblib`, `sonuclar_v2/modeller/scaler_70_15_15.joblib` |
| YAYINLAMA | G6, G9 | `sonuclar_v2_dedup/modeller/mlp_best_70_15_15_dedup.keras`, `sonuclar_v2_dedup/modeller/mlp_final_70_15_15_dedup.keras`, `sonuclar_v2_dedup/modeller/rf_70_15_15_matched_dedup.joblib`, `sonuclar_v2_dedup/modeller/rf_80_20_dedup.joblib`, `sonuclar_v2_dedup/modeller/scaler_70_15_15_dedup.joblib` |
| YAYINLAMA | G6, G13 | `sonuclar_ysa/best_model.keras`, `sonuclar_ysa/ysa_final_model.keras`, `sonuclar_ysa/scaler.pkl` — ağırlık olmalarının yanında Git geçmişinde/HEAD'de de mevcut. |

**Klasör kararı:** `A1_2/70_15_15/modeller/`, `sonuclar_v2/modeller/` ve `sonuclar_v2_dedup/modeller/` tamamen yayın dışıdır. `sonuclar_ysa/` klasörü ağırlık dosyaları nedeniyle yayın dışı; klasörün yalnızca toplu çıktıları opsiyonel olarak değerlendirilebilir.

### 2.2 Ham sesler ve kayıt bazlı türevler — G7 / G9 / G10

| Karar | Gerekçe | Dosyalar |
|---|---|---|
| YAYINLAMA | G7, G9 | `data/**` — CREMA-D, RAVDESS, TESS, SAVEE ve **BNC** ham WAV'leri (16.503 dosya). İndirilebilirlik dağıtım izni değildir; BNC rızası yok, SAVEE lisansı doğrulanamadı. |
| YAYINLAMA | G7 | `temp_mic.wav` — mikrofon kaydı, kişisel ses içerebilir. |
| YAYINLAMA | G10 | `final_oznitelik_seti.csv`, `final_oznitelik_seti_v2.csv`, `final_oznitelik_seti_v2_dedup.csv`, `A1_2/final_oznitelik_seti_A1_2.csv`, `A1_1/final_oznitelik_seti_A1_1.csv`, `A3/final_oznitelik_seti_A3.csv` |
| YAYINLAMA | G7, G9 | `ham_veri_listesi.csv`, `manifest_v2.csv`, `manifest_v2_dedup_draft.csv`, `manifest_v2_dedup_removed_duplicates.csv`, `manifest_v2_dedup_review_qc.csv`, `manifest_v2_local.csv` |
| YAYINLAMA | G7, G9 | `A1_2/augmentation_plani.csv`, `A1_2/f0_istatistikleri.csv`, `A1_2/dis_gercek_ses_testi/aday_havuzu.csv`, `A1_2/dis_gercek_ses_testi/korunan_dosya_sha256.csv`, `A1_2/dis_gercek_ses_testi/secim_durumu.json`, `A1_2/dis_gercek_ses_testi/test_manifest.csv` |
| YAYINLAMA | G7, G9 | `A1_2/kfold/oof_tahminler.csv`, `A1_2/70_15_15/karsilastirma/70_15_15_mlp_test_predictions_A1_2.csv`, `A1_2/70_15_15/karsilastirma/70_15_15_rf_matched_test_predictions_A1_2.csv` |
| YAYINLAMA | G7 | `A1_1/augmentation_plani.csv`, `A1_1/kfold/oof_tahminler.csv`, `A3/augmentation_plani.csv`, `A3/kfold/oof_tahminler.csv`, `A3/pencere_secimleri.csv` |
| YAYINLAMA | G7 | `sonuclar_v2/kfold/oof_tahminler.csv`, `sonuclar_v2/karsilastirma/70_15_15_mlp_test_predictions.csv`, `sonuclar_v2/karsilastirma/70_15_15_rf_matched_test_predictions.csv`, `sonuclar_v2/istatistik/eslesmis_tahminler_70_15_15.csv`, `sonuclar_v2/istatistik/eslesmis_tahminler_oof.csv`, `sonuclar_v2/klasik/80_20_rf_test_predictions.npz` |
| YAYINLAMA | G7 | `sonuclar_v2_dedup/kfold/oof_tahminler.csv`, `sonuclar_v2_dedup/karsilastirma/70_15_15_mlp_test_predictions_dedup.csv`, `sonuclar_v2_dedup/karsilastirma/70_15_15_rf_matched_test_predictions_dedup.csv`, `sonuclar_v2_dedup/klasik/80_20_rf_test_predictions_dedup.npz` |
| YAYINLAMA | G10 | `A1_2/f0_raw_results.pkl` |

### 2.3 Kişisel bilgi, denetim kanıtı ve gereksiz dosyalar

| Karar | Gerekçe | Dosyalar |
|---|---|---|
| YAYINLAMA | G14 | `github_yayin_oncesi_denetim.md` — olduğu gibi yayınlanmaz; kişisel yol ve yazar kimliği kanıtları içerir. Temizlenmiş bir türev ayrıca hazırlanabilir (bu aşamada yapılmadı). |
| YAYINLAMA | G11 | `yedek_eski_kodlar/3_modelleri_egit.py`, `yedek_eski_kodlar/5_kfold_dogrulama.py`, `yedek_eski_kodlar/6_istatistik_testleri.py` |
| YAYINLAMA | G11 | `.venv/**`, `__pycache__/**`, `.mplconfig/`, `A1_2/70_15_15/.mplconfig/` — ortam, bayt kodu ve matplotlib önbelleği. |

---

## 3. Lisans veya gizlilik incelemesi bekleyen dosyalar

| Karar | Gerekçe | Dosyalar | Koşul |
|---|---|---|---|
| YAYINLA | G17 (tamamlandı) | `README.md`, `LICENSE`, `LICENSES.md`, `ATIF.md`, `MODEL_LICENSE_REVIEW.md` | 2026-11-27 onayıyla yayın kapsamına alındı; dört koşul karşılandı ve §1.1'e taşındı. |
| KOŞULLU | G12 | `screenshots/arayuz.png`, `screenshots/duygu_tahmini.png` | Piksel içeriği insan gözüyle kontrol edilmeden yayınlanmaz; yerel yol/kullanıcı bilgisi olmadığı doğrulanmalı. |
| KOŞULLU | G8 | `A1_2/70_15_15/artifact_manifest_A1_2.json` | İçindeki mutlak yollar göreli yollara çevrilmeden yayınlanmaz; ayrıca model hash'lerini içerdiği için weights yayın kararıyla birlikte değerlendirilir. |
| KOŞULLU | G8 | `A1_2/on_kontrol_log.txt`, `A1_2/kfold/egitim_log.txt`, `A1_2/kfold/istatistik_kontrolleri.csv`, `A1_2/kfold/istatistik_kontrolleri.md`, `A1_2/kfold/istatistiksel_karsilastirma_raporu.md`, `A3/kfold/egitim_log.txt` | Kişisel mutlak yollar temizlenmeden yayınlanmaz. A1.2 kfold dosyaları temizlendiğinde G2 kapsamına geçer. |
| KOŞULLU | G6, G9 | `A1_2/70_15_15/modeller/**`, `sonuclar_v2/modeller/**`, `sonuclar_v2_dedup/modeller/**`, `sonuclar_ysa/*.keras`, `sonuclar_ysa/scaler.pkl` | Yalnızca hak sahiplerinden açık yazılı model dağıtım izni alınıp hukuki inceleme tamamlanırsa yeniden değerlendirilir. Aksi hâlde kalıcı `LICENSE_REVIEW`. |
| KOŞULLU | G9 | SAVEE ve BNC kaynaklı tüm ham dosyalar, kayıt bazlı türevler ve bunlarla eğitilmiş ağırlıklar | SAVEE için yetkili lisans/rıza, BNC için tek konuşmacının yayın onayı belgelenmeden yayın yok. Üçüncü taraf repolar hak kaynağı sayılmaz. |
| KOŞULLU | G15 | `A1_2/70_15_15/modeller/mlp_final_70_15_15_A1_2.keras` | Manifestte hash eksikliği giderilmeden ve lisans/onay verilmeden yayın yok. |
| KOŞULLU | G13 | `sonuclar_ysa/best_model.keras`, `sonuclar_ysa/ysa_final_model.keras`, `sonuclar_ysa/scaler.pkl`, `sonuclar_ysa/label_map.json` | HEAD ve Git geçmişinde mevcut. `.gitignore` eklemek geçmişten kaldırmaz; ya history rewrite ya da belgelenmiş onay gerekir (kapsam dışı). |
| YAYINLA | G1 (tamamlandı) | Tüm Python kaynak dosyaları | Proje geneli `LICENSE` eklendi; kod lisansı tanımlı. |

**Ek gizlilik notu:** Git yazar metadata'sında kişisel e-posta adresi 4 commit'in tamamında yer alır. Bu, dosya listesinden bağımsız bir depo metadata konusudur; temizleme history rewrite gerektirir ve bu çalışmada yapılmadı.

---

## 4. `.gitignore` dosyasına eklenmesi gereken kurallar (öneri — uygulanmadı)

Mevcut `.gitignore` yalnızca `data/`, `.venv/`, `*.log`, `temp_mic.wav`, `final_oznitelik_seti.csv`, `final_oznitelik_seti_v2.csv`, `manifest_v2_local.csv`, `sonuclar_v2/modeller/`, `/scaler.pkl` ve `yedek_eski_kodlar/` gibi kalıpları kapsıyor. Aşağıdaki ekler önerilir:

```gitignore
# Eğitilmiş ağırlıklar ve serileştirilmiş dosyalar
*.keras
*.joblib
*.pkl
*.npz
A1_2/70_15_15/modeller/
sonuclar_v2_dedup/modeller/
sonuclar_ysa/best_model.keras
sonuclar_ysa/ysa_final_model.keras
sonuclar_ysa/scaler.pkl
A1_2/f0_raw_results.pkl

# Kayıt bazlı türevler, büyük özellik tabloları ve manifestler
final_oznitelik_seti_v2_dedup.csv
A1_2/final_oznitelik_seti_A1_2.csv
A1_1/final_oznitelik_seti_A1_1.csv
A3/final_oznitelik_seti_A3.csv
A1_2/augmentation_plani.csv
A1_2/f0_istatistikleri.csv
manifest_v2*.csv
A1_2/kfold/oof_tahminler.csv
A1_2/70_15_15/karsilastirma/*_test_predictions_A1_2.csv
A1_2/dis_gercek_ses_testi/aday_havuzu.csv
A1_2/dis_gercek_ses_testi/korunan_dosya_sha256.csv
A1_2/dis_gercek_ses_testi/secim_durumu.json
A1_2/dis_gercek_ses_testi/test_manifest.csv
A1_1/augmentation_plani.csv
A1_1/kfold/oof_tahminler.csv
A3/augmentation_plani.csv
A3/kfold/oof_tahminler.csv
A3/pencere_secimleri.csv
sonuclar_v2/kfold/oof_tahminler.csv
sonuclar_v2/karsilastirma/*_test_predictions.csv
sonuclar_v2/istatistik/eslesmis_tahminler_*.csv
sonuclar_v2/klasik/*_rf_test_predictions.npz
sonuclar_v2_dedup/kfold/oof_tahminler.csv
sonuclar_v2_dedup/karsilastirma/*_test_predictions_dedup.csv
sonuclar_v2_dedup/klasik/*_rf_test_predictions_dedup.npz

# Kişisel yol içeren log/rapor ve manifest dosyaları
*_log.txt
A1_2/70_15_15/artifact_manifest_A1_2.json
A1_2/kfold/istatistik_kontrolleri.csv
A1_2/kfold/istatistik_kontrolleri.md
A1_2/kfold/istatistiksel_karsilastirma_raporu.md

# Önbellek ve üretilen yapılandırma
.mplconfig/
*.mplconfig/
A1_2/70_15_15/.mplconfig/

# Eski deney katmanları (opsiyonel; yayına dahil edilmeyecek)
A1_1/
A3/
sonuclar_klasik/
sonuclar_istatistik/
sonuclar_v2/
sonuclar_v2_dedup/
sonuclar_ysa/
```

Notlar:
- `*.keras`/`*.joblib`/`*.pkl`/`*.npz` kuralları yalnızca ağırlıkların yayınlanmaması kararı korunduğu sürece eklenmelidir. İleride açık izin alınırsa istisna kalıpları (`!...`) yeniden değerlendirilir.
- `.gitignore`, Git geçmişinde zaten izlenmiş olan `sonuclar_ysa/*.keras` ve `scaler.pkl` dosyalarını **kaldırmaz**; bu dosyalar hâlâ HEAD içindedir.
- Kişisel yol içeren dosyalar `.gitignore` ile çözülmez; içeriklerinin düzeltilmesi gerekir. Bu çalışmada hiçbir içerik düzeltilmedi.

---

## 5. GitHub deposunda bulunması gereken nihai klasör yapısı

Yalnızca **YAYINLA** kararı verilen dosyalar yer alır. `README.md` ve dört lisans/atıf/model belgesi 2026-11-27 onayıyla kapsama alındı; bölüm 6'daki düzeltmeler tamamlandı. Diğer `KOŞULLU` dosyalar koşullar tamamlanmadan yayınlanmaz.

```text
ses-duygu-tanima/
├── .gitignore
├── LICENSE                              # YAYINLA: kaynak kod lisansı
├── LICENSES.md / ATIF.md                # YAYINLA: veri lisansları ve atıflar
├── MODEL_LICENSE_REVIEW.md              # YAYINLA: ağırlıkların yayın kararı
├── README.md                            # YAYINLA: güncellendi
├── requirements.txt
├── requirements-gui.txt
├── github_son_yayin_dosyasi_listesi.md
├── model_agirliklari_lisans_incelemesi.md
├── model_secimi_paketleme_oncesi_inceleme.md
├── A1_2_RF_B_GUI_entegrasyon_raporu.md
├── 1_veri_hazirlama.py
├── 1_veri_hazirlama_v2.py
├── 2_oznitelik_cikarimi.py
├── 2_oznitelik_cikarimi_v2.py
├── 2_oznitelik_cikarimi_v2_dedup.py
├── 3_modelleri_egit.py
├── 3_modelleri_egit_v2.py
├── 4_ysa_modeli.py
├── 4_ysa_modeli_v2.py
├── 5_kfold_dogrulama.py
├── 5_kfold_dogrulama_v2.py
├── 6_istatistik_testleri.py
├── 6_istatistik_testleri_v2.py
├── 7_final_arayuz.py
├── arayuz_A1_2_adapter.py
├── test_arayuz_A1_2.py
├── scaler_kaydet.py
├── screenshots/                         # KOŞULLU: görsel kontrol sonrası
│   ├── arayuz.png
│   └── duygu_tahmini.png
└── A1_2/
    ├── A0_karsilastirma_A1_2.csv
    ├── augmentation_dagilim.csv
    ├── f0_augmentasyon_kontrolu.csv
    ├── f0_dagilim_raporu.md
    ├── f0_kalite_kontrol.csv
    ├── f0_ozellik_tanimlari.csv
    ├── f0_sinif_kaynak_istatistikleri.csv
    ├── fold_sayilari.csv
    ├── kaynak_sayilari.csv
    ├── leakage_kontrolleri.csv
    ├── on_kontrol_A1_2.csv
    ├── on_kontrol_A1_2.md
    ├── on_kontrol_A1_2.py
    ├── ozellik_cikarimi_A1_2.py
    ├── ozellik_cikarimi_raporu.md
    ├── sha256_source_id_kontrolleri.csv
    ├── sinif_sayilari.csv
    ├── split_sayilari.csv
    ├── veri_hazirlama_raporu.md
    ├── kfold/
    │   ├── 5_kfold_A1_2.py
    │   ├── cv_config.json
    │   ├── dogrulama_sonrasi.csv
    │   ├── dogrulama_sonrasi.md
    │   ├── dogrulama_sonrasi_A1_2.py
    │   ├── f0_imputasyon_degerleri.csv
    │   ├── istatistik_bootstrap_farklari.csv
    │   ├── istatistik_genel.csv
    │   ├── istatistik_konfig.json
    │   ├── istatistik_mcnemar.csv
    │   ├── istatistik_sinif_f1.csv
    │   ├── istatistiksel_karsilastirma_A1_2.py
    │   ├── karsilastirma_A0_A1_2.csv
    │   ├── karsilastirma_A0_A1_2.md
    │   ├── kfold_ozet.csv
    │   ├── kfold_per_fold.csv
    │   ├── oof_cm_MLP.png
    │   ├── oof_cm_RF.png
    │   ├── oof_karisiklik.csv
    │   ├── oof_kaynak_bazli.csv
    │   ├── oof_metrikler.csv
    │   ├── oof_sinif_f1.csv
    │   ├── sizdirma_kontrolleri.csv
    │   ├── sizdirma_kontrolleri.md
    │   └── son_rapor_A1_2.md
    ├── 70_15_15/
    │   ├── 4_ysa_A1_2.py
    │   ├── son_rapor.md
    │   ├── dogrulama_kontrolleri.csv
    │   ├── dogrulama_kontrolleri.md
    │   ├── f0_imputasyon_degerleri.csv
    │   ├── filtre_dogrulama.csv
    │   ├── karsilastirma/
    │   │   ├── A0_vs_A1_2_karisiklik.csv
    │   │   ├── A0_vs_A1_2_sinif_metrikleri.csv
    │   │   └── A0_vs_A1_2_test_metrikleri.csv
    │   ├── modeller/                    # YOK: ağırlıklar yerel kalır
    │   └── ysa/                         # yalnızca toplu çıktılar
    │       ├── 70_15_15_label_map_A1_2.json
    │       ├── 70_15_15_accuracy_A1_2.png
    │       ├── 70_15_15_confusion_matrix_A1_2.csv
    │       ├── 70_15_15_egitim_val_farki_A1_2.csv
    │       ├── 70_15_15_karsilastirma_metrikleri_A1_2.csv
    │       ├── 70_15_15_loss_A1_2.png
    │       ├── 70_15_15_mlp_cm_A1_2.png
    │       ├── 70_15_15_rf_matched_cm_A1_2.png
    │       ├── 70_15_15_sinif_metrikleri_A1_2.csv
    │       └── 70_15_15_train_val_test_metrikleri_A1_2.csv
    └── dis_gercek_ses_testi/
        ├── 00_secim_ve_cakismalar.py
        ├── cakisma_ve_dogrulama_raporu.md
        └── lisans_durumu.csv
```

Bu ağaçta **yoktur**: `data/`, `.venv/`, tüm `.keras`/`.joblib`/`.pkl`/`.npz`, büyük özellik CSV'leri, kayıt bazlı tahminler, `A1_1/`, `A3/`, `sonuclar_*`, `yedek_eski_kodlar/`, `github_yayin_oncesi_denetim.md`.

**Kullanıcı etkisi:** GUI kodu depoda olmasına rağmen gerekli 4 A1.2 artefaktı depada olmadığından arayüz kurulum sonrası doğrudan çalıştırılamaz. README bunu açıkça belirtmelidir.

---

## 6. README ve lisans/atıf belgelerindeki eksiklikler (kayıt)

> **Güncelleme 2026-11-27:** 1–9 ve 12 numaralı maddeler kapatıldı; 11 numaralı `.gitignore` önerileri uygulandı. 10 numaralı `screenshots/` kararı `KOŞULLU G12` olarak korunuyor. Aşağıdaki tablo geçmiş denetim kaydı olarak saklanmıştır.

| # | Eksik | Yapılması gereken (denetim anındaki öneri) |
|---|---|---|
| 1 | Proje geneli `LICENSE` dosyası yok. | Bir kaynak kod lisansı seçip kök dizine `LICENSE` ekle; README'de kod lisansının veri/model lisanslarından ayrı olduğunu belirt. |
| 2 | Ayrı bir `LICENSES.md` / `ATIF.md` yok. | CREMA-D (ODbL 1.0 + DbCL 1.0), RAVDESS (CC BY-NC-SA 4.0), TESS (CC BY-NC 4.0) için tam yaratıcı/atıf bilgisi, lisans URI'si, erişim tarihi ve gerekiyorsa değişiklik bildirimi ekle. SAVEE için "doğrulanamadı", BNC için "yayın rızası belgelenmedi" notunu koru. |
| 3 | Model lisansı/atıf belgesi yok. | `MODEL_LICENSE_REVIEW.md` ekle; RF-B, scaler ve MLP'nin depaya dahil edilmediğini, `LICENSE_REVIEW` olduğunu, açık yazılı izin + hukuki inceleme olmadan yayınlanmayacağını yaz. |
| 4 | README "Project Structure" bölümü `modeller/` altında `.joblib`/`.keras` dosyalarını ve `A1_2/final_oznitelik_seti_A1_2.csv` dosyasını gösteriyor. | Bu girdileri kaldır; yerine "artifacts are local, not distributed" notu ekle. Aksi hâlde README, bu raporun kararıyla çelişir. |
| 5 | README, GUI'nin 4 yerel artefakt gerektirdiğini söylüyor ancak bunların neden depoda olmadığını açıklamıyor. | Eksik artefakt nedeniyle GUI'nin depodan çalıştırılamayacağını, kullanıcıların kendi lisanslı verileriyle yeniden eğitmesi gerektiğini belirt. |
| 6 | SAVEE/BNC uyarısı yalnızca "limitation" maddelerinde. | Lisans/atıf bölümünde ve model arasında da tekrarla: bu kaynaklarla eğitilmiş ağırlıklar da yayın dışıdır. |
| 7 | İndirilebilirlik ≠ dağıtım izni ilkesi README'de açık yok. | Kısa bir uyarı ekle: veri setine erişim, yeniden dağıtım veya model ağırlığı izni vermez. |
| 8 | Yayın kararının kendisi README'de yok. | Bu dosya listesine ve lisans incelemesine referans veren bir "Release decisions" bölümü ekle. |
| 9 | Eski MLP artefaktlarının Git geçmişinde olduğu README'de belirtilmiyor. | Geçmiş riskini ve history rewrite gerektiğini (yapılmadığını) belge et. |
| 10 | Ekran görüntüsü onayı yok. | İnsan kontrolü tamamlanana kadar `screenshots/`'un yayın durumunu KOŞULLU tut. |
| 11 | `.gitignore` önerileri README'ye yansımadı. | Bölüm 4'teki kuralların uygulanması planlandığında README "Development" notuna ekle. |
| 12 | Kod lisansı olmadığı hâlde Python dosyaları "yayınlanabilir" sayılıyor. | Lisans eklenene kadar hiçbir Python dosyasını stage etme; bu rapordaki YAYINLA kararı içerik güvenliği anlamındadır. |

---

## 7. Git geçmişinde zaten bulunan hassas dosyalar

Aşağıdaki dosyalar yalnızca çalışma ağacında değil, **HEAD (`f9c6951`) ve geçmiş commit'lerdedir**:

- `sonuclar_ysa/best_model.keras`
- `sonuclar_ysa/ysa_final_model.keras`
- `sonuclar_ysa/scaler.pkl`
- `sonuclar_ysa/label_map.json`

Ek olarak:

- Git yazar kimliğinde kişisel e-posta adresi 4 commit'in tamamında yer alır.
- Ham WAV dosyaları geçmişte izlenmemiştir; `data/` yalnızca kaynak kod metinlerinde geçer.
- `.gitignore` eklemek bu dosyaları geçmişten çıkarmaz. Kaldırma işlemi history rewrite gerektirir; bu çalışmada yapılmadı ve yetki kapsamı dışındadır.
- Bu dosyaların varlığı, A1.2 RF-B anlatılırken eski MLP artefaktlarının repoda görünmesi nedeniyle hem karışıklık hem lisans riski yaratır.

---

## 8. Kapsam ve doğrulama

- İncelenen 351 dosyanın tamamı yukarıdaki karar gruplarından birinde yer alır.
- SAVEE lisansı `DOGRULANAMADI`; BNC için yayın rızası belgelenmedi. Bu belirsizlikler korundu.
- Model ağırlıkları, scaler dosyaları, ham sesler ve kayıt bazlı türev veriler hiçbir yerde `YAYINLA` olarak işaretlenmedi.
- `.gitignore` değiştirilmedi; dosya silinmedi/taşınmadı; commit/push yapılmadı.
- Bu raporun kendisi dışında yeni dosya oluşturulmadı.
- Kararlar, lisans belirsizlikleri hakkında hukuki kesinlik iddiası içermez; gerekirse avukat incelemesi gerekir.

---

## 9. Son onay güncellemesi (2026-11-27)

| Karar | Dosyalar |
|---|---|
| YAYINLA (yeni onay) | `README.md`, `LICENSE`, `LICENSES.md`, `ATIF.md`, `MODEL_LICENSE_REVIEW.md` |
| YAYINLAMA (korundu) | `github_lisans_ve_readme_raporu.md`, `github_son_yayin_temizlik_raporu.md` ve denetim/karar artefaktları |
| YAYINLAMA / `LICENSE_REVIEW` (korundu) | Tüm model, scaler, ham ses ve kayıt bazlı türev dosyalar |
| Silme kararı (korundu) | 26 yayın dışı dosya; Git ağacına yansıması için ayrıca stage + commit gerekir |

Beş dosya için G17/G1 koşulları karşılandı; içeriklerinde kişisel mutlak yol veya e-posta yoktur. README'nin 95 yolu diskteki dosyalarla eşleşir. Kesin stage listesi **88 giriş** olarak güncellendi: 80 önceden onaylanmış yeni dosya + 4 lisans/atıf/model belgesi + `README.md` + 3 önceden onaylanmış değiştirilmiş dosya. Hiçbir stage/commit işlemi yapılmadı.
