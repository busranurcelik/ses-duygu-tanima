# A1.2 70/15/15 — veri sızıntısı ve doğrulama kontrolleri

- Kontrol: **42**, PASS: **42**, FAIL: **0**

| kapsam | kontrol | beklenen | gerçek | durum | detay |
|---|---|---|---|---|---|
| global | 202 kolon / 192 ozellik | 202 | 202 | PASS | metadata=10, ozellik=192 |
| global | ilk_184_A0_ozelligi | 184 | 184 | PASS |  |
| global | group_id_kaynak_on_eki | 0 | 0 | PASS |  |
| global | bir_group_tek_kaynak | 0 | 0 | PASS |  |
| global | satir_sayisi | 23268 | 23268 | PASS |  |
| global | A0_satir_sayisi | 23268 | 23268 | PASS |  |
| global | A0_A1_2_satir_eslesmesi | 23268 | 23268 | PASS |  |
| global | metadata_A0_ile_birebir | 0 | 0 | PASS | source_id+aug_id ile eslesildi |
| global | ilk_184_ozellik_max_diff | <=1e-10 | 5.684e-14 | PASS |  |
| global | f0_detay_satir_sayisi | 23268 | 23268 | PASS |  |
| global | f0_detay_tekrar | 0 | 0 | PASS |  |
| global | fallback_bayragi_var | 0 | 0 | PASS |  |
| global | fallback_sayisi | 1606 | 1606 | PASS |  |
| global | fallback_satir_sentinel_sifir | 0 | 0 | PASS |  |
| global | ozellik_matrisi_finite | True | True | PASS |  |
| global | EXCLUDED_filtrelendi | 0 | 3664 | PASS | 3664 satir atildi |
| global | train_satir | 17536 | 17536 | PASS |  |
| global | val_satir | 1268 | 1268 | PASS |  |
| global | test_satir | 800 | 800 | PASS |  |
| global | val_test_yalnizca_orig | 0 | 0 | PASS |  |
| global | train-val_grup_kesisimi | 0 | 0 | PASS |  |
| global | train-val_source_kesisimi | 0 | 0 | PASS |  |
| global | train-test_grup_kesisimi | 0 | 0 | PASS |  |
| global | train-test_source_kesisimi | 0 | 0 | PASS |  |
| global | val-test_grup_kesisimi | 0 | 0 | PASS |  |
| global | val-test_source_kesisimi | 0 | 0 | PASS |  |
| global | source_tek_split | 0 | 0 | PASS |  |
| global | test_kaynagi | TESS | TESS | PASS |  |
| global | test_grup | TESS_YAF | TESS_YAF | PASS | Test yalnizca TESS_YAF konusmacisidir |
| global | train_aug_varyantlari | True | True | PASS |  |
| global | imputasyon_medyan_finite | True | True | PASS | n_train_fallbacksiz=16342 |
| global | voiced_frame_ratio_degismedi | True | True | PASS | 191 impute edilmedi |
| global | imputasyon_sonrasi_finite | True | True | PASS |  |
| global | scaler_train_only | 17536 | 17536 | PASS |  |
| global | sinif_sirasi | ['angry', 'fear', 'happy', 'sad'] | ['angry', 'fear', 'happy', 'sad'] | PASS |  |
| global | test_satir_sayisi_karsilastirma | 800 | 800 | PASS |  |
| A0_karsilastirma | RF-B_test_kumesi | 800 | 800 | PASS |  |
| A0_karsilastirma | RF-B_y_true_ayni | True | True | PASS |  |
| A0_karsilastirma | RF-B_A0_metrik_reproduksiyon | <=1e-6 | 0.00e+00 | PASS |  |
| A0_karsilastirma | MLP_test_kumesi | 800 | 800 | PASS |  |
| A0_karsilastirma | MLP_y_true_ayni | True | True | PASS |  |
| A0_karsilastirma | MLP_A0_metrik_reproduksiyon | <=1e-6 | 0.00e+00 | PASS |  |

- A0 ile aynı `split_70_15_15` atamaları, aynı train/val/test kayıtları kullanıldı.
- İlk 184 özellik A0 CSV'siyle birebir doğrulandı.
- F0 medyanları yalnızca train'in fallback'siz satırlarından öğrenildi.
- `voiced_frame_ratio` (191) impute edilmedi; gerçek sıfırlar korundu.
- Scaler yalnızca 70/15/15 train'e fit edildi.
- 10-fold OOF, McNemar, ROC ve mevcut A1.2 raporu değiştirilmedi.
