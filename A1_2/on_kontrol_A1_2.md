# A1.2 — Hazırlık sonrası kapsamlı ön kontroller

- Kontrol sayısı: **64**, PASS: **64**, FAIL: **0**
- Girdiler salt okunur; eğitim/CV/tahmin yapılmadı.

| kategori | kontrol | beklenen | gerçek | durum |
|---|---|---|---|---|
| sema | satir_sayisi | 23268 | 23268 | PASS |
| sema | kolon_sayisi | 202 | 202 | PASS |
| sema | metadata_kolonlari | 10 | 10 | PASS |
| sema | ilk_184_kolon | 184 | 184 | PASS |
| sema | f0_kolonlari_184_191 | 8 | 8 | PASS |
| A0_eslesme | meta_source_id_birebir | True | True | PASS |
| A0_eslesme | meta_group_id_birebir | True | True | PASS |
| A0_eslesme | meta_source_birebir | True | True | PASS |
| A0_eslesme | meta_label_birebir | True | True | PASS |
| A0_eslesme | meta_aug_id_birebir | True | True | PASS |
| A0_eslesme | meta_split_80_20_birebir | True | True | PASS |
| A0_eslesme | meta_split_70_15_15_birebir | True | True | PASS |
| A0_eslesme | meta_fold_birebir | True | True | PASS |
| A0_eslesme | meta_labels_birebir | True | True | PASS |
| A0_eslesme | meta_sha256_birebir | True | True | PASS |
| A0_eslesme | ilk_184_max_abs_diff | <=1e-10 | 5.68e-14 | PASS |
| A0_eslesme | source_id_kumesi | 7780 | 7780 | PASS |
| veri | REVIEW_LABEL_yok | 0 | 0 | PASS |
| veri | REVIEW_QC_yok | 0 | 0 | PASS |
| veri | fold_manifest_birebir | True | True | PASS |
| veri | split_80_20_manifest_birebir_A0_kurali | True | True | PASS |
| veri | split_70_15_15_manifest_birebir_A0_kurali | True | True | PASS |
| veri | tekil_source_id | 7780 | 7780 | PASS |
| veri | tekil_sha256 | 7780 | 7780 | PASS |
| veri | source_id_aug_tekrari | 0 | 0 | PASS |
| augmentation | varyant_kumeleri_manifest | 0 | 0 | PASS |
| augmentation | satir_sayisi | 23268 | 23268 | PASS |
| leakage | split_80_20_val_test_aug_satiri | 0 | 0 | PASS |
| leakage | split_70_15_15_val_test_aug_satiri | 0 | 0 | PASS |
| leakage | source_id_tek_fold | 0 | 0 | PASS |
| leakage | group_tek_kaynak | 0 | 0 | PASS |
| leakage | fold_1_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_2_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_3_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_4_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_5_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_6_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_7_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_8_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_9_group_source_kesisim | 0 | 0 | PASS |
| leakage | fold_10_group_source_kesisim | 0 | 0 | PASS |
| leakage | leakage_csv_tum_PASS | 12 | 12 | PASS |
| f0 | 8_kolon_finite | True | True | PASS |
| f0 | f0_araligi | 0 veya [60,400] | min=0.00, max=394.67 | PASS |
| f0 | voiced_oran_aralik | [0,1] | [0.0000,0.9815] | PASS |
| f0 | range=max-min | True | True | PASS |
| f0 | min<=medyan<=max | True | True | PASS |
| f0 | kare_sayisi_108 | hicbiri 108 degil | 0 hatali satir | PASS |
| f0 | voiced_ratio_payda_tutarli | True | True | PASS |
| f0 | fallback_satirlarda_7_sifat_sifir | 0 | 0 | PASS |
| f0 | normal_satirlarda_yeterli_voiced | >=3 | 0 | PASS |
| f0 | CSV_F0_vs_detay_birebir | True | True | PASS |
| hata_raporu | pyin_cagrisi_sayisi | 23268 | 23268 | PASS |
| hata_raporu | pyin_hata_sayisi_tutarli | 0 | 0 | PASS |
| hata_raporu | fallback_sayisi_tutarli | 1606 | 1606 | PASS |
| hata_raporu | wav_okuma_hatasi | 0 | 0 | PASS |
| hata_raporu | sha256_uyusmazligi | 0 | 0 | PASS |
| raporlar | f0_sinif_kaynak_istatistikleri.csv_var | True | True | PASS |
| raporlar | f0_augmentasyon_kontrolu.csv_var | True | True | PASS |
| raporlar | f0_ozellik_tanimlari.csv_var | True | True | PASS |
| ses_tutarli | yeniden_f0_ornek_sayisi | >=30 | 36 | PASS |
| ses_tutarli | yeniden_f0_hata | 0 | 0 | PASS |
| ses_tutarli | yeniden_f0_max_abs_diff | <=1e-6 | 5.68e-14 | PASS |
| ses_tutarli | ses_zinciri | load->augment->normalize->2.5s->pyin | ayni | PASS |

## Özet

- F0 yeniden hesaplama: **36** örnek, maksimum mutlak fark **5.68e-14** Hz.
- pYIN hatalı kayıt: **0**, fallback uygulanan: **1606** (ayrıntı: `f0_kalite_kontrol.csv`).

Eğitim veya CV başlatılmadı; onay bekleniyor.
