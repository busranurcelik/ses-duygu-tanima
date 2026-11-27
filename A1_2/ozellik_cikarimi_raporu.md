# Özellik Çıkarım Raporu — A1.2

İlk 184 özellik A0 CSV'sinden birebir kopyalandı; 8 yeni F0 özelliği her varyantın gerçek sinyalinden A0 zinciriyle çıkarıldı. Scaler ve model değiştirilmedi.

## Çıktı

- `final_oznitelik_seti_A1_2.csv`: **23268** satır × **202** kolon (10 metadata + 184 A0 + 8 F0)
- Boyut: **87347989** byte

## 8 F0 özelliği

| kolon | ozellik | birim | tanim | fallback |
|---|---|---|---|---|
| 184 | f0_mean | Hz | Gecerli voiced karelerin F0 ortalamasi | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 185 | f0_median | Hz | Gecerli voiced karelerin F0 medyani | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 186 | f0_std | Hz (ddof=0) | Gecerli voiced karelerin F0 std (ddof=0) | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 187 | f0_min | Hz | Gecerli voiced karelerin minimum F0 | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 188 | f0_max | Hz | Gecerli voiced karelerin maksimum F0 | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 189 | f0_range | Hz | maksimum - minimum F0 | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 190 | f0_slope | Hz/s (gercek zaman) | np.polyfit(zaman[voiced], F0[voiced], 1) egitimi; zaman=i*512/22050 s | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |
| 191 | voiced_frame_ratio | oran | n_voiced_valid / 108 (2,5 s pencerenin toplam pYIN kare sayisi) | Yetersiz voiced (<3) veya hata: ilk 7 alan 0.0; voiced_ratio gercek paydadan uretilir |

## pYIN / fallback

| kontrol | deger |
|---|---|
| pyin_cagrisi | 23268 |
| wav_okuma_hatasi | 0 |
| sha256_uyusmazligi | 0 |
| eksik_varyant | 0 |
| pyin_hatali_kayit | 0 |
| fallback_uygulanan_kayit | 1606 |
| voiced_kare_sifir_kayit | 1606 |
| voiced_flag_var_f0_gecersiz_kare | 0 |
| f0_sonlu_olmayan_kare | 1471990 |
| beklenen_kare_sayisi | 108 |
| kare_sayisi_108_olmayan_kayit | 0 |
| fallback_sebebi:no_valid_voiced | 1606 |
| kaynak:BNC:fallback | 0 |
| kaynak:CREMA:fallback | 1586 |
| kaynak:RAVDESS:fallback | 20 |
| kaynak:SAVEE:fallback | 0 |
| kaynak:TESS:fallback | 0 |
| aug:noise:fallback | 540 |
| aug:noise2:fallback | 0 |
| aug:orig:fallback | 537 |
| aug:pitch_down:fallback | 0 |
| aug:pitch_up:fallback | 529 |
| aug:stretch_0.9:fallback | 0 |
| aug:stretch_1.1:fallback | 0 |

## A0 eşleşmesi

| kontrol | sonuc |
|---|---|
| meta_source_id_birebir | True |
| meta_group_id_birebir | True |
| meta_source_birebir | True |
| meta_label_birebir | True |
| meta_aug_id_birebir | True |
| meta_split_80_20_birebir | True |
| meta_split_70_15_15_birebir | True |
| meta_fold_birebir | True |
| meta_labels_birebir | True |
| meta_sha256_birebir | True |
| ilk_184_max_abs_diff | 0.0 |
| ilk_184_allclose_atol_1e-10 | True |
| f0_8_kolon_finite | True |
| f0_8_kolon_sabit_olmayan | 8 |

## Leakage

| kapsam | train_rows | eval_rows | excluded_rows | eval_aug_rows | excluded_orig_rows | group_leak | source_leak | multi_role_source | sonuc |
|---|---|---|---|---|---|---|---|---|---|
| 80_20 | 18513 | 1609 | 3146 | 0 | 0 | 0 | 0 | 0 | PASS |
| 70_15_15 | 17536 | 2068 | 3664 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_1 | 21156 | 704 | 1408 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_2 | 21532 | 640 | 1096 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_3 | 21234 | 678 | 1356 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_4 | 21436 | 632 | 1200 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_5 | 21103 | 759 | 1406 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_6 | 20271 | 999 | 1998 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_7 | 21393 | 625 | 1250 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_8 | 20939 | 643 | 1686 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_9 | 21496 | 628 | 1144 | 0 | 0 | 0 | 0 | 0 | PASS |
| fold_10 | 18852 | 1472 | 2944 | 0 | 0 | 0 | 0 | 0 | PASS |

## Satır/kaynak/sınıf/fold

| source | feature_rows |
|---|---|
| BNC | 700 |
| CREMA | 15019 |
| RAVDESS | 2149 |
| SAVEE | 600 |
| TESS | 4800 |

| label | feature_rows |
|---|---|
| angry | 5823 |
| fear | 5817 |
| happy | 5823 |
| sad | 5805 |

| fold | feature_rows |
|---|---|
| 1 | 2112 |
| 10 | 4416 |
| 2 | 1736 |
| 3 | 2034 |
| 4 | 1832 |
| 5 | 2165 |
| 6 | 2997 |
| 7 | 1875 |
| 8 | 2329 |
| 9 | 1772 |

| source | aug_id | n_rows |
|---|---|---|
| BNC | noise | 100 |
| BNC | noise2 | 100 |
| BNC | orig | 100 |
| BNC | pitch_down | 100 |
| BNC | pitch_up | 100 |
| BNC | stretch_0.9 | 100 |
| BNC | stretch_1.1 | 100 |
| CREMA | noise | 4969 |
| CREMA | orig | 5081 |
| CREMA | pitch_up | 4969 |
| RAVDESS | noise | 695 |
| RAVDESS | orig | 759 |
| RAVDESS | pitch_up | 695 |
| SAVEE | noise | 180 |
| SAVEE | orig | 240 |
| SAVEE | pitch_up | 180 |
| TESS | noise | 1600 |
| TESS | orig | 1600 |
| TESS | pitch_up | 1600 |

Not: Fold OOF test satırları `fold == k & aug_id == 'orig'` ile seçilir.

Tüm doğrulamalar PASS; eğitim/CV/tahmin yapılmadı.
