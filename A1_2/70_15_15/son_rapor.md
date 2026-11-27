# A1.2 — 70/15/15 Değerlendirme Raporu

Bu çalışma, mevcut A0 70/15/15 protokolü birebir korunarak A1.2'nin 192 özellikli veri seti üzerinde çalıştırılmıştır. 10-fold OOF sonuçları ve mevcut A1.2 raporu değiştirilmemiştir.

## 1. Kritik kapsam uyarısı

**Mevcut 70/15/15 test kümesinin tamamı `TESS_YAF` konuşmacısındandır (800 özgün kayıt, kaynak=TESS). Bu sonuç konuşmacıdan bağımsız genelleme başarısı olarak sunulamaz; yalnızca görülmemiş tek bir TESS konuşmacısındaki performansı gösterir.**

## 2. Protokol ve veri

- A0 ile aynı split kolonu: `split_70_15_15`.
- Satırlar: girdi 23.268. EXCLUDED atılan 3.664. train 17.536. val 1.268. test 800.
- Train aug_id'leri: noise,noise2,orig,pitch_down,pitch_up,stretch_0.9,stretch_1.1; val/test: orig.
- Test kaynağı/grubu: TESS / TESS_YAF.
- Seed 42, MLP 200 epoch (erken durma ile), batch 32; RF-B 200 ağaç, `class_weight='balanced'`, `random_state=42`.
- Scaler yalnızca train'e fit edildi; RF-B ve MLP aynı scaleri kullandı.

## 3. F0 imputasyonu

7 sentinel sütun (184..190) yalnızca train'in fallback'siz satırlarından hesaplanan medyanlarla dolduruldu; val ve test için aynı medyanlar kullanıldı. `voiced_frame_ratio` (191) impute edilmedi.

| özellik | train medyanı | train fallback'siz | train impute | val impute | test impute |
|---|---:|---:|---:|---:|---:|
| f0_mean | 223.567904 | 16.342 | 1.194 | 147 | 0 |
| f0_median | 220.080970 | 16.342 | 1.194 | 147 | 0 |
| f0_std | 25.407571 | 16.342 | 1.194 | 147 | 0 |
| f0_min | 177.731706 | 16.342 | 1.194 | 147 | 0 |
| f0_max | 282.131498 | 16.342 | 1.194 | 147 | 0 |
| f0_range | 93.326705 | 16.342 | 1.194 | 147 | 0 |
| f0_slope | -18.247327 | 16.342 | 1.194 | 147 | 0 |

Train'de 16.342 fallback'siz satır. 1.194 impute edilen satır; val'de 147. test'te 0 impute uygulandı.

## 4. Test performansı: A0 vs A1.2

A0 değerleri kaydedilmiş test tahminlerinden, A1.2 değerleri bu çalışmadaki tahminlerden hesaplanmıştır. Test kümesi iki deney için birebir aynıdır.

| Model | Metrik | A0 | A1.2 | Fark |
|---|---|---:|---:|---:|
| RF-B | accuracy | 0,6450 | 0,6562 | +0,0112 |
| RF-B | precision_macro | 0,7107 | 0,7257 | +0,0150 |
| RF-B | recall_macro | 0,6450 | 0,6562 | +0,0112 |
| RF-B | f1_macro | 0,6101 | 0,6252 | +0,0151 |
| RF-B | f1_angry | 0,6899 | 0,6850 | -0,0049 |
| RF-B | f1_fear | 0,3665 | 0,4125 | +0,0459 |
| RF-B | f1_happy | 0,3840 | 0,4033 | +0,0193 |
| RF-B | f1_sad | 1,0000 | 1,0000 | +0,0000 |
| MLP | accuracy | 0,5575 | 0,6075 | +0,0500 |
| MLP | precision_macro | 0,6040 | 0,6302 | +0,0262 |
| MLP | recall_macro | 0,5575 | 0,6075 | +0,0500 |
| MLP | f1_macro | 0,4805 | 0,5548 | +0,0743 |
| MLP | f1_angry | 0,7505 | 0,7486 | -0,0019 |
| MLP | f1_fear | 0,2212 | 0,3373 | +0,1161 |
| MLP | f1_happy | 0,1705 | 0,2656 | +0,0951 |
| MLP | f1_sad | 0,7797 | 0,8677 | +0,0880 |

## 5. Sınıf bazlı precision / recall / F1 (A1.2 test)

| Model | Sınıf | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| RF-B | angry | 0,5223 | 0,9950 | 0,6850 | 200 |
| RF-B | fear | 0,9298 | 0,2650 | 0,4125 | 200 |
| RF-B | happy | 0,4506 | 0,3650 | 0,4033 | 200 |
| RF-B | sad | 1,0000 | 1,0000 | 1,0000 | 200 |
| MLP | angry | 0,6075 | 0,9750 | 0,7486 | 200 |
| MLP | fear | 0,8571 | 0,2100 | 0,3373 | 200 |
| MLP | happy | 0,2899 | 0,2450 | 0,2656 | 200 |
| MLP | sad | 0,7663 | 1,0000 | 0,8677 | 200 |

## 6. Confusion matrix (A1.2 test)

**RF-B**

| gerçek \ tahmin | angry | fear | happy | sad |
|---|---:|---:|---:|---:|
| angry | 199 | 0 | 1 | 0 |
| fear | 59 | 53 | 88 | 0 |
| happy | 123 | 4 | 73 | 0 |
| sad | 0 | 0 | 0 | 200 |

**MLP**

| gerçek \ tahmin | angry | fear | happy | sad |
|---|---:|---:|---:|---:|
| angry | 195 | 1 | 0 | 4 |
| fear | 19 | 42 | 120 | 19 |
| happy | 107 | 6 | 49 | 38 |
| sad | 0 | 0 | 0 | 200 |

## 7. Fear ↔ happy karışıklığı

| Model | Yön | A0 | A1.2 | Fark |
|---|---|---:|---:|---:|
| RF-B | fear_to_happy | 101 | 88 | -13 |
| RF-B | happy_to_fear | 5 | 4 | -1 |
| RF-B | fear_to_happy_rate | 0,5050 | 0,4400 | -0,0650 |
| RF-B | happy_to_fear_rate | 0,0250 | 0,0200 | -0,0050 |
| MLP | fear_to_happy | 122 | 120 | -2 |
| MLP | happy_to_fear | 1 | 6 | +5 |
| MLP | fear_to_happy_rate | 0,6100 | 0,6000 | -0,0100 |
| MLP | happy_to_fear_rate | 0,0050 | 0,0300 | +0,0250 |

F0 eklenmesine rağmen fear/happy karışıklığı tamamen çözülmemiştir; yön ve büyüklük modele göre değişmektedir.

## 8. Eğitim / validation performans farkı

Train metrikleri augmentation'lı eğitim satırlarını içerir; val ve test yalnızca özgün kayıtlardır.

| Model | Metrik | Train | Val | Train−Val | Test |
|---|---|---:|---:|---:|---:|
| RF-B | accuracy | 1,0000 | 0,6096 | +0,3904 | 0,6562 |
| RF-B | precision_macro | 1,0000 | 0,6041 | +0,3959 | 0,7257 |
| RF-B | recall_macro | 1,0000 | 0,6096 | +0,3904 | 0,6562 |
| RF-B | f1_macro | 1,0000 | 0,5986 | +0,4014 | 0,6252 |
| MLP | accuracy | 0,8606 | 0,6151 | +0,2454 | 0,6075 |
| MLP | precision_macro | 0,8646 | 0,6178 | +0,2468 | 0,6302 |
| MLP | recall_macro | 0,8606 | 0,6151 | +0,2454 | 0,6075 |
| MLP | f1_macro | 0,8599 | 0,6070 | +0,2529 | 0,5548 |

## 9. Eğitim süreleri ve epoch

- MLP: **31.09 sn**, çalıştırılan epoch **35** / 200, en iyi val_accuracy epoch **25**.
- RF-B: **3.54 sn**.

## 10. Doğrulama ve sızıntı kontrolleri

- Kontrol: **42**, PASS: **42**, FAIL: **0**.
- İlk 184 özellik A0 ile birebir doğrulandı; grup/source sızıntısı yok; val/test yalnızca orig; scaler ve F0 medyanları yalnızca train'den.
- Ayrıntılar: `dogrulama_kontrolleri.csv` / `.md`.

## 11. Sınırlılıklar

1. Test yalnızca `TESS_YAF` konuşmacısından oluşur; konuşmacıdan bağımsız genelleme iddiası yoktur.
2. TESS'te test için yalnızca 4 grup/konuşmacı vardır ve bu split'te test tek konuşmacıdır.
3. Train metrikleri augmentation nedeniyle bağımsız değildir.
4. McNemar/ROC/10-fold CV bu deneyde çalıştırılmamıştır.
5. A0 70/15/15 modeli yeniden eğitilmemiş; kaydedilmiş test tahminleri kullanılmıştır.

## Çıktılar

- `ysa/`: metrikler, sınıf metrikleri, confusion matrix, eğitim grafikleri, label map.
- `modeller/`: scaler, RF-B, en iyi ve son MLP.
- `karsilastirma/`: A1.2 test tahminleri ve A0 karşılaştırma tabloları.
- `dogrulama_kontrolleri.*`, `filtre_dogrulama.csv`, `f0_imputasyon_degerleri.csv`, `artifact_manifest_A1_2.json`, `egitim_log.txt`.
