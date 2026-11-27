# A1.2 — F0 betimsel analiz (yalnızca hazırlık)

Yalnızca betimsel istatistik kullanıldı. Model eğitimi, CV, tahmin veya test sonucu kullanılmadı. F0 istatistikleri geçerli voiced karelerden, voiced paydası 108 analiz karesinden hesaplandı.

## 1. Kaynak ve sınıf bazında F0 (yalnızca `orig`, fallback'siz kayıtlar)

| source | label | n | n_valid | f0_mean_ort | f0_mean_std | f0_mean_medyan | f0_mean_p10 | f0_mean_p90 | voiced_oran_ort | fallback_oran |
|---|---|---|---|---|---|---|---|---|---|---|
| BNC | angry | 25 | 25 | 255.3985 | 33.1729 | 255.1856 | 213.7775 | 294.0545 | 0.5893 | 0.0000 |
| BNC | fear | 25 | 25 | 310.3880 | 26.5370 | 310.7309 | 270.1516 | 344.5932 | 0.5230 | 0.0000 |
| BNC | happy | 25 | 25 | 267.1942 | 34.0756 | 262.5908 | 228.4364 | 313.1402 | 0.6000 | 0.0000 |
| BNC | sad | 25 | 25 | 271.2677 | 27.2311 | 266.3406 | 243.7431 | 302.8867 | 0.5396 | 0.0000 |
| CREMA | angry | 1270 | 1217 | 226.9912 | 61.4372 | 226.3751 | 145.2917 | 309.0582 | 0.4365 | 0.0417 |
| CREMA | fear | 1270 | 1160 | 209.1013 | 69.3525 | 208.4609 | 114.2416 | 301.3299 | 0.3896 | 0.0866 |
| CREMA | happy | 1271 | 1163 | 216.2901 | 67.1929 | 220.1984 | 122.1596 | 303.2066 | 0.3570 | 0.0850 |
| CREMA | sad | 1270 | 1011 | 165.2686 | 57.0974 | 168.3022 | 97.6549 | 232.8908 | 0.3743 | 0.2039 |
| RAVDESS | angry | 192 | 187 | 243.3096 | 70.4227 | 259.9725 | 137.5484 | 326.7438 | 0.4653 | 0.0260 |
| RAVDESS | fear | 190 | 189 | 243.9537 | 76.7373 | 263.8873 | 136.0564 | 341.7641 | 0.5049 | 0.0053 |
| RAVDESS | happy | 191 | 191 | 224.6979 | 63.5626 | 234.0435 | 126.2115 | 304.6861 | 0.5372 | 0.0000 |
| RAVDESS | sad | 186 | 185 | 196.7040 | 68.7437 | 196.7127 | 106.9679 | 294.8420 | 0.5354 | 0.0054 |
| SAVEE | angry | 60 | 60 | 178.4167 | 28.3435 | 177.4735 | 142.3522 | 218.1547 | 0.6375 | 0.0000 |
| SAVEE | fear | 60 | 60 | 222.7164 | 46.5099 | 227.1033 | 167.8328 | 274.5288 | 0.6242 | 0.0000 |
| SAVEE | happy | 60 | 60 | 204.6860 | 25.8491 | 203.5495 | 175.7972 | 240.9605 | 0.6289 | 0.0000 |
| SAVEE | sad | 60 | 60 | 161.0871 | 30.0283 | 169.3841 | 120.2197 | 195.0165 | 0.5370 | 0.0000 |
| TESS | angry | 400 | 400 | 252.8317 | 15.3474 | 251.7195 | 233.6870 | 274.3097 | 0.5362 | 0.0000 |
| TESS | fear | 400 | 400 | 305.6636 | 13.5776 | 305.7198 | 287.5479 | 323.1833 | 0.4697 | 0.0000 |
| TESS | happy | 400 | 400 | 246.0078 | 27.4396 | 249.4861 | 207.7884 | 279.2004 | 0.4884 | 0.0000 |
| TESS | sad | 400 | 400 | 207.8772 | 14.3057 | 210.1932 | 188.2087 | 225.1686 | 0.6848 | 0.0000 |
| TUMU | angry | 1947 | 1889 | 232.9115 | 56.8418 | 240.0223 | 154.0288 | 299.6016 | 0.4688 | 0.0298 |
| TUMU | fear | 1945 | 1834 | 235.5795 | 73.3023 | 243.3712 | 127.2655 | 320.3041 | 0.4285 | 0.0571 |
| TUMU | happy | 1947 | 1839 | 223.9406 | 60.4710 | 230.0079 | 133.2056 | 295.4178 | 0.4165 | 0.0555 |
| TUMU | sad | 1941 | 1681 | 180.2942 | 55.2759 | 192.1869 | 104.1766 | 237.3547 | 0.4742 | 0.1340 |

## 2. Fear vs happy F0 dağılımı (`orig`, fallback'siz)

| label | n_valid | mean | std | median | p10 | p25 | p75 | p90 |
|---|---|---|---|---|---|---|---|---|
| fear | 1834 | 235.5795 | 73.3023 | 243.3712 | 127.2655 | 180.5397 | 299.1283 | 320.3041 |
| happy | 1839 | 223.9406 | 60.4710 | 230.0079 | 133.2056 | 186.7327 | 266.7451 | 295.4178 |

- Ortalama F0 farkı (fear − happy): **+11.64 Hz**; havuzlanmış standart sapma cinsinden etki büyüklüğü (deskriptif d): **+0.173**.
- p10–p90 aralıklarının kesişimi: **[133.21, 295.42] Hz**; kaba örtüşme oranı: **0.84** (1.0 = tam örtüşme).
- Bu değerler sınıflandırma performansı veya istatistiksel anlamlılık kanıtı değildir.

## 3. Pitch augmentation'ın F0'ya etkisi

| aug_id | n_cift | orig_f0_mean_ort | aug_f0_mean_ort | ortani_arac | oran_p10 | oran_p90 | beklenen_oran | beklenen_fark |
|---|---|---|---|---|---|---|---|---|
| noise | 7001 | 219.5329 | 220.5953 | 1.0000 | 0.9978 | 1.0087 | 1.0000 | pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; noise/stretch F0'yu degistirmemeli |
| pitch_up | 6878 | 220.5473 | 231.2563 | 1.0592 | 0.9819 | 1.1032 | 1.0595 | pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; noise/stretch F0'yu degistirmemeli |
| pitch_down | 100 | 276.0621 | 258.0212 | 0.9017 | 0.8764 | 1.0405 | 0.8909 | pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; noise/stretch F0'yu degistirmemeli |
| stretch_0.9 | 100 | 276.0621 | 279.5193 | 1.0123 | 0.9696 | 1.0582 | 1.0000 | pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; noise/stretch F0'yu degistirmemeli |
| stretch_1.1 | 100 | 276.0621 | 274.8605 | 1.0000 | 0.9533 | 1.0365 | 1.0000 | pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; noise/stretch F0'yu degistirmemeli |
| noise2 | 100 | 276.0621 | 276.1484 | 1.0000 | 0.9833 | 1.0180 | 1.0000 | pitch_up ~ +1 semiton; pitch_down ~ -2 semiton; noise/stretch F0'yu degistirmemeli |

- Beklenen oranlar: `pitch_up` için 2^(1/12) ≈ 1.0595 (+1 semiton), `pitch_down` için 2^(-2/12) ≈ 0.8909 (−2 semiton). Noise ve time-stretch varyantlarında F0 değişmemeli (oran ≈ 1).
- Oranlar aynı `source_id` için eşleştirilmiş `aug_f0_mean / orig_f0_mean` değerlerinin medyanıdır; fallback uygulanan kayıtlar hariç tutuldu.

## 4. Voiced frame oranı ve F0 çıkılamama

| source | n_kayit | voiced_oran_ort | voiced_oran_medyan | voiced_sifir_sayisi | fallback_sayisi |
|---|---|---|---|---|---|
| BNC | 100 | 0.5630 | 0.5694 | 0 | 0 |
| CREMA | 5081 | 0.3497 | 0.3611 | 530 | 530 |
| RAVDESS | 759 | 0.5061 | 0.5185 | 7 | 7 |
| SAVEE | 240 | 0.6069 | 0.6204 | 0 | 0 |
| TESS | 1600 | 0.5448 | 0.5370 | 0 | 0 |

| label | n_kayit | voiced_oran_ort | f0_ort | f0_std | fallback_sayisi |
|---|---|---|---|---|---|
| angry | 1947 | 0.4549 | 225.9732 | 68.5932 | 58 |
| fear | 1945 | 0.4040 | 222.1351 | 89.7618 | 111 |
| happy | 1947 | 0.3934 | 211.5187 | 78.0033 | 108 |
| sad | 1941 | 0.4107 | 156.1435 | 80.1276 | 260 |

Not: BNC yalnızca 100 kayıt/1 kaynak grubu içerir; bu kaynakta F0 güvenilirliği örneklem büyüklüğü nedeniyle sınırlıdır. TESS 2, SAVEE 4 grup içerir.
