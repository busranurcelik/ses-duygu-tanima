# A1.2 10-fold OOF — A0 ile karsilastirma

A0 ve A1.2 ayni fold atamalari, ayni seed'ler (43..52), ayni RF/MLP ayarlari ve ayni degerlendirme protokolunu kullanir (test = ilgili fold'un `orig` satirlari). A1.2'de 184 A0 ozelligine 8 F0 ozelligi eklendi; fallback sentinel'leri her fold'da yalnizca o fold train'inden ogrenilen medyanla impute edildi.
McNemar, ROC/AUC, olasilik kaydi, 70/15/15 ve model kaydi YAPILMADI.

## 1. Genel performans ve fear/happy

| model | metrik | A0 | A1.2 | fark (A1.2-A0) |
|---|---|---:|---:|---:|
| RF | accuracy | 0.5860 | 0.5978 | +0.0118 |
| RF | f1_macro | 0.5810 | 0.5922 | +0.0112 |
| RF | f1_fear | 0.4606 | 0.4645 | +0.0039 |
| RF | f1_happy | 0.5182 | 0.5497 | +0.0315 |
| RF | fear_to_happy | 394.0000 | 402.0000 | +8.0000 |
| RF | happy_to_fear | 274.0000 | 220.0000 | -54.0000 |
| RF | fear_to_happy_rate | 0.2026 | 0.2067 | +0.0041 |
| RF | happy_to_fear_rate | 0.1407 | 0.1130 | -0.0277 |
| MLP | accuracy | 0.5796 | 0.5950 | +0.0154 |
| MLP | f1_macro | 0.5755 | 0.5912 | +0.0158 |
| MLP | f1_fear | 0.4533 | 0.4848 | +0.0315 |
| MLP | f1_happy | 0.5123 | 0.5421 | +0.0299 |
| MLP | fear_to_happy | 587.0000 | 458.0000 | -129.0000 |
| MLP | happy_to_fear | 215.0000 | 225.0000 | +10.0000 |
| MLP | fear_to_happy_rate | 0.3018 | 0.2355 | -0.0663 |
| MLP | happy_to_fear_rate | 0.1104 | 0.1156 | +0.0051 |

## 2. Kaynak bazli performans

| model | kaynak | metrik | A0 | A1.2 | fark |
|---|---|---|---:|---:|---:|
| RF | BNC | accuracy | 0.3500 | 0.3600 | +0.0100 |
| RF | BNC | f1_macro | 0.2568 | 0.2572 | +0.0004 |
| RF | CREMA | accuracy | 0.5869 | 0.5971 | +0.0102 |
| RF | CREMA | f1_macro | 0.5751 | 0.5856 | +0.0105 |
| RF | RAVDESS | accuracy | 0.5033 | 0.5020 | -0.0013 |
| RF | RAVDESS | f1_macro | 0.4971 | 0.4949 | -0.0022 |
| RF | SAVEE | accuracy | 0.5125 | 0.4958 | -0.0167 |
| RF | SAVEE | f1_macro | 0.5088 | 0.4887 | -0.0201 |
| RF | TESS | accuracy | 0.6481 | 0.6756 | +0.0275 |
| RF | TESS | f1_macro | 0.6445 | 0.6743 | +0.0298 |
| MLP | BNC | accuracy | 0.3300 | 0.3200 | -0.0100 |
| MLP | BNC | f1_macro | 0.2159 | 0.2092 | -0.0068 |
| MLP | CREMA | accuracy | 0.5934 | 0.6113 | +0.0179 |
| MLP | CREMA | f1_macro | 0.5846 | 0.6051 | +0.0205 |
| MLP | RAVDESS | accuracy | 0.4980 | 0.5310 | +0.0329 |
| MLP | RAVDESS | f1_macro | 0.5031 | 0.5332 | +0.0301 |
| MLP | SAVEE | accuracy | 0.4500 | 0.5250 | +0.0750 |
| MLP | SAVEE | f1_macro | 0.4503 | 0.5214 | +0.0711 |
| MLP | TESS | accuracy | 0.6094 | 0.6012 | -0.0081 |
| MLP | TESS | f1_macro | 0.6024 | 0.5913 | -0.0110 |

## 3. Fold tutarligi

| model | metrik | A0 | A1.2 | fark |
|---|---|---:|---:|---:|
| RF | accuracy_mean | 0.5795 | 0.5901 | +0.0106 |
| RF | accuracy_std | 0.0445 | 0.0496 | +0.0051 |
| RF | f1_macro_mean | 0.5654 | 0.5751 | +0.0098 |
| RF | f1_macro_std | 0.0401 | 0.0424 | +0.0023 |
| MLP | accuracy_mean | 0.5756 | 0.5927 | +0.0172 |
| MLP | accuracy_std | 0.0406 | 0.0434 | +0.0028 |
| MLP | f1_macro_mean | 0.5619 | 0.5811 | +0.0192 |
| MLP | f1_macro_std | 0.0344 | 0.0417 | +0.0073 |

| model | fold | accuracy A0 | accuracy A1.2 | fark | F1 A0 | F1 A1.2 | fark |
|---|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 0.6293 | 0.6463 | +0.0170 | 0.6189 | 0.6377 | +0.0188 |
| RF | 2 | 0.4875 | 0.5109 | +0.0234 | 0.4751 | 0.4979 | +0.0228 |
| RF | 3 | 0.6165 | 0.6313 | +0.0147 | 0.5952 | 0.6078 | +0.0126 |
| RF | 4 | 0.5870 | 0.5981 | +0.0111 | 0.5802 | 0.5897 | +0.0095 |
| RF | 5 | 0.5916 | 0.5968 | +0.0053 | 0.5738 | 0.5775 | +0.0037 |
| RF | 6 | 0.6166 | 0.6617 | +0.0450 | 0.5732 | 0.6172 | +0.0440 |
| RF | 7 | 0.5184 | 0.5248 | +0.0064 | 0.5141 | 0.5211 | +0.0070 |
| RF | 8 | 0.5505 | 0.5412 | -0.0093 | 0.5533 | 0.5426 | -0.0107 |
| RF | 9 | 0.5812 | 0.5669 | -0.0143 | 0.5732 | 0.5571 | -0.0161 |
| RF | 10 | 0.6162 | 0.6230 | +0.0068 | 0.5968 | 0.6028 | +0.0059 |
| MLP | 1 | 0.6207 | 0.6548 | +0.0341 | 0.6089 | 0.6456 | +0.0367 |
| MLP | 2 | 0.5000 | 0.5406 | +0.0406 | 0.4949 | 0.5309 | +0.0361 |
| MLP | 3 | 0.6018 | 0.6283 | +0.0265 | 0.5817 | 0.6153 | +0.0337 |
| MLP | 4 | 0.5854 | 0.6234 | +0.0380 | 0.5840 | 0.6192 | +0.0352 |
| MLP | 5 | 0.6126 | 0.6377 | +0.0250 | 0.5991 | 0.6278 | +0.0286 |
| MLP | 6 | 0.6136 | 0.6036 | -0.0100 | 0.5656 | 0.5540 | -0.0116 |
| MLP | 7 | 0.5328 | 0.5472 | +0.0144 | 0.5341 | 0.5466 | +0.0124 |
| MLP | 8 | 0.5194 | 0.5179 | -0.0016 | 0.5198 | 0.5184 | -0.0014 |
| MLP | 9 | 0.5828 | 0.5780 | -0.0048 | 0.5770 | 0.5752 | -0.0017 |
| MLP | 10 | 0.5863 | 0.5958 | +0.0095 | 0.5539 | 0.5781 | +0.0242 |

Not: `fear_to_happy` / `happy_to_fear` sayilari OOF karisiklik matrisinden alinmistir; oranlar ilgili gercek sinif satir sayisina bolunmustur. Farklar betimseldir; istatistiksel anlamlilik testi yapilmadi.
