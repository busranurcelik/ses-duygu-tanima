# A1.2 — Model seçimi ve paketleme öncesi inceleme

Bu rapor yalnızca mevcut sonuçların ve dosyaların incelenmesiyle hazırlanmıştır. Model eğitilmemiş, model/scaler dosyası oluşturulmamış veya değiştirilmemiş, `7_final_arayuz.py` dahil GUI kodu değiştirilmemiş, mevcut sonuç dosyaları üzerine yazılmamış ve commit/push yapılmamıştır.

## 1. Model seçimi

### 1.1. Dört deneyin OOF karşılaştırması

Tüm değerler 7.780 canonical kayıt üzerindeki 10-fold group-aware OOF sonuçlarından alınmıştır. Deney tanımları: **A0** 184 özellik, **A1.1** 264 özellik (A0 + 40 delta MFCC + 40 delta-delta MFCC), **A3** 184 özellik + etiketsiz deterministik 2,5 s pencere seçimi, **A1.2** 192 özellik (A0 + 8 F0/pitch özelliği).

| Deney | RF accuracy | RF macro-F1 | MLP accuracy | MLP macro-F1 |
|---|---:|---:|---:|---:|
| A0 | 0,5860 | 0,5810 | 0,5796 | 0,5755 |
| A1.1 | 0,5766 | 0,5699 | 0,5630 | 0,5591 |
| A3 | 0,5811 | 0,5764 | 0,5815 | 0,5769 |
| **A1.2** | **0,5978** | **0,5922** | **0,5950** | **0,5912** |

A1.2, hem RF hem MLP için accuracy ve macro-F1'in tamamında en yüksek değere sahiptir. A0'a göre OOF artışı RF'de **+0,0118 accuracy / +0,0112 macro-F1**, MLP'de **+0,0154 accuracy / +0,0158 macro-F1**'dir. A1.2'nin kaynak tabakalı küme bootstrap %95 aralıkları bu dört genel metrik için sıfır içermez; ancak sınıf bazında yalnızca **happy F1** artışı iki modelde de sıfır içermeyen aralığa sahiptir. MLP fear F1 artışı pozitif olmasına rağmen güven aralığı sıfırı içerir.

A1.1 genel olarak A0'ın altında kalmıştır. A3 karışık sonuç vermiştir: RF'de hafif gerileme, MLP'de marjinal iyileşme vardır. A1.1 ve A3 için kaydedilmiş model/scaler bulunmamaktadır; bu deneyler yalnızca OOF kanıtıdır ve paketleme adayı değildir.

A0 için 70/15/15 referansı: RF-B **0,6450 accuracy / 0,6101 macro-F1**, MLP **0,5575 accuracy / 0,4805 macro-F1**. A1.2 70/15/15 sonuçları bölüm 1.3'te verilmiştir.

### 1.2. A1.2 RF ve MLP: 10-fold OOF karşılaştırması

| Model | Accuracy | Macro-F1 | Angry F1 | Fear F1 | Happy F1 | Sad F1 |
|---|---:|---:|---:|---:|---:|---:|
| RF | **0,597815** | **0,592168** | **0,634441** | 0,464516 | **0,549716** | **0,720000** |
| MLP | 0,594987 | 0,591238 | 0,627674 | **0,484813** | 0,542121 | 0,710343 |
| RF − MLP | +0,002828 | +0,000931 | +0,006767 | −0,020297 | +0,007595 | +0,009657 |

RF'nin OOF üstünlüğü **çok küçüktür**: accuracy farkı 0,0028, macro-F1 farkı 0,0009. Bu nedenle RF'nin MLP'den istatistiksel olarak üstün olduğu iddia edilemez; RF ile MLP arasında doğrudan cluster-bootstrap veya eşleştirilmiş anlamlılık testi çalıştırılmamıştır. Buna karşılık A0'a göre iyileşme kanıtı RF ve MLP için ayrı ayrı desteklenmiştir.

**Fear ↔ happy karışıklığı, OOF:**

| Model | Fear→Happy | Happy→Fear | Toplam | Fear→Happy oranı | Happy→Fear oranı |
|---|---:|---:|---:|---:|---:|
| RF | 402 | 220 | 622 | 0,2067 | 0,1130 |
| MLP | 458 | 225 | 683 | 0,2355 | 0,1156 |

RF, OOF'ta fear→happy karışıklığını MLP'ye göre 56 örnek daha az yapmıştır. Ancak F0 eklenmesine rağmen fear/happy ayrımı çözülmüş değildir.

**Fold tutarlılığı:**

| Model | Accuracy ort. | Accuracy std | Min | Max | A0'a göre iyileşen fold |
|---|---:|---:|---:|---:|---:|
| RF | 0,5901 | 0,0523 | 0,5109 (fold 2) | 0,6617 (fold 6) | 8/10 |
| MLP | 0,5927 | 0,0457 | 0,5179 (fold 8) | 0,6548 (fold 1) | 7/10 |

MLP fold'lar arasında biraz daha kararlıdır (daha düşük std). RF daha yüksek tepe performansa sahip olmasına karşın fold değişkenliği daha büyüktür. Düşük performanslı fold'lar kaynak bileşiminden etkilenmiştir (ör. fold 8'de BNC, fold 10'da TESS YAF). Bu nedenle tek bir ortalamaya bakarak genelleme iddiası güçlü değildir.

### 1.3. A1.2 RF ve MLP: 70/15/15 sonuçları

Aynı A0 split'i kullanılmıştır: train 17.536, val 1.268, test 800. Test kümesinin tamamı **TESS_YAF** konuşmacısındandır.

| Model | Accuracy | Macro-F1 | Angry F1 | Fear F1 | Happy F1 | Sad F1 |
|---|---:|---:|---:|---:|---:|---:|
| RF-B | **0,65625** | **0,625198** | 0,685026 | **0,412451** | **0,403315** | **1,000000** |
| MLP | 0,60750 | 0,554793 | **0,748560** | 0,337349 | 0,265583 | 0,867679 |
| RF − MLP | +0,04875 | +0,070405 | −0,063534 | +0,075102 | +0,137732 | +0,132321 |

RF-B, görülmemiş TESS_YAF konuşmacısında MLP'den belirgin biçimde daha iyidir. MLP yalnızca angry F1'de öndedir; RF fear, happy ve sad sınıflarında üstündür. A0'a göre test artışları RF'de **+0,0112 accuracy / +0,0151 macro-F1**, MLP'de **+0,0500 accuracy / +0,0743 macro-F1**'dir.

**Test confusion matrix, fear ↔ happy:**

| Model | Fear→Happy | Happy→Fear |
|---|---:|---:|
| RF-B | 88 | 4 |
| MLP | 120 | 6 |

MLP ayrıca fear→sad 19 ve happy→sad 38 hata yapmıştır; RF bu yönlerde sıfır hata yapmıştır. RF'nin sad sınıfı test F1'i 1,0000, MLP'nin 0,8677'dir.

**Train/validation farkı ve süre:**

| Model | Train accuracy | Val accuracy | Train−Val accuracy | Train macro-F1 | Val macro-F1 | Train−Val F1 | Süre |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF-B | 1,0000 | 0,6096 | 0,3904 | 1,0000 | 0,5986 | 0,4014 | 3,54 sn |
| MLP | 0,8606 | 0,6151 | 0,2454 | 0,8599 | 0,6070 | 0,2529 | 31,09 sn |

MLP 35/200 epoch çalıştırılmış, test tahminleri **en iyi val_accuracy checkpoint'i** (epoch 25) ile üretilmiştir. RF-B'nin train metriklerinin 1,0 olması ve büyük train-val farkı ciddi ezberleme eğilimini göstermektedir; buna rağmen val/test performansı MLP'den yüksektir.

### 1.4. Güçlü ve zayıf yönler

| Model | Güçlü yönler | Zayıf yönler |
|---|---|---|
| **RF-B** | OOF ve 70/15/15'in ikisinde de en yüksonuç; fear/happy karışıklığı OOF ve test'te daha düşük; sad testi hatasız; deterministik (`random_state=42`); 3,54 sn ile hızlı; dağıtımı basit (`joblib` + scikit-learn) | OOF'ta MLP'ye üstünlüğü marjinal; fold std'si daha yüksek; train=1,0 ile çok büyük train-val farkı; fear/happy karışıklığı hâlâ ciddi; kaynak bileşimine duyarlı |
| **MLP** | Fold'lar arasında daha kararlı; OOF fear F1 ve 70/15/15 angry F1 daha iyi; A0'a göre OOF iyileşmesi biraz daha büyük; olasılık çıktısı mevcut | Mutlak test performansı RF'nin gerisinde; test'te fear/happy ve happy/sad karışıklığı daha kötü; TensorFlow/Keras bağımlılığı; 31,09 sn; best/final checkpoint ayrımı var; train-val farkı hâlâ 0,25 |

### 1.5. Metodolojik değerlendirme

**RF'yi ana model adayı, MLP'yi alternatif olarak tutmak metodolojik olarak uygundur.** Gerekçe:

1. RF, OOF'ta marjinal de olsa öndedir; 70/15/15 testinde ise açıkça daha iyidir.
2. RF, fear/happy gibi problematik sınıfta hem OOF hem test'te daha iyi davranır.
3. RF daha hızlı, deterministik ve paketlenmesi daha basittir.
4. A1.2'nin A0'a göre iyileşmesi bootstrap ile desteklenmiştir; bu kanıt RF için de geçerlidir.

Ancak bu karar **kesin üstünlük iddiası değil, aday seçimi** olarak sunulmalıdır. RF-MLP farkı OOF'ta fold gürültüsüyle karşılaştırılabilir düzeydedir; 70/15/15 testi tek konuşmacıdır ve konuşmacıdan bağımsız genelleme göstermez. MLP alternatif olarak korunmalı; farklı bir bağımsız test kümesinde karşılaştırma yapılmadan "RF daha geneller" denmemelidir.

## 2. Paketleme öncesi inceleme

### 2.1. GUI için gerekli bileşenler ve mevcut durum

| Bileşen | Kullanılması gereken kaynak | Deney/provenans | Uyum durumu |
|---|---|---|---|
| Model (ana aday) | `A1_2/70_15_15/modeller/rf_70_15_15_matched_A1_2.joblib` | A1.2 70/15/15, seed 42, 200 ağaç, `class_weight='balanced'` | **Uygun.** `n_features_in_=192`, `classes_=[0,1,2,3]`, A0 RF-B ayarlarıyla aynı. SHA-256: `e3b63152e0b2802efdd9ea2b54f0cf669bb9cb077fdbf87d8ede437f576be15d`. |
| Model (alternatif) | `A1_2/70_15_15/modeller/mlp_best_70_15_15_A1_2.keras` | A1.2 70/15/15, en iyi `val_accuracy` checkpoint'i, epoch 25 | **Uygun, ikinci tercih.** Input `(None,192)`, output `(None,4)`. `mlp_final_...keras` test metriklerinin kaynağı değildir; kullanılmamalıdır. SHA-256: `be6e282a49eef73d2b4ff91b3907b41e64984ec75db16b2da2852c3083659f8e`. |
| Scaler | `A1_2/70_15_15/modeller/scaler_A1_2.joblib` | Yalnızca 70/15/15 train (17.536 satır) | **Uygun.** `n_features_in_=192`; RF-B ve MLP aynı scaleri kullandı. SHA-256: `835152626cf33da3b7bbea0dcd436fd4ca1a3ea52770d4847a0b4d65b5297b10`. |
| F0 imputasyon medyanları | `A1_2/70_15_15/f0_imputasyon_degerleri.csv` | Yalnızca 70/15/15 train'in fallback'siz satırları | **Uygun.** 184..190 için medyanlar; 191 impute edilmez. `f0_mean` medyanı 223,567904 Hz. `A1_2/kfold/f0_imputasyon_degerleri.csv` fold-bazlıdır ve bu modelle **karıştırılmamalıdır**. |
| Özellik sırası / isimleri | `A1_2/final_oznitelik_seti_A1_2.csv` + `A1_2/f0_ozellik_tanimlari.csv` + A0 çıkarım kodu | 10 metadata + 192 özellik; ilk 184 A0 ile aynı | **Şartname hazır, GUI uygulaması eksik.** Kolonlar konumsal `"0".."191"`; model/scaler isim tutmaz. GUI şu anda 184 özellik üretir. |
| Label map | `A1_2/70_15_15/ysa/70_15_15_label_map_A1_2.json` | LabelEncoder ve RF `classes_` ile aynı | **Uygun.** `["angry","fear","happy","sad"]` → 0,1,2,3. |
| Ses ön işleme | `A1_2/ozellik_cikarimi_A1_2.py` (satır 73–77, 146–149, 187–245) | A0 zinciri + pYIN | **Şartname hazır, GUI'de F0 kısmı eksik.** A0 penceresi korunur; A3 `select_window()` kullanılmaz. |
| Provenans / doğrulama | `A1_2/70_15_15/artifact_manifest_A1_2.json`, `dogrulama_kontrolleri.*`, `filtre_dogrulama.csv` | Aynı 70/15/15 çalışması | **Uygun.** 42/42 doğrulama PASS; hash'ler manifest ile tutarlı. |

### 2.2. Doğru çıkarım zinciri (paketleme şartnamesi)

1. `librosa.load(path, sr=22050)` ile mono ses yüklenir; çıkarımda augmentation yoktur (`orig`).
2. `librosa.util.normalize` uygulanır.
3. `librosa.util.fix_length(y, 55125)` ile ilk 2,5 s kullanılır; kısa kayıtlar sıfırla doldurulur. Bu davranış A0 ve A1.2 ile aynıdır; A3 pencere seçimi A1.2 için geçerli değildir.
4. İlk 184 özellik üretilir: MFCC (`n_mfcc=40`) ortalama 40, chroma-STFT ortalama 12, mel-spektrogram (`n_mels=128`) ortalama 128, RMS ortalama/std 2, ZCR ortalama/std 2 → toplam 184.
5. Aynı 2,5 s pencereden F0 çıkarılır: `librosa.pyin`, `sr=22050`, `fmin=60`, `fmax=400`, `frame_length=2048`, `hop_length=512`. Geçerli voiced kare sayısı < 3 ise ilk 7 F0 alanı sentinel `0.0` kabul edilir.
6. 184..190 için sentinel değerleri 70/15/15 train medyanlarıyla doldurulur; 191 (`voiced_frame_ratio = n_valid/108`) **olduğu gibi bırakılır**. Bu adım scaler'dan önce yapılmalıdır.
7. 192 özellik **tam bu sırayla** scaler'a verilir: `[0..183 A0] + [184 f0_mean, 185 f0_median, 186 f0_std, 187 f0_min, 188 f0_max, 189 f0_range, 190 f0_slope, 191 voiced_frame_ratio]`.
8. RF için `predict_proba`, MLP için `predict` çağrılır; sütun/sınıf sırası label map ile eşleştirilir.

### 2.3. Mevcut GUI ile uyum kontrolü

`7_final_arayuz.py` **incelendi, değiştirilmedi**. Tespit edilenler:

- GUI şu anda `sonuclar_ysa/best_model.keras`, `sonuclar_ysa/scaler.pkl` ve `sonuclar_ysa/label_map.json` yolunu kullanıyor. Bu dosyaların model/scaler hash'leri ne A0 dedup ne de A1.2 artefaktlarıyla aynıdır. Scaler ve model girdisi **184**'tür. Bu varlıklar eski/ayrı bir eğitim çıktısıdır ve A1.2 ile uyumlu değildir.
- GUI'nin `extract_features()` fonksiyonu yalnızca 184 özellik döndürür. A1.2 scaler'ına (192) bağlanırsa `feats.shape[1] != self.scaler.n_features_in_` kontrolü hatayı başlatır. F0 çıkarımı ve imputasyon adımı GUI'de yoktur.
- GUI yalnızca `tf.keras.models.load_model()` kullanır; RF için `joblib.load()` + `predict_proba()` yolu henüz yoktur.
- GUI'de `HAPPY_MIN=0.28`, `HAPPY_MARGIN=0.06/0.08/0.07` gibi **eğitim/değerlendirme protokolünde olmayan** post-hoc happy eşik düzeltmesi vardır. Bu düzeltme korunursa raporlanan metrikler GUI çıktısıyla reprodükle edilemez; entegrasyonda devre dışı bırakılmalıdır.
- GUI'nin ilk 2,5 s pencere ve `librosa.util.normalize` davranışı A1.2 ile uyumludur. A3 notundaki `select_window()` henüz uygulanmamıştır; A1.2 seçilirse bu bir eksik değildir.
- `sonuclar_v2_dedup/modeller/` altındaki A0 dedup model/scaler 184 özelliktir ve kendi içinde tutarlıdır; ancak GUI onları da kullanmıyor. A0, karşılaştırma tabanı olarak korunmalıdır.
- A1.1 ve A3 dizinlerinde kaydedilmiş model/scaler dosyası bulunamadı; bu deneyler OOF kanıtıdır, GUI adayı değildir.

### 2.4. Açıkça hazır olan ve eksik olan

**Dosya düzeyinde GUI'ye alınmaya hazır olan:** **A1.2 RF-B** artefakt setidir (`rf_70_15_15_matched_A1_2.joblib` + `scaler_A1_2.joblib` + `f0_imputasyon_degerleri.csv` + 192 özellik şartnamesi + `70_15_15_label_map_A1_2.json`). Aynı setin MLP karşılığı da hazırdır; `mlp_best_...keras` alternatif olarak kullanılabilir, `mlp_final_...keras` kullanılmamalıdır.

**Doğrudan çalışan GUI entegrasyonu henüz hazır değildir.** Eksikler:

1. GUI'de 192 özellik çıkarımı ve F0/pYIN bloğu yoktur.
2. Sentinel/fallback tespiti ve 70/15/15 medyanlarıyla imputasyon çalışma zamanında uygulanmamıştır. Medyan dosyası hazır olsa da yeni ses/mikrofon kaydı için `source_id`/`aug_id` olmadığından karar, `extract_f0` mantığıyla üretilen `n_valid < 3` bilgisinden alınmalıdır.
3. RF yükleme/tahmin yolu (joblib, `predict_proba`, `classes_`→label map) yoktur.
4. GUI asset yolları A1.2 paketine yönlendirilmemiştir; A1.2 için ayrı bir paketleme klasörü/manifesti oluşturulmamıştır.
5. Protokol dışı happy eşik düzeltmesi kaldırılmamıştır.
6. Entegrasyon sonrası örnek bir test kaydıyla uçtan uca smoke test yapılmamıştır.

Bu eksiklerin hiçbiri bu incelemede uygulanmamıştır; GUI kodu ve model/scaler dosyaları değiştirilmemiştir.

## 3. Son karar ve sınırlar

- **Ana model adayı:** `A1_2/70_15_15/modeller/rf_70_15_15_matched_A1_2.joblib`.
- **Alternatif:** `A1_2/70_15_15/modeller/mlp_best_70_15_15_A1_2.keras`.
- **Karar gerekçesi:** A1.2, OOF'ta dört deneyin en iyisidir; RF 70/15/15 testinde MLP'den açıkça üstündür ve dağıtım maliyeti daha düşüktür. Ancak RF-MLP OOF farkı küçüktür ve istatistiksel üstünlük testi yapılmamıştır.
- **Kritik sınır:** 70/15/15 test kümesinin tamamı `TESS_YAF`'tır. Bu sonuçlar konuşmacıdan bağımsız genelleme başarısı olarak sunulamaz.
- **Paketleme durumu:** Artefakt seti hazır; çalışan GUI entegrasyonu için çıkarım adaptörü ve asset yönlendirmesi gerekir.
- **Bu çalışmada yapılmayan:** eğitim, model/scaler oluşturma/değiştirme, GUI değişikliği, mevcut sonuç değişikliği, 10-fold OOF veya 70/15/15 yeniden çalıştırma, commit/push.

### Okunan kaynaklar

- `A1_2/kfold/`: `oof_metrikler.csv`, `oof_sinif_f1.csv`, `kfold_per_fold.csv`, `kfold_ozet.csv`, `karsilastirma_A0_A1_2.csv`, `istatistik_genel.csv`, `istatistik_sinif_f1.csv`, `son_rapor_A1_2.md`, `cv_config.json`, `f0_imputasyon_degerleri.csv`.
- `A1_2/70_15_15/`: `son_rapor.md`, `artifact_manifest_A1_2.json`, `dogrulama_kontrolleri.*`, `f0_imputasyon_degerleri.csv`, `ysa/*.csv`, `karsilastirma/*.csv`, `modeller/*`, `4_ysa_A1_2.py`.
- `A1_2/ozellik_cikarimi_A1_2.py`, `A1_2/f0_ozellik_tanimlari.csv`, `A1_2/final_oznitelik_seti_A1_2.csv`.
- `A1_1/kfold/karsilastirma_A0_A1_1.md`, `A3/kfold/karsilastirma_A0_A3.md`, `A3/GUI_uygulama_notu.md`.
- `sonuclar_v2_dedup/`: `artifact_manifest_dedup.json`, `ysa/70_15_15_karsilastirma_metrikleri_dedup.csv`, `modeller/*`.
- `7_final_arayuz.py` (salt okunur), `sonuclar_ysa/*`, `README.md`.

Not: `A1_2/kfold/son_rapor_A1_2.md` metni 70/15/15'in yapılmadığını söyleyen eski bir durum satırı içerir. Mevcut sonuç değiştirilmediği için bu satır düzeltilmemiştir; 70/15/15 kanıtı `A1_2/70_15_15/` dizinindeki yeni rapor ve artefaktlardır.
