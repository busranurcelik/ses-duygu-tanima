# A1.2 — Konuşma Duygu Tanıma Sisteminde Temel Frekans (F0) Özelliklerinin Etkisinin İncelenmesi

Bu rapor, mevcut OOF tahminleri, bootstrap çıktıları ve doğrulama dosyalarından üretilmiştir. Eğitim, 10-fold CV, 70/15/15, model/scaler kaydı, commit veya push yapılmamıştır.

## 1. Çalışmanın Amacı

Bu deneyin amacı, konuşma duygu tanıma sisteminde kullanılan mevcut akustik özelliklere temel frekans (F0/pitch) tabanlı özellikler eklemenin sınıflandırma performansına etkisini incelemektir.

Dört duygu sınıfı ele alınmıştır:

- Angry
- Fear
- Happy
- Sad

A0 deneyinde kullanılan 184 özellikli yapı karşılaştırma tabanı olarak korunmuştur. A1.2 deneyinde bu yapıya sekiz F0 özelliği eklenerek özellik sayısı 192'ye çıkarılmıştır.

Temel araştırma sorusu, F0 özelliklerinin özellikle birbirine karıştırılması zor olan fear ve happy sınıflarının ayırt edilmesine katkı sağlayıp sağlamadığıdır.

## 2. Veri Seti

Beş farklı veri kaynağı kullanılmıştır:

| Veri kaynağı | Kayıt sayısı |
|---|---:|
| CREMA-D | 5.081 |
| TESS | 1.600 |
| RAVDESS | 759 |
| SAVEE | 240 |
| BNC | 100 |
| **Toplam** | **7.780** |

Veri hazırlama aşamasında yinelenen kayıtlar, etiket çakışması bulunan 2 kayıt (`REVIEW_LABEL`) ve kalite kontrolü nedeniyle ayrılan 17 kayıt (`REVIEW_QC`) değerlendirme dışında bırakılmıştır. Sonuçta 7.780 özgün kayıt, 7.780 tekil `source_id` ve 122 konuşmacı/grup kümesi kullanılmıştır.

Konuşmacı/grup bazlı ayrım, aynı gruba ait kayıtların eğitim ve test bölümlerine dağılmasını önlemek amacıyla korunmuştur.

## 3. Özellik Çıkarımı

A0 sistemindeki 184 özellik birebir korunmuş, bunlara sekiz temel frekans özelliği eklenmiştir. Sonuç veri seti 23.268 satır × 202 kolondur: 10 metadata + 184 A0 özelliği + 8 F0 özelliği.

F0 özellikleri `librosa.pyin` ile çıkarılmıştır. Sabit parametreler:

- Örnekleme frekansı: 22.050 Hz
- Minimum F0: 60 Hz
- Maksimum F0: 400 Hz
- `frame_length`: 2.048
- `hop_length`: 512

Eklenen sekiz özellik:

1. F0 mean
2. F0 median
3. F0 standard deviation (`ddof=0`)
4. F0 minimum
5. F0 maximum
6. F0 range
7. F0 slope
8. Voiced frame ratio

F0 slope, voiced karelerde gerçek zaman ekseninde `t = i × 512 / 22050` s kullanılarak `np.polyfit` eğimiyle hesaplanmıştır. Voiced frame ratio, geçerli voiced F0 karelerinin 2,5 s penceredeki **108 analiz karesine** oranıdır.

23.268 pYIN çağrısının tamamı hatasız tamamlanmıştır: 0 WAV okuma hatası, 0 SHA uyuşmazlığı, 0 eksik varyant, 0 pYIN exception.

## 4. Eksik F0 Değerlerinin Yönetimi

Bazı kayıtlarda geçerli F0 tespit edilememiştir. Toplam **1.606 fallback satırı** vardır; tamamı `no_valid_voiced` nedeniyledir. Kaynak dağılımı CREMA 1.586, RAVDESS 20'dir; varyant dağılımı `orig` 537, `noise` 540, `pitch_up` 529'dur.

Bu kayıtlar için F0 istatistiklerinin ham `0.0` sentinel olarak değerlendirilmesi yanıltıcı olacağından, aşağıdaki yedi sütun eksik değer olarak ele alınmıştır:

- Mean
- Median
- Standard deviation
- Minimum
- Maximum
- Range
- Slope

Eksik değerlerin doldurulmasında her fold için **yalnızca ilgili fold'un eğitim verisindeki fallback'siz satırlardan** hesaplanan medyan kullanılmıştır. Test verisinin istatistikleri eğitime sızmamıştır. `voiced_frame_ratio` sütunundaki gerçek sıfır değerleri korunmuş, impute edilmemiştir.

Bağımsız doğrulamada medyan hesaplamalarının yeniden üretimi arasındaki maksimum fark **2,84 × 10⁻¹⁴** olarak raporlanmıştır.

## 5. Deneysel Yöntem

A0 ve A1.2 deneylerinde aynı 10-fold grup bazlı çapraz doğrulama yapısı kullanılmıştır. Fold atamaları `manifest_v2_dedup_draft.csv` kaynağından gelmiş, `REVIEW_LABEL` kayıtları çıkarılmıştır.

Her fold için:

- Test kümesi ilgili fold'un özgün kayıtlarından (`orig`) oluşturulmuştur.
- Eğitim kümesi diğer dokuz fold'un özgün ve artırılmış kayıtlarından oluşturulmuştur.
- Veri artırma işlemleri yalnızca eğitim verisinde kullanılmıştır.
- Ölçeklendirme yalnızca ilgili fold'un eğitim verisine fit edilmiştir.
- F0 eksik değer doldurma medyanları yalnızca ilgili fold'un eğitim verisinden öğrenilmiştir.
- Seed'ler fold başına 43–52 arasında sabitlenmiştir.

RF ve MLP modelleri aynı fold atamaları, aynı seed'ler ve aynı değerlendirme protokolüyle karşılaştırılmıştır. OOF değerlendirmesinde 7.780 özgün kayıt için birer tahmin elde edilmiştir. Model/scaler kaydı yapılmamıştır.

## 6. Deney Sonuçları

### 6.1. Genel Performans

| Model | Deney | Accuracy | Macro-F1 |
|---|---|---:|---:|
| RF | A0 | 0,5860 | 0,5810 |
| RF | A1.2 | **0,5978** | **0,5922** |
| MLP | A0 | 0,5796 | 0,5755 |
| MLP | A1.2 | **0,5950** | **0,5912** |

A1.2, her iki modelde de accuracy ve macro-F1 değerlerini artırmıştır:

- RF: accuracy **+1,18 yüzde puan**, macro-F1 **+1,12 yüzde puan**.
- MLP: accuracy **+1,54 yüzde puan**, macro-F1 **+1,58 yüzde puan**.

### 6.2. Sınıf Bazlı Performans

| Model | Sınıf | A0 F1 | A1.2 F1 | Fark |
|---|---|---:|---:|---:|
| RF | Angry | 0,6265 | 0,6344 | +0,0079 |
| RF | Fear | 0,4606 | 0,4645 | +0,0039 |
| RF | Happy | 0,5182 | **0,5497** | **+0,0315** |
| RF | Sad | 0,7186 | 0,7200 | +0,0014 |
| MLP | Angry | 0,6343 | 0,6277 | −0,0066 |
| MLP | Fear | 0,4533 | **0,4848** | **+0,0315** |
| MLP | Happy | 0,5123 | **0,5421** | **+0,0299** |
| MLP | Sad | 0,7020 | 0,7103 | +0,0083 |

Her iki modelde de happy sınıfının F1 değeri yükselmiştir: RF'de +3,15 yüzde puan, MLP'de +2,99 yüzde puan. MLP'de fear F1 de +3,15 yüzde puan artmıştır; ancak bu artışın bootstrap güven aralığı sıfırı içermektedir (bkz. Bölüm 7.3).

Fear–happy karışıklığı devam etmektedir:

| Model | Yön | A0 | A1.2 | Fark |
|---|---|---:|---:|---:|
| RF | Fear → Happy | 394 | 402 | +8 |
| RF | Happy → Fear | 274 | 220 | −54 |
| MLP | Fear → Happy | 587 | 458 | −129 |
| MLP | Happy → Fear | 215 | 225 | +10 |

Dolayısıyla F0 özelliklerinin fear–happy ayrımını tamamen çözdüğü söylenemez. RF'de `happy→fear` karışıklığı azalırken `fear→happy` hafif artmış; MLP'de ise `fear→happy` belirgin biçimde azalmış, `happy→fear` hafif artmıştır.

### 6.3. Kaynak Bazlı Sonuçlar

A0 ve A1.2 değerleri `A1_2/kfold/karsilastirma_A0_A1_2.csv` dosyasından alınmıştır.

**RF**

| Kaynak | A0 accuracy | A1.2 accuracy | Δ accuracy | A0 macro-F1 | A1.2 macro-F1 | Δ macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| BNC | 0,3500 | 0,3600 | +0,0100 | 0,2568 | 0,2572 | +0,0004 |
| CREMA-D | 0,5869 | 0,5971 | +0,0102 | 0,5751 | 0,5856 | +0,0105 |
| RAVDESS | 0,5033 | 0,5020 | −0,0013 | 0,4971 | 0,4949 | −0,0022 |
| SAVEE | 0,5125 | 0,4958 | −0,0167 | 0,5088 | 0,4887 | −0,0201 |
| TESS | 0,6481 | 0,6756 | +0,0275 | 0,6445 | 0,6743 | +0,0298 |

**MLP**

| Kaynak | A0 accuracy | A1.2 accuracy | Δ accuracy | A0 macro-F1 | A1.2 macro-F1 | Δ macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| BNC | 0,3300 | 0,3200 | −0,0100 | 0,2159 | 0,2092 | −0,0068 |
| CREMA-D | 0,5934 | 0,6113 | +0,0179 | 0,5846 | 0,6051 | +0,0205 |
| RAVDESS | 0,4980 | 0,5310 | +0,0329 | 0,5031 | 0,5332 | +0,0301 |
| SAVEE | 0,4500 | 0,5250 | +0,0750 | 0,4503 | 0,5214 | +0,0711 |
| TESS | 0,6094 | 0,6013 | −0,0081 | 0,6024 | 0,5913 | −0,0110 |

Sonuçlar kaynaklar arasında farklılık göstermektedir. MLP'de SAVEE ve RAVDESS belirgin iyileşirken TESS hafif gerilemiştir. RF'de TESS iyileşirken SAVEE ve RAVDESS hafif gerilemiştir. BNC yalnızca 100 kayıt ve tek konuşmacı/grup içerdiğinden bu kaynak için genelleme yorumu sınırlıdır. Kaynak bazlı sonuçlar tek başına bağımsız veri seti başarısını kanıtlamaz.

### 6.4. Fold Tutarlılığı

Standart sapmalar `A1_2/kfold/karsilastirma_A0_A1_2.csv` içindeki population std (`ddof=0`) değerleridir.

| Model | Deney | Accuracy ort. ± std | Macro-F1 ort. ± std |
|---|---|---:|---:|
| RF | A0 | 0,5795 ± 0,0445 | 0,5654 ± 0,0401 |
| RF | A1.2 | 0,5901 ± 0,0496 | 0,5751 ± 0,0424 |
| MLP | A0 | 0,5756 ± 0,0406 | 0,5619 ± 0,0344 |
| MLP | A1.2 | 0,5927 ± 0,0434 | 0,5811 ± 0,0417 |

RF modelinde 10 fold'un 8'inde, MLP modelinde ise 7'sinde A1.2 sonucu A0'dan yüksek olmuştur. Bununla birlikte fold standart sapmaları hafif artmıştır; ortalama performans artışına rağmen fold'lar arası değişkenlik dikkate alınmalıdır.

## 7. İstatistiksel Karşılaştırma

### 7.1. Yöntem

- **Eşleştirme anahtarı:** `source_id` + `y_true`. 7.780 kaydın tamamı eşleşmiş; `group_id`, `source` ve `fold` değerlerinin birebir aynı olduğu doğrulanmıştır.
- **Bootstrap:** eşleştirilmiş, kaynak tabakalı küme bootstrap. Küme `group_id`, tabaka `source`'tur.
- **Örnekleme:** her kaynak için o kaynağa ait küme sayısı korunarak `with replacement` örnekleme yapılmış; aynı örneklenen satırlar A0 ve A1.2 için birlikte kullanılmıştır. Küme boyutları farklı olduğundan örneklenen satır sayısı 7.742–7.804 aralığında değişmiştir.
- **Tekrar sayısı:** 10.000; **sabit seed:** 42.
- **Güven aralığı:** yüzdelik (percentile) %95 aralık, [2,5; 97,5].
- **Kümeler:** BNC=1, CREMA=91, RAVDESS=24, SAVEE=4, TESS=2; toplam 122 küme.

Bu tasarım, aynı konuşmacı/gruba ait kayıtların birlikte örneklenmesini sağlayarak grup bağımlılığını korumuştur. Ancak BNC tek küme, SAVEE 4 küme ve TESS 2 küme içerdiğinden bu kaynaklarda bootstrap belirsizlik değerlendirmesi sınırlıdır.

### 7.2. Genel Metriklerin İstatistiksel Karşılaştırması

| Model | Metrik | A0 | A1.2 | Fark | %95 güven aralığı | Sıfır içeriyor |
|---|---|---:|---:|---:|---|---|
| RF | Accuracy | 0,5860 | 0,5978 | +0,0118 | [+0,0036; +0,0200] | Hayır |
| RF | Macro-F1 | 0,5810 | 0,5922 | +0,0112 | [+0,0023; +0,0195] | Hayır |
| MLP | Accuracy | 0,5796 | 0,5950 | +0,0154 | [+0,0062; +0,0248] | Hayır |
| MLP | Macro-F1 | 0,5755 | 0,5912 | +0,0158 | [+0,0060; +0,0269] | Hayır |

Dört genel metriğin güven aralığı sıfırı içermemiştir. Bu bulgu, kullanılan bootstrap yöntemi ve örnekleme tasarımı kapsamında A1.2 lehine performans artışını desteklemektedir.

### 7.3. Sınıf Bazlı F1 Farklarının Tamamı

A1.2 − A0 farkları ve %95 bootstrap güven aralıkları aşağıdadır:

| Model | Sınıf | A0 F1 | A1.2 F1 | Fark | %95 güven aralığı | Sıfır içeriyor |
|---|---|---:|---:|---:|---|---|
| RF | Angry | 0,6265 | 0,6344 | +0,0079 | [−0,0027; +0,0178] | Evet |
| RF | Fear | 0,4606 | 0,4645 | +0,0039 | [−0,0105; +0,0172] | Evet |
| RF | Happy | 0,5182 | 0,5497 | +0,0315 | [+0,0069; +0,0556] | **Hayır** |
| RF | Sad | 0,7186 | 0,7200 | +0,0014 | [−0,0074; +0,0103] | Evet |
| MLP | Angry | 0,6343 | 0,6277 | −0,0066 | [−0,0245; +0,0146] | Evet |
| MLP | Fear | 0,4533 | 0,4848 | +0,0315 | [−0,0068; +0,0780] | Evet |
| MLP | Happy | 0,5123 | 0,5421 | +0,0299 | [+0,0122; +0,0468] | **Hayır** |
| MLP | Sad | 0,7020 | 0,7103 | +0,0083 | [−0,0017; +0,0184] | Evet |

Sınıf bazında yalnızca happy F1 artışı iki modelde de sıfır içermeyen güven aralığına sahiptir. MLP fear F1 farkı pozitif olmakla birlikte güven aralığı sıfırı içermektedir; bu nedenle fear sınıfındaki iyileşme istatistiksel olarak kesinleşmiş kabul edilmemelidir.

### 7.4. McNemar Testi (Yardımcı Analiz)

Eşleştirilmiş doğruluk kararları üzerinde exact binomial (two-sided) McNemar testi uygulanmıştır. `n01` A0 doğru / A1.2 yanlış, `n10` A0 yanlış / A1.2 doğru kayıtlarını gösterir.

| Model | n01 | n10 | Uyuşmayan toplam | Exact p |
|---|---:|---:|---:|---:|
| RF | 384 | 476 | 860 | 0,001898 |
| MLP | 474 | 594 | 1.068 | 0,000268 |

Her iki modelde de `n10 > n01` yönü A1.2 lehinedir. Ancak McNemar testi grup/konuşmacı bağımlılığını modellemediğinden **yalnızca yardımcı analizdir**; birincil kanıt Bölüm 7.2 ve 7.3'teki cluster bootstrap güven aralıklarıdır.

### 7.5. Çoklu Karşılaştırma

4 genel metrik ve 8 sınıf bazlı F1 farkı için çoklu karşılaştırma düzeltmesi (örneğin Bonferroni veya FDR) **uygulanmamıştır**. %95 güven aralıkları tek tek yorumlanmıştır. Bu nedenle özellikle sınıf bazlı sonuçlar ve aynı veri üzerinde yapılan çok sayıda karşılaştırma birlikte değerlendirilmelidir.

## 8. A0, A1.1 ve A3 ile Karşılaştırma

A1.1 ve A3 değerleri kendi `kfold/karsilastirma_oof_metrikler.csv` dosyalarından; A1.2 değerleri `A1_2/kfold/karsilastirma_A0_A1_2.csv` dosyasından alınmıştır. Dört deney de 7.780 canonical kayıt üzerinde 10-fold OOF değerlendirmesi içermektedir.

| Model | Metrik | A0 | A1.1 | A3 | A1.2 | En yüksek |
|---|---|---:|---:|---:|---:|---|
| RF | Accuracy | 0,5860 | 0,5766 | 0,5811 | **0,5978** | A1.2 |
| RF | Macro-F1 | 0,5810 | 0,5699 | 0,5764 | **0,5922** | A1.2 |
| MLP | Accuracy | 0,5796 | 0,5630 | 0,5815 | **0,5950** | A1.2 |
| MLP | Macro-F1 | 0,5755 | 0,5591 | 0,5769 | **0,5912** | A1.2 |

Doğrulanan bu tabloya göre A1.2, hem RF hem MLP için accuracy ve macro-F1 değerlerinin tamamında A0, A1.1 ve A3'ten yüksektir. A0 karşılaştırma tabanı olarak korunmalıdır.

## 9. Sınırlılıklar

1. Veri kaynakları farklı kayıt koşulları, konuşmacılar ve veri dağılımları içermektedir.
2. Bazı kaynaklarda konuşmacı/grup sayısı düşüktür: BNC 1, TESS 2, SAVEE 4 küme. Bu durum kaynak bazlı bootstrap belirsizliğinin sınırlı değerlendirilmesine yol açar.
3. Fear ve happy sınıfları arasındaki karışıklık devam etmektedir.
4. F0 çıkarımı 1.606 kayıtta geçerli temel frekans tespit edememiştir; tamamı `no_valid_voiced` nedeniyledir ve ağırlıklı olarak CREMA kaynaklıdır.
5. Bootstrap grup bağımlılığını koruyacak şekilde tasarlanmış olsa da az sayıda küme içeren kaynaklarda belirsizlik tam olarak temsil edilemeyebilir.
6. 4 genel metrik ve 8 sınıf metriği için çoklu karşılaştırma düzeltmesi yapılmamıştır.
7. Sonuçlar aynı veri havuzundaki çapraz doğrulama değerlendirmesine dayanmaktadır; tamamen bağımsız bir veri setinde doğrulama yapılmış değildir.
8. Kaynak bazlı sonuçlar tek başına bağımsız veri seti başarısını kanıtlamaz.
9. 70/15/15 değerlendirmesi gerçekleştirilmemiştir.

## 10. Sonuç ve Değerlendirme

A1.2 deneyinde mevcut 184 akustik özelliğe sekiz temel frekans özelliği eklenmiştir. 10-fold grup bazlı çapraz doğrulama sonucunda hem RF hem MLP modellerinde accuracy ve macro-F1 değerlerinde artış gözlenmiştir.

Eşleştirilmiş, kaynak tabakalı küme bootstrap karşılaştırmasında dört genel performans metriğinin %95 güven aralıklarının sıfırı içermemesi, kullanılan tasarım kapsamında A1.2 lehine destekleyici bir bulgudur. Sınıf bazında happy F1 artışı iki modelde de sıfır içermeyen aralıklarla desteklenmiştir. Fear sınıfında MLP'de olumlu bir eğilim görülmekle birlikte güven aralığı sıfırı içerdiğinden bu iyileşme kesinleşmiş sayılmaz.

Doğrulanan karşılaştırma tablosu göre A1.2, RF ve MLP için accuracy ve macro-F1'de A0, A1.1 ve A3'ten yüksektir. Bu nedenle A1.2, mevcut deneyler arasında en güçlü iyileştirme adayı olarak değerlendirilmiştir. A0 karşılaştırma tabanı olarak korunmalıdır.

Bununla birlikte sonuçların bağımsız veri kaynaklarında doğrulanması, konuşmacıdan bağımsız genelleme başarısının ayrıca değerlendirilmesi ve 70/15/15 protokolünün uygulanması gerekmektedir.

**Genel değerlendirme:** Temel frekans özelliklerinin sisteme eklenmesi, mevcut veri ve değerlendirme protokolü kapsamında performansı artırmıştır. Bu iyileşmenin özellikle happy sınıfında belirgin olduğu görülmüştür.

## Kaynak Dosyalar

- `A1_2/kfold/oof_tahminler.csv`, `oof_metrikler.csv`, `oof_kaynak_bazli.csv`, `oof_karisiklik.csv`
- `A1_2/kfold/karsilastirma_A0_A1_2.csv`, `kfold_ozet.csv`
- `A1_2/kfold/istatistik_genel.csv`, `istatistik_sinif_f1.csv`, `istatistik_mcnemar.csv`, `istatistik_konfig.json`, `istatistik_kontrolleri.csv`, `istatistiksel_karsilastirma_raporu.md`
- `A1_2/ozellik_cikarimi_raporu.md`, `f0_ozellik_tanimlari.csv`, `kaynak_sayilari.csv`, `veri_hazirlama_raporu.md`
- `A1_1/kfold/karsilastirma_oof_metrikler.csv`
- `A3/kfold/karsilastirma_oof_metrikler.csv`

Bu rapor dışında mevcut deney dosyaları ve sonuçları değiştirilmedi; eğitim, CV, 70/15/15, commit ve push yapılmadı.
