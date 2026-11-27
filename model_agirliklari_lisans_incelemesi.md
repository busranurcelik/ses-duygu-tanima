# A1.2 Model Ağırlıkları Lisans İncelemesi (RF-B / MLP)

Tarih: 2026-11-27\
Kapsam: `ses_duygu_projesi` içindeki A1.2 RF-B ve MLP model dosyalarının ve bunların dayandığı veri kaynaklarının lisans/yeniden dağıtım durumu.\
Bu rapor bir **hukuki görüş değildir**. Lisans metinlerinin yorumu, özellikle eğitilmiş model ağırlıklarının veri lisansları karşısındaki durumu, yetkili bir hukukçu tarafından değerlendirilmelidir.

> **Uygulanan kural:** Lisansı veya dağıtım izni belirsiz olan hiçbir dosya `PUBLIC` sayılmadı. RF-B, MLP ve scaler dosyaları `LICENSE_REVIEW` olarak tutuldu. Bu rapor hiçbir lisansın ağırlıklara nasıl uygulanacağına dair kesin hüküm vermez; belirsiz noktalar açıkça belirsiz olarak işaretlenmiştir.

---

## 1. Yöntem ve kaynaklar

### 1.1 Yerel olarak incelenen dosyalar

| Dosya | İnceleme amacı |
|---|---|
| `A1_2/70_15_15/modeller/` (4 dosya) | RF-B, scaler, best/final MLP artefaktlarının envanteri |
| `A1_2/70_15_15/artifact_manifest_A1_2.json` | Model dosya adları, SHA-256 değerleri, eğitim protokolü, sürümler |
| `A1_2/kfold/son_rapor_A1_2.md` | Beş veri kaynağı, kayıt sayıları, OOF protokolü ve sonuçlar |
| `A1_2/70_15_15/son_rapor.md` | 70/15/15 protokolü, test kapsamı, metrikler, sızıntı kontrolleri |
| `github_yayin_oncesi_denetim.md` | Önceki sınıflandırma, SAVEE/BNC durumu, kişisel veri bulguları |
| `A1_2/dis_gercek_ses_testi/lisans_durumu.csv` | Kaynak bazlı lisans/rıza kayıtları |
| `A1_2/dis_gercek_ses_testi/cakisma_ve_dogrulama_raporu.md` | BNC/SAVEE ve çakışma denetimi notları |

Model dosyalarının ikili içeriği açılmadı, değiştirilmedi veya yeniden eğitilmedi.

### 1.2 Birincil lisans metinleri (2026-11-27'de okundu)

| Kaynak | Belirtilen lisans | Birincil adres |
|---|---|---|
| CREMA-D | Veritabanı: ODbL 1.0; bireysel içerikler: DbCL 1.0 | `https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/LICENSE.txt`; `https://opendatacommons.org/licenses/odbl/1-0/`; `https://opendatacommons.org/licenses/dbcl/1-0/` |
| RAVDESS | CC BY-NC-SA 4.0 | Zenodo kayıt 1188976: `https://zenodo.org/records/1188976`; yasa metni: `https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en` |
| TESS | CC BY-NC 4.0 | `https://doi.org/10.5683/SP2/E8H2MF`; yasa metni: `https://creativecommons.org/licenses/by-nc/4.0/legalcode.en` |
| SAVEE | Yetkili lisans/rıza metni bulunamadı | Yerel kayıt ve erişilebilir kurumsal adresler doğrulanamadı; üçüncü taraf repolar hak kaynağı sayılmadı |
| BNC (yerel kayıt) | Yayın rızası belgelenmedi | `A1_2/dis_gercek_ses_testi/lisans_durumu.csv`, `cakisma_ve_dogrulama_raporu.md` |

**İkincil, bağlayıcı olmayan rehber:** Creative Commons'un "Using CC-Licensed Works for AI Training" sayfası (`https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/`), yukarıdaki yasa metinlerinin yerine geçmez. CC bu konuda "copyright law to AI training is complex" diyerek yargı alanları arasındaki farklara işaret eder; sunduğu pratik rehberi "likely to lead to overcompliance" (aşırı uyum) olarak nitelendirir. Bu nedenle aşağıda yalnızca tartışma bağlamı olarak kullanılmıştır, kesin sonuç kaynağı değildir.

---

## 2. Genel çerçeve: kod, veri ve ağırlık ayrımı

1. **Kod lisansı ayrıdır.** Depoda proje geneli için bir `LICENSE` dosyası yoktur. Kaynak kod lisansı seçilmeden önce veri lisanslarının koda otomatik uygulandığı varsayılmamalıdır.
2. **Veri lisansı, model ağırlıklarına otomatik olarak uygulanmaz; ancak ağırlıkların dağıtımı için de otomatik bir izin doğmaz.** İki yönlü belirsizlik söz konusudur: lisansın ağırlıklara uygulanıp uygulanmayacağı ve uygulanıyorsa hangi koşulların doğacağı açıkça düzenlenmemiştir.
3. **İndirilebilirlik ≠ yeniden dağıtım izni.** Bir veri setine erişilebilir olması, onu GitHub'da yeniden yayınlama veya onunla eğitilmiş ağırlıkları paylaşma hakkı vermez.
4. **Eğitim/araştırma kullanımı ile yayın ayrıdır.** Lisans metinlerinde eğitim amaçlı kullanım ile eğitilmiş çıktının kamuya sunulması aynı şekilde ele alınmaz; bazı koşullar yalnızca "Share" / "Publicly Convey" anında gündeme gelir.
5. **CC lisansları telif ve ilgili hakları kapsar; kişilik haklarını kapsamaz.** CC BY-NC(-SA) 4.0 §2(b) metni, mahremiyet, tanıtım ve benzeri kişilik haklarının lisanslanmadığını, yalnızca hak sahibinin lehine sınırlı ölçüde feragat edildiğini belirtir. BNC gibi tek konuşmacılı kayıtlarda rıza belgesi yokluğu bu nedenle ayrı bir risktir.
6. **İstisna ve sınırlamalar.** CC metinleri §2(a)(2) ve §8(a) ile,Telif hakkı istisnası/zorunlu lisans uygulanan kullanımlarda lisans koşullarının devreye girmeyebileceğini söyler. Bu rapor hiçbir yargı alanındaki istisna/zorunlu lisans rejimini incelemez; bu konuda kesinlik iddiası yoktur.

---

## 3. Veri kaynaklarının lisans boyutları

### 3.1 CREMA-D — ODbL 1.0 / DbCL 1.0

CREMA-D `LICENSE.txt` metni, veritabanını ODbL 1.0, bireysel içeriklerdeki hakları DbCL 1.0 altında sunduğunu belirtir.

| Boyut | Birincil metinde yer alan durum |
|---|---|
| Eğitim / araştırma | ODbL §2: kullanım hakları "explicitly include commercial use", ticarî olmayan bir kısıt yoktur. DbCL §2.1 de içerikler üzerindeki telif hakları için dünya çapında, ücretsiz, ticari kullanımı açıkça içeren bir lisans verir. |
| Türev veri / model ağırlığı | ODbL §1, "Derivative Database"i veritabanına dayalı çeviri, uyarlama, düzenleme, değiştirme veya içeriğin tamamının/önemli kısmının yeni bir veritabanına çıkarılması olarak tanımlar. §4.5(b), bir "Produced Work" oluşturmanın kendi başına "Derivative Database" yaratmadığını söyler. **Eğitilmiş bir sınıflandırıcının bu tanımlardan hangisine girdiği, ağırlıkların "önemli kısım" içerip içermediği veya salt çıktı/üretim eseri sayılıp sayılmayacağı lisans metninde açıkça çözülmemiştir.** Bu rapor bu soruna kesin cevap vermez. |
| Ticari kısıt | ODbL/DbCL'de NC benzeri bir kısıt yoktur; ancak ODbL'in §4 koşulları (bildirim, share-alike, erişim) kamuya açık kullanımda devreye girebilir. |
| Yeniden dağıtım | ODbL §4.2, veritabanını veya türev veritabanını "Publicly Convey" eden tarafın lisans metnini/URI'sini ve bildirimleri korumasını ister. §4.6, türev veritabanı veya ondan üretilen Produced Work kamuya açık kullanıldığında, alıcılara türev veritabanının makine okunabilir kopyasını veya değişiklikleri/yöntemi sunma yükümlülüğünden söz eder. **İkili model dosyasının bu kapsama girip girmediği belirsizdir.** |
| Atıf / ShareAlike | ODbL §4.3, Produced Work'ün kamuya açık kullanımunda kaynağın veritabanından alındığına dair bildirim ister; §4.4, kamuya açık kullanılan türev veritabanlarını aynı/uyumlu lisansa bağlar. DbCL §2.2, ODbL'e uyumu şart koşar. |

**Ağırlık etkisi:** CREMA-D tek başına "ticari kullanıma izin veriyor" diye özetlenebilir; ancak bu, karma veriyle eğitilmiş ağırlıkların dağıtım izni anlamına gelmez. Kullanım koşullarının ağırlıklara nasıl yansıyacağı yukarıdaki belirsizlik nedeniyle açık değildir.

### 3.2 RAVDESS — CC BY-NC-SA 4.0

Zenodo kayıt 1188976 lisans alanı CC BY-NC-SA 4.0 olarak doğrulandı.

| Boyut | Birincil metinde yer alan durum |
|---|---|
| Eğitim / araştırma | §2(a)(1)-(2), Licensed Material ve Adapted Material'in çoğaltılması/paylaşılması hakkını **"for NonCommercial purposes only"** ile sınırlar. §2(a)(2), yalnızca teknik değişikliklerin Adapted Material oluşturmadığını belirtir. |
| Türev veri / model ağırlığı | §1, Adapted Material'i lisanslı materyale dayalı, telif izni gerektirecek şekilde çevrilmiş/değiştirilmiş/dönüştürülmüş materyal olarak tanımlar. **Ses örneklerinden istatistiksel özellik öğrenmenin bu tanımı karşılayıp karşılamadığı metinde açıkça yazmaz.** CC'nin ikincil rehberi, ihtiyatlı yaklaşımda modelin kendisinin de SA kapsamında değerlendirilebileceğini söyler; ancak aynı rehber konunun karmaşık ve yargı alanına bağlı olduğunu vurgular. Bu rapor kesin bir sınıflandırma yapmaz. |
| Ticari kısıt | "NonCommercial", §1'de "not primarily intended for or directed towards commercial advantage or monetary compensation" olarak tanımlanır. §2(b), izin verilmeyen kullanımlarda telif ücreti toplama hakkını açıkça saklı tutar. Bir portföy yayınının bu tanıma girip girmeyeceği ve ileride ticarî kullanım olasılığı olguya bağlıdır; burada karar verilmez. |
| Yeniden dağıtım | §2(a), paylaşımı NC amaçlarla sınırlar; §3(a) atıf koşulları "Share" durumunda devreye girer. Ham WAV'lerin yeniden dağıtımı için ayrıca izin yoktur; ağırlıklar için de açık bir dağıtım izni bulunmaz. |
| Atıf / ShareAlike | §3(a), yaratıcı kimliği, telif/licans bildirimlerini, URI'yi, değişiklik göstergesini korumayı ister. §3(b), paylaşılan Adapted Material için aynı License Elements'a sahip bir CC lisansı (Adapter's License) zorunluluğunu düzenler. §2(b), mahremiyet/tanıtım gibi kişilik haklarının lisans dışında olduğunu belirtir. |

**Ağırlık etkisi:** NC ve SA koşulları, ağırlıkların Adapted Material sayılması halinde gündeme gelebilecek ek yükümlülükler oluşturur. Bu ihtimal kesinleştirilmediği için ağırlıklar `LICENSE_REVIEW` kalır; "NC olduğu için kesin yasaktır" veya "SA kesin uygulanır" biçiminde hüküm verilmez.

### 3.3 TESS — CC BY-NC 4.0

University of Toronto Borealis kayıt sayfasında lisans CC BY-NC 4.0 olarak doğrulandı.

| Boyut | Birincil metinde yer alan durum |
|---|---|
| Eğitim / araştırma | §2(a)(1)-(2), RAVDESS ile aynı biçimde çoğaltma/paylaşımı **NonCommercial** amaçlarla sınırlar. |
| Türev veri / model ağırlığı | §1'deki Adapted Material tanımı RAVDESS ile aynıdır. BY-NC 4.0'da ShareAlike koşulu yoktur; §3'te yalnızca, paylaşılan Adapted Material için uygulanan lisansın alıcıların bu lisans koşullarına uymasını engellememesi gerektiği belirtilir. **Ağırlıkların Adapted Material olup olmadığı sorusu burada da açık değildir.** |
| Ticari kısıt | NC tanımı ve §2(b)'deki telif ücreti rezervasyonu RAVDESS ile aynıdır. |
| Yeniden dağıtım | Ham WAV'ler için ek izin yoktur. Ağırlıklar için açık izin yoktur. |
| Atıf / ShareAlike | §3(a) atıf zorunluluğu vardır; **SA zorunluluğu yoktur**. Ancak atıfın ağırlıklara nasıl uygulanacağı (model kartı mı, README mi, yoksa gerekli olup olmadığı) metinde özel olarak düzenlenmemiştir. |

**Ağırlık etkisi:** A1.2 70/15/15 test kümesinin tamamı `TESS_YAF` konuşmacısına aittir (800 özgün kayıt). Metrikler bu NC lisanslı kaynaktan türetilmiştir. Toplu metriklerin yayını, ham ses veya kayıt bazlı eşleştirilebilir tahminlerin yayınıyla aynı değildir; yine de TESS atıf notu korunmalıdır. Ağırlık dağıtımı için buradan bir izin çıkarılmaz.

### 3.4 SAVEE — DOGRULANAMADI

- `lisans_durumu.csv` ve denetim raporunda durum **DOGRULANAMADI**, kapsam "Belirtilmedi", kanıt "-", not "Yerelde lisans/rıza belgesi yok; kurumsal kaynak adresleri 404 dönüyor. Üçüncü taraf repolar hak kaynağı sayılmadı."
- A1.2 veri setinde 240 SAVEE kaydı eğitim/evaluasyon havuzunda yer alır (`A1_2/kfold/son_rapor_A1_2.md`, Bölüm 2 ve 6.3).
- **Eğitim/araştırma:** Bu proje içindeki yerel kullanımın hukuki dayanağı belgelenememiştir; "internette bulundu" bir lisans/rıza belgesi değildir.
- **Türev veri / model ağırlığı:** İzin belgelenmediği için ağırlıklara yansıyacak bir hak da doğrulanamaz. Bu, ağırlıkların kesin yasak olduğu anlamına gelmez; **belirsizlik nedeniyle dağıtım yapılmayacağı** anlamına gelir.
- **Ticari / yeniden dağıtım / atıf:** Lisans metni olmadığından bu başlıklar için doğrulanabilir bir koşul yoktur. Üçüncü taraf beyanları hak kaynağı kabul edilmez.

**Ağırlık etkisi:** SAVEE, karma eğitim havuzunun bir parçasıdır. Diğer dört kaynak için izin alınsa bile SAVEE belirsizliği tek başına ağırlık yayınını engeller.

### 3.5 BNC (yerel kayıt) — yayın rızası belgelenmedi

- 100 kayıt, tek konuşmacı/grup `BNC_SPEAKER_01`; `lisans_durumu.csv` durumu **HARIC**, not: rıza belgelenmedi, değerlendirmeye alınmaz.
- Aynı 100 kayıt A1.2 OOF veri setinde yer alır; kaynak bazlı sonuçlar raporda BNC için ayrıca sunulur.
- **Eğitim/araştırma:** Yerel değerlendirmede kullanılmış olsa da yayın rızası belgelenmediği için kamuya açık dağıtım dayanağı yoktur.
- **Türev veri / model ağırlığı:** Tek konuşmacıya ait ses özelliklerinin model parametrelerine yansıma olasılığı, lisansın ötesinde **mahremiyet / kişilik hakkı** riski yaratır. CC metinlerinin §2(b)'de kişilik haklarını lisanslamadığına dair notu bu riski azaltmaz; burada zaten geçerli bir lisans/rıza yoktur.
- **Ticari / yeniden dağıtım / atıf:** Rıza olmadığından bu başlıklarda izin doğrulanamaz.
- Mevcut denetimde BNC ham WAV'leri dış test aday havuzundan çıkarılmış ve "kesinlikle yayınlanmamalı" olarak sınıflandırılmıştır.

**Ağırlık etkisi:** BNC, ağırlık yayını için en yüksek riskli bileşendir. Modelin BNC kayıtlarıyla eğitilmiş olması, ağırlıkların "yalnızca genel istatistik" olduğu varsayımıyla giderilemez; bu varsayım doğrulanmamıştır.

---

## 4. Artefakt bazlı değerlendirme

### 4.1 Envanter

| Artefakt | Manifest / rapordaki kanıt | Sınıflandırma |
|---|---|---|
| `modeller/rf_70_15_15_matched_A1_2.joblib` | SHA-256 `e3b63152…be15d`; RF-B 200 ağaç, `class_weight='balanced'`, seed 42; 192 özellik; scaler ile aynı eğitim akışı | `LICENSE_REVIEW` |
| `modeller/scaler_A1_2.joblib` | SHA-256 `83515262…97b10`; yalnızca 70/15/15 train'e fit edildi | `LICENSE_REVIEW` |
| `modeller/mlp_best_70_15_15_A1_2.keras` | SHA-256 `be6e282a…59f8e`; 35/200 epoch, en iyi epoch 25; aynı scaler | `LICENSE_REVIEW` |
| `modeller/mlp_final_70_15_15_A1_2.keras` | Dizinde mevcut; **manifestte SHA-256 kaydı yok** | `LICENSE_REVIEW` |

`f0_imputasyon_degerleri.csv` ve `ysa/70_15_15_label_map_A1_2.json` ham ses veya model ağırlığı değil, toplu/eşik değerlerdir; önceki denetimde `PUBLIC` tarafında tutulmuştur. Bu dosyaların yayını, model ağırlıklarının yayını için bir izin veya emsal oluşturmaz.

### 4.2 RF-B (ana model)

- A1.2 RF-B, GUI'nin kullandığı ana modeldir; arayüz `predict_proba` çıktısını doğrudan kullanır, ek bir "happy bias" düzeltmesi uygulanmaz.
- 70/15/15 sonucu: accuracy **0,6562**, macro-F1 **0,6252**; ancak test kümesinin tamamı `TESS_YAF` olduğundan bu, konuşmacıdan bağımsız genelleme başarısı değildir. OOF sonuçları: **0,5978 / 0,5922**.
- `joblib` dosyası, eğitim satırlarından öğrenilmiş bölme eşikleri ve yaprak değerleri içeren serileştirilmiş bir tahmincidir. Bu değerler toplu bir metrik tablosu değil, veriden türetilmiş parametrelerdir. **Veri kayıtlarının birebir kopyası olup olmadığı veya üyelik çıkarımına (membership inference) izin verip vermediği bu incelemede test edilmemiştir; bu nedenle "yalnızca agregattır, risksizdir" denemez.**
- Eğitim havuzu CREMA-D, TESS, RAVDESS, SAVEE ve BNC karışımıdır. SAVEE lisansı ve BNC rızası doğrulanmadığı için, CC lisanslarının ağırlıklara uygulanması sorunu çözülmese bile RF-B yayını engellenir.
- **Karar:** Açık yazılı dağıtım izni ve hukuki değerlendirme olmadan `LICENSE_REVIEW`; GitHub'a yüklenmez.

### 4.3 MLP (alternatif model)

- `mlp_best_70_15_15_A1_2.keras` ve `mlp_final_70_15_15_A1_2.keras`, aynı veri havuzu ve aynı scaler ile eğitilmiştir; GUI'ye entegre değildir, yalnızca karşılaştırma modelidir.
- 70/15/15: accuracy **0,6075**, macro-F1 **0,5548**; OOF: **0,5950 / 0,5912**. Eğitim 35 epoch'ta durmuş, en iyi val_accuracy 25. epoch'ta kaydedilmiştir.
- Sinir ağı ağırlıkları, eğitim örnekleri üzerinde optimize edilmiş sayısal parametrelerdir. RF-B için geçerli olan karma veri kaynaklı SAVEE/BNC engeli ve CC BY-NC(-SA) belirsizliği aynen geçerlidir.
- Ek belirsizlik: `mlp_final_70_15_15_A1_2.keras` dosyasının manifestte SHA-256 kaydı bulunmaz; proverans/bütünlük belgelendirmesi best model kadar eksiksiz değildir.
- Daha eski `sonuclar_ysa/` MLP artefaktları Git geçmişinde ve HEAD içinde mevcuttur; geçmişten kaldırmak history rewrite gerektirir ve bu çalışmanın kapsamı dışındadır.
- **Karar:** RF-B ile aynı gerekçelerle `LICENSE_REVIEW`; GitHub'a yüklenmez.

---

## 5. Açıkça belirsiz bırakılan noktalar

Bu rapor aşağıdaki sorulara kesin cevap vermez:

1. Eğitilmiş RF-B/MLP ağırlıkları CC BY-NC-SA 4.0 veya CC BY-NC 4.0 anlamında **Adapted Material** midir, yoksa lisanslı materyalin kopyasını/uyarlamasını içermeyen bir çıktı mıdır? Birincil metinler bunu açıkça düzenlemez.
2. Ağırlık dosyasının paylaşılması, CC metinlerindeki **"Share"** veya ODbL'deki **"Publicly Convey"** tanımına girer mi? Tanımlar kapsamlıdır ancak ikili model parametrelerini açıkça ele almaz.
3. ShareAlike koşulu, veri kümesi değil **yalnızca eğitilmiş model** paylaşıldığında tetiklenir mi? CC metninde bu senaryoya ilişkin açık bir hüküm yoktur; CC'nin kendi rehberi ihtiyatlı yaklaşımı önerir, bağlayıcı değildir.
4. ODbL açısından model bir **"Produced Work"** mı, yoksa önemli miktarda içeriğin çıkarıldığı bir **"Derivative Database"** mı? §4.5(b) Produced Work için türev veritabanı oluşmadığını söyler; ancak bu dosyaların somut sınıflandırması yapılmamıştır.
5. Ağırlıklar CREMA-D/RAVDESS/TESS/SAVEE/BNC kayıtlarını ezberlemiş veya üyelik çıkarımına izin verecek bilgi taşıyor mu? Bu incelemede model içeriği analiz edilmedi; varsayım yok.
6. Portföy amaçlı bir GitHub yayını CC tanımındaki **NonCommercial** kapsamına girer mi? Gelecekteki ticarî kullanım olasılığı bilinmemektedir.
7. Uygulanabilir telif istisnası/zorunlu lisans rejimi var mı? Hiçbir yargı alanı için değerlendirme yapılmadı.
8. SAVEE için yetkili lisans/rıza ve BNC için yayın onayı belgelenemedi. Üçüncü taraf repolar hak kaynağı kabul edilmedi.
9. Kişilik hakları, ses biyometrisi ve tek konuşmacılı BNC kayıtlarıyla ilgili rıza kapsamı belirsizdir.
10. Mevcut Git geçmişindeki eski MLP dosyalarının yayın riski nasıl giderilecektir? History rewrite yapılmadı; bu rapor geçmişi değiştirmez.
11. Hak sahiplerinden model ağırlıklarının dağıtımına ilişkin **açık yazılı izin** belgesi yoktur.

Bu belirsizlikler nedeniyle: **CC BY-NC-SA 4.0 ve CC BY-NC 4.0'nın eğitilmiş ağırlıklara tam olarak nasıl uygulanacağına dair kesin bir hüküm verilmemektedir.** Dağıtım düşünülüyorsa nitelikli bir avukatın lisans metinlerini, ülke mevzuatını ve modelin somut içeriğini incelemesi gerekir.

---

## 6. GitHub yayın kararı

| Öğe | Karar | Gerekçe |
|---|---|---|
| RF-B `.joblib`, scaler `.joblib` | **Yayınlama — `LICENSE_REVIEW`** | Karma veri, SAVEE/BNC belirsizliği, CC/ODbL belirsizliği, açık izin yok |
| `mlp_best` / `mlp_final` `.keras` | **Yayınlama — `LICENSE_REVIEW`** | Yukarıdaki gerekçeler; final dosyada manifest hash eksikliği |
| Ham WAV'ler (`data/**`) | **Yayınlama** | İndirilebilirlik dağıtım izni değildir; BNC rızası yok, SAVEE lisansı yok |
| Kayıt bazlı CSV/NPZ türevleri | **Yayınlama** | Kaynak kimlikleri/hash'ler ve belirsiz kaynaklar nedeniyle önceki denetimde `LICENSE_REVIEW` |
| Kod ve kurulum dosyaları | Ayrı değerlendirilir | Proje geneli `LICENSE` yok; kod lisansı veri/model lisansından bağımsız seçilmelidir |
| Toplu metrik raporları / label map / F0 medyanları | Koşullu olarak yayınlanabilir | Kişisel yollar temizlenmeli, kaynak atıfları korunmalı; model izni anlamına gelmez |
| `github_yayin_oncesi_denetim.md` | Olduğu gibi yayınlanmaz | Kişisel yol kanıtları içerir |
| Eski `sonuclar_ysa/*.keras` | Geçmişte kalıyor | Kaldırma history rewrite ister; kapsam dışı, risk not olarak kaydedildi |

**Karar özeti:** Açık yazılı izin ve hukuki değerlendirme alınana kadar hiçbir model ağırlığı veya scaler dosyası stage edilmez, commit'lenmez veya push edilmez. Bu inceleme sırasında commit/push yapılmadı.

---

## 7. Güvenli alternatifler

1. **Ağırlıkları yerel tut, yöntemi yayınla.** Mimari, özellik listesi (192), hiperparametreler, split protokolü, toplu metrikler ve eğitim kodu paylaşılabilir; ağırlık dosyası paylaşılmaz.
2. **Açık izin al.** Hak sahiplerinden (CREMA-D, RAVDESS, TESS kayıt sahipleri; SAVEE için yetkili taraf; BNC için konuşmacı/rıza mercileri) **eğitilmiş model ağırlıklarının dağıtımını açıkça kapsayan** yazılı izin veya ayrı bir model lisansı iste. Genel veri lisansı, model izni yerine geçmez.
3. **Lisansı temiz veriyle yeniden eğit.** BNC ve SAVEE'yi ve izin alınamayan NC/SA kaynaklarını içermeyen, dağıtım izni belgelenmiş bir veri havuzuyla sıfırdan eğitim yap; proveransı yeni bir manifest ile kaydet. Bu, mevcut ağırlıkların lisansını değiştirmez.
4. **Yalnızca agregate çıktı paylaş.** Sınıf bazlı F1 ve confusion matrix gibi toplu sonuçlar yayınlanabilir; kayıt bazlı tahminler, `source_id`/SHA eşleştirmeleri ve ham veri türevleri paylaşılmaz. Yine de kaynak atıfları korunur.
5. **Bağımsız hukuki inceleme.** Yayın kararı verilmeden önce bir avukattan CC/ODbL/DbCL koşullarının somut ağırlık dosyalarına uygulanması, NC/SA yükümlülükleri ve BNC mahremiyet riski hakkında yazılı görüş al.
6. **Kod lisansını ayrıca belirle.** Proje için bir `LICENSE` dosyası seç; README'de kod lisansı ile veri/model lisanslarının farklı olduğunu açıkça belirt. Kod lisansı hiçbir veri veya ağırlık izni sağlamaz.
7. **Erişim kısıtlı, yayın dışı kullanım.** Yerel veya kurum içi kullanım, kamuya dağıtımdan ayrıdır; ancak erişim kontrolü tek başına lisans/onay eksikliğini gidermez.

---

## 8. Kaynaklar ve sınırlar

**Birincil kaynaklar:** CREMA-D `LICENSE.txt`; ODbL 1.0 ve DbCL 1.0 resmî metinler; CC BY-NC-SA 4.0 ve CC BY-NC 4.0 legalcode metinleri; Zenodo 1188976 (RAVDESS); University of Toronto Borealis TESS kaydı.

**Yerel kanıtlar:** `A1_2/70_15_15/artifact_manifest_A1_2.json`, `A1_2/kfold/son_rapor_A1_2.md`, `A1_2/70_15_15/son_rapor.md`, `A1_2/70_15_15/modeller/`, `github_yayin_oncesi_denetim.md`, `A1_2/dis_gercek_ses_testi/lisans_durumu.csv`, `A1_2/dis_gercek_ses_testi/cakisma_ve_dogrulama_raporu.md`.

**Yapılmayanlar:** Dosya silme/değiştirme/taşıma, model dosyalarına dokunma, yeniden eğitim, paket kurulumu, Git geçmişi değişikliği, commit veya push. Bu rapor dışında yeni bir dosya oluşturulmadı.

**Sınırlar:** Model dosyalarının ikili içeriği, bellek/ezerleme (memorization) testleri, kişilik hakkı değerlendirmesi ve yargı alanına özgü telif istisnaları incelenmedi. SAVEE lisansı ve BNC yayın rızası doğrulanamadı. Bu rapor hukuki görüş veya lisans onayı değildir; belirsizlikler nedeniyle tüm model ağırlıkları `LICENSE_REVIEW` olarak kalır.
