# Dis gercek ses testi: secim, lisans ve cakisma dogrulama raporu

Olusturulma (UTC): 2026-11-27T04:00:40+00:00

Bu rapor prediction **oncesinde** uretildi. Secilen kayitlar orijinal WAV dosyalardir; augmentation, yeniden kodlama veya kesme uygulanmamis, model tarafinda 22050 Hz / 2.5 s on isleme yapilacaktir.

## 1. Kapsam ve lisans durumu

| Kaynak | Durum | Kapsam | Dayanak |
|---|---|---|---|
| CREMA-D | DOGRULANDI | Veritabani ODbL 1.0; bireysel icerikler DbCL 1.0 | https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt |
| RAVDESS | DOGRULANDI | CC BY-NC-SA 4.0 | https://zenodo.org/records/1188976 |
| SAVEE | DOGRULANAMADI | Belirtilmedi | - |
| TESS | DOGRULANDI | CC BY-NC 4.0 | https://doi.org/10.5683/SP2/E8H2MF |
| BNC | HARIC | Riza belgelenmedi | - |

- CREMA-D ve RAVDESS lisans kosullari erisilebilir birincil kaynaklardan okundu.
- SAVEE icin yetkili lisans/riza metni yerelde ve erisilebilir kurumsal adreslerde bulunamadi; bu nedenle lisans durumu **DOGRULANAMADI** olarak isaretlendi. Ucuncu taraf GitHub repolari hak kaynagi kabul edilmedi. Bu test raporu SAVEE lisansina iliskin yasal bir iddia tasimaz; kayitlar mevcut proje kapsaminda yerel degerlendirme icin kullanilir, yeniden dagitilmaz.
- BNC (`BNCS-SES-WAV`) 100 kaydi tek konusmaciya (`BNC_SPEAKER_01`) aittir ve riza/privacy belgesi bulunamadigi icin degerlendirme disinda birakilmistir; sadece envanterde raporlanir.
- TESS lisansi U of T Borealis metadata'sindan dogrulandi (CC BY-NC 4.0); ancak TESS mevcut 70/15/15 test bolumunun kaynagidir. 3200 hedef-etiketli TESS WAV'in tamami ham_veri_listesi.csv'envanterindedir; modeling manifestlerinde yer almayan 1600 WAV ise icerik SHA-256 bakimindan modeling dosyalariyla birebir aynidir. Bu nedenle TESS icinde de hash-temiz, kullanilmamis hedef-etiketli aday kalmamistir.

## 2. Cakisma denetimi

- `ham_veri_listesi.csv` benzersiz yol sayisi: 10160
- Bu yollardan okunan dosya icerik SHA-256 sayisi: 7790
- Egitimde gorulen benzersiz yol-hash sayisi: 20320
- Mutlak cakisma kurali: secilen her dosya; (a) egitim manifestindeki yollarla, (b) egitim dosyalarinin icerik SHA-256 degerleriyle, (c) dedup manifesti ve A1.2 ozellik CSV'lerindeki source_id/yol hash'leriyle karsilastirildi. Eslesen adaylar secilmedi.
- Adaylarin kendi aralarinda ayni icerik SHA-256'ya sahip olmasi durumunda ilgili grup tamamen dislandi.

| Kaynak | Toplam WAV | Hedef disi | Hedef etiketli | Egitim yolu | Icerik hash | Yol hash | Uygun | Secilen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CREMA-D | 7442 | 2359 | 5083 | 5083 | 5083 | 5083 | 0 | 0 |
| RAVDESS | 2880 | 1344 | 1536 | 1536 | 1536 | 1536 | 0 | 0 |
| SAVEE | 480 | 240 | 240 | 240 | 240 | 240 | 0 | 0 |
| TESS | 5600 | 2400 | 3200 | 3200 | 3200 | 3200 | 0 | 0 |

## 3. Kaynak/sinif dagilimi ve eksikler

| Kaynak | Sinif | Uygun aday | Secilen | Eksik | Konusmaci sayisi | Konusmacilar |
|---|---|---:|---:|---:|---:|---|
| CREMA-D | angry | 0 | 0 | 15 | 0 |  |
| CREMA-D | fear | 0 | 0 | 15 | 0 |  |
| CREMA-D | happy | 0 | 0 | 15 | 0 |  |
| CREMA-D | sad | 0 | 0 | 15 | 0 |  |
| RAVDESS | angry | 0 | 0 | 15 | 0 |  |
| RAVDESS | fear | 0 | 0 | 15 | 0 |  |
| RAVDESS | happy | 0 | 0 | 15 | 0 |  |
| RAVDESS | sad | 0 | 0 | 15 | 0 |  |
| SAVEE | angry | 0 | 0 | 15 | 0 |  |
| SAVEE | fear | 0 | 0 | 15 | 0 |  |
| SAVEE | happy | 0 | 0 | 15 | 0 |  |
| SAVEE | sad | 0 | 0 | 15 | 0 |  |
| TESS | angry | 0 | 0 | 15 | 0 |  |
| TESS | fear | 0 | 0 | 15 | 0 |  |
| TESS | happy | 0 | 0 | 15 | 0 |  |
| TESS | sad | 0 | 0 | 15 | 0 |  |

Eksik sinif kaydi: CREMA-D/angry: -15, CREMA-D/fear: -15, CREMA-D/happy: -15, CREMA-D/sad: -15, RAVDESS/angry: -15, RAVDESS/fear: -15, RAVDESS/happy: -15, RAVDESS/sad: -15, SAVEE/angry: -15, SAVEE/fear: -15, SAVEE/happy: -15, SAVEE/sad: -15, TESS/angry: -15, TESS/fear: -15, TESS/happy: -15, TESS/sad: -15

## 4. Otomatik kontroller

- PASS: Tum secilen yollar egitim manifestinde yok
- PASS: Tum secilen icerik hash'leri egitim hash'lerinde yok
- PASS: Tum secilen yol hash'leri egitim source_id'lerinde yok
- PASS: Secilenler icinde tekrar eden icerik hash yok
- FAIL: Her kaynak/sinif en az 15 kayit
- PASS: BNC degerlendirmeye alinmadi
- PASS: Tum secilen WAV basliklari okunabildi

**Prediction durumu:** ENGEL
Hedef etiketli, ham envanterde yer almayan ve icerik/yol hash acisindan egitimle cakismayan WAV bulunamadigi icin prediction baslatilmadi. Bu rapor eksiklik raporudur; basari metrigi uretilmedi.

## 5. Yorum siniri

Bu secim rastgele bir referans test kumesi degildir. Egitimde kullanilmamis orijinal kayitlardan kaynak/sinif dengeli bir **ek degerlendirme** alt kumesidir. Konusmacilar egitimdeki kayitlarla ayni olabilir; sonuclar gorulmemis konusmaciya veya genel populyasyona genelleme olarak okunmamalidir.
