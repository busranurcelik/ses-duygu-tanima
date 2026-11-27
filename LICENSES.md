# License Inventory

Tarih: 2026-11-27\
Bu dosya, `LICENSE` dosyasının kapsamını sınırlandırır ve yayın adayındaki her dosya sınıfının lisans durumunu ayrı belgeler. **Bu dosya hukuki görüş değildir.** Belirsizlikler korunur; hak sahipliği doğrulanmayan dosyalara yayın izni verilmez.

## 1. MIT kapsamı — yalnızca özgün kaynak kod

Aşağıdaki 24 Python dosyası, depanın Git geçmişindeki yazar kaydı (`Busra Nur Celik`) ve dosya üst bilgileriyle bu proje için yazılmış özgün kaynak kod olarak değerlendirildi. Yalnızca bu dosyalar `LICENSE` altındadır.

```text
1_veri_hazirlama.py
1_veri_hazirlama_v2.py
2_oznitelik_cikarimi.py
2_oznitelik_cikarimi_v2.py
2_oznitelik_cikarimi_v2_dedup.py
3_modelleri_egit.py
3_modelleri_egit_v2.py
4_ysa_modeli.py
4_ysa_modeli_v2.py
5_kfold_dogrulama.py
5_kfold_dogrulama_v2.py
6_istatistik_testleri.py
6_istatistik_testleri_v2.py
7_final_arayuz.py
arayuz_A1_2_adapter.py
scaler_kaydet.py
test_arayuz_A1_2.py
A1_2/70_15_15/4_ysa_A1_2.py
A1_2/dis_gercek_ses_testi/00_secim_ve_cakismalar.py
A1_2/kfold/5_kfold_A1_2.py
A1_2/kfold/dogrulama_sonrasi_A1_2.py
A1_2/kfold/istatistiksel_karsilastirma_A1_2.py
A1_2/on_kontrol_A1_2.py
A1_2/ozellik_cikarimi_A1_2.py
```

Bu dosyalarda üçüncü taraf kaynak kodundan kopyalanmış bir blok bulunmadı; dosyalar yalnızca üçüncü taraf kütüphaneleri **import eder**. Kütüphanelerin kendisi MIT kapsamında değildir. `scaler_kaydet.py` bir scaler dosyası üretir; kod MIT olsa bile üretilen scaler `.pkl` dosyası MIT kapsamında değildir.

## 2. Üçüncü taraf kod — MIT dışı

Yayın adayında üçüncü taraf kaynak kod dosyası bulunmadı. Aşağıdaki paketler yalnızca bağımlılıktır; kaynak kodları bu depoya dahil değildir ve kendi lisanslarına tabidir:

- `joblib==1.5.3`
- `librosa==0.11.0`
- `matplotlib==3.10.8`
- `numpy==2.3.5`
- `pandas==2.3.3`
- `scikit-learn==1.8.0`
- `scipy==1.16.3`
- `seaborn==0.13.2`
- `sounddevice==0.5.3`
- `tensorflow==2.20.0` (yalnızca tam kurulumda)

`requirements.txt` ve `requirements-gui.txt` bu bağımlılıkları listeler; kendileri MIT kapsamındaki kaynak kod değildir.

**Sahipliği/lisansı belirsiz kod:** Bu incelemede öyle bir dosya tespit edilmedi. İleride böyle bir dosya bulunursa `LICENSE` kapsamından çıkarılır ve ayrı olarak işaretlenir; otomatik MIT izni verilmez.

## 3. Veri ve lisans/atıf belgeleri — MIT dışı

| Dosya | Durum |
|---|---|
| `ATIF.md` | Veri kaynakları ve atıflar; MIT dışı |
| `LICENSES.md` | Bu envanter; MIT dışı |
| `LICENSE` | MIT metni; kapsamı yalnızca bölüm 1'deki kod |
| `MODEL_LICENSE_REVIEW.md` | Model/scaler dağıtım kararı; MIT dışı |
| `model_agirliklari_lisans_incelemesi.md` | Ağırlık lisans incelemesi; MIT dışı |
| `A1_2/dis_gercek_ses_testi/lisans_durumu.csv` | Kaynak lisans/rıza kaydı; MIT dışı |
| `A1_2/dis_gercek_ses_testi/cakisma_ve_dogrulama_raporu.md` | Lisans/çakışma denetimi; MIT dışı |
| `github_son_yayin_dosyasi_listesi.md` | Yayın karar envanteri; MIT dışı |

Veri lisansları ve rıza belirsizlikleri `ATIF.md` içinde korunur. **İndirilebilirlik, yeniden dağıtım izni değildir.**

## 4. Toplu raporlar ve türetilmiş çıktılar — MIT dışı, koşullu yayın

A1.2 metrik tabloları, doğrulama özetleri, eşik değerleri, label map dosyaları ve grafikler kayıt bazlı ham veri veya model ağırlığı içermez; önceki dosya kararında `YAYINLA` olarak işaretlenmiştir. Bunlar **MIT lisansına dahil değildir**. Yayında CREMA-D, RAVDESS, TESS, SAVEE ve BNC atıf/notları korunmalı; kayıt bazlı `source_id`, SHA-256, mutlak yol ve ağırlık dosyası eklenmemelidir.

Bu kategorideki dosyalar için hukuki kesinlik iddiası yoktur; yalnızca toplu/anonim oldukları ve önceki denetimde kayıt bazlı risk taşımadıkları belgelenmiştir.

## 5. Model ağırlıkları ve scaler — MIT dışı, dağıtım izni yok

| Dosya | Karar |
|---|---|
| `A1_2/70_15_15/modeller/rf_70_15_15_matched_A1_2.joblib` | `LICENSE_REVIEW` — dağıtım yok |
| `A1_2/70_15_15/modeller/scaler_A1_2.joblib` | `LICENSE_REVIEW` — dağıtım yok |
| `A1_2/70_15_15/modeller/mlp_best_70_15_15_A1_2.keras` | `LICENSE_REVIEW` — dağıtım yok |
| `A1_2/70_15_15/modeller/mlp_final_70_15_15_A1_2.keras` | `LICENSE_REVIEW` — dağıtım yok, ayrıca manifest hash'i eksik |
| `sonuclar_ysa/best_model.keras`, `ysa_final_model.keras`, `scaler.pkl` | Geçmişten çıkarıldı; dağıtım yok |

Bu dosyalar yayın adayında bulunmaz. Ayrıntılar `MODEL_LICENSE_REVIEW.md` ve `model_agirliklari_lisans_incelemesi.md` içindedir. **Bu belgeler dağıtım izni sonucuna varmaz.**

## 6. Diğer / yayın dışı

| Dosya grubu | Durum |
|---|---|
| `.gitignore` | Yapılandırma; MIT kapsamında değil |
| `README.md` | Proje belgesi; MIT kapsamında değil |
| `github_lisans_ve_readme_raporu.md` | Denetim raporu; MIT kapsamında değil |
| `sonuclar_klasik/`, `sonuclar_istatistik/`, `sonuclar_ysa/` içindeki eski çıktılar | Opsiyonel/koşullu; yayın izni yok, çıkarılması güvenli |

Temiz temelden kalıtılmış 27 yayın dışı dosyadan `README.md` dışındaki 26'sı çalışma ağacından kaldırıldı; Git geçmişi ve HEAD değiştirilmedi. Bunlar yayın manifestine alınmaz.

## 7. Uygulama ekran görüntüleri — MIT dışı, yayın izni verildi

Aşağıdaki iki görsel, proje sahibi `Busra Nur Celik` tarafından kendi uygulaması (`7_final_arayuz.py` ve `arayuz_A1_2_adapter.py`) çalıştırılarak alınmış ekran görüntüleridir. Proje sahibi kendi ürettiği bu görsellerin yayınlanmasına izin vermiştir; eski `screenshots/arayuz.png` ve `screenshots/duygu_tahmini.png` dosyalarına ilişkin “görsel onayı yok / yayın izni yok” kararı bu iki yeni dosya için geçerli değildir.

| Dosya | Lisans ve yayın durumu |
|---|---|
| `screenshots/application-interface.png` | Proje sahibinin kendi uygulamasından alınmış ekran görüntüsü; yayın izni **var**; `LICENSE` (MIT) kapsamında **değil** |
| `screenshots/emotion-prediction-example.png` | Proje sahibinin kendi uygulamasından alınmış ekran görüntüsü; yayın izni **var**; `LICENSE` (MIT) kapsamında **değil** |

`LICENSE` metninin kapsamı yalnızca bölüm 1'de listelenen 24 Python dosyasıdır. Ekran görüntüleri kaynak kod olmadığından bu kapsama **otomatik olarak dahil edilmez** ve 24 dosyalık liste değiştirilmez. Bu onay yalnızca ekran görüntülerinin yayınına ilişkindir; görüntülerde görünebilecek üçüncü taraf veri, model veya kütüphane içerikleri için ayrı bir dağıtım izni verilmez. Model ağırlıkları ve scaler dosyalarının durumu bölüm 5'teki `LICENSE_REVIEW` kararıyla korunur.
