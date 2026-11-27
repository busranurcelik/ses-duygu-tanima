# Model ve Scaler Dağıtım İncelemesi

Tarih: 2026-11-27\
Bu dosya, RF-B, MLP ve scaler dosyalarının yayın adayındaki dağıtım durumunu ayrı açıklar. **Bu bir izin belgesi veya hukuki görüş değildir.** Açık yazılı dağıtım izni ve bağımsız hukuki değerlendirme olmadan hiçbir ağırlık dosyası stage edilmez, commit'lenmez veya push edilmez.

## 1. Ortak gerekçe

A1.2 modelleri CREMA-D, RAVDESS, TESS, SAVEE ve yerel BNC kayıtlarının karışımıyla eğitildi. CREMA-D ODbL/DbCL, RAVDESS CC BY-NC-SA 4.0, TESS CC BY-NC 4.0 altındadır; SAVEE lisansı/rızası `DOGRULANAMADI`, BNC için yayın rızası belgelenmemiştir. Eğitilmiş ağırlıkların bu lisanslar anlamında Adapted Material, Produced Work veya yalnızca çıktı sayılıp sayılmayacağı birincil metinlerde açıkça çözülmemiştir. Ağırlıkların ezberleme/üyelik çıkarımı riski de bu incelemede test edilmemiştir.

Bu nedenle: **model ağırlıklarının dağıtımına izin verildiği sonucuna varılmamıştır.**

## 2. RF-B (ana model)

| Öğe | Durum |
|---|---|
| Dosya | `A1_2/70_15_15/modeller/rf_70_15_15_matched_A1_2.joblib` (yayın adayında yok) |
| Rol | GUI'nin kullandığı ana `RandomForestClassifier`; 200 ağaç, `class_weight="balanced"`, `random_state=42`, 192 özellik |
| Eğitim | 70/15/15 train; scaler yalnızca train'e fit edildi |
| Performans | 70/15/15 accuracy 0,6562 / macro-F1 0,6252; OOF 0,5978 / 0,5922. Test yalnızca `TESS_YAF` |
| Lisans kararı | **`LICENSE_REVIEW` — dağıtım yok** |
| Gerekçe | Karma veri, SAVEE/BNC belirsizliği, CC/ODbL belirsizliği, açık yazılı izin yok |

RF-B serileştirilmiş bölme eşikleri ve yaprak değerleri içerir; bunlar toplu metrik değil, veriden öğrenilmiş parametrelerdir.

## 3. MLP (alternatif model)

| Öğe | Durum |
|---|---|
| Dosyalar | `mlp_best_70_15_15_A1_2.keras`, `mlp_final_70_15_15_A1_2.keras` (yayın adayında yok) |
| Rol | RF-B ile karşılaştırma modeli; GUI'ye entegre değil |
| Eğitim | RF-B ile aynı 70/15/15 train ve aynı scaler |
| Performans | 70/15/15 accuracy 0,6075 / macro-F1 0,5548; OOF 0,5950 / 0,5912 |
| Lisans kararı | **`LICENSE_REVIEW` — dağıtım yok** |
| Ek belirsizlik | `mlp_final_70_15_15_A1_2.keras` için manifest SHA-256 kaydı eksik |
| Gerekçe | RF-B ile aynı karma veri ve izin belirsizlikleri |

MLP sinir ağı ağırlıkları da eğitim örnekleri üzerinde optimize edilmiş sayısal parametrelerdir; "yalnızca genel istatistik" varsayımı doğrulanmamıştır.

## 4. Scaler

| Öğe | Durum |
|---|---|
| Dosya | `A1_2/70_15_15/modeller/scaler_A1_2.joblib` (yayın adayında yok) |
| Rol | 70/15/15 train satırlarına fit edilen `StandardScaler`; RF-B ve MLP aynı scaleri kullanır |
| Lisans kararı | **`LICENSE_REVIEW` — dağıtım yok** |
| Gerekçe | Scaler, lisansı belirsiz/kısıtlı kayıtları içeren eğitim verisinden öğrenilmiş parametreler taşır; ayrı dağıtım izni yok |

`scaler_kaydet.py` kaynak kodu MIT kapsamındadır; ancak bu kodun ürettiği `scaler.pkl` / `scaler.joblib` dosyası MIT kapsamına girmez.

## 5. Eski MLP artefaktları ve Git geçmişi

`sonuclar_ysa/best_model.keras`, `sonuclar_ysa/ysa_final_model.keras` ve `sonuclar_ysa/scaler.pkl` önceki MLP çalışmasına ait ağırlık/scaler dosyalarıdır. İzole test deposunda bu üç dosya tüm commit geçmişinden çıkarılmış ve doğrulanmıştır; yayın adayı o temiz kopyadan oluşturulmuştur. Bu durum, diğer ağırlıkların dağıtımına izin verildiği anlamına gelmez.

## 6. Karar özeti

| Artefakt | Dağıtım |
|---|---|
| RF-B `.joblib` | **Yok — LICENSE_REVIEW** |
| MLP `.keras` dosyaları | **Yok — LICENSE_REVIEW** |
| Scaler `.joblib` / `.pkl` | **Yok — LICENSE_REVIEW** |
| Ham WAV'ler | **Yok** |
| Kayıt bazlı CSV/NPZ türevleri | **Yok** |
| Toplu metrikler / label map / F0 medyanları | Kaynak atıfı korunmak koşuluyla ayrı değerlendirilir; model izni vermez |
| Kod (bölüm 1'deki 24 dosya) | MIT — yalnızca kod |

Açık yazılı izin, hukuki inceleme ve proverans doğrulaması tamamlanmadan bu karar değiştirilmez.
