# Veri Hazırlama Raporu — A1.2 (A0 girdisi, F0 hazırlığı)

Yalnızca veri hazırlama ve F0 çıkarımı yapıldı. Eğitim, CV, 70/15/15, tahmin, model/scaler kaydı veya GUI değişikliği yapılmadı.

## Girdiler

- `manifest_v2_dedup_draft.csv`: 7782 satır (draft)
- `manifest_v2_dedup_review_qc.csv`: 17 satır (REVIEW_QC)
- `final_oznitelik_seti_v2_dedup.csv`: 23268 satır × 194 kolon (ilk 184 özellik kaynağı)

## Dışlananlar

- REVIEW_LABEL: 2 kayıt / 1 SHA256 — etiket değiştirilmedi.
- REVIEW_QC: 17 kayıt — canonical kümeyle kesişim 0.

## Canonical küme

- Kayıt: **7780**, tekil source_id: **7780**, tekil SHA256: **7780**, group_id: **122**

| source | canonical_kayit |
|---|---|
| BNC | 100 |
| CREMA | 5081 |
| RAVDESS | 759 |
| SAVEE | 240 |
| TESS | 1600 |

| label | canonical_kayit |
|---|---|
| angry | 1947 |
| fear | 1945 |
| happy | 1947 |
| sad | 1941 |

## Augmentation planı (A0 ile aynı)

Bir kayıt iki değerlendirmenin en az birinde train ise augmentation alır; BNC 6, diğer kaynaklar 2 varyant. Train olmayan kayıtlar yalnızca `orig`.

- Beklenen feature satırı: **23268**, üretilen: **23268**

## Ön leakage

- Tekrarlayan SHA256: **0**
- Çoklu fold/source: **0**
