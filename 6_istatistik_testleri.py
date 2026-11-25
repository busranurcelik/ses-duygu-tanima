
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.models import load_model
from scipy.stats import binomtest

# ==================================================
# AYARLAR
# ==================================================
INPUT_FILE = "final_oznitelik_seti.csv"
MODEL_PATH = "sonuclar_ysa/best_model.keras"
SCALER_PATH = "sonuclar_ysa/scaler.pkl"
RESULT_FOLDER = "sonuclar_istatistik"

SEED = 42

os.makedirs(RESULT_FOLDER, exist_ok=True)

print("=== 6. ADIM: MCNEMAR TESTİ & ROC ANALİZİ ===")

# ==================================================
# 1. VERİYİ YÜKLE
# ==================================================
df = pd.read_csv(INPUT_FILE).dropna()

label_col = "labels"
X = df.drop(label_col, axis=1).values
y = df[label_col].values

# Label Encoding
lb = LabelEncoder()
y_encoded = lb.fit_transform(y)
classes = lb.classes_

print("Sınıf sırası:", classes)

# ==================================================
# 2. YSA İLE AYNI TRAIN / VAL / TEST BÖLMESİ
# ==================================================

# İlk bölme: %70 eğitim, %30 geçici
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    stratify=y_encoded,
    random_state=SEED,
    shuffle=True
)

# İkinci bölme: %15 doğrulama, %15 test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED,
    shuffle=True
)

print("\nVeri seti boyutları:")
print("Eğitim:", X_train.shape)
print("Doğrulama:", X_val.shape)
print("Test:", X_test.shape)

# ==================================================
# 3. ÖLÇEKLENDİRME
# ==================================================

# YSA eğitiminde kaydedilen scaler yüklenir.
scaler = joblib.load(SCALER_PATH)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nYSA ile aynı scaler kullanılıyor.")

# ==================================================
# 4. YSA MODELİ
# ==================================================
print("\nYSA modeli yükleniyor...")

model_ysa = load_model(MODEL_PATH)

# Sınıf sırası kontrolü
expected_classes = ["angry", "fear", "happy", "sad"]

if list(classes) != expected_classes:
    raise ValueError(
        f"Sınıf sırası beklenenden farklı: {list(classes)}"
    )

# Model çıkış boyutu kontrolü
if model_ysa.output_shape[-1] != len(classes):
    raise ValueError(
        "YSA çıkış boyutu sınıf sayısıyla uyuşmuyor!"
    )

# YSA tahminleri
y_pred_probs_ysa = model_ysa.predict(
    X_test_scaled,
    verbose=1
)

y_pred_ysa = np.argmax(y_pred_probs_ysa, axis=1)

# ==================================================
# 5. RANDOM FOREST MODELİ
# ==================================================
print("\nRandom Forest eğitiliyor...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

# ==================================================
# 6. MODEL DOĞRULUKLARI
# ==================================================
ysa_accuracy = np.mean(y_pred_ysa == y_test)
rf_accuracy = np.mean(y_pred_rf == y_test)

print("\n=== TEST SONUÇLARI ===")
print(f"YSA Accuracy: {ysa_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# ==================================================
# 7. MCNEMAR TESTİ
# ==================================================
print("\n=== MCNEMAR TESTİ ===")

ysa_correct = (y_pred_ysa == y_test)
rf_correct = (y_pred_rf == y_test)

n00 = np.sum(ysa_correct & rf_correct)
n01 = np.sum(~ysa_correct & rf_correct)
n10 = np.sum(ysa_correct & ~rf_correct)
n11 = np.sum(~ysa_correct & ~rf_correct)

mcnemar_table = np.array([
    [n00, n01],
    [n10, n11]
])

print("\nMcNemar Tablosu:")
print(mcnemar_table)

print(f"İki model de doğru: {n00}")
print(f"RF doğru / YSA yanlış: {n01}")
print(f"YSA doğru / RF yanlış: {n10}")
print(f"İki model de yanlış: {n11}")

discordant = n01 + n10

if discordant == 0:
    p_value = 1.0
else:
    p_value = binomtest(
        k=min(n01, n10),
        n=discordant,
        p=0.5,
        alternative="two-sided"
    ).pvalue

if p_value < 0.05:
    yorum = "İki model arasında istatistiksel olarak anlamlı fark vardır."
else:
    yorum = "İki model arasında istatistiksel olarak anlamlı fark yoktur."

print(f"\nMcNemar p-değeri: {p_value:.6f}")
print("Sonuç:", yorum)

# McNemar sonuçlarını kaydet
with open(
    f"{RESULT_FOLDER}/mcnemar_sonuclari.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("McNemar Testi (YSA vs Random Forest)\n")
    f.write(f"Test örneği sayısı: {len(y_test)}\n")
    f.write(f"YSA Accuracy: {ysa_accuracy:.4f}\n")
    f.write(f"RF Accuracy: {rf_accuracy:.4f}\n")
    f.write(f"İki model de doğru: {n00}\n")
    f.write(f"B (RF doğru / YSA yanlış): {n01}\n")
    f.write(f"C (YSA doğru / RF yanlış): {n10}\n")
    f.write(f"İki model de yanlış: {n11}\n")
    f.write(f"p-değeri: {p_value:.6f}\n")
    f.write(f"Yorum: {yorum}\n")

# ==================================================
# 8. ROC EĞRİLERİ (MULTICLASS - YSA)
# ==================================================
print("\n=== ROC ANALİZİ ===")

y_test_bin = label_binarize(
    y_test,
    classes=np.arange(len(classes))
)

n_classes = y_test_bin.shape[1]

if y_pred_probs_ysa.shape[1] != n_classes:
    raise ValueError(
        "ROC için tahmin olasılıklarının boyutu hatalı!"
    )

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_pred_probs_ysa[:, i]
    )

    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC grafiği
plt.figure(figsize=(10, 8))

colors = ["red", "blue", "green", "orange"]

for i, color in zip(range(n_classes), colors):

    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    lw=2
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("YSA Modeli - Çok Sınıflı ROC Eğrisi")

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{RESULT_FOLDER}/ysa_roc_egrisi.png",
    dpi=300
)

plt.close()

# ==================================================
# 9. ROC AUC SONUÇLARINI KAYDET
# ==================================================
roc_results = pd.DataFrame({
    "Sınıf": classes,
    "AUC": [roc_auc[i] for i in range(n_classes)]
})

roc_results.to_csv(
    f"{RESULT_FOLDER}/roc_auc_sonuclari.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nROC AUC sonuçları:")
print(roc_results)

# ==================================================
# 10. TAMAMLANDI
# ==================================================
print("\n=== TÜM İSTATİSTİKSEL ANALİZLER TAMAMLANDI ===")
print(f"Çıktılar: {RESULT_FOLDER}")
