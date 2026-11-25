import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from tensorflow.keras.models import load_model

# ==================================================
# AYARLAR
# ==================================================
INPUT_FILE = "final_oznitelik_seti.csv"
MODEL_PATH = "sonuclar_ysa/ysa_final_model.keras"   
SCALER_PATH = "sonuclar_ysa/scaler.pkl"
RESULT_FOLDER = "sonuclar_istatistik"

os.makedirs(RESULT_FOLDER, exist_ok=True)

print("=== 6. ADIM: MCNEMAR TESTİ & ROC ANALİZİ ===")

# ==================================================
# 1. VERİYİ YÜKLE
# ==================================================
df = pd.read_csv(INPUT_FILE).dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

lb = LabelEncoder()
y_encoded = lb.fit_transform(y)
classes = lb.classes_

scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# AYNI BÖLME 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"Test Seti Boyutu: {len(y_test)}")

# ==================================================
# 2. MODELLER
# ==================================================

# --- YSA ---
print("\n YSA modeli yükleniyor...")
model_ysa = load_model(MODEL_PATH)

y_pred_probs_ysa = model_ysa.predict(X_test)
y_pred_ysa = np.argmax(y_pred_probs_ysa, axis=1)

# --- RANDOM FOREST ---
print(" Random Forest eğitiliyor...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ==================================================
# 3. MCNEMAR TESTİ
# ==================================================
print("\n McNemar Testi Hesaplanıyor...")

ysa_correct = (y_pred_ysa == y_test)
rf_correct  = (y_pred_rf  == y_test)

n00 = np.sum(ysa_correct & rf_correct)      # ikisi doğru
n01 = np.sum(~ysa_correct & rf_correct)     # RF doğru, YSA yanlış
n10 = np.sum(ysa_correct & ~rf_correct)     # YSA doğru, RF yanlış
n11 = np.sum(~ysa_correct & ~rf_correct)    # ikisi yanlış

mcnemar_table = np.array([[n00, n01],
                           [n10, n11]])

print("\nMcNemar Tablosu:")
print(mcnemar_table)

print(f"RF Doğru - YSA Yanlış (B): {n01}")
print(f"YSA Doğru - RF Yanlış (C): {n10}")

chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
print(f"Chi-Square Değeri: {chi2:.4f}")

if chi2 > 3.84:
    yorum = "İstatistiksel olarak ANLAMLI fark vardır."
else:
    yorum = "İstatistiksel olarak anlamlı fark yoktur."

print("SONUÇ:", yorum)

with open(f"{RESULT_FOLDER}/mcnemar_sonuclari.txt", "w", encoding="utf-8") as f:
    f.write("McNemar Testi (YSA vs Random Forest)\n")
    f.write(f"B (RF doğru / YSA yanlış): {n01}\n")
    f.write(f"C (YSA doğru / RF yanlış): {n10}\n")
    f.write(f"Chi2: {chi2:.4f}\n")
    f.write(f"Yorum: {yorum}\n")

# ==================================================
# 4. ROC EĞRİLERİ (MULTICLASS – YSA)
# ==================================================
print("\n ROC Eğrileri çiziliyor...")

y_test_bin = label_binarize(y_test, classes=np.arange(len(classes)))
n_classes = y_test_bin.shape[1]

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs_ysa[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

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

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("YSA Modeli – Çok Sınıflı ROC Eğrisi")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULT_FOLDER}/ysa_roc_egrisi.png", dpi=300)
plt.close()

print("\n TÜM İSTATİSTİKSEL ANALİZLER TAMAMLANDI")
print(f"Çıktılar: {RESULT_FOLDER}")
