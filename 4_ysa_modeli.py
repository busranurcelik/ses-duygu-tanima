import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ==================================================
# DETERMINISTIC AYARLAR
# ==================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==================================================
# AYARLAR
# ==================================================
DATA_FILE = "final_oznitelik_seti.csv"
OUT_DIR = "sonuclar_ysa"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 200          # hocanın şartı
BATCH_SIZE = 32

# ==================================================
# VERİYİ OKU
# ==================================================
df = pd.read_csv(DATA_FILE).dropna()

label_col = "labels"
X = df.drop(label_col, axis=1).values
y = df[label_col].values

# ==================================================
# LABEL ENCODING
# ==================================================
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = list(le.classes_)

with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print("Sınıf sırası:", classes)

y_cat = to_categorical(y_enc)

# ==================================================
# TRAIN / VAL / TEST (70 / 15 / 15)
# ==================================================
X_train, X_temp, y_train, y_temp, y_enc_train, y_enc_temp = train_test_split(
    X, y_cat, y_enc,
    test_size=0.30,
    stratify=y_enc,
    random_state=SEED,
    shuffle=True
)

X_val, X_test, y_val, y_test, y_enc_val, y_enc_test = train_test_split(
    X_temp, y_temp, y_enc_temp,
    test_size=0.50,
    stratify=np.argmax(y_temp, axis=1),
    random_state=SEED,
    shuffle=True
)

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ==================================================
# SCALER
# ==================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
print(" Scaler kaydedildi")

# Temel class_weight (balanced)
cw = compute_class_weight(class_weight="balanced",
                          classes=np.unique(y_enc_train),
                          y=y_enc_train)
class_weight = {i: float(w) for i, w in enumerate(cw)}

if "happy" in classes:
    class_weight[classes.index("happy")] *= 1.25
if "sad" in classes:
    class_weight[classes.index("sad")] *= 1.15

print("🎯 class_weight:", {classes[k]: round(v, 3) for k, v in class_weight.items()})

# ==================================================
# MODEL
# ==================================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(256, kernel_initializer="he_normal"),
    BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    Dropout(0.40),

    Dense(128, kernel_initializer="he_normal"),
    BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    Dropout(0.35),

    Dense(64, kernel_initializer="he_normal"),
    BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    Dropout(0.30),

    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# CALLBACKS
# ==================================================
ckpt_path = os.path.join(OUT_DIR, "best_model.keras")
checkpoint = ModelCheckpoint(
    ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=12,
    min_lr=1e-5,
    verbose=1
)

# ==================================================
# EĞİTİM
# ==================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight,          
    verbose=1
)

# ==================================================
# MODEL KAYDET (final)
# ==================================================
final_model_path = os.path.join(OUT_DIR, "ysa_final_model.keras")
model.save(final_model_path)
print(" Final model kaydedildi:", final_model_path)
print(" Best model kaydedildi :", ckpt_path)

# ==================================================
# EĞİTİM GRAFİKLERİ
# ==================================================
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("YSA Eğitim & Doğrulama Kayıp Grafiği")
plt.savefig(os.path.join(OUT_DIR, "ysa_loss.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("YSA Eğitim & Doğrulama Doğruluk Grafiği")
plt.savefig(os.path.join(OUT_DIR, "ysa_accuracy.png"), dpi=300)
plt.close()

print("Eğitim grafikleri kaydedildi")

# ==================================================
# TEST DEĞERLENDİRME (BEST MODEL ile)
# ==================================================
best_model = tf.keras.models.load_model(ckpt_path)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy (BEST): {test_acc:.4f}")

y_pred = best_model.predict(X_test, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true_cls,
    y_pred_cls,
    target_names=classes
))

# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_true_cls, y_pred_cls)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("YSA - Confusion Matrix (BEST)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ysa_confusion_matrix.png"), dpi=300)
plt.close()

print("Confusion matrix kaydedildi")

# ==================================================
# METRİK TABLOSU (RAPORLUK)
# ==================================================
labels = classes
rows = []
total = cm.sum()

for i, label in enumerate(labels):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = total - (TP + FN + FP)

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    rows.append([
        label,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(specificity, 4),
        round(f1, 4)
    ])

df_metrics = pd.DataFrame(
    rows,
    columns=[
        "Sınıf",
        "Doğruluk (Accuracy)",
        "Kesinlik (Precision)",
        "Duyarlılık (Recall)",
        "Özgüllük (Specificity)",
        "F1-Skoru"
    ]
)

print("\n YSA Performans Tablosu")
print(df_metrics)

df_metrics.to_csv(
    os.path.join(OUT_DIR, "ysa_sonuclari_tablo.csv"),
    index=False,
    encoding="utf-8-sig"
)
print(" YSA sonuç tablosu kaydedildi: ysa_sonuclari_tablo.csv")

print("\n TÜM YSA ÇIKTILARI OLUŞTURULDU")
print("Klasör:", OUT_DIR)
