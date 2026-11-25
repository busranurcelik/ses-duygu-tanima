import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Modeller
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

# =========================
# AYARLAR
# =========================
INPUT_FILE = "final_oznitelik_seti.csv"
RESULT_FOLDER = "sonuclar_klasik"
os.makedirs(RESULT_FOLDER, exist_ok=True)

print("=== 1. ADIM: VERİ HAZIRLIĞI ===")

# Veri
df = pd.read_csv(INPUT_FILE)
df = df.dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Label encode
lb = LabelEncoder()
y = lb.fit_transform(y)
print(f"Sınıflar: {lb.classes_}")

# Ölçekleme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified split (ÇOK ÖNEMLİ)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# =========================
# MODELLER
# =========================
classifiers = {
    "1-Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "2-SVM": SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    ),

    "3-KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    ),

    "4-Logistic Regression": LogisticRegression(
    max_iter=1500,
    solver="lbfgs",
    random_state=42
    ),


    "5-Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

ensembles = {
    "Ensemble-Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

voting_clf = VotingClassifier(
    estimators=[
        ('svm', classifiers["2-SVM"]),
        ('rf', classifiers["5-Random Forest"]),
        ('lr', classifiers["4-Logistic Regression"])
    ],
    voting='soft'
)

ensembles["Ensemble-Voting Classifier"] = voting_clf

all_models = {**classifiers, **ensembles}

# =========================
# EĞİTİM
# =========================
print("\n=== 2. ADIM: MODELLER EĞİTİLİYOR ===")
results = []

for name, model in all_models.items():
    print(f"\n--> {name}")
    start = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    sure = time.time() - start
    print(f"Accuracy: %{acc*100:.2f} | Süre: {sure:.2f} sn")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "Sure (sn)": sure
    })

    # --- Confusion Matrix (Raw) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=lb.classes_,
                yticklabels=lb.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.savefig(f"{RESULT_FOLDER}/{name}_cm_raw.png")
    plt.close()

    # --- Confusion Matrix (Normalized) ---
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=lb.classes_,
                yticklabels=lb.classes_)
    plt.title(f'{name} - Confusion Matrix (Normalized)')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.savefig(f"{RESULT_FOLDER}/{name}_cm_normalized.png")
    plt.close()

# =========================
# SONUÇLAR
# =========================
print("\n=== 3. ADIM: SONUÇLAR ===")
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print(results_df)
results_df.to_csv(f"{RESULT_FOLDER}/genel_sonuc_tablosu.csv", index=False)

print(f"\nTüm çıktılar '{RESULT_FOLDER}' klasörüne kaydedildi.")
