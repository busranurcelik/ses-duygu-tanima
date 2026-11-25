import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# =========================
# MODELLER
# =========================
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# =========================
# AYARLAR
# =========================
INPUT_FILE = "final_oznitelik_seti.csv"
K_FOLD_VALUE = 10
RANDOM_STATE = 42

print(f"=== {K_FOLD_VALUE}-KAT ÇAPRAZ DOĞRULAMA (5 + 2 MODEL) ===")

# =========================
# VERİYİ YÜKLE
# =========================
df = pd.read_csv(INPUT_FILE).dropna()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Ölçekleme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 5 KLASİK MODEL
# =========================
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    ),

    # =========================
    # 2 ENSEMBLE
    # =========================
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
}

# Voting Ensemble (SVM + RF + LR)
voting_clf = VotingClassifier(
    estimators=[
        ("svm", classifiers["SVM"]),
        ("rf", classifiers["Random Forest"]),
        ("lr", classifiers["Logistic Regression"])
    ],
    voting="hard"
)

classifiers["Voting Ensemble"] = voting_clf

# =========================
# K-FOLD
# =========================
results = []

print("-" * 70)
print(f"{'MODEL':<25} | {'ORT. ACC (%)':<15} | {'STD (%)'}")
print("-" * 70)

kfold = StratifiedKFold(
    n_splits=K_FOLD_VALUE,
    shuffle=True,
    random_state=RANDOM_STATE
)

for name, model in classifiers.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="accuracy",
        n_jobs=-1
    )

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"{name:<25} | {mean_acc*100:.2f}          | ± {std_acc*100:.2f}")

    results.append({
        "Model": name,
        "K-Fold Mean Accuracy": mean_acc,
        "Standard Deviation": std_acc
    })

# =========================
# KAYDET
# =========================
os.makedirs("sonuclar_klasik", exist_ok=True)

results_df = pd.DataFrame(results).sort_values(
    by="K-Fold Mean Accuracy",
    ascending=False
)

results_df.to_csv(
    "sonuclar_klasik/k_fold_sonuclari.csv",
    index=False,
    encoding="utf-8-sig"
)

print("-" * 70)
print("K-Fold sonuçları kaydedildi: sonuclar_klasik/k_fold_sonuclari.csv")
