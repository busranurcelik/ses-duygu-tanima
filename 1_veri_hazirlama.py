import os
import pandas as pd

# =========================
# AYARLAR
# =========================
DATA_PATH = "./data"

# KULLANILACAK DUYGULAR
TARGET_EMOTIONS = ['happy', 'sad', 'angry', 'fear']

data_list = []

print("=== VERİ TARAMA (V4 - FEAR EKLİ / NEUTRAL YOK) BAŞLIYOR ===")

# ---------------------------------------------------------
# 1. RAVDESS
# ---------------------------------------------------------
ravdess_map = {
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear'      
}

ravdess_dir = os.path.join(DATA_PATH, "RAVDESS")
ravdess_count = 0

if os.path.exists(ravdess_dir):
    for root, _, files in os.walk(ravdess_dir):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) > 2:
                    emotion_code = parts[2]
                    if emotion_code in ravdess_map:
                        data_list.append({
                            'path': os.path.join(root, file),
                            'label': ravdess_map[emotion_code],
                            'source': 'RAVDESS'
                        })
                        ravdess_count += 1
    print(f"[OK] RAVDESS eklendi: {ravdess_count}")
else:
    print("[HATA] RAVDESS bulunamadı")

# ---------------------------------------------------------
# 2. TESS
# ---------------------------------------------------------
tess_dir = os.path.join(DATA_PATH, "TESS")
tess_count = 0

if os.path.exists(tess_dir):
    for root, _, files in os.walk(tess_dir):
        for file in files:
            if file.endswith('.wav'):
                f = file.lower()
                emotion = None

                if 'happy' in f:
                    emotion = 'happy'
                elif 'sad' in f:
                    emotion = 'sad'
                elif 'angry' in f:
                    emotion = 'angry'
                elif 'fear' in f:
                    emotion = 'fear'

                if emotion in TARGET_EMOTIONS:
                    data_list.append({
                        'path': os.path.join(root, file),
                        'label': emotion,
                        'source': 'TESS'
                    })
                    tess_count += 1

    print(f"[OK] TESS eklendi: {tess_count}")
else:
    print("[HATA] TESS bulunamadı")

# ---------------------------------------------------------
# 3. SAVEE
# ---------------------------------------------------------
savee_map = {
    '_a': 'angry',
    '_h': 'happy',
    '_sa': 'sad',
    '_f': 'fear'      
}

savee_dir = os.path.join(DATA_PATH, "SAVEE")
savee_count = 0

if os.path.exists(savee_dir):
    for root, _, files in os.walk(savee_dir):
        for file in files:
            if file.endswith('.wav'):
                f = file.lower()
                for key, emotion in savee_map.items():
                    if key in f:
                        data_list.append({
                            'path': os.path.join(root, file),
                            'label': emotion,
                            'source': 'SAVEE'
                        })
                        savee_count += 1
                        break

    print(f"[OK] SAVEE eklendi: {savee_count}")
else:
    print("[HATA] SAVEE bulunamadı")

# ---------------------------------------------------------
# 4. CREMA-D
# ---------------------------------------------------------
crema_map = {
    'HAP': 'happy',
    'SAD': 'sad',
    'ANG': 'angry',
    'FEA': 'fear'     
}

crema_dir = os.path.join(DATA_PATH, "CREMA-D")
crema_count = 0

if os.path.exists(crema_dir):
    for root, _, files in os.walk(crema_dir):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('_')
                if len(parts) >= 3:
                    code = parts[2]
                    if code in crema_map:
                        data_list.append({
                            'path': os.path.join(root, file),
                            'label': crema_map[code],
                            'source': 'CREMA'
                        })
                        crema_count += 1
    print(f"[OK] CREMA-D eklendi: {crema_count}")
else:
    print("[HATA] CREMA-D bulunamadı")

# ---------------------------------------------------------
# 5. BNC (KENDİ VERİSETİM)
# ---------------------------------------------------------
bnc_dir = os.path.join(DATA_PATH, "BNCS-SES-WAV")
bnc_count = 0

if os.path.exists(bnc_dir):
    for emotion in TARGET_EMOTIONS:
        emotion_dir = os.path.join(bnc_dir, emotion)
        if os.path.exists(emotion_dir):
            for file in os.listdir(emotion_dir):
                if file.endswith('.wav'):
                    data_list.append({
                        'path': os.path.join(emotion_dir, file),
                        'label': emotion,
                        'source': 'BNC'
                    })
                    bnc_count += 1

    print(f"[OK] BNC eklendi: {bnc_count}")
else:
    print("[HATA] BNC bulunamadı")

# ---------------------------------------------------------
# RAPOR & KAYIT
# ---------------------------------------------------------
print("\n------------------------------------------------")

if len(data_list) > 0:
    df = pd.DataFrame(data_list)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("ham_veri_listesi.csv", index=False)

    print("İŞLEM TAMAMLANDI ")
    print(f"Toplam dosya sayısı: {len(df)}")

    print("\nKaynak Dağılımı:")
    print(df['source'].value_counts())

    print("\nSınıf Dağılımı:")
    print(df['label'].value_counts())
else:
    print("HATA: Hiç veri bulunamadı!")
