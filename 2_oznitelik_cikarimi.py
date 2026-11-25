import pandas as pd
import numpy as np
import librosa
import os
import sys

# =========================
# AYARLAR
# =========================
INPUT_FILE = "ham_veri_listesi.csv"
OUTPUT_FILE = "final_oznitelik_seti.csv"

SAMPLE_RATE = 22050
DURATION = 2.5  # saniye

print("NOT: Bu işlem 20–40 dk sürebilir")

# =========================
# VERİYİ OKU
# =========================
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"İşlenecek dosya sayısı: {len(df)}")
except FileNotFoundError:
    print("HATA: ham_veri_listesi.csv yok!")
    sys.exit()

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def preprocess_audio(y, sr):
    y = librosa.util.normalize(y)
    target_len = int(DURATION * sr)
    y = librosa.util.fix_length(y, size=target_len)
    return y

def extract_features_from_audio(y, sr):
    y = preprocess_audio(y, sr)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return np.hstack([
        mfcc,
        chroma,
        mel,
        np.mean(rms),
        np.std(rms),
        np.mean(zcr),
        np.std(zcr)
    ])

def extract_features(y, sr=SAMPLE_RATE):
    return extract_features_from_audio(y, sr)

# =========================
# AUGMENTATION
# =========================
def add_noise(y, level=0.015):
    noise_amp = level * np.random.uniform() * np.max(np.abs(y))
    return y + noise_amp * np.random.normal(size=len(y))

def pitch_shift(y, sr, steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y=y, rate=rate)

# =========================
# ANA DÖNGÜ
# =========================
X_data = []
Y_labels = []

counter = 0
error_count = 0
total = len(df)

print("İşlem başlıyor...")

for _, row in df.iterrows():
    path = row["path"]
    emotion = row["label"]
    source = row["source"]

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)

        # -----------------
        # ORİJİNAL
        # -----------------
        X_data.append(extract_features(y, sr))
        Y_labels.append(emotion)

        # -----------------
        # HERKES İÇİN NOISE
        # -----------------
        y_noise = add_noise(y)
        X_data.append(extract_features(y_noise, sr))
        Y_labels.append(emotion)

        # -----------------
        # FEAR / ANGRY / SAD / HAPPY → PITCH
        # -----------------
        y_pitch = pitch_shift(y, sr, steps=1)
        X_data.append(extract_features(y_pitch, sr))
        Y_labels.append(emotion)

        # -----------------
        # BNC BOOST 
        # -----------------
        if source == "BNC":
            # pitch down 
            y_pitch_low = pitch_shift(y, sr, steps=-2)
            X_data.append(extract_features(y_pitch_low, sr))
            Y_labels.append(emotion)

            # slow stretch
            y_stretch_slow = time_stretch(y, 0.9)
            X_data.append(extract_features(y_stretch_slow, sr))
            Y_labels.append(emotion)

            # fast stretch
            y_stretch_fast = time_stretch(y, 1.1)
            X_data.append(extract_features(y_stretch_fast, sr))
            Y_labels.append(emotion)

            # extra noise (mikrofon simülasyonu)
            y_noise2 = add_noise(add_noise(y, 0.02), 0.02)
            X_data.append(extract_features(y_noise2, sr))
            Y_labels.append(emotion)

        counter += 1

    except Exception as e:
        error_count += 1
        if error_count <= 5:
            print(f" HATA ({path}): {e}")
        continue

    if counter % 200 == 0:
        print(f"İşlenen: {counter}/{total}", end="\r")

# =========================
# KAYDET
# =========================
print("\n\nİşlem tamamlandı")
print(f"Başarılı: {counter}")
print(f"Hatalı: {error_count}")

df_out = pd.DataFrame(X_data)
df_out["labels"] = Y_labels
df_out.to_csv(OUTPUT_FILE, index=False)

print(f" '{OUTPUT_FILE}' oluşturuldu")
print(f"Toplam satır: {len(df_out)}")
print(f"Özellik sayısı: {df_out.shape[1]-1}")
