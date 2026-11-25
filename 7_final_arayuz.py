import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import librosa
import tensorflow as tf
import joblib
import sounddevice as sd
from scipy.io.wavfile import write

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "sonuclar_ysa")

MODEL_PATH = os.path.join(RESULT_DIR, "best_model.keras")
SCALER_PATH = os.path.join(RESULT_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(RESULT_DIR, "label_map.json")

# =========================
# AYARLAR (EĞİTİMLE AYNI)
# =========================
SAMPLE_RATE = 22050
DURATION = 2.5
TEMP_WAV = os.path.join(BASE_DIR, "temp_mic.wav")

# =========================
# FEATURE EXTRACTION (184)
# =========================
def preprocess_audio(y, sr):
    y = librosa.util.normalize(y)
    y = librosa.util.fix_length(y, size=int(DURATION * sr))
    return y

def extract_features(y, sr):
    y = preprocess_audio(y, sr)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), axis=1)

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

# =========================
# GUI
# =========================
class DuyguAnaliziApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Ses Duygu Analizi (YSA)")
        self.root.geometry("760x680")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.configure(
            "green.Horizontal.TProgressbar",
            foreground="green",
            background="green"
        )

        header = tk.Frame(root, bg="#2c3e50")
        header.pack(fill="x")
        tk.Label(
            header, text="🎙️ Ses Duygu Analizi (YSA)",
            font=("Arial", 16, "bold"),
            bg="#2c3e50", fg="white", pady=14
        ).pack()

        main = tk.Frame(root, padx=20, pady=15)
        main.pack(fill="both", expand=True)

        tk.Button(
            main, text="📂 WAV DOSYA SEÇ",
            command=self.dosya_sec,
            bg="#3498db", fg="white",
            font=("Arial", 12), width=26
        ).pack(pady=5)

        tk.Button(
            main, text="🎤 MİKROFONDAN KAYIT AL",
            command=self.mikrofon_kayit,
            bg="#9b59b6", fg="white",
            font=("Arial", 12), width=26
        ).pack(pady=5)

        self.lbl_file = tk.Label(main, text="Dosya / Mikrofon bekleniyor...", fg="#555")
        self.lbl_file.pack()

        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=12)

        self.btn_analyze = tk.Button(
            main, text="ANALİZ ET",
            command=self.analiz_et,
            bg="#27ae60", fg="white",
            font=("Arial", 14, "bold"),
            width=26,
            state="disabled"
        )
        self.btn_analyze.pack(pady=10)

        self.lbl_result = tk.Label(main, text="—", font=("Arial", 26, "bold"))
        self.lbl_result.pack(pady=8)

        self.lbl_probs = tk.Label(main, text="", font=("Consolas", 10))
        self.lbl_probs.pack(pady=6)

        self.bars = []
        for _ in range(4):
            row = tk.Frame(main)
            row.pack(fill="x", pady=2)
            name = tk.Label(row, text="—", width=10, anchor="w")
            name.pack(side="left")
            bar = ttk.Progressbar(row, length=430)
            bar.pack(side="left", padx=6)
            val = tk.Label(row, text="%0", width=6)
            val.pack(side="left")
            self.bars.append((name, bar, val))

        self.status = tk.Label(root, text="Model yükleniyor...", bd=1,
                               relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_path = None
        self._load_assets()

    # =========================
    def _load_assets(self):
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)

            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.classes = json.load(f)

            for i, (n, _, _) in enumerate(self.bars):
                n.config(text=self.classes[i])

            self.status.config(text="✅ Sistem hazır.")
        except Exception as e:
            self.status.config(text=f"❌ Yükleme hatası: {e}")

    # =========================
    def dosya_sec(self):
        path = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if path:
            self.file_path = path
            self.lbl_file.config(text=os.path.basename(path))
            self.btn_analyze.config(state="normal")

    # =========================
    def mikrofon_kayit(self):
        try:
            self.status.config(text="🎤 Kayıt alınıyor...")
            self.root.update()

            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            write(TEMP_WAV, SAMPLE_RATE, recording)
            self.file_path = TEMP_WAV

            self.lbl_file.config(text="Mikrofondan kayıt alındı")
            self.btn_analyze.config(state="normal")
            self.status.config(text="✅ Kayıt tamamlandı.")
        except Exception as e:
            messagebox.showerror("Mikrofon Hatası", str(e))

    # =========================
    def analiz_et(self):
        try:
            self.btn_analyze.config(state="disabled", text="Analiz ediliyor...")
            self.root.update()

            # --- Ses yükle ---
            y, sr = librosa.load(self.file_path, sr=SAMPLE_RATE)

            # --- Feature çıkar ---
            feats = extract_features(y, sr).reshape(1, -1)

            if feats.shape[1] != self.scaler.n_features_in_:
                raise ValueError("Feature sayısı uyuşmuyor!")

            # --- Ölçekle ---
            feats = self.scaler.transform(feats)

            # --- Tahmin ---
            probs = self.model.predict(feats, verbose=0)[0]

            # ================================
            # Amaç: Gerçekte happy iken sad/angry/fear'a kaymasını azaltmak
            # ================================
            idx = int(np.argmax(probs))
            max_prob = probs[idx]

            if "happy" in self.classes:
                happy_idx = self.classes.index("happy")
                happy_p = probs[happy_idx]

                # Ayarlanabilir eşikler
                HAPPY_MIN = 0.28        # happy bu değerin altındaysa zorlamıyoruz
                HAPPY_MARGIN = 0.06     # kazanana çok yakınsa happy seç

                # Eğer kazanan fear ise (happy ile sık karışıyorsa) toleransı artır
                if self.classes[idx] == "fear":
                    HAPPY_MARGIN = 0.08

                # Eğer kazanan sad ise (happy->sad kayması varsa) küçük bir tolerans daha
                if self.classes[idx] == "sad":
                    HAPPY_MARGIN = max(HAPPY_MARGIN, 0.07)

                # Eğer happy, max'a çok yakınsa ve yeterince büyükse -> happy seç
                if (happy_p >= HAPPY_MIN) and ((max_prob - happy_p) <= HAPPY_MARGIN):
                    idx = happy_idx
                    max_prob = happy_p
            # ================================

            # --- Hoca formatı çıktı ---
            self.lbl_result.config(
                text=f"%{max_prob*100:.1f} olasılıkla {self.classes[idx]}"
            )

            # --- Olasılık yazıları ---
            order = np.argsort(probs)[::-1]
            self.lbl_probs.config(
                text="   ".join(
                    [f"{self.classes[i]}: %{probs[i]*100:.1f}" for i in order]
                )
            )

            # --- Barlar ---
            for i, (n, b, v) in enumerate(self.bars):
                p = probs[i] * 100
                b["value"] = p
                v.config(text=f"%{p:.0f}")
                if i == idx:
                    b.configure(style="green.Horizontal.TProgressbar")
                else:
                    b.configure(style="Horizontal.TProgressbar")

            self.status.config(text="✅ Analiz tamamlandı.")

        except Exception as e:
            messagebox.showerror("Analiz Hatası", str(e))

        finally:
            self.btn_analyze.config(state="normal", text="ANALİZ ET")


# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = DuyguAnaliziApp(root)
    root.mainloop()
