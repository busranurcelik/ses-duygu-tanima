import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write

import arayuz_A1_2_adapter as a12

# =========================
# PATH / AYARLAR
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = a12.SAMPLE_RATE
DURATION = a12.DURATION
TEMP_WAV = os.path.join(BASE_DIR, "temp_mic.wav")

# =========================
# GUI
# =========================
class DuyguAnaliziApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Ses Duygu Analizi")
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
            header, text="🎙️ Ses Duygu Analizi — A1.2 RF-B",
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

        self.status = tk.Label(root, text="A1.2 RF-B (Random Forest) modeli yükleniyor...", bd=1,
                               relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_path = None
        self._load_assets()

    # =========================
    def _load_assets(self):
        self.model = None
        self.scaler = None
        self.classes = None
        self.medians = None
        self.ready = False
        try:
            self.model, self.scaler, self.classes, self.medians = a12.load_assets()
            self.ready = True

            for i, (n, _, _) in enumerate(self.bars):
                n.config(text=self.classes[i])

            self.status.config(text="✅ A1.2 RF-B (Random Forest) hazır.")
        except Exception as e:
            msg = f"A1.2 RF-B model/scaler yüklenemedi; analiz devre dışı: {e}"
            self.status.config(text=f"❌ {msg}")
            self.btn_analyze.config(state="disabled")
            messagebox.showerror("A1.2 RF-B model yükleme hatası", msg)

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
        if not self.ready or self.model is None or self.scaler is None:
            msg = ("A1.2 RF-B modeli veya scaler yüklenemedi; tahmin yapılamaz. "
                   "A1.2 RF-B artefaktlarını kontrol edin.")
            self.status.config(text=f"❌ {msg}")
            messagebox.showerror("Model hatası", msg)
            self.btn_analyze.config(state="disabled")
            return

        try:
            self.btn_analyze.config(state="disabled", text="Analiz ediliyor...")
            self.root.update()

            # --- Ses yükle ---
            y, sr = librosa.load(self.file_path, sr=SAMPLE_RATE)

            # --- A1.2 192 özellik + scaler + RF-B tahmin ---
            probs, feats, f0_fallback = a12.predict_audio(
                self.model, self.scaler, self.classes, self.medians, y, sr
            )

            if feats.shape[0] != a12.N_FEATURES:
                raise RuntimeError(f"Özellik sayısı uyuşmuyor: {feats.shape[0]}")
            if len(probs) != len(self.classes):
                raise RuntimeError("RF-B olasılık sayısı label map ile uyuşmuyor.")

            idx = int(np.argmax(probs))
            max_prob = float(probs[idx])

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
                p = float(probs[i]) * 100
                b["value"] = p
                v.config(text=f"%{p:.0f}")
                if i == idx:
                    b.configure(style="green.Horizontal.TProgressbar")
                else:
                    b.configure(style="Horizontal.TProgressbar")

            note = " | F0 imputasyonu: evet" if f0_fallback else " | F0 imputasyonu: hayır"
            self.status.config(text="✅ Analiz tamamlandı." + note)

        except Exception as e:
            self.status.config(text=f"❌ Analiz hatası: {e}")
            messagebox.showerror("Analiz Hatası", str(e))

        finally:
            if self.ready:
                self.btn_analyze.config(state="normal", text="ANALİZ ET")


# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = DuyguAnaliziApp(root)
    root.mainloop()
