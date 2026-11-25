import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Dosya yolları
INPUT_FILE = "final_oznitelik_seti.csv"
SCALER_FILE = "scaler.pkl"

print("Veri seti okunuyor...")
try:
    df = pd.read_csv(INPUT_FILE)
    
    # Sadece özellikleri al (son sütun etiket olduğu için atıyoruz)
    X = df.iloc[:, :-1].values
    
    print("Scaler eğitiliyor...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    print("Scaler kaydediliyor...")
    joblib.dump(scaler, SCALER_FILE)
    
    print(f"BAŞARILI! '{SCALER_FILE}' dosyası oluşturuldu.")
    print("Şimdi arayüz kodunu çalıştırabilirsin.")
    
except Exception as e:
    print(f"HATA: {e}")