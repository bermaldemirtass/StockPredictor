import pandas as pd

# CSV'den veriyi oku
df = pd.read_csv(
    "aapl.csv",
    skiprows=2,
    names=["Date", "Close", "High", "Low", "Open", "Volume"],
    parse_dates=["Date"]
)



df.set_index("Date", inplace=True)

# Günlük getiri (Return)
df["Return"] = df["Close"].pct_change()

# Gecikmeli kapanış fiyatları
df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)

# Hareketli ortalamalar
df["SMA_5"] = df["Close"].rolling(window=5).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# Volatilite (standart sapma)
df["Volatility"] = df["Close"].rolling(window=5).std()

# NaN olan satırları temizle
df.dropna(inplace=True)

# Yeni veri setini kaydet
df.to_csv("aapl_features.csv")

print("Feature engineering tamamlandı. 'aapl_features.csv' dosyası oluşturuldu.")


