import yfinance as yf

# Apple hissesi, 2 yıllık veri
data = yf.download("AAPL", start="2022-01-01", end="2024-12-31")

# 'Date' sütununu veri içine al, index olarak bırakma
data.reset_index(inplace=True)

# CSV'ye düzgün şekilde yaz (ekstra başlıklar olmadan)
data.to_csv("aapl.csv", index=False)

print("Veri temiz şekilde 'aapl.csv' dosyasına kaydedildi.")




