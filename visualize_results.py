import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Veriyi oku
df = pd.read_csv("aapl_features.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df = df.dropna()

# Gerçek değerler
actual = df["Close"].iloc[-len(df)//5:].values  # Son %20 test verisi

# Linear Regression tahmini oku
linear_pred = pd.read_csv("linear_predictions.csv")["y_pred"].values

# LSTM tahmini oku
lstm_pred = pd.read_csv("lstm_predictions.csv")["y_pred"].values

# Grafik
plt.figure(figsize=(12,6))
plt.plot(actual, label="Gerçek Değerler", linewidth=2)
plt.plot(linear_pred, label="Linear Regression", linestyle='--')
plt.plot(lstm_pred, label="LSTM", linestyle=':')
plt.title("Model Karşılaştırması")
plt.xlabel("Zaman")
plt.ylabel("Kapanış Fiyatı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()



