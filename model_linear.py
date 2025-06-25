import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Veriyi oku
df = pd.read_csv("aapl_features.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Özellikler ve hedef
features = ["Lag_1", "Lag_2", "SMA_5", "SMA_20", "Volatility"]
X = df[features]
y = df["Close"]

# Veriyi train/test olarak ayır (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Modeli eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Hata hesapla
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression MSE:", mse)

# Tahminleri dosyaya yaz
pd.DataFrame({"y_pred": y_pred}).to_csv("linear_predictions.csv", index=False)


# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
predictions = model.predict(X_test)

# Hata hesapla
mse = mean_squared_error(y_test, predictions)
print("Linear Regression MSE:", mse)


