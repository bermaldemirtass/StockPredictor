import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veriyi oku
df = pd.read_csv("aapl_features.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Yalnızca kapanış fiyatını kullanalım (başlangıç için sade yapı)
data = df[["Close"]].values

# Normalize et (LSTM için önemli)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Zaman penceresi oluştur (örneğin 10 günlük geçmişe bakarak tahmin yap)
def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step])
        y.append(dataset[i+time_step])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(data_scaled, time_step)

# Veriyi train/test olarak ayır (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM girişine uygun şekle getir
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# LSTM modelini kur
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Eğit
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Tahmin yap
y_pred = model.predict(X_test)

# Tahmini eski ölçeğe geri dönüştür
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Hata hesapla
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print("LSTM MSE:", mse)
import pandas as pd
pd.DataFrame({"y_pred": y_pred_rescaled.flatten()}).to_csv("lstm_predictions.csv", index=False)




