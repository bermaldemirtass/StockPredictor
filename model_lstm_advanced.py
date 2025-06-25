import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veriyi oku
df = pd.read_csv("aapl_features.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Özellikleri belirle
features = ["Close", "Lag_1", "Lag_2", "SMA_5"]

data = df[features].dropna().values  # NaN satırları sil

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Zaman serisi verisini pencereleme
def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step])
        y.append(dataset[i+time_step][0])  # Close sütunu hedef
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(data_scaled, time_step)

# Eğitim ve test ayrımı
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Şekillendirme
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Eğitim
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)


# Tahmin
y_pred = model.predict(X_test)

# Sadece Close için inverse_transform
close_min = scaler.data_min_[0]
close_scale = scaler.data_range_[0]
y_pred_rescaled = y_pred * close_scale + close_min
y_test_rescaled = y_test * close_scale + close_min

# Hata metriği
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print("İyileştirilmiş LSTM MSE:", mse)


