# lstm_model.py - Deep learning with LSTM
# pip install tensorflow

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("📥 Downloading Bitcoin data...")
btc = yf.download('BTC-USD', start='2020-01-01', end='2026-08-13')

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = ['_'.join(col).strip() for col in btc.columns.values]
btc.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Use only closing price
data = btc['Close'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences (look back 60 days)
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training data: {len(X_train)} sequences")
print(f"Testing data: {len(X_test)} sequences")

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n🧠 Training LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Predict
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
y_test_actual = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
mae = mean_absolute_error(y_test_actual, pred)

print(f"\n✅ LSTM RMSE: ${rmse:,.2f}")
print(f"✅ LSTM MAE: ${mae:,.2f}")

# Predict tomorrow
last_60 = data_scaled[-seq_length:].reshape(1, seq_length, 1)
tomorrow_scaled = model.predict(last_60)
tomorrow = scaler.inverse_transform(tomorrow_scaled)[0][0]
current = data[-1][0]

print(f"\n💰 Current Price: ${current:,.2f}")
print(f"📈 Tomorrow's Prediction (LSTM): ${tomorrow:,.2f}")