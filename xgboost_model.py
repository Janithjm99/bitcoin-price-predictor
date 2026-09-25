# xgboost_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("📥 Downloading Bitcoin data...")
btc = yf.download('BTC-USD', start='2020-01-01', end='2026-08-13')

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = ['_'.join(col).strip() for col in btc.columns.values]
btc.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Simple features (same as before but we'll use XGBoost)
df = btc[['Close']].copy()
df['MA_7'] = df['Close'].rolling(7).mean()
df['MA_30'] = df['Close'].rolling(30).mean()
df['Volatility'] = df['Close'].pct_change().rolling(30).std()
df['Returns'] = df['Close'].pct_change()
df['Momentum'] = df['Close'] - df['MA_7']

for i in range(1, 4):
    df[f'Lag_{i}'] = df['Close'].shift(i)

df = df.dropna()
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

# Split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

features = ['MA_7', 'MA_30', 'Volatility', 'Returns', 'Momentum', 'Lag_1', 'Lag_2', 'Lag_3']
X_train = train[features]
y_train = train['Target']
X_test = test[features]
y_test = test['Target']

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training XGBoost model...")
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

pred = xgb.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print(f"\n✅ XGBoost RMSE: ${rmse:,.2f}")
print(f"✅ XGBoost MAE: ${mae:,.2f}")