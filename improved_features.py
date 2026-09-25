# improved_features.py - Add more technical indicators

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("📥 Downloading Bitcoin data...")
btc = yf.download('BTC-USD', start='2020-01-01', end='2026-08-13')

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = ['_'.join(col).strip() for col in btc.columns.values]
btc.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

print("Creating ADVANCED features...")
df = btc[['Close', 'High', 'Low', 'Open', 'Volume']].copy()

# 1. Price-based features
df['MA_7'] = df['Close'].rolling(7).mean()
df['MA_30'] = df['Close'].rolling(30).mean()
df['MA_90'] = df['Close'].rolling(90).mean()  # NEW - longer trend

# 2. Volatility features
df['Volatility_7'] = df['Close'].pct_change().rolling(7).std()  # NEW
df['Volatility_30'] = df['Close'].pct_change().rolling(30).std()

# 3. RSI - Relative Strength Index (NEW - very popular indicator)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Close'], 14)

# 4. Bollinger Bands (NEW)
df['BB_Middle'] = df['Close'].rolling(20).mean()
df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(20).std()
df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(20).std()
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])  # 0=lower, 1=upper

# 5. Volume-based features (NEW)
df['Volume_MA'] = df['Volume'].rolling(30).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

# 6. Price patterns (NEW)
df['High_Low_Ratio'] = df['High'] / df['Low']
df['Close_Open_Ratio'] = df['Close'] / df['Open']

# 7. Lag features (more of them)
for i in range(1, 6):  # 5 days of lags instead of 3
    df[f'Lag_{i}'] = df['Close'].shift(i)

# 8. Day of week (NEW - crypto sometimes has weekend patterns)
df['DayOfWeek'] = df.index.dayofweek  # Monday=0, Sunday=6
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 9. Month (NEW - seasonal patterns)
df['Month'] = df.index.month

# Remove NaN rows
df = df.dropna()

# Create target
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

print(f"✅ Created {len(df.columns)} features")

# Define all features (exclude non-feature columns)
exclude_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Target']
features = [col for col in df.columns if col not in exclude_cols]
print(f"📊 Using {len(features)} features")

# Split and train
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train[features]
y_train = train['Target']
X_test = test[features]
y_test = test['Target']

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest with more features...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

pred = rf.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print(f"\n✅ Improved RMSE: ${rmse:,.2f} (was $7,477.69)")
print(f"✅ Improved MAE: ${mae:,.2f} (was $5,216.12)")

# Feature importance
importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
print("\n🔥 Top 10 Features:")
print(importance.head(10))

# Predict tomorrow
latest = df.iloc[-1:][features]
latest_scaled = scaler.transform(latest)
tomorrow = rf.predict(latest_scaled)[0]
current = df['Close'].iloc[-1]

print(f"\n💰 Current Price: ${current:,.2f}")
print(f"📈 Tomorrow's Prediction: ${tomorrow:,.2f}")
print(f"📊 Change: {(tomorrow/current - 1)*100:.2f}%")