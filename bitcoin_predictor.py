# bitcoin_predictor.py - FIXED VERSION

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# STEP 1: DOWNLOAD DATA
# ============================================

print("📥 Downloading Bitcoin data...")
btc = yf.download('BTC-USD', start='2020-01-01', end='2026-08-13')
print(f"✅ Data downloaded! {len(btc)} rows")

# FIX: Flatten the column structure if it's multi-index
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = ['_'.join(col).strip() for col in btc.columns.values]

# Rename columns to simple names
btc.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

print("\n📊 First 5 rows of data:")
print(btc.head())

print(f"\n🔍 Missing values: {btc.isnull().sum().sum()}")

# ============================================
# STEP 2: EXPLORE THE DATA
# ============================================

print("\n📈 Basic Statistics:")
print(btc['Close'].describe())

# Create a figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Closing price over time
axes[0, 0].plot(btc.index, btc['Close'], color='blue', linewidth=1)
axes[0, 0].set_title('Bitcoin Closing Price Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Volume over time - FIXED: use .values to get 1D array
axes[0, 1].bar(btc.index, btc['Volume'].values, color='orange', alpha=0.7, width=1)
axes[0, 1].set_title('Bitcoin Trading Volume')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Histogram of returns
returns = btc['Close'].pct_change().dropna()
axes[1, 0].hist(returns, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Daily Returns')
axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)

# Plot 4: Box plot of closing prices by year
btc['Year'] = btc.index.year
# FIX: Use boxplot with positions
btc.boxplot(column='Close', by='Year', ax=axes[1, 1])
axes[1, 1].set_title('Closing Prices by Year')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Price (USD)')

plt.tight_layout()
plt.savefig('bitcoin_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Exploration plots saved as 'bitcoin_exploration.png'")

# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================

print("\n" + "="*50)
print("STEP 3: Creating features...")
print("="*50)

# Create a copy of the close price
df = btc[['Close']].copy()

# Add technical indicators (features)
print("   ➤ Adding moving averages...")
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

print("   ➤ Adding volatility...")
df['Volatility'] = df['Close'].pct_change().rolling(window=30).std()

print("   ➤ Adding returns...")
df['Returns'] = df['Close'].pct_change()

print("   ➤ Adding momentum...")
df['Momentum'] = df['Close'] - df['MA_7']

print("   ➤ Adding lag features...")
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)

# Remove rows with NaN values (from the rolling calculations)
df_clean = df.dropna()

print(f"\n✅ Cleaned data: {len(df_clean)} rows remaining")
print(f"✅ Feature columns: {len(df_clean.columns)} features")

# ============================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================

print("\n" + "="*50)
print("STEP 4: Preparing data for modeling...")
print("="*50)

# We want to predict tomorrow's price using today's features
# So we shift the target (Close) backwards by 1 day
df_clean['Target'] = df_clean['Close'].shift(-1)

# Remove the last row (where we don't have a target)
df_model = df_clean.dropna()

print(f"✅ Model data: {len(df_model)} rows with targets")

# Split data into training (80%) and testing (20%)
# IMPORTANT: For time series, DON'T shuffle! Use chronological split
split_idx = int(len(df_model) * 0.8)

train = df_model.iloc[:split_idx]
test = df_model.iloc[split_idx:]

print(f"📊 Training data: {len(train)} rows ({((len(train)/len(df_model))*100):.1f}%)")
print(f"📊 Testing data: {len(test)} rows ({((len(test)/len(df_model))*100):.1f}%)")

# Define features and target
feature_cols = ['MA_7', 'MA_30', 'Volatility', 'Returns', 'Momentum', 'Lag_1', 'Lag_2', 'Lag_3']
X_train = train[feature_cols]
y_train = train['Target']
X_test = test[feature_cols]
y_test = test['Target']

# Scale the features
print("   ➤ Scaling features...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features ready: {X_train_scaled.shape[1]} features")

# ============================================
# STEP 5A: ARIMA MODEL
# ============================================

print("\n" + "="*50)
print("STEP 5A: Training ARIMA model...")
print("="*50)

train_close = train['Close']
test_close = test['Close']

try:
    # ARIMA(order=(p,d,q)) - p=autoregressive, d=differencing, q=moving average
    arima_model = ARIMA(train_close, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    
    # Make predictions
    arima_pred = arima_fit.forecast(steps=len(test_close))
    
    # Calculate errors
    arima_rmse = np.sqrt(mean_squared_error(test_close, arima_pred))
    arima_mae = mean_absolute_error(test_close, arima_pred)
    
    print(f"✅ ARIMA Model Performance:")
    print(f"   RMSE: ${arima_rmse:,.2f}")
    print(f"   MAE:  ${arima_mae:,.2f}")
    
except Exception as e:
    print(f"⚠️ ARIMA failed: {e}")
    arima_pred = None
    arima_rmse = None

# ============================================
# STEP 5B: RANDOM FOREST MODEL
# ============================================

print("\n" + "="*50)
print("STEP 5B: Training Random Forest model...")
print("="*50)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("   ➤ Training model...")
rf_model.fit(X_train_scaled, y_train)

# Make predictions
print("   ➤ Making predictions...")
rf_pred = rf_model.predict(X_test_scaled)

# Calculate errors
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"\n✅ Random Forest Performance:")
print(f"   RMSE: ${rf_rmse:,.2f}")
print(f"   MAE:  ${rf_mae:,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔥 Feature Importance:")
print(feature_importance.to_string(index=False))

# ============================================
# STEP 6: VISUALIZE RESULTS
# ============================================

print("\n" + "="*50)
print("STEP 6: Visualizing predictions...")
print("="*50)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Actual vs Predictions
ax1 = axes[0, 0]
ax1.plot(test.index, test_close, label='Actual Price', color='blue', linewidth=2)
if arima_pred is not None:
    ax1.plot(test.index, arima_pred, label='ARIMA Prediction', color='red', linestyle='--', linewidth=2)
ax1.plot(test.index, rf_pred, label='Random Forest Prediction', color='green', linestyle='-.', linewidth=2)
ax1.set_title('Bitcoin Price Prediction - Models Comparison')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Random Forest Performance
ax2 = axes[0, 1]
ax2.scatter(y_test, rf_pred, alpha=0.6, color='purple')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax2.set_title('Random Forest: Predicted vs Actual')
ax2.set_xlabel('Actual Price (USD)')
ax2.set_ylabel('Predicted Price (USD)')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals (errors)
ax3 = axes[1, 0]
residuals = y_test - rf_pred
ax3.plot(test.index, residuals, color='orange', linewidth=1)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax3.set_title('Prediction Errors (Residuals)')
ax3.set_xlabel('Date')
ax3.set_ylabel('Error (USD)')
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance
ax4 = axes[1, 1]
feature_importance.plot(x='Feature', y='Importance', kind='bar', ax=ax4, color='teal', legend=False)
ax4.set_title('Feature Importance - Random Forest')
ax4.set_xlabel('Features')
ax4.set_ylabel('Importance Score')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bitcoin_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Prediction plots saved as 'bitcoin_predictions.png'")

# ============================================
# STEP 7: MAKE FUTURE PREDICTIONS
# ============================================

print("\n" + "="*50)
print("STEP 7: Predicting tomorrow's price...")
print("="*50)

# Get the latest available data
latest_data = df_clean.iloc[-1:][feature_cols]

# Scale it
latest_scaled = scaler.transform(latest_data)

# Make prediction
tomorrow_price = rf_model.predict(latest_scaled)[0]
current_price = df_clean['Close'].iloc[-1]

print(f"\n💰 Current Bitcoin Price: ${current_price:,.2f}")
print(f"📈 Predicted Price for Tomorrow: ${tomorrow_price:,.2f}")
print(f"📊 Expected Change: ${tomorrow_price - current_price:,.2f} ({(tomorrow_price/current_price - 1)*100:.2f}%)")

# Prediction interval (rough estimate)
test_errors = np.abs(y_test - rf_pred)
avg_error = np.mean(test_errors)

print(f"\n🎯 Prediction Range: ${tomorrow_price - avg_error:,.2f} to ${tomorrow_price + avg_error:,.2f} (with ±{avg_error:,.2f} average error)")

# Save model for later use
import joblib
joblib.dump(rf_model, 'bitcoin_rf_model.pkl')
joblib.dump(scaler, 'bitcoin_scaler.pkl')
print("\n💾 Model and scaler saved as 'bitcoin_rf_model.pkl' and 'bitcoin_scaler.pkl'")

print("\n" + "="*50)
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)
print("\n📁 Generated files:")
print("  • bitcoin_exploration.png - Initial data exploration")
print("  • bitcoin_predictions.png - Model predictions")
print("  • bitcoin_rf_model.pkl - Saved machine learning model")
print("  • bitcoin_scaler.pkl - Saved data scaler")