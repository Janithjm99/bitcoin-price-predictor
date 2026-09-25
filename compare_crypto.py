# compare_crypto.py - Predict multiple cryptos

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

cryptos = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum', 
    'SOL-USD': 'Solana',
    'ADA-USD': 'Cardano'
}

results = []

for symbol, name in cryptos.items():
    print(f"\n🔄 Analyzing {name} ({symbol})...")
    
    try:
        data = yf.download(symbol, start='2020-01-01', end='2026-08-13', progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        df = data[['Close']].copy()
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
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        pred = rf.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        
        current = df['Close'].iloc[-1]
        tomorrow = rf.predict(scaler.transform(df.iloc[-1:][features]))[0]
        
        results.append({
            'Crypto': name,
            'Symbol': symbol,
            'Current Price': f"${current:,.2f}",
            'Tomorrow Price': f"${tomorrow:,.2f}",
            'Expected Change': f"{(tomorrow/current - 1)*100:+.2f}%",
            'RMSE': f"${rmse:,.2f}",
            'MAE': f"${mae:,.2f}",
            'Predictability': 'High' if mae/current < 0.05 else 'Medium' if mae/current < 0.10 else 'Low'
        })
        
    except Exception as e:
        print(f"⚠️ Failed: {e}")

# Display results
print("\n" + "="*80)
print("📊 CRYPTOCURRENCY PRICE PREDICTION COMPARISON")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find the most predictable
best = results_df.loc[results_df['MAE'].str.replace('$', '').str.replace(',', '').astype(float).idxmin()]
print(f"\n🏆 Most predictable: {best['Crypto']} (MAE: {best['MAE']})")