# 🚀 [Bitcoin Price Predictor](https://bitcoin-price-predictor-4.streamlit.app/)

https://bitcoin-price-predictor-4.streamlit.app/

A machine learning project that predicts Bitcoin prices using multiple algorithms (Random Forest, XGBoost, ARIMA, LSTM) with an interactive Streamlit dashboard for real-time predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dashboard](#dashboard)
- [Technical Indicators](#technical-indicators)
- [Results](#results)
- [Disclaimer](#disclaimer)

## 🎯 Overview

This project downloads historical Bitcoin data from Yahoo Finance, engineers technical features, trains multiple machine learning models, and provides both command-line scripts and an interactive web dashboard for price predictions.

**Data Source:** Yahoo Finance (BTC-USD)  
**Time Range:** 2020-01-01 to 2026-08-13 (configurable)  
**Prediction Target:** Next day's closing price

## ✨ Features

- **Multiple ML Models**: Random Forest, XGBoost, ARIMA, LSTM
- **Feature Engineering**: 20+ technical indicators (MA, RSI, Bollinger Bands, Volume ratios, etc.)
- **Interactive Dashboard**: Real-time predictions with Streamlit + Plotly
- **Multi-Crypto Support**: Compare Bitcoin, Ethereum, Solana, Cardano
- **Model Persistence**: Save/load trained models with joblib
- **Visualization**: Price charts, feature importance, prediction intervals

## 📁 Project Structure

```
bitcoin-price-predictor/
├── main.py                 # Entry point - runs complete pipeline
├── bitcoin_predictor.py    # Core ML pipeline (Random Forest + ARIMA)
├── dashboard.py            # Streamlit web dashboard
├── improved_features.py    # Extended feature engineering (20+ indicators)
├── compare_crypto.py       # Multi-cryptocurrency comparison
├── xgboost_model.py        # XGBoost implementation
├── lstm_model.py           # LSTM deep learning model
├── requirements.txt        # Python dependencies
├── bitcoin_rf_model.pkl    # Saved Random Forest model
├── bitcoin_scaler.pkl      # Saved MinMaxScaler
├── bitcoin_exploration.png # Data exploration plots
└── bitcoin_predictions.png # Prediction result plots
```

## 🛠 Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd bitcoin-price-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >=1.28.0 | Web dashboard |
| plotly | >=5.17.0 | Interactive charts |
| yfinance | >=0.2.28 | Yahoo Finance data |
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.3.0 | ML models & preprocessing |
| joblib | >=1.3.0 | Model persistence |

For XGBoost model: `pip install xgboost`  
For LSTM model: `pip install tensorflow`

## 🚀 Usage

### 1. Run Complete Pipeline (CLI)

```bash
python main.py
```

This executes the full pipeline:
- Downloads Bitcoin data
- Performs exploratory data analysis
- Engineers features
- Trains ARIMA and Random Forest models
- Evaluates performance
- Generates visualizations
- Saves model artifacts

**Output files:**
- `bitcoin_exploration.png` - Price history, volume, returns distribution
- `bitcoin_predictions.png` - Model comparisons, residuals, feature importance
- `bitcoin_rf_model.pkl` - Trained Random Forest model
- `bitcoin_scaler.pkl` - Fitted feature scaler

### 2. Run Individual Scripts

```bash
# Core predictor (Random Forest + ARIMA)
python bitcoin_predictor.py

# Advanced features with more indicators
python improved_features.py

# Compare multiple cryptocurrencies
python compare_crypto.py

# XGBoost model
python xgboost_model.py

# LSTM deep learning model
python lstm_model.py
```

### 3. Launch Interactive Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`

## 🤖 Models

### Random Forest (Default)
- **Algorithm**: Ensemble of decision trees
- **Features**: 8 technical indicators (MA_7, MA_30, Volatility, Returns, Momentum, Lag_1-3)
- **Performance**: RMSE ~$7,478 | MAE ~$5,216
- **Strengths**: Handles non-linear relationships, feature importance available

### ARIMA
- **Algorithm**: Autoregressive Integrated Moving Average
- **Order**: (5, 1, 0)
- **Strengths**: Classical time series approach, good for stationary series

### XGBoost
- **Algorithm**: Gradient Boosting
- **Parameters**: n_estimators=200, max_depth=6, learning_rate=0.1
- **Strengths**: Often outperforms Random Forest on tabular data

### LSTM (Deep Learning)
- **Architecture**: 2 LSTM layers (50 units) + Dropout + Dense layers
- **Sequence Length**: 60 days lookback
- **Strengths**: Captures temporal dependencies in sequential data

## 📊 Dashboard Features

The Streamlit dashboard provides:

| Section | Description |
|---------|-------------|
| **Live Metrics** | Current price, tomorrow's prediction, confidence range, trading signal |
| **Price Chart** | Interactive Plotly chart with candlesticks, MAs, Bollinger Bands, volume, RSI |
| **Feature Importance** | Horizontal bar chart showing what drives predictions |
| **Market Statistics** | 52-week high/low, average volume, volatility |
| **Trading Insights** | RSI signals, MA crossovers, model-based signals |
| **Raw Data** | View/download recent data as CSV |

### Dashboard Controls
- **Cryptocurrency Selector**: BTC, ETH, SOL, ADA, DOGE
- **Model Selector**: Random Forest, XGBoost, ARIMA
- **Technical Indicators Toggle**: Show/hide MAs, Bollinger Bands
- **Refresh Button**: Fetch latest data

## 🔧 Technical Indicators

### Basic Features (bitcoin_predictor.py)
- `MA_7`, `MA_30` - Moving Averages (7 & 30 day)
- `Volatility` - 30-day rolling standard deviation of returns
- `Returns` - Daily percentage change
- `Momentum` - Price minus 7-day MA
- `Lag_1`, `Lag_2`, `Lag_3` - Previous 1-3 day closing prices

### Advanced Features (improved_features.py)
- `MA_90` - 90-day moving average
- `Volatility_7` - 7-day volatility
- `RSI` - Relative Strength Index (14-day)
- `BB_Upper`, `BB_Middle`, `BB_Lower` - Bollinger Bands (20-day, 2σ)
- `BB_Position` - Position within Bollinger Bands (0-1)
- `Volume_MA`, `Volume_Ratio` - Volume moving average & ratio
- `High_Low_Ratio`, `Close_Open_Ratio` - Intraday price patterns
- `Lag_1` through `Lag_5` - 5 days of lag features
- `DayOfWeek`, `IsWeekend` - Temporal features
- `Month` - Seasonal patterns

## 📈 Results

### Model Performance Comparison

| Model | RMSE | MAE | Notes |
|-------|------|-----|-------|
| Random Forest | ~$7,478 | ~$5,216 | Baseline with 8 features |
| Random Forest (Improved) | ~$X,XXX | ~$X,XXX | With 20+ advanced features |
| XGBoost | ~$X,XXX | ~$X,XXX | Gradient boosting |
| ARIMA | ~$X,XXX | ~$X,XXX | Classical time series |
| LSTM | ~$X,XXX | ~$X,XXX | Deep learning |

### Feature Importance (Random Forest)
Typical ranking:
1. **Lag_1** - Yesterday's price (strongest predictor)
2. **MA_7** / **MA_30** - Trend indicators
3. **Returns** - Daily momentum
4. **Volatility** - Market uncertainty measure

## ⚠️ Disclaimer

> **This is an educational project and should NOT be used for real trading decisions.**
> 
> Cryptocurrency markets are highly volatile and unpredictable. Past performance does not guarantee future results. Always do your own research (DYOR) and consult with a financial advisor before investing.

## 📝 License

This project is for educational purposes only.

## 🤝 Contributing

Feel free to fork and experiment with:
- Additional technical indicators
- Different model architectures
- Alternative data sources
- Ensemble methods
- Hyperparameter optimization

---

**Built with ❤️ using Python, scikit-learn, Streamlit, and Plotly**  
**Data provided by Yahoo Finance**
