# dashboard.py - Complete Bitcoin Price Predictor Dashboard

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS (makes it look professional)
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f7931a 0%, #ffd700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 0.5rem;
    }
    .positive {
        color: #00ff88;
    }
    .negative {
        color: #ff4444;
    }
    .disclaimer {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin-top: 2rem;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🚀 Bitcoin Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning powered Bitcoin price prediction for tomorrow</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - Settings
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Symbol selection
    symbol = st.selectbox(
        "Select Cryptocurrency",
        ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"],
        index=0
    )
    
    # Date range
    st.markdown("### 📅 Data Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    st.info(f"📊 Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Model selection
    st.markdown("### 🤖 Model")
    model_type = st.radio(
        "Select Model",
        ["Random Forest", "XGBoost", "ARIMA"],
        index=0,
        help="Random Forest is the default model currently loaded"
    )
    
    # Feature toggle
    st.markdown("### 📊 Show Advanced Features")
    show_feature_importance = st.checkbox("Feature Importance", value=True)
    show_technical_indicators = st.checkbox("Technical Indicators", value=True)
    
    # Action buttons
    st.markdown("---")
    refresh_data = st.button("🔄 Refresh Data", use_container_width=True)
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('bitcoin_rf_model.pkl')
        scaler = joblib.load('bitcoin_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run bitcoin_predictor.py first.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

model, scaler = load_model()

# ============================================
# FETCH DATA
# ============================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(symbol, start_date, end_date):
    """Fetch cryptocurrency data from Yahoo Finance"""
    try:
        with st.spinner(f'📥 Fetching {symbol} data...'):
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
        if data.empty:
            st.error("❌ No data found for this symbol")
            return None
            
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        
        # Standardize column names
        data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

# Fetch data
data = fetch_data(symbol, start_date, end_date)

if data is None:
    st.stop()

# ============================================
# CALCULATE FEATURES
# ============================================
def calculate_features(df):
    """Calculate all technical features"""
    df = df.copy()
    
    # Moving averages
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['MA_30'] = df['Close'].rolling(30).mean()
    df['MA_90'] = df['Close'].rolling(90).mean()
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(30).std()
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Momentum
    df['Momentum'] = df['Close'] - df['MA_7']
    
    # RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(20).std()
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Lag features
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df

features_df = calculate_features(data)

# Define feature columns (must match the training data)
feature_cols = ['MA_7', 'MA_30', 'Volatility', 'Returns', 'Momentum', 'Lag_1', 'Lag_2', 'Lag_3']

# ============================================
# MAKE PREDICTION
# ============================================
def make_prediction(model, scaler, features_df, feature_cols):
    """Make a prediction for tomorrow's price"""
    if model is None or scaler is None:
        return None, None, None
    
    try:
        # Get latest data
        latest = features_df.iloc[-1:][feature_cols]
        
        # Scale
        latest_scaled = scaler.transform(latest)
        
        # Predict
        prediction = model.predict(latest_scaled)[0]
        
        # Current price
        current_price = features_df['Close'].iloc[-1]
        
        return prediction, current_price, None
    except Exception as e:
        return None, None, str(e)

# Make prediction
prediction, current_price, error = make_prediction(model, scaler, features_df, feature_cols)

# ============================================
# MAIN METRICS DISPLAY
# ============================================
st.markdown("## 📊 Live Dashboard")

# Create 4 columns for key metrics
col1, col2, col3, col4 = st.columns(4)

# Column 1: Current Price
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">💰 Current Price</div>
        <div class="metric-value">${:,.2f}</div>
    </div>
    """.format(current_price if current_price else 0), unsafe_allow_html=True)

# Column 2: Prediction
with col2:
    if prediction:
        change = prediction - current_price
        change_pct = (change / current_price) * 100 if current_price else 0
        color_class = "positive" if change > 0 else "negative"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📈 Tomorrow's Prediction</div>
            <div class="metric-value">${:,.2f}</div>
            <div style="margin-top: 0.5rem;" class="{}">
                {:+.2f} ({:+.2f}%)
            </div>
        </div>
        """.format(prediction, color_class, change, change_pct), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📈 Tomorrow's Prediction</div>
            <div class="metric-value" style="font-size: 1rem;">Not Available</div>
        </div>
        """, unsafe_allow_html=True)

# Column 3: Confidence Range
with col3:
    if prediction and current_price:
        error_margin = 5216.12  # Average error from training
        low = prediction - error_margin
        high = prediction + error_margin
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🎯 Confidence Range</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                ${:,.0f} - ${:,.0f}
            </div>
            <div style="margin-top: 0.5rem; color: #aaa;">
                ±${:,.2f} average error
            </div>
        </div>
        """.format(low, high, error_margin), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🎯 Confidence Range</div>
            <div class="metric-value" style="font-size: 1rem;">Not Available</div>
        </div>
        """, unsafe_allow_html=True)

# Column 4: Signal
with col4:
    if prediction and current_price:
        signal = "🟢 BUY" if prediction > current_price else "🔴 SELL" if prediction < current_price else "⚪ HOLD"
        confidence = abs((prediction - current_price) / current_price * 100) if current_price else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📊 Trading Signal</div>
            <div class="metric-value" style="font-size: 2.5rem;">{}</div>
            <div style="margin-top: 0.5rem; color: #aaa;">
                Confidence: {:.1f}%
            </div>
        </div>
        """.format(signal, confidence), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">📊 Trading Signal</div>
            <div class="metric-value" style="font-size: 1rem;">Not Available</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PRICE CHART
# ============================================
st.markdown("---")
st.subheader("📈 Price History & Technical Analysis")

# Create the chart
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("Price & Technical Indicators", "Volume", "RSI")
)

# Add candlestick chart (or line chart as fallback)
fig.add_trace(
    go.Scatter(
        x=data.index[-180:],  # Last 180 days for better visibility
        y=data['Close'][-180:],
        mode='lines',
        name='Close Price',
        line=dict(color='#f7931a', width=2)
    ),
    row=1, col=1
)

# Add moving averages if enabled
if show_technical_indicators:
    fig.add_trace(
        go.Scatter(
            x=features_df.index[-180:],
            y=features_df['MA_7'][-180:],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#00ff88', width=1, dash='dash')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=features_df.index[-180:],
            y=features_df['MA_30'][-180:],
            mode='lines',
            name='30-Day MA',
            line=dict(color='#ff6b6b', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'BB_Upper' in features_df.columns:
        fig.add_trace(
            go.Scatter(
                x=features_df.index[-180:],
                y=features_df['BB_Upper'][-180:],
                mode='lines',
                name='Upper BB',
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=features_df.index[-180:],
                y=features_df['BB_Lower'][-180:],
                mode='lines',
                name='Lower BB',
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255,255,255,0.05)',
                showlegend=False
            ),
            row=1, col=1
        )

# Add prediction point
if prediction and current_price:
    tomorrow_date = data.index[-1] + timedelta(days=1)
    fig.add_trace(
        go.Scatter(
            x=[tomorrow_date],
            y=[prediction],
            mode='markers+text',
            name='Prediction',
            marker=dict(
                color='#ffd700',
                size=20,
                symbol='star',
                line=dict(color='#f7931a', width=2)
            ),
            text=['📈'],
            textposition='top center'
        ),
        row=1, col=1
    )

# Volume chart
fig.add_trace(
    go.Bar(
        x=data.index[-180:],
        y=data['Volume'][-180:],
        name='Volume',
        marker_color='rgba(247, 147, 26, 0.6)',
        opacity=0.7
    ),
    row=2, col=1
)

# RSI chart
if 'RSI' in features_df.columns:
    fig.add_trace(
        go.Scatter(
            x=features_df.index[-180:],
            y=features_df['RSI'][-180:],
            mode='lines',
            name='RSI',
            line=dict(color='#9b59b6', width=2)
        ),
        row=3, col=1
    )
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    hovermode='x unified',
    template='plotly_dark',
    title_font_size=16,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update axes
fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=2, col=1)
fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=2, col=1)
fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=3, col=1)
fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# FEATURE IMPORTANCE (if model is loaded)
# ============================================
if show_feature_importance and model is not None:
    st.subheader("🔍 What Drives Bitcoin Price?")
    
    # Get feature importance from the model
    try:
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Create horizontal bar chart
        fig2 = go.Figure(data=[
            go.Bar(
                y=importance['Feature'],
                x=importance['Importance'],
                orientation='h',
                marker=dict(
                    color=importance['Importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{x:.1%}" for x in importance['Importance']],
                textposition='outside'
            )
        ])
        
        fig2.update_layout(
            title="Feature Importance - Random Forest Model",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Explanation
        st.info("""
        💡 **Understanding Feature Importance:**
        - **Lag_1**: Yesterday's price (most important - price tends to follow momentum)
        - **MA_7/MA_30**: Moving averages (trend indicators)
        - **Returns**: Daily price change (volatility measure)
        - **Volatility**: Price stability measure
        """)
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")

# ============================================
# STATISTICAL SUMMARY
# ============================================
st.subheader("📊 Market Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    price_high = data['High'].max()
    st.metric("📈 52-Week High", f"${price_high:,.2f}")

with col2:
    price_low = data['Low'].min()
    st.metric("📉 52-Week Low", f"${price_low:,.2f}")

with col3:
    avg_volume = data['Volume'].mean()
    st.metric("📊 Avg Daily Volume", f"{avg_volume:,.0f}")

with col4:
    volatility = data['Close'].pct_change().std() * 100
    st.metric("🌊 Volatility", f"{volatility:.2f}%")

# ============================================
# TRADING SIGNALS (Additional indicators)
# ============================================
if prediction and current_price:
    st.subheader("📊 Trading Insights")
    
    # Create 3 columns for different signals
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # RSI Signal
        if 'RSI' in features_df.columns:
            current_rsi = features_df['RSI'].iloc[-1]
            if current_rsi > 70:
                rsi_signal = "🔴 Overbought (RSI > 70) - Potential Sell"
            elif current_rsi < 30:
                rsi_signal = "🟢 Oversold (RSI < 30) - Potential Buy"
            else:
                rsi_signal = "⚪ Neutral (30 < RSI < 70)"
            st.metric("RSI Signal", rsi_signal, f"RSI: {current_rsi:.1f}")
    
    with c2:
        # Moving Average Signal
        if 'MA_7' in features_df.columns and 'MA_30' in features_df.columns:
            ma7 = features_df['MA_7'].iloc[-1]
            ma30 = features_df['MA_30'].iloc[-1]
            if ma7 > ma30:
                ma_signal = "🟢 Golden Cross - Bullish"
            else:
                ma_signal = "🔴 Death Cross - Bearish"
            st.metric("Moving Average Signal", ma_signal, f"MA7 vs MA30")
    
    with c3:
        # Model Prediction Signal
        if prediction > current_price:
            signal_text = "📈 Bullish - Price Expected to Rise"
            color = "green"
        else:
            signal_text = "📉 Bearish - Price Expected to Fall"
            color = "red"
        st.metric("Model Signal", signal_text, f"Target: ${prediction:,.2f}")

# ============================================
# DISCLAIMER
# ============================================
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This is an educational project and should NOT be used for real trading decisions.
    Cryptocurrency markets are highly volatile and unpredictable. Always do your own research before investing.
</div>
""", unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Data from Yahoo Finance | Model trained on historical Bitcoin data")

# ============================================
# EXPANDABLE: RAW DATA VIEW
# ============================================
with st.expander("📋 View Raw Data"):
    st.dataframe(
        data.tail(10),
        use_container_width=True,
        height=300
    )
    
    # Download button for data
    csv = data.tail(100).to_csv()
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"{symbol}_data.csv",
        mime="text/csv"
    )