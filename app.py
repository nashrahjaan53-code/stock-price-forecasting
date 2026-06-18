import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="📈 GOOGL Stock Forecaster", page_icon="📈", layout="wide")

@st.cache_resource
def load_all():
    arima   = joblib.load("model/arima_model.pkl")
    prophet = joblib.load("model/prophet_model.pkl")
    results = joblib.load("model/results.pkl")
    df      = pd.read_csv("model/GOOGL.csv", parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return arima, prophet, results, df

arima_model, prophet_model, results, df = load_all()

COLORS = {'ARIMA': '#e74c3c', 'Prophet': '#2ecc71', 'LSTM': '#9b59b6'}
FORECAST_DAYS = 60
train = df['Close'][:-FORECAST_DAYS]
test  = df['Close'][-FORECAST_DAYS:]

# ── Header ────────────────────────────────────────────────────
st.title("Google Stock Price Forecasting")
st.markdown("**ARIMA vs Prophet vs LSTM | 5 Years of GOOGL Data**")
st.markdown("---")

# ── Metrics at top ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}",
            f"{((df['Close'].iloc[-1]/df['Close'].iloc[-2])-1)*100:.2f}%")
col2.metric(" Best Model",  "Prophet", "MAPE: 1.25%")
col3.metric("ARIMA MAPE",    f"{results['ARIMA']['mape']:.2f}%")
col4.metric("LSTM MAPE",     f"{results['LSTM']['mape']:.2f}%")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "EDA", " Model Comparison", " Model Info"])

# ── Tab 1: Forecast ───────────────────────────────────────────
with tab1:
    st.subheader(" Forecasts vs Actual Price")

    models_to_show = st.multiselect(
        "Select models to display:",
        ["ARIMA", "Prophet", "LSTM"],
        default=["ARIMA", "Prophet", "LSTM"]
    )
    history_days = st.slider("Days of historical data to show:", 30, 300, 120)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train[-history_days:].index, train[-history_days:],
            color='steelblue', linewidth=1.5, label='Historical Price')
    ax.plot(test.index, test, color='black', linewidth=2.5,
            label='Actual Price', zorder=5)

    for name in models_to_show:
        ax.plot(test.index, results[name]['pred'],
                color=COLORS[name], linewidth=2,
                linestyle='--', label=f'{name} Forecast')

    ax.axvline(test.index[0], color='gray', linestyle=':', linewidth=1.5)
    ax.set_title('GOOGL Stock Price Forecast', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Forecast Values (Last 10 days)")
    forecast_df = pd.DataFrame({'Actual': test.values}, index=test.index)
    for name in models_to_show:
        forecast_df[name] = results[name]['pred'].values
    st.dataframe(forecast_df.tail(10).style.format("${:.2f}"), use_container_width=True)

# ── Tab 2: EDA ────────────────────────────────────────────────
with tab2:
    st.subheader(" Exploratory Data Analysis")
    plots = {
        "Price History & Moving Averages": "model/eda.png",
        "Seasonal Decomposition":          "model/decomposition.png",
        "Price & Returns Distribution":    "model/distributions.png",
        "LSTM Training Loss":              "model/lstm_training.png",
    }
    for title, path in plots.items():
        st.markdown(f"#### {title}")
        try:
            st.image(path, use_container_width=True)
        except:
            st.warning(f"Run pipeline first: {path}")

# ── Tab 3: Comparison ─────────────────────────────────────────
with tab3:
    st.subheader(" Model Performance Comparison")
    st.image("model/forecast_comparison.png", use_container_width=True)
    st.image("model/metrics_comparison.png",  use_container_width=True)
    st.image("model/individual_forecasts.png", use_container_width=True)

    st.markdown("###  Metrics Summary")
    summary = pd.DataFrame({
        'Model'   : ['ARIMA', 'Prophet', 'LSTM'],
        'MAE ($)' : [f"${results[m]['mae']:.2f}"  for m in ['ARIMA','Prophet','LSTM']],
        'RMSE ($)': [f"${results[m]['rmse']:.2f}" for m in ['ARIMA','Prophet','LSTM']],
        'MAPE (%)'  : [f"{results[m]['mape']:.2f}%" for m in ['ARIMA','Prophet','LSTM']],
        'Winner'  : ['', ' Best', '']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Tab 4: Model Info ─────────────────────────────────────────
with tab4:
    st.subheader("About the Models")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ###  ARIMA
        **AutoRegressive Integrated Moving Average**
        - Classical statistical model
        - Uses past values to predict future
        - Order: (5, 1, 0)
        - Best for: linear, stationary data
        - MAPE: 3.69%
        """)
    with col2:
        st.markdown("""
        ###  Prophet
        **Facebook's Forecasting Tool**
        - Handles seasonality automatically
        - Robust to missing data
        - Detects trend changepoints
        - Best for: business time series
        - MAPE: 1.25% 
        """)
    with col3:
        st.markdown("""
        ###  LSTM
        **Long Short-Term Memory (Deep Learning)**
        - Learns complex patterns
        - 2 LSTM layers + Dropout
        - Sequence length: 60 days
        - Best for: complex non-linear patterns
        - MAPE: 3.73%
        """)

    st.markdown("---")
    st.markdown("""
    ###  Which model to use in real life?
    - **Short term (1–7 days):** LSTM
    - **Medium term (1–3 months):** Prophet
    - **Quick baseline:** ARIMA
    - **Production:** Ensemble of all 3!
    """)

st.markdown("---")
st.markdown(" *This is for educational purposes only. Not financial advice.*")
st.markdown("Built with  using **ARIMA + Prophet + LSTM + Streamlit**")