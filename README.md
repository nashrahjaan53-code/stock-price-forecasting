# Google Stock Price Forecasting — ARIMA vs Prophet vs LSTM

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?style=for-the-badge&logo=tensorflow)
![Prophet](https://img.shields.io/badge/Prophet-Facebook-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit)

> An advanced Time Series Forecasting project that predicts Google (GOOGL) stock prices using 3 different approaches — ARIMA (statistical), Prophet (Facebook), and LSTM (Deep Learning) — with a full comparison dashboard built in Streamlit.

---

## What does this project do?

Forecasts the next **60 days** of GOOGL stock price using:
-  **ARIMA** — classical statistical model
-  **Prophet** — Facebook's forecasting library
-  **LSTM** — deep learning with sequential memory

Then compares all 3 using **MAE, RMSE, and MAPE**.

---

##  Results

| Model | MAE ($) | RMSE ($) | MAPE (%) | Winner |
|-------|---------|----------|----------|--------|
| ARIMA | $93.08 | $106.33 | 3.69% | |
| Prophet | $31.52 | $37.56 | **1.25%** | 🏆 Best |
| LSTM | $94.09 | $108.13 | 3.73% | |

**Prophet wins** with the lowest error across all metrics!

---

##  Pipeline

```
5 Years of GOOGL Stock Data (2019–2024)
           ↓
EDA (Moving Averages, Returns, Decomposition, Volume)
           ↓
Train/Test Split (Last 60 days = Test)
           ↓
┌─────────────┬──────────────┬──────────────┐
│   ARIMA     │   Prophet    │     LSTM     │
│  (5,1,0)   │ Changepoints │ 2 Layers +   │
│             │ + Seasonality│   Dropout    │
└─────────────┴──────────────┴──────────────┘
           ↓
Evaluate (MAE, RMSE, MAPE)
           ↓
Compare & Visualize
           ↓
Streamlit Dashboard
```

---

## Project Structure

```
stock-price-forecasting/
│
├── notebook/
│   └── stock_forecasting.py      # Full ML pipeline
│
├── model/
│   ├── GOOGL.csv                 # Stock data
│   ├── arima_model.pkl           # Saved ARIMA
│   ├── prophet_model.pkl         # Saved Prophet
│   ├── lstm_model.keras          # Saved LSTM
│   ├── scaler.pkl                # MinMaxScaler
│   ├── results.pkl               # All results
│   ├── eda.png
│   ├── decomposition.png
│   ├── distributions.png
│   ├── lstm_training.png
│   ├── forecast_comparison.png
│   ├── metrics_comparison.png
│   └── individual_forecasts.png
│
├── app.py                        # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

##  How to Run

```bash
# Step 1 - Clone
git clone https://github.com/YOUR_USERNAME/stock-price-forecasting.git
cd stock-price-forecasting

# Step 2 - Install
pip install -r requirements.txt

# Step 3 - Run pipeline
cd notebook
python stock_forecasting.py

# Step 4 - Launch app
cd ..
streamlit run app.py
```

🌐 Opens at `http://localhost:8501`

---

##  Key Concepts

### ARIMA (AutoRegressive Integrated Moving Average)
A classical statistical model that uses past values (AR), differencing for stationarity (I), and past forecast errors (MA) to predict future values.

### Prophet
Facebook's open-source tool designed for business time series. Automatically detects trend changepoints, handles seasonality (weekly/yearly), and is robust to missing data.

### LSTM (Long Short-Term Memory)
A type of recurrent neural network that can learn long-term dependencies in sequential data. Uses memory cells and gates to remember patterns over 60-day windows.

---

##  Key Visualizations
- Price history with MA-20, MA-50, MA-200
- Seasonal decomposition (trend, seasonality, residual)
- Daily returns distribution
- All 3 forecasts vs actual price
- Error metrics comparison (MAE, RMSE, MAPE)
- LSTM training loss curve

---

##  Tech Stack
- **Statsmodels** — ARIMA
- **Prophet** — Facebook forecasting
- **TensorFlow/Keras** — LSTM
- **Scikit-learn** — Preprocessing & metrics
- **Streamlit** — Interactive dashboard
- **Matplotlib & Seaborn** — Visualizations

---

##  Disclaimer
This project is for **educational purposes only**. It is not financial advice. Stock prices are inherently unpredictable and past performance does not guarantee future results.

---

Made by Nashrah k(https://github.com/nashrahjaan53-code)

⭐ Star this repo if you found it helpful!