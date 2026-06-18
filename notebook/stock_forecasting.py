import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

os.makedirs("../model", exist_ok=True)
print(" All libraries imported!")

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n📦 Loading GOOGL stock data...")
df = pd.read_csv('../model/GOOGL.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

print(f"    {len(df)} trading days | {df.index[0].date()} → {df.index[-1].date()}")
print(f"   Price range: ${df['Close'].min():.2f} — ${df['Close'].max():.2f}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n Running EDA...")

# Full price history + moving averages
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('GOOGL Stock — Exploratory Data Analysis', fontsize=15, fontweight='bold')

# Price + MA
ax = axes[0]
ax.plot(df.index, df['Close'], color='steelblue', linewidth=1, label='Close Price')
ax.plot(df.index, df['Close'].rolling(20).mean(),  color='orange',  linewidth=1.5, label='MA-20')
ax.plot(df.index, df['Close'].rolling(50).mean(),  color='red',     linewidth=1.5, label='MA-50')
ax.plot(df.index, df['Close'].rolling(200).mean(), color='purple',  linewidth=1.5, label='MA-200')
ax.set_title('Closing Price with Moving Averages', fontweight='bold')
ax.set_ylabel('Price ($)'); ax.legend(); ax.grid(alpha=0.3)

# Daily returns
returns = df['Close'].pct_change().dropna()
axes[1].plot(returns.index, returns*100, color='green', linewidth=0.7, alpha=0.8)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].fill_between(returns.index, returns*100, 0,
                      where=returns>0, color='green', alpha=0.3)
axes[1].fill_between(returns.index, returns*100, 0,
                      where=returns<0, color='red', alpha=0.3)
axes[1].set_title('Daily Returns (%)', fontweight='bold')
axes[1].set_ylabel('Return (%)'); axes[1].grid(alpha=0.3)

# Volume
axes[2].bar(df.index, df['Volume']/1e6, color='steelblue', alpha=0.6, width=1)
axes[2].set_title('Trading Volume (Millions)', fontweight='bold')
axes[2].set_ylabel('Volume (M)'); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../model/eda.png', dpi=150)
plt.close()
print("    EDA plot saved.")

# Seasonal decomposition
decomp_series = df['Close'].resample('W').mean().dropna()
decomp = seasonal_decompose(decomp_series, model='additive', period=52)
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle('Seasonal Decomposition of GOOGL Stock', fontsize=14, fontweight='bold')
for ax, data, title in zip(axes,
    [decomp_series, decomp.trend, decomp.seasonal, decomp.resid],
    ['Original', 'Trend', 'Seasonality', 'Residual']):
    ax.plot(data, linewidth=1.2)
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../model/decomposition.png', dpi=150)
plt.close()
print("   Decomposition plot saved.")

# Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['Close'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Price Distribution', fontweight='bold')
axes[0].set_xlabel('Price ($)')
axes[1].hist(returns*100, bins=50, color='green', edgecolor='black', alpha=0.7)
axes[1].set_title('Returns Distribution', fontweight='bold')
axes[1].set_xlabel('Daily Return (%)')
plt.tight_layout()
plt.savefig('../model/distributions.png', dpi=150)
plt.close()
print("   Distribution plots saved.")

# ── 3. TRAIN / TEST SPLIT ────────────────────────────────────
FORECAST_DAYS = 60
train = df['Close'][:-FORECAST_DAYS]
test  = df['Close'][-FORECAST_DAYS:]
print(f"\n Train: {len(train)} days | Test (forecast): {FORECAST_DAYS} days")

results = {}

# ── 4. MODEL 1: ARIMA ─────────────────────────────────────────
print("\n Training ARIMA...")

# ADF Test
adf_result = adfuller(train)
print(f"   ADF p-value: {adf_result[1]:.4f} ({'Stationary' if adf_result[1]<0.05 else 'Non-stationary'})")

# Fit ARIMA
arima_model = ARIMA(train, order=(5, 1, 0))
arima_fit   = arima_model.fit()
arima_pred  = arima_fit.forecast(steps=FORECAST_DAYS)
arima_pred  = pd.Series(arima_pred.values, index=test.index)

mae_arima  = mean_absolute_error(test, arima_pred)
rmse_arima = np.sqrt(mean_squared_error(test, arima_pred))
mape_arima = np.mean(np.abs((test - arima_pred) / test)) * 100

results['ARIMA'] = {'pred': arima_pred, 'mae': mae_arima,
                     'rmse': rmse_arima, 'mape': mape_arima}
print(f"  ARIMA — MAE: {mae_arima:.2f} | RMSE: {rmse_arima:.2f} | MAPE: {mape_arima:.2f}%")
joblib.dump(arima_fit, '../model/arima_model.pkl')

# ── 5. MODEL 2: PROPHET ───────────────────────────────────────
print("\n Training Prophet...")
prophet_df = train.reset_index()
prophet_df.columns = ['ds', 'y']

prophet_model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True
)
prophet_model.fit(prophet_df)

future         = prophet_model.make_future_dataframe(periods=FORECAST_DAYS, freq='B')
forecast       = prophet_model.predict(future)
prophet_pred   = forecast[['ds', 'yhat']].tail(FORECAST_DAYS)
prophet_pred   = pd.Series(prophet_pred['yhat'].values, index=test.index)

mae_prophet  = mean_absolute_error(test, prophet_pred)
rmse_prophet = np.sqrt(mean_squared_error(test, prophet_pred))
mape_prophet = np.mean(np.abs((test - prophet_pred) / test)) * 100

results['Prophet'] = {'pred': prophet_pred, 'mae': mae_prophet,
                       'rmse': rmse_prophet, 'mape': mape_prophet}
print(f"  Prophet — MAE: {mae_prophet:.2f} | RMSE: {rmse_prophet:.2f} | MAPE: {mape_prophet:.2f}%")
joblib.dump(prophet_model, '../model/prophet_model.pkl')

# ── 6. MODEL 3: LSTM ──────────────────────────────────────────
print("\n Training LSTM...")

scaler      = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1,1))

SEQ_LEN = 60
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build LSTM
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(patience=5, restore_best_weights=True)

history = lstm_model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    validation_split=0.1,
    callbacks=[es], verbose=0
)
print(f"   Trained for {len(history.history['loss'])} epochs")

# LSTM Forecast
all_data   = scaler.transform(df['Close'].values.reshape(-1,1))
last_seq   = all_data[-(FORECAST_DAYS + SEQ_LEN):-FORECAST_DAYS]
lstm_preds = []
current    = last_seq.copy()

for _ in range(FORECAST_DAYS):
    inp  = current[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    pred = lstm_model.predict(inp, verbose=0)[0,0]
    lstm_preds.append(pred)
    current = np.append(current, [[pred]], axis=0)

lstm_pred = scaler.inverse_transform(np.array(lstm_preds).reshape(-1,1)).flatten()
lstm_pred = pd.Series(lstm_pred, index=test.index)

mae_lstm  = mean_absolute_error(test, lstm_pred)
rmse_lstm = np.sqrt(mean_squared_error(test, lstm_pred))
mape_lstm = np.mean(np.abs((test - lstm_pred) / test)) * 100

results['LSTM'] = {'pred': lstm_pred, 'mae': mae_lstm,
                    'rmse': rmse_lstm, 'mape': mape_lstm}
print(f"   ✅ LSTM — MAE: {mae_lstm:.2f} | RMSE: {rmse_lstm:.2f} | MAPE: {mape_lstm:.2f}%")
lstm_model.save('../model/lstm_model.keras')
joblib.dump(scaler, '../model/scaler.pkl')

# LSTM training loss plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'], label='Train Loss', color='steelblue')
ax.plot(history.history['val_loss'], label='Val Loss', color='orange')
ax.set_title('LSTM Training Loss', fontsize=13, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../model/lstm_training.png', dpi=150)
plt.close()

# ── 7. COMPARISON PLOTS ───────────────────────────────────────
print("\n📈 Generating comparison plots...")
colors = {'ARIMA': '#e74c3c', 'Prophet': '#2ecc71', 'LSTM': '#9b59b6'}

# All 3 forecasts vs actual
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train[-120:].index, train[-120:], color='steelblue',
        linewidth=1.5, label='Historical Price')
ax.plot(test.index, test, color='black', linewidth=2, label='Actual Price', zorder=5)
for name, res in results.items():
    ax.plot(test.index, res['pred'], linewidth=2,
            color=colors[name], label=f'{name} Forecast', linestyle='--')
ax.axvline(test.index[0], color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.set_title('GOOGL Stock — ARIMA vs Prophet vs LSTM Forecast', fontsize=14, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Price ($)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../model/forecast_comparison.png', dpi=150)
plt.close()
print("  Forecast comparison saved.")

# Metrics bar charts
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Model Comparison — Error Metrics', fontsize=14, fontweight='bold')
model_names = list(results.keys())
model_colors = [colors[n] for n in model_names]

for ax, metric, label in zip(axes,
    ['mae', 'rmse', 'mape'],
    ['MAE ($)', 'RMSE ($)', 'MAPE (%)']):
    vals = [results[n][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=model_colors, edgecolor='black')
    ax.set_title(label, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../model/metrics_comparison.png', dpi=150)
plt.close()
print("  Metrics comparison saved.")

# Individual forecast plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Individual Forecasts vs Actual', fontsize=14, fontweight='bold')
for ax, (name, res) in zip(axes, results.items()):
    ax.plot(test.index, test.values, color='black', linewidth=2, label='Actual')
    ax.plot(test.index, res['pred'].values, color=colors[name],
            linewidth=2, linestyle='--', label=f'{name}')
    ax.set_title(f'{name}\nMAE:{res["mae"]:.1f} | RMSE:{res["rmse"]:.1f} | MAPE:{res["mape"]:.1f}%',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('../model/individual_forecasts.png', dpi=150)
plt.close()
print("  Individual forecasts saved.")

# ── 8. SUMMARY ───────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL RESULTS SUMMARY")
print("="*55)
print(f"{'Model':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
print("-"*45)
for name, res in results.items():
    print(f"{name:<10} ${res['mae']:>8.2f} ${res['rmse']:>8.2f} {res['mape']:>8.2f}%")

best = min(results, key=lambda x: results[x]['mape'])
print(f"\n Best Model by MAPE: {best}")
print("="*55)

joblib.dump(results, '../model/results.pkl')
print("\n All done! Models trained and plots saved.")