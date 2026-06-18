import numpy as np
import pandas as pd
import os

os.makedirs('../model', exist_ok=True)

np.random.seed(42)
dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='B')
n = len(dates)

t = np.arange(n)
trend = 1000 + t * 0.85
seasonality = 60 * np.sin(2 * np.pi * t / 252) + 30 * np.sin(2 * np.pi * t / 63)
noise = np.cumsum(np.random.normal(0, 8, n))
price = trend + seasonality + noise
price = np.maximum(price, 100)

df = pd.DataFrame({
    'Date':   dates,
    'Open':   price * (1 - np.abs(np.random.normal(0, 0.005, n))),
    'High':   price * (1 + np.abs(np.random.normal(0, 0.012, n))),
    'Low':    price * (1 - np.abs(np.random.normal(0, 0.012, n))),
    'Close':  price,
    'Volume': np.random.randint(20000000, 60000000, n)
})

df.to_csv('../model/GOOGL.csv', index=False)
print(f'✅ GOOGL.csv generated! {len(df)} trading days')
print(f'   Price range: ${df.Close.min():.2f} — ${df.Close.max():.2f}')