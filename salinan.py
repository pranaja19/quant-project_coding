import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import datetime as dt
from statsmodels.regression.rolling import RollingOLS
import pandas_ta
import statsmodels.api as sm
import warnings
import requests, zipfile, os

warnings.filterwarnings("ignore")

# =====================
# Ambil data S&P 500
# =====================
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

tables = pd.read_html(response.text)
sandp500 = tables[0]
sandp500["Symbol"] = sandp500["Symbol"].str.replace(".", "-", regex=False)
symbol_list = sandp500["Symbol"].unique().tolist()

end_date = "2025-08-25"
start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 8)

df = yf.download(
    tickers=symbol_list,
    start=start_date,
    end=end_date
)

df = df.stack()
df.index.names = ["Date", "Ticker"]
df.columns = df.columns.str.lower()

# =====================
# Indikator teknikal
# =====================
df["garman_klass_vol"] = np.sqrt(
    0.5 * (np.log(df["high"] / df["low"])) ** 2
    - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
)

df["rsi"] = df.groupby(level=1)["close"].transform(lambda x: pandas_ta.rsi(x, length=20))
df["bb_low"] = df.groupby(level=1)["close"].transform(lambda x: pandas_ta.bbands(np.log1p(x), length=20).iloc[:, 0])
df["bb_mid"] = df.groupby(level=1)["close"].transform(lambda x: pandas_ta.bbands(np.log1p(x), length=20).iloc[:, 1])
df["bb_high"] = df.groupby(level=1)["close"].transform(lambda x: pandas_ta.bbands(np.log1p(x), length=20).iloc[:, 2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data["high"], low=stock_data["low"], close=stock_data["close"], length=14)
    return atr.sub(atr.mean()).div(atr.std())

df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())

df["macd"] = df.groupby(level=1, group_keys=False)["close"].apply(compute_macd)

df["dollar_vol"] = (df["close"] * df["volume"]) / 1e6
df["ema20"] = df.groupby(level=1)["close"].transform(lambda x: pandas_ta.ema(x, length=20))
df["ema50"] = df.groupby(level=1)["close"].transform(lambda x: pandas_ta.ema(x, length=50))

# =====================
# Monthly data & ranking
# =====================
last_cols = [c for c in df.columns.unique(0) if c not in ["dollar_vol", "garman_klass_vol", "volume", "open", "high", "low"]]

data = pd.concat([
    df.unstack("Ticker")["dollar_vol"].resample("M").mean().stack("Ticker").to_frame("dollar_vol"),
    df.unstack()[last_cols].resample("M").last().stack("Ticker"),
], axis=1).dropna()

data["dollar_vol"] = (data["dollar_vol"].unstack("Ticker").rolling(5 * 12).mean().stack())
data["dollar_vol_rank"] = data.groupby("Date")["dollar_vol"].rank(ascending=False)
data = data[data["dollar_vol_rank"] < 150].drop(["dollar_vol", "dollar_vol_rank"], axis=1)

# =====================
# Return Calculation
# =====================
def calculate_return(df):
    outliers_cutoff = 0.005
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        df[f"return_{lag}"] = (
            df["close"].pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(outliers_cutoff), upper=x.quantile(1 - outliers_cutoff)))
        ).add(1).pow(1 / lag).sub(1)
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_return).dropna()

# =====================
# Fama-French Factors
# =====================
# ==== Fama-French dari file lokal ====
csv_path = r"C:\Users\pranaja\Downloads\F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

fama = pd.read_csv(csv_path, skiprows=3)
fama = fama.dropna().rename(columns={fama.columns[0]: "Date"})

# ubah kolom Date ke datetime & jadikan index
fama["Date"] = pd.to_datetime(fama["Date"], errors="coerce")
fama = fama.set_index("Date").sort_index()

# ubah semua kolom (kecuali Date) ke persen desimal
fama = fama.apply(pd.to_numeric, errors="coerce") / 100

# filter mulai 2010
fama = fama[fama.index >= "2010-01-01"]
fama = fama.join(data['return_1']).sort_index()
print(fama)


