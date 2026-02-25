# src/data/preprocess.py
import os
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler



def download_data(ticker: str) -> str:
    """
    Downloads data
    """
    
    # Date range for data
    START = "2015-01-01"
    END = "2025-12-31"
    RAW_DATA_DIRC = "data/raw"

    raw_data = yf.download(ticker, start=START, end=END, auto_adjust=False)

    # create directory if does not exist
    os.makedirs(RAW_DATA_DIRC, exist_ok=True)
    
    output_path = os.path.join(RAW_DATA_DIRC, f"{ticker}_raw.csv")
    raw_data.to_csv(output_path)

    return output_path


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaNs and ensure dtypes are numeric where needed.
    """
    cur_df = df.copy()
    # remove first two rows 
    cur_df = cur_df.iloc[2:]
    
    # forward fill NaN values, if first value is NaN back fill
    cur_df.ffill(inplace=True)
    cur_df.bfill(inplace=True)


    # Rename 1st col to Date and make values datetime  
    cur_df.rename(columns={"Price": "Date"}, inplace=True)
    cur_df["Date"] = pd.to_datetime(cur_df["Date"])

    # make adj close, close, open, high, low, volume numeric
    for col in cur_df.columns:
        if col == "Date":
            continue
        if cur_df[col].dtype not in ['float64', 'int64']:
            cur_df[col] = pd.to_numeric(cur_df[col], errors='coerce')
    
    return cur_df


def compute_log_returns( df: pd.DataFrame, price_col: str = "Adj Close" ) -> pd.Series:
    """
    Compute daily log returns: log(P_t / P_{t-1}).
    """
    log_return_series = np.log(df[price_col] / df[price_col].shift(1))
    log_return_series = log_return_series.dropna()

    return log_return_series


def compute_rolling_vol(returns: pd.Series, window: int = 20,) -> pd.Series:
    """
    Rolling volatility (standard deviation of log returns).
    """
    rolling_volatility = returns.rolling(window=window).std()
    rolling_volatility = rolling_volatility[1:]

    return rolling_volatility
   
def compute_drawdown( df: pd.DataFrame, price_col: str = "Adj Close",) -> pd.Series:
    """
    Drawdown: (price / rolling_max - 1).
    """

    rolling_max = df[price_col].cummax()
    drawdown = (df[price_col] / rolling_max) - 1

    return drawdown

def normalize_features(df: pd.DataFrame, feature_cols, fit=True, scaler=None) -> pd.DataFrame:
    """
    Normalize selected features using StandardScaler.
    
    If fit=True, fit the scaler on the data and return 
    the scaled DataFrame and the fitted scaler.
    
    If fit=False, use the provided scaler to transform 
    the data and return the scaled DataFrame.
    """
    # fit scaler on train, transform on val/test
    if fit:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df_scaled, scaler
    else:
        if scaler is None:
            raise ValueError("When fit=False, a scaler must be provided.")
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        return df_scaled

    

def create_sequences(df: pd.DataFrame, seq_length: int =60, forecast_horizon: int = 1, target_col: str = 'Adj Close') -> Tuple:
    """
    Create sequences for time series forecasting.    
    """    
    # Extract target and feature values
    target = df[target_col].values
    features = df.drop(columns=[target_col]).values

    X, y = [], []
    for i in range(len(df) - seq_length - forecast_horizon + 1):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length+forecast_horizon-1])

    return np.array(X), np.array(y)


def preprocess_single_ticker(ticker: str, output_dir: str = "data/processed", vol_window: int = 20, ) -> Tuple[str, pd.DataFrame]:
    """
    Load raw <ticker> CSV, clean it, compute features, and save processed CSV.
    """                     

    path = download_data(ticker)
    df = pd.read_csv(path)
    df = basic_clean(df)


    df['log_returns'] = compute_log_returns(df)
    df['rolling_vol'] = compute_rolling_vol(df["log_returns"], window=vol_window)
    df["drawdown"] = compute_drawdown(df)
    
    # price returns 
    df['returns'] = df['Adj Close'].pct_change()
    
    # momentum indicators
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['Adj Close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close= df['Adj Close']).macd_diff()
    df['stoch_k'] = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Adj Close'], window=14
    ).stoch()
    
    # volatility indicators
    df['atr_14'] = ta.volatility.AverageTrueRange(
         high=df['High'], low=df['Low'], close=df['Adj Close'], window=14
    ).average_true_range()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Adj Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    # Realized volatility (20-day rolling std of returns)
    df['realized_vol'] = df['returns'].rolling(20).std()
    
    # Simple moving averages
    df['sma_20'] = df['Adj Close'].rolling(20).mean()
    df['sma_50'] = df['Adj Close'].rolling(50).mean()
    
    # Exponential moving average
    df['ema_12'] = df['Adj Close'].ewm(span=12, adjust=False).mean()
    
    # Price momentum (5-day return)
    df['price_momentum'] = df['Adj Close'].pct_change(periods=5)
    
    # volatility ratio
    df['vol_ratio'] = df['realized_vol'] / df['realized_vol'].rolling(30).mean()

    # Close to SMA ratio
    df['close_to_sma'] = (df['Adj Close'] - df['sma_20']) / df['sma_20']
    
    
    # drop NaN values produced by rolling volatility 
    df = df.dropna()

    # save as csv
    os.makedirs("data/processed", exist_ok=True)
    out_path = os.path.join(output_dir, f"{ticker}_processed.csv")

    df.to_csv(out_path)

    return out_path, df


tickers = [
    "SPY",     # Market baseline
    "QQQ",     # Tech
    "TLT",     # Bonds 
    "KO",      # Low volatility
    "JPM",     # Financials
    "GLD",     # Commodities
    "REGN",    # United Rentals
    "URI",     # Commodities
    "ULTA",    # Ulta
    "AVGO",    # Broadcom
    "ANET",    # Arista Networks
    "AXON",    # Axon
    "XOM",     # Exon
    "CVX",     # Cheveron
    "SHEL",    # Shell
    "ADBE",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "AXP",
    "CAT",
    "BA",
    "HON",
    "MMM",
    "DD",
    "JNJ",
    "PFE",
    "MRK",
    "UNH",
    "WMT",
    "HD",
    "NKE",
    "MCD",
    "SBUX",
    "COP",
    "V",
    "MA",
    "PG",
    "PEP",
    "CSCO",
    "SCHW",
    "BLK",
    "MKTX",
    "ABBV",
    "AMGN",
    "LLY",
    "ISRG",
    "BIIB",
    "GILD",
    "MNST",
    "DE",
    "ETN",
    "LIN",
    "U",
    "INTU",
    "TTD",
    "COST",
    "LMT",
    "XLE",
    "XLRE",
    "XLU",
    "VO",
    "IJR",
    "VWO",
    "VEA",
    "VXX",
    "UVXY",
    "TIP",
    "DBC",
    "LQD",
    "HYG",
    "JNK"
]


for ticker in tickers:
    preprocess_single_ticker(ticker)

