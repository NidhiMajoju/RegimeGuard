# src/data/preprocess.py
import os
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf



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
    
def preprocess_single_ticker(ticker: str, output_dir: str = "data/processed", vol_window: int = 20, ) -> Tuple[str, pd.DataFrame]:
    """
    Load raw <ticker> CSV, clean it, compute features, and save processed CSV.
    """                     

    path = download_data(ticker)
    df = pd.read_csv(path)
    df = basic_clean(df)
    df["log_returns"] = compute_log_returns(df)
    df["rolling_vol"] = compute_rolling_vol(df["log_returns"], window=vol_window)
    df["drawdown"] = compute_drawdown(df)
    
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
    "AAPL",    # Mega cap tech
    "NVDA",    # High growth/volatility
    "KO",      # Low volatility
    "JPM",     # Financials
    "GLD",     # Commodities
    "ALGN",    # Align Tech
    "REGN",    # United Rentals
    "URI",     # Commodities
    "AMZN",    # Amazaon
    "ULTA",    # Ulta
    "AVGO",    # Broadcom
    "ANET",    # Arista Networks
    "AXON",    # Axon
    "XOM",     # Exon
    "CVX",     # Cheveron
    "SHEL",    # Shell

]


for ticker in tickers:
    preprocess_single_ticker(ticker)

