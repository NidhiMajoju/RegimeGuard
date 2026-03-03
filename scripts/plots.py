# plotting utilities for RegimeGuard data
import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Helper functions -----------------------------------------------------------

def load_processed(ticker: str, processed_dir: str = "data/processed") -> pd.DataFrame:
    """Load a processed CSV for a given ticker into a DataFrame.

    The returned DataFrame is indexed by ``Date``.
    """

    path = os.path.join(processed_dir, f"{ticker}_processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file for {ticker} not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def ensure_report_dir(report_dir: str = "data/reports") -> str:
    # If a file exists at the path, remove it and replace with a directory.
    if os.path.exists(report_dir) and not os.path.isdir(report_dir):
        os.remove(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


# Plotting functions --------------------------------------------------------

def plot_price(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot adjusted closing price as a time series."""

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(df.index, df["Adj Close"], label="Adj Close")
    ax.set_title(f"{ticker} Price" if ticker else "Price")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig, ax


def plot_volume(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot daily volume as a bar chart."""

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.bar(df.index, df["Volume"], width=1.0, color="gray")
    ax.set_title(f"{ticker} Volume" if ticker else "Volume")
    ax.set_ylabel("Volume")
    ax.grid(True)
    return fig, ax


def plot_returns(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax_ts: Optional[plt.Axes] = None,
    ax_hist: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Show log returns time series and a histogram/density plot."""

    returns = df["log_returns"].dropna()

    # time series
    if ax_ts is None and ax_hist is None:
        fig, (ax_ts, ax_hist) = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    elif ax_ts is None:
        fig = ax_hist.figure
        ax_ts = fig.add_subplot(211, sharex=ax_hist)
    elif ax_hist is None:
        fig = ax_ts.figure
        ax_hist = fig.add_subplot(212, sharex=ax_ts)
    else:
        fig = ax_ts.figure

    ax_ts.plot(returns.index, returns, label="log returns", color="tab:blue")
    ax_ts.set_title(f"{ticker} Log Returns" if ticker else "Log Returns")
    ax_ts.set_ylabel("Returns")
    ax_ts.grid(True)

    sns.histplot(returns, ax=ax_hist, kde=True, color="tab:blue", stat="density")
    ax_hist.set_title("Distribution of Returns")
    ax_hist.set_xlabel("Log return")
    ax_hist.grid(True)

    return fig, (ax_ts, ax_hist)


def plot_drawdown(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    highlight_max: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot drawdown series (negative or zero).

    If ``highlight_max`` is True the date/level of the worst (most negative)
    drawdown is annotated on the chart.
    """

    dd = df["drawdown"].dropna()
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(dd.index, dd, label="drawdown", color="tab:red")
    if highlight_max and not dd.empty:
        idx = dd.idxmin()
        val = dd.loc[idx]
        ax.scatter([idx], [val], color="black", zorder=5)
        ax.annotate(f"min {val:.2%}", xy=(idx, val), xytext=(5, -15),
                    textcoords="offset points", color="black")

    ax.set_title(f"{ticker} Drawdown" if ticker else "Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    return fig, ax


def plot_rolling_volatility(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot rolling volatility computed on log returns."""

    vol = df["rolling_vol"].dropna()
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(vol.index, vol, label="rolling vol", color="tab:green")
    ax.set_title(f"{ticker} Rolling Volatility" if ticker else "Rolling Volatility")
    ax.set_ylabel("Std dev")
    ax.grid(True)
    return fig, ax


def plot_cumulative_returns(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot cumulative log returns (exp(cumsum) - 1)."""

    cum = (df["log_returns"].cumsum().apply(np.exp) - 1).dropna()
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(cum.index, cum, label="cumulative return")
    ax.set_title(f"{ticker} Cumulative Returns" if ticker else "Cumulative Returns")
    ax.set_ylabel("Cumulative return")
    ax.grid(True)
    return fig, ax




def plot_moving_averages(
    df: pd.DataFrame,
    windows: List[int] = [50, 200],
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay simple moving averages on price."""

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(df.index, df["Adj Close"], label="Adj Close", color="black")
    for w in windows:
        ma = df["Adj Close"].rolling(window=w).mean()
        ax.plot(df.index, ma, label=f"MA{w}")

    ax.set_title(f"{ticker} Price and MAs" if ticker else "Price and Moving Averages")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    return fig, ax


def plot_log_price(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot log of adjusted closing price."""

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(df.index, np.log(df["Adj Close"]), label="log price")
    ax.set_title(f"{ticker} Log Price" if ticker else "Log Price")
    ax.set_ylabel("log(price)")
    ax.grid(True)
    return fig, ax


def compute_rolling_sharpe(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = False,
) -> pd.Series:
    """Rolling Sharpe ratio: mean / std of log returns over a window.

    If ``annualize`` we scale by sqrt(252).
    """

    r = df["log_returns"].dropna()
    roll_mean = r.rolling(window=window).mean()
    roll_std = r.rolling(window=window).std()
    sr = roll_mean / roll_std
    if annualize:
        sr = sr * np.sqrt(252)
    return sr


def plot_rolling_sharpe(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = False,
    ticker: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot rolling Sharpe ratio."""

    sr = compute_rolling_sharpe(df, window=window, annualize=annualize)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(sr.index, sr, label="rolling sharpe", color="tab:purple")
    ax.set_title(f"{ticker} Rolling Sharpe" if ticker else "Rolling Sharpe")
    ax.set_ylabel("Sharpe")
    ax.grid(True)
    return fig, ax


def summary_statistics(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """Return a table of basic statistics for price and returns."""

    returns = df["log_returns"].dropna()
    stats = {
        "price_mean": df[price_col].mean(),
        "price_std": df[price_col].std(),
        "return_mean": returns.mean(),
        "return_std": returns.std(),
        "return_skew": returns.skew(),
        "return_kurt": returns.kurtosis(),
        "return_min": returns.min(),
        "return_max": returns.max(),
    }
    return pd.Series(stats).to_frame("value")


def plot_correlation_matrix(
    tickers: List[str],
    processed_dir: str = "data/processed",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Calculate and plot correlation of log returns between tickers."""

    dfs = {}
    for t in tickers:
        dfs[t] = load_processed(t, processed_dir)
    returns = pd.concat({t: dfs[t]["log_returns"] for t in tickers}, axis=1)
    returns.columns = tickers
    corr = returns.corr()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix of log returns")
    return fig, ax


# Convenience script --------------------------------------------------------

def make_all_plots(
    tickers: List[str],
    processed_dir: str = "data/processed",
    report_dir: str = "data/reports",
    save: bool = True,
):
    """Generate a standard set of charts for each ticker and write to ``report_dir``.

    The following plots are created:

    * Price
    * Volume
    * Log returns (series + histogram)
    * Drawdown
    * Rolling volatility
    * Cumulative returns
    """

    ensure_report_dir(report_dir)

    for t in tickers:
        df = load_processed(t, processed_dir)

        # compute & store summary stats
        stats = summary_statistics(df)
        if save:
            stats.to_csv(os.path.join(report_dir, f"{t}_stats.csv"))

        for func, kwargs in [
            (plot_price, {}),
            (plot_volume, {}),
            (plot_returns, {}),
            # highlight worst drawdown point on the drawdown chart
            (plot_drawdown, {"highlight_max": True}),
            (plot_rolling_volatility, {}),
            (plot_cumulative_returns, {}),
            (plot_moving_averages, {}),
            (plot_log_price, {}),
            (plot_rolling_sharpe, {}),
        ]:
            fig_or_fig_ax = func(df, ticker=t, **kwargs)
            # ``plot_returns`` returns (fig, (ax_ts, ax_hist))
            if isinstance(fig_or_fig_ax, tuple) and isinstance(fig_or_fig_ax[1], tuple):
                fig = fig_or_fig_ax[0]
            else:
                fig = fig_or_fig_ax[0]

            if save and fig is not None:
                fname = f"{t}_{func.__name__}.png"
                fig.savefig(os.path.join(report_dir, fname), dpi=150)
                plt.close(fig)

    # one extra chart for correlations across all tickers
    if save:
        fig, ax = plot_correlation_matrix(tickers, processed_dir)
        fig.savefig(os.path.join(report_dir, "correlation_matrix.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots and summary statistics from processed RegimeGuard data.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers to plot (default: all files in data/processed)",
    )
    parser.add_argument(
        "--report-dir",
        default="data/reports",
        help="Directory where plots and stats are saved",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers_list = args.tickers
    else:
        # inspect processed folder
        files = os.listdir("data/processed")
        tickers_list = [f.split("_")[0] for f in files if f.endswith("_processed.csv")]

    make_all_plots(tickers_list, report_dir=args.report_dir)
