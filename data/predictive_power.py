from typing import List, Optional
import numpy as np
import pandas as pd

def compute_future_return(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    horizons: List[int] = [5, 20],
    use_log: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    prices = pd.to_numeric(out[price_col], errors="coerce")

    for h in horizons:
        # Shift prices BACKWARDS to look into the future
        fwd_price = prices.shift(-h)
        if use_log:
            out[f"future_ret_{h}"] = np.log(fwd_price / prices)
        else:
            out[f"future_ret_{h}"] = (fwd_price / prices) - 1.0
    return out

def feature_predictive_power_by_regime(
    df: pd.DataFrame,
    regimes: pd.Series,
    feature_cols: Optional[List[str]] = None,
    future_return_cols: Optional[List[str]] = None,
    min_count: int = 50,
) -> pd.DataFrame:
    # Default columns if none provided
    if feature_cols is None:
        ret_col = "log_returns" if "log_returns" in df.columns else "log_return"
        feature_cols = [c for c in [ret_col, "drawdown"] if c in df.columns]
        vol_cols = [c for c in df.columns if c.startswith("rolling_vol")]
        if vol_cols: feature_cols.append(vol_cols[0])

    if future_return_cols is None:
        future_return_cols = [c for c in df.columns if c.startswith("future_ret_")]

    # Combine data
    combined = df.copy()
    combined["regime"] = regimes
    
    results = []
    for rid, group in combined.groupby("regime"):
        for feat in feature_cols:
            for target in future_return_cols:
                valid_data = group[[feat, target]].dropna()
                if len(valid_data) >= min_count:
                    correlation = valid_data[feat].corr(valid_data[target])
                    results.append({
                        "regime": rid,
                        "feature": feat,
                        "target": target,
                        "correlation": correlation,
                        "sample_size": len(valid_data)
                    })
    
    return pd.DataFrame(results)