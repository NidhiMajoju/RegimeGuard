import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class RegimeDetector:
    """
    Gaussian HMM for market regime detection.
    Features: log_return, rolling_vol_*, drawdown
    """
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def _build_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        # Prefer 'log_returns' (from preprocess), fallback to 'log_return'
        ret_col = "log_returns" if "log_returns" in df.columns else "log_return"
        vol_cols = [c for c in df.columns if c.startswith("rolling_vol")]
        vol_col = vol_cols[0] if vol_cols else "rolling_vol"
        dd_col = "drawdown"

        feature_cols = [ret_col, vol_col, dd_col]
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain feature column: {col}")

        valid = df[feature_cols].dropna()
        valid_idx = valid.index
        X = valid.values.astype(np.float64)
        return X, valid_idx

    def fit_and_predict(self, df: pd.DataFrame) -> pd.Series:
        X, valid_idx = self._build_feature_matrix(df)
        self.model.fit(X)
        labels = self.model.predict(X)
        return pd.Series(labels, index=valid_idx, name="regime")

    @staticmethod
    def compute_regime_stats(df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        ret_col = "log_returns" if "log_returns" in df.columns else "log_return"
        combined = pd.DataFrame({"return": df[ret_col], "regime": regimes}).dropna()
        
        if combined.empty:
            return pd.DataFrame()

        ann = np.sqrt(252)
        stats = (
            combined.groupby("regime")["return"]
            .agg(mean_return="mean", vol="std", count="size")
            .reset_index()
        )
        stats["sharpe_ann"] = np.where(
            stats["vol"].gt(0),
            ann * stats["mean_return"] / stats["vol"],
            np.nan,
        )
        return stats

    def interpret_regimes(self, stats: pd.DataFrame) -> Dict[int, str]:
        """
        Logic:
        - High Mean Return + Low Vol = Bull
        - Low/Negative Mean Return + High Vol = Bear
        - Middle = Sideways
        """
        if stats.empty:
            return {}
        
        # Sort by mean return to identify regimes
        # Highest mean = Bull, Lowest mean = Bear, Middle = Sideways
        sorted_stats = stats.sort_values("mean_return", ascending=False).reset_index()
        
        mapping = {
            sorted_stats.iloc[0]["regime"]: "Bull",
            sorted_stats.iloc[1]["regime"]: "Sideways",
            sorted_stats.iloc[2]["regime"]: "Bear"
        }
        return mapping

def run_regime_detection(
    input_path: str,
    labels_output_path: str = "data/processed/regime_labels.csv",
    stats_output_path: str = "data/reports/regime_characteristics.csv",
    n_regimes: int = 3,
) -> Tuple[pd.Series, pd.DataFrame]:
    
    df = pd.read_csv(input_path)
    # Date handling
    for cand in ["Date", "date", "datetime"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand).sort_index()
            break

    detector = RegimeDetector(n_regimes=n_regimes)
    regimes = detector.fit_and_predict(df)
    stats = detector.compute_regime_stats(df, regimes)

    # Use the logic to label them
    mapping = detector.interpret_regimes(stats)
    regime_names = regimes.map(mapping)

    # Save Results
    os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)

    output_df = pd.DataFrame({"regime_id": regimes, "regime_name": regime_names})
    output_df.to_csv(labels_output_path)

    stats["regime_label"] = stats["regime"].map(mapping)
    stats.to_csv(stats_output_path, index=False)

    return regimes, stats