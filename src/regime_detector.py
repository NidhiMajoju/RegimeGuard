import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes

        self.feature_cols = [
            "log_returns",
            "rolling_vol",
            "drawdown",
            "realized_vol",
            "price_momentum",
            "vol_ratio",
            "close_to_sma",
        ]

        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def _build_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:

        if "log_returns" in df.columns:
            ret_col = "log_returns"
        elif "log_return" in df.columns:
            ret_col = "log_return"
        else:
            raise ValueError("Missing return column")

        vol_cols = [c for c in df.columns if c.startswith("rolling_vol")]
        vol_col = vol_cols[0] if vol_cols else "rolling_vol"

        base_features = [ret_col, vol_col, "drawdown"]

        extended = [
            col for col in self.feature_cols
            if col in df.columns and col not in base_features
        ]

        final_features = base_features + extended

        valid = df[final_features].dropna()
        X = valid.values.astype(np.float64)

        return X, valid.index

    def fit_and_predict(self, df: pd.DataFrame) -> pd.Series:

        X, valid_idx = self._build_feature_matrix(df)

        self.model.fit(X)
        labels = self.model.predict(X)

        return pd.Series(labels, index=valid_idx, name="regime")

    @staticmethod
    def compute_regime_stats(df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:

        if "log_returns" in df.columns:
            ret_col = "log_returns"
        elif "log_return" in df.columns:
            ret_col = "log_return"
        else:
            raise ValueError("Missing return column")

        combined = pd.DataFrame({
            "return": df[ret_col],
            "regime": regimes
        }).dropna()

        if combined.empty:
            return pd.DataFrame()

        ann = np.sqrt(252)

        stats = (
            combined.groupby("regime")["return"]
            .agg(mean_return="mean", vol="std", count="size")
            .reset_index()
        )

        stats["sharpe_ann"] = np.where(
            stats["vol"] > 0,
            ann * stats["mean_return"] / stats["vol"],
            np.nan,
        )

        return stats

    def interpret_regimes(self, stats: pd.DataFrame) -> Dict[int, str]:

        if stats.empty:
            return {}

        stats = stats.copy()

        # -------------------------------------------------
        # TOGGLE REGIME CLASSIFICATION METHOD
        # -------------------------------------------------

        # METHOD 1: SCORE-BASED (default)
        stats["score"] = stats["mean_return"] - 0.5 * stats["vol"]
        stats_sorted = stats.sort_values("score", ascending=False)

        # METHOD 2: MEAN-RETURN BASED
        # Uncomment the next line and comment out the score method above to switch
        # stats_sorted = stats.sort_values("mean_return", ascending=False)

        stats_sorted = stats_sorted.reset_index(drop=True)

        labels = ["Bull", "Sideways", "Bear"]
        mapping = {}

        for i in range(min(len(stats_sorted), len(labels))):
            mapping[int(stats_sorted.iloc[i]["regime"])] = labels[i]

        return mapping

    def segment_data_by_regime(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:

        if "regime" not in df.columns:
            raise ValueError("DataFrame must contain 'regime' column.")

        segments = {}

        for regime_id in sorted(df["regime"].dropna().unique()):
            regime_df = df[df["regime"] == regime_id].copy()
            segments[int(regime_id)] = regime_df

        return segments


def run_regime_detection(
    input_path: str,
    n_regimes: int = 3,
) -> Tuple[pd.Series, pd.DataFrame]:

    df = pd.read_csv(input_path)

    for cand in ["Date", "date", "datetime"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand).sort_index()
            break

    detector = RegimeDetector(n_regimes=n_regimes)

    regimes = detector.fit_and_predict(df)
    stats = detector.compute_regime_stats(df, regimes)

    mapping = detector.interpret_regimes(stats)
    regime_names = regimes.map(mapping)

    if not stats.empty:
        stats["regime_label"] = stats["regime"].map(mapping)

    return regimes, stats


if __name__ == "__main__":

    input_file = "data/processed/SPY_processed.csv"

    if os.path.exists(input_file):

        regimes, stats = run_regime_detection(input_file)

        df = pd.read_csv(input_file)

        for cand in ["Date", "date", "datetime"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand])
                df = df.set_index(cand).sort_index()
                break

        df["regime"] = regimes

        detector = RegimeDetector(n_regimes=3)
        regime_segments = detector.segment_data_by_regime(df)

        print("\n" + "=" * 70)
        print("PER-REGIME STATISTICS")
        print("=" * 70)

        ret_col = "log_returns" if "log_returns" in df.columns else "log_return"
        vol_cols = [c for c in df.columns if c.startswith("rolling_vol")]
        vol_col = vol_cols[0] if vol_cols else "rolling_vol"

        for regime_id, regime_df in sorted(regime_segments.items()):

            print(f"\nRegime {regime_id}:")
            print(f"  Sample count: {len(regime_df)}")

            mean_return = regime_df[ret_col].mean()
            print(f"  Mean return: {mean_return:.6f}")

            mean_vol = regime_df[vol_col].mean()
            print(f"  Mean volatility: {mean_vol:.6f}")

            std_return = regime_df[ret_col].std()
            if std_return > 0:
                sharpe = mean_return / std_return
                print(f"  Sharpe ratio: {sharpe:.6f}")
            else:
                print("  Sharpe ratio: N/A (zero std)")

        print("\n" + "=" * 70)

    else:
        print(f"File not found: {input_file}")