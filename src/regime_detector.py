import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:

    def __init__(
        self,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = 3

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
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def _build_feature_matrix(self, df: pd.DataFrame):

        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        valid = df[self.feature_cols].dropna()
        X = valid.values.astype(np.float64)

        return X, valid.index

    def fit_and_predict(self, df: pd.DataFrame):

        X, valid_idx = self._build_feature_matrix(df)

        self.model.fit(X)
        labels = self.model.predict(X)

        return pd.Series(labels, index=valid_idx, name="regime")

    @staticmethod
    def compute_regime_stats(df: pd.DataFrame, regimes: pd.Series):

        combined = pd.DataFrame({
            "return": df["log_returns"],
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

    def interpret_regimes(self, stats: pd.DataFrame):

        if len(stats) != 3:
            raise ValueError("Model must produce exactly 3 regimes.")

        stats = stats.copy()

        # -----------------------------
        # SCORE-BASED METHOD
        # -----------------------------
        stats["score"] = stats["mean_return"] - 0.5 * stats["vol"]
        stats_sorted = stats.sort_values("score", ascending=False)

        # -----------------------------
        # MEAN-RETURN METHOD
        # -----------------------------
        # stats_sorted = stats.sort_values("mean_return", ascending=False)

        stats_sorted = stats_sorted.reset_index(drop=True)

        mapping = {
            int(stats_sorted.iloc[0]["regime"]): "Bull",
            int(stats_sorted.iloc[1]["regime"]): "Sideways",
            int(stats_sorted.iloc[2]["regime"]): "Bear",
        }

        return mapping


def run_regime_detection(input_path: str, output_dir: str):

    df = pd.read_csv(input_path)

    for cand in ["Date", "date", "datetime"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand).sort_index()
            break

    detector = RegimeDetector()

    regimes = detector.fit_and_predict(df)
    stats = detector.compute_regime_stats(df, regimes)
    mapping = detector.interpret_regimes(stats)

    regime_names = regimes.map(mapping)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    labels_path = os.path.join(output_dir, f"{base_name}_regime_labels.csv")
    stats_path = os.path.join(output_dir, f"{base_name}_regime_stats.csv")

    labels_df = pd.DataFrame({
        "regime_id": regimes,
        "regime_name": regime_names
    })

    labels_df.to_csv(labels_path)
    stats["regime_label"] = stats["regime"].map(mapping)
    stats.to_csv(stats_path, index=False)

    print(f"Processed: {base_name}")


if __name__ == "__main__":

    processed_folder = "data/processed"
    output_folder = "data/regime_output"

    os.makedirs(output_folder, exist_ok=True)

    csv_files = [
        f for f in os.listdir(processed_folder)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print("No CSV files found in processed folder.")

    for file in csv_files:
        file_path = os.path.join(processed_folder, file)
        run_regime_detection(file_path, output_folder)

    print("\nAll files processed successfully.\n")