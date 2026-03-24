import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RegimeGuardVisualizer:

    @staticmethod
    def plot_backtest_results(backtest_results, save_path=None):
        """Plot cumulative P&L and trading signals"""

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top plot — cumulative P&L
        axes[0].plot(
            backtest_results["cumulative_pnl_strat"],
            label="Strategy Cumulative P&L"
        )
        axes[0].plot(
            backtest_results["cumulative_pnl_bh"],
            label="Buy & Hold Cumulative P&L"
        )
        axes[0].set_ylabel("Cumulative P&L")
        axes[0].set_title("Backtest Results")
        axes[0].legend()
        axes[0].grid(True)

        # Bottom plot — trading signals
        axes[1].plot(
            np.asarray(backtest_results["signals"]).flatten(),
            label="Trading Signals"
        )
        axes[1].axhline(0, linestyle="--", linewidth=1, label="Zero Line")
        axes[1].set_ylabel("Signal")
        axes[1].set_xlabel("Time")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_feature_confidence(trust_scores, feature_names, save_path=None):
        """Plot feature trust scores over time"""

        plt.figure(figsize=(14, 6))

        trust_scores = np.asarray(trust_scores)

        for i, feature_name in enumerate(feature_names):
            plt.plot(trust_scores[:, i], label=feature_name)

        plt.ylabel("Trust Score")
        plt.xlabel("Time")
        plt.title("Feature Trust Scores Over Time")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_regime_analysis(df, regime_labels, save_path=None):
        """Visualize detected market regimes"""

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        regime_labels = np.asarray(regime_labels)
        unique_regimes = np.unique(regime_labels)

        # Top plot — price colored by regime
        for regime in unique_regimes:
            mask = regime_labels == regime
            axes[0].scatter(
                df.index[mask],
                df.loc[mask, "Adj Close"],
                label=f"Regime {regime}",
                s=12
            )
        axes[0].set_ylabel("Price")
        axes[0].set_title("Price by Detected Regime")
        axes[0].legend()
        axes[0].grid(True)

        # Middle plot — realized volatility line
        axes[1].plot(df.index, df["realized_vol"], label="Realized Volatility")
        axes[1].set_ylabel("Realized Vol")
        axes[1].grid(True)

        # Bottom plot — regime sequence as bar chart
        axes[2].bar(df.index, regime_labels, label="Regime Labels")
        axes[2].set_ylabel("Regime")
        axes[2].set_xlabel("Time")
        axes[2].set_title("Regime Sequence")
        axes[2].grid(True)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()