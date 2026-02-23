#!/usr/bin/env python3
"""Simple smoke test to verify `regime.detector` and `predictive_power` functions.

Run from repository root:
    python scripts/smoke_test_all.py
"""
import os
import sys
import importlib.util
import traceback
import numpy as np
import pandas as pd


def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths to target files
    rd_path = os.path.join(repo_root, "src", "regime.detector.py")
    pp_path = os.path.join(repo_root, "data", "predictive_power.py")

    errors = []

    # Load modules
    try:
        rd_mod = load_module_from_path(rd_path, "regime_detector_module")
    except Exception:
        print("Failed to load regime.detector:")
        traceback.print_exc()
        sys.exit(1)

    try:
        pp_mod = load_module_from_path(pp_path, "predictive_power_module")
    except Exception:
        print("Failed to load predictive_power:")
        traceback.print_exc()
        sys.exit(2)

    # Validate presence of expected symbols
    if not hasattr(rd_mod, "RegimeDetector"):
        print("RegimeDetector class not found in regime.detector")
        sys.exit(3)

    compute_future_return = getattr(pp_mod, "compute_future_return", None)
    feature_pred = getattr(pp_mod, "feature_predictive_power_by_regime", None)
    if compute_future_return is None or feature_pred is None:
        print("predictive_power functions missing in data/predictive_power.py")
        sys.exit(4)

    # Build synthetic dataset
    np.random.seed(0)
    n = 260
    rets = np.random.normal(0.0006, 0.01, size=n)
    price = 100 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Adj Close": price}, index=dates)
    df["log_returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["rolling_vol_20"] = df["log_returns"].rolling(20).std()
    df["drawdown"] = df["Adj Close"] / df["Adj Close"].cummax() - 1
    df = df.dropna()

    # Run RegimeDetector
    try:
        RegimeDetector = rd_mod.RegimeDetector
        det = RegimeDetector(n_regimes=3)
        regimes = det.fit_and_predict(df)
        stats = det.compute_regime_stats(df, regimes)
        mapping = det.interpret_regimes(stats) if hasattr(det, "interpret_regimes") else {}
        print("Regime detector: OK\n", stats)
        print("Mapping:", mapping)
    except Exception:
        print("Regime detector failed:")
        traceback.print_exc()
        errors.append("regime")

    # Run predictive power functions
    try:
        df2 = compute_future_return(df.copy(), horizons=[1, 5, 20])
        pp = feature_pred(df2, regimes)
        print("Predictive power: OK\n", pp.head())
    except Exception:
        print("Predictive power failed:")
        traceback.print_exc()
        errors.append("predictive_power")

    if errors:
        print("SMOKE TEST: FAIL", errors)
        sys.exit(10)

    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
