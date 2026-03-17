#!/usr/bin/env python3
"""Smoke test for RegimeGuard model.
Run from repository root:
    python scripts/smoke_test_regime_guard.py
"""
import sys
import os
import traceback
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regime_guard import RegimeGuard

def main():
    config = {
        "seq_length": 60,
        "n_features": 15,
        "hidden_size": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.3,
    }

    torch.manual_seed(42)
    x = torch.randn(8, config["seq_length"], config["n_features"])

    try:
        model = RegimeGuard(config)
        output = model(x)

        assert output["prediction"].shape == (8, 1)
        assert output["trust_scores"].shape == (8, 15)
        assert output["uncertainty"].shape == (8, 15)
        assert not torch.isnan(output["prediction"]).any()
        print("Forward pass:  OK")

        model.zero_grad()
        output["prediction"].mean().backward()
        print("Backward pass: OK")

        print("\nSMOKE TEST: PASS")
    except Exception:
        traceback.print_exc()
        print("\nSMOKE TEST: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()