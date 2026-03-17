"""
Quick smoke test for TemporalEncoder.
Run: python scripts/smoke_temporal_encoder.py
"""
from pathlib import Path
import sys
import torch

# Ensure project root on sys.path so `src` imports work when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_encoder import TemporalEncoder


def main() -> None:
    batch, seq_len, features = 4, 10, 8
    x = torch.randn(batch, seq_len, features)

    model = TemporalEncoder(
        input_size=features,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
    )

    out, (h, c) = model(x)

    assert out.shape == (batch, seq_len, 512), f"Unexpected out shape {out.shape}"
    assert h.shape == (4, batch, 256), f"Unexpected hidden shape {h.shape}"
    assert c.shape == (4, batch, 256), f"Unexpected cell shape {c.shape}"

    print("Smoke test passed.")
    print(f"out shape: {out.shape}")
    print(f"h shape:   {h.shape}")
    print(f"c shape:   {c.shape}")


if __name__ == "__main__":
    main()
