import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for time-series inputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Bidirectional LSTM so the effective hidden size doubles
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.hidden_size = hidden_size * 2

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch, seq_len, features)

        Returns:
            output: LSTM outputs for all time steps.
            (hidden, cell): Final hidden and cell states from both directions.
        """
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)
