# src/models/regime_guard.py

import torch
import torch.nn as nn
from .temporal_encoder import TemporalEncoder
from .attention_model import MultiHeadAttention
from .trust_head import FeatureTrustHead


class RegimeGuard(nn.Module):
    """
    Complete RegimeGuard architecture:
    Temporal Encoder → Multi-Head Attention → Feature Trust → Prediction
    """

    def __init__(self, config):
        super().__init__()

        # unpack config fields
        self.seq_length = config["seq_length"]
        self.n_features = config["n_features"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.3)

        # initialize TemporalEncoder
        self.encoder = TemporalEncoder(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        encoder_out_size = self.hidden_size * 2 # bidirectional

        # initialize MultiHeadAttention
        self.attention = MultiHeadAttention(
            hidden_size=encoder_out_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # initialize FeatureTrustHead
        self.trust_head = FeatureTrustHead(
            input_size=encoder_out_size,
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        # initialize prediction head (nn.Sequential)
        # input: encoder hidden + n_features → 64 → 32 → 1
        # include ReLU activations and Dropout between layers
        self.prediction_head = nn.Sequential(
            nn.Linear(encoder_out_size + self.n_features, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )


    def forward(self, x):
        """
        Args:
            x: [batch, seq_length, n_features]
        Returns:
            pred: [batch, 1] return prediction
            trust_scores: [batch, n_features] signal reliability
            uncertainty: [batch, n_features] epistemic uncertainty
        """

        # Temporal encoding  ->  [batch, seq_len, hidden*2]
        encoder_out, (hidden, cell) = self.encoder(x)

        # Self-attention (Q=K=V)  ->  [batch, seq_len, hidden*2], attn weights
        attn_out, attn_weights = self.attention(encoder_out, encoder_out, encoder_out)

        # pool attention output to single vector (last timestep) ->[batch, hidden*2]
        pooled = attn_out[:, -1, :]

        # get trust_scores and uncertainty from trust head
        trust_scores, uncertainty = self.trust_head(pooled)

        # apply confidence weighting to trust scores
        # scale from [0, 1] → [0.5, 1.0]
        weighted_trust = 0.5 + 0.5 * trust_scores

        # concatenate pooled vector and weighted trust scores
        # then pass through prediction head
        combined = torch.cat([pooled, weighted_trust], dim=-1)
        prediction = self.prediction_head(combined)

        return {
            "prediction": prediction,
            "trust_scores": trust_scores,
            "uncertainty": uncertainty,
            "attention_weights": attn_weights,
        }
