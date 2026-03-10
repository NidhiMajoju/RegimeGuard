# src/models/trust_head.py
import torch
import torch.nn as nn

class FeatureTrustHead(nn.Module):
    """
    Bayesian uncertainty module for signal reliability
    Based on Kendall & Gal (2017): aleatoric + epistemic uncertainty
    """
    def __init__(self, input_size, n_features, hidden_size=128, dropout=0.3):
        super().__init__()
        self.n_features = n_features

        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, n_features),
            nn.Softplus()
        )
 
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_size, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [batch, input_size] from encoder output
        Returns:
            trust_scores: [batch, n_features] confidence per signal
            uncertainty: [batch, n_features] epistemic uncertainty
        """
        
        features = self.trunk(x)
        trust_scores = self.reliability_head(features)
        uncertainty = self.uncertainty_head(features)

        return trust_scores, uncertainty
