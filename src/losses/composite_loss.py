# src/losses/composite_loss.py

import torch
import torch.nn as nn


class CompositeRegimeLoss(nn.Module):
    """
    Composite loss function:
    L = w1*MSE + w2*Reliability + w3*IRM
    """

    def __init__(self, w_mse=0.7, w_reliability=0.2, w_irm=0.1):
        super().__init__()

        self.w_mse = w_mse
        self.w_reliability = w_reliability
        self.w_irm = w_irm

    def mse_loss(self, pred, target):
        """Prediction accuracy"""
        return torch.mean((pred - target) ** 2)

    def reliability_loss(self, uncertainty, pred_error):
        """
        Penalize excessive confidence in noisy signals.
        Based on Kendall & Gal:
        L_u = -log(var) + (pred_error^2 / var)
        """
        eps = 1e-6

        pred_error_sq = pred_error ** 2
        loss = -torch.log(uncertainty + eps) + (pred_error_sq / (uncertainty + eps))

        return torch.mean(loss)

    def irm_loss(self, risk_regimes, model_params):
        """
        Invariant Risk Minimization.
        Encourage similar decision boundaries across regimes.
        """
        if len(risk_regimes) < 2:
            return torch.tensor(0.0, device=risk_regimes[0].device)

        grads = []

        for risk in risk_regimes:
            grad = torch.autograd.grad(
                risk,
                model_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )

            # Flatten gradients (ignore None values)
            flat_grad = torch.cat([
                g.reshape(-1) for g in grad if g is not None
            ])

            grads.append(flat_grad)

        grads = torch.stack(grads)  # [num_regimes, num_params]

        # Compute variance across regimes
        grad_var = torch.var(grads, dim=0)

        return torch.mean(grad_var)

    def forward(self, pred, target, uncertainty, regime_indicator=None, model_params=None):
        """
        Args:
            pred: [batch, 1]
            target: [batch, 1]
            uncertainty: [batch, n_features]
            regime_indicator: [batch]
            model_params: iterable of model parameters
        """

        device = pred.device

        # Prediction error
        pred_error = pred - target

        # MSE
        mse = self.mse_loss(pred, target)

        # Reliability (use mean uncertainty across features)
        uncertainty_mean = torch.mean(uncertainty, dim=1, keepdim=True)
        reliability = self.reliability_loss(uncertainty_mean, pred_error)

        # IRM loss
        irm = torch.tensor(0.0, device=device)

        if regime_indicator is not None and model_params is not None:
            unique_regimes = torch.unique(regime_indicator)

            risk_regimes = []

            for r in unique_regimes:
                mask = regime_indicator == r

                if torch.sum(mask) == 0:
                    continue

                pred_r = pred[mask]
                target_r = target[mask]

                # Per-regime risk (MSE)
                risk_r = torch.mean((pred_r - target_r) ** 2)
                risk_regimes.append(risk_r)

            if len(risk_regimes) > 0:
                irm = self.irm_loss(risk_regimes, model_params)

        # Total loss
        total = (
            self.w_mse * mse +
            self.w_reliability * reliability +
            self.w_irm * irm
        )

        return {
            "total": total,
            "mse": mse,
            "reliability": reliability,
            "irm": irm
        }
