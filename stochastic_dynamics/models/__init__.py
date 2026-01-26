"""
Models for Time-Varying AR Estimation
======================================

This module contains neural network and classical models for estimating
time-varying autoregressive (TVAR) coefficients.

Models:
- ARModel: Classic AR(p) with OLS (baseline)
- NeuralODE_TVAR: Neural ODE with Levinson-Durbin stability
- LagAttentionTVAR: Transformer attention over lag bank
- LagAttentionTVARFast: Bilinear scoring (no transformer)
- TransformerAR: CLS token over fixed lag sequence
- MLPTVAR: Hyper-network MLP
- TVAROperator: Neural operator with continuous delay kernel
"""

from .ar_ols import ARModel
from .neural_ode import NeuralODE_TVAR, levinson_order2, levinson_durbin
from .lag_attention import LagAttentionTVAR, LagAttentionTVARFast, build_lag_bank
from .transformer_ar import TransformerAR
from .mlp_tvar import MLPTVAR
from .neural_operator import TVAROperator, fractional_delay_samples

__all__ = [
    # Baseline
    "ARModel",
    # Neural ODE
    "NeuralODE_TVAR",
    "levinson_order2",
    "levinson_durbin",
    # Lag Attention
    "LagAttentionTVAR",
    "LagAttentionTVARFast",
    "build_lag_bank",
    # Transformer
    "TransformerAR",
    # MLP
    "MLPTVAR",
    # Neural Operator
    "TVAROperator",
    "fractional_delay_samples",
]
