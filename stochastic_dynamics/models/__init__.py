"""
Models
======

This module contains neural network and classical models for estimating
time-varying autoregressive (TVAR) coefficients.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Class
     - Description
     - Validated
   * - :class:`ARModel`
     - Classic AR(p) with OLS (baseline)
     - ✅
   * - :class:`NeuralODE_TVAR`
     - Neural ODE with Levinson-Durbin stability
     - ✅
   * - :class:`LagAttentionTVAR`
     - Transformer attention over lag bank
     - ✅
   * - :class:`LagAttentionTVARFast`
     - Bilinear scoring (no transformer)
     - ✅
   * - :class:`TransformerAR`
     - CLS token over fixed lag sequence
     - ✅
   * - :class:`MLPTVAR`
     - Hyper-network MLP
     - ✅
   * - :class:`TVAROperator`
     - Neural operator with continuous delay kernel
     - ✅
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
