"""
Generators
==========

Signal generators for various stochastic and deterministic systems.

Functions
---------
- lorenz: Classic 3D chaotic attractor (RK4)
- tvar: Univariate time-varying AR(2)
- tvvar: Multivariate 3D time-varying VAR(2)
- ou: Ornstein-Uhlenbeck process
- noise: White and colored (1/f) noise
"""

from .lorenz import lorenz
from .tvar_generator import TVARGenerator
from .tvvar import TVVARGenerator
from .ou import OUGenerator
from .noise import NoiseGenerator

__all__ = [
    "lorenz",
    "TVARGenerator",
    "TVVARGenerator",
    "OUGenerator",
    "NoiseGenerator",
]

