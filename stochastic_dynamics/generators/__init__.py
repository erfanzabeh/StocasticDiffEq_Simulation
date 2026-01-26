"""
Generators
==========

Signal generators for various stochastic and deterministic systems.

Available Generators:
- LorenzGenerator: Classic 3D chaotic attractor (RK4)
- TVARGenerator: Univariate time-varying AR(2)
- TVVARGenerator: Multivariate 3D time-varying VAR(2)
- OUGenerator: Ornstein-Uhlenbeck process
- NoiseGenerator: White and colored (1/f) noise
"""

from .lorenz import LorenzGenerator
from .tvar_generator import TVARGenerator
from .tvvar import TVVARGenerator
from .ou import OUGenerator
from .noise import NoiseGenerator

__all__ = [
    "LorenzGenerator",
    "TVARGenerator",
    "TVVARGenerator",
    "OUGenerator",
    "NoiseGenerator",
]

