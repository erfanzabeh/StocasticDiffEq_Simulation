"""
Generators
==========

Signal generators for various stochastic and deterministic systems.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Class
     - Description
     - Validated
   * - :class:`LorenzGenerator`
     - Classic 3D chaotic attractor (RK4)
     - ✅
   * - :class:`TVARGenerator`
     - Univariate time-varying AR(2)
     - ✅
   * - :class:`TVVARGenerator`
     - Multivariate 3D time-varying VAR(2)
     - ✅
   * - :class:`OUGenerator`
     - Ornstein-Uhlenbeck process
     - ✅
   * - :class:`NoiseGenerator`
     - White and colored (1/f) noise
     - ✅
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

