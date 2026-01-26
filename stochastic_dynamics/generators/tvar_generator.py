"""
Time-Varying AR Generator
=========================

Univariate AR(2) with sinusoidally varying coefficients.

Model:
    x(t) = a1(t) * x(t-1) + a2(t) * x(t-2) + ε(t)
    
where a1(t) and a2(t) vary smoothly over time:
    a1(t) = a1_base + a1_amp * sin(2π * t / a1_period + a1_phase)
    a2(t) = a2_base + a2_amp * sin(2π * t / a2_period + a2_phase)

Source: TimeVariying_AR_Simualtion.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Tuple

from ..abstract import Generator


@partial(jax.jit, static_argnums=(0, 1))
def _tvar_core(
    n_steps: int,
    burn_in: int,
    a1_base: float,
    a1_amp: float,
    a1_period: float,
    a1_phase: float,
    a2_base: float,
    a2_amp: float,
    a2_period: float,
    a2_phase: float,
    noise_std: float,
    key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate univariate TV-AR(2) process.
    
    Returns
    -------
    tuple
        (data, a1, a2) where:
        - data: (n_steps, 1) signal
        - a1: (n_steps,) first AR coefficient over time
        - a2: (n_steps,) second AR coefficient over time
    """
    T_total = n_steps + burn_in
    
    # Generate time-varying coefficients
    t = jnp.arange(T_total, dtype=jnp.float32)
    a1_full = a1_base + a1_amp * jnp.sin(2.0 * jnp.pi * t / a1_period + a1_phase)
    a2_full = a2_base + a2_amp * jnp.sin(2.0 * jnp.pi * t / a2_period + a2_phase)
    
    # Generate noise
    eps = random.normal(key, shape=(T_total,)) * noise_std
    
    # Simulate AR(2) using scan
    def ar2_step(carry, inputs):
        x_prev1, x_prev2 = carry
        a1_t, a2_t, eps_t = inputs
        x_t = a1_t * x_prev1 + a2_t * x_prev2 + eps_t
        return (x_t, x_prev1), x_t
    
    init_carry = (0.0, 0.0)
    inputs = (a1_full, a2_full, eps)
    _, x_full = jax.lax.scan(ar2_step, init_carry, inputs)
    
    # Remove burn-in and reshape to (n_steps, 1)
    data = x_full[burn_in:].reshape(-1, 1)
    a1 = a1_full[burn_in:]
    a2 = a2_full[burn_in:]
    
    return data, a1, a2


class TVARGenerator(Generator):
    """
    Univariate Time-Varying AR(2) generator.
    
    Coefficients vary sinusoidally:
        a1(t) = a1_base + a1_amp * sin(2π * t / a1_period + a1_phase)
        a2(t) = a2_base + a2_amp * sin(2π * t / a2_period + a2_phase)
    
    Parameters
    ----------
    a1_base, a1_amp, a1_period, a1_phase : float
        Parameters for a1(t) coefficient.
    a2_base, a2_amp, a2_period, a2_phase : float
        Parameters for a2(t) coefficient.
    noise_std : float
        Standard deviation of innovation noise.
    burn_in : int
        Number of initial samples to discard (default: 200).
    seed : int
        Random seed for reproducibility.
    
    Attributes (after __call__)
    ---------------------------
    a1 : np.ndarray
        Ground truth a1(t) coefficients of shape (n_steps,).
    a2 : np.ndarray
        Ground truth a2(t) coefficients of shape (n_steps,).
    
    Examples
    --------
    >>> gen = TVARGenerator(a1_base=0.6, a1_amp=0.3, a1_period=400)
    >>> data = gen(n_steps=3000, dt=0.001)
    >>> data.shape
    (3000, 1)
    >>> gen.a1.shape  # Ground truth coefficients
    (3000,)
    """
    
    _core = staticmethod(_tvar_core)
    
    def __init__(
        self,
        a1_base: float = 0.6,
        a1_amp: float = 0.3,
        a1_period: float = 400.0,
        a1_phase: float = 0.0,
        a2_base: float = -0.3,
        a2_amp: float = 0.2,
        a2_period: float = 600.0,
        a2_phase: float = 1.2,
        noise_std: float = 0.1,
        burn_in: int = 200,
        seed: int = 42
    ):
        super().__init__()
        self.a1_base = a1_base
        self.a1_amp = a1_amp
        self.a1_period = a1_period
        self.a1_phase = a1_phase
        self.a2_base = a2_base
        self.a2_amp = a2_amp
        self.a2_period = a2_period
        self.a2_phase = a2_phase
        self.noise_std = noise_std
        self.burn_in = burn_in
        self.seed = seed
        
        # Ground truth (populated after call)
        self._a1: Optional[np.ndarray] = None
        self._a2: Optional[np.ndarray] = None
    
    def __call__(self, n_steps: int, dt: float = 0.001) -> np.ndarray:
        """
        Generate TV-AR(2) signal.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps (after burn-in).
        dt : float
            Time step size (default: 0.001).
        
        Returns
        -------
        np.ndarray
            Signal of shape (n_steps, 1).
        """
        data, a1, a2 = self._core(
            n_steps,
            self.burn_in,
            self.a1_base, self.a1_amp, self.a1_period, self.a1_phase,
            self.a2_base, self.a2_amp, self.a2_period, self.a2_phase,
            self.noise_std,
            random.PRNGKey(self.seed)
        )
        
        self._dt = dt
        self._n_steps = n_steps
        self._a1 = np.asarray(a1)  # Convert JAX array to NumPy
        self._a2 = np.asarray(a2)
        
        return np.asarray(data)  # Convert JAX array to NumPy
    
    @property
    def dim(self) -> int:
        return 1
    
    @property
    def a1(self) -> Optional[np.ndarray]:
        """Ground truth a1(t) coefficients. None if not yet called."""
        return self._a1
    
    @property
    def a2(self) -> Optional[np.ndarray]:
        """Ground truth a2(t) coefficients. None if not yet called."""
        return self._a2
    
    @property
    def ground_truth(self) -> Optional[dict]:
        """Ground truth coefficients as dict. None if not yet called."""
        if self._a1 is None:
            return None
        return {'a1': self._a1, 'a2': self._a2}
    
    @property
    def params(self) -> dict:
        return {
            'a1_base': self.a1_base,
            'a1_amp': self.a1_amp,
            'a1_period': self.a1_period,
            'a1_phase': self.a1_phase,
            'a2_base': self.a2_base,
            'a2_amp': self.a2_amp,
            'a2_period': self.a2_period,
            'a2_phase': self.a2_phase,
            'noise_std': self.noise_std,
            'burn_in': self.burn_in,
            'seed': self.seed
        }
