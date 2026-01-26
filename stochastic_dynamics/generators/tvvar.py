"""
Multivariate Time-Varying VAR Generator
========================================

3D VAR(2) process with rotating coupling matrices for guaranteed stability.

Model:
    X(t) = A1(t) @ X(t-1) + A2(t) @ X(t-2) + ε(t)

where A_i(t) = R(t) @ diag(λ) @ R(t).T with orthogonal rotation R(t).

This ensures:
- Eigenvalues remain fixed (controllable stability)
- Coupling structure rotates smoothly over time
- Spectral radius < 1 for stable dynamics

Source: TimeVariying_AR_Simualtion.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Tuple

from ..abstract import Generator


def _rotation_z(theta: float) -> jnp.ndarray:
    """3D rotation matrix around z-axis."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def _rotation_y(theta: float) -> jnp.ndarray:
    """3D rotation matrix around y-axis."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ])


@partial(jax.jit, static_argnums=(0, 1))
def _tvvar3_core(
    n_steps: int,
    burn_in: int,
    alpha: jnp.ndarray,  # (3,) eigenvalues for A1
    beta: jnp.ndarray,   # (3,) eigenvalues for A2
    rotation_period1: float,
    rotation_period2: float,
    noise_scale: float,
    key: jnp.ndarray
) -> jnp.ndarray:
    """
    Simulate 3D time-varying VAR(2) with rotating coupling matrices.
    """
    T_total = n_steps + burn_in
    
    # Generate noise
    noise = random.normal(key, shape=(T_total, 3)) * noise_scale
    
    def step(carry, inputs):
        X_prev1, X_prev2 = carry  # X(t-1), X(t-2)
        k, eps = inputs
        
        # Time-varying rotation angles
        th1 = 2.0 * jnp.pi * k / rotation_period1
        th2 = 2.0 * jnp.pi * k / rotation_period2 + 0.7
        
        # Rotation matrices
        R1 = _rotation_z(th1) @ _rotation_y(th1 * 0.5)
        R2 = _rotation_y(th2) @ _rotation_z(th2 * 0.5)
        
        # A = R @ diag(eigenvalues) @ R.T
        A1 = R1 @ jnp.diag(alpha) @ R1.T
        A2 = R2 @ jnp.diag(beta) @ R2.T
        
        X_new = A1 @ X_prev1 + A2 @ X_prev2 + eps
        
        return (X_new, X_prev1), X_new
    
    # Initial state
    init_carry = (jnp.zeros(3), jnp.zeros(3))
    inputs = (jnp.arange(T_total, dtype=jnp.float32), noise)
    
    _, X_full = jax.lax.scan(step, init_carry, inputs)
    
    # Remove burn-in
    return X_full[burn_in:]


class TVVARGenerator(Generator):
    """
    3D Time-Varying VAR(2) with rotating coupling matrices.
    
    Ensures stability by using orthogonal rotations to preserve eigenvalues
    while the coupling structure evolves smoothly over time.
    
    Parameters
    ----------
    alpha : array-like
        Eigenvalues for first-lag matrix A1 (length 3).
        Default: [0.45, 0.35, 0.30]
    beta : array-like
        Eigenvalues for second-lag matrix A2 (length 3).
        Negative values help with damping.
        Default: [-0.10, -0.08, -0.06]
    rotation_period1 : float
        Period for first rotation (samples).
    rotation_period2 : float
        Period for second rotation (samples).
    noise_scale : float
        Scale of innovation noise.
    burn_in : int
        Number of initial samples to discard.
    seed : int
        Random seed for JAX PRNG.
    
    Example
    -------
    >>> tvvar = TVVARGenerator()
    >>> data = tvvar(n_steps=5000, dt=1.0)  # (5000, 3)
    >>> X1, X2, X3 = data[:, 0], data[:, 1], data[:, 2]
    """
    
    _core = staticmethod(_tvvar3_core)
    
    def __init__(
        self,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        rotation_period1: float = 500.0,
        rotation_period2: float = 900.0,
        noise_scale: float = 1.0,
        burn_in: int = 300,
        seed: int = 0
    ):
        super().__init__()
        
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.45, 0.35, 0.30])
        self.beta = np.array(beta) if beta is not None else np.array([-0.10, -0.08, -0.06])
        self.rotation_period1 = rotation_period1
        self.rotation_period2 = rotation_period2
        self.noise_scale = noise_scale
        self.burn_in = burn_in
        self.seed = seed
        self._key = random.PRNGKey(seed)
    
    def __call__(self, n_steps: int, dt: float = 1.0) -> np.ndarray:
        """
        Generate 3D TV-VAR trajectory.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps (after burn-in).
        dt : float
            Time step size (for time property).
        
        Returns
        -------
        np.ndarray
            Signal of shape (n_steps, 3).
        """
        self._dt = dt
        self._n_steps = n_steps
        
        self._key, subkey = random.split(self._key)
        
        data = self._core(
            n_steps,
            self.burn_in,
            jnp.array(self.alpha),
            jnp.array(self.beta),
            self.rotation_period1,
            self.rotation_period2,
            self.noise_scale,
            subkey
        )
        
        return np.asarray(data)
    
    @property
    def dim(self) -> int:
        return 3
    
    @property
    def params(self) -> dict:
        return {
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "rotation_period1": self.rotation_period1,
            "rotation_period2": self.rotation_period2,
            "noise_scale": self.noise_scale,
            "burn_in": self.burn_in,
            "seed": self.seed,
        }
