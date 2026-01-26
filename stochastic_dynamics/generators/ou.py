"""
Ornstein-Uhlenbeck Generator
============================

Mean-reverting stochastic process defined by the SDE:
    dx = -θ(x - μ)dt + σdW

where:
- θ = reversion rate (1/θ = characteristic time scale)
- μ = long-term mean
- σ = noise strength
- W = Wiener process

Supports two methods:
- exact: Analytical solution (statistically exact)
- euler: Euler-Maruyama approximation (first-order)

Source: ornstein_uhlenbeck_how_to_simulate.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Literal

from ..abstract import Generator


@partial(jax.jit, static_argnums=(0,))
def _ou_exact_core(
    n_steps: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    dt: float,
    key: jnp.ndarray
) -> jnp.ndarray:
    """
    Simulate OU process using exact analytical solution.
    
    The conditional distribution P(x_{t+dt} | x_t) is Gaussian with:
        mean = x_t * exp(-θ*dt) + μ * (1 - exp(-θ*dt))
        var  = σ² / (2θ) * (1 - exp(-2θ*dt))
    """
    # Precompute constants
    exp_neg_theta_dt = jnp.exp(-theta * dt)
    one_minus_exp = 1.0 - exp_neg_theta_dt
    var = (sigma**2 / (2.0 * theta)) * (1.0 - jnp.exp(-2.0 * theta * dt))
    sd = jnp.sqrt(var)
    
    # Generate all random increments
    noise = random.normal(key, shape=(n_steps,)) * sd
    
    # Simulate using scan
    def step(x_prev, eps):
        mean = x_prev * exp_neg_theta_dt + mu * one_minus_exp
        x_new = mean + eps
        return x_new, x_new
    
    _, x = jax.lax.scan(step, x0, noise)
    
    return x.reshape(-1, 1)


@partial(jax.jit, static_argnums=(0,))
def _ou_euler_core(
    n_steps: int,
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    dt: float,
    key: jnp.ndarray
) -> jnp.ndarray:
    """
    Simulate OU process using Euler-Maruyama approximation.
    
    Approximation (first-order in dt):
        x_{t+dt} ≈ x_t - θ*dt*(x_t - μ) + σ*sqrt(dt)*ε
    """
    sd = sigma * jnp.sqrt(dt)
    noise = random.normal(key, shape=(n_steps,)) * sd
    
    def step(x_prev, eps):
        mean = x_prev - theta * dt * (x_prev - mu)
        x_new = mean + eps
        return x_new, x_new
    
    _, x = jax.lax.scan(step, x0, noise)
    
    return x.reshape(-1, 1)


class OUGenerator(Generator):
    """
    Ornstein-Uhlenbeck process generator.
    
    Mean-reverting SDE:
        dx = -θ(x - μ)dt + σdW
    
    Parameters
    ----------
    theta : float
        Reversion rate. Characteristic time scale τ = 1/θ.
    mu : float
        Long-term mean.
    sigma : float
        Noise strength.
    x0 : float
        Initial state.
    method : str
        Integration method: 'exact' or 'euler'.
    seed : int
        Random seed for JAX PRNG.
    
    Example
    -------
    >>> ou = OUGenerator(theta=2.0, mu=1.0, sigma=0.4)
    >>> data = ou(n_steps=10000, dt=0.005)  # (10000, 1)
    >>> t = ou.time  # Time array
    """
    
    _core_exact = staticmethod(_ou_exact_core)
    _core_euler = staticmethod(_ou_euler_core)
    
    def __init__(
        self,
        theta: float = 2.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        x0: float = 0.0,
        method: Literal['exact', 'euler'] = 'exact',
        seed: int = 0
    ):
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.method = method
        self.seed = seed
        self._key = random.PRNGKey(seed)
    
    def __call__(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Generate OU trajectory.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps.
        dt : float
            Time step size.
        
        Returns
        -------
        np.ndarray
            Signal of shape (n_steps, 1).
        """
        self._dt = dt
        self._n_steps = n_steps
        
        # Split key for reproducibility on repeated calls
        self._key, subkey = random.split(self._key)
        
        if self.method == 'exact':
            data = self._core_exact(
                n_steps, self.theta, self.mu, self.sigma, 
                self.x0, dt, subkey
            )
        else:
            data = self._core_euler(
                n_steps, self.theta, self.mu, self.sigma,
                self.x0, dt, subkey
            )
        
        return np.asarray(data)
    
    @property
    def dim(self) -> int:
        return 1
    
    @property
    def tau(self) -> float:
        """Characteristic time scale (1/theta)."""
        return 1.0 / self.theta
    
    @property
    def stationary_std(self) -> float:
        """Stationary standard deviation: σ / sqrt(2θ)."""
        return self.sigma / np.sqrt(2.0 * self.theta)
    
    @property
    def params(self) -> dict:
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "x0": self.x0,
            "method": self.method,
            "seed": self.seed,
        }
