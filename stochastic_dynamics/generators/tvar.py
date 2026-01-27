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

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Tuple


@partial(jax.jit, static_argnums=(0, 1))
def tvar(n_steps: int, burn_in: int = 200, a1_base: float = 0.6, a1_amp: float = 0.3, 
         a1_period: float = 400.0, a1_phase: float = 0.0, a2_base: float = -0.3, 
         a2_amp: float = 0.2, a2_period: float = 600.0, a2_phase: float = 1.2,
         noise_std: float = 0.1, seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Univariate Time-Varying AR(2) generator.
    
    Coefficients vary sinusoidally:
        a1(t) = a1_base + a1_amp * sin(2π * t / a1_period + a1_phase)
        a2(t) = a2_base + a2_amp * sin(2π * t / a2_period + a2_phase)
    
    Parameters
    ----------
    n_steps : int
        Number of time steps (after burn-in).
    burn_in : int
        Number of initial samples to discard.
    a1_base, a1_amp, a1_period, a1_phase : float
        Parameters for a1(t) coefficient.
    a2_base, a2_amp, a2_period, a2_phase : float
        Parameters for a2(t) coefficient.
    noise_std : float
        Standard deviation of innovation noise.
    seed : int
        Random seed.
    
    Returns
    -------
    tuple
        (data, a1, a2) where:
        - data: (n_steps,) signal
        - a1: (n_steps,) first AR coefficient over time
        - a2: (n_steps,) second AR coefficient over time
    
    Example
    -------
    >>> x, a1, a2 = tvar(3000)
    """
    key = random.PRNGKey(seed)
    T_total = n_steps + burn_in
    
    t = jnp.arange(T_total, dtype=jnp.float32)
    a1_full = a1_base + a1_amp * jnp.sin(2.0 * jnp.pi * t / a1_period + a1_phase)
    a2_full = a2_base + a2_amp * jnp.sin(2.0 * jnp.pi * t / a2_period + a2_phase)
    
    eps = random.normal(key, shape=(T_total,)) * noise_std
    
    def ar2_step(carry, inputs):
        x_prev1, x_prev2 = carry
        a1_t, a2_t, eps_t = inputs
        x_t = a1_t * x_prev1 + a2_t * x_prev2 + eps_t
        return (x_t, x_prev1), x_t
    
    init_carry = (0.0, 0.0)
    inputs = (a1_full, a2_full, eps)
    _, x_full = jax.lax.scan(ar2_step, init_carry, inputs)
    
    data = x_full[burn_in:]
    a1 = a1_full[burn_in:]
    a2 = a2_full[burn_in:]
    
    return data, a1, a2
