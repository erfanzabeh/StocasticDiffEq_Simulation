"""
Lorenz System Generator
=======================

Classic 3D chaotic attractor with RK4 integration.

Equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz

Classic parameters: σ=10, ρ=28, β=8/3

Source: DeepLagAttention.ipynb, NeuralOperator.ipynb, TimeVariying_AR_Simualtion.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..abstract import Generator

@partial(jax.jit, static_argnums=(0,))
def _lorenz_core(
    n_steps: int,
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
    x0: float,
    y0: float,
    z0: float
) -> jnp.ndarray:
    """
    RK4 integration of the Lorenz system.
    
    Returns
    -------
    jnp.ndarray
        Trajectory of shape (n_steps, 3).
    """
    def lorenz_derivs(state, sigma, rho, beta):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return jnp.array([dx, dy, dz])
    
    def rk4_step(state, _):
        k1 = lorenz_derivs(state, sigma, rho, beta)
        k2 = lorenz_derivs(state + 0.5 * dt * k1, sigma, rho, beta)
        k3 = lorenz_derivs(state + 0.5 * dt * k2, sigma, rho, beta)
        k4 = lorenz_derivs(state + dt * k3, sigma, rho, beta)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state, new_state
    
    init_state = jnp.array([x0, y0, z0])
    _, trajectory = jax.lax.scan(rk4_step, init_state, None, length=n_steps)
    
    return trajectory


class LorenzGenerator(Generator):
    """
    Lorenz system generator with RK4 integration.
    
    Parameters
    ----------
    sigma : float
        σ parameter (default: 10.0)
    rho : float
        ρ parameter (default: 28.0)
    beta : float
        β parameter (default: 8/3)
    x0 : tuple
        Initial conditions (x, y, z). Default: (1.0, 1.0, 1.0)
    
    Examples
    --------
    >>> gen = LorenzGenerator(sigma=10, rho=28, beta=8/3)
    >>> data = gen(n_steps=20000, dt=0.005)
    >>> data.shape
    (20000, 3)
    >>> gen.time.shape
    (20000,)
    >>> x_component = data[:, 0]  # Extract x(t) for embedding
    """
    
    # Attach JIT core as class attribute
    _core = staticmethod(_lorenz_core)
    
    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8/3,
        x0: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x0 = x0
    
    def __call__(self, n_steps: int, dt: float = 0.005) -> np.ndarray:
        """
        Generate Lorenz trajectory.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps.
        dt : float
            Time step size (default: 0.005).
        
        Returns
        -------
        np.ndarray
            Trajectory of shape (n_steps, 3).
        """
        data = self._core(
            n_steps, dt,
            self.sigma, self.rho, self.beta,
            self.x0[0], self.x0[1], self.x0[2]
        )
        self._dt = dt
        self._n_steps = n_steps
        return np.asarray(data)  # Convert JAX array to NumPy
    
    @property
    def dim(self) -> int:
        return 3
    
    @property
    def params(self) -> dict:
        return {
            'sigma': self.sigma,
            'rho': self.rho,
            'beta': self.beta,
            'x0': self.x0
        }
