"""
Delay Embedder (Takens)
=======================

State-space reconstruction from scalar time series via delay embedding.

Takens' Theorem:
    y_t = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]

This reconstructs an m-dimensional manifold that is diffeomorphic 
to the original attractor.

Source: 1overf_sim.ipynb, TimeVariying_AR_Simualtion.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from ..abstract import Embedder

@partial(jax.jit, static_argnums=(1, 2))
def _delay_embed_core(
    x: jnp.ndarray,
    m: int,
    tau: int
) -> jnp.ndarray:
    """
    Construct delay-embedded matrix from scalar time series.
    
    Original logic from 1overf_sim.ipynb:
        T = len(x) - (m - 1) * tau
        return np.column_stack([x[i*tau : i*tau + T] for i in range(m)])
    
    Parameters
    ----------
    x : jnp.ndarray
        1D time series of shape (n_steps,).
    m : int
        Embedding dimension.
    tau : int
        Delay in samples.
    
    Returns
    -------
    jnp.ndarray
        Embedded matrix of shape (n_steps - (m-1)*tau, m).
    """
    n_rows = len(x) - (m - 1) * tau
    
    # Build indices for each delay column
    # Column i contains x[i*tau : i*tau + n_rows]
    indices = jnp.arange(n_rows).reshape(-1, 1) + jnp.arange(m) * tau
    
    return x[indices]


class DelayEmbedder(Embedder):
    """
    Takens delay embedding for state-space reconstruction.
    
    Constructs m-dimensional embedding from scalar time series:
        y_t = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
    
    Parameters
    ----------
    m : int
        Embedding dimension (default: 3).
    tau : int
        Delay in samples (default: 1).
    
    Attributes (after __call__)
    ---------------------------
    n_embedded : int
        Number of embedded points (n_steps - (m-1)*tau).
    
    Examples
    --------
    >>> emb = DelayEmbedder(m=3, tau=15)
    >>> x = lorenz_gen(n_steps=10000)[:, 0]  # Get x(t) component
    >>> embedded = emb(x)
    >>> embedded.shape
    (9970, 3)
    
    Notes
    -----
    The choice of τ is critical:
    - Too small: Points cluster near diagonal (redundant information)
    - Too large: Structure distorted (decorrelated)
    - Heuristic: Use first minimum of autocorrelation function
    """
    
    _core = staticmethod(_delay_embed_core)
    
    def __init__(self, m: int = 3, tau: int = 1):
        self.m = m
        self.tau = tau
        self._n_embedded: Optional[int] = None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Embed 1D signal.
        
        Parameters
        ----------
        x : np.ndarray
            1D input signal of shape (n_steps,).
        
        Returns
        -------
        np.ndarray
            Embedded signal of shape (n_steps - (m-1)*tau, m).
        
        Raises
        ------
        ValueError
            If embedding parameters are too large for signal length.
        """
        # Validate input
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {x.shape}")
        
        n_rows = len(x) - (self.m - 1) * self.tau
        if n_rows <= 0:
            raise ValueError(
                f"Embedding too large for signal length {len(x)}; "
                f"reduce m={self.m} or tau={self.tau}. "
                f"Need at least {(self.m - 1) * self.tau + 1} samples."
            )
        
        # Convert to JAX array, compute embedding, convert back to NumPy
        x_jax = jnp.asarray(x)
        embedded = self._core(x_jax, self.m, self.tau)
        
        self._n_embedded = n_rows
        
        return np.asarray(embedded)
    
    @property
    def embedding_dim(self) -> int:
        return self.m
    
    @property
    def n_embedded(self) -> Optional[int]:
        """Number of embedded points from last call. None if not yet called."""
        return self._n_embedded
    
    @property
    def params(self) -> dict:
        return {'m': self.m, 'tau': self.tau}
