"""
Noise Generators
================

White and colored (1/f^α) noise generators.

Available noise types:
- white: Flat PSD, no temporal correlation
- pink (α=1): 1/f noise, long-range correlations
- brown/red (α=2): 1/f² noise, random walk-like

Source: 1overf_sim.ipynb
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Optional, Literal

from ..abstract import Generator


@partial(jax.jit, static_argnums=(0,))
def _white_noise_core(
    n_steps: int,
    key: jnp.ndarray
) -> jnp.ndarray:
    """Generate standard white noise."""
    return random.normal(key, shape=(n_steps, 1))


@partial(jax.jit, static_argnums=(0,))
def _colored_noise_core(
    n_steps: int,
    alpha: float,
    key: jnp.ndarray
) -> jnp.ndarray:
    """
    Generate 1/f^α colored noise via spectral shaping.
    
    Method: Generate white noise in frequency domain, scale magnitudes 
    by 1/f^(α/2), then inverse FFT back to time domain.
    """
    # Generate complex white noise in frequency domain
    key1, key2 = random.split(key)
    n_freq = n_steps // 2 + 1
    
    re = random.normal(key1, shape=(n_freq,))
    im = random.normal(key2, shape=(n_freq,))
    X = re + 1j * im
    
    # Frequency axis (normalized, 0 to 0.5)
    freqs = jnp.fft.rfftfreq(n_steps)
    
    # Shape magnitude: 1/f^(alpha/2) so PSD ∝ 1/f^alpha
    # Handle DC component (freq=0) separately
    mag_shape = jnp.where(
        freqs > 0,
        1.0 / (freqs ** (alpha / 2.0)),
        0.0
    )
    
    X_shaped = X * mag_shape
    x = jnp.fft.irfft(X_shaped, n=n_steps)
    
    # Standardize to zero mean, unit variance
    x = (x - jnp.mean(x)) / (jnp.std(x) + 1e-12)
    
    return x.reshape(-1, 1)


class NoiseGenerator(Generator):
    """
    White and colored noise generator.
    
    Parameters
    ----------
    noise_type : str
        Type of noise: 'white', 'pink', 'brown', or 'colored'.
    alpha : float
        Spectral exponent for colored noise.
        - α=0: white noise (flat PSD)
        - α=1: pink noise (1/f)
        - α=2: brown/red noise (1/f²)
        Ignored if noise_type is 'white', 'pink', or 'brown'.
    seed : int
        Random seed for JAX PRNG.
    
    Example
    -------
    >>> # White noise
    >>> white = NoiseGenerator(noise_type='white')
    >>> data = white(n_steps=10000, dt=0.001)
    
    >>> # Pink (1/f) noise
    >>> pink = NoiseGenerator(noise_type='pink')
    >>> data = pink(n_steps=10000, dt=0.001)
    
    >>> # Custom colored noise
    >>> colored = NoiseGenerator(noise_type='colored', alpha=1.5)
    >>> data = colored(n_steps=10000, dt=0.001)
    """
    
    _core_white = staticmethod(_white_noise_core)
    _core_colored = staticmethod(_colored_noise_core)
    
    # Preset alpha values
    PRESETS = {
        'white': 0.0,
        'pink': 1.0,
        'brown': 2.0,
        'red': 2.0,
    }
    
    def __init__(
        self,
        noise_type: Literal['white', 'pink', 'brown', 'red', 'colored'] = 'white',
        alpha: float = 1.0,
        seed: int = 0
    ):
        super().__init__()
        self.noise_type = noise_type
        
        # Set alpha from preset or use provided value
        if noise_type in self.PRESETS:
            self.alpha = self.PRESETS[noise_type]
        else:
            self.alpha = alpha
        
        self.seed = seed
        self._key = random.PRNGKey(seed)
    
    def __call__(self, n_steps: int, dt: float = 1.0) -> np.ndarray:
        """
        Generate noise signal.
        
        Parameters
        ----------
        n_steps : int
            Number of samples.
        dt : float
            Time step size (used for time property, doesn't affect noise).
        
        Returns
        -------
        np.ndarray
            Noise signal of shape (n_steps, 1), zero mean, unit variance.
        """
        self._dt = dt
        self._n_steps = n_steps
        
        self._key, subkey = random.split(self._key)
        
        if self.alpha == 0.0:
            data = self._core_white(n_steps, subkey)
        else:
            data = self._core_colored(n_steps, self.alpha, subkey)
        
        return np.asarray(data)
    
    @property
    def dim(self) -> int:
        return 1
    
    @property
    def params(self) -> dict:
        return {
            "noise_type": self.noise_type,
            "alpha": self.alpha,
            "seed": self.seed,
        }
