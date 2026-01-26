"""
Surrogate Generation Tools
==========================

IAAFT and Fourier phase-randomized surrogates for null hypothesis testing.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class IAFFTSurrogateTool(Tool):
    """
    Iterative Amplitude-Adjusted Fourier Transform surrogate.
    
    Preserves:
    - Power spectrum (approximately)
    - Amplitude distribution (exactly)
    
    Destroys:
    - Phase correlations / nonlinear structure
    
    Parameters
    ----------
    n_iter : int
        Number of iterations (typically 10-100).
    seed : int, optional
        Random seed for reproducibility.
    
    Example
    -------
    >>> iaaft = IAFFTSurrogateTool(n_iter=50, seed=42)
    >>> surrogate = iaaft(signal, fs=500)
    """
    
    def __init__(self, n_iter: int = 50, seed: Optional[int] = None):
        self.n_iter = n_iter
        self.seed = seed
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Generate IAAFT surrogate.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency (not used, but required by interface).
        
        Returns
        -------
        np.ndarray
            Surrogate time series.
        """
        n_iter = kwargs.get('n_iter', self.n_iter)
        seed = kwargs.get('seed', self.seed)
        rng = np.random.default_rng(seed)
        
        n = len(x)
        x_sorted = np.sort(x)
        
        # Target spectrum from original signal
        X_target = np.fft.rfft(x)
        amp_target = np.abs(X_target)
        
        # Initialize with shuffled version
        idx = rng.permutation(n)
        s = x[idx].copy()
        
        for _ in range(n_iter):
            # FFT and impose target amplitude spectrum
            S = np.fft.rfft(s)
            S_new = amp_target * np.exp(1j * np.angle(S))
            s = np.fft.irfft(S_new, n=n)
            
            # Rank-order to match original amplitude distribution
            ranks = np.argsort(np.argsort(s))
            s = x_sorted[ranks]
        
        return s
    
    @property
    def name(self) -> str:
        return "IAAFT Surrogate"
    
    @property
    def params(self) -> dict:
        return {"n_iter": self.n_iter, "seed": self.seed}


class FourierSurrogateTool(Tool):
    """
    Simple Fourier surrogate with randomized phases.
    
    Preserves power spectrum exactly, amplitude distribution only approximately.
    Faster than IAAFT but less accurate for non-Gaussian signals.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    
    Example
    -------
    >>> fourier_surr = FourierSurrogateTool(seed=42)
    >>> surrogate = fourier_surr(signal, fs=500)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Generate Fourier surrogate with random phases.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency (not used, but required by interface).
        
        Returns
        -------
        np.ndarray
            Surrogate time series.
        """
        seed = kwargs.get('seed', self.seed)
        rng = np.random.default_rng(seed)
        
        X = np.fft.rfft(x)
        random_phases = rng.uniform(0, 2 * np.pi, len(X))
        
        # Keep DC and Nyquist components real
        random_phases[0] = 0
        if len(x) % 2 == 0:
            random_phases[-1] = 0
        
        X_surrogate = np.abs(X) * np.exp(1j * random_phases)
        return np.fft.irfft(X_surrogate, n=len(x))
    
    @property
    def name(self) -> str:
        return "Fourier Surrogate"
    
    @property
    def params(self) -> dict:
        return {"seed": self.seed}
