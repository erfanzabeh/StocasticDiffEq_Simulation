"""
Hilbert Transform & Envelope Tools
===================================

Analytic signal, envelope, and instantaneous phase extraction.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class HilbertTool(Tool):
    """
    Compute analytic signal via Hilbert transform.
    
    Parameters
    ----------
    use_scipy : bool
        If True, use scipy.signal.hilbert. If False, use custom FFT-based.
    
    Example
    -------
    >>> hilbert = HilbertTool()
    >>> analytic = hilbert(signal, fs=500)
    >>> envelope = np.abs(analytic)
    >>> phase = np.angle(analytic)
    """
    
    def __init__(self, use_scipy: bool = True):
        self.use_scipy = use_scipy
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute analytic signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D, real-valued).
        fs : float
            Sampling frequency in Hz (not used, but required by interface).
        
        Returns
        -------
        np.ndarray
            Complex analytic signal.
        """
        if self.use_scipy:
            from scipy.signal import hilbert
            return hilbert(x)
        else:
            return self._hilbert_custom(x)
    
    @staticmethod
    def _hilbert_custom(x: np.ndarray) -> np.ndarray:
        """Custom Hilbert transform via FFT."""
        n = len(x)
        X = np.fft.fft(x)
        
        # Create frequency-domain filter
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        
        return np.fft.ifft(X * h)
    
    @property
    def name(self) -> str:
        return "Hilbert Transform"
    
    @property
    def params(self) -> dict:
        return {"use_scipy": self.use_scipy}


class EnvelopeTool(Tool):
    """
    Extract amplitude envelope from signal.
    
    Parameters
    ----------
    use_scipy : bool
        If True, use scipy for Hilbert transform.
    
    Example
    -------
    >>> env_tool = EnvelopeTool()
    >>> envelope = env_tool(signal, fs=500)
    """
    
    def __init__(self, use_scipy: bool = True):
        self._hilbert = HilbertTool(use_scipy=use_scipy)
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute envelope (instantaneous amplitude).
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        np.ndarray
            Envelope of the signal.
        """
        analytic = self._hilbert(x, fs)
        return np.abs(analytic)
    
    def instantaneous_phase(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Compute instantaneous phase (unwrapped)."""
        analytic = self._hilbert(x, fs)
        return np.unwrap(np.angle(analytic))
    
    def instantaneous_frequency(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Compute instantaneous frequency in Hz."""
        phase = self.instantaneous_phase(x, fs)
        return np.gradient(phase) * fs / (2 * np.pi)
    
    @property
    def name(self) -> str:
        return "Envelope"
    
    @property
    def params(self) -> dict:
        return {"use_scipy": self._hilbert.use_scipy}
