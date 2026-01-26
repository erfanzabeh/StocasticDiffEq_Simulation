"""
Spectrogram Tool
================

Time-frequency analysis via Short-Time Fourier Transform.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class SpectrogramTool(Tool):
    """
    Compute spectrogram (STFT) with optional Gaussian smoothing.
    
    Parameters
    ----------
    nperseg : int
        Segment length for STFT.
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 2.
    smooth_sigma : float, optional
        Gaussian smoothing sigma. None means no smoothing.
    
    Example
    -------
    >>> spec = SpectrogramTool(nperseg=256, smooth_sigma=1.0)
    >>> freqs, times, Sxx = spec(signal, fs=500)
    """
    
    def __init__(
        self,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        smooth_sigma: Optional[float] = None
    ):
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.smooth_sigma = smooth_sigma
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        freqs : np.ndarray
            Frequency axis.
        times : np.ndarray
            Time axis.
        Sxx : np.ndarray
            Power spectral density (freqs x times).
        """
        from scipy.signal import spectrogram
        
        nperseg = kwargs.get('nperseg', self.nperseg)
        noverlap = kwargs.get('noverlap', self.noverlap)
        smooth_sigma = kwargs.get('smooth_sigma', self.smooth_sigma)
        
        freqs, times, Sxx = spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap)
        
        if smooth_sigma is not None:
            from scipy.ndimage import gaussian_filter
            Sxx = gaussian_filter(Sxx, sigma=smooth_sigma)
        
        return freqs, times, Sxx
    
    @property
    def params(self) -> dict:
        return {
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "smooth_sigma": self.smooth_sigma,
        }
