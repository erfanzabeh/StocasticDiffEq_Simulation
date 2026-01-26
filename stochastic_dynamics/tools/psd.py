"""
Power Spectral Density Tool
===========================

PSD estimation via Welch's method.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class PSDTool(Tool):
    """
    Power Spectral Density estimation using Welch's method.
    
    Parameters
    ----------
    nperseg : int
        Length of each segment for Welch's method.
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 2.
    use_scipy : bool
        If True, use scipy.signal.welch. If False, use custom implementation.
    
    Example
    -------
    >>> psd = PSDTool(nperseg=256)
    >>> freqs, power = psd(signal, fs=500)
    """
    
    def __init__(
        self,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        use_scipy: bool = True
    ):
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.use_scipy = use_scipy
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute PSD of signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        freqs : np.ndarray
            Frequency axis in Hz.
        power : np.ndarray
            Power spectral density.
        """
        nperseg = kwargs.get('nperseg', self.nperseg)
        noverlap = kwargs.get('noverlap', self.noverlap)
        
        if self.use_scipy:
            return self._welch_scipy(x, fs, nperseg, noverlap)
        else:
            return self._welch_custom(x, fs, nperseg, noverlap)
    
    @staticmethod
    def _welch_scipy(
        x: np.ndarray,
        fs: float,
        nperseg: int,
        noverlap: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD using scipy."""
        from scipy.signal import welch
        return welch(x, fs, nperseg=nperseg, noverlap=noverlap)
    
    @staticmethod
    def _welch_custom(
        x: np.ndarray,
        fs: float,
        nperseg: int,
        noverlap: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Custom Welch PSD implementation (NumPy only)."""
        step = nperseg - noverlap
        n_seg = (len(x) - nperseg) // step + 1
        
        window = np.hanning(nperseg)
        scale = 1.0 / (fs * (window ** 2).sum())
        
        psd_accum = np.zeros(nperseg // 2 + 1)
        
        for i in range(n_seg):
            seg = x[i * step : i * step + nperseg]
            seg = (seg - seg.mean()) * window
            fft_seg = np.fft.rfft(seg)
            psd_accum += np.abs(fft_seg) ** 2
        
        psd_accum *= scale / n_seg
        freqs = np.fft.rfftfreq(nperseg, 1 / fs)
        
        return freqs, psd_accum
    
    @property
    def name(self) -> str:
        return "PSD (Welch)"
    
    @property
    def params(self) -> dict:
        return {
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "use_scipy": self.use_scipy,
        }
