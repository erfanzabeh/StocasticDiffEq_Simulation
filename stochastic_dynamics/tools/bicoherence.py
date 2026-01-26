"""
Bicoherence Tool
================

Higher-order spectral analysis for detecting quadratic phase coupling.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class BicoherenceTool(Tool):
    """
    Compute bicoherence for detecting quadratic phase coupling.
    
    Bicoherence values near 1 indicate strong phase-locking between 
    frequency components f1, f2, and f1+f2.
    
    Parameters
    ----------
    nperseg : int
        Segment length for FFT.
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 2.
    fmax : float, optional
        Maximum frequency to compute. Default is fs / 4.
    
    Example
    -------
    >>> bic = BicoherenceTool(nperseg=512, fmax=50)
    >>> freqs, bic_matrix = bic(signal, fs=500)
    """
    
    def __init__(
        self,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        fmax: Optional[float] = None
    ):
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.fmax = fmax
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute bicoherence map.
        
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
        bic : np.ndarray
            Bicoherence matrix (f1 x f2).
        """
        nperseg = kwargs.get('nperseg', self.nperseg)
        noverlap = kwargs.get('noverlap', self.noverlap)
        fmax = kwargs.get('fmax', self.fmax)
        
        if fmax is None:
            fmax = fs / 4
        
        # Segment signal
        step = nperseg - noverlap
        n_seg = (len(x) - nperseg) // step + 1
        
        # Frequency resolution
        freqs = np.fft.rfftfreq(nperseg, 1 / fs)
        mask = freqs <= fmax
        freqs_out = freqs[mask]
        nf = len(freqs_out)
        
        # Accumulators
        bispectrum = np.zeros((nf, nf), dtype=complex)
        norm1 = np.zeros((nf, nf))
        norm2 = np.zeros((nf, nf))
        
        window = np.hanning(nperseg)
        
        for i in range(n_seg):
            seg = x[i * step : i * step + nperseg]
            seg = (seg - seg.mean()) * window
            
            X = np.fft.rfft(seg)
            X_trunc = X[:nf]
            
            for i1 in range(nf):
                for i2 in range(i1, nf):
                    i3 = i1 + i2
                    if i3 < len(X):
                        prod = X_trunc[i1] * X_trunc[i2] * np.conj(X[i3])
                        bispectrum[i1, i2] += prod
                        norm1[i1, i2] += np.abs(X_trunc[i1] * X_trunc[i2]) ** 2
                        norm2[i1, i2] += np.abs(X[i3]) ** 2
        
        # Compute bicoherence
        with np.errstate(divide='ignore', invalid='ignore'):
            bic = np.abs(bispectrum) ** 2 / (norm1 * norm2 + 1e-12)
        
        # Mirror to lower triangle
        bic = bic + bic.T - np.diag(np.diag(bic))
        
        return freqs_out, bic
    
    @property
    def params(self) -> dict:
        return {
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "fmax": self.fmax,
        }
