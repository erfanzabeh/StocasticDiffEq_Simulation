"""
Notch Filter Tool
=================

IIR notch filtering for removing line noise.
"""

import numpy as np
from typing import Tuple, Union
from ..abstract import Tool


class NotchTool(Tool):
    """
    IIR notch filter for removing line noise.
    
    Parameters
    ----------
    freqs : tuple or float
        Frequency or frequencies to notch out (Hz).
    Q : float
        Quality factor. Higher Q means narrower notch.
    
    Example
    -------
    >>> notch = NotchTool(freqs=(60.0, 120.0))  # Remove 60 Hz and harmonics
    >>> filtered = notch(signal, fs=1000)
    """
    
    def __init__(
        self,
        freqs: Union[float, Tuple[float, ...]] = (60.0, 120.0),
        Q: float = 30.0
    ):
        if isinstance(freqs, (int, float)):
            freqs = (float(freqs),)
        self.freqs = tuple(freqs)
        self.Q = Q
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Apply notch filter(s) to signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        from scipy.signal import iirnotch, filtfilt
        
        y = np.asarray(x).astype(float)
        nyquist = fs / 2
        
        for f0 in self.freqs:
            if f0 < nyquist:
                b, a = iirnotch(f0, self.Q, fs)
                y = filtfilt(b, a, y)
        
        return y
    
    @property
    def params(self) -> dict:
        return {
            "freqs": self.freqs,
            "Q": self.Q,
        }
