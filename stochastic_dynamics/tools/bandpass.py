"""
Bandpass Filter Tool
====================

Butterworth bandpass filtering.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class BandpassTool(Tool):
    """
    Butterworth bandpass filter.
    
    Parameters
    ----------
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    order : int
        Filter order.
    
    Example
    -------
    >>> bp = BandpassTool(lowcut=8, highcut=12)  # Alpha band
    >>> filtered = bp(signal, fs=500)
    """
    
    def __init__(
        self,
        lowcut: float,
        highcut: float,
        order: int = 4
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Apply bandpass filter.
        
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
        from scipy.signal import butter, filtfilt
        
        lowcut = kwargs.get('lowcut', self.lowcut)
        highcut = kwargs.get('highcut', self.highcut)
        order = kwargs.get('order', self.order)
        
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, x)
    
    @property
    def params(self) -> dict:
        return {
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
        }
