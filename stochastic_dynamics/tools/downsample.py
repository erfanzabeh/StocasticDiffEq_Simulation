"""
Downsampling Tool
=================

Signal downsampling with anti-aliasing.
"""

import numpy as np
from typing import Tuple
from ..abstract import Tool


class DownsampleTool(Tool):
    """
    Downsample signal with anti-aliasing filter.
    
    Parameters
    ----------
    target_fs : float
        Target sampling frequency in Hz.
    ftype : str
        Filter type for anti-aliasing ('iir' or 'fir').
    
    Example
    -------
    >>> ds = DownsampleTool(target_fs=250)
    >>> downsampled, new_fs = ds(signal, fs=1000)
    """
    
    def __init__(
        self,
        target_fs: float,
        ftype: str = 'iir'
    ):
        self.target_fs = target_fs
        self.ftype = ftype
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Downsample signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Original sampling frequency in Hz.
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (downsampled_signal, new_fs)
        """
        if fs <= self.target_fs:
            return np.asarray(x), fs
        
        from scipy.signal import decimate
        
        r = int(np.floor(fs / self.target_fs))
        x_down = decimate(x, r, ftype=self.ftype, zero_phase=True)
        new_fs = fs / r
        
        return x_down, new_fs
    
    @property
    def params(self) -> dict:
        return {
            "target_fs": self.target_fs,
            "ftype": self.ftype,
        }
