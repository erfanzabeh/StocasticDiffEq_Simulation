"""
Autocorrelation Tool
====================

Compute autocorrelation function of a signal.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class ACFTool(Tool):
    """
    Autocorrelation function estimation.
    
    Parameters
    ----------
    max_lag : int, optional
        Maximum lag to compute. Default is n_samples // 4.
    normalize : bool
        If True, normalize ACF to range [-1, 1].
    
    Example
    -------
    >>> acf_tool = ACFTool(max_lag=100)
    >>> lags, acf = acf_tool(signal, fs=500)
    """
    
    def __init__(
        self,
        max_lag: Optional[int] = None,
        normalize: bool = True
    ):
        self.max_lag = max_lag
        self.normalize = normalize
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute autocorrelation function.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        lags : np.ndarray
            Lag values in seconds.
        acf : np.ndarray
            Autocorrelation values.
        """
        max_lag = kwargs.get('max_lag', self.max_lag)
        if max_lag is None:
            max_lag = len(x) // 4
        
        acf = self._compute_acf(x, max_lag, self.normalize)
        lags = np.arange(max_lag + 1) / fs
        
        return lags, acf
    
    @staticmethod
    def _compute_acf(
        x: np.ndarray,
        max_lag: int,
        normalize: bool
    ) -> np.ndarray:
        """Compute ACF using np.correlate."""
        x = x - x.mean()
        n = len(x)
        
        # Full autocorrelation
        acf_full = np.correlate(x, x, mode='full')
        acf = acf_full[n - 1 : n + max_lag]
        
        if normalize:
            acf = acf / acf[0]
        
        return acf
    
    @property
    def params(self) -> dict:
        return {
            "max_lag": self.max_lag,
            "normalize": self.normalize,
        }
