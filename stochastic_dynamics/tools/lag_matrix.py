"""
Lag Matrix Tool
===============

Construct design matrix for autoregressive (AR) modeling.

Unlike delay embedding (for state-space reconstruction), lag matrix 
returns (X, y) pairs suitable for regression/fitting.

Source: TimeVariying_AR_Simualtion.ipynb, MiceData.ipynb, MonkeyData.ipynb
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class LagMatrixTool(Tool):
    """
    Construct lag matrix for AR(p) regression.
    
    Given time series x, constructs:
    - X: Design matrix with lagged values [x(t-1), x(t-2), ..., x(t-p)]
    - y: Target vector x(t)
    
    Parameters
    ----------
    p : int
        AR order (number of lags).
    include_intercept : bool
        Whether to include a column of ones (default: True).
    
    Example
    -------
    >>> lag_tool = LagMatrixTool(p=10, include_intercept=True)
    >>> X, y = lag_tool(signal, fs=500)
    >>> coef = np.linalg.lstsq(X, y, rcond=None)[0]
    """
    
    def __init__(self, p: int = 10, include_intercept: bool = True):
        self.p = p
        self.include_intercept = include_intercept
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float = 1.0,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct lag matrix from signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency (not used, but required by Tool interface).
        
        Returns
        -------
        X : np.ndarray
            Design matrix of shape (N-p, p+1) or (N-p, p).
        y : np.ndarray
            Target vector of shape (N-p,).
        """
        p = kwargs.get('p', self.p)
        include_intercept = kwargs.get('include_intercept', self.include_intercept)
        
        N = len(x)
        n_eff = N - p
        
        if n_eff <= 0:
            raise ValueError(f"Signal too short ({N}) for AR order p={p}")
        
        y = x[p:].copy()
        
        if include_intercept:
            X = np.zeros((n_eff, p + 1))
            X[:, 0] = 1.0
            for i in range(p):
                X[:, i + 1] = x[p - 1 - i : N - 1 - i]
        else:
            X = np.zeros((n_eff, p))
            for i in range(p):
                X[:, i] = x[p - 1 - i : N - 1 - i]
        
        return X, y
    
    @property
    def params(self) -> dict:
        return {
            "p": self.p,
            "include_intercept": self.include_intercept,
        }
