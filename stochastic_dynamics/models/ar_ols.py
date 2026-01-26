"""
AR(p) Model with OLS Estimation
===============================

Classic autoregressive model — the baseline for comparison with TVAR models.
"""

import numpy as np
from typing import Optional


class ARModel:
    """
    Classic AR(p) model with OLS estimation.
    
    A simple baseline for comparison with time-varying models.
    Uses numpy for fitting (no PyTorch needed).
    
    Parameters
    ----------
    p : int
        AR order (number of lags)
    
    Attributes
    ----------
    coeffs : ndarray, shape (p,)
        Estimated AR coefficients [a_1, ..., a_p]
    intercept : float
        Estimated intercept term
    residual_std : float
        Standard deviation of residuals
    
    Example
    -------
    >>> model = ARModel(p=3)
    >>> model.fit(x_train)
    >>> y_pred = model.predict(x_test)
    >>> forecast = model.forecast(x_test, horizon=10)
    """
    
    def __init__(self, p: int):
        self.p = p
        self.coeffs: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.residual_std: Optional[float] = None
    
    def fit(self, x: np.ndarray) -> "ARModel":
        """
        Fit AR(p) model using OLS.
        
        Parameters
        ----------
        x : ndarray, shape (T,)
            Time series to fit
        
        Returns
        -------
        self : ARModel
            Fitted model
        """
        T = len(x)
        p = self.p
        
        # Build design matrix X and target y
        X = np.column_stack([x[p-k:T-k] for k in range(1, p+1)])  # (T-p, p)
        y = x[p:]  # (T-p,)
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(y)), X])  # (T-p, p+1)
        
        # OLS: β = (X'X)^{-1} X'y
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        self.intercept = beta[0]
        self.coeffs = beta[1:]
        
        # Compute residuals
        y_hat = X_with_intercept @ beta
        residuals = y - y_hat
        self.residual_std = float(np.std(residuals))
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        One-step ahead prediction (teacher-forced).
        
        Parameters
        ----------
        x : ndarray, shape (T,)
            Input time series
        
        Returns
        -------
        y_hat : ndarray, shape (T-p,)
            Predictions for x[p], x[p+1], ...
        """
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        T = len(x)
        p = self.p
        
        X = np.column_stack([x[p-k:T-k] for k in range(1, p+1)])
        y_hat = self.intercept + X @ self.coeffs
        
        return y_hat
    
    def forecast(self, x: np.ndarray, horizon: int) -> np.ndarray:
        """
        Multi-step ahead forecast (autoregressive rollout).
        
        Parameters
        ----------
        x : ndarray, shape (T,)
            Input time series (uses last p values as initial condition)
        horizon : int
            Number of steps to forecast
        
        Returns
        -------
        forecast : ndarray, shape (horizon,)
            Forecasted values
        """
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        p = self.p
        buffer = list(x[-p:])  # Last p values
        forecast = []
        
        for _ in range(horizon):
            lags = np.array(buffer[-p:][::-1])  # [x_{t-1}, ..., x_{t-p}]
            y_next = self.intercept + np.dot(self.coeffs, lags)
            forecast.append(y_next)
            buffer.append(y_next)
        
        return np.array(forecast)
    
    def is_stable(self) -> bool:
        """
        Check if AR polynomial has all roots inside the unit circle.
        
        Returns
        -------
        stable : bool
            True if all roots are inside unit circle
        """
        if self.coeffs is None:
            raise ValueError("Model not fitted.")
        
        # AR polynomial: 1 - a1*z - a2*z^2 - ... - ap*z^p
        poly_coeffs = np.concatenate([[1], -self.coeffs])
        roots = np.roots(poly_coeffs)
        return bool(np.all(np.abs(roots) > 1))
    
    def __repr__(self):
        if self.coeffs is None:
            return f"ARModel(p={self.p}, fitted=False)"
        return f"ARModel(p={self.p}, coeffs={self.coeffs.round(4)}, intercept={self.intercept:.4f})"
