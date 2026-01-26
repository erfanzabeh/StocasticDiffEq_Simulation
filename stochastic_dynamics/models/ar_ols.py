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
    >>> y_pred = model.predict(x_train)
    >>> print(model.aic(x_train[p:], y_pred))
    >>> print(model.bic(x_train[p:], y_pred))
    """
    
    def __init__(self, p: int):
        self.p = p
        self.coeffs: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.residual_std: Optional[float] = None
        self._n_samples: Optional[int] = None  # sample size used for fitting
    
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
        self._n_samples = len(y)
        
        return self
    
    def aic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Akaike Information Criterion.
        
        Parameters
        ----------
        y_true : ndarray
            True values
        y_pred : ndarray
            Predicted values
        
        Returns
        -------
        aic : float
            AIC value (lower is better)
        """
        n = len(y_true)
        k = self.p + 1  # p coefficients + intercept
        rss = np.sum((y_true - y_pred)**2)
        sigma2 = rss / max(n, 1)
        return n * np.log(sigma2 + 1e-12) + 2 * k
    
    def bic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion.
        
        Parameters
        ----------
        y_true : ndarray
            True values
        y_pred : ndarray
            Predicted values
        
        Returns
        -------
        bic : float
            BIC value (lower is better)
        """
        n = len(y_true)
        k = self.p + 1  # p coefficients + intercept
        rss = np.sum((y_true - y_pred)**2)
        sigma2 = rss / max(n, 1)
        return n * np.log(sigma2 + 1e-12) + k * np.log(max(n, 2))
    
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
    
    def hybrid_predict(
        self, 
        x: np.ndarray, 
        start_idx: int, 
        n_steps: int, 
        refresh_every: int = 1
    ) -> np.ndarray:
        """
        Controllable k-step prediction with periodic ground-truth refresh.
        
        This is a hybrid between teacher forcing (refresh_every=1) and 
        pure free-running forecast (refresh_every >= n_steps). Useful for
        evaluating how prediction error accumulates over multiple steps.
        
        Parameters
        ----------
        x : ndarray, shape (T,)
            Full time series (ground truth)
        start_idx : int
            Index in x where prediction begins (must be >= p)
        n_steps : int
            Number of steps to predict
        refresh_every : int, default=1
            How often to reset history buffer to ground truth.
            - 1: classic one-step (teacher forcing every step)
            - k: run k-1 steps open-loop, then refresh
            - >= n_steps: pure free-running (no refresh)
        
        Returns
        -------
        predictions : ndarray, shape (n_steps,)
            Predicted values starting at index start_idx
        
        Example
        -------
        >>> model = ARModel(p=5).fit(x_train)
        >>> # 5-step lookahead with refresh every 5 steps
        >>> y_pred = model.hybrid_predict(x_full, start_idx=1000, n_steps=500, refresh_every=5)
        >>> # Compare: pure 1-step (same as predict on x[1000:1500])
        >>> y_1step = model.hybrid_predict(x_full, start_idx=1000, n_steps=500, refresh_every=1)
        """
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        p = self.p
        if start_idx < p:
            raise ValueError(f"start_idx must be >= p={p}")
        if start_idx + n_steps > len(x):
            raise ValueError(f"Not enough data: need {start_idx + n_steps} samples, have {len(x)}")
        
        refresh_every = max(1, int(refresh_every))
        predictions = []
        
        # Initialize history buffer with true values
        history = list(x[start_idx - p : start_idx])
        
        for t in range(n_steps):
            # Refresh history with ground truth periodically
            if (t % refresh_every) == 0:
                abs_idx = start_idx + t
                history = list(x[abs_idx - p : abs_idx])
            
            # Predict next value: coeffs are [a_1, ..., a_p] for lags [x_{t-1}, ..., x_{t-p}]
            lags = np.array(history[::-1])  # reverse to match [x_{t-1}, ..., x_{t-p}]
            y_hat = self.intercept + np.dot(self.coeffs, lags)
            predictions.append(y_hat)
            
            # Update history buffer (slide window)
            history.pop(0)
            history.append(y_hat)
        
        return np.array(predictions)
    
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
