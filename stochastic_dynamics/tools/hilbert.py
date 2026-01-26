"""
Hilbert Transform & Envelope Tools
===================================

Analytic signal, envelope, and instantaneous phase extraction.
"""

import numpy as np
from typing import Optional
from ..abstract import Tool


class HilbertTool(Tool):
    """
    Compute analytic signal via Hilbert transform.
    
    Parameters
    ----------
    use_scipy : bool
        If True, use scipy.signal.hilbert. If False, use custom FFT-based.
    
    Example
    -------
    >>> hilbert = HilbertTool()
    >>> analytic = hilbert(signal, fs=500)
    >>> envelope = np.abs(analytic)
    >>> phase = np.angle(analytic)
    """
    
    def __init__(self, use_scipy: bool = True):
        self.use_scipy = use_scipy
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute analytic signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D, real-valued).
        fs : float
            Sampling frequency in Hz (not used, but required by interface).
        
        Returns
        -------
        np.ndarray
            Complex analytic signal.
        """
        if self.use_scipy:
            from scipy.signal import hilbert
            return hilbert(x)
        else:
            return self._hilbert_custom(x)
    
    @staticmethod
    def _hilbert_custom(x: np.ndarray) -> np.ndarray:
        """Custom Hilbert transform via FFT."""
        n = len(x)
        X = np.fft.fft(x)
        
        # Create frequency-domain filter
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        
        return np.fft.ifft(X * h)
    
    @property
    def params(self) -> dict:
        return {"use_scipy": self.use_scipy}


class EnvelopeTool(Tool):
    """
    Extract amplitude envelope from signal.
    
    Parameters
    ----------
    use_scipy : bool
        If True, use scipy for Hilbert transform.
    
    Example
    -------
    >>> env_tool = EnvelopeTool()
    >>> envelope = env_tool(signal, fs=500)
    """
    
    def __init__(self, use_scipy: bool = True):
        self._hilbert = HilbertTool(use_scipy=use_scipy)
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute envelope (instantaneous amplitude).
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        np.ndarray
            Envelope of the signal.
        """
        analytic = self._hilbert(x, fs)
        return np.abs(analytic)
    
    def instantaneous_phase(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Compute instantaneous phase (unwrapped)."""
        analytic = self._hilbert(x, fs)
        return np.unwrap(np.angle(analytic))
    
    def instantaneous_frequency(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Compute instantaneous frequency in Hz."""
        phase = self.instantaneous_phase(x, fs)
        return np.gradient(phase) * fs / (2 * np.pi)
    
    @property
    def params(self) -> dict:
        return {"use_scipy": self._hilbert.use_scipy}


class EnvelopeNormalizeTool(Tool):
    """
    Envelope normalization: bandpass, extract envelope, divide out amplitude.
    
    This normalizes a signal by dividing out the slowly-varying envelope,
    producing a "flattened" signal with constant amplitude modulation.
    Useful for removing amplitude variations while preserving phase structure.
    
    Parameters
    ----------
    fband : tuple
        Bandpass frequency range (low, high) in Hz.
    env_lp_hz : float or None
        Lowpass cutoff for envelope smoothing. If None, no smoothing.
    bp_order : int
        Bandpass filter order.
    lp_order : int
        Lowpass filter order for envelope smoothing.
    eps : float
        Small constant to avoid division by zero.
    keep_scale : bool
        If True, rescale output by median envelope to preserve typical amplitude.
    
    Example
    -------
    >>> norm_tool = EnvelopeNormalizeTool(fband=(1, 80), env_lp_hz=3.0)
    >>> x_flat, x_bp, envelope = norm_tool(signal, fs=500)
    """
    
    def __init__(
        self,
        fband: tuple = (1.0, 50.0),
        env_lp_hz: float = 5.0,
        bp_order: int = 4,
        lp_order: int = 2,
        eps: float = 1e-6,
        keep_scale: bool = True
    ):
        self.fband = fband
        self.env_lp_hz = env_lp_hz
        self.bp_order = bp_order
        self.lp_order = lp_order
        self.eps = eps
        self.keep_scale = keep_scale
        self._hilbert = HilbertTool(use_scipy=True)
    
    def __call__(
        self,
        x: np.ndarray,
        fs: float,
        **kwargs
    ) -> tuple:
        """
        Apply envelope normalization.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (1D).
        fs : float
            Sampling frequency in Hz.
        
        Returns
        -------
        tuple
            (x_normalized, x_bandpassed, envelope)
            - x_normalized: envelope-normalized signal
            - x_bandpassed: bandpassed signal before normalization
            - envelope: estimated envelope (possibly smoothed)
        """
        from scipy.signal import butter, filtfilt
        
        fband = kwargs.get('fband', self.fband)
        env_lp_hz = kwargs.get('env_lp_hz', self.env_lp_hz)
        
        # 1) Bandpass filter
        nyq = fs / 2.0
        b_bp, a_bp = butter(self.bp_order, np.array(fband) / nyq, btype='bandpass')
        x_bp = filtfilt(b_bp, a_bp, x)
        
        # 2) Compute envelope via Hilbert transform
        analytic = self._hilbert(x_bp, fs)
        envelope = np.abs(analytic)
        
        # 3) Optionally smooth envelope with lowpass filter
        if env_lp_hz is not None and env_lp_hz > 0:
            b_lp, a_lp = butter(self.lp_order, env_lp_hz / nyq, btype='low')
            envelope = filtfilt(b_lp, a_lp, envelope)
        
        # 4) Divide out envelope
        x_flat = x_bp / (envelope + self.eps)
        
        # 5) Optionally rescale to preserve typical amplitude
        if self.keep_scale:
            x_flat *= np.median(envelope)
        
        return x_flat, x_bp, envelope
    
    @property
    def params(self) -> dict:
        return {
            "fband": self.fband,
            "env_lp_hz": self.env_lp_hz,
            "bp_order": self.bp_order,
            "lp_order": self.lp_order,
            "eps": self.eps,
            "keep_scale": self.keep_scale,
        }
