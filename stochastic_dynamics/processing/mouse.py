"""
Mouse LFP Processing Pipeline
==============================

Processing pipeline for mouse hippocampal LFP recordings,
with focus on sharp-wave ripple (SWR) detection.

Typical sampling rates: 20-30 kHz (Neuropixels, high-density probes)
"""

import numpy as np
from typing import Optional, List, Tuple
from ..abstract import Processing
from ..tools import BandpassTool, EnvelopeTool, SpectrogramTool


class MouseLFPPipeline(Processing):
    """
    Processing pipeline for mouse hippocampal LFP.
    
    Workflow:
    1. Bandpass filter for ripple band (80-140 Hz default)
    2. Compute Hilbert envelope
    3. Threshold-based ripple detection
    4. Optional spectrogram computation
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    ripple_band : tuple
        (low, high) frequency bounds for ripple band in Hz.
    
    Example
    -------
    >>> pipeline = MouseLFPPipeline(fs=20000, ripple_band=(80, 140))
    >>> processed = pipeline(raw_lfp)
    >>> ripples = pipeline.detect_ripples(threshold_sd=3.0)
    >>> print(f"Found {len(ripples)} ripple events")
    """
    
    def __init__(
        self,
        fs: float,
        ripple_band: Tuple[float, float] = (80.0, 140.0),
        filter_order: int = 4
    ):
        super().__init__(fs)
        self.ripple_band = ripple_band
        self.filter_order = filter_order
        
        # Compose tools
        self._bandpass = BandpassTool(
            lowcut=ripple_band[0],
            highcut=ripple_band[1],
            order=filter_order
        )
        self._envelope_tool = EnvelopeTool()
        self._spectrogram_tool = SpectrogramTool()
        
        # State
        self._raw: Optional[np.ndarray] = None
        self._ripple_band_lfp: Optional[np.ndarray] = None
        self._envelope: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run full preprocessing pipeline.
        
        Parameters
        ----------
        x : np.ndarray
            Raw LFP signal (1D).
        
        Returns
        -------
        np.ndarray
            Ripple-band filtered signal.
        """
        self._raw = np.asarray(x).ravel()
        self._time = np.arange(len(self._raw)) / self.fs
        
        # Step 1: Bandpass filter for ripple band
        self._ripple_band_lfp = self._bandpass(self._raw, self.fs)
        self._intermediate['ripple_band_lfp'] = self._ripple_band_lfp
        
        # Step 2: Compute envelope
        self._envelope = self._envelope_tool(self._ripple_band_lfp, self.fs)
        self._intermediate['envelope'] = self._envelope
        
        self._preprocessed = self._ripple_band_lfp
        return self._ripple_band_lfp
    
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Alias for __call__ for interface compliance."""
        return self.__call__(x)
    
    def detect_ripples(
        self,
        threshold_sd: float = 3.0,
        min_duration_ms: float = 15.0,
        max_duration_ms: float = 250.0
    ) -> List[Tuple[float, float]]:
        """
        Detect ripple events based on envelope threshold crossing.
        
        Parameters
        ----------
        threshold_sd : float
            Number of standard deviations above mean for detection.
        min_duration_ms : float
            Minimum ripple duration in milliseconds.
        max_duration_ms : float
            Maximum ripple duration in milliseconds.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (start_time, end_time) tuples for each ripple.
        
        Raises
        ------
        ValueError
            If pipeline has not been run yet.
        """
        if self._envelope is None:
            raise ValueError("Must run pipeline first: pipeline(raw_lfp)")
        
        threshold = np.mean(self._envelope) + threshold_sd * np.std(self._envelope)
        min_samples = int(min_duration_ms * self.fs / 1000)
        max_samples = int(max_duration_ms * self.fs / 1000)
        
        # Find threshold crossings
        above_threshold = self._envelope > threshold
        
        # Detect start/end of events
        ripple_intervals = []
        in_ripple = False
        start_idx = 0
        
        for i, above in enumerate(above_threshold):
            if above and not in_ripple:
                start_idx = i
                in_ripple = True
            elif not above and in_ripple:
                duration = i - start_idx
                if min_samples <= duration <= max_samples:
                    start_time = self._time[start_idx]
                    end_time = self._time[i]
                    ripple_intervals.append((start_time, end_time))
                in_ripple = False
        
        # Store results
        self._intermediate['ripple_intervals'] = ripple_intervals
        self._intermediate['ripple_threshold'] = threshold
        
        return ripple_intervals
    
    def compute_spectrogram(
        self,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        smooth_sigma: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram of raw LFP using SpectrogramTool.
        
        Parameters
        ----------
        nperseg : int
            Segment length for STFT.
        noverlap : int, optional
            Overlap between segments. Default is nperseg - nperseg//8.
        smooth_sigma : float
            Gaussian smoothing sigma.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (frequencies, times, Sxx) spectrogram arrays.
        """
        if self._raw is None:
            raise ValueError("Must run pipeline first: pipeline(raw_lfp)")
        
        if noverlap is None:
            noverlap = nperseg - nperseg // 8
        
        # Use SpectrogramTool
        f, t, Sxx = self._spectrogram_tool(
            self._raw, self.fs,
            nperseg=nperseg, noverlap=noverlap, smooth_sigma=smooth_sigma
        )
        
        self._intermediate['spectrogram'] = (f, t, Sxx)
        
        return f, t, Sxx
    
    # Properties for accessing intermediate results
    @property
    def raw_(self) -> Optional[np.ndarray]:
        """Raw input signal from last run."""
        return self._raw
    
    @property
    def ripple_band_lfp_(self) -> Optional[np.ndarray]:
        """Ripple-band filtered signal from last run."""
        return self._ripple_band_lfp
    
    @property
    def envelope_(self) -> Optional[np.ndarray]:
        """Hilbert envelope from last run."""
        return self._envelope
    
    @property
    def time_(self) -> Optional[np.ndarray]:
        """Time vector from last run."""
        return self._time
    
    @property
    def name(self) -> str:
        return "Mouse LFP Pipeline"
    
    @property
    def params(self) -> dict:
        return {
            "fs": self.fs,
            "ripple_band": self.ripple_band,
            "filter_order": self.filter_order,
        }
