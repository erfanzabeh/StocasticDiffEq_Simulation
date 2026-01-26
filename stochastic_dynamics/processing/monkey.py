"""
Monkey LFP Processing Pipeline
===============================

Processing pipeline for monkey visual cortex (V1/V4) LFP recordings
from Utah arrays.

Typical sampling rates: 500-1000 Hz
"""

import numpy as np
from typing import Optional, Tuple
from ..abstract import Processing
from ..tools import BandpassTool, PSDTool, NotchTool, DownsampleTool


class MonkeyLFPPipeline(Processing):
    """
    Processing pipeline for monkey visual cortex LFP.
    
    Workflow:
    1. Notch filter for line noise (60/120 Hz)
    2. Bandpass filter (0.5-200 Hz)
    3. Optional downsampling
    4. Channel grouping by brain region (V1 vs V4)
    5. PSD computation and comparison
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    notch_freqs : tuple
        Frequencies to notch out (Hz).
    bandpass : tuple
        (low, high) frequency bounds for bandpass filter.
    target_fs : float, optional
        Target sampling rate for downsampling. None to skip.
    
    Example
    -------
    >>> pipeline = MonkeyLFPPipeline(fs=500)
    >>> processed = pipeline(raw_lfp)  # Single channel
    >>> processed_multi = pipeline.preprocess_multichannel(lfp_array)
    """
    
    def __init__(
        self,
        fs: float,
        notch_freqs: Tuple[float, ...] = (60.0, 120.0),
        bandpass: Tuple[float, float] = (0.5, 200.0),
        target_fs: Optional[float] = None,
        filter_order: int = 4,
        notch_Q: float = 30.0
    ):
        super().__init__(fs)
        self.notch_freqs = notch_freqs
        self.bandpass = bandpass
        self.target_fs = target_fs
        self.filter_order = filter_order
        self.notch_Q = notch_Q
        
        # Compose tools (all filtering logic lives in tools/)
        self._notch_tool = NotchTool(freqs=notch_freqs, Q=notch_Q)
        self._bandpass_tool = BandpassTool(
            lowcut=bandpass[0],
            highcut=bandpass[1],
            order=filter_order
        )
        self._downsample_tool = DownsampleTool(target_fs=target_fs) if target_fs else None
        self._psd_tool = PSDTool(nperseg=4096)
        
        # State
        self._raw: Optional[np.ndarray] = None
        self._processed: Optional[np.ndarray] = None
        self._current_fs: float = fs
        self._time: Optional[np.ndarray] = None
    
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run full preprocessing pipeline on single channel.
        
        Parameters
        ----------
        x : np.ndarray
            Raw LFP signal (1D).
        
        Returns
        -------
        np.ndarray
            Preprocessed signal.
        """
        self._raw = np.asarray(x).ravel()
        
        # Step 1: Notch filter (using NotchTool)
        y = self._notch_tool(self._raw, self.fs)
        self._intermediate['notch_filtered'] = y
        
        # Step 2: Bandpass filter (using BandpassTool)
        y = self._bandpass_tool(y, self.fs)
        self._intermediate['bandpass_filtered'] = y
        
        # Step 3: Downsample (using DownsampleTool)
        if self._downsample_tool is not None:
            y, self._current_fs = self._downsample_tool(y, self.fs)
        else:
            self._current_fs = self.fs
        self._intermediate['downsampled'] = y
        
        self._processed = y
        self._preprocessed = y
        self._time = np.arange(len(y)) / self._current_fs
        
        return y
    
    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Alias for __call__ for interface compliance."""
        return self.__call__(x)
    
    def preprocess_multichannel(
        self,
        lfp_array: np.ndarray,
        channel_regions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Preprocess multi-channel LFP array.
        
        Parameters
        ----------
        lfp_array : np.ndarray
            LFP array of shape (n_samples, n_channels).
        channel_regions : np.ndarray, optional
            Array of region labels ('V1', 'V4', etc.) for each channel.
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (processed_array, new_fs)
        """
        n_samples, n_channels = lfp_array.shape
        
        # Process first channel to get output size
        y0 = self(lfp_array[:, 0])
        out = np.zeros((len(y0), n_channels))
        out[:, 0] = y0
        
        # Process remaining channels
        for ch in range(1, n_channels):
            # Reset state for each channel
            out[:, ch] = self(lfp_array[:, ch])
        
        # Store channel regions if provided
        if channel_regions is not None:
            self._intermediate['channel_regions'] = channel_regions
            self._intermediate['v1_channels'] = np.where(channel_regions == 'V1')[0]
            self._intermediate['v4_channels'] = np.where(channel_regions == 'V4')[0]
        
        self._intermediate['multichannel_processed'] = out
        
        return out, self._current_fs
    
    def compute_psd(
        self,
        x: Optional[np.ndarray] = None,
        fmin: float = 0.5,
        fmax: float = 200.0,
        smooth_sigma: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density using the PSDTool.
        
        Parameters
        ----------
        x : np.ndarray, optional
            Signal to analyze. If None, uses last processed signal.
        fmin : float
            Minimum frequency to return.
        fmax : float
            Maximum frequency to return.
        smooth_sigma : float
            Gaussian smoothing sigma along frequency axis.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (frequencies, psd) arrays.
        """
        from scipy.ndimage import gaussian_filter1d
        
        if x is None:
            if self._processed is None:
                raise ValueError("Must run pipeline first or provide signal")
            x = self._processed
            fs = self._current_fs
        else:
            fs = self.fs
        
        # Use the PSDTool
        freqs, psd = self._psd_tool(x, fs, nperseg=min(4096, len(x)//4))
        
        # Trim to frequency range
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mask]
        psd = psd[mask]
        
        # Smooth
        if smooth_sigma > 0:
            psd = gaussian_filter1d(psd, sigma=smooth_sigma)
        
        return freqs, psd
    
    def compute_group_psd(
        self,
        lfp_array: np.ndarray,
        channel_indices: np.ndarray,
        **psd_kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute average PSD for a group of channels.
        
        Parameters
        ----------
        lfp_array : np.ndarray
            Processed LFP array of shape (n_samples, n_channels).
        channel_indices : np.ndarray
            Indices of channels to include in group.
        **psd_kwargs
            Arguments passed to compute_psd.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (frequencies, individual_psds, average_psd)
        """
        psds = []
        freqs = None
        
        for ch in channel_indices:
            f, p = self.compute_psd(lfp_array[:, ch], **psd_kwargs)
            if freqs is None:
                freqs = f
            psds.append(p)
        
        psds = np.array(psds)
        avg_psd = np.mean(psds, axis=0)
        
        return freqs, psds, avg_psd
    
    def prepare_for_tvar(
        self,
        x: Optional[np.ndarray] = None,
        band: Tuple[float, float] = (1.0, 50.0),
        tvar_target_fs: float = 250.0,
        zscore: bool = True
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Prepare signal for TVAR modeling.
        
        Parameters
        ----------
        x : np.ndarray, optional
            Signal to prepare. If None, uses last processed signal.
        band : tuple
            (low, high) bandpass filter for TVAR.
        tvar_target_fs : float
            Target sampling rate for TVAR.
        zscore : bool
            Whether to z-score normalize.
        
        Returns
        -------
        Tuple[np.ndarray, float, np.ndarray]
            (prepared_signal, new_fs, time_vector)
        """
        if x is None:
            if self._processed is None:
                raise ValueError("Must run pipeline first or provide signal")
            x = self._processed
            fs = self._current_fs
        else:
            fs = self.fs
        
        # Use BandpassTool for target band
        tvar_bandpass = BandpassTool(lowcut=band[0], highcut=band[1], order=4)
        x_filt = tvar_bandpass(x, fs)
        
        # Use DownsampleTool for TVAR
        new_fs = fs
        if fs > tvar_target_fs:
            tvar_downsample = DownsampleTool(target_fs=tvar_target_fs)
            x_filt, new_fs = tvar_downsample(x_filt, fs)
        
        # Z-score (simple operation, not worth a separate tool)
        if zscore:
            x_filt = (x_filt - np.mean(x_filt)) / (np.std(x_filt) + 1e-12)
        
        time = np.arange(len(x_filt)) / new_fs
        
        return x_filt.astype(float), new_fs, time
    
    # Properties
    @property
    def raw_(self) -> Optional[np.ndarray]:
        """Raw input signal from last run."""
        return self._raw
    
    @property
    def processed_(self) -> Optional[np.ndarray]:
        """Processed signal from last run."""
        return self._processed
    
    @property
    def current_fs_(self) -> float:
        """Current sampling frequency (after downsampling)."""
        return self._current_fs
    
    @property
    def time_(self) -> Optional[np.ndarray]:
        """Time vector from last run."""
        return self._time
    
    @property
    def params(self) -> dict:
        return {
            "fs": self.fs,
            "notch_freqs": self.notch_freqs,
            "bandpass": self.bandpass,
            "target_fs": self.target_fs,
            "filter_order": self.filter_order,
            "notch_Q": self.notch_Q,
        }
