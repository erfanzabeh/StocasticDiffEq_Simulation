"""
Analysis Tools
==============

Spectral, correlation, and signal analysis tools.

All tools inherit from the Tool abstract base class.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Class
     - Description
     - Validated
   * - :class:`PSDTool`
     - Power spectral density (Welch method)
     - ✅
   * - :class:`ACFTool`
     - Autocorrelation function
     - ✅
   * - :class:`HilbertTool`
     - Hilbert transform (analytic signal)
     - ✅
   * - :class:`EnvelopeTool`
     - Amplitude envelope extraction
     - ✅
   * - :class:`IAFFTSurrogateTool`
     - IAAFT surrogate generation
     - ✅
   * - :class:`FourierSurrogateTool`
     - Fourier surrogate with random phases
     - ✅
   * - :class:`BicoherenceTool`
     - Bicoherence (quadratic phase coupling)
     - ✅
   * - :class:`SpectrogramTool`
     - Time-frequency spectrogram
     - ✅
   * - :class:`BandpassTool`
     - Butterworth bandpass filter
     - ✅
   * - :class:`LagMatrixTool`
     - Lag matrix construction for AR
     - ✅
   * - :class:`NotchTool`
     - Notch filter (line noise removal)
     - ✅
   * - :class:`DownsampleTool`
     - Anti-aliased downsampling
     - ✅
"""

from .psd import PSDTool
from .acf import ACFTool
from .hilbert import HilbertTool, EnvelopeTool
from .surrogate import IAFFTSurrogateTool, FourierSurrogateTool
from .bicoherence import BicoherenceTool
from .spectrogram import SpectrogramTool
from .bandpass import BandpassTool
from .lag_matrix import LagMatrixTool
from .notch import NotchTool
from .downsample import DownsampleTool

__all__ = [
    "PSDTool",
    "ACFTool",
    "HilbertTool",
    "EnvelopeTool",
    "IAFFTSurrogateTool",
    "FourierSurrogateTool",
    "BicoherenceTool",
    "SpectrogramTool",
    "BandpassTool",
    "LagMatrixTool",
    "NotchTool",
    "DownsampleTool",
]
