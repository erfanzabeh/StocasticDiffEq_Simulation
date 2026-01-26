"""
Abstract Base Classes
=====================

Core abstractions for generators and embedders.
All concrete implementations inherit from these.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Generator(ABC):
    """
    Abstract base class for all signal generators.
    
    Contract:
    - __call__(n_steps, dt) -> np.ndarray of shape (n_steps, dim)
    - All parameters stored on instance
    - Metadata (dt, time, params) accessible via properties after call
    - _core: staticmethod pointing to JIT-compiled implementation
    
    Example
    -------
    >>> gen = LorenzGenerator(sigma=10, rho=28)
    >>> data = gen(n_steps=10000, dt=0.005)  # (10000, 3)
    >>> t = gen.time  # (10000,)
    """
    
    # Subclasses should override with staticmethod(_jit_function)
    _core = None
    
    def __init__(self):
        self._dt: Optional[float] = None
        self._n_steps: Optional[int] = None
    
    @abstractmethod
    def __call__(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Generate signal.
        
        Parameters
        ----------
        n_steps : int
            Number of time steps to generate.
        dt : float
            Time step size (seconds).
        
        Returns
        -------
        np.ndarray
            Signal of shape (n_steps, dim). Always 2D.
        """
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output signal."""
        pass
    
    @property
    def dt(self) -> Optional[float]:
        """Time step used in last call. None if not yet called."""
        return self._dt
    
    @property
    def n_steps(self) -> Optional[int]:
        """Number of steps in last call. None if not yet called."""
        return self._n_steps
    
    @property
    def time(self) -> Optional[np.ndarray]:
        """Time array for last generated signal. None if not yet called."""
        if self._n_steps is None or self._dt is None:
            return None
        return np.arange(self._n_steps) * self._dt
    
    @property
    @abstractmethod
    def params(self) -> dict:
        """Dictionary of generator parameters."""
        pass


class Embedder(ABC):
    """
    Abstract base class for all embedders.
    
    Contract:
    - __call__(x) -> np.ndarray of shape (n_embedded, embedding_dim)
    - Input is 1D array (n_steps,)
    - Parameters stored on instance
    
    Example
    -------
    >>> emb = DelayEmbedder(m=3, tau=15)
    >>> data = gen(n_steps=10000, dt=0.005)[:, 0]  # Get scalar observable
    >>> embedded = emb(data)  # (n_embedded, 3)
    """
    
    # Subclasses should override with staticmethod(_jit_function)
    _core = None
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Embed 1D signal.
        
        Parameters
        ----------
        x : np.ndarray
            1D input signal of shape (n_steps,).
        
        Returns
        -------
        np.ndarray
            Embedded signal of shape (n_embedded, embedding_dim).
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedded space."""
        pass
    
    @property
    @abstractmethod
    def params(self) -> dict:
        """Dictionary of embedder parameters."""
        pass


class Tool(ABC):
    """
    Abstract base class for all analysis tools.
    
    Contract:
    - __call__(x, fs, **kwargs) -> result (type depends on tool)
    - Input is typically 1D array (n_steps,) and sampling frequency
    - Parameters stored on instance
    - Stateless: same input always produces same output
    
    Example
    -------
    >>> psd = PSDTool(method='welch', nperseg=256)
    >>> freqs, power = psd(signal, fs=500)
    """
    
    @abstractmethod
    def __call__(self, x: np.ndarray, fs: float, **kwargs):
        """
        Apply analysis tool to signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal (typically 1D).
        fs : float
            Sampling frequency in Hz.
        **kwargs
            Additional tool-specific parameters.
        
        Returns
        -------
        Result type depends on specific tool.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analysis tool."""
        pass
    
    @property
    @abstractmethod
    def params(self) -> dict:
        """Dictionary of tool parameters."""
        pass
