"""
Stochastic Dynamics Package
============================

Tools for simulating stochastic processes, chaotic systems, 
and time-varying autoregressive models.

Submodules:
- generators: Signal generators (Lorenz, OU, TVAR, noise, etc.)
- embedders: Delay embedding and lag matrix construction
- tools: Spectral, correlation, and signal analysis tools
- models: Neural network and classical models for TVAR estimation
- processing: Domain-specific processing pipelines (Mouse/Monkey LFP)
"""

from .abstract import Generator, Embedder, Tool, Processing
from .generators import (
    LorenzGenerator,
    TVARGenerator,
    TVVARGenerator,
    OUGenerator,
    NoiseGenerator,
)
from .embedders import DelayEmbedder
from .tools import (
    PSDTool,
    ACFTool,
    HilbertTool,
    EnvelopeTool,
    IAFFTSurrogateTool,
    FourierSurrogateTool,
    BicoherenceTool,
    SpectrogramTool,
    BandpassTool,
    LagMatrixTool,
    NotchTool,
    DownsampleTool,
)
from .models import (
    ARModel,
    NeuralODE_TVAR,
    LagAttentionTVAR,
    LagAttentionTVARFast,
    TransformerAR,
    MLPTVAR,
    TVAROperator,
)
from .processing import (
    MouseLFPPipeline,
    MonkeyLFPPipeline,
)

__version__ = "0.1.0"
__all__ = [
    # Abstract base classes
    "Generator", 
    "Embedder",
    "Tool",
    "Processing",
    # Generators
    "LorenzGenerator",
    "TVARGenerator",
    "TVVARGenerator",
    "OUGenerator",
    "NoiseGenerator",
    # Embedders
    "DelayEmbedder",
    # Tools
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
    # Models
    "ARModel",
    "NeuralODE_TVAR",
    "LagAttentionTVAR",
    "LagAttentionTVARFast",
    "TransformerAR",
    "MLPTVAR",
    "TVAROperator",
    # Processing pipelines
    "MouseLFPPipeline",
    "MonkeyLFPPipeline",
]

