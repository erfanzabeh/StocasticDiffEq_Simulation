"""
Processing Pipelines
====================

Domain-specific processing pipelines for neural data.

All pipelines inherit from the Processing abstract base class
and compose Tools into complete workflows.

Available Pipelines:
- MouseLFPPipeline: Hippocampal LFP with ripple detection
- MonkeyLFPPipeline: Visual cortex (V1/V4) multi-channel processing
"""

from .mouse import MouseLFPPipeline
from .monkey import MonkeyLFPPipeline

__all__ = [
    "MouseLFPPipeline",
    "MonkeyLFPPipeline",
]
