"""
Processing Pipelines
====================

Domain-specific processing pipelines for neural data.

All pipelines inherit from the Processing abstract base class
and compose Tools into complete workflows.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Class
     - Description
     - Validated
   * - :class:`MouseLFPPipeline`
     - Hippocampal LFP with ripple detection
     - ✅
   * - :class:`MonkeyLFPPipeline`
     - Visual cortex (V1/V4) multi-channel processing
     - ✅
"""

from .mouse import MouseLFPPipeline
from .monkey import MonkeyLFPPipeline

__all__ = [
    "MouseLFPPipeline",
    "MonkeyLFPPipeline",
]
