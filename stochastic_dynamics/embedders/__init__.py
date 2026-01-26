"""
Embedders
=========

Delay embedding for state-space reconstruction (Takens' theorem).

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Class
     - Description
     - Validated
   * - :class:`DelayEmbedder`
     - Takens delay embedding
     - ✅
"""

from .delay_embedder import DelayEmbedder

__all__ = [
    "DelayEmbedder",
]
