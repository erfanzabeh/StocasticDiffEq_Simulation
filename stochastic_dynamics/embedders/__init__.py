"""
Embedders
=========

Delay embedding for state-space reconstruction (Takens' theorem).

Available Embedders:
- DelayEmbedder: Takens delay embedding
"""

from .delay_embedder import DelayEmbedder

__all__ = [
    "DelayEmbedder",
]
