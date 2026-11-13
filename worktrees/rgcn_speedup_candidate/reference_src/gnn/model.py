"""Backward-compatible imports; implementation lives in src.model."""
from ..model.layers import EdgeRGCNLayer, AttentionPool
from ..model.heads import SemanticGroupedHead
from ..model.network import EdgeCentricRGCN

__all__ = ["EdgeRGCNLayer", "AttentionPool", "SemanticGroupedHead", "EdgeCentricRGCN"]
