"""
Attention-based modules for AIRR-ML-25.

This package contains attention mechanisms and Multiple Instance Learning
implementations for immune repertoire classification.
"""

from .attention_mil import (
    KmerEncoder,
    SequenceEncoder,
    AttentionPooling,
    AttentionMIL,
    AIRRDataset,
)

__all__ = [
    'KmerEncoder',
    'SequenceEncoder',
    'AttentionPooling',
    'AttentionMIL',
    'AIRRDataset',
]
