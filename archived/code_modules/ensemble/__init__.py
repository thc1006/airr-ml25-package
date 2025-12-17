"""
Ensemble learning modules for AIRR-ML-25.

This package contains stacking, blending, and other ensemble
methods for combining multiple models.
"""

from .stacking import (
    StackingClassifier,
    HillClimbingEnsemble,
    get_default_base_learners,
    quick_stack,
    check_gpu_availability,
)

__all__ = [
    'StackingClassifier',
    'HillClimbingEnsemble',
    'get_default_base_learners',
    'quick_stack',
    'check_gpu_availability',
]
