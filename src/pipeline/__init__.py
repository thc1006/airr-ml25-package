"""
Pipeline modules for AIRR-ML-25.

This package contains end-to-end training and prediction pipelines.
"""

from .champion_pipeline import (
    ImmuneStatePredictor,
    PipelineConfig,
    StackedEnsembleClassifier,
)

__all__ = [
    'ImmuneStatePredictor',
    'PipelineConfig',
    'StackedEnsembleClassifier',
]
