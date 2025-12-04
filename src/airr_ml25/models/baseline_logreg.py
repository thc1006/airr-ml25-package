"""Baseline L1‑regularized logistic regression for AIRR‑ML‑25.

This is not meant to be competitive on the leaderboard, but to provide a
transparent starting point that is easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..config import DatasetConfig
from ..data import load_repertoire
from ..features import aggregate_repertoire_features


@dataclass
class TrainedBaseline:
    """Container for a trained baseline model and feature metadata."""

    pipeline: Pipeline
    label_col: str
    repertoire_id_col: str = "repertoire_id"


def train_on_dataset(
    dataset: DatasetConfig,
    train_root: Path | str,
    cdr3_col: str = "cdr3_aa",
    label_col: str = "label",
    repertoire_id_col: str = "repertoire_id",
    random_state: int = 42,
) -> TrainedBaseline:
    """Train a baseline model on a single dataset.

    Parameters
    ----------
    dataset:
        Dataset configuration.
    train_root:
        Root directory containing the train_datasets/ folder.
    """
    rep = load_repertoire(dataset, train_root)
    if label_col not in rep.metadata.columns:
        raise KeyError(f"Label column {label_col!r} not found in metadata.csv")

    feats = aggregate_repertoire_features(
        rep.sequences,
        cdr3_col=cdr3_col,
        repertoire_id_col=repertoire_id_col,
    )
    merged = pd.merge(
        rep.metadata,
        feats,
        on=repertoire_id_col,
        how="inner",
        validate="one_to_one",
    )

    X = merged.drop(columns=[label_col, repertoire_id_col])
    y = merged[label_col].values

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=1.0,
                    max_iter=200,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    return TrainedBaseline(
        pipeline=pipeline,
        label_col=label_col,
        repertoire_id_col=repertoire_id_col,
    )


def predict_proba(
    model: TrainedBaseline,
    dataset: DatasetConfig,
    test_root: Path | str,
    cdr3_col: str = "cdr3_aa",
) -> pd.DataFrame:
    """Predict positive‑class probabilities for a dataset.

    Returns a DataFrame with columns:
    - repertoire_id
    - prob
    """
    rep = load_repertoire(dataset, test_root)
    feats = aggregate_repertoire_features(
        rep.sequences,
        cdr3_col=cdr3_col,
        repertoire_id_col=model.repertoire_id_col,
    )
    X = feats.drop(columns=[model.repertoire_id_col])
    prob = model.pipeline.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            model.repertoire_id_col: feats[model.repertoire_id_col].values,
            "prob": prob,
        }
    )
