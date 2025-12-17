"""Simple submission helpers and CLI entrypoint.

This module is intentionally conservative: it focuses on producing a table of
repertoire‑level probabilities. You should adapt it to the exact Kaggle
`sample_submissions.csv` schema once you have it available in your environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import DatasetConfig, default_train_datasets
from .models.baseline_logreg import TrainedBaseline, predict_proba, train_on_dataset


@dataclass
class SubmissionConfig:
    train_root: Path
    test_root: Path
    datasets: List[DatasetConfig]


def build_repertoire_probabilities(cfg: SubmissionConfig) -> pd.DataFrame:
    """Train a baseline per dataset and collect repertoire probabilities."""
    records: List[pd.DataFrame] = []
    for ds in cfg.datasets:
        print(f"[baseline] Training on dataset: {ds.name}")
        model = train_on_dataset(ds, cfg.train_root)
        print(f"[baseline] Predicting on dataset: {ds.name}")
        probs = predict_proba(model, ds, cfg.test_root)
        probs.insert(0, "dataset", ds.name)
        records.append(probs)

    return pd.concat(records, ignore_index=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AIRR‑ML‑25 baseline submission helper")

    parser.add_argument(
        "--train-root",
        type=str,
        required=True,
        help="Path to train_datasets root directory",
    )
    parser.add_argument(
        "--test-root",
        type=str,
        required=True,
        help="Path to test_datasets root directory",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Where to write the baseline probabilities CSV",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    datasets = default_train_datasets()
    cfg = SubmissionConfig(
        train_root=train_root,
        test_root=test_root,
        datasets=datasets,
    )

    df = build_repertoire_probabilities(cfg)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[baseline] Wrote probabilities to {out_path}")
    print("NOTE: You must still align this table to Kaggle's sample_submissions.csv schema.")


if __name__ == "__main__":  # pragma: no cover
    main()
