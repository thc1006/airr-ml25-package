"""Data loading utilities for AIRR‑ML‑25.

These helpers are intentionally lightweight and avoid any competition‑specific
logic that might change. The idea is that Claude (or you) can extend them
as you learn more about the exact file layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import DatasetConfig


SequenceLevel = Literal["amino_acid", "nucleotide"]


@dataclass
class RepertoireData:
    """Container for a single dataset's metadata and sequences.

    Attributes
    ----------
    metadata:
        DataFrame with repertoire‑level information, including labels for train.
    sequences:
        DataFrame with sequence‑level information. Column names are expected
        to follow the competition schema (e.g. `repertoire_id`, `sequence_id`,
        CDR3 sequence, V/J gene, counts, etc.), but we keep this generic.
    """

    metadata: pd.DataFrame
    sequences: pd.DataFrame


def load_metadata(dataset: DatasetConfig, root: Path | str) -> pd.DataFrame:
    """Load `metadata.csv` for a given dataset.

    The function is robust to missing files: if no metadata is found, it
    returns an empty DataFrame.
    """
    root = Path(root)
    meta_path = root / dataset.name / "metadata.csv"
    if not meta_path.exists():
        return pd.DataFrame()

    return pd.read_csv(meta_path)


def load_sequences(
    dataset: DatasetConfig,
    root: Path | str,
    filename: str = "sequences.csv",
) -> pd.DataFrame:
    """Load sequence‑level data for a given dataset.

    Parameters
    ----------
    dataset:
        Dataset configuration.
    root:
        Train or test root directory.
    filename:
        Name of the sequence file; update this if the competition uses a
        different convention (e.g. `airr.tsv`).
    """
    root = Path(root)
    seq_path = root / dataset.name / filename
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")

    if seq_path.suffix == ".csv":
        return pd.read_csv(seq_path)
    elif seq_path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(seq_path, sep="\t")
    else:
        raise ValueError(f"Unsupported sequence file extension: {seq_path.suffix}")


def load_repertoire(
    dataset: DatasetConfig,
    root: Path | str,
    sequence_filename: str = "sequences.csv",
) -> RepertoireData:
    """Convenience wrapper to load both metadata and sequences."""
    metadata = load_metadata(dataset, root)
    sequences = load_sequences(dataset, root, filename=sequence_filename)
    return RepertoireData(metadata=metadata, sequences=sequences)
