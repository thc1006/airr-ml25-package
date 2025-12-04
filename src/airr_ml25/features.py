"""Feature engineering utilities for AIRR‑ML‑25.

This module intentionally keeps features simple so that you can quickly get a
baseline running and then iteratively extend it with Claude.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _kmer_counts(
    sequences: Iterable[str],
    k: int = 3,
) -> Counter:
    """Compute k‑mer counts from an iterable of sequences."""
    counts: Counter = Counter()
    for seq in sequences:
        if not isinstance(seq, str):
            continue
        s = seq.strip()
        if len(s) < k:
            continue
        for i in range(len(s) - k + 1):
            kmer = s[i : i + k]
            counts[kmer] += 1
    return counts


def aggregate_repertoire_features(
    sequences: pd.DataFrame,
    cdr3_col: str = "cdr3_aa",
    repertoire_id_col: str = "repertoire_id",
    count_col: str | None = None,
    k: int = 3,
) -> pd.DataFrame:
    """Create simple repertoire‑level features.

    The idea is to collapse all sequences belonging to the same repertoire into
    a single feature vector.

    Currently we compute:
    - sequence count
    - total counts (if a count column is available)
    - mean / std / max of sequence length
    - a small set of top‑K k‑mer frequencies (global top‑K across all repertoires)
    """
    df = sequences.copy()

    if count_col is None or count_col not in df.columns:
        df["_count"] = 1.0
        count_col = "_count"

    df["_len"] = df[cdr3_col].astype(str).str.len()

    # Basic aggregate stats per repertoire
    if "sequence_id" in df.columns:
        count_col_name = "sequence_id"
    else:
        count_col_name = cdr3_col

    agg = df.groupby(repertoire_id_col).agg(
        seq_count=(count_col_name, "count"),
        total_count=(count_col, "sum"),
        len_mean=("_len", "mean"),
        len_std=("_len", "std"),
        len_max=("_len", "max"),
    )

    # Global top‑k k‑mers
    global_counts = _kmer_counts(df[cdr3_col].dropna().astype(str).tolist(), k=k)
    top_kmers = [k for k, _ in global_counts.most_common(64)]

    for kmer in top_kmers:
        mask = df[cdr3_col].astype(str).str.contains(kmer, na=False)
        agg[f"kmer_{kmer}"] = (
            df.loc[mask].groupby(repertoire_id_col)[count_col].sum().reindex(agg.index, fill_value=0.0)
        )

    return agg.reset_index()
