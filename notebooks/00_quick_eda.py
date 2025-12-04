"""Quick EDA script for AIRR‑ML‑25.

You can run this as:

    python notebooks/00_quick_eda.py --train-root ./data/train_datasets

or copy its content into a Jupyter notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(train_root: str) -> None:
    root = Path(train_root)
    print(f"[EDA] Listing datasets under {root}")
    for study in sorted(p.name for p in root.iterdir() if p.is_dir()):
        meta_path = root / study / "metadata.csv"
        if not meta_path.exists():
            print(f"  - {study}: no metadata.csv found")
            continue
        meta = pd.read_csv(meta_path)
        print(f"  - {study}: {len(meta)} repertoires")
        if "label" in meta.columns:
            print(meta["label"].value_counts(dropna=False))
        print()

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root",
        type=str,
        required=True,
        help="Path to train_datasets root directory",
    )
    args = parser.parse_args()
    main(args.train_root)
