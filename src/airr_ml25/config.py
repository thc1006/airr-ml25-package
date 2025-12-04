"""Configuration helpers for AIRR‑ML‑25 project.

Centralizes path handling so that we don't hard‑code Kaggle‑specific paths
inside the core feature / model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DEFAULT_TRAIN_ROOT = Path("./data/train_datasets")
DEFAULT_TEST_ROOT = Path("./data/test_datasets")


@dataclass
class DatasetConfig:
    """Configuration for a single AIRR‑ML‑25 dataset/study.

    Attributes
    ----------
    name:
        Dataset name, e.g. "study1". This should match the folder name under
        `train_datasets/` and `test_datasets/` when running on Kaggle.
    has_metadata:
        Whether this dataset comes with a `metadata.csv` file containing labels.
        For test datasets this will normally be False.
    """

    name: str
    has_metadata: bool = True

    def train_dir(self, root: Path | str) -> Path:
        root = Path(root)
        return root / self.name

    def test_dir(self, root: Path | str) -> Path:
        root = Path(root)
        return root / self.name


def default_train_datasets() -> List[DatasetConfig]:
    """Return a conservative default list of train datasets.

    The actual list of studies can be obtained from the Kaggle data folder;
    this function is only meant as a starting point and should be updated
    once you have the full folder listing available.
    """

    # NOTE: Update this list after inspecting the real train_datasets/ folder.
    names = ["study1", "study2", "study3"]
    return [DatasetConfig(name=n, has_metadata=True) for n in names]


def infer_dataset_names(root: Path | str) -> List[str]:
    """Infer dataset/study folder names from a train root directory."""
    root = Path(root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
