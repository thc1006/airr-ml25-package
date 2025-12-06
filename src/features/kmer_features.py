"""
K-mer Feature Extraction for AIRR-ML-25.

Multi-scale k-mer frequency extraction with GPU acceleration support.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional
from itertools import product
import logging

logger = logging.getLogger(__name__)

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def generate_all_kmers(k: int) -> List[str]:
    """Generate all possible k-mers from amino acids.

    Args:
        k: Length of k-mers

    Returns:
        List of all possible k-mers (20^k items)
    """
    return [''.join(p) for p in product(AMINO_ACIDS, repeat=k)]


def extract_kmers_from_sequence(sequence: str, k: int) -> List[str]:
    """Extract all k-mers from a single sequence.

    Args:
        sequence: Amino acid sequence
        k: Length of k-mers

    Returns:
        List of k-mers found in sequence
    """
    if not sequence or len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def count_kmers_in_repertoire(
    repertoire: pd.DataFrame,
    k: int,
    sequence_col: str = 'junction_aa',
    binary: bool = True
) -> Counter:
    """Count k-mers across all sequences in a repertoire.

    Args:
        repertoire: DataFrame with sequences
        k: Length of k-mers
        sequence_col: Column containing sequences
        binary: If True, count each k-mer once per sequence (presence/absence)

    Returns:
        Counter of k-mer counts
    """
    kmer_counts = Counter()

    if sequence_col not in repertoire.columns:
        return kmer_counts

    for seq in repertoire[sequence_col].dropna():
        if not isinstance(seq, str):
            continue
        kmers = extract_kmers_from_sequence(seq, k)
        if binary:
            # Count each unique k-mer once per sequence
            kmer_counts.update(set(kmers))
        else:
            # Count all occurrences
            kmer_counts.update(kmers)

    return kmer_counts


def extract_kmer_features(
    repertoire: pd.DataFrame,
    k: int,
    sequence_col: str = 'junction_aa',
    binary: bool = True,
    normalize: bool = True,
    kmer_vocab: Optional[List[str]] = None
) -> np.ndarray:
    """Extract k-mer frequency features for a single repertoire.

    Args:
        repertoire: DataFrame with sequences
        k: Length of k-mers
        sequence_col: Column containing sequences
        binary: If True, use binary k-mer presence
        normalize: If True, normalize to frequencies
        kmer_vocab: Pre-defined k-mer vocabulary (if None, uses all possible)

    Returns:
        Array of shape (n_kmers,) with k-mer frequencies
    """
    if kmer_vocab is None:
        kmer_vocab = generate_all_kmers(k)

    kmer_counts = count_kmers_in_repertoire(repertoire, k, sequence_col, binary)

    # Create feature vector
    features = np.array([kmer_counts.get(kmer, 0) for kmer in kmer_vocab], dtype=np.float32)

    if normalize and features.sum() > 0:
        features = features / features.sum()

    return features


class KmerFeaturizer:
    """
    K-mer based feature extraction for immune repertoires.

    Supports multi-scale k-mers and efficient feature extraction.

    Usage:
        featurizer = KmerFeaturizer(k_values=[3, 4, 5])
        featurizer.fit(train_repertoires)  # Optional: learn top k-mers
        X = featurizer.transform_many(repertoires)
    """

    def __init__(
        self,
        k_values: List[int] = [3, 4],
        sequence_col: str = 'junction_aa',
        binary: bool = True,
        normalize: bool = True,
        max_features_per_k: Optional[int] = None
    ):
        """
        Args:
            k_values: List of k values to use (e.g., [3, 4, 5])
            sequence_col: Column name for sequences
            binary: Use binary presence/absence
            normalize: Normalize to frequencies
            max_features_per_k: If set, only keep top features per k
        """
        self.k_values = k_values
        self.sequence_col = sequence_col
        self.binary = binary
        self.normalize = normalize
        self.max_features_per_k = max_features_per_k

        # Will be set during fit
        self.kmer_vocabs_: Dict[int, List[str]] = {}
        self.feature_names_: List[str] = []
        self._is_fitted = False

    def fit(self, repertoires: List[pd.DataFrame]) -> 'KmerFeaturizer':
        """Learn k-mer vocabulary from training data.

        If max_features_per_k is set, learns the most frequent k-mers.
        Otherwise, uses all possible k-mers.
        """
        for k in self.k_values:
            if self.max_features_per_k is None:
                # Use all possible k-mers
                self.kmer_vocabs_[k] = generate_all_kmers(k)
            else:
                # Count k-mers across all repertoires and keep top ones
                total_counts = Counter()
                for rep in repertoires:
                    counts = count_kmers_in_repertoire(rep, k, self.sequence_col, self.binary)
                    total_counts.update(counts)

                # Keep top k-mers
                top_kmers = [kmer for kmer, _ in total_counts.most_common(self.max_features_per_k)]
                self.kmer_vocabs_[k] = top_kmers

        # Build feature names
        self.feature_names_ = []
        for k in self.k_values:
            for kmer in self.kmer_vocabs_[k]:
                self.feature_names_.append(f"kmer_{k}_{kmer}")

        self._is_fitted = True
        logger.info(f"KmerFeaturizer fitted with {len(self.feature_names_)} features")

        return self

    def transform(self, repertoire: pd.DataFrame) -> np.ndarray:
        """Extract k-mer features for a single repertoire.

        Returns:
            Array of shape (n_features,)
        """
        if not self._is_fitted:
            # Auto-fit with default vocabulary
            self.fit([repertoire])

        features_list = []
        for k in self.k_values:
            feat = extract_kmer_features(
                repertoire, k, self.sequence_col,
                self.binary, self.normalize, self.kmer_vocabs_[k]
            )
            features_list.append(feat)

        return np.concatenate(features_list)

    def transform_many(
        self,
        repertoires: List[pd.DataFrame],
        n_jobs: int = 1,
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract features for multiple repertoires.

        Args:
            repertoires: List of repertoire DataFrames
            n_jobs: Number of parallel jobs (not yet implemented)
            show_progress: Show progress bar

        Returns:
            Array of shape (n_repertoires, n_features)
        """
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(repertoires, desc="Extracting k-mer features")
        else:
            iterator = repertoires

        features = [self.transform(rep) for rep in iterator]
        return np.vstack(features)

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names_

    def get_kmer_to_index(self, k: int) -> Dict[str, int]:
        """Get mapping from k-mer to feature index for a specific k."""
        offset = sum(len(self.kmer_vocabs_[kk]) for kk in self.k_values if kk < k)
        return {kmer: offset + i for i, kmer in enumerate(self.kmer_vocabs_[k])}


class MultiScaleKmerFeaturizer:
    """
    Multi-scale k-mer featurizer with configurable scales.

    Optimized for AIRR-ML-25 competition.
    """

    def __init__(
        self,
        k_range: Tuple[int, int] = (3, 5),
        sequence_col: str = 'junction_aa',
        binary: bool = True,
        top_k_per_scale: Optional[int] = 5000
    ):
        """
        Args:
            k_range: Range of k values (inclusive)
            sequence_col: Column for sequences
            binary: Binary k-mer presence
            top_k_per_scale: Keep only top k-mers per scale to reduce dimensionality
        """
        self.k_range = k_range
        k_values = list(range(k_range[0], k_range[1] + 1))

        self.featurizer = KmerFeaturizer(
            k_values=k_values,
            sequence_col=sequence_col,
            binary=binary,
            normalize=True,
            max_features_per_k=top_k_per_scale
        )

    def fit(self, repertoires: List[pd.DataFrame]) -> 'MultiScaleKmerFeaturizer':
        """Fit on training repertoires."""
        self.featurizer.fit(repertoires)
        return self

    def transform(self, repertoire: pd.DataFrame) -> np.ndarray:
        """Transform single repertoire."""
        return self.featurizer.transform(repertoire)

    def transform_many(self, repertoires: List[pd.DataFrame], **kwargs) -> np.ndarray:
        """Transform multiple repertoires."""
        return self.featurizer.transform_many(repertoires, **kwargs)

    @property
    def feature_names(self) -> List[str]:
        """Return feature names."""
        return self.featurizer.feature_names


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Create synthetic repertoires
    def random_cdr3(length=12):
        return ''.join(np.random.choice(AMINO_ACIDS, length))

    repertoires = [
        pd.DataFrame({'junction_aa': [random_cdr3() for _ in range(100)]})
        for _ in range(5)
    ]

    # Test single-scale
    featurizer = KmerFeaturizer(k_values=[3])
    featurizer.fit(repertoires)
    X = featurizer.transform_many(repertoires, show_progress=False)
    print(f"Single-scale shape: {X.shape}")
    print(f"Feature names: {len(featurizer.feature_names)}")

    # Test multi-scale with top-k
    ms_featurizer = MultiScaleKmerFeaturizer(k_range=(3, 4), top_k_per_scale=1000)
    ms_featurizer.fit(repertoires)
    X2 = ms_featurizer.transform_many(repertoires, show_progress=False)
    print(f"Multi-scale shape: {X2.shape}")
