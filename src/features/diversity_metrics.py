"""
Diversity Metrics for Immune Repertoire Analysis

This module implements clonality and diversity metrics commonly used in
adaptive immune receptor repertoire (AIRR) analysis.

References:
- Greiff et al. (2015). Systems Analysis Reveals High Genetic and Antigen-Driven
  Predetermination of Antibody Repertoires throughout B Cell Development.
- DeWolf et al. (2018). Quantifying size and diversity of the human T cell
  alloresponse. JCI Insight.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


def shannon_entropy(counts: np.ndarray) -> float:
    """
    Calculate Shannon entropy of clonotype distribution.

    Shannon entropy measures the uncertainty/diversity in the distribution.
    Higher values indicate more diverse repertoires.

    H = -sum(p_i * log2(p_i))

    Parameters
    ----------
    counts : np.ndarray
        Array of clone counts/frequencies

    Returns
    -------
    float
        Shannon entropy in bits

    Examples
    --------
    >>> counts = np.array([100, 50, 25, 25])  # 4 clones
    >>> shannon_entropy(counts)
    1.75
    """
    if len(counts) == 0:
        return 0.0

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]  # Remove zeros

    if len(counts) == 0:
        return 0.0

    # Convert to probabilities
    probs = counts / counts.sum()

    # Calculate entropy using scipy (handles edge cases)
    return float(scipy_entropy(probs, base=2))


def gini_coefficient(counts: np.ndarray) -> float:
    """
    Calculate Gini coefficient of clonotype distribution.

    Gini coefficient measures inequality in the distribution:
    - 0 = perfect equality (all clones have same frequency)
    - 1 = perfect inequality (one clone dominates)

    Parameters
    ----------
    counts : np.ndarray
        Array of clone counts/frequencies

    Returns
    -------
    float
        Gini coefficient [0, 1]

    Examples
    --------
    >>> counts = np.array([1, 1, 1, 1])  # Equal distribution
    >>> gini_coefficient(counts)
    0.0
    >>> counts = np.array([100, 1, 1, 1])  # One dominant clone
    >>> gini_coefficient(counts)
    0.485...
    """
    if len(counts) == 0:
        return 0.0

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]

    if len(counts) == 0:
        return 0.0

    if len(counts) == 1:
        return 1.0

    # Sort counts
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)

    # Calculate Gini coefficient
    # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

    return float(gini)


def simpson_index(counts: np.ndarray) -> float:
    """
    Calculate Simpson diversity index.

    Simpson index measures the probability that two randomly selected
    receptors belong to different clones. Higher values = more diverse.

    D = 1 - sum(p_i^2)

    Parameters
    ----------
    counts : np.ndarray
        Array of clone counts/frequencies

    Returns
    -------
    float
        Simpson diversity index [0, 1]

    Examples
    --------
    >>> counts = np.array([50, 50])  # Two equal clones
    >>> simpson_index(counts)
    0.5
    """
    if len(counts) == 0:
        return 0.0

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]

    if len(counts) == 0:
        return 0.0

    # Convert to probabilities
    probs = counts / counts.sum()

    # Simpson index = 1 - sum of squared probabilities
    simpson = 1.0 - np.sum(probs ** 2)

    return float(simpson)


def d50_index(counts: np.ndarray) -> float:
    """
    Calculate D50 index.

    D50 is the fraction of unique clones that make up 50% of the repertoire.
    Lower values indicate less diversity (few clones dominate).

    Parameters
    ----------
    counts : np.ndarray
        Array of clone counts/frequencies

    Returns
    -------
    float
        D50 index as fraction [0, 1]

    Examples
    --------
    >>> counts = np.array([40, 30, 20, 10])  # First 2 clones = 70%
    >>> d50_index(counts)
    0.5  # 2/4 clones needed for 50%
    """
    if len(counts) == 0:
        return 0.0

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]

    if len(counts) == 0:
        return 0.0

    # Sort in descending order
    sorted_counts = np.sort(counts)[::-1]
    total = sorted_counts.sum()

    # Find how many clones make up 50% of repertoire
    cumsum = np.cumsum(sorted_counts)
    threshold = total * 0.5
    n_clones_for_50pct = np.searchsorted(cumsum, threshold) + 1

    # Return as fraction of total clones
    d50 = n_clones_for_50pct / len(counts)

    return float(d50)


def clonality(counts: np.ndarray) -> float:
    """
    Calculate clonality metric.

    Clonality is the inverse of normalized Shannon entropy:
    C = 1 - (H / H_max)

    where H_max = log2(N) for N unique clones

    - 0 = maximally diverse (all clones equal frequency)
    - 1 = no diversity (single clone)

    Parameters
    ----------
    counts : np.ndarray
        Array of clone counts/frequencies

    Returns
    -------
    float
        Clonality [0, 1]

    Examples
    --------
    >>> counts = np.array([1, 1, 1, 1])  # Equal distribution
    >>> clonality(counts)
    0.0
    >>> counts = np.array([100])  # Single clone
    >>> clonality(counts)
    1.0
    """
    if len(counts) == 0:
        return 0.0

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]

    if len(counts) == 0:
        return 0.0

    if len(counts) == 1:
        return 1.0

    # Calculate Shannon entropy
    h = shannon_entropy(counts)

    # Maximum possible entropy for N clones
    h_max = np.log2(len(counts))

    if h_max == 0:
        return 1.0

    # Clonality = 1 - normalized entropy
    clonal = 1.0 - (h / h_max)

    return float(clonal)


def extract_diversity_features(
    repertoire: pd.DataFrame,
    count_col: str = 'duplicate_count',
    sequence_col: str = 'junction_aa'
) -> Dict[str, float]:
    """
    Extract all diversity features from a repertoire.

    Parameters
    ----------
    repertoire : pd.DataFrame
        Repertoire data with sequence information
    count_col : str, optional
        Column name for clone counts. If missing, treats each row as count=1
    sequence_col : str, optional
        Column name for sequences (used for grouping if count_col missing)

    Returns
    -------
    Dict[str, float]
        Dictionary with diversity metrics:
        - shannon_entropy: Shannon entropy in bits
        - gini_coefficient: Gini coefficient [0, 1]
        - simpson_index: Simpson diversity [0, 1]
        - d50_index: Fraction of clones for 50% coverage
        - clonality: 1 - normalized Shannon entropy
        - n_unique_clones: Number of unique clones
        - top_clone_fraction: Fraction of repertoire from top clone
        - top10_clone_fraction: Fraction from top 10 clones (or all if <10)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'junction_aa': ['CASSLGQAY', 'CASSLGQAY', 'CASSFTQY'],
    ...     'duplicate_count': [100, 50, 25]
    ... })
    >>> features = extract_diversity_features(df)
    >>> features['n_unique_clones']
    2.0
    """
    if len(repertoire) == 0:
        return {
            'shannon_entropy': 0.0,
            'gini_coefficient': 0.0,
            'simpson_index': 0.0,
            'd50_index': 0.0,
            'clonality': 0.0,
            'n_unique_clones': 0.0,
            'top_clone_fraction': 0.0,
            'top10_clone_fraction': 0.0,
        }

    # Get counts
    if count_col in repertoire.columns:
        counts = repertoire[count_col].values
        counts = counts[~pd.isna(counts)]
        if len(counts) == 0:
            counts = np.ones(len(repertoire))
    else:
        # No count column - group by sequence
        if sequence_col in repertoire.columns:
            counts = repertoire.groupby(sequence_col).size().values
        else:
            # No sequence column either - treat each row as unique
            counts = np.ones(len(repertoire))

    counts = counts[counts > 0]

    if len(counts) == 0:
        return {
            'shannon_entropy': 0.0,
            'gini_coefficient': 0.0,
            'simpson_index': 0.0,
            'd50_index': 0.0,
            'clonality': 0.0,
            'n_unique_clones': 0.0,
            'top_clone_fraction': 0.0,
            'top10_clone_fraction': 0.0,
        }

    # Calculate diversity metrics
    features = {
        'shannon_entropy': shannon_entropy(counts),
        'gini_coefficient': gini_coefficient(counts),
        'simpson_index': simpson_index(counts),
        'd50_index': d50_index(counts),
        'clonality': clonality(counts),
        'n_unique_clones': float(len(counts)),
    }

    # Calculate top clone fractions
    total = counts.sum()
    sorted_counts = np.sort(counts)[::-1]

    features['top_clone_fraction'] = float(sorted_counts[0] / total) if len(sorted_counts) > 0 else 0.0
    features['top10_clone_fraction'] = float(sorted_counts[:10].sum() / total) if len(sorted_counts) > 0 else 0.0

    return features


def cdr3_length_features(
    repertoire: pd.DataFrame,
    sequence_col: str = 'junction_aa',
    count_col: Optional[str] = 'duplicate_count'
) -> Dict[str, float]:
    """
    Extract CDR3 length distribution features.

    Parameters
    ----------
    repertoire : pd.DataFrame
        Repertoire data with sequence column
    sequence_col : str, optional
        Column name for CDR3 sequences
    count_col : str or None, optional
        Column name for clone counts. If provided, calculates weighted statistics.

    Returns
    -------
    Dict[str, float]
        Dictionary with length features:
        - mean_cdr3_length: Mean length
        - std_cdr3_length: Standard deviation
        - median_cdr3_length: Median length
        - min_cdr3_length: Minimum length
        - max_cdr3_length: Maximum length
        - cdr3_length_entropy: Shannon entropy of length distribution

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'junction_aa': ['CASSLGQAY', 'CASSFTQY', 'CAW'],
    ...     'duplicate_count': [100, 50, 25]
    ... })
    >>> features = cdr3_length_features(df)
    >>> features['mean_cdr3_length']
    7.428...
    """
    if len(repertoire) == 0 or sequence_col not in repertoire.columns:
        return {
            'mean_cdr3_length': 0.0,
            'std_cdr3_length': 0.0,
            'median_cdr3_length': 0.0,
            'min_cdr3_length': 0.0,
            'max_cdr3_length': 0.0,
            'cdr3_length_entropy': 0.0,
        }

    # Get sequences
    sequences = repertoire[sequence_col].dropna()

    if len(sequences) == 0:
        return {
            'mean_cdr3_length': 0.0,
            'std_cdr3_length': 0.0,
            'median_cdr3_length': 0.0,
            'min_cdr3_length': 0.0,
            'max_cdr3_length': 0.0,
            'cdr3_length_entropy': 0.0,
        }

    # Calculate lengths
    lengths = sequences.str.len().values

    # Get counts for weighting
    if count_col is not None and count_col in repertoire.columns:
        counts = repertoire.loc[sequences.index, count_col].fillna(1).values
    else:
        counts = np.ones(len(lengths))

    # Filter out invalid entries
    valid_mask = (lengths > 0) & (counts > 0)
    lengths = lengths[valid_mask]
    counts = counts[valid_mask]

    if len(lengths) == 0:
        return {
            'mean_cdr3_length': 0.0,
            'std_cdr3_length': 0.0,
            'median_cdr3_length': 0.0,
            'min_cdr3_length': 0.0,
            'max_cdr3_length': 0.0,
            'cdr3_length_entropy': 0.0,
        }

    # Calculate weighted statistics
    mean_length = np.average(lengths, weights=counts)
    variance = np.average((lengths - mean_length) ** 2, weights=counts)
    std_length = np.sqrt(variance)

    # Median (approximate for weighted case)
    sorted_indices = np.argsort(lengths)
    cumsum = np.cumsum(counts[sorted_indices])
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    median_length = lengths[sorted_indices[median_idx]]

    # Calculate length distribution entropy
    length_counts = {}
    for l, c in zip(lengths, counts):
        length_counts[l] = length_counts.get(l, 0) + c

    length_dist = np.array(list(length_counts.values()))
    length_entropy = shannon_entropy(length_dist)

    return {
        'mean_cdr3_length': float(mean_length),
        'std_cdr3_length': float(std_length),
        'median_cdr3_length': float(median_length),
        'min_cdr3_length': float(lengths.min()),
        'max_cdr3_length': float(lengths.max()),
        'cdr3_length_entropy': float(length_entropy),
    }


class DiversityFeaturizer:
    """
    Featurizer for extracting diversity and clonality metrics from repertoires.

    This class provides a scikit-learn compatible interface for extracting
    diversity features from immune repertoires.

    Parameters
    ----------
    count_col : str, optional
        Column name for clone counts
    sequence_col : str, optional
        Column name for CDR3 sequences
    include_length_features : bool, optional
        Whether to include CDR3 length distribution features

    Attributes
    ----------
    feature_names_ : List[str]
        List of feature names (available after first transform)

    Examples
    --------
    >>> featurizer = DiversityFeaturizer()
    >>> df = pd.DataFrame({'junction_aa': ['CASSLGQAY', 'CASSFTQY']})
    >>> features = featurizer.transform(df)
    >>> featurizer.feature_names
    ['shannon_entropy', 'gini_coefficient', ...]
    """

    def __init__(
        self,
        count_col: str = 'duplicate_count',
        sequence_col: str = 'junction_aa',
        include_length_features: bool = True
    ):
        self.count_col = count_col
        self.sequence_col = sequence_col
        self.include_length_features = include_length_features
        self.feature_names_ = None

    def transform(self, repertoire: pd.DataFrame) -> Dict[str, float]:
        """
        Extract all diversity features from a single repertoire.

        Parameters
        ----------
        repertoire : pd.DataFrame
            Repertoire data

        Returns
        -------
        Dict[str, float]
            Dictionary of features
        """
        # Extract diversity features
        features = extract_diversity_features(
            repertoire,
            count_col=self.count_col,
            sequence_col=self.sequence_col
        )

        # Extract length features if requested
        if self.include_length_features:
            length_features = cdr3_length_features(
                repertoire,
                sequence_col=self.sequence_col,
                count_col=self.count_col
            )
            features.update(length_features)

        # Store feature names on first call
        if self.feature_names_ is None:
            self.feature_names_ = sorted(features.keys())

        return features

    def transform_many(self, repertoires: List[pd.DataFrame]) -> np.ndarray:
        """
        Extract features for multiple repertoires.

        Parameters
        ----------
        repertoires : List[pd.DataFrame]
            List of repertoire DataFrames

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_repertoires, n_features)
        """
        feature_list = []

        for repertoire in repertoires:
            features = self.transform(repertoire)
            feature_list.append(features)

        # Convert to array with consistent ordering
        if self.feature_names_ is None:
            if len(feature_list) > 0:
                self.feature_names_ = sorted(feature_list[0].keys())
            else:
                self.feature_names_ = []

        # Build feature matrix
        n_samples = len(feature_list)
        n_features = len(self.feature_names_)
        X = np.zeros((n_samples, n_features))

        for i, features in enumerate(feature_list):
            for j, name in enumerate(self.feature_names_):
                X[i, j] = features.get(name, 0.0)

        return X

    @property
    def feature_names(self) -> List[str]:
        """
        Return list of feature names.

        Returns
        -------
        List[str]
            Feature names in order
        """
        if self.feature_names_ is None:
            # Default feature names
            names = [
                'shannon_entropy',
                'gini_coefficient',
                'simpson_index',
                'd50_index',
                'clonality',
                'n_unique_clones',
                'top_clone_fraction',
                'top10_clone_fraction',
            ]
            if self.include_length_features:
                names.extend([
                    'mean_cdr3_length',
                    'std_cdr3_length',
                    'median_cdr3_length',
                    'min_cdr3_length',
                    'max_cdr3_length',
                    'cdr3_length_entropy',
                ])
            return names
        return self.feature_names_

    def __repr__(self) -> str:
        return (
            f"DiversityFeaturizer("
            f"count_col={self.count_col!r}, "
            f"sequence_col={self.sequence_col!r}, "
            f"include_length_features={self.include_length_features})"
        )
