"""Gene usage feature extraction for AIRR repertoires.

This module provides functions and classes to extract V gene, J gene, and VJ pair
usage features from adaptive immune receptor repertoires.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import pandas as pd


def normalize_gene_name(gene: str) -> str:
    """Normalize gene names to handle variants.

    Removes allele information (e.g., *01, *02) to group functional variants
    together. This is important because alleles of the same gene typically have
    similar function.

    Args:
        gene: Gene name, possibly with allele suffix

    Returns:
        Normalized gene name without allele information

    Examples:
        >>> normalize_gene_name('TRBV20-1*01')
        'TRBV20-1'
        >>> normalize_gene_name('TRBV20-1*02')
        'TRBV20-1'
        >>> normalize_gene_name('TRBJ2-7')
        'TRBJ2-7'
        >>> normalize_gene_name(None)
        'UNKNOWN'
    """
    if pd.isna(gene):
        return 'UNKNOWN'
    # Remove allele info after *
    return str(gene).split('*')[0].strip()


def extract_v_gene_usage(
    repertoire: pd.DataFrame,
    v_col: str = 'v_call',
    normalize: bool = True
) -> Dict[str, float]:
    """Extract V gene usage frequencies from a repertoire.

    Args:
        repertoire: DataFrame with V gene calls
        v_col: Column name for V genes
        normalize: If True, return frequencies; else counts

    Returns:
        Dict mapping V gene names to usage values (frequencies or counts)

    Examples:
        >>> df = pd.DataFrame({'v_call': ['TRBV20-1*01', 'TRBV20-1*02', 'TRBV7-3']})
        >>> extract_v_gene_usage(df)
        {'TRBV20-1': 0.6667, 'TRBV7-3': 0.3333}
    """
    if v_col not in repertoire.columns:
        return {}

    # Normalize gene names
    v_genes = repertoire[v_col].apply(normalize_gene_name)

    # Count occurrences
    counts = Counter(v_genes)

    # Remove UNKNOWN if it exists (we don't want missing data as a feature)
    if 'UNKNOWN' in counts:
        del counts['UNKNOWN']

    if not counts:
        return {}

    if normalize:
        total = sum(counts.values())
        return {gene: count / total for gene, count in counts.items()}
    else:
        return dict(counts)


def extract_j_gene_usage(
    repertoire: pd.DataFrame,
    j_col: str = 'j_call',
    normalize: bool = True
) -> Dict[str, float]:
    """Extract J gene usage frequencies from a repertoire.

    Args:
        repertoire: DataFrame with J gene calls
        j_col: Column name for J genes
        normalize: If True, return frequencies; else counts

    Returns:
        Dict mapping J gene names to usage values (frequencies or counts)

    Examples:
        >>> df = pd.DataFrame({'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1']})
        >>> extract_j_gene_usage(df)
        {'TRBJ2-7': 0.6667, 'TRBJ1-1': 0.3333}
    """
    if j_col not in repertoire.columns:
        return {}

    # Normalize gene names
    j_genes = repertoire[j_col].apply(normalize_gene_name)

    # Count occurrences
    counts = Counter(j_genes)

    # Remove UNKNOWN if it exists
    if 'UNKNOWN' in counts:
        del counts['UNKNOWN']

    if not counts:
        return {}

    if normalize:
        total = sum(counts.values())
        return {gene: count / total for gene, count in counts.items()}
    else:
        return dict(counts)


def extract_vj_pair_usage(
    repertoire: pd.DataFrame,
    v_col: str = 'v_call',
    j_col: str = 'j_call',
    normalize: bool = True,
    top_k: Optional[int] = 50
) -> Dict[str, float]:
    """Extract VJ pair combination frequencies from a repertoire.

    VJ pairing patterns can be informative as certain disease states may show
    preferential pairing of specific V and J genes.

    Args:
        repertoire: DataFrame with V and J gene calls
        v_col: Column name for V genes
        j_col: Column name for J genes
        normalize: If True, return frequencies; else counts
        top_k: If specified, return only top k pairs; None returns all

    Returns:
        Dict mapping VJ pair names to usage values (frequencies or counts)
        Keys are formatted as 'V_GENE_J_GENE' (e.g., 'TRBV20-1_TRBJ2-7')

    Examples:
        >>> df = pd.DataFrame({
        ...     'v_call': ['TRBV20-1', 'TRBV20-1', 'TRBV7-3'],
        ...     'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1']
        ... })
        >>> extract_vj_pair_usage(df, top_k=2)
        {'TRBV20-1_TRBJ2-7': 0.6667, 'TRBV7-3_TRBJ1-1': 0.3333}
    """
    if v_col not in repertoire.columns or j_col not in repertoire.columns:
        return {}

    # Normalize gene names
    v_genes = repertoire[v_col].apply(normalize_gene_name)
    j_genes = repertoire[j_col].apply(normalize_gene_name)

    # Create VJ pair strings
    vj_pairs = [f"{v}_{j}" for v, j in zip(v_genes, j_genes)]

    # Count occurrences
    counts = Counter(vj_pairs)

    # Remove pairs with UNKNOWN
    counts = {pair: count for pair, count in counts.items()
              if 'UNKNOWN' not in pair}

    if not counts:
        return {}

    # Get top k if specified
    if top_k is not None:
        counts = dict(Counter(counts).most_common(top_k))

    if normalize:
        total = sum(counts.values())
        return {pair: count / total for pair, count in counts.items()}
    else:
        return counts


class GeneUsageFeaturizer:
    """Featurizer for V/J gene usage patterns in immune repertoires.

    This class learns the most common V genes, J genes, and VJ pairs from
    training data, then extracts frequency features for these learned patterns.

    The featurizer follows sklearn conventions:
    - fit() learns the vocabulary from training data
    - transform() extracts features using the learned vocabulary
    - feature_names provides interpretable feature labels

    Attributes:
        v_col: Column name for V gene calls
        j_col: Column name for J gene calls
        top_v_genes: Number of most common V genes to use as features
        top_j_genes: Number of most common J genes to use as features
        top_vj_pairs: Number of most common VJ pairs to use as features
        v_genes_: List of V gene names (set during fit)
        j_genes_: List of J gene names (set during fit)
        vj_pairs_: List of VJ pair names (set during fit)

    Examples:
        >>> # Train on multiple repertoires
        >>> featurizer = GeneUsageFeaturizer(top_v_genes=50, top_j_genes=15)
        >>> featurizer.fit(train_repertoires)
        >>>
        >>> # Extract features for a single repertoire
        >>> features = featurizer.transform(test_repertoire)
        >>>
        >>> # Get feature names
        >>> print(featurizer.feature_names[:5])
        ['v_TRBV20-1', 'v_TRBV7-3', 'v_TRBV12-1', 'j_TRBJ2-7', 'j_TRBJ1-1']
    """

    def __init__(
        self,
        v_col: str = 'v_call',
        j_col: str = 'j_call',
        top_v_genes: int = 50,
        top_j_genes: int = 15,
        top_vj_pairs: int = 100
    ):
        """Initialize the featurizer.

        Args:
            v_col: Column name for V gene calls
            j_col: Column name for J gene calls
            top_v_genes: Number of most common V genes to track
            top_j_genes: Number of most common J genes to track
            top_vj_pairs: Number of most common VJ pairs to track
        """
        self.v_col = v_col
        self.j_col = j_col
        self.top_v_genes = top_v_genes
        self.top_j_genes = top_j_genes
        self.top_vj_pairs = top_vj_pairs

        # Will be set during fit
        self.v_genes_: Optional[List[str]] = None
        self.j_genes_: Optional[List[str]] = None
        self.vj_pairs_: Optional[List[str]] = None

    def fit(self, repertoires: List[pd.DataFrame]) -> 'GeneUsageFeaturizer':
        """Learn the top V/J genes and pairs from training data.

        This method counts all V genes, J genes, and VJ pairs across all
        repertoires and selects the top-k most common of each to use as
        features.

        Args:
            repertoires: List of repertoire DataFrames from training data

        Returns:
            self (for method chaining)
        """
        # Aggregate counts across all repertoires
        v_counter = Counter()
        j_counter = Counter()
        vj_counter = Counter()

        for repertoire in repertoires:
            # V gene counts
            v_usage = extract_v_gene_usage(
                repertoire,
                v_col=self.v_col,
                normalize=False
            )
            v_counter.update(v_usage)

            # J gene counts
            j_usage = extract_j_gene_usage(
                repertoire,
                j_col=self.j_col,
                normalize=False
            )
            j_counter.update(j_usage)

            # VJ pair counts
            vj_usage = extract_vj_pair_usage(
                repertoire,
                v_col=self.v_col,
                j_col=self.j_col,
                normalize=False,
                top_k=None  # Get all pairs during fit
            )
            vj_counter.update(vj_usage)

        # Select top-k of each
        self.v_genes_ = [gene for gene, _ in v_counter.most_common(self.top_v_genes)]
        self.j_genes_ = [gene for gene, _ in j_counter.most_common(self.top_j_genes)]
        self.vj_pairs_ = [pair for pair, _ in vj_counter.most_common(self.top_vj_pairs)]

        return self

    def transform(self, repertoire: pd.DataFrame) -> np.ndarray:
        """Extract gene usage features for a single repertoire.

        Args:
            repertoire: DataFrame with V and J gene calls

        Returns:
            Array of shape (n_features,) with:
            - top_v_genes frequencies
            - top_j_genes frequencies
            - top_vj_pairs frequencies

        Raises:
            ValueError: If fit() has not been called yet
        """
        if self.v_genes_ is None or self.j_genes_ is None or self.vj_pairs_ is None:
            raise ValueError("Must call fit() before transform()")

        # Extract usage dictionaries
        v_usage = extract_v_gene_usage(repertoire, v_col=self.v_col, normalize=True)
        j_usage = extract_j_gene_usage(repertoire, j_col=self.j_col, normalize=True)
        vj_usage = extract_vj_pair_usage(
            repertoire,
            v_col=self.v_col,
            j_col=self.j_col,
            normalize=True,
            top_k=None
        )

        # Create feature vector
        features = []

        # V gene features
        for gene in self.v_genes_:
            features.append(v_usage.get(gene, 0.0))

        # J gene features
        for gene in self.j_genes_:
            features.append(j_usage.get(gene, 0.0))

        # VJ pair features
        for pair in self.vj_pairs_:
            features.append(vj_usage.get(pair, 0.0))

        return np.array(features, dtype=np.float32)

    def transform_many(self, repertoires: List[pd.DataFrame]) -> np.ndarray:
        """Extract features for multiple repertoires.

        Args:
            repertoires: List of repertoire DataFrames

        Returns:
            Array of shape (n_repertoires, n_features)
        """
        return np.vstack([self.transform(rep) for rep in repertoires])

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names in order.

        Returns:
            List of feature names with prefixes:
            - 'v_' for V gene features
            - 'j_' for J gene features
            - 'vj_' for VJ pair features

        Raises:
            ValueError: If fit() has not been called yet
        """
        if self.v_genes_ is None or self.j_genes_ is None or self.vj_pairs_ is None:
            raise ValueError("Must call fit() before accessing feature_names")

        names = []

        # V gene feature names
        for gene in self.v_genes_:
            names.append(f"v_{gene}")

        # J gene feature names
        for gene in self.j_genes_:
            names.append(f"j_{gene}")

        # VJ pair feature names
        for pair in self.vj_pairs_:
            names.append(f"vj_{pair}")

        return names

    @property
    def n_features(self) -> int:
        """Return the number of features.

        Returns:
            Total number of features (sum of V, J, and VJ features)

        Raises:
            ValueError: If fit() has not been called yet
        """
        if self.v_genes_ is None or self.j_genes_ is None or self.vj_pairs_ is None:
            raise ValueError("Must call fit() before accessing n_features")

        return len(self.v_genes_) + len(self.j_genes_) + len(self.vj_pairs_)


def get_gene_diversity(
    repertoire: pd.DataFrame,
    gene_col: str,
    metric: str = 'shannon'
) -> float:
    """Calculate diversity of gene usage in a repertoire.

    Diversity metrics quantify how evenly distributed gene usage is across
    different genes. Low diversity indicates dominance by few genes (clonal
    expansion), while high diversity indicates balanced usage.

    Args:
        repertoire: DataFrame with gene calls
        gene_col: Column name for gene calls
        metric: Diversity metric to use ('shannon', 'simpson', 'gini')

    Returns:
        Diversity score

    Examples:
        >>> df = pd.DataFrame({'v_call': ['TRBV20-1'] * 90 + ['TRBV7-3'] * 10})
        >>> get_gene_diversity(df, 'v_call', metric='shannon')
        0.469  # Low diversity (one gene dominates)
    """
    if gene_col not in repertoire.columns:
        return 0.0

    # Normalize and count
    genes = repertoire[gene_col].apply(normalize_gene_name)
    genes = genes[genes != 'UNKNOWN']

    if len(genes) == 0:
        return 0.0

    # Get frequencies
    counts = Counter(genes)
    total = sum(counts.values())
    freqs = np.array([count / total for count in counts.values()])

    if metric == 'shannon':
        # Shannon entropy: -sum(p * log(p))
        return -np.sum(freqs * np.log(freqs + 1e-10))

    elif metric == 'simpson':
        # Simpson index: 1 - sum(p^2)
        return 1.0 - np.sum(freqs ** 2)

    elif metric == 'gini':
        # Gini coefficient
        n = len(freqs)
        sorted_freqs = np.sort(freqs)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_freqs)) / (n * np.sum(sorted_freqs)) - (n + 1) / n

    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    # Simple test
    import doctest
    doctest.testmod()

    # Example usage
    print("Gene Usage Featurizer Example")
    print("=" * 50)

    # Create sample data
    sample_rep = pd.DataFrame({
        'v_call': ['TRBV20-1*01', 'TRBV20-1*02', 'TRBV7-3', 'TRBV12-1'] * 10,
        'j_call': ['TRBJ2-7', 'TRBJ2-7', 'TRBJ1-1', 'TRBJ1-2'] * 10
    })

    # Test individual functions
    print("\nV Gene Usage:")
    v_usage = extract_v_gene_usage(sample_rep)
    for gene, freq in sorted(v_usage.items(), key=lambda x: -x[1])[:5]:
        print(f"  {gene}: {freq:.3f}")

    print("\nJ Gene Usage:")
    j_usage = extract_j_gene_usage(sample_rep)
    for gene, freq in sorted(j_usage.items(), key=lambda x: -x[1])[:5]:
        print(f"  {gene}: {freq:.3f}")

    print("\nVJ Pair Usage:")
    vj_usage = extract_vj_pair_usage(sample_rep, top_k=3)
    for pair, freq in sorted(vj_usage.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {freq:.3f}")

    # Test featurizer
    print("\n\nTesting GeneUsageFeaturizer:")
    print("=" * 50)

    featurizer = GeneUsageFeaturizer(top_v_genes=3, top_j_genes=2, top_vj_pairs=3)
    featurizer.fit([sample_rep])

    features = featurizer.transform(sample_rep)
    print(f"\nExtracted {len(features)} features")
    print(f"Feature shape: {features.shape}")

    print("\nFeature names and values:")
    for name, value in zip(featurizer.feature_names, features):
        if value > 0:
            print(f"  {name}: {value:.3f}")

    print("\nDiversity metrics:")
    print(f"  Shannon entropy (V): {get_gene_diversity(sample_rep, 'v_call', 'shannon'):.3f}")
    print(f"  Simpson index (V): {get_gene_diversity(sample_rep, 'v_call', 'simpson'):.3f}")
    print(f"  Gini coefficient (V): {get_gene_diversity(sample_rep, 'v_call', 'gini'):.3f}")
