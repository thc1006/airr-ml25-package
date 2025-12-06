"""
Public Clonotype Features for AIRR-ML-25.

Public clonotypes are CDR3 sequences that appear across multiple individuals.
These often represent convergent immune responses to common antigens.

Key insight: Disease-associated public clonotypes can be powerful biomarkers.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PublicClonotypeStats:
    """Statistics about public clonotypes."""
    total_unique_sequences: int
    n_public: int
    frac_public: float
    n_disease_enriched: int
    n_healthy_enriched: int
    top_public: List[Tuple[str, int]]


def find_public_clonotypes(
    repertoires: List[pd.DataFrame],
    sequence_col: str = 'junction_aa',
    min_individuals: int = 2
) -> Set[str]:
    """
    Find CDR3 sequences that appear in multiple individuals.

    Args:
        repertoires: List of sequence DataFrames (one per repertoire)
        sequence_col: Column name containing CDR3 sequences
        min_individuals: Minimum number of individuals to be considered "public"

    Returns:
        Set of public CDR3 sequences
    """
    sequence_individuals: Dict[str, Set[int]] = defaultdict(set)

    for i, rep in enumerate(repertoires):
        if sequence_col not in rep.columns:
            continue
        unique_seqs = set(rep[sequence_col].dropna().unique())
        for seq in unique_seqs:
            sequence_individuals[seq].add(i)

    public = {
        seq for seq, individuals in sequence_individuals.items()
        if len(individuals) >= min_individuals
    }

    return public


def find_disease_associated_public(
    repertoires: List[pd.DataFrame],
    labels: List[int],
    sequence_col: str = 'junction_aa',
    public_set: Optional[Set[str]] = None,
    min_count: int = 5
) -> Dict[str, float]:
    """
    Find public clonotypes enriched in disease (positive) cases.

    Args:
        repertoires: List of sequence DataFrames
        labels: List of binary labels (1=disease, 0=healthy)
        sequence_col: Column name containing CDR3 sequences
        public_set: Pre-computed public clonotype set (optional)
        min_count: Minimum total count for a sequence to be considered

    Returns:
        Dict mapping sequence to disease enrichment score (0-1)
    """
    if public_set is None:
        public_set = find_public_clonotypes(repertoires, sequence_col)

    disease_counts: Counter = Counter()
    healthy_counts: Counter = Counter()

    for rep, label in zip(repertoires, labels):
        if sequence_col not in rep.columns:
            continue
        unique_seqs = set(rep[sequence_col].dropna().unique())
        relevant_seqs = unique_seqs & public_set

        if label == 1:
            disease_counts.update(relevant_seqs)
        else:
            healthy_counts.update(relevant_seqs)

    # Calculate enrichment scores
    enrichment = {}
    for seq in public_set:
        d = disease_counts.get(seq, 0)
        h = healthy_counts.get(seq, 0)
        total = d + h
        if total >= min_count:
            # Disease enrichment = P(disease | seq)
            enrichment[seq] = d / (total + 1e-6)

    return enrichment


def extract_public_clonotype_features(
    repertoire: pd.DataFrame,
    public_set: Set[str],
    disease_enriched: Optional[Dict[str, float]] = None,
    sequence_col: str = 'junction_aa'
) -> Dict[str, float]:
    """
    Extract public clonotype features for a single repertoire.

    Features:
    - n_public: Number of public clonotypes in repertoire
    - frac_public: Fraction of unique sequences that are public
    - n_disease_enriched: Number of disease-enriched public clonotypes
    - avg_disease_enrichment: Average enrichment of public clonotypes
    - max_disease_enrichment: Max enrichment of public clonotypes
    - public_template_fraction: Fraction of total templates from public clonotypes

    Args:
        repertoire: Sequence DataFrame
        public_set: Set of public clonotype sequences
        disease_enriched: Dict of sequence -> disease enrichment score
        sequence_col: Column name for sequences

    Returns:
        Dict of feature name -> value
    """
    if sequence_col not in repertoire.columns:
        return {
            'n_public': 0,
            'frac_public': 0.0,
            'n_disease_enriched': 0,
            'avg_disease_enrichment': 0.0,
            'max_disease_enrichment': 0.0,
            'public_template_fraction': 0.0,
        }

    unique_seqs = set(repertoire[sequence_col].dropna().unique())
    public_in_rep = unique_seqs & public_set

    n_public = len(public_in_rep)
    frac_public = n_public / max(len(unique_seqs), 1)

    features = {
        'n_public': n_public,
        'frac_public': frac_public,
    }

    # Disease enrichment features
    if disease_enriched:
        enrichments = [
            disease_enriched.get(seq, 0.5)
            for seq in public_in_rep
            if seq in disease_enriched
        ]

        if enrichments:
            features['n_disease_enriched'] = sum(1 for e in enrichments if e > 0.6)
            features['avg_disease_enrichment'] = np.mean(enrichments)
            features['max_disease_enrichment'] = np.max(enrichments)
        else:
            features['n_disease_enriched'] = 0
            features['avg_disease_enrichment'] = 0.0
            features['max_disease_enrichment'] = 0.0
    else:
        features['n_disease_enriched'] = 0
        features['avg_disease_enrichment'] = 0.0
        features['max_disease_enrichment'] = 0.0

    # Template fraction (if templates column exists)
    if 'templates' in repertoire.columns or 'duplicate_count' in repertoire.columns:
        template_col = 'templates' if 'templates' in repertoire.columns else 'duplicate_count'

        mask = repertoire[sequence_col].isin(public_in_rep)
        public_templates = repertoire.loc[mask, template_col].sum()
        total_templates = repertoire[template_col].sum()

        features['public_template_fraction'] = (
            public_templates / max(total_templates, 1)
        )
    else:
        features['public_template_fraction'] = 0.0

    return features


class PublicClonotypeFeaturizer:
    """
    Featurizer for public clonotype-based features.

    Usage:
        featurizer = PublicClonotypeFeaturizer()
        featurizer.fit(train_repertoires, train_labels)
        features = featurizer.transform(test_repertoire)
    """

    def __init__(
        self,
        sequence_col: str = 'junction_aa',
        min_individuals: int = 2,
        min_count: int = 5,
        enrichment_threshold: float = 0.6
    ):
        self.sequence_col = sequence_col
        self.min_individuals = min_individuals
        self.min_count = min_count
        self.enrichment_threshold = enrichment_threshold

        self.public_set_: Optional[Set[str]] = None
        self.disease_enriched_: Optional[Dict[str, float]] = None
        self.top_sequences_: Optional[List[Tuple[str, float]]] = None

    def fit(
        self,
        repertoires: List[pd.DataFrame],
        labels: List[int]
    ) -> 'PublicClonotypeFeaturizer':
        """Fit the featurizer on training data."""
        # Find public clonotypes
        self.public_set_ = find_public_clonotypes(
            repertoires,
            self.sequence_col,
            self.min_individuals
        )

        # Find disease-associated public clonotypes
        self.disease_enriched_ = find_disease_associated_public(
            repertoires,
            labels,
            self.sequence_col,
            self.public_set_,
            self.min_count
        )

        # Store top disease-enriched sequences for Task B
        self.top_sequences_ = sorted(
            self.disease_enriched_.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return self

    def transform(self, repertoire: pd.DataFrame) -> Dict[str, float]:
        """Extract features for a single repertoire."""
        if self.public_set_ is None:
            raise ValueError("Featurizer not fitted. Call fit() first.")

        return extract_public_clonotype_features(
            repertoire,
            self.public_set_,
            self.disease_enriched_,
            self.sequence_col
        )

    def transform_many(
        self,
        repertoires: List[pd.DataFrame]
    ) -> np.ndarray:
        """Extract features for multiple repertoires."""
        features = [self.transform(rep) for rep in repertoires]

        # Convert to array
        feature_names = list(features[0].keys())
        X = np.array([
            [f[name] for name in feature_names]
            for f in features
        ])

        return X

    def get_important_sequences(
        self,
        top_k: int = 50000
    ) -> List[Tuple[str, str, str, float]]:
        """
        Get top disease-associated public clonotypes for Task B.

        Returns:
            List of (junction_aa, v_call, j_call, score) tuples
        """
        if self.top_sequences_ is None:
            raise ValueError("Featurizer not fitted. Call fit() first.")

        # Note: We don't have V/J info from public clonotypes alone
        # These would need to be looked up from the original data
        return [
            (seq, "UNKNOWN", "UNKNOWN", score)
            for seq, score in self.top_sequences_[:top_k]
        ]

    def get_stats(self) -> PublicClonotypeStats:
        """Get statistics about public clonotypes."""
        if self.public_set_ is None:
            raise ValueError("Featurizer not fitted.")

        n_disease_enriched = sum(
            1 for score in self.disease_enriched_.values()
            if score > self.enrichment_threshold
        )
        n_healthy_enriched = sum(
            1 for score in self.disease_enriched_.values()
            if score < (1 - self.enrichment_threshold)
        )

        return PublicClonotypeStats(
            total_unique_sequences=0,  # Not tracked
            n_public=len(self.public_set_),
            frac_public=0.0,  # Not meaningful without total
            n_disease_enriched=n_disease_enriched,
            n_healthy_enriched=n_healthy_enriched,
            top_public=self.top_sequences_[:10] if self.top_sequences_ else []
        )


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Create synthetic repertoires
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def random_cdr3(length=12):
        return "".join(np.random.choice(amino_acids, length))

    # Create repertoires with some shared sequences
    shared_seqs = [random_cdr3() for _ in range(100)]

    repertoires = []
    labels = []
    for i in range(20):
        n_seqs = np.random.randint(100, 200)
        seqs = [random_cdr3() for _ in range(n_seqs - 20)]

        # Add some shared sequences
        seqs.extend(np.random.choice(shared_seqs, 20, replace=False))

        rep = pd.DataFrame({'junction_aa': seqs})
        repertoires.append(rep)
        labels.append(i % 2)  # Alternate labels

    # Test featurizer
    featurizer = PublicClonotypeFeaturizer()
    featurizer.fit(repertoires, labels)

    print(f"Public clonotypes found: {len(featurizer.public_set_)}")
    print(f"Disease-enriched: {len([s for s in featurizer.disease_enriched_.values() if s > 0.6])}")

    # Transform
    features = featurizer.transform(repertoires[0])
    print(f"Features: {features}")

    # Get stats
    stats = featurizer.get_stats()
    print(f"Stats: {stats}")
