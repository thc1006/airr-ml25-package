"""
Feature engineering modules for AIRR-ML-25.

This package contains various feature extraction methods for
immune repertoire classification.
"""

from .public_clonotypes import (
    PublicClonotypeFeaturizer,
    find_public_clonotypes,
    find_disease_associated_public,
    extract_public_clonotype_features,
)

from .kmer_features import (
    KmerFeaturizer,
    MultiScaleKmerFeaturizer,
    extract_kmer_features,
    extract_kmers_from_sequence,
)

from .diversity_metrics import (
    DiversityFeaturizer,
    extract_diversity_features,
    cdr3_length_features,
    shannon_entropy,
    gini_coefficient,
    simpson_index,
    d50_index,
    clonality,
)

from .gene_usage import (
    GeneUsageFeaturizer,
    extract_v_gene_usage,
    extract_j_gene_usage,
    extract_vj_pair_usage,
    normalize_gene_name,
)

__all__ = [
    # Public clonotypes
    'PublicClonotypeFeaturizer',
    'find_public_clonotypes',
    'find_disease_associated_public',
    'extract_public_clonotype_features',
    # K-mer features
    'KmerFeaturizer',
    'MultiScaleKmerFeaturizer',
    'extract_kmer_features',
    'extract_kmers_from_sequence',
    # Diversity metrics
    'DiversityFeaturizer',
    'extract_diversity_features',
    'cdr3_length_features',
    'shannon_entropy',
    'gini_coefficient',
    'simpson_index',
    'd50_index',
    'clonality',
    # Gene usage
    'GeneUsageFeaturizer',
    'extract_v_gene_usage',
    'extract_j_gene_usage',
    'extract_vj_pair_usage',
    'normalize_gene_name',
]
