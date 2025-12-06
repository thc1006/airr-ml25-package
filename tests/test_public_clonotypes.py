"""
Tests for Public Clonotype Features module.

Following TDD principles:
- Each test verifies a specific behavior
- Tests are independent and isolated
"""

import pytest
import numpy as np
import pandas as pd


class TestFindPublicClonotypes:
    """Test public clonotype identification."""

    def test_empty_repertoires(self):
        """Should handle empty repertoire list."""
        from src.features.public_clonotypes import find_public_clonotypes

        result = find_public_clonotypes([])
        assert len(result) == 0

    def test_no_shared_sequences(self):
        """Should return empty set when no sequences are shared."""
        from src.features.public_clonotypes import find_public_clonotypes

        rep1 = pd.DataFrame({'junction_aa': ['CASS1', 'CASS2']})
        rep2 = pd.DataFrame({'junction_aa': ['CASS3', 'CASS4']})

        result = find_public_clonotypes([rep1, rep2])
        assert len(result) == 0

    def test_shared_sequences(self):
        """Should find sequences appearing in multiple repertoires."""
        from src.features.public_clonotypes import find_public_clonotypes

        rep1 = pd.DataFrame({'junction_aa': ['CASS1', 'CASS2', 'SHARED']})
        rep2 = pd.DataFrame({'junction_aa': ['CASS3', 'SHARED', 'CASS4']})
        rep3 = pd.DataFrame({'junction_aa': ['CASS5', 'SHARED']})

        result = find_public_clonotypes([rep1, rep2, rep3])
        assert 'SHARED' in result

    def test_min_individuals_threshold(self):
        """Should respect minimum individuals threshold."""
        from src.features.public_clonotypes import find_public_clonotypes

        rep1 = pd.DataFrame({'junction_aa': ['CASS1', 'SHARED']})
        rep2 = pd.DataFrame({'junction_aa': ['CASS2', 'SHARED']})
        rep3 = pd.DataFrame({'junction_aa': ['CASS3']})

        # With threshold 2, SHARED should be found
        result = find_public_clonotypes([rep1, rep2, rep3], min_individuals=2)
        assert 'SHARED' in result

        # With threshold 3, SHARED should not be found
        result = find_public_clonotypes([rep1, rep2, rep3], min_individuals=3)
        assert 'SHARED' not in result


class TestDiseaseAssociatedPublic:
    """Test disease enrichment calculation."""

    def test_disease_enrichment_calculation(self):
        """Should calculate correct enrichment scores."""
        from src.features.public_clonotypes import find_disease_associated_public

        # Disease repertoires have "DISEASE_SEQ"
        rep1 = pd.DataFrame({'junction_aa': ['DISEASE_SEQ', 'SEQ1']})
        rep2 = pd.DataFrame({'junction_aa': ['DISEASE_SEQ', 'SEQ2']})
        rep3 = pd.DataFrame({'junction_aa': ['DISEASE_SEQ', 'SEQ3']})
        # Healthy repertoires have "HEALTHY_SEQ"
        rep4 = pd.DataFrame({'junction_aa': ['HEALTHY_SEQ', 'SEQ4']})
        rep5 = pd.DataFrame({'junction_aa': ['HEALTHY_SEQ', 'SEQ5']})
        rep6 = pd.DataFrame({'junction_aa': ['HEALTHY_SEQ', 'SEQ6']})

        repertoires = [rep1, rep2, rep3, rep4, rep5, rep6]
        labels = [1, 1, 1, 0, 0, 0]

        public = {'DISEASE_SEQ', 'HEALTHY_SEQ'}
        result = find_disease_associated_public(
            repertoires, labels,
            public_set=public,
            min_count=2
        )

        # DISEASE_SEQ should have high enrichment (appears in disease only)
        assert result['DISEASE_SEQ'] > 0.8

        # HEALTHY_SEQ should have low enrichment (appears in healthy only)
        assert result['HEALTHY_SEQ'] < 0.2


class TestExtractFeatures:
    """Test feature extraction for repertoires."""

    def test_feature_names(self):
        """Should extract expected feature names."""
        from src.features.public_clonotypes import extract_public_clonotype_features

        rep = pd.DataFrame({'junction_aa': ['CASS1', 'CASS2']})
        public_set = {'CASS1'}

        features = extract_public_clonotype_features(rep, public_set)

        expected_keys = [
            'n_public', 'frac_public',
            'n_disease_enriched', 'avg_disease_enrichment',
            'max_disease_enrichment', 'public_template_fraction'
        ]
        assert all(key in features for key in expected_keys)

    def test_n_public_count(self):
        """Should count public clonotypes correctly."""
        from src.features.public_clonotypes import extract_public_clonotype_features

        rep = pd.DataFrame({'junction_aa': ['CASS1', 'CASS2', 'CASS3']})
        public_set = {'CASS1', 'CASS2'}

        features = extract_public_clonotype_features(rep, public_set)
        assert features['n_public'] == 2

    def test_frac_public(self):
        """Should calculate fraction correctly."""
        from src.features.public_clonotypes import extract_public_clonotype_features

        rep = pd.DataFrame({'junction_aa': ['CASS1', 'CASS2', 'CASS3', 'CASS4']})
        public_set = {'CASS1', 'CASS2'}

        features = extract_public_clonotype_features(rep, public_set)
        assert features['frac_public'] == 0.5

    def test_empty_repertoire(self):
        """Should handle repertoire with missing column."""
        from src.features.public_clonotypes import extract_public_clonotype_features

        rep = pd.DataFrame({'other_col': ['data']})
        public_set = {'CASS1'}

        features = extract_public_clonotype_features(rep, public_set)
        assert features['n_public'] == 0
        assert features['frac_public'] == 0.0


class TestPublicClonotypeFeaturizer:
    """Test the featurizer class."""

    def test_fit(self):
        """Featurizer should fit on training data."""
        from src.features.public_clonotypes import PublicClonotypeFeaturizer

        rep1 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ1']})
        rep2 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ2']})
        labels = [1, 0]

        featurizer = PublicClonotypeFeaturizer()
        featurizer.fit([rep1, rep2], labels)

        assert featurizer.public_set_ is not None
        assert 'SHARED' in featurizer.public_set_

    def test_transform(self):
        """Featurizer should transform repertoires to features."""
        from src.features.public_clonotypes import PublicClonotypeFeaturizer

        rep1 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ1']})
        rep2 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ2']})
        labels = [1, 0]

        featurizer = PublicClonotypeFeaturizer()
        featurizer.fit([rep1, rep2], labels)

        test_rep = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ3']})
        features = featurizer.transform(test_rep)

        assert isinstance(features, dict)
        assert 'n_public' in features

    def test_transform_without_fit(self):
        """Should raise error if transform called without fit."""
        from src.features.public_clonotypes import PublicClonotypeFeaturizer

        featurizer = PublicClonotypeFeaturizer()
        test_rep = pd.DataFrame({'junction_aa': ['CASS1']})

        with pytest.raises(ValueError):
            featurizer.transform(test_rep)

    def test_transform_many(self):
        """Should transform multiple repertoires to array."""
        from src.features.public_clonotypes import PublicClonotypeFeaturizer

        rep1 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ1']})
        rep2 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ2']})
        labels = [1, 0]

        featurizer = PublicClonotypeFeaturizer()
        featurizer.fit([rep1, rep2], labels)

        X = featurizer.transform_many([rep1, rep2])
        assert X.shape[0] == 2
        assert X.shape[1] == 6  # Number of features

    def test_get_important_sequences(self):
        """Should return ranked disease-associated sequences."""
        from src.features.public_clonotypes import PublicClonotypeFeaturizer

        rep1 = pd.DataFrame({'junction_aa': ['DISEASE_SEQ', 'SEQ1']})
        rep2 = pd.DataFrame({'junction_aa': ['DISEASE_SEQ', 'SEQ2']})
        rep3 = pd.DataFrame({'junction_aa': ['HEALTHY_SEQ', 'SEQ3']})
        rep4 = pd.DataFrame({'junction_aa': ['HEALTHY_SEQ', 'SEQ4']})
        labels = [1, 1, 0, 0]

        featurizer = PublicClonotypeFeaturizer(min_count=1)
        featurizer.fit([rep1, rep2, rep3, rep4], labels)

        important = featurizer.get_important_sequences(top_k=10)
        assert len(important) > 0

        # First element should be tuple (seq, v_call, j_call, score)
        assert len(important[0]) == 4


class TestStats:
    """Test statistics functionality."""

    def test_get_stats(self):
        """Should return PublicClonotypeStats object."""
        from src.features.public_clonotypes import PublicClonotypeFeaturizer

        rep1 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ1']})
        rep2 = pd.DataFrame({'junction_aa': ['SHARED', 'SEQ2']})
        labels = [1, 0]

        featurizer = PublicClonotypeFeaturizer()
        featurizer.fit([rep1, rep2], labels)

        stats = featurizer.get_stats()
        assert stats.n_public >= 0
        assert isinstance(stats.top_public, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
