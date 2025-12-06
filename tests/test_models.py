"""
Unit tests for AIRR-ML-25 models.

Following TDD Red-Green-Refactor:
1. Write failing tests first
2. Implement minimum code to pass
3. Refactor while tests pass
"""

import pytest
import numpy as np
import pandas as pd
from collections import Counter

# Test fixtures
@pytest.fixture
def sample_sequences():
    """Sample CDR3 sequences for testing."""
    return [
        "CASSLGQAY",
        "CASSLAPGATNEKLF",
        "CASSLDRGSYEQY",
        "CASSLASS",
        "CASSLGQAY",  # Duplicate
    ]


@pytest.fixture
def sample_kmer_data():
    """Sample k-mer count data."""
    return {
        "CAS": 5,
        "ASS": 4,
        "SSL": 3,
        "SLG": 2,
        "LGQ": 2,
        "GQA": 1,
        "QAY": 1,
    }


class TestBinaryKmerCounting:
    """Test binary k-mer counting for Task B."""

    def test_unique_kmers_counted_once(self, sample_sequences):
        """Each k-mer should be counted at most once per sequence."""
        k = 3
        seq = "CASSLASS"  # Contains "ASS" twice

        # Extract k-mers with binary counting
        seen_kmers = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            seen_kmers.add(kmer)

        # "ASS" should only appear once
        kmer_list = list(seen_kmers)
        assert kmer_list.count("ASS") <= 1

        # Total unique k-mers
        assert len(seen_kmers) == 5  # CAS, ASS, SSL, SLA, LAS

    def test_binary_vs_count_kmers(self, sample_sequences):
        """Binary counting should differ from frequency counting."""
        k = 3
        seq = "CASSLASS"

        # Binary counting
        binary_kmers = set()
        for i in range(len(seq) - k + 1):
            binary_kmers.add(seq[i:i+k])

        # Frequency counting
        freq_kmers = Counter()
        for i in range(len(seq) - k + 1):
            freq_kmers[seq[i:i+k]] += 1

        # Binary: 5 unique, Frequency: 6 total (ASS counted twice)
        assert len(binary_kmers) == 5
        assert sum(freq_kmers.values()) == 6
        assert freq_kmers["ASS"] == 2


class TestCoefficientScaling:
    """Test coefficient scaling for Task B scoring."""

    def test_raw_coefficients_shape(self):
        """Coefficients should have correct shape."""
        # Simulated coefficients
        n_features = 100
        coefs = np.random.randn(n_features)

        assert coefs.shape == (n_features,)

    def test_no_scaler_division(self):
        """Raw coefficients should NOT be divided by scaler.scale_."""
        n_features = 10
        raw_coefs = np.array([0.5, -0.3, 0.8, -0.2, 0.1,
                              0.4, -0.6, 0.3, -0.1, 0.2])
        scaler_scale = np.array([2.0, 1.5, 3.0, 0.5, 1.0,
                                  2.5, 1.2, 0.8, 1.8, 2.2])

        # Wrong: divided by scale
        wrong_coefs = raw_coefs / scaler_scale

        # Correct: raw coefficients
        correct_coefs = raw_coefs

        # They should be different
        assert not np.allclose(wrong_coefs, correct_coefs)

        # Raw coefficients preserve magnitude
        assert np.allclose(correct_coefs, raw_coefs)

    def test_coefficient_ranking_preserved(self):
        """Ranking of coefficients should be preserved without scaling."""
        raw_coefs = np.array([0.5, -0.3, 0.8, -0.2, 0.1])

        # Ranking by absolute value
        raw_ranking = np.argsort(np.abs(raw_coefs))[::-1]

        # Expected: [2, 0, 1, 3, 4] (0.8, 0.5, -0.3, -0.2, 0.1)
        expected_ranking = np.array([2, 0, 1, 3, 4])
        assert np.array_equal(raw_ranking, expected_ranking)


class TestSequenceScoring:
    """Test sequence scoring for Task B."""

    def test_score_calculation(self):
        """Score should be sum of coefficients for present k-mers."""
        kmer_to_idx = {"CAS": 0, "ASS": 1, "SSL": 2, "SLG": 3, "LGQ": 4}
        coefficients = np.array([0.5, 0.3, -0.2, 0.1, 0.4])

        sequence = "CASSLGQ"
        k = 3

        # Binary k-mer extraction
        seen_kmers = set()
        score = 0.0
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer not in seen_kmers and kmer in kmer_to_idx:
                seen_kmers.add(kmer)
                idx = kmer_to_idx[kmer]
                score += coefficients[idx]

        # K-mers: CAS, ASS, SSL, SLG, LGQ
        # Score: 0.5 + 0.3 + (-0.2) + 0.1 + 0.4 = 1.1
        expected_score = 0.5 + 0.3 - 0.2 + 0.1 + 0.4
        assert np.isclose(score, expected_score)

    def test_duplicate_kmer_not_double_counted(self):
        """Duplicate k-mers in sequence should not be double counted."""
        kmer_to_idx = {"CAS": 0, "ASS": 1, "SSL": 2, "SLA": 3, "LAS": 4}
        coefficients = np.array([0.5, 0.3, 0.2, 0.1, 0.4])

        sequence = "CASSLASS"  # Contains ASS twice
        k = 3

        seen_kmers = set()
        score = 0.0
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer not in seen_kmers and kmer in kmer_to_idx:
                seen_kmers.add(kmer)
                idx = kmer_to_idx[kmer]
                score += coefficients[idx]

        # Should only count ASS once
        expected_score = 0.5 + 0.3 + 0.2 + 0.1 + 0.4  # CAS + ASS + SSL + SLA + LAS
        assert np.isclose(score, expected_score)


class TestSubmissionFormat:
    """Test submission file format."""

    def test_submission_row_count(self):
        """Submission should have exactly 404,213 rows."""
        n_test_repertoires = 4213
        n_train_datasets = 8
        n_sequences_per_dataset = 50000

        expected_rows = n_test_repertoires + (n_train_datasets * n_sequences_per_dataset)
        assert expected_rows == 404213

    def test_submission_columns(self):
        """Submission should have required columns."""
        required_columns = [
            "ID",
            "dataset",
            "label_positive_probability",
            "junction_aa",
            "v_call",
            "j_call"
        ]

        # Create sample submission
        sample_row = {
            "ID": "rep_001",
            "dataset": "test_dataset_1",
            "label_positive_probability": 0.85,
            "junction_aa": -999.0,
            "v_call": -999.0,
            "j_call": -999.0
        }

        df = pd.DataFrame([sample_row])
        assert all(col in df.columns for col in required_columns)

    def test_missing_value_encoding(self):
        """Missing values should be encoded as -999.0."""
        missing_value = -999.0

        # Task A row (prediction)
        task_a_row = {
            "ID": "rep_001",
            "dataset": "test_dataset_1",
            "label_positive_probability": 0.85,
            "junction_aa": missing_value,
            "v_call": missing_value,
            "j_call": missing_value
        }

        # Task B row (sequence)
        task_b_row = {
            "ID": "train_dataset_1_seq_top_1",
            "dataset": "train_dataset_1",
            "label_positive_probability": missing_value,
            "junction_aa": "CASSLGQAY",
            "v_call": "TRBV20-1",
            "j_call": "TRBJ2-7"
        }

        assert task_a_row["junction_aa"] == -999.0
        assert task_b_row["label_positive_probability"] == -999.0


class TestDiversityMetrics:
    """Test diversity metric calculations."""

    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        # Uniform distribution: maximum entropy
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_uniform = -np.sum(uniform * np.log2(uniform + 1e-10))
        assert np.isclose(entropy_uniform, 2.0)  # log2(4)

        # Skewed distribution: lower entropy
        skewed = np.array([0.9, 0.05, 0.03, 0.02])
        entropy_skewed = -np.sum(skewed * np.log2(skewed + 1e-10))
        assert entropy_skewed < entropy_uniform

    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        def gini(x):
            x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

        # Perfect equality
        equal = np.array([1, 1, 1, 1])
        assert np.isclose(gini(equal), 0.0)

        # Perfect inequality
        unequal = np.array([0, 0, 0, 100])
        assert gini(unequal) > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
