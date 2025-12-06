"""
Tests for Attention-based Multiple Instance Learning module.

Following TDD principles:
- Each test verifies a specific behavior
- Tests are independent and isolated
"""

import pytest
import torch
import numpy as np


class TestKmerEncoder:
    """Test k-mer encoding functionality."""

    def test_vocab_size(self):
        """Vocabulary size should be 20^k for amino acids."""
        from src.attention.attention_mil import KmerEncoder

        encoder = KmerEncoder(k=3)
        assert len(encoder.kmer_to_idx) == 20 ** 3  # 8000 k-mers

        encoder = KmerEncoder(k=2)
        assert len(encoder.kmer_to_idx) == 20 ** 2  # 400 k-mers

    def test_sequence_to_indices(self):
        """Sequence should be converted to k-mer indices."""
        from src.attention.attention_mil import KmerEncoder

        encoder = KmerEncoder(k=3)
        indices = encoder.sequence_to_indices("CASSLGQAY")

        # Sequence length 9, k=3 -> 7 k-mers
        assert len(indices) == 7

        # All indices should be positive (0 is padding)
        assert all(idx > 0 for idx in indices)

    def test_embedding_shape(self):
        """Embedding output should have correct shape."""
        from src.attention.attention_mil import KmerEncoder

        encoder = KmerEncoder(k=3, embedding_dim=64)
        indices = encoder.sequence_to_indices("CASSLGQAY")
        indices = indices.unsqueeze(0)  # Add batch dimension

        embeddings = encoder(indices)
        assert embeddings.shape == (1, 7, 64)


class TestSequenceEncoder:
    """Test sequence-level encoding."""

    def test_output_shape(self):
        """Sequence encoder should produce fixed-size output."""
        from src.attention.attention_mil import SequenceEncoder

        encoder = SequenceEncoder(input_dim=64, hidden_dim=128, output_dim=64)
        input_tensor = torch.randn(5, 10, 64)  # 5 seqs, 10 k-mers, 64 dim

        output = encoder(input_tensor)
        assert output.shape == (5, 64)

    def test_masked_pooling(self):
        """Masked pooling should ignore padded positions."""
        from src.attention.attention_mil import SequenceEncoder

        encoder = SequenceEncoder(input_dim=64, hidden_dim=128, output_dim=64)

        # Create input with some padding
        input_tensor = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 7:] = False  # First sequence only has 7 valid positions
        mask[1, 5:] = False  # Second sequence only has 5 valid positions

        output = encoder(input_tensor, mask)
        assert output.shape == (2, 64)


class TestAttentionPooling:
    """Test attention-based repertoire aggregation."""

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1."""
        from src.attention.attention_mil import AttentionPooling

        attention = AttentionPooling(input_dim=64, hidden_dim=32)
        input_tensor = torch.randn(2, 100, 64)  # 2 repertoires, 100 seqs each

        _, weights = attention(input_tensor)
        assert weights.shape == (2, 100)

        # Check weights sum to 1 (approximately)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_masked_attention(self):
        """Masked positions should have zero attention."""
        from src.attention.attention_mil import AttentionPooling

        attention = AttentionPooling(input_dim=64, hidden_dim=32)
        input_tensor = torch.randn(1, 10, 64)

        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 7:] = False  # Only first 7 positions are valid

        _, weights = attention(input_tensor, mask)

        # Masked positions should have zero weight
        assert torch.allclose(weights[0, 7:], torch.zeros(3), atol=1e-6)


class TestAttentionMIL:
    """Test full Attention MIL model."""

    def test_model_creation(self):
        """Model should be created with default parameters."""
        from src.attention.attention_mil import AttentionMIL

        model = AttentionMIL()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_encode_sequences(self):
        """Model should encode sequences to embeddings."""
        from src.attention.attention_mil import AttentionMIL

        model = AttentionMIL(k=3, seq_output_dim=64)
        sequences = ["CASSLGQAY", "CASSLAPGATNEKLF", "CASSLDRGSYEQY"]

        embeddings = model.encode_sequences(sequences)
        assert embeddings.shape == (3, 64)

    def test_forward_pass(self):
        """Forward pass should produce logits and attention."""
        from src.attention.attention_mil import AttentionMIL

        model = AttentionMIL(k=3, seq_output_dim=64)
        embeddings = torch.randn(2, 50, 64)  # 2 repertoires, 50 seqs

        logits, attention = model(embeddings)
        assert logits.shape == (2, 1)
        assert attention.shape == (2, 50)

    def test_sequence_importance(self):
        """Should rank sequences by attention importance."""
        from src.attention.attention_mil import AttentionMIL

        model = AttentionMIL(k=3)
        sequences = ["CASSLGQAY", "CASSLAPGATNEKLF", "CASSLDRGSYEQY"]
        attention_weights = torch.tensor([0.5, 0.3, 0.2])

        ranked = model.get_sequence_importance(sequences, attention_weights)
        assert len(ranked) == 3
        assert ranked[0][0] == "CASSLGQAY"  # Highest attention
        assert ranked[0][1] == 0.5


class TestAIRRDataset:
    """Test dataset for AIRR repertoires."""

    def test_dataset_length(self):
        """Dataset length should match number of repertoires."""
        from src.attention.attention_mil import AIRRDataset

        repertoires = [
            ["CASSLGQAY", "CASSLAPGATNEKLF"],
            ["CASSLDRGSYEQY", "CASSLASS"],
        ]
        labels = [0, 1]

        dataset = AIRRDataset(repertoires, labels)
        assert len(dataset) == 2

    def test_getitem_returns_tuple(self):
        """Dataset should return (sequences, label) tuple."""
        from src.attention.attention_mil import AIRRDataset

        repertoires = [["CASSLGQAY", "CASSLAPGATNEKLF"]]
        labels = [1]

        dataset = AIRRDataset(repertoires, labels)
        sequences, label = dataset[0]

        assert isinstance(sequences, list)
        assert label == 1

    def test_max_sequences_sampling(self):
        """Should sample if repertoire exceeds max_sequences."""
        from src.attention.attention_mil import AIRRDataset

        # Create large repertoire
        large_rep = [f"CASS{'A' * i}Y" for i in range(200)]
        repertoires = [large_rep]
        labels = [0]

        dataset = AIRRDataset(repertoires, labels, max_sequences=100)
        sequences, _ = dataset[0]

        assert len(sequences) == 100


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_training(self):
        """Test training on small synthetic data."""
        from src.attention.attention_mil import AttentionMIL, AIRRDataset

        # Create synthetic data
        np.random.seed(42)
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        def random_cdr3(length=12):
            return "".join(np.random.choice(amino_acids, length))

        train_repertoires = [
            [random_cdr3() for _ in range(50)]
            for _ in range(10)
        ]
        train_labels = [i % 2 for i in range(10)]

        # Create model and do one forward pass
        model = AttentionMIL(k=3)
        model.eval()

        # Process one repertoire
        rep = train_repertoires[0]
        embeds = model.encode_sequences(rep)
        embeds = embeds.unsqueeze(0)
        logits, attention = model(embeds)

        assert logits.shape == (1, 1)
        assert attention.shape == (1, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
