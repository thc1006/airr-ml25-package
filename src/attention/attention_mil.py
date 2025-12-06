"""
Attention-based Multiple Instance Learning for AIRR-ML-25.

This module implements attention-based aggregation of sequence features
for repertoire-level classification. Following the DeepRC approach.

References:
- Widrich et al. "Modern Hopfield Networks and Attention for Immune
  Repertoire Classification" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter


class KmerEncoder(nn.Module):
    """Encode CDR3 sequences using k-mer features."""

    def __init__(self, k: int = 3, embedding_dim: int = 64):
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.vocab_size = len(self.amino_acids) ** k

        # K-mer vocabulary
        self._build_vocab()

        # Learnable k-mer embeddings
        self.embedding = nn.Embedding(self.vocab_size + 1, embedding_dim, padding_idx=0)

    def _build_vocab(self):
        """Build k-mer to index mapping."""
        from itertools import product
        self.kmer_to_idx = {}
        for i, kmer in enumerate(product(self.amino_acids, repeat=self.k)):
            self.kmer_to_idx[''.join(kmer)] = i + 1  # 0 reserved for padding
        self.idx_to_kmer = {v: k for k, v in self.kmer_to_idx.items()}

    def sequence_to_indices(self, seq: str) -> torch.Tensor:
        """Convert CDR3 sequence to k-mer indices."""
        indices = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if kmer in self.kmer_to_idx:
                indices.append(self.kmer_to_idx[kmer])
            else:
                indices.append(0)  # Unknown k-mer
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, kmer_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kmer_indices: (batch, max_seq_len) tensor of k-mer indices

        Returns:
            (batch, max_seq_len, embedding_dim) tensor of k-mer embeddings
        """
        return self.embedding(kmer_indices)


class SequenceEncoder(nn.Module):
    """Encode CDR3 sequences to fixed-size embeddings."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, kmer_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            kmer_embeddings: (batch, seq_len, input_dim)
            mask: (batch, seq_len) boolean mask for valid positions

        Returns:
            (batch, output_dim) sequence embedding
        """
        # Mean pooling over valid k-mers
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            pooled = (kmer_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = kmer_embeddings.mean(dim=1)

        return self.encoder(pooled)


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequences in a repertoire."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Multi-head attention
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

        # Gated attention (like DeepRC)
        self.gate_w = nn.Linear(input_dim, hidden_dim)
        self.gate_u = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_embeddings: (batch, n_sequences, input_dim)
            mask: (batch, n_sequences) boolean mask for valid sequences

        Returns:
            repertoire_embedding: (batch, input_dim)
            attention_weights: (batch, n_sequences)
        """
        batch_size, n_seq, input_dim = sequence_embeddings.shape

        # Gated attention mechanism (DeepRC style)
        gate = torch.sigmoid(self.gate_w(sequence_embeddings))
        hidden = torch.tanh(self.gate_u(sequence_embeddings))
        gated = gate * hidden  # (batch, n_seq, hidden_dim)

        # Attention scores
        scores = gated.sum(dim=-1)  # (batch, n_seq)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (batch, n_seq)

        # Weighted aggregation
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        repertoire_embedding = (sequence_embeddings * attention_weights_expanded).sum(dim=1)

        return repertoire_embedding, attention_weights


class AttentionMIL(nn.Module):
    """
    Full Attention-based Multiple Instance Learning model.

    Architecture:
    1. K-mer encoding of each CDR3 sequence
    2. Sequence-level encoding (mean pool + MLP)
    3. Attention-based repertoire aggregation
    4. Classification head
    """

    def __init__(
        self,
        k: int = 3,
        kmer_embed_dim: int = 64,
        seq_hidden_dim: int = 128,
        seq_output_dim: int = 64,
        attention_hidden_dim: int = 32,
        n_attention_heads: int = 4,
        n_classes: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.k = k

        # Layers
        self.kmer_encoder = KmerEncoder(k=k, embedding_dim=kmer_embed_dim)
        self.seq_encoder = SequenceEncoder(
            input_dim=kmer_embed_dim,
            hidden_dim=seq_hidden_dim,
            output_dim=seq_output_dim,
            dropout=dropout
        )
        self.attention = AttentionPooling(
            input_dim=seq_output_dim,
            hidden_dim=attention_hidden_dim,
            n_heads=n_attention_heads,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(seq_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Encode a list of CDR3 sequences to embeddings."""
        max_len = max(len(s) - self.k + 1 for s in sequences)
        batch_indices = []
        masks = []

        for seq in sequences:
            indices = self.kmer_encoder.sequence_to_indices(seq)
            # Pad to max length
            padded = F.pad(indices, (0, max_len - len(indices)))
            batch_indices.append(padded)
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:len(indices)] = True
            masks.append(mask)

        batch_indices = torch.stack(batch_indices)  # (n_seq, max_len)
        masks = torch.stack(masks)  # (n_seq, max_len)

        # Get k-mer embeddings
        kmer_embeds = self.kmer_encoder(batch_indices)  # (n_seq, max_len, embed_dim)

        # Encode to sequence embeddings
        seq_embeds = self.seq_encoder(kmer_embeds, masks)  # (n_seq, seq_output_dim)

        return seq_embeds

    def forward(
        self,
        repertoire_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            repertoire_embeddings: (batch, n_sequences, seq_output_dim)
            mask: (batch, n_sequences)

        Returns:
            logits: (batch, n_classes)
            attention_weights: (batch, n_sequences)
        """
        rep_embed, attn_weights = self.attention(repertoire_embeddings, mask)
        logits = self.classifier(rep_embed)
        return logits, attn_weights

    def get_sequence_importance(
        self,
        sequences: List[str],
        attention_weights: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """Get sequences ranked by attention importance."""
        scores = attention_weights.detach().cpu().numpy()
        ranked = [(seq, float(score)) for seq, score in zip(sequences, scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class AIRRDataset(torch.utils.data.Dataset):
    """Dataset for AIRR repertoires."""

    def __init__(
        self,
        repertoire_sequences: List[List[str]],
        labels: Optional[List[int]] = None,
        max_sequences: int = 1000,
        k: int = 3
    ):
        self.repertoires = repertoire_sequences
        self.labels = labels
        self.max_sequences = max_sequences
        self.k = k

    def __len__(self):
        return len(self.repertoires)

    def __getitem__(self, idx):
        sequences = self.repertoires[idx]

        # Sample if too many sequences
        if len(sequences) > self.max_sequences:
            import random
            sequences = random.sample(sequences, self.max_sequences)

        if self.labels is not None:
            return sequences, self.labels[idx]
        return sequences


def collate_repertoires(batch, model: AttentionMIL, device: torch.device):
    """Collate function for variable-size repertoires."""
    if isinstance(batch[0], tuple):
        sequences_list, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
    else:
        sequences_list = batch
        labels = None

    # Encode all sequences
    all_embeddings = []
    max_n_seq = max(len(seqs) for seqs in sequences_list)

    for sequences in sequences_list:
        with torch.no_grad():
            embeds = model.encode_sequences(sequences).to(device)

        # Pad to max number of sequences
        if len(sequences) < max_n_seq:
            pad_size = max_n_seq - len(sequences)
            padding = torch.zeros(pad_size, embeds.size(-1), device=device)
            embeds = torch.cat([embeds, padding], dim=0)

        all_embeddings.append(embeds)

    # Stack into batch
    batch_embeddings = torch.stack(all_embeddings)  # (batch, max_n_seq, embed_dim)

    # Create mask
    masks = []
    for seqs in sequences_list:
        mask = torch.zeros(max_n_seq, dtype=torch.bool, device=device)
        mask[:len(seqs)] = True
        masks.append(mask)
    masks = torch.stack(masks)

    return batch_embeddings, masks, labels


def train_attention_mil(
    train_repertoires: List[List[str]],
    train_labels: List[int],
    val_repertoires: Optional[List[List[str]]] = None,
    val_labels: Optional[List[int]] = None,
    n_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> AttentionMIL:
    """Train the Attention MIL model."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = AttentionMIL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = AIRRDataset(train_repertoires, train_labels)

    best_val_auc = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        # Simple batch iteration
        indices = np.random.permutation(len(train_dataset))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [train_dataset[j] for j in batch_idx]

            embeddings, masks, labels = collate_repertoires(batch, model, device)

            optimizer.zero_grad()
            logits, _ = model(embeddings, masks)
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(indices) // batch_size)

        # Validation
        if val_repertoires is not None and (epoch + 1) % 5 == 0:
            from sklearn.metrics import roc_auc_score
            model.eval()
            val_preds = []

            val_dataset = AIRRDataset(val_repertoires, val_labels)
            for j in range(len(val_dataset)):
                batch = [val_dataset[j]]
                embeddings, masks, _ = collate_repertoires(batch, model, device)
                with torch.no_grad():
                    logits, _ = model(embeddings, masks)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())

            val_preds = np.concatenate(val_preds)
            val_auc = roc_auc_score(val_labels, val_preds)

            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def get_important_sequences_from_model(
    model: AttentionMIL,
    repertoires: List[List[str]],
    labels: List[int],
    top_k: int = 50000,
    device: str = 'cuda'
) -> List[Tuple[str, str, str, float]]:
    """
    Extract most important sequences based on attention weights.

    Returns list of (junction_aa, v_call, j_call, score) tuples.
    Note: v_call and j_call will be placeholders since we don't have that info
    from pure sequence analysis.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Collect attention scores for positive-labeled sequences
    sequence_scores: Dict[str, float] = Counter()

    for sequences, label in zip(repertoires, labels):
        if label == 0:  # Only use positive cases
            continue

        dataset = AIRRDataset([sequences], [label])
        batch = [dataset[0]]
        embeddings, masks, _ = collate_repertoires(batch, model, device)

        with torch.no_grad():
            _, attention_weights = model(embeddings, masks)

        # Accumulate attention scores
        for seq, score in zip(sequences, attention_weights[0].cpu().numpy()):
            sequence_scores[seq] += float(score)

    # Sort by score
    sorted_sequences = sorted(
        sequence_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Return with placeholder v_call and j_call
    return [(seq, "UNKNOWN", "UNKNOWN", score) for seq, score in sorted_sequences]


if __name__ == "__main__":
    # Quick test
    model = AttentionMIL(k=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test encoding
    test_sequences = ["CASSLGQAY", "CASSLAPGATNEKLF", "CASSLDRGSYEQY"]
    embeds = model.encode_sequences(test_sequences)
    print(f"Sequence embeddings shape: {embeds.shape}")

    # Test forward pass
    embeds = embeds.unsqueeze(0)  # Add batch dimension
    logits, attn = model(embeds)
    print(f"Logits shape: {logits.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print(f"Attention weights: {attn[0].detach().numpy()}")
