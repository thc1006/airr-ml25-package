#!/usr/bin/env python3
"""
🏆 Championship Deep Learning Pipeline for AIRR-ML-25
Target: Beat GROZD team (0.81364) → Achieve 0.82+

Architecture:
- ESM-2 (650M params) for sequence embeddings
- Multi-head attention aggregation over repertoires
- Hybrid features: Deep + Traditional (V/J usage, clonality)
- Leave-one-dataset-out cross-validation
- GPU-optimized batch processing on RTX 5080
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# ESM-2 Feature Extractor
# ============================================================================

class ESM2FeatureExtractor:
    """Extract sequence embeddings using ESM-2 protein language model."""

    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda"):
        """
        Initialize ESM-2 model.

        Args:
            model_name: ESM-2 model variant
                - esm2_t12_35M_UR50D (fast, 35M params)
                - esm2_t30_150M_UR50D (balanced, 150M params)
                - esm2_t33_650M_UR50D (powerful, 650M params) ← Default
                - esm2_t36_3B_UR50D (best, 3B params, requires 12GB VRAM)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"Loading ESM-2 model: {model_name}...")

        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✓ ESM-2 loaded successfully on {device}")
        except Exception as e:
            print(f"✗ Failed to load ESM-2: {e}")
            print("  Installing fair-esm package...")
            os.system("pip install fair-esm")
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = self.model.to(device).eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            print(f"✓ ESM-2 loaded successfully on {device}")

    def extract_embeddings(self, sequences: List[str], batch_size: int = 32, max_seqs: int = 1000) -> np.ndarray:
        """
        Extract sequence embeddings in batches.

        Args:
            sequences: List of CDR3 amino acid sequences
            batch_size: Number of sequences to process at once
            max_seqs: Maximum sequences to process (for memory efficiency)

        Returns:
            numpy array of shape (n_sequences, embedding_dim)
        """
        # Sample sequences if too many
        if len(sequences) > max_seqs:
            np.random.seed(42)
            indices = np.random.choice(len(sequences), max_seqs, replace=False)
            sequences = [sequences[i] for i in sorted(indices)]

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]

                # Prepare batch
                batch_labels = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_labels)
                batch_tokens = batch_tokens.to(self.device)

                # Extract representations
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

                # Get mean pooled representation for each sequence
                token_representations = results["representations"][33]

                # Mean pool over sequence length (excluding special tokens)
                for j, seq_len in enumerate([len(s) for s in batch_seqs]):
                    # tokens: <cls> seq... <eos>
                    seq_repr = token_representations[j, 1:seq_len+1].mean(0)
                    embeddings.append(seq_repr.cpu().numpy())

                # Clear GPU memory
                del batch_tokens, results, token_representations
                torch.cuda.empty_cache()

        return np.array(embeddings)


# ============================================================================
# Traditional Feature Extractors (for Hybrid Model)
# ============================================================================

def extract_vj_features(df: pd.DataFrame, top_n: int = 50) -> Dict[str, float]:
    """Extract V/J gene usage patterns."""
    features = {}
    total = len(df)

    if total == 0:
        return features

    # V gene usage
    if 'v_call' in df.columns:
        v_counts = df['v_call'].value_counts().head(top_n)
        for gene, count in v_counts.items():
            if pd.notna(gene):
                features[f"v_{gene}"] = count / total

    # J gene usage
    if 'j_call' in df.columns:
        j_counts = df['j_call'].value_counts().head(top_n)
        for gene, count in j_counts.items():
            if pd.notna(gene):
                features[f"j_{gene}"] = count / total

    # VJ pair combinations
    if 'v_call' in df.columns and 'j_call' in df.columns:
        vj_pairs = df[['v_call', 'j_call']].apply(
            lambda x: f"{x['v_call']}_{x['j_call']}", axis=1
        )
        vj_counts = vj_pairs.value_counts().head(top_n)
        for pair, count in vj_counts.items():
            if pd.notna(pair):
                features[f"vj_{pair}"] = count / total

    return features


def extract_clonality_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extract clonality and diversity metrics."""
    features = {}

    if 'junction_aa' not in df.columns or len(df) == 0:
        return features

    seq_counts = df['junction_aa'].value_counts()
    frequencies = seq_counts.values / seq_counts.sum()

    # Shannon entropy
    features['shannon_entropy'] = entropy(frequencies)

    # Gini-Simpson index
    features['gini_simpson'] = 1 - np.sum(frequencies ** 2)

    # D50 - number of clones accounting for 50% of repertoire
    cumsum = np.cumsum(np.sort(frequencies)[::-1])
    features['d50'] = np.sum(cumsum <= 0.5)

    # Clonality score
    max_entropy = np.log(len(seq_counts))
    if max_entropy > 0:
        features['clonality'] = 1 - (features['shannon_entropy'] / max_entropy)
    else:
        features['clonality'] = 0

    # CDR3 length statistics
    lengths = df['junction_aa'].str.len()
    features['mean_length'] = lengths.mean()
    features['std_length'] = lengths.std()
    features['min_length'] = lengths.min()
    features['max_length'] = lengths.max()

    # Top clone frequency
    features['top_clone_freq'] = frequencies[0] if len(frequencies) > 0 else 0

    return features


def extract_features_from_repertoire(tsv_path: str) -> Dict[str, float]:
    """Extract all traditional features from a repertoire TSV file."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')

        # Extract features
        vj_features = extract_vj_features(df)
        clonality_features = extract_clonality_features(df)

        # Combine
        all_features = {**vj_features, **clonality_features}

        return all_features, df['junction_aa'].tolist()
    except Exception as e:
        print(f"Error processing {tsv_path}: {e}")
        return {}, []


# ============================================================================
# Attention-based Repertoire Aggregator (DeepRC-style)
# ============================================================================

class AttentionAggregator(nn.Module):
    """Multi-head attention aggregation over variable-length repertoires."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Query token (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, sequence_embeddings: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            sequence_embeddings: (batch_size, n_sequences, embedding_dim)
            mask: (batch_size, n_sequences) - True for valid positions

        Returns:
            aggregated: (batch_size, embedding_dim)
        """
        batch_size = sequence_embeddings.size(0)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Self-attention aggregation
        attn_output, attn_weights = self.attention(
            query=query,
            key=sequence_embeddings,
            value=sequence_embeddings,
            key_padding_mask=~mask if mask is not None else None
        )

        # Squeeze and normalize
        aggregated = self.norm(attn_output.squeeze(1))

        return aggregated, attn_weights


# ============================================================================
# Championship Classifier Model
# ============================================================================

class ChampionshipClassifier(nn.Module):
    """Hybrid Deep Learning + Traditional Features Classifier."""

    def __init__(self, esm_dim: int = 1280, trad_dim: int = 100,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.3):
        super().__init__()

        # Attention aggregator for ESM embeddings
        self.attention = AttentionAggregator(
            input_dim=esm_dim,
            hidden_dim=256,
            num_heads=4
        )

        # MLP for combined features
        layers = []
        input_dim = esm_dim + trad_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, esm_embeddings: torch.Tensor, trad_features: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Args:
            esm_embeddings: (batch_size, n_sequences, esm_dim)
            trad_features: (batch_size, trad_dim)
            mask: (batch_size, n_sequences)

        Returns:
            logits: (batch_size, 1)
            attention_weights: attention scores
        """
        # Aggregate ESM embeddings with attention
        aggregated_esm, attn_weights = self.attention(esm_embeddings, mask)

        # Concatenate with traditional features
        combined = torch.cat([aggregated_esm, trad_features], dim=1)

        # Final classification
        logits = self.mlp(combined)

        return logits, attn_weights


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class RepertoireDataset(Dataset):
    """Dataset for variable-length repertoires."""

    def __init__(self, repertoire_data: List[Dict]):
        """
        Args:
            repertoire_data: List of dicts containing:
                - 'esm_embeddings': numpy array (n_seqs, esm_dim)
                - 'trad_features': numpy array (trad_dim,)
                - 'label': int (0 or 1)
                - 'repertoire_id': str
                - 'dataset_id': int
        """
        self.data = repertoire_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_repertoires(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length repertoires."""

    # Find max length in batch
    max_len = max(item['esm_embeddings'].shape[0] for item in batch)
    esm_dim = batch[0]['esm_embeddings'].shape[1]
    trad_dim = batch[0]['trad_features'].shape[0]
    batch_size = len(batch)

    # Initialize padded tensors
    esm_padded = torch.zeros(batch_size, max_len, esm_dim)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    trad_features = torch.zeros(batch_size, trad_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)

    # Fill in the data
    for i, item in enumerate(batch):
        seq_len = item['esm_embeddings'].shape[0]
        esm_padded[i, :seq_len] = torch.from_numpy(item['esm_embeddings'])
        masks[i, :seq_len] = True
        trad_features[i] = torch.from_numpy(item['trad_features'])
        labels[i] = item['label']

    return {
        'esm_embeddings': esm_padded.float(),
        'trad_features': trad_features.float(),
        'masks': masks,
        'labels': labels.float(),
        'repertoire_ids': [item['repertoire_id'] for item in batch]
    }


# ============================================================================
# Data Loading Functions
# ============================================================================

def standardize_features(feature_dict: Dict[str, float], all_feature_names: List[str]) -> np.ndarray:
    """Convert feature dict to standardized numpy array."""
    feature_vector = np.zeros(len(all_feature_names))
    for i, name in enumerate(all_feature_names):
        if name in feature_dict:
            feature_vector[i] = feature_dict[name]
    return feature_vector


def load_dataset(dataset_path: str, dataset_id: int, esm_extractor: ESM2FeatureExtractor,
                 all_feature_names: List[str]) -> List[Dict]:
    """Load one dataset and extract all features."""
    print(f"\n📂 Loading dataset {dataset_id} from {dataset_path}")

    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)

    repertoire_data = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Dataset {dataset_id}"):
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = 1 if row['label_positive'] else 0

        # Full path to TSV file
        tsv_path = os.path.join(dataset_path, filename)

        if not os.path.exists(tsv_path):
            print(f"⚠️  File not found: {tsv_path}")
            continue

        try:
            # Extract traditional features and sequences
            trad_features_dict, sequences = extract_features_from_repertoire(tsv_path)

            if len(sequences) == 0:
                print(f"⚠️  No sequences in {repertoire_id}")
                continue

            # Extract ESM-2 embeddings (sample if too many sequences)
            esm_embeddings = esm_extractor.extract_embeddings(sequences, batch_size=32, max_seqs=1000)

            # Standardize traditional features
            trad_features = standardize_features(trad_features_dict, all_feature_names)

            repertoire_data.append({
                'esm_embeddings': esm_embeddings,
                'trad_features': trad_features,
                'label': label,
                'repertoire_id': repertoire_id,
                'dataset_id': dataset_id,
                'sequences': sequences[:1000]  # Store for Task B
            })

        except Exception as e:
            print(f"❌ Error processing {repertoire_id}: {e}")
            continue

    print(f"✓ Loaded {len(repertoire_data)} repertoires from dataset {dataset_id}")
    return repertoire_data


def load_all_training_data(train_root: str, esm_extractor: ESM2FeatureExtractor) -> Tuple[List[Dict], List[str]]:
    """Load all 8 training datasets."""
    print("\n" + "="*70)
    print("📊 LOADING ALL TRAINING DATA")
    print("="*70)

    # First pass: collect all feature names
    print("\n🔍 Phase 1: Collecting all feature names...")
    all_feature_names_set = set()

    for dataset_id in range(1, 9):
        dataset_path = os.path.join(train_root, f'train_dataset_{dataset_id}')
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        metadata = pd.read_csv(metadata_path)

        # Sample a few files to get feature names
        for filename in metadata['filename'].head(10):
            tsv_path = os.path.join(dataset_path, filename)
            features, _ = extract_features_from_repertoire(tsv_path)
            all_feature_names_set.update(features.keys())

    all_feature_names = sorted(list(all_feature_names_set))
    print(f"✓ Found {len(all_feature_names)} unique features")

    # Second pass: load all data with standardized features
    print("\n📥 Phase 2: Loading datasets with ESM-2 extraction...")
    all_data = []

    for dataset_id in range(1, 9):
        dataset_path = os.path.join(train_root, f'train_dataset_{dataset_id}')
        dataset_data = load_dataset(dataset_path, dataset_id, esm_extractor, all_feature_names)
        all_data.extend(dataset_data)

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Total loaded: {len(all_data)} repertoires across 8 datasets")
    return all_data, all_feature_names


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        esm_emb = batch['esm_embeddings'].to(device)
        trad_feat = batch['trad_features'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(esm_emb, trad_feat, masks)
        logits = logits.squeeze()

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            esm_emb = batch['esm_embeddings'].to(device)
            trad_feat = batch['trad_features'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits, _ = model(esm_emb, trad_feat, masks)
            logits = logits.squeeze()

            # Compute loss
            loss = criterion(logits, labels)

            # Record metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc, all_preds


def train_one_fold(fold_id: int, train_data: List[Dict], val_data: List[Dict],
                   trad_dim: int, device: str, epochs: int = 25):
    """Train model for one fold of leave-one-dataset-out CV."""
    print(f"\n{'='*70}")
    print(f"🎯 TRAINING FOLD {fold_id}/8")
    print(f"{'='*70}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create datasets and dataloaders
    train_dataset = RepertoireDataset(train_data)
    val_dataset = RepertoireDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=collate_repertoires, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                           collate_fn=collate_repertoires, num_workers=4)

    # Initialize model
    model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim,
                                   hidden_dims=[512, 256], dropout=0.3).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_auc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Train
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

        # Validate
        val_loss, val_auc, _ = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save model
            os.makedirs('./models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc,
                'epoch': epoch
            }, f'./models/championship_fold{fold_id}.pt')
            print(f"✓ New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n✅ Fold {fold_id} complete. Best Val AUC: {best_auc:.4f}")

    # Load best model
    checkpoint = torch.load(f'./models/championship_fold{fold_id}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, best_auc


# ============================================================================
# Main Championship Pipeline
# ============================================================================

def main():
    """Championship training pipeline."""

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship Deep Learning Pipeline 🏆      ║
    ║                                                              ║
    ║  Target: Beat GROZD (0.81364) → Achieve 0.82+               ║
    ║  Method: ESM-2 + Attention + Hybrid Features                ║
    ║  GPU: RTX 5080 16GB                                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Paths
    train_root = './data/train_datasets/train_datasets'
    test_root = './data/test_datasets/test_datasets'

    # Initialize ESM-2 extractor
    print("\n🔧 Initializing ESM-2 Feature Extractor...")
    esm_extractor = ESM2FeatureExtractor(model_name="esm2_t33_650M_UR50D", device=device)

    # Load all training data
    all_data, all_feature_names = load_all_training_data(train_root, esm_extractor)
    trad_dim = len(all_feature_names)
    print(f"\n📊 Traditional feature dimension: {trad_dim}")

    # Leave-one-dataset-out cross-validation
    print("\n" + "="*70)
    print("🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
    print("="*70)

    fold_results = []
    trained_models = []

    for test_dataset_id in range(1, 9):
        # Split data
        train_data = [d for d in all_data if d['dataset_id'] != test_dataset_id]
        val_data = [d for d in all_data if d['dataset_id'] == test_dataset_id]

        # Train fold
        model, val_auc = train_one_fold(
            fold_id=test_dataset_id,
            train_data=train_data,
            val_data=val_data,
            trad_dim=trad_dim,
            device=device,
            epochs=25
        )

        fold_results.append({'fold': test_dataset_id, 'val_auc': val_auc})
        trained_models.append(model)

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Print CV results
    print("\n" + "="*70)
    print("📈 CROSS-VALIDATION RESULTS")
    print("="*70)
    for result in fold_results:
        print(f"Fold {result['fold']}: Val AUC = {result['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    print(f"\n🎯 Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    print("\n✅ Training complete! Ready for inference on test datasets.")
    print("   Next step: Implement generate_predictions() to create submission file.")


if __name__ == '__main__':
    main()
