#!/usr/bin/env python3
"""
Hybrid Submission Generator for AIRR-ML-25 Competition
=======================================================

Uses the best model for each dataset:
- DeepRC: Datasets 1, 2, 3, 4, 5, 8 (AUC: 0.55-0.67)
- XGBoost: Datasets 6, 7 (AUC: 0.69-0.74)

Test datasets mapping:
- test_dataset_1 -> train_dataset_1 (DeepRC)
- test_dataset_2 -> train_dataset_2 (DeepRC)
- test_dataset_3 -> train_dataset_3 (DeepRC)
- test_dataset_4 -> train_dataset_4 (DeepRC)
- test_dataset_5 -> train_dataset_5 (DeepRC)
- test_dataset_6 -> train_dataset_6 (XGBoost)
- test_dataset_7_1, 7_2 -> train_dataset_7 (XGBoost)
- test_dataset_8_1, 8_2, 8_3 -> train_dataset_8 (DeepRC)
"""

import os
import json
import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from tqdm import tqdm
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    TRAIN_ROOT = './data/train_datasets/train_datasets'
    TEST_ROOT = './data/test_datasets/test_datasets'
    DEEPRC_CHECKPOINT_DIR = './checkpoints_deeprc'
    XGB_CHECKPOINT_DIR = './checkpoints_xgb'
    FEATURE_NAMES_FILE = './checkpoints/feature_names.json'
    SUBMISSION_DIR = './submissions'

    # DeepRC model architecture (must match training)
    EMBEDDING_DIM = 64
    CNN_CHANNELS = [128, 256]
    CNN_KERNELS = [5, 9]
    ATTENTION_DIM = 128
    CLASSIFIER_DIMS = [256, 128]
    DROPOUT = 0.3
    MAX_SEQ_LENGTH = 30
    MAX_SEQS_PER_REP = 5000

    # Task B
    TOP_K_SEQUENCES = 50000

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model selection strategy based on CV results
    MODEL_SELECTION = {
        1: 'deeprc',  # DeepRC: 0.6129 vs XGB: 0.5286
        2: 'deeprc',  # DeepRC: 0.6317 vs XGB: 0.5451
        3: 'deeprc',  # DeepRC: 0.5509 vs XGB: 0.5463
        4: 'deeprc',  # DeepRC: 0.5788 vs XGB: 0.5614
        5: 'deeprc',  # DeepRC: 0.6654 vs XGB: 0.6026
        6: 'xgboost', # DeepRC: 0.5103 vs XGB: 0.7407 *** XGBoost much better
        7: 'xgboost', # DeepRC: 0.5802 vs XGB: 0.6930 *** XGBoost better
        8: 'deeprc',  # DeepRC: 0.6095 vs XGB: N/A
    }

config = Config()

# ============================================================================
# Amino Acid Encoding (for DeepRC) - Must match champion_deeprc.py
# ============================================================================
AA_VOCAB = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21, '*': 22, '-': 23, '.': 24,  # Special characters
}
VOCAB_SIZE = len(AA_VOCAB) + 1  # +1 for padding (0)

def encode_sequence(seq: str, max_len: int = 30) -> np.ndarray:
    """Encode amino acid sequence to integer array."""
    encoded = np.zeros(max_len, dtype=np.int32)
    for i, aa in enumerate(seq[:max_len]):
        encoded[i] = AA_VOCAB.get(aa.upper(), AA_VOCAB['X'])
    return encoded

def encode_sequences_batch(sequences: List[str], max_len: int = 30) -> np.ndarray:
    """Encode batch of sequences efficiently."""
    batch = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            batch[i, j] = AA_VOCAB.get(aa.upper(), AA_VOCAB['X'])
    return batch

# ============================================================================
# DeepRC Model Architecture - MUST MATCH champion_deeprc.py EXACTLY
# ============================================================================
class SequenceEncoder(nn.Module):
    """Encode CDR3 sequences using embedding + 1D-CNN."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 64,
                 cnn_channels: List[int] = [128, 256], cnn_kernels: List[int] = [5, 9],
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale 1D-CNN for capturing different motif sizes
        self.convs = nn.ModuleList()
        for channels, kernel in zip(cnn_channels, cnn_kernels):
            self.convs.append(nn.Sequential(
                nn.Conv1d(embed_dim, channels, kernel, padding=kernel//2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output_dim = sum(cnn_channels)

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            out = conv(embedded)
            # Global max pooling over sequence length
            pooled = F.adaptive_max_pool1d(out, 1).squeeze(-1)
            conv_outputs.append(pooled)

        # Concatenate multi-scale features
        combined = torch.cat(conv_outputs, dim=1)
        # Project
        output = self.projection(combined)
        return output

class AttentionAggregator(nn.Module):
    """Attention-based aggregation over sequences in a repertoire."""

    def __init__(self, input_dim: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, seq_embeddings: torch.Tensor,
                return_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores: (n_seqs, 1)
        attn_scores = self.attention(seq_embeddings)
        # Softmax over sequences: (n_seqs, 1)
        attn_weights = F.softmax(attn_scores, dim=0)
        # Weighted sum: (hidden_dim,)
        repertoire_embedding = (seq_embeddings * attn_weights).sum(dim=0)

        if return_weights:
            return repertoire_embedding, attn_weights.squeeze(-1)
        return repertoire_embedding, None

class RepertoireClassifier(nn.Module):
    """Full model: Sequence Encoder + Attention Aggregator + Classifier"""

    def __init__(self, embedding_dim: int, cnn_channels: List[int], cnn_kernels: List[int],
                 attention_dim: int, classifier_dims: List[int], dropout: float = 0.3):
        super().__init__()

        self.sequence_encoder = SequenceEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=embedding_dim,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            dropout=dropout
        )

        hidden_dim = self.sequence_encoder.output_dim

        self.attention_aggregator = AttentionAggregator(
            input_dim=hidden_dim,
            attention_dim=attention_dim
        )

        # MLP Classifier (using LayerNorm instead of BatchNorm for single-sample support)
        layers = []
        input_dim = hidden_dim
        for dim in classifier_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward_single_repertoire(self, sequences: torch.Tensor,
                                   return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode all sequences: (n_seqs, hidden_dim)
        seq_embeddings = self.sequence_encoder(sequences)
        # Aggregate with attention: (hidden_dim,)
        rep_embedding, attn_weights = self.attention_aggregator(
            seq_embeddings, return_weights=return_attention
        )
        # Classify: (1,) -> scalar
        logit = self.classifier(rep_embedding.unsqueeze(0)).squeeze()
        return logit, attn_weights

# ============================================================================
# Feature Extraction for XGBoost
# ============================================================================
def load_feature_names() -> List[str]:
    """Load feature names from checkpoint."""
    with open(config.FEATURE_NAMES_FILE, 'r') as f:
        return json.load(f)

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

    df_clean = df['junction_aa'].dropna()
    if len(df_clean) == 0:
        return features

    seq_counts = df_clean.value_counts()
    frequencies = seq_counts.values / seq_counts.sum()

    # Shannon entropy
    features['shannon_entropy'] = entropy(frequencies)

    # Gini-Simpson index
    features['gini_simpson'] = 1 - np.sum(frequencies ** 2)

    # D50
    cumsum = np.cumsum(np.sort(frequencies)[::-1])
    features['d50'] = np.sum(cumsum <= 0.5)

    # Clonality score
    max_entropy = np.log(len(seq_counts))
    if max_entropy > 0:
        features['clonality'] = 1 - (features['shannon_entropy'] / max_entropy)
    else:
        features['clonality'] = 0

    # CDR3 length statistics
    lengths = df_clean.str.len()
    features['mean_length'] = lengths.mean() if len(lengths) > 0 else 0.0
    features['std_length'] = lengths.std() if len(lengths) > 1 else 0.0
    features['min_length'] = lengths.min() if len(lengths) > 0 else 0.0
    features['max_length'] = lengths.max() if len(lengths) > 0 else 0.0
    features['top_clone_fraction'] = frequencies[0] if len(frequencies) > 0 else 0.0

    return features

def extract_all_traditional_features(tsv_path: str) -> Dict[str, float]:
    """Extract all traditional features from a repertoire file."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except:
        return {}

    features = {}
    features.update(extract_vj_features(df))
    features.update(extract_clonality_features(df))

    return features

def standardize_features(feature_dict: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """Standardize feature dictionary to fixed-length array."""
    feature_vector = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        feature_vector[i] = feature_dict.get(name, 0.0)
    return feature_vector

# ============================================================================
# Model Loading
# ============================================================================
def load_deeprc_model(train_dataset_id: int) -> RepertoireClassifier:
    """Load DeepRC model for a specific dataset."""
    model_path = Path(config.DEEPRC_CHECKPOINT_DIR) / f"train_dataset_{train_dataset_id}_model.pt"

    model = RepertoireClassifier(
        embedding_dim=config.EMBEDDING_DIM,
        cnn_channels=config.CNN_CHANNELS,
        cnn_kernels=config.CNN_KERNELS,
        attention_dim=config.ATTENTION_DIM,
        classifier_dims=config.CLASSIFIER_DIMS,
        dropout=0.0  # No dropout at inference
    )

    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()

    return model

def load_xgboost_model(train_dataset_id: int) -> Tuple[xgb.XGBClassifier, object]:
    """Load XGBoost model and scaler for a specific dataset."""
    model_path = Path(config.XGB_CHECKPOINT_DIR) / f"xgb_ds{train_dataset_id}.pkl"

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return data['final_model'], data['scaler']

# ============================================================================
# Prediction Functions
# ============================================================================
def predict_with_deeprc(model: RepertoireClassifier, test_dir: str, dataset_name: str) -> List[Dict]:
    """Generate predictions using DeepRC model."""
    test_path = Path(test_dir)
    tsv_files = sorted(test_path.glob('*.tsv'))
    predictions = []

    # Determine max sequences based on dataset
    if '7' in dataset_name or '8' in dataset_name:
        max_seqs = 8000
    else:
        max_seqs = config.MAX_SEQS_PER_REP

    with torch.no_grad():
        for tsv_file in tqdm(tsv_files, desc=f"  DeepRC {dataset_name}", leave=False):
            repertoire_id = tsv_file.stem

            try:
                df = pd.read_csv(tsv_file, sep='\t', usecols=['junction_aa'])
                sequences = df['junction_aa'].dropna().astype(str).tolist()

                if len(sequences) == 0:
                    prob = 0.5
                else:
                    if len(sequences) > max_seqs:
                        random.seed(42)
                        sequences = random.sample(sequences, max_seqs)

                    encoded = encode_sequences_batch(sequences, config.MAX_SEQ_LENGTH)
                    seq_tensor = torch.from_numpy(encoded).long().to(config.DEVICE)

                    logit, _ = model.forward_single_repertoire(seq_tensor)
                    prob = torch.sigmoid(logit).item()

                predictions.append({
                    'ID': repertoire_id,
                    'dataset': dataset_name,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

            except Exception as e:
                print(f"    Error {repertoire_id}: {e}")
                predictions.append({
                    'ID': repertoire_id,
                    'dataset': dataset_name,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

    return predictions

def predict_with_xgboost(model: xgb.XGBClassifier, scaler, test_dir: str,
                          dataset_name: str, feature_names: List[str]) -> List[Dict]:
    """Generate predictions using XGBoost model."""
    test_path = Path(test_dir)
    tsv_files = sorted(test_path.glob('*.tsv'))
    predictions = []

    for tsv_file in tqdm(tsv_files, desc=f"  XGBoost {dataset_name}", leave=False):
        repertoire_id = tsv_file.stem

        try:
            # Extract features
            feat_dict = extract_all_traditional_features(str(tsv_file))
            feat_vector = standardize_features(feat_dict, feature_names)

            # Scale and predict
            feat_scaled = scaler.transform(feat_vector.reshape(1, -1))
            prob = model.predict_proba(feat_scaled)[0, 1]

            predictions.append({
                'ID': repertoire_id,
                'dataset': dataset_name,
                'label_positive_probability': float(prob),
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

        except Exception as e:
            print(f"    Error {repertoire_id}: {e}")
            predictions.append({
                'ID': repertoire_id,
                'dataset': dataset_name,
                'label_positive_probability': 0.5,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

    return predictions

# ============================================================================
# Task B: Sequence Identification
# ============================================================================
def identify_important_sequences_deeprc(model: RepertoireClassifier, train_dir: str,
                                         dataset_name: str, top_k: int = 50000) -> pd.DataFrame:
    """Identify important sequences using DeepRC attention weights."""
    train_path = Path(train_dir)
    metadata_path = train_path / 'metadata.csv'

    if not metadata_path.exists():
        print(f"    Warning: metadata.csv not found for {dataset_name}")
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_path)
    positive_rows = metadata[metadata['label_positive'] == 1]

    all_sequences = []

    # Determine max sequences
    if '7' in dataset_name or '8' in dataset_name:
        max_seqs = 8000
    else:
        max_seqs = config.MAX_SEQS_PER_REP

    with torch.no_grad():
        for _, row in tqdm(positive_rows.iterrows(), desc=f"  TaskB {dataset_name}",
                           total=len(positive_rows), leave=False):
            tsv_path = train_path / row['filename']
            if not tsv_path.exists():
                continue

            try:
                df = pd.read_csv(tsv_path, sep='\t')
                required_cols = ['junction_aa', 'v_call', 'j_call']
                if not all(c in df.columns for c in required_cols):
                    continue

                df = df.dropna(subset=['junction_aa'])
                if len(df) == 0:
                    continue

                if len(df) > max_seqs:
                    df = df.sample(n=max_seqs, random_state=42)

                sequences = df['junction_aa'].astype(str).tolist()
                encoded = encode_sequences_batch(sequences, config.MAX_SEQ_LENGTH)
                seq_tensor = torch.from_numpy(encoded).long().to(config.DEVICE)

                _, attention = model.forward_single_repertoire(seq_tensor, return_attention=True)

                if attention is not None:
                    attention = attention.cpu().numpy()
                    for i, (_, seq_row) in enumerate(df.iterrows()):
                        all_sequences.append({
                            'junction_aa': seq_row['junction_aa'],
                            'v_call': seq_row['v_call'],
                            'j_call': seq_row['j_call'],
                            'attention': attention[i] if i < len(attention) else 0.0
                        })
            except Exception as e:
                continue

    if not all_sequences:
        return pd.DataFrame()

    seq_df = pd.DataFrame(all_sequences)

    # Group by sequence and sum attention weights
    grouped = seq_df.groupby(['junction_aa', 'v_call', 'j_call']).agg({
        'attention': 'sum'
    }).reset_index()

    # Get top k
    grouped = grouped.sort_values('attention', ascending=False).head(top_k)

    # Format for submission
    result = []
    for i, row in grouped.iterrows():
        result.append({
            'ID': f"{dataset_name}_seq_top_{len(result)+1}",
            'dataset': dataset_name,
            'label_positive_probability': -999.0,
            'junction_aa': row['junction_aa'],
            'v_call': row['v_call'],
            'j_call': row['j_call']
        })

    return pd.DataFrame(result)

def identify_important_sequences_xgboost(train_dir: str, dataset_name: str,
                                          top_k: int = 50000) -> pd.DataFrame:
    """Identify important sequences using frequency in positive samples."""
    train_path = Path(train_dir)
    metadata_path = train_path / 'metadata.csv'

    if not metadata_path.exists():
        return pd.DataFrame()

    metadata = pd.read_csv(metadata_path)
    positive_rows = metadata[metadata['label_positive'] == 1]
    negative_rows = metadata[metadata['label_positive'] == 0]

    pos_sequences = {}
    neg_sequences = {}

    # Count sequences in positive repertoires
    for _, row in positive_rows.iterrows():
        tsv_path = train_path / row['filename']
        if not tsv_path.exists():
            continue
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            if 'junction_aa' not in df.columns:
                continue
            for _, seq_row in df.dropna(subset=['junction_aa']).iterrows():
                key = (seq_row['junction_aa'],
                       seq_row.get('v_call', 'unknown'),
                       seq_row.get('j_call', 'unknown'))
                pos_sequences[key] = pos_sequences.get(key, 0) + 1
        except:
            continue

    # Count sequences in negative repertoires (for comparison)
    for _, row in negative_rows.iterrows():
        tsv_path = train_path / row['filename']
        if not tsv_path.exists():
            continue
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            if 'junction_aa' not in df.columns:
                continue
            for _, seq_row in df.dropna(subset=['junction_aa']).iterrows():
                key = (seq_row['junction_aa'],
                       seq_row.get('v_call', 'unknown'),
                       seq_row.get('j_call', 'unknown'))
                neg_sequences[key] = neg_sequences.get(key, 0) + 1
        except:
            continue

    # Calculate differential score (positive - negative ratio)
    scores = []
    for key, pos_count in pos_sequences.items():
        neg_count = neg_sequences.get(key, 0)
        # Higher score for sequences more frequent in positive repertoires
        score = pos_count - neg_count * 0.5
        scores.append((*key, score))

    # Sort and get top k
    scores.sort(key=lambda x: x[3], reverse=True)
    top_sequences = scores[:top_k]

    # Format for submission
    result = []
    for i, (junction_aa, v_call, j_call, _) in enumerate(top_sequences):
        result.append({
            'ID': f"{dataset_name}_seq_top_{i+1}",
            'dataset': dataset_name,
            'label_positive_probability': -999.0,
            'junction_aa': junction_aa,
            'v_call': v_call,
            'j_call': j_call
        })

    return pd.DataFrame(result)

# ============================================================================
# Main Pipeline
# ============================================================================
def generate_hybrid_submission():
    """Generate complete submission using hybrid model strategy."""
    print("=" * 70)
    print("HYBRID SUBMISSION GENERATOR")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Model selection: DeepRC(1-5,8), XGBoost(6,7)")
    print("=" * 70)

    # Create output directory
    Path(config.SUBMISSION_DIR).mkdir(parents=True, exist_ok=True)

    # Load feature names for XGBoost
    feature_names = load_feature_names()
    print(f"Loaded {len(feature_names)} feature names")

    # Test dataset mapping
    test_to_train = {
        'test_dataset_1': 1,
        'test_dataset_2': 2,
        'test_dataset_3': 3,
        'test_dataset_4': 4,
        'test_dataset_5': 5,
        'test_dataset_6': 6,
        'test_dataset_7_1': 7,
        'test_dataset_7_2': 7,
        'test_dataset_8_1': 8,
        'test_dataset_8_2': 8,
        'test_dataset_8_3': 8,
    }

    # Cache loaded models
    deeprc_models = {}
    xgb_models = {}

    # ============================================
    # Task A: Predictions
    # ============================================
    print("\n" + "=" * 70)
    print("TASK A: Generating Test Predictions")
    print("=" * 70)

    all_predictions = []
    test_root = Path(config.TEST_ROOT)

    for test_dir in sorted(test_root.iterdir()):
        if not test_dir.is_dir():
            continue

        test_name = test_dir.name
        train_id = test_to_train.get(test_name)

        if train_id is None:
            print(f"  Skipping unknown dataset: {test_name}")
            continue

        model_type = config.MODEL_SELECTION[train_id]

        if model_type == 'deeprc':
            # Load DeepRC model if not cached
            if train_id not in deeprc_models:
                print(f"\nLoading DeepRC model for dataset {train_id}...")
                deeprc_models[train_id] = load_deeprc_model(train_id)

            predictions = predict_with_deeprc(
                deeprc_models[train_id], str(test_dir), test_name
            )
        else:  # xgboost
            # Load XGBoost model if not cached
            if train_id not in xgb_models:
                print(f"\nLoading XGBoost model for dataset {train_id}...")
                xgb_models[train_id] = load_xgboost_model(train_id)

            model, scaler = xgb_models[train_id]
            predictions = predict_with_xgboost(
                model, scaler, str(test_dir), test_name, feature_names
            )

        all_predictions.extend(predictions)
        print(f"  {test_name}: {len(predictions)} predictions ({model_type})")

    task_a_df = pd.DataFrame(all_predictions)
    print(f"\nTotal Task A predictions: {len(task_a_df)}")

    # ============================================
    # Task B: Sequence Identification
    # ============================================
    print("\n" + "=" * 70)
    print("TASK B: Identifying Important Sequences")
    print("=" * 70)

    all_sequences = []
    train_root = Path(config.TRAIN_ROOT)

    for ds_id in range(1, 9):
        train_name = f"train_dataset_{ds_id}"
        train_dir = train_root / train_name

        if not train_dir.exists():
            print(f"  Skipping {train_name}: directory not found")
            continue

        model_type = config.MODEL_SELECTION[ds_id]

        if model_type == 'deeprc':
            if ds_id not in deeprc_models:
                deeprc_models[ds_id] = load_deeprc_model(ds_id)

            seq_df = identify_important_sequences_deeprc(
                deeprc_models[ds_id], str(train_dir), train_name
            )
        else:  # xgboost - use frequency-based identification
            seq_df = identify_important_sequences_xgboost(
                str(train_dir), train_name
            )

        if len(seq_df) > 0:
            all_sequences.append(seq_df)
            print(f"  {train_name}: {len(seq_df)} sequences ({model_type})")
        else:
            print(f"  {train_name}: No sequences found")

    if all_sequences:
        task_b_df = pd.concat(all_sequences, ignore_index=True)
    else:
        task_b_df = pd.DataFrame()

    print(f"\nTotal Task B sequences: {len(task_b_df)}")

    # ============================================
    # Combine and Save
    # ============================================
    print("\n" + "=" * 70)
    print("GENERATING FINAL SUBMISSION")
    print("=" * 70)

    # Combine
    final_df = pd.concat([task_a_df, task_b_df], ignore_index=True)

    # Ensure column order
    final_df = final_df[['ID', 'dataset', 'label_positive_probability',
                          'junction_aa', 'v_call', 'j_call']]

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = Path(config.SUBMISSION_DIR) / f"hybrid_submission_{timestamp}.csv"
    final_df.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")
    print(f"Total rows: {len(final_df)}")
    print(f"  Task A (predictions): {len(task_a_df)}")
    print(f"  Task B (sequences): {len(task_b_df)}")

    # Validation
    print("\n" + "-" * 40)
    print("Validation:")
    print(f"  Expected Task A: 4,213 rows")
    print(f"  Expected Task B: 400,000 rows (8 datasets x 50,000)")
    print(f"  Expected Total: 404,213 rows")
    print(f"  Actual Total: {len(final_df)} rows")

    if len(final_df) == 404213:
        print("  STATUS: PASS")
    else:
        print("  STATUS: MISMATCH - Review submission!")

    return submission_path


if __name__ == "__main__":
    submission_path = generate_hybrid_submission()
