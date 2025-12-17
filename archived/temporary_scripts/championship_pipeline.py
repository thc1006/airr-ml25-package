#!/usr/bin/env python3
"""
🏆 AIRR-ML-25 Championship Complete Pipeline

This script handles the complete workflow:
1. Load pre-extracted features from checkpoints
2. Train 8-fold leave-one-dataset-out models
3. Generate Task A predictions (4,213 test repertoires)
4. Generate Task B sequence identification (8 × 50,000 sequences)
5. Create and validate submission.csv (404,213 rows)

Target: Beat GROZD (0.81364) → Achieve 0.82+
"""

import os
import sys
import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import psutil
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR = './checkpoints'
MODELS_DIR = './models'
SUBMISSION_DIR = './submissions'
TRAIN_ROOT = './data/train_datasets/train_datasets'
TEST_ROOT = './data/test_datasets/test_datasets'
SAMPLE_SUBMISSION = './data/sample_submissions.csv'

# Test datasets mapping (11 test sets)
TEST_DATASETS = [
    'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
    'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
    'test_dataset_7_1', 'test_dataset_7_2',
    'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
]

# Training constants
MAX_SEQS_PER_REPERTOIRE = 500
BATCH_SIZE = 4
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-4

# Task B constants
TOP_K_SEQUENCES = 50000

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# Model Architecture (Copied from championship_dl.py)
# ============================================================================

class AttentionAggregator(nn.Module):
    """Multi-head attention aggregation over variable-length repertoires."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, sequence_embeddings: torch.Tensor, mask: torch.Tensor = None):
        batch_size = sequence_embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, attn_weights = self.attention(
            query=query,
            key=sequence_embeddings,
            value=sequence_embeddings,
            key_padding_mask=~mask if mask is not None else None
        )
        aggregated = self.norm(attn_output.squeeze(1))
        return aggregated, attn_weights


class ChampionshipClassifier(nn.Module):
    """Hybrid Deep Learning + Traditional Features Classifier."""

    def __init__(self, esm_dim: int = 1280, trad_dim: int = 100,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.3):
        super().__init__()

        self.attention = AttentionAggregator(
            input_dim=esm_dim,
            hidden_dim=256,
            num_heads=4
        )

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

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, esm_embeddings: torch.Tensor, trad_features: torch.Tensor,
                mask: torch.Tensor = None):
        aggregated_esm, attn_weights = self.attention(esm_embeddings, mask)
        combined = torch.cat([aggregated_esm, trad_features], dim=1)
        logits = self.mlp(combined)
        return logits, attn_weights


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class RepertoireDataset(Dataset):
    """Dataset for variable-length repertoires."""

    def __init__(self, repertoire_data: List[Dict]):
        self.data = repertoire_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_repertoires(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length repertoires."""
    max_len = max(item['esm_embeddings'].shape[0] for item in batch)
    esm_dim = batch[0]['esm_embeddings'].shape[1]
    trad_dim = batch[0]['trad_features'].shape[0]
    batch_size = len(batch)

    esm_padded = torch.zeros(batch_size, max_len, esm_dim)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    trad_features = torch.zeros(batch_size, trad_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['esm_embeddings'].shape[0]
        esm_padded[i, :seq_len] = torch.from_numpy(item['esm_embeddings'])
        masks[i, :seq_len] = True
        trad_features[i] = torch.from_numpy(item['trad_features'])
        labels[i] = item.get('label', 0)

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

def load_extraction_status() -> Dict:
    """Load extraction status."""
    status_file = os.path.join(CHECKPOINT_DIR, 'extraction_status.json')
    if not os.path.exists(status_file):
        return {'completed_datasets': [], 'timestamp': None}
    with open(status_file, 'r') as f:
        return json.load(f)


def load_feature_names() -> List[str]:
    """Load feature names."""
    feature_file = os.path.join(CHECKPOINT_DIR, 'feature_names.json')
    if not os.path.exists(feature_file):
        return []
    with open(feature_file, 'r') as f:
        return json.load(f)


def load_dataset_checkpoint(dataset_id: int) -> List[Dict]:
    """Load a single dataset's checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'dataset_{dataset_id}.npz')

    if not os.path.exists(checkpoint_path):
        return []

    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        dataset_data = []

        # Check format
        if 'processed_data' in data.keys():
            # Dataset 8 format
            processed_data = data['processed_data']
            for item in processed_data:
                dataset_data.append({
                    'esm_embeddings': item['esm_embeddings'],
                    'trad_features': item['trad_features'],
                    'label': int(item['label']),
                    'repertoire_id': str(item['repertoire_id']),
                    'dataset_id': int(item['dataset_id'])
                })
        else:
            # Standard format
            esm_embeddings = data['esm_embeddings']
            trad_features = data['trad_features']
            labels = data['labels']
            repertoire_ids = data['repertoire_ids']
            dataset_ids = data['dataset_ids']

            for i in range(len(labels)):
                dataset_data.append({
                    'esm_embeddings': esm_embeddings[i],
                    'trad_features': trad_features[i],
                    'label': int(labels[i]),
                    'repertoire_id': str(repertoire_ids[i]),
                    'dataset_id': int(dataset_ids[i])
                })

        return dataset_data
    except Exception as e:
        print(f"⚠️ Error loading dataset {dataset_id}: {e}")
        return []


def load_all_training_data() -> Tuple[List[Dict], List[str]]:
    """Load all training data from checkpoints."""
    print("\n" + "="*70)
    print("📊 LOADING TRAINING DATA FROM CHECKPOINTS")
    print("="*70)

    status = load_extraction_status()
    completed = status.get('completed_datasets', [])

    if len(completed) < 8:
        raise RuntimeError(f"Not all datasets extracted! Found: {completed}")

    all_data = []
    for dataset_id in sorted(completed):
        dataset_data = load_dataset_checkpoint(dataset_id)
        if len(dataset_data) > 0:
            print(f"   ✓ Dataset {dataset_id}: {len(dataset_data)} repertoires")
            all_data.extend(dataset_data)
        else:
            raise RuntimeError(f"Failed to load dataset {dataset_id}")

    feature_names = load_feature_names()

    print(f"\n✅ Total loaded: {len(all_data)} repertoires")
    print(f"   Feature dimension: {len(feature_names)}")

    return all_data, feature_names


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for batch in tqdm(dataloader, desc="Training", leave=False):
        esm_emb = batch['esm_embeddings'].to(device)
        trad_feat = batch['trad_features'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(esm_emb, trad_feat, masks)
                logits = logits.squeeze()
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(esm_emb, trad_feat, masks)
            logits = logits.squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_repertoire_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            esm_emb = batch['esm_embeddings'].to(device)
            trad_feat = batch['trad_features'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(esm_emb, trad_feat, masks)
            logits = logits.squeeze()
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_repertoire_ids.extend(batch['repertoire_ids'])

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc, all_preds, all_repertoire_ids


def train_fold(fold_id: int, train_data: List[Dict], val_data: List[Dict],
               trad_dim: int, device: str):
    """Train model for one fold."""
    print(f"\n{'='*60}")
    print(f"🎯 TRAINING FOLD {fold_id}/8")
    print(f"{'='*60}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Create dataloaders
    train_dataset = RepertoireDataset(train_data)
    val_dataset = RepertoireDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_repertoires, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_repertoires, num_workers=2)

    # Initialize model
    model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim,
                                   hidden_dims=[512, 256], dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_auc = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        print(f"Epoch {epoch+1:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'val_auc': val_auc,
                'trad_dim': trad_dim
            }, f'{MODELS_DIR}/fold{fold_id}.pt')
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    print(f"✅ Fold {fold_id} best AUC: {best_auc:.4f}")

    # Load best model
    checkpoint = torch.load(f'{MODELS_DIR}/fold{fold_id}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, best_auc


def train_all_folds(all_data: List[Dict], trad_dim: int) -> Dict:
    """Train all 8 folds with leave-one-dataset-out CV."""
    print("\n" + "="*70)
    print("🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
    print("="*70)

    fold_results = []

    for test_dataset_id in range(1, 9):
        train_data = [d for d in all_data if d['dataset_id'] != test_dataset_id]
        val_data = [d for d in all_data if d['dataset_id'] == test_dataset_id]

        model, val_auc = train_fold(
            fold_id=test_dataset_id,
            train_data=train_data,
            val_data=val_data,
            trad_dim=trad_dim,
            device=device
        )

        fold_results.append({'fold': test_dataset_id, 'val_auc': val_auc})

        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "="*70)
    print("📈 CROSS-VALIDATION RESULTS")
    print("="*70)
    for result in fold_results:
        print(f"Fold {result['fold']}: Val AUC = {result['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    print(f"\n🎯 Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    return {
        'fold_results': fold_results,
        'mean_auc': mean_auc,
        'std_auc': std_auc
    }


# ============================================================================
# ESM-2 Feature Extraction for Test Data
# ============================================================================

class ESM2FeatureExtractor:
    """Extract sequence embeddings using ESM-2."""

    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda"):
        self.device = device
        print(f"Loading ESM-2 model: {model_name}...")

        import esm
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print(f"✓ ESM-2 loaded on {device}")

    def extract_embeddings(self, sequences: List[str], batch_size: int = 16,
                          max_seqs: int = 500) -> np.ndarray:
        """Extract sequence embeddings."""
        if len(sequences) > max_seqs:
            np.random.seed(42)
            indices = np.random.choice(len(sequences), max_seqs, replace=False)
            sequences = [sequences[i] for i in sorted(indices)]

        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        cleaned_sequences = []
        for seq in sequences:
            cleaned_seq = ''.join(c if c in valid_aa else 'X' for c in seq.upper())
            if len(cleaned_seq) > 0:
                cleaned_sequences.append(cleaned_seq)

        if len(cleaned_sequences) == 0:
            return np.zeros((1, 1280))

        sequences = cleaned_sequences
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                batch_labels = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_labels)
                batch_tokens = batch_tokens.to(self.device)

                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]

                for j, seq_len in enumerate([len(s) for s in batch_seqs]):
                    seq_repr = token_representations[j, 1:seq_len+1].mean(0)
                    embeddings.append(seq_repr.cpu().numpy())

                del batch_tokens, results, token_representations
                torch.cuda.empty_cache()

        return np.array(embeddings)


# ============================================================================
# Traditional Feature Extraction
# ============================================================================

def extract_vj_features(df: pd.DataFrame, top_n: int = 50) -> Dict[str, float]:
    """Extract V/J gene usage patterns."""
    features = {}
    total = len(df)

    if total == 0:
        return features

    if 'v_call' in df.columns:
        v_counts = df['v_call'].value_counts().head(top_n)
        for gene, count in v_counts.items():
            if pd.notna(gene):
                features[f"v_{gene}"] = count / total

    if 'j_call' in df.columns:
        j_counts = df['j_call'].value_counts().head(top_n)
        for gene, count in j_counts.items():
            if pd.notna(gene):
                features[f"j_{gene}"] = count / total

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

    features['shannon_entropy'] = entropy(frequencies)
    features['gini_simpson'] = 1 - np.sum(frequencies ** 2)

    cumsum = np.cumsum(np.sort(frequencies)[::-1])
    features['d50'] = np.sum(cumsum <= 0.5)

    max_entropy = np.log(len(seq_counts))
    if max_entropy > 0:
        features['clonality'] = 1 - (features['shannon_entropy'] / max_entropy)
    else:
        features['clonality'] = 0

    lengths = df_clean.str.len()
    features['mean_length'] = lengths.mean() if len(lengths) > 0 else 0.0
    features['std_length'] = lengths.std() if len(lengths) > 1 else 0.0
    features['min_length'] = lengths.min() if len(lengths) > 0 else 0.0
    features['max_length'] = lengths.max() if len(lengths) > 0 else 0.0
    features['top_clone_freq'] = frequencies[0] if len(frequencies) > 0 else 0.0

    features = {k: (0.0 if pd.isna(v) or np.isinf(v) else float(v)) for k, v in features.items()}

    return features


def standardize_features(feature_dict: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """Convert feature dict to standardized array."""
    feature_vector = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        if name in feature_dict:
            val = feature_dict[name]
            if pd.isna(val) or np.isinf(val):
                feature_vector[i] = 0.0
            else:
                feature_vector[i] = float(val)
    return feature_vector


# ============================================================================
# Task A: Test Predictions
# ============================================================================

def predict_test_repertoires(esm_extractor, feature_names: List[str], trad_dim: int) -> pd.DataFrame:
    """Generate predictions for all test repertoires."""
    print("\n" + "="*70)
    print("🔮 TASK A: GENERATING TEST PREDICTIONS")
    print("="*70)

    all_predictions = []

    for test_dataset in TEST_DATASETS:
        print(f"\n📂 Processing {test_dataset}...")
        test_path = os.path.join(TEST_ROOT, test_dataset)

        # Determine which training dataset model to use
        # For test_dataset_X_Y, use model trained without dataset X
        if test_dataset.startswith('test_dataset_7'):
            model_fold = 7
        elif test_dataset.startswith('test_dataset_8'):
            model_fold = 8
        else:
            model_fold = int(test_dataset.split('_')[-1])

        # Load model
        model_path = f'{MODELS_DIR}/fold{model_fold}.pt'
        checkpoint = torch.load(model_path)
        model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim,
                                       hidden_dims=[512, 256], dropout=0.3).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"   Using model from fold {model_fold}")

        # Get list of TSV files
        tsv_files = list(Path(test_path).glob('*.tsv'))
        print(f"   Found {len(tsv_files)} repertoires")

        # Process each repertoire
        for tsv_file in tqdm(tsv_files, desc=f"   {test_dataset}"):
            repertoire_id = tsv_file.stem

            try:
                # Read repertoire
                df = pd.read_csv(tsv_file, sep='\t')

                # Extract traditional features
                vj_features = extract_vj_features(df)
                clonality_features = extract_clonality_features(df)
                all_features = {**vj_features, **clonality_features}
                trad_features = standardize_features(all_features, feature_names)

                # Get sequences for ESM-2
                sequences = df['junction_aa'].dropna().astype(str).tolist()
                if len(sequences) == 0:
                    prob = 0.5  # Default probability
                else:
                    # Extract ESM-2 embeddings
                    esm_embeddings = esm_extractor.extract_embeddings(
                        sequences, batch_size=16, max_seqs=MAX_SEQS_PER_REPERTOIRE
                    )

                    # Prepare batch
                    esm_tensor = torch.from_numpy(esm_embeddings).unsqueeze(0).float().to(device)
                    trad_tensor = torch.from_numpy(trad_features).unsqueeze(0).float().to(device)
                    mask = torch.ones(1, esm_embeddings.shape[0], dtype=torch.bool).to(device)

                    # Predict
                    with torch.no_grad():
                        logits, _ = model(esm_tensor, trad_tensor, mask)
                        prob = torch.sigmoid(logits).item()

                all_predictions.append({
                    'ID': repertoire_id,
                    'dataset': test_dataset,
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

            except Exception as e:
                print(f"      ⚠️ Error processing {repertoire_id}: {e}")
                all_predictions.append({
                    'ID': repertoire_id,
                    'dataset': test_dataset,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })

        gc.collect()
        torch.cuda.empty_cache()

    predictions_df = pd.DataFrame(all_predictions)
    print(f"\n✅ Generated {len(predictions_df)} Task A predictions")

    return predictions_df


# ============================================================================
# Task B: Sequence Identification
# ============================================================================

def identify_important_sequences(all_data: List[Dict], feature_names: List[str]) -> pd.DataFrame:
    """Identify top 50,000 important sequences per training dataset."""
    print("\n" + "="*70)
    print("🧬 TASK B: IDENTIFYING IMPORTANT SEQUENCES")
    print("="*70)

    all_sequences = []

    for dataset_id in range(1, 9):
        print(f"\n📂 Processing train_dataset_{dataset_id}...")

        # Load model for this dataset
        model_path = f'{MODELS_DIR}/fold{dataset_id}.pt'
        checkpoint = torch.load(model_path)
        trad_dim = checkpoint['trad_dim']
        model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim,
                                       hidden_dims=[512, 256], dropout=0.3).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get data for this dataset
        dataset_data = [d for d in all_data if d['dataset_id'] == dataset_id]
        print(f"   {len(dataset_data)} repertoires")

        # Collect all sequences with their importance scores
        sequence_scores = defaultdict(lambda: {'score': 0, 'count': 0, 'v_call': None, 'j_call': None})

        # Read original TSV files to get sequence details
        train_path = os.path.join(TRAIN_ROOT, f'train_dataset_{dataset_id}')
        metadata_path = os.path.join(train_path, 'metadata.csv')
        metadata = pd.read_csv(metadata_path)

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"   Dataset {dataset_id}"):
            tsv_path = os.path.join(train_path, row['filename'])
            label = 1 if row['label_positive'] else 0

            try:
                df = pd.read_csv(tsv_path, sep='\t')

                # For positive samples, sequences are more important
                weight = 2.0 if label == 1 else 0.5

                for _, seq_row in df.iterrows():
                    junction_aa = seq_row.get('junction_aa')
                    if pd.isna(junction_aa):
                        continue

                    junction_aa = str(junction_aa)
                    v_call = str(seq_row.get('v_call', '')) if pd.notna(seq_row.get('v_call')) else ''
                    j_call = str(seq_row.get('j_call', '')) if pd.notna(seq_row.get('j_call')) else ''

                    # Score based on frequency in positive vs negative samples
                    sequence_scores[junction_aa]['score'] += weight
                    sequence_scores[junction_aa]['count'] += 1
                    if v_call and not sequence_scores[junction_aa]['v_call']:
                        sequence_scores[junction_aa]['v_call'] = v_call
                    if j_call and not sequence_scores[junction_aa]['j_call']:
                        sequence_scores[junction_aa]['j_call'] = j_call

            except Exception as e:
                print(f"      ⚠️ Error processing {row['filename']}: {e}")
                continue

        # Sort by score and select top 50,000
        sorted_seqs = sorted(
            sequence_scores.items(),
            key=lambda x: (x[1]['score'], x[1]['count']),
            reverse=True
        )[:TOP_K_SEQUENCES]

        # Create rows for submission
        for rank, (junction_aa, info) in enumerate(sorted_seqs, 1):
            all_sequences.append({
                'ID': f'train_dataset_{dataset_id}_seq_top_{rank}',
                'dataset': f'train_dataset_{dataset_id}',
                'label_positive_probability': -999.0,
                'junction_aa': junction_aa,
                'v_call': info['v_call'] if info['v_call'] else 'TRBV20-1',  # Default if missing
                'j_call': info['j_call'] if info['j_call'] else 'TRBJ2-7'   # Default if missing
            })

        print(f"   ✓ Selected {len(sorted_seqs)} sequences")
        gc.collect()

    sequences_df = pd.DataFrame(all_sequences)
    print(f"\n✅ Generated {len(sequences_df)} Task B sequences")

    return sequences_df


# ============================================================================
# Submission Generation
# ============================================================================

def generate_submission(task_a_df: pd.DataFrame, task_b_df: pd.DataFrame) -> pd.DataFrame:
    """Combine Task A and Task B into final submission."""
    print("\n" + "="*70)
    print("📝 GENERATING SUBMISSION FILE")
    print("="*70)

    # Combine
    submission_df = pd.concat([task_a_df, task_b_df], ignore_index=True)

    # Ensure correct column order
    submission_df = submission_df[['ID', 'dataset', 'label_positive_probability',
                                    'junction_aa', 'v_call', 'j_call']]

    # Validate
    expected_rows = 4213 + 8 * 50000
    actual_rows = len(submission_df)

    print(f"   Expected rows: {expected_rows}")
    print(f"   Actual rows: {actual_rows}")

    if actual_rows != expected_rows:
        print(f"   ⚠️ Row count mismatch!")
    else:
        print(f"   ✓ Row count matches!")

    # Save
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(SUBMISSION_DIR, f'submission_{timestamp}.csv')
    submission_df.to_csv(output_path, index=False)

    print(f"\n💾 Saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    # Also save as latest
    latest_path = os.path.join(SUBMISSION_DIR, 'submission_latest.csv')
    submission_df.to_csv(latest_path, index=False)

    return submission_df


def validate_submission(submission_df: pd.DataFrame):
    """Validate submission format."""
    print("\n" + "="*70)
    print("✅ VALIDATING SUBMISSION")
    print("="*70)

    # Load sample submission
    sample_df = pd.read_csv(SAMPLE_SUBMISSION)

    # Check columns
    expected_cols = list(sample_df.columns)
    actual_cols = list(submission_df.columns)

    print(f"   Expected columns: {expected_cols}")
    print(f"   Actual columns: {actual_cols}")
    print(f"   Columns match: {expected_cols == actual_cols}")

    # Check Task A
    task_a_rows = submission_df[submission_df['junction_aa'] == -999.0]
    print(f"\n   Task A rows: {len(task_a_rows)}")

    # Check Task B
    task_b_rows = submission_df[submission_df['label_positive_probability'] == -999.0]
    print(f"   Task B rows: {len(task_b_rows)}")

    # Check for NaN
    nan_count = submission_df.isna().sum().sum()
    print(f"\n   NaN values: {nan_count}")

    # Check probability range
    probs = task_a_rows['label_positive_probability']
    print(f"   Prob min: {probs.min():.4f}")
    print(f"   Prob max: {probs.max():.4f}")
    print(f"   Prob mean: {probs.mean():.4f}")

    if nan_count == 0 and len(submission_df) == 404213:
        print("\n🎉 SUBMISSION VALIDATION PASSED!")
        return True
    else:
        print("\n❌ SUBMISSION VALIDATION FAILED!")
        return False


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Complete championship pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship Complete Pipeline 🏆               ║
    ║                                                                  ║
    ║  Target: Beat GROZD (0.81364) → Achieve 0.82+                   ║
    ║  Method: ESM-2 (650M) + Attention + Hybrid Features             ║
    ║  GPU: RTX 5080 16GB                                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Load training data
    all_data, feature_names = load_all_training_data()
    trad_dim = len(feature_names)

    # Step 2: Check for existing models or train
    models_exist = all(
        os.path.exists(f'{MODELS_DIR}/fold{i}.pt')
        for i in range(1, 9)
    )

    if models_exist:
        print("\n✅ All 8 fold models found! Skipping training.")
    else:
        print("\n🎓 Training models...")
        results = train_all_folds(all_data, trad_dim)

        # Save results
        with open(os.path.join(CHECKPOINT_DIR, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Step 3: Initialize ESM-2 for test predictions
    print("\n🔧 Initializing ESM-2 for test predictions...")
    esm_extractor = ESM2FeatureExtractor(device=str(device))

    # Step 4: Generate Task A predictions
    task_a_df = predict_test_repertoires(esm_extractor, feature_names, trad_dim)

    # Step 5: Generate Task B sequences
    task_b_df = identify_important_sequences(all_data, feature_names)

    # Step 6: Generate and validate submission
    submission_df = generate_submission(task_a_df, task_b_df)
    validate_submission(submission_df)

    print("\n" + "="*70)
    print("🏆 CHAMPIONSHIP PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nSubmission file: {SUBMISSION_DIR}/submission_latest.csv")
    print("\nTo submit to Kaggle:")
    print(f"  kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\")
    print(f"    -f {SUBMISSION_DIR}/submission_latest.csv -m 'Championship DL submission'")


if __name__ == '__main__':
    main()
