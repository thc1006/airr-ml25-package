#!/usr/bin/env python3
"""
BREAKTHROUGH: PyTorch Attention MIL on ESM-2 Embeddings
========================================================
True GPU deep learning - NOT tree-based models!

Key innovations:
1. Uses FULL 1280-dim ESM-2 embeddings (not reduced to 50 dims)
2. Attention MIL learns which sequences matter
3. Per-dataset training (not LODO)
4. GPU-accelerated training with PyTorch

Target: > 0.74006 (beat V5)
"""

import os
import gc
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    TRAIN_ROOT = Path('/app/data/train_datasets')
    TEST_ROOT = Path('/app/data/test_datasets')
    OUTPUT_DIR = Path('/app/outputs/submissions')
    CACHE_GB10 = Path('/app/cache/esm2_gb10')

    # Model architecture
    EMB_DIM = 1280
    HIDDEN_DIM = 256
    ATTENTION_DIM = 128
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 100
    PATIENCE = 15
    N_SPLITS = 5
    RANDOM_STATE = 42

    # Per-dataset class weights
    SCALE_POS_WEIGHT = {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
        7: 5.0, 8: 2.0
    }

    TOP_K_SEQUENCES = 50000


# ============================================================================
# Attention MIL Model
# ============================================================================
class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning.

    Input: (batch, n_instances, emb_dim) - ESM-2 embeddings per repertoire
    Output: (batch, 1) - disease probability
    """

    def __init__(self, emb_dim: int = 1280, hidden_dim: int = 256,
                 attention_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        # Instance-level transformation
        self.instance_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gated Attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1)

        # Bag-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (batch, n_instances, emb_dim)
        Returns:
            logits: (batch, 1)
            attention_weights: (batch, n_instances) if return_attention
        """
        # Encode instances
        h = self.instance_encoder(x)  # (batch, n_instances, hidden_dim)

        # Gated attention
        A_V = self.attention_V(h)  # (batch, n_instances, attention_dim)
        A_U = self.attention_U(h)  # (batch, n_instances, attention_dim)
        A = self.attention_w(A_V * A_U)  # (batch, n_instances, 1)
        A = A.squeeze(-1)  # (batch, n_instances)

        # Softmax attention weights
        attention_weights = F.softmax(A, dim=1)  # (batch, n_instances)

        # Weighted aggregation
        M = torch.bmm(attention_weights.unsqueeze(1), h)  # (batch, 1, hidden_dim)
        M = M.squeeze(1)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(M)  # (batch, 1)

        if return_attention:
            return logits, attention_weights
        return logits


# ============================================================================
# Dataset
# ============================================================================
class RepertoireDataset(Dataset):
    """Dataset for repertoire-level classification."""

    def __init__(self, embeddings: Dict[str, np.ndarray], labels: Dict[str, int],
                 max_instances: int = 1000):
        self.rep_ids = list(embeddings.keys())
        self.embeddings = embeddings
        self.labels = labels
        self.max_instances = max_instances

    def __len__(self):
        return len(self.rep_ids)

    def __getitem__(self, idx):
        rep_id = self.rep_ids[idx]
        emb = self.embeddings[rep_id]
        label = self.labels.get(rep_id, 0)

        # Pad or truncate
        if len(emb) > self.max_instances:
            # Random sample
            indices = np.random.choice(len(emb), self.max_instances, replace=False)
            emb = emb[indices]
        elif len(emb) < self.max_instances:
            # Pad with zeros
            pad = np.zeros((self.max_instances - len(emb), emb.shape[1]), dtype=np.float32)
            emb = np.vstack([emb, pad])

        return {
            'rep_id': rep_id,
            'embeddings': torch.from_numpy(emb.astype(np.float32)),
            'label': torch.tensor(label, dtype=torch.float32),
        }


# ============================================================================
# Training
# ============================================================================
class AttentionMILTrainer:
    """Per-dataset Attention MIL trainer."""

    def __init__(self, device: str = 'cuda', random_state: int = 42):
        self.device = device
        self.random_state = random_state

    def train_dataset(self, embeddings: Dict[str, np.ndarray],
                      labels: Dict[str, int], ds_id: int) -> Tuple[float, nn.Module]:
        """Train model for a single dataset with cross-validation."""

        # Prepare data
        rep_ids = list(embeddings.keys())
        y = np.array([labels.get(rid, 0) for rid in rep_ids])

        pos_weight = torch.tensor([Config.SCALE_POS_WEIGHT.get(ds_id, 1.0)]).to(self.device)

        pos = int(y.sum())
        neg = len(y) - pos
        min_class = max(2, min(pos, neg))
        n_splits = min(Config.N_SPLITS, min_class)

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_preds = np.zeros(len(y))
        cv_scores = []
        best_model = None
        best_score = 0

        for fold, (train_idx, val_idx) in enumerate(kf.split(rep_ids, y)):
            train_reps = [rep_ids[i] for i in train_idx]
            val_reps = [rep_ids[i] for i in val_idx]

            train_emb = {rid: embeddings[rid] for rid in train_reps}
            val_emb = {rid: embeddings[rid] for rid in val_reps}
            train_labels = {rid: labels.get(rid, 0) for rid in train_reps}
            val_labels = {rid: labels.get(rid, 0) for rid in val_reps}

            train_ds = RepertoireDataset(train_emb, train_labels)
            val_ds = RepertoireDataset(val_emb, val_labels)

            train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                                      shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                                    shuffle=False, num_workers=0)

            # Model
            model = GatedAttentionMIL(
                emb_dim=Config.EMB_DIM,
                hidden_dim=Config.HIDDEN_DIM,
                attention_dim=Config.ATTENTION_DIM,
                dropout=Config.DROPOUT,
            ).to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                          weight_decay=Config.WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.N_EPOCHS)

            best_val_auc = 0
            patience_counter = 0

            for epoch in range(Config.N_EPOCHS):
                # Train
                model.train()
                train_loss = 0
                for batch in train_loader:
                    emb = batch['embeddings'].to(self.device)
                    y_batch = batch['label'].to(self.device).unsqueeze(1)

                    optimizer.zero_grad()
                    logits = model(emb)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                scheduler.step()

                # Validate
                model.eval()
                val_preds = []
                val_labels_list = []
                with torch.no_grad():
                    for batch in val_loader:
                        emb = batch['embeddings'].to(self.device)
                        logits = model(emb)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        val_preds.extend(probs.flatten())
                        val_labels_list.extend(batch['label'].numpy())

                val_auc = roc_auc_score(val_labels_list, val_preds)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= Config.PATIENCE:
                        break

            # Load best model for this fold
            model.load_state_dict(best_state)

            # Get OOF predictions
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    rep_ids_batch = batch['rep_id']
                    emb = batch['embeddings'].to(self.device)
                    logits = model(emb)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    for rid, prob in zip(rep_ids_batch, probs):
                        idx = rep_ids.index(rid)
                        oof_preds[idx] = prob

            fold_auc = best_val_auc
            cv_scores.append(fold_auc)

            if fold_auc > best_score:
                best_score = fold_auc
                best_model = model

        mean_auc = np.mean(cv_scores)

        # Retrain on all data for final model
        all_ds = RepertoireDataset(embeddings, labels)
        all_loader = DataLoader(all_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

        final_model = GatedAttentionMIL(
            emb_dim=Config.EMB_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            attention_dim=Config.ATTENTION_DIM,
            dropout=Config.DROPOUT,
        ).to(self.device)

        optimizer = torch.optim.AdamW(final_model.parameters(), lr=Config.LEARNING_RATE,
                                      weight_decay=Config.WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(50):  # Fewer epochs for final training
            final_model.train()
            for batch in all_loader:
                emb = batch['embeddings'].to(self.device)
                y_batch = batch['label'].to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                logits = final_model(emb)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        return mean_auc, final_model

    def predict(self, model: nn.Module, embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Predict on test data."""
        model.eval()
        predictions = {}

        for rep_id, emb in embeddings.items():
            # Pad or truncate
            if len(emb) > 1000:
                indices = np.random.choice(len(emb), 1000, replace=False)
                emb = emb[indices]
            elif len(emb) < 1000:
                pad = np.zeros((1000 - len(emb), emb.shape[1]), dtype=np.float32)
                emb = np.vstack([emb, pad])

            emb_tensor = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(emb_tensor)
                prob = torch.sigmoid(logits).item()

            predictions[rep_id] = prob

        return predictions


# ============================================================================
# Data Loading
# ============================================================================
def load_esm2_embeddings(ds_idx: int) -> Optional[Dict]:
    """Load ESM-2 GB10 embeddings for a dataset."""
    cache_file = Config.CACHE_GB10 / f'train_dataset_{ds_idx}_embeddings.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def load_test_embeddings(test_name: str) -> Optional[Dict]:
    """Load test embeddings."""
    # Try multiple possible naming conventions
    for pattern in [f'{test_name}_embeddings.pkl', f'{test_name.replace("test_", "")}_embeddings.pkl']:
        cache_file = Config.CACHE_GB10 / pattern
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    return None


def get_labels(ds_path: Path) -> Dict[str, int]:
    """Load labels from metadata."""
    meta_file = ds_path / 'metadata.csv'
    if meta_file.exists():
        meta = pd.read_csv(meta_file)
        return dict(zip(meta['repertoire_id'], meta['label_positive']))
    return {}


def identify_top_sequences(ds_path: Path, model: nn.Module, embeddings: Dict,
                           top_k: int = 50000, device: str = 'cuda') -> pd.DataFrame:
    """Identify top sequences using attention weights."""

    # Read all sequences
    meta = pd.read_csv(ds_path / 'metadata.csv')
    all_seqs = []

    for _, row in meta.iterrows():
        rep_id = row['repertoire_id']
        tsv_path = ds_path / row['filename']
        if tsv_path.exists():
            df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
            df = df.dropna(subset=['junction_aa']).head(1000)

            # Get attention weights for this repertoire
            if rep_id in embeddings:
                emb = embeddings[rep_id]
                if len(emb) > 1000:
                    emb = emb[:1000]
                elif len(emb) < 1000:
                    pad = np.zeros((1000 - len(emb), emb.shape[1]), dtype=np.float32)
                    emb = np.vstack([emb, pad])

                emb_tensor = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    _, attention = model(emb_tensor, return_attention=True)
                    attention = attention.cpu().numpy().flatten()[:len(df)]

                df['attention_score'] = attention[:len(df)] if len(attention) >= len(df) else \
                    list(attention) + [0.0] * (len(df) - len(attention))
                df['dataset'] = ds_path.name
                all_seqs.append(df)

    if not all_seqs:
        return pd.DataFrame()

    all_df = pd.concat(all_seqs, ignore_index=True)

    # Select top sequences by attention score
    all_df = all_df.nlargest(top_k, 'attention_score')

    result = []
    for i, row in all_df.iterrows():
        result.append({
            'ID': f"{ds_path.name}_seq_top_{len(result)+1}",
            'dataset': ds_path.name,
            'label_positive_probability': -999.0,
            'junction_aa': row['junction_aa'],
            'v_call': str(row.get('v_call', 'TRBV1'))[:20],
            'j_call': str(row.get('j_call', 'TRBJ1-1'))[:20],
        })

    return pd.DataFrame(result)


# ============================================================================
# Main
# ============================================================================
def main():
    print('=' * 70)
    print('BREAKTHROUGH: PyTorch Attention MIL on ESM-2')
    print('True GPU Deep Learning - Target > 0.74006')
    print('=' * 70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_sets = sorted([d.name for d in Config.TRAIN_ROOT.iterdir() if d.is_dir()])
    test_sets = sorted([d.name for d in Config.TEST_ROOT.iterdir() if d.is_dir()])

    print(f'\nTrain datasets: {train_sets}')
    print(f'Test datasets: {test_sets}')

    trainer = AttentionMILTrainer(device=device, random_state=Config.RANDOM_STATE)

    bundles = {}
    all_preds = []
    all_seqs = []

    # Train per dataset
    for ds_name in train_sets:
        ds_path = Config.TRAIN_ROOT / ds_name
        ds_id = int(ds_name.split('_')[-1])

        print(f'\n[{ds_name}] Training (id={ds_id})')

        # Load embeddings
        data = load_esm2_embeddings(ds_id)
        if data is None or 'embeddings' not in data:
            print(f'  [SKIP] No embeddings found')
            continue

        embeddings = data['embeddings']
        labels = get_labels(ds_path)

        print(f'  Repertoires: {len(embeddings)}')
        print(f'  Labels: {sum(labels.values())} positive / {len(labels) - sum(labels.values())} negative')

        # Train
        cv_auc, model = trainer.train_dataset(embeddings, labels, ds_id)
        print(f'  CV AUC: {cv_auc:.4f}')

        bundles[ds_name] = {
            'model': model,
            'score': cv_auc,
            'embeddings': embeddings,
        }

    # Calculate mean CV
    mean_cv = np.mean([b['score'] for b in bundles.values()])
    print(f'\n{"="*70}')
    print(f'MEAN CV AUC: {mean_cv:.4f}')
    print(f'Target: > 0.74006 (V5)')
    print(f'Gap: {0.74006 - mean_cv:+.4f}')
    print(f'{"="*70}')

    # Predict on test sets
    print('\n[TEST PREDICTIONS]')
    for test_name in test_sets:
        test_path = Config.TEST_ROOT / test_name

        # Map test to train dataset
        base_id = int(test_name.split('_')[2].split('_')[0])
        train_ds_name = f'train_dataset_{base_id}'

        if train_ds_name not in bundles:
            print(f'  {test_name}: No model (skip)')
            continue

        model = bundles[train_ds_name]['model']

        # Load test embeddings
        test_data = load_test_embeddings(test_name)
        if test_data is None:
            # Use dummy predictions
            test_files = list(test_path.glob('*.tsv'))
            for tsv in test_files:
                all_preds.append({
                    'ID': tsv.stem,
                    'dataset': test_name,
                    'label_positive_probability': 0.5,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0,
                })
            print(f'  {test_name}: {len(test_files)} (dummy predictions)')
            continue

        test_emb = test_data.get('embeddings', {})
        predictions = trainer.predict(model, test_emb)

        for rep_id, prob in predictions.items():
            all_preds.append({
                'ID': rep_id,
                'dataset': test_name,
                'label_positive_probability': prob,
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0,
            })

        print(f'  {test_name}: {len(predictions)} predictions')

    # Task B: Top sequences
    print('\n[TASK B: TOP SEQUENCES]')
    for ds_name in train_sets:
        ds_path = Config.TRAIN_ROOT / ds_name

        if ds_name not in bundles:
            continue

        model = bundles[ds_name]['model']
        embeddings = bundles[ds_name]['embeddings']

        seq_df = identify_top_sequences(ds_path, model, embeddings,
                                        Config.TOP_K_SEQUENCES, device)
        if len(seq_df) > 0:
            all_seqs.append(seq_df)
            print(f'  {ds_name}: {len(seq_df)} sequences')

    # Create submission
    print('\n[CREATING SUBMISSION]')
    task_a = pd.DataFrame(all_preds)
    task_b = pd.concat(all_seqs, ignore_index=True) if all_seqs else pd.DataFrame()

    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability',
                            'junction_aa', 'v_call', 'j_call']]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Config.OUTPUT_DIR / f'pytorch_attention_mil_{timestamp}.csv'
    submission.to_csv(output_path, index=False)

    latest_path = Config.OUTPUT_DIR / 'pytorch_attention_mil_latest.csv'
    submission.to_csv(latest_path, index=False)

    print(f'\nSubmission: {output_path}')
    print(f'Total rows: {len(submission)} (expected: 404213)')
    print(f'Task A: {len(task_a)}')
    print(f'Task B: {len(task_b)}')

    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)
    for ds_name, bundle in bundles.items():
        print(f"  {ds_name}: CV AUC = {bundle['score']:.4f}")
    print(f'\n  MEAN CV AUC: {mean_cv:.4f}')
    print(f'  Target: > 0.74006 (V5)')
    print(f'  Gap to 1st: {0.84590 - mean_cv:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
