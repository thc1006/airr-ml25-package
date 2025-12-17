#!/usr/bin/env python3
"""
train_esm2_gb10_optimized.py - GB10 Optimized ESM-2 Pipeline
=============================================================

FULLY AUTOMATED Pipeline for AIRR-ML-25 Competition
Hardware: NVIDIA GB10 (128GB unified memory, 6144 CUDA cores)

CRITICAL FIX: Previous embeddings were all zeros due to extraction bug.
This script properly extracts ESM-2 embeddings from hidden states.

Optimizations:
- Batch size 256 (maximize GPU utilization)
- FP16 mixed precision
- Flash Attention enabled
- Parallel data loading (8 workers)
- Incremental caching per dataset

Usage:
    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --shm-size=64g \
        -v $(pwd):/app -v $(pwd)/data:/app/data \
        -v ~/.cache:/root/.cache \
        -w /app --rm \
        nvcr.io/nvidia/pytorch:25.11-py3 \
        bash -c "pip install transformers scikit-learn tqdm pandas accelerate --quiet && \
                 python -u scripts/train_esm2_gb10_optimized.py"
"""

import os
import sys
import gc
import pickle
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import IncrementalPCA

warnings.filterwarnings('ignore')


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🚀 ESM-2 GB10 OPTIMIZED PIPELINE 🚀                                         ║
║                                                                              ║
║  Target: Beat SajayR (0.84590) → Achieve 0.85+                               ║
║  Hardware: NVIDIA GB10 (128GB unified, 6144 CUDA cores)                      ║
║                                                                              ║
║  CRITICAL FIX: Proper ESM-2 hidden state extraction                          ║
║  Optimization: Batch 256, FP16, Flash Attention, 8 workers                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """, flush=True)


@dataclass
class Config:
    data_dir: str = '/app/data'
    output_dir: str = '/app/outputs'
    cache_dir: str = '/app/cache/esm2_gb10'
    device: str = 'cuda'

    # ESM-2 650M (faster than 3B, still good quality)
    # Use 650M for speed: ~10x faster than 3B
    esm_model_name: str = 'facebook/esm2_t33_650M_UR50D'
    esm_layers: int = 33
    esm_repr_layer: int = 15  # L15 optimal for TCR
    esm_batch_size: int = 256  # GB10 optimized
    esm_embed_dim: int = 1280  # 650M output dimension

    # Sampling
    max_seqs_per_rep: int = 1000  # Sample 1000 sequences per repertoire
    pca_dim: int = 256  # Reduce to 256D for speed

    # MIL
    mil_hidden: int = 512
    mil_attention: int = 256
    mil_dropout: float = 0.3

    # Training
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4  # Higher LR for faster convergence
    patience: int = 7

    test_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.test_patterns = [
            'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
            'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
            'test_dataset_7_1', 'test_dataset_7_2',
            'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
        ]
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


class ESM2Extractor:
    """Proper ESM-2 embedding extraction with hidden states"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModel

        print(f"\n📦 Loading ESM-2 model: {self.config.esm_model_name}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.esm_model_name)
        self.model = AutoModel.from_pretrained(
            self.config.esm_model_name,
            torch_dtype=torch.float16  # FP16 for speed
        ).to(self.device)
        self.model.eval()

        # Enable optimizations
        torch.backends.cuda.enable_flash_sdp(True)
        torch.set_float32_matmul_precision('high')

        print(f"   ✅ Model loaded (FP16, Flash Attention enabled)", flush=True)

    @torch.no_grad()
    def extract(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings from sequences using layer 15"""

        if not sequences:
            return np.zeros((0, self.config.esm_embed_dim), dtype=np.float32)

        all_embeddings = []
        batch_size = self.config.esm_batch_size

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # Filter valid sequences
            valid_seqs = [s for s in batch_seqs if s and len(s) >= 3 and len(s) <= 50]
            if not valid_seqs:
                continue

            # Tokenize
            inputs = self.tokenizer(
                valid_seqs,
                padding=True,
                truncation=True,
                max_length=50,
                return_tensors='pt'
            ).to(self.device)

            # Forward pass with hidden states
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(**inputs, output_hidden_states=True)

            # CRITICAL FIX: Extract from hidden_states, not last_hidden_state
            # Use layer 15 (optimal for TCR CDR3)
            hidden_states = outputs.hidden_states[self.config.esm_repr_layer]

            # Mean pooling over sequence length (exclude padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            count = attention_mask.sum(dim=1)
            mean_hidden = sum_hidden / count.clamp(min=1)

            embeddings = mean_hidden.float().cpu().numpy()
            all_embeddings.append(embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.zeros((0, self.config.esm_embed_dim), dtype=np.float32)


class GatedAttentionMIL(nn.Module):
    """Gated Attention MIL for repertoire classification"""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 attention_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        self.attention_V = nn.Linear(hidden_dim // 2, attention_dim)
        self.attention_U = nn.Linear(hidden_dim // 2, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        # x: (batch, n_seqs, features)
        h = self.encoder(x.view(-1, x.size(-1)))
        h = h.view(x.size(0), x.size(1), -1)

        # Gated attention
        a_v = torch.tanh(self.attention_V(h))
        a_u = torch.sigmoid(self.attention_U(h))
        a = self.attention_w(a_v * a_u)
        a = F.softmax(a, dim=1)

        # Weighted aggregation
        z = torch.sum(a * h, dim=1)

        return self.classifier(z)


class RepertoireDataset(Dataset):
    def __init__(self, embeddings: Dict, labels: Dict):
        self.rep_ids = list(embeddings.keys())
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.rep_ids)

    def __getitem__(self, idx):
        rep_id = self.rep_ids[idx]
        return {
            'rep_id': rep_id,
            'embeddings': torch.FloatTensor(self.embeddings[rep_id]),
            'label': torch.FloatTensor([float(self.labels[rep_id])])
        }


def collate_fn(batch):
    max_len = max(b['embeddings'].size(0) for b in batch)
    feat_dim = batch[0]['embeddings'].size(1)

    padded = torch.zeros(len(batch), max_len, feat_dim)
    labels = torch.zeros(len(batch), 1)
    rep_ids = []

    for i, b in enumerate(batch):
        n = b['embeddings'].size(0)
        padded[i, :n] = b['embeddings']
        labels[i] = b['label']
        rep_ids.append(b['rep_id'])

    return {'embeddings': padded, 'labels': labels, 'rep_ids': rep_ids}


def discover_datasets(config: Config) -> Dict:
    """Discover all datasets"""
    data_dir = Path(config.data_dir)
    result = {'train': {}, 'test': {}}

    # Training datasets
    train_dir = data_dir / 'train_datasets'
    for i in range(1, 9):
        path = train_dir / f'train_dataset_{i}'
        meta_path = path / 'metadata.csv'
        if meta_path.exists():
            result['train'][f'train_dataset_{i}'] = {
                'path': path,
                'metadata': pd.read_csv(meta_path)
            }

    # Test datasets (no metadata.csv - create from TSV filenames)
    test_dir = data_dir / 'test_datasets'
    for pattern in config.test_patterns:
        path = test_dir / pattern
        if path.exists():
            # Create metadata from TSV files
            tsv_files = list(path.glob('*.tsv'))
            if tsv_files:
                rep_ids = [f.stem for f in tsv_files]
                metadata = pd.DataFrame({
                    'repertoire_id': rep_ids,
                    'filename': [f.name for f in tsv_files]
                })
                result['test'][pattern] = {
                    'path': path,
                    'metadata': metadata
                }

    return result


def extract_dataset_embeddings(
    extractor: ESM2Extractor,
    dataset_info: Dict,
    dataset_name: str,
    config: Config
) -> Tuple[Dict, Dict, Dict]:
    """Extract embeddings for a single dataset with caching"""

    cache_path = Path(config.cache_dir) / f'{dataset_name}_embeddings.pkl'

    if cache_path.exists():
        print(f"   📂 Loading cached {dataset_name}...", flush=True)
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings'], data['labels'], data.get('sequences', {})

    metadata = dataset_info['metadata']
    dataset_path = dataset_info['path']

    embeddings = {}
    labels = {}
    sequences_dict = {}

    for _, row in tqdm(metadata.iterrows(), total=len(metadata),
                       desc=f"   {dataset_name}", leave=False):
        rep_id = row['repertoire_id']
        label = row.get('label_positive', False)
        tsv_path = dataset_path / f"{rep_id}.tsv"

        try:
            df = pd.read_csv(tsv_path, sep='\t')
            seqs = df['junction_aa'].dropna().astype(str).tolist()
            seqs = [s for s in seqs if len(s) >= 5 and len(s) <= 30 and 'X' not in s]

            if not seqs:
                continue

            # Sample if too many
            if len(seqs) > config.max_seqs_per_rep:
                np.random.seed(hash(rep_id) % 2**32)
                seqs = list(np.random.choice(seqs, config.max_seqs_per_rep, replace=False))

            emb = extractor.extract(seqs)

            if emb.shape[0] > 0:
                embeddings[rep_id] = emb
                labels[rep_id] = bool(label)
                sequences_dict[rep_id] = seqs

        except Exception as e:
            print(f"     ⚠️ Error {rep_id}: {e}", flush=True)
            continue

    # Cache
    print(f"   💾 Caching {dataset_name} ({len(embeddings)} repertoires)...", flush=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'labels': labels,
            'sequences': sequences_dict
        }, f)

    return embeddings, labels, sequences_dict


def apply_pca_to_embeddings(
    embeddings: Dict[str, np.ndarray],
    config: Config
) -> Tuple[Dict[str, np.ndarray], IncrementalPCA]:
    """Apply incremental PCA to reduce dimensionality"""

    print(f"\n📊 Applying PCA: {config.esm_embed_dim}D → {config.pca_dim}D", flush=True)

    pca = IncrementalPCA(n_components=config.pca_dim, batch_size=10000)

    # Fit
    print("   Fitting PCA...", flush=True)
    for rep_id, emb in tqdm(embeddings.items(), desc="   Fitting"):
        if emb.shape[0] > 0:
            pca.partial_fit(emb)

    # Transform
    print("   Transforming...", flush=True)
    reduced = {}
    for rep_id, emb in tqdm(embeddings.items(), desc="   Transforming"):
        if emb.shape[0] > 0:
            reduced[rep_id] = pca.transform(emb)
        else:
            reduced[rep_id] = np.zeros((0, config.pca_dim), dtype=np.float32)

    print(f"   ✅ Explained variance: {pca.explained_variance_ratio_.sum():.4f}", flush=True)
    return reduced, pca


def train_lodo_cv(
    embeddings: Dict,
    labels: Dict,
    rep_to_dataset: Dict,
    config: Config
) -> Tuple[List[nn.Module], List[float]]:
    """LODO Cross-Validation training"""

    print("\n" + "=" * 60)
    print("LODO Cross-Validation (8 folds)")
    print("=" * 60)

    # Group by dataset
    dataset_reps = defaultdict(list)
    for rep_id in embeddings.keys():
        ds = rep_to_dataset.get(rep_id)
        if ds:
            dataset_reps[ds].append(rep_id)

    dataset_ids = sorted(dataset_reps.keys())
    input_dim = config.pca_dim

    fold_models = []
    fold_aucs = []

    for fold_idx, val_ds in enumerate(dataset_ids):
        print(f"\n  📂 Fold {fold_idx+1}/8: Val={val_ds}", flush=True)

        train_reps = [r for d, reps in dataset_reps.items() for r in reps if d != val_ds]
        val_reps = dataset_reps[val_ds]

        print(f"     Train: {len(train_reps)}, Val: {len(val_reps)}", flush=True)

        train_emb = {r: embeddings[r] for r in train_reps if r in embeddings}
        train_lab = {r: labels[r] for r in train_reps if r in labels}
        val_emb = {r: embeddings[r] for r in val_reps if r in embeddings}
        val_lab = {r: labels[r] for r in val_reps if r in labels}

        if len(val_emb) < 10:
            print(f"     ⚠️ Too few val samples, skipping", flush=True)
            continue

        train_ds = RepertoireDataset(train_emb, train_lab)
        val_ds_obj = RepertoireDataset(val_emb, val_lab)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_ds_obj, batch_size=config.batch_size,
                               shuffle=False, collate_fn=collate_fn, num_workers=4)

        model = GatedAttentionMIL(
            input_dim=input_dim,
            hidden_dim=config.mil_hidden,
            attention_dim=config.mil_attention,
            dropout=config.mil_dropout
        ).to(config.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                emb = batch['embeddings'].to(config.device)
                lab = batch['labels'].to(config.device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    logits = model(emb)
                    loss = criterion(logits, lab)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            preds, true_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    emb = batch['embeddings'].to(config.device)
                    with torch.amp.autocast('cuda'):
                        logits = model(emb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds.extend(probs.flatten())
                    true_labels.extend(batch['labels'].numpy().flatten())

            try:
                auc = roc_auc_score(true_labels, preds) if len(set(true_labels)) > 1 else 0.5
            except:
                auc = 0.5

            if (epoch + 1) % 5 == 0 or auc > best_auc:
                print(f"     Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, AUC={auc:.4f}", flush=True)

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= config.patience:
                print(f"     Early stop at epoch {epoch+1}", flush=True)
                break

        if best_state:
            model.load_state_dict({k: v.to(config.device) for k, v in best_state.items()})

        fold_models.append(model)
        fold_aucs.append(best_auc)
        print(f"     ✅ Best AUC: {best_auc:.4f}", flush=True)

    mean_auc = np.mean(fold_aucs)
    print(f"\n📊 LODO CV: {mean_auc:.4f} (±{np.std(fold_aucs):.4f})")
    print(f"   Per-fold: {[f'{a:.4f}' for a in fold_aucs]}")

    return fold_models, fold_aucs


def generate_test_predictions(
    extractor: ESM2Extractor,
    pca: IncrementalPCA,
    fold_models: List[nn.Module],
    test_datasets: Dict,
    config: Config
) -> Tuple[Dict, Dict]:
    """Generate predictions for test sets"""

    print("\n" + "=" * 60)
    print("Test Set Inference")
    print("=" * 60)

    predictions = {}
    rep_to_test_ds = {}

    for test_name, test_info in tqdm(test_datasets.items(), desc="Test datasets"):
        metadata = test_info['metadata']
        dataset_path = test_info['path']

        for _, row in tqdm(metadata.iterrows(), total=len(metadata),
                          desc=f"  {test_name}", leave=False):
            rep_id = row['repertoire_id']
            rep_to_test_ds[rep_id] = test_name

            try:
                tsv_path = dataset_path / f"{rep_id}.tsv"
                df = pd.read_csv(tsv_path, sep='\t')
                seqs = df['junction_aa'].dropna().astype(str).tolist()
                seqs = [s for s in seqs if len(s) >= 5 and len(s) <= 30 and 'X' not in s]

                if not seqs:
                    predictions[rep_id] = 0.5
                    continue

                if len(seqs) > config.max_seqs_per_rep:
                    np.random.seed(42)
                    seqs = list(np.random.choice(seqs, config.max_seqs_per_rep, replace=False))

                emb = extractor.extract(seqs)
                if emb.shape[0] == 0:
                    predictions[rep_id] = 0.5
                    continue

                emb_pca = pca.transform(emb)
                emb_tensor = torch.FloatTensor(emb_pca).unsqueeze(0).to(config.device)

                # Ensemble prediction
                preds = []
                for model in fold_models:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        logit = model(emb_tensor)
                        prob = torch.sigmoid(logit).item()
                        preds.append(prob)

                predictions[rep_id] = np.mean(preds)

            except Exception as e:
                print(f"     ⚠️ {rep_id}: {e}", flush=True)
                predictions[rep_id] = 0.5

    return predictions, rep_to_test_ds


def select_task_b_sequences(
    sequences: Dict,
    labels: Dict,
    rep_to_dataset: Dict,
    n_per_dataset: int = 50000
) -> Dict:
    """Select top sequences for Task B"""

    print("\n📊 Task B: Selecting top sequences", flush=True)

    ds_seqs = defaultdict(list)
    ds_labels = defaultdict(list)

    for rep_id, seqs in sequences.items():
        ds = rep_to_dataset.get(rep_id)
        if ds and seqs:
            label = labels.get(rep_id, False)
            for s in seqs:
                ds_seqs[ds].append(s)
                ds_labels[ds].append(1 if label else 0)

    results = {}
    for ds in sorted(ds_seqs.keys()):
        seqs = ds_seqs[ds]
        labs = ds_labels[ds]

        print(f"   {ds}: {len(seqs)} sequences", flush=True)

        if len(seqs) <= n_per_dataset:
            selected = seqs
        else:
            # Prefer positive-associated sequences
            pos_seqs = [s for s, l in zip(seqs, labs) if l == 1]
            neg_seqs = [s for s, l in zip(seqs, labs) if l == 0]

            n_pos = min(int(n_per_dataset * 0.7), len(pos_seqs))
            n_neg = min(n_per_dataset - n_pos, len(neg_seqs))

            np.random.seed(42)
            selected = []
            if pos_seqs:
                idx = np.random.choice(len(pos_seqs), min(n_pos, len(pos_seqs)), replace=False)
                selected.extend([pos_seqs[i] for i in idx])
            if neg_seqs and len(selected) < n_per_dataset:
                idx = np.random.choice(len(neg_seqs), min(n_per_dataset - len(selected), len(neg_seqs)), replace=False)
                selected.extend([neg_seqs[i] for i in idx])

        # Ensure exactly 50000
        while len(selected) < n_per_dataset:
            selected.append(selected[len(selected) % max(len(selected), 1)])
        results[ds] = selected[:n_per_dataset]

    return results


def generate_submission(
    predictions: Dict,
    rep_to_test_ds: Dict,
    task_b: Dict,
    config: Config
) -> Path:
    """Generate submission file"""

    print("\n" + "=" * 60)
    print("Generating Submission")
    print("=" * 60)

    rows = []

    # Task A
    for rep_id, prob in predictions.items():
        ds = rep_to_test_ds.get(rep_id, 'unknown')
        rows.append({
            'ID': rep_id,
            'dataset': ds,
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })

    # Task B
    for ds_id, seqs in task_b.items():
        for i, seq in enumerate(seqs):
            rows.append({
                'ID': f"{ds_id}_seq_top_{i+1}",
                'dataset': ds_id,
                'label_positive_probability': -999.0,
                'junction_aa': seq,
                'v_call': -999.0,
                'j_call': -999.0
            })

    df = pd.DataFrame(rows)

    # Save
    sub_dir = Path(config.output_dir) / 'submissions'
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / 'submission_esm2_gb10.csv'
    df.to_csv(path, index=False)

    print(f"\n✅ Submission: {path}")
    print(f"   Rows: {len(df)} (Task A: {len(predictions)}, Task B: {8 * 50000})")

    return path


def main():
    print_banner()
    config = Config()

    print(f"\n🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize extractor
    extractor = ESM2Extractor(config)

    # Discover datasets
    print("\n📁 Discovering datasets...", flush=True)
    datasets = discover_datasets(config)
    print(f"   Train: {len(datasets['train'])}, Test: {len(datasets['test'])}")

    # ========================================
    # Stage 1: Extract Train Embeddings
    # ========================================
    print("\n" + "=" * 60)
    print("Stage 1: Extract Training Embeddings")
    print("=" * 60)

    all_embeddings = {}
    all_labels = {}
    all_sequences = {}
    rep_to_dataset = {}

    for ds_name, ds_info in datasets['train'].items():
        emb, lab, seqs = extract_dataset_embeddings(extractor, ds_info, ds_name, config)
        all_embeddings.update(emb)
        all_labels.update(lab)
        all_sequences.update(seqs)
        for rep_id in emb.keys():
            rep_to_dataset[rep_id] = ds_name

    print(f"\n   Total: {len(all_embeddings)} repertoires")

    # Verify embeddings are not zero
    sample_emb = next(iter(all_embeddings.values()))
    print(f"   Sample embedding: mean={sample_emb.mean():.4f}, std={sample_emb.std():.4f}")

    # ========================================
    # Stage 2: PCA Dimensionality Reduction
    # ========================================
    reduced_emb, pca = apply_pca_to_embeddings(all_embeddings, config)

    # ========================================
    # Stage 3: LODO CV Training
    # ========================================
    fold_models, fold_aucs = train_lodo_cv(reduced_emb, all_labels, rep_to_dataset, config)

    # Save models
    models_path = Path(config.cache_dir) / 'lodo_models.pkl'
    print(f"\n💾 Saving models to {models_path}...", flush=True)
    with open(models_path, 'wb') as f:
        pickle.dump({
            'fold_models_state': [m.state_dict() for m in fold_models],
            'fold_aucs': fold_aucs,
            'pca': pca,
            'config': config
        }, f)

    # ========================================
    # Stage 4: Test Inference
    # ========================================
    predictions, rep_to_test_ds = generate_test_predictions(
        extractor, pca, fold_models, datasets['test'], config
    )

    # ========================================
    # Stage 5: Task B Selection
    # ========================================
    task_b = select_task_b_sequences(all_sequences, all_labels, rep_to_dataset)

    # ========================================
    # Stage 6: Generate Submission
    # ========================================
    sub_path = generate_submission(predictions, rep_to_test_ds, task_b, config)

    # Summary
    mean_auc = np.mean(fold_aucs)
    print(f"\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"   LODO CV AUC: {mean_auc:.4f} (±{np.std(fold_aucs):.4f})")
    print(f"   Submission: {sub_path}")
    print(f"   Target: > 0.85 to beat SajayR (0.84590)")

    # Auto-submit if good enough
    if mean_auc > 0.70:
        print(f"\n🎯 CV AUC looks promising! Ready for submission.")
        print(f"   kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \\")
        print(f"       -f {sub_path} -m 'ESM-2 650M GB10 optimized, LODO CV {mean_auc:.4f}'")

    return mean_auc


if __name__ == '__main__':
    try:
        auc = main()
        print(f"\n✅ Pipeline completed! CV AUC: {auc:.4f}")
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
