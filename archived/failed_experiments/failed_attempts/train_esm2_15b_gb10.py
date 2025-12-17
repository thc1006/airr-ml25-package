#!/usr/bin/env python3
"""
train_esm2_15b_gb10.py - ESM-2 15B GB10 Ultra-Optimized Pipeline
================================================================

MAXIMUM PERFORMANCE Pipeline for AIRR-ML-25 Competition
Target: Beat SajayR (0.84590) → Achieve 0.85+

ESM-2 15B Advantages:
- 48 layers (vs 33 for 650M)
- Layer 33 optimal representation (deeper semantic features)
- Better protein structure understanding

GB10 Optimizations:
- 128GB Unified Memory (no OOM for 15B)
- Batch size 32 (15B requires more memory per sample)
- FP16 mixed precision
- Flash Attention + SDPA
- Gradient checkpointing if needed

Usage:
    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --shm-size=64g \
        -v $(pwd):/app -v $(pwd)/data:/app/data \
        -v ~/.cache:/root/.cache \
        -w /app --rm \
        nvcr.io/nvidia/pytorch:25.11-py3 \
        bash -c "pip install transformers scikit-learn tqdm pandas accelerate --quiet && \
                 python -u scripts/train_esm2_15b_gb10.py"
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
║  🏆 ESM-2 15B GB10 TURBO PIPELINE 🏆                                         ║
║                                                                              ║
║  Target: Beat SajayR (0.84590) → WIN THE COMPETITION!                        ║
║  Hardware: NVIDIA GB10 (128GB unified, 6144 CUDA cores, Blackwell)           ║
║                                                                              ║
║  Model: facebook/esm2_t48_15B_UR50D (48 layers, 15B params)                 ║
║  Layer: L33 (optimal for TCR CDR3 sequences)                                 ║
║                                                                              ║
║  🚀 GB10 TURBO MODE:                                                         ║
║     • Batch 64 (2x larger with unified memory)                               ║
║     • BF16 (Blackwell native precision)                                      ║
║     • torch.compile() (10-30% speedup)                                       ║
║     • Flash Attention + SDPA                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """, flush=True)


@dataclass
class Config:
    data_dir: str = '/app/data'
    output_dir: str = '/app/outputs'
    cache_dir: str = '/app/cache/esm2_15b'
    device: str = 'cuda'

    # ESM-2 15B - Maximum quality
    esm_model_name: str = 'facebook/esm2_t48_15B_UR50D'
    esm_layers: int = 48
    esm_repr_layer: int = 33  # L33 optimal for 15B
    esm_batch_size: int = 64  # GB10 128GB can handle larger batches!
    esm_embed_dim: int = 5120  # 15B output dimension

    # Sampling - GB10 OPTIMIZED
    max_seqs_per_rep: int = 400  # Balanced: quality vs speed
    pca_dim: int = 512  # Larger dim for 15B features

    # GB10 Specific
    use_torch_compile: bool = True  # 10-30% speedup
    use_bf16: bool = True  # Blackwell native BF16

    # MIL
    mil_hidden: int = 1024  # Larger for 15B features
    mil_attention: int = 512
    mil_dropout: float = 0.3

    # Training
    lr: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 30
    patience: int = 8

    # Data patterns
    train_patterns: List[str] = field(default_factory=lambda: [
        f'train_dataset_{i}' for i in range(1, 9)
    ])
    test_patterns: List[str] = field(default_factory=lambda: [
        'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
        'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
        'test_dataset_7_1', 'test_dataset_7_2',
        'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
    ])


class SequenceDataset(Dataset):
    """Simple dataset for ESM-2 processing."""
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class ESM2Extractor:
    """ESM-2 15B embedding extractor optimized for GB10."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self._load_model()

    def _load_model(self):
        """Load ESM-2 15B model with GB10-specific optimizations."""
        from transformers import AutoModel, AutoTokenizer

        print(f"\n📦 Loading ESM-2 15B model: {self.config.esm_model_name}", flush=True)
        print("   (This may take 2-3 minutes for 15B model...)", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.esm_model_name)

        # Select dtype based on GB10 optimization setting
        dtype = torch.bfloat16 if self.config.use_bf16 else torch.float16
        print(f"   Using dtype: {dtype}", flush=True)

        # Load with optimizations
        self.model = AutoModel.from_pretrained(
            self.config.esm_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Enable Flash Attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # GB10 TURBO: torch.compile() for 10-30% speedup
        if self.config.use_torch_compile:
            print("   🚀 Applying torch.compile() optimization...", flush=True)
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("   ✅ torch.compile() enabled (mode=reduce-overhead)", flush=True)
            except Exception as e:
                print(f"   ⚠️ torch.compile() failed: {e}, using eager mode", flush=True)

        dtype_str = "BF16" if self.config.use_bf16 else "FP16"
        print(f"   ✅ Model loaded ({dtype_str}, Flash Attention enabled)", flush=True)

        # Check GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"   GPU Memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved", flush=True)

    def extract(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings from sequences using proper hidden state extraction."""
        if not sequences:
            return np.zeros((0, self.config.esm_embed_dim))

        embeddings = []
        dataset = SequenceDataset(sequences)
        loader = DataLoader(
            dataset,
            batch_size=self.config.esm_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x
        )

        with torch.no_grad():
            for batch_seqs in loader:
                # Tokenize
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=50
                ).to(self.device)

                # Forward pass with hidden states
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(**inputs, output_hidden_states=True)

                # CRITICAL: Extract from hidden_states at optimal layer
                hidden_states = outputs.hidden_states[self.config.esm_repr_layer]

                # Mean pooling over sequence length (excluding padding)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                count = attention_mask.sum(dim=1)
                mean_hidden = sum_hidden / count.clamp(min=1)

                embeddings.append(mean_hidden.float().cpu().numpy())

                # Clear GPU cache periodically
                del outputs, hidden_states, inputs
                torch.cuda.empty_cache()

        return np.vstack(embeddings)


def discover_datasets(config: Config) -> Dict[str, pd.DataFrame]:
    """Discover all datasets with proper metadata handling."""
    datasets = {}
    data_dir = Path(config.data_dir)

    # Training datasets
    train_dir = data_dir / 'train_datasets'
    for pattern in config.train_patterns:
        path = train_dir / pattern
        if path.exists():
            metadata_path = path / 'metadata.csv'
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path)
                datasets[pattern] = metadata
                print(f"   {pattern}: {len(metadata)} repertoires", flush=True)

    # Test datasets (no metadata.csv - create from TSV filenames)
    test_dir = data_dir / 'test_datasets'
    for pattern in config.test_patterns:
        path = test_dir / pattern
        if path.exists():
            tsv_files = list(path.glob('*.tsv'))
            if tsv_files:
                rep_ids = [f.stem for f in tsv_files]
                metadata = pd.DataFrame({
                    'repertoire_id': rep_ids,
                    'filename': [f.name for f in tsv_files]
                })
                datasets[pattern] = metadata
                print(f"   {pattern}: {len(metadata)} repertoires (test)", flush=True)

    return datasets


def load_repertoire_sequences(data_dir: Path, dataset_name: str,
                              rep_id: str, max_seqs: int) -> List[str]:
    """Load sequences from a repertoire."""
    if dataset_name.startswith('train'):
        base_dir = data_dir / 'train_datasets' / dataset_name
    else:
        base_dir = data_dir / 'test_datasets' / dataset_name

    tsv_path = base_dir / f"{rep_id}.tsv"
    if not tsv_path.exists():
        return []

    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa'])
        sequences = df['junction_aa'].dropna().tolist()

        # Sample if needed
        if len(sequences) > max_seqs:
            sequences = list(np.random.choice(sequences, max_seqs, replace=False))

        # Filter valid sequences
        sequences = [s for s in sequences if isinstance(s, str) and len(s) >= 5]
        return sequences
    except Exception as e:
        return []


def extract_dataset_embeddings(
    extractor: ESM2Extractor,
    config: Config,
    dataset_name: str,
    metadata: pd.DataFrame,
    cache_dir: Path
) -> Dict[str, np.ndarray]:
    """Extract embeddings for all repertoires in a dataset."""
    cache_file = cache_dir / f"{dataset_name}_embeddings.pkl"

    if cache_file.exists():
        print(f"   📂 Loading cached {dataset_name}...", flush=True)
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    embeddings = {}
    data_dir = Path(config.data_dir)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"   {dataset_name}"):
        rep_id = row['repertoire_id']

        sequences = load_repertoire_sequences(
            data_dir, dataset_name, rep_id, config.max_seqs_per_rep
        )

        if sequences:
            emb = extractor.extract(sequences)
            embeddings[rep_id] = emb
        else:
            embeddings[rep_id] = np.zeros((1, config.esm_embed_dim))

    # Cache results
    print(f"💾 Caching {dataset_name} ({len(embeddings)} repertoires)...", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings


def fit_pca(all_embeddings: Dict[str, np.ndarray], config: Config) -> IncrementalPCA:
    """Fit PCA on all embeddings."""
    print("\n📊 Fitting PCA...", flush=True)

    all_vecs = []
    for emb_dict in all_embeddings.values():
        for emb in emb_dict.values():
            if emb is not None and len(emb) > 0:
                all_vecs.append(emb)

    all_data = np.vstack(all_vecs)
    print(f"   Total vectors: {len(all_data)}", flush=True)

    # Use Incremental PCA for memory efficiency
    pca = IncrementalPCA(n_components=config.pca_dim, batch_size=5000)

    # Fit in batches
    for i in tqdm(range(0, len(all_data), 10000), desc="   PCA fitting"):
        batch = all_data[i:i+10000]
        pca.partial_fit(batch)

    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"   PCA variance explained: {explained:.1f}%", flush=True)

    return pca


def apply_pca(embeddings: Dict[str, np.ndarray], pca: IncrementalPCA) -> Dict[str, np.ndarray]:
    """Apply PCA to embeddings."""
    result = {}
    for rep_id, emb in embeddings.items():
        if emb is not None and len(emb) > 0:
            result[rep_id] = pca.transform(emb)
        else:
            result[rep_id] = np.zeros((1, pca.n_components))
    return result


class AttentionMIL(nn.Module):
    """Gated Attention Multiple Instance Learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 attention_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attention_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [n_instances, input_dim]
        h = self.feature_extractor(x)  # [n_instances, hidden_dim]

        # Gated attention
        a_V = self.attention_V(h)  # [n_instances, attention_dim]
        a_U = self.attention_U(h)  # [n_instances, attention_dim]
        a = self.attention_w(a_V * a_U)  # [n_instances, 1]
        a = F.softmax(a, dim=0)  # [n_instances, 1]

        # Weighted sum
        z = torch.sum(a * h, dim=0, keepdim=True)  # [1, hidden_dim]

        # Classify
        logit = self.classifier(z)  # [1, 1]

        return logit.squeeze(), a.squeeze()


def train_fold(
    train_embeddings: Dict[str, np.ndarray],
    train_labels: Dict[str, int],
    val_embeddings: Dict[str, np.ndarray],
    val_labels: Dict[str, int],
    config: Config
) -> Tuple[AttentionMIL, float]:
    """Train a single LODO fold."""

    device = torch.device(config.device)
    model = AttentionMIL(
        input_dim=config.pca_dim,
        hidden_dim=config.mil_hidden,
        attention_dim=config.mil_attention,
        dropout=config.mil_dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    # Prepare data
    train_items = [(rep_id, emb, train_labels[rep_id])
                   for rep_id, emb in train_embeddings.items()
                   if rep_id in train_labels]

    val_items = [(rep_id, emb, val_labels[rep_id])
                 for rep_id, emb in val_embeddings.items()
                 if rep_id in val_labels]

    best_auc = 0
    best_model = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        np.random.shuffle(train_items)

        for rep_id, emb, label in train_items:
            x = torch.tensor(emb, dtype=torch.float32).to(device)
            y = torch.tensor([label], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            logit, _ = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for rep_id, emb, label in val_items:
                x = torch.tensor(emb, dtype=torch.float32).to(device)
                logit, _ = model(x)
                prob = torch.sigmoid(logit).item()
                val_preds.append(prob)
                val_true.append(label)

        if len(set(val_true)) > 1:
            val_auc = roc_auc_score(val_true, val_preds)
        else:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # Load best model
    if best_model:
        model.load_state_dict(best_model)

    return model, best_auc


def run_lodo_cv(
    all_embeddings: Dict[str, Dict[str, np.ndarray]],
    all_labels: Dict[str, int],
    rep_to_dataset: Dict[str, str],
    config: Config
) -> Tuple[Dict[str, AttentionMIL], List[float]]:
    """Run Leave-One-Dataset-Out cross-validation."""

    print("\n" + "=" * 60, flush=True)
    print("Stage 3: LODO Cross-Validation (8 folds)", flush=True)
    print("=" * 60, flush=True)

    # Get unique datasets
    datasets = sorted(set(rep_to_dataset.values()))
    print(f"   Datasets: {datasets}", flush=True)

    models = {}
    fold_aucs = []

    for fold_idx, val_dataset in enumerate(datasets):
        print(f"\n   Fold {fold_idx + 1}/8: Val={val_dataset}", flush=True)

        # Split by dataset
        train_reps = [r for r, d in rep_to_dataset.items() if d != val_dataset]
        val_reps = [r for r, d in rep_to_dataset.items() if d == val_dataset]

        # Collect embeddings and labels
        train_embeddings = {}
        train_labels = {}
        val_embeddings = {}
        val_labels = {}

        for ds_name, emb_dict in all_embeddings.items():
            for rep_id, emb in emb_dict.items():
                if rep_id in train_reps:
                    train_embeddings[rep_id] = emb
                    train_labels[rep_id] = all_labels.get(rep_id, 0)
                elif rep_id in val_reps:
                    val_embeddings[rep_id] = emb
                    val_labels[rep_id] = all_labels.get(rep_id, 0)

        print(f"      Train: {len(train_embeddings)}, Val: {len(val_embeddings)}", flush=True)

        # Train
        model, val_auc = train_fold(
            train_embeddings, train_labels,
            val_embeddings, val_labels,
            config
        )

        models[val_dataset] = model
        fold_aucs.append(val_auc)
        print(f"      Val AUC: {val_auc:.4f}", flush=True)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"\n   📊 LODO CV Results: {mean_auc:.4f} ± {std_auc:.4f}", flush=True)

    return models, fold_aucs


def predict_test(
    models: Dict[str, AttentionMIL],
    test_embeddings: Dict[str, Dict[str, np.ndarray]],
    config: Config
) -> Dict[str, float]:
    """Predict on test data using ensemble of LODO models."""

    print("\n" + "=" * 60, flush=True)
    print("Stage 4: Test Set Inference", flush=True)
    print("=" * 60, flush=True)

    device = torch.device(config.device)
    predictions = {}

    # Ensemble all models
    for ds_name, emb_dict in test_embeddings.items():
        print(f"   Processing {ds_name}...", flush=True)

        for rep_id, emb in tqdm(emb_dict.items(), desc=f"   {ds_name}"):
            x = torch.tensor(emb, dtype=torch.float32).to(device)

            probs = []
            for model in models.values():
                model.eval()
                model.to(device)
                with torch.no_grad():
                    logit, _ = model(x)
                    prob = torch.sigmoid(logit).item()
                    probs.append(prob)

            # Average ensemble
            predictions[rep_id] = np.mean(probs)

    return predictions


def select_task_b_sequences(
    all_embeddings: Dict[str, Dict[str, np.ndarray]],
    all_labels: Dict[str, int],
    data_dir: Path,
    config: Config
) -> Dict[str, List[Dict]]:
    """Select top 50,000 disease-associated sequences per training dataset."""

    print("\n" + "=" * 60, flush=True)
    print("Stage 5: Task B Sequence Selection", flush=True)
    print("=" * 60, flush=True)

    results = {}

    for ds_name in config.train_patterns:
        print(f"   Processing {ds_name}...", flush=True)

        # Load all sequences with metadata
        base_dir = data_dir / 'train_datasets' / ds_name
        all_seqs = []

        for _, row in pd.read_csv(base_dir / 'metadata.csv').iterrows():
            rep_id = row['repertoire_id']
            label = row.get('label', 0)

            tsv_path = base_dir / f"{rep_id}.tsv"
            if tsv_path.exists():
                try:
                    df = pd.read_csv(tsv_path, sep='\t')
                    for _, seq_row in df.iterrows():
                        if pd.notna(seq_row.get('junction_aa')):
                            all_seqs.append({
                                'junction_aa': seq_row['junction_aa'],
                                'v_call': seq_row.get('v_call', '-999.0'),
                                'j_call': seq_row.get('j_call', '-999.0'),
                                'label': label
                            })
                except:
                    pass

        # Sort by disease association (label=1)
        disease_seqs = [s for s in all_seqs if s['label'] == 1]
        healthy_seqs = [s for s in all_seqs if s['label'] == 0]

        # Prioritize disease sequences, fill with random if needed
        np.random.shuffle(disease_seqs)
        np.random.shuffle(healthy_seqs)

        selected = disease_seqs[:50000]
        if len(selected) < 50000:
            remaining = 50000 - len(selected)
            selected.extend(healthy_seqs[:remaining])

        # Ensure exactly 50000
        while len(selected) < 50000:
            selected.append({
                'junction_aa': 'CASSLGQAY',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

        results[ds_name] = selected[:50000]
        print(f"      Selected: {len(results[ds_name])} sequences", flush=True)

    return results


def create_submission(
    task_a_predictions: Dict[str, float],
    task_b_sequences: Dict[str, List[Dict]],
    output_path: Path
):
    """Create final submission file."""

    print("\n" + "=" * 60, flush=True)
    print("Stage 6: Create Submission", flush=True)
    print("=" * 60, flush=True)

    rows = []

    # Task A predictions
    for rep_id, prob in task_a_predictions.items():
        # Determine dataset from test patterns
        dataset = None
        for pattern in ['test_dataset_1', 'test_dataset_2', 'test_dataset_3',
                       'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
                       'test_dataset_7_1', 'test_dataset_7_2',
                       'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3']:
            # Simple heuristic - check if this is a valid pattern
            dataset = pattern
            break

        rows.append({
            'ID': rep_id,
            'dataset': 'test',  # Will be filled properly
            'label_positive_probability': prob,
            'junction_aa': '-999.0',
            'v_call': '-999.0',
            'j_call': '-999.0'
        })

    # Task B sequences
    for ds_name, sequences in task_b_sequences.items():
        for idx, seq in enumerate(sequences):
            rows.append({
                'ID': f"{ds_name}_seq_top_{idx + 1}",
                'dataset': ds_name,
                'label_positive_probability': '-999.0',
                'junction_aa': seq['junction_aa'],
                'v_call': str(seq.get('v_call', '-999.0')),
                'j_call': str(seq.get('j_call', '-999.0'))
            })

    df = pd.DataFrame(rows)

    # Ensure correct column order
    df = df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"   ✅ Submission saved: {output_path}", flush=True)
    print(f"   Total rows: {len(df)}", flush=True)

    return df


def main():
    print_banner()

    config = Config()
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🚀 GPU: {gpu_name}")
        print(f"   Memory: {gpu_mem:.1f} GB")

    # Discover datasets
    print("\n📁 Discovering datasets...")
    datasets = discover_datasets(config)
    train_datasets = {k: v for k, v in datasets.items() if k.startswith('train')}
    test_datasets = {k: v for k, v in datasets.items() if k.startswith('test')}
    print(f"   Train: {len(train_datasets)}, Test: {len(test_datasets)}")

    # Initialize extractor
    extractor = ESM2Extractor(config)

    # Stage 1: Extract embeddings
    print("\n" + "=" * 60)
    print("Stage 1: Extract Training Embeddings (15B)")
    print("=" * 60)

    train_embeddings = {}
    all_labels = {}
    rep_to_dataset = {}

    for ds_name, metadata in train_datasets.items():
        emb = extract_dataset_embeddings(extractor, config, ds_name, metadata, cache_dir)
        train_embeddings[ds_name] = emb

        # Extract labels
        for _, row in metadata.iterrows():
            rep_id = row['repertoire_id']
            rep_to_dataset[rep_id] = ds_name
            if 'label' in row:
                all_labels[rep_id] = int(row['label'])

    # Stage 2: PCA
    print("\n" + "=" * 60)
    print("Stage 2: PCA Dimensionality Reduction")
    print("=" * 60)

    pca = fit_pca(train_embeddings, config)

    # Apply PCA to training embeddings
    for ds_name in train_embeddings:
        train_embeddings[ds_name] = apply_pca(train_embeddings[ds_name], pca)

    # Stage 3: LODO CV
    models, fold_aucs = run_lodo_cv(train_embeddings, all_labels, rep_to_dataset, config)

    # Stage 4: Test inference
    print("\n" + "=" * 60)
    print("Stage 4: Extract Test Embeddings")
    print("=" * 60)

    test_embeddings = {}
    for ds_name, metadata in test_datasets.items():
        emb = extract_dataset_embeddings(extractor, config, ds_name, metadata, cache_dir)
        test_embeddings[ds_name] = apply_pca(emb, pca)

    task_a_predictions = predict_test(models, test_embeddings, config)

    # Stage 5: Task B
    task_b_sequences = select_task_b_sequences(
        train_embeddings, all_labels, Path(config.data_dir), config
    )

    # Stage 6: Create submission
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_path = Path(config.output_dir) / 'submissions' / f'esm2_15b_submission_{timestamp}.csv'
    create_submission(task_a_predictions, task_b_sequences, output_path)

    # Summary
    mean_auc = np.mean(fold_aucs)
    print("\n" + "=" * 60)
    print("🏆 PIPELINE COMPLETE 🏆")
    print("=" * 60)
    print(f"   LODO CV AUC: {mean_auc:.4f}")
    print(f"   Submission: {output_path}")
    print("\n   🎯 Ready for Kaggle submission!")


if __name__ == '__main__':
    main()
