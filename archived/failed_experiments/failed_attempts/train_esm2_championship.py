#!/usr/bin/env python3
"""
train_esm2_championship.py - ESM-2 Deep Learning Championship Pipeline
=======================================================================

Target: Beat GROZD (0.81364) → Achieve 0.82+
Hardware: NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory)

KEY IMPROVEMENTS:
- HuggingFace ESM-2 (more reliable download)
- Timeout & retry for model loading
- ESM-2 L15 (optimal for TCR CDR3)
- Attention-based MIL (EAMIL/DeepRC style)
- LODO Cross-Validation
- Real-time progress output

Usage:
    docker run --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --shm-size=64g \
        -v $(pwd):/app -v $(pwd)/data:/app/data \
        -v ~/.cache:/root/.cache \
        -w /app --rm \
        nvcr.io/nvidia/pytorch:25.11-py3 \
        bash -c "pip install transformers scikit-learn tqdm pandas accelerate --quiet && \
                 python -u scripts/train_esm2_championship.py"
"""

import os
import sys
import gc
import json
import pickle
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler  # PyTorch 2.10+ compatible

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings('ignore')


def print_banner():
    """Print championship banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🏆 ESM-2 3B CHAMPIONSHIP PIPELINE 🏆                                        ║
║                                                                              ║
║  Target: Beat GROZD (0.81364) → Achieve 0.82+                                ║
║  Hardware: NVIDIA DGX Spark (GB10, 128GB unified memory)                     ║
║                                                                              ║
║  ARCHITECTURE (OPTIMIZED FOR DEADLINE):                                      ║
║  ✅ ESM-2 3B (esm2_t36_3B_UR50D) - 3 BILLION parameters!                     ║
║  ✅ Layer 24 embeddings (optimal for 36-layer model)                         ║
║  ✅ Gated Attention MIL (EAMIL/DeepRC)                                       ║
║  ✅ LODO Cross-Validation                                                    ║
║  ✅ Task B: LogisticRegression coefficients                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """, flush=True)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """Championship Configuration"""
    # Paths
    data_dir: str = '/app/data'
    output_dir: str = '/app/outputs'
    checkpoint_dir: str = '/app/cache/esm2_championship'

    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_workers: int = 4

    # ESM-2 Settings (HuggingFace version) - 3B MODEL (PRACTICAL for deadline)
    # esm2_t33_650M_UR50D (650M params, 33 layers) - Small/Fast
    # esm2_t36_3B_UR50D (3B params, 36 layers) - Good balance (USING THIS)
    # esm2_t48_15B_UR50D (15B params, 48 layers) - Too slow (~67h per dataset!)
    esm_model_name: str = 'facebook/esm2_t36_3B_UR50D'
    esm_repr_layer: int = 24  # Layer 24 is optimal for 36-layer model (2/3 depth)
    esm_batch_size: int = 128  # 🔥 MAX: 榨乾GPU (2x batch = ~1.3x速度)
    esm_reduced_dim: int = 1024  # 🧠 SMART: 大維度利用RAM (不影響速度)

    # PrimeSeq Strategy - 🧠 SMART: 速度與質量平衡
    primeseq_n_select: int = 1500  # 🧠 SMART: 比原始多50%信號，但保持速度
    primeseq_freq_ratio: float = 0.5  # 50% frequency-based

    # MIL Settings - 🧠 SMART: 大網路利用RAM (不影響Stage1速度)
    mil_hidden_dim: int = 1024  # 🧠 SMART: 大網路提升質量
    mil_attention_dim: int = 512  # 🧠 SMART: 強注意力機制
    mil_dropout: float = 0.3

    # Training - 🧠 SMART: 大batch利用RAM加速MIL訓練
    train_epochs: int = 50  # 🧠 SMART: 有early stopping保護
    train_batch_size: int = 128  # 🧠 SMART: 大batch更穩定
    train_lr: float = 1e-4

    # Test patterns
    test_patterns: List[str] = None

    def __post_init__(self):
        self.test_patterns = [
            'test_dataset_1', 'test_dataset_2', 'test_dataset_3',
            'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
            'test_dataset_7_1', 'test_dataset_7_2',
            'test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3'
        ]
        # Create dirs
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# =============================================================================
# ESM-2 HuggingFace Extractor (More Reliable)
# =============================================================================
class ESM2HFExtractor:
    """HuggingFace-based ESM-2 Embedding Extractor"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.model = None
        self.tokenizer = None
        self.pca = None

    def load_model(self, max_retries: int = 10):
        """Load ESM-2 from HuggingFace with progress and retry mechanism (Bug #8 Fix)"""
        from transformers import AutoTokenizer, EsmModel
        import os

        # Disable XET for more reliable downloads
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

        print("📥 Loading ESM-2 from HuggingFace...", flush=True)
        print(f"   Model: {self.config.esm_model_name}", flush=True)
        print(f"   XET disabled for stability, using standard HTTP download", flush=True)
        print(f"   This may take 30-60 minutes for 15B model (~60GB)...", flush=True)

        start_time = time.time()

        # Bug #8 Fix: Enhanced retry mechanism for ESM-2 download
        for attempt in range(max_retries):
            try:
                print(f"   🔄 Attempt {attempt+1}/{max_retries}...", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.esm_model_name,
                    resume_download=True,
                    local_files_only=False
                )
                self.model = EsmModel.from_pretrained(
                    self.config.esm_model_name,
                    resume_download=True,
                    local_files_only=False
                )
                self.model = self.model.to(self.device).eval()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # Exponential backoff: 30s, 60s, 90s...
                    print(f"   ⚠️ Attempt {attempt+1} failed: {str(e)[:100]}...", flush=True)
                    print(f"   Retrying in {wait_time} seconds...", flush=True)
                    time.sleep(wait_time)
                    # Clean incomplete files and retry
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                else:
                    raise RuntimeError(f"Failed to load ESM-2 after {max_retries} attempts: {e}")

        elapsed = time.time() - start_time
        print(f"✅ ESM-2 loaded on {self.device} in {elapsed:.1f}s", flush=True)
        print(f"   Using representation layer: {self.config.esm_repr_layer}", flush=True)

        # 🚀 OPTIMIZATION: torch.compile() for speedup
        # Note: Use "default" mode (NOT "reduce-overhead") to avoid CUDAGraph issues with ESM-2
        try:
            print("🔥 Applying torch.compile() optimization (default mode)...", flush=True)
            self.model = torch.compile(self.model, mode="default", fullgraph=False)
            print("   ✅ torch.compile() successful!", flush=True)
        except Exception as e:
            print(f"   ⚠️ torch.compile() failed (using uncompiled model): {str(e)[:80]}", flush=True)

    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings for a batch of sequences"""
        if self.model is None:
            self.load_model()

        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50  # CDR3 sequences are short
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 🚀 OPTIMIZATION: torch.inference_mode() + autocast for maximum speed
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(**inputs, output_hidden_states=True)

        # Get L15 representations
        hidden_states = outputs.hidden_states[self.config.esm_repr_layer]

        # Mean pooling (excluding padding)
        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
        sum_repr = (hidden_states * attention_mask).sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        mean_repr = sum_repr / lengths

        return mean_repr.cpu().float().numpy()

    def extract_all(self, sequences: List[str], show_progress: bool = True) -> np.ndarray:
        """Extract embeddings for all sequences"""
        all_embeddings = []
        batch_size = self.config.esm_batch_size

        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="ESM-2 embedding", file=sys.stdout)

        for i in iterator:
            batch = sequences[i:i+batch_size]
            embeddings = self.extract_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA for dimensionality reduction"""
        print(f"Fitting PCA: {embeddings.shape[1]} → {self.config.esm_reduced_dim}", flush=True)

        # Handle NaN values - replace with 0 (neutral embedding)
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            print(f"  ⚠️ Found {nan_count} NaN values, replacing with 0...", flush=True)
            embeddings = np.nan_to_num(embeddings, nan=0.0)

        # Also handle inf values
        inf_count = np.isinf(embeddings).sum()
        if inf_count > 0:
            print(f"  ⚠️ Found {inf_count} Inf values, clipping...", flush=True)
            embeddings = np.clip(embeddings, -1e6, 1e6)

        self.pca = PCA(n_components=self.config.esm_reduced_dim, random_state=42)
        self.pca.fit(embeddings)
        print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}", flush=True)

    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA transformation"""
        if self.pca is None:
            raise ValueError("PCA not fitted!")
        # Handle NaN/Inf before transform
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        embeddings = np.clip(embeddings, -1e6, 1e6)
        return self.pca.transform(embeddings)


# =============================================================================
# Gated Attention MIL (EAMIL/DeepRC Style)
# =============================================================================
class GatedAttention(nn.Module):
    """Gated Attention mechanism"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, n_instances, input_dim)
            mask: (batch, n_instances)
        Returns:
            aggregated: (batch, input_dim)
            attention_weights: (batch, n_instances)
        """
        v = torch.tanh(self.attention_V(x))
        u = torch.sigmoid(self.attention_U(x))
        scores = self.attention_w(v * u)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        weights = F.softmax(scores, dim=1)
        aggregated = (x * weights).sum(dim=1)

        return aggregated, weights.squeeze(-1)


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning Model"""

    def __init__(self, config: Config):
        super().__init__()

        input_dim = config.esm_reduced_dim
        hidden_dim = config.mil_hidden_dim
        attention_dim = config.mil_attention_dim
        dropout = config.mil_dropout

        # Instance encoder
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention
        self.attention = GatedAttention(hidden_dim, attention_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, n_instances, input_dim)
            mask: (batch, n_instances)
        Returns:
            logits: (batch, 1)
            attention_weights: (batch, n_instances)
        """
        # Encode instances
        encoded = self.instance_encoder(x)

        # Attention aggregation
        aggregated, weights = self.attention(encoded, mask)

        # Classify
        logits = self.classifier(aggregated)

        return logits, weights


# =============================================================================
# Data Loading
# =============================================================================
def load_repertoire_data(tsv_path: Path, n_select: int = 1000) -> pd.DataFrame:
    """Load and sample repertoire data"""
    # Only use columns that exist: junction_aa, v_call, j_call
    df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])
    df = df.dropna(subset=['junction_aa'])
    df = df[df['junction_aa'].str.len() >= 5]
    df = df[df['junction_aa'].str.len() <= 30]

    if len(df) > n_select:
        # Random sample since templates column doesn't exist
        df = df.sample(n=n_select, random_state=42)

    return df


def discover_datasets(config: Config) -> Dict:
    """Discover train and test datasets"""
    data_dir = Path(config.data_dir)

    # Training datasets
    train_dir = data_dir / 'train_datasets'
    train_datasets = {}

    for i in range(1, 9):
        dataset_path = train_dir / f'train_dataset_{i}'
        if dataset_path.exists():
            metadata_path = dataset_path / 'metadata.csv'
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path)
                train_datasets[f'train_dataset_{i}'] = {
                    'path': dataset_path,
                    'metadata': metadata,
                    'n_repertoires': len(metadata)
                }

    # Test datasets
    test_dir = data_dir / 'test_datasets'
    test_datasets = {}

    for pattern in config.test_patterns:
        test_path = test_dir / pattern
        if test_path.exists():
            # Count TSV files
            tsv_files = list(test_path.glob('*.tsv'))
            test_datasets[pattern] = {
                'path': test_path,
                'n_repertoires': len(tsv_files)
            }

    return {'train': train_datasets, 'test': test_datasets}


# =============================================================================
# MIL Dataset
# =============================================================================
class RepertoireDataset(Dataset):
    """Dataset for MIL training"""

    def __init__(self, embeddings_dict: Dict[str, np.ndarray],
                 labels_dict: Dict[str, int],
                 max_instances: int = 1000):
        self.rep_ids = list(embeddings_dict.keys())
        self.embeddings = embeddings_dict
        self.labels = labels_dict
        self.max_instances = max_instances

    def __len__(self):
        return len(self.rep_ids)

    def __getitem__(self, idx):
        rep_id = self.rep_ids[idx]
        emb = self.embeddings[rep_id]
        label = self.labels[rep_id]

        # Pad/truncate
        n_instances = len(emb)
        if n_instances > self.max_instances:
            indices = np.random.choice(n_instances, self.max_instances, replace=False)
            emb = emb[indices]
            n_instances = self.max_instances

        # Create mask
        mask = np.ones(self.max_instances, dtype=bool)
        mask[n_instances:] = False

        # Pad embeddings
        padded = np.zeros((self.max_instances, emb.shape[1]), dtype=np.float32)
        padded[:n_instances] = emb

        return {
            'embeddings': torch.tensor(padded, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.float32),
            'rep_id': rep_id
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'embeddings': torch.stack([b['embeddings'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'rep_id': [b['rep_id'] for b in batch]
    }


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, optimizer, scaler, device):
    """Train one epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _ = model(embeddings, mask)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_rep_ids = []

    for batch in dataloader:
        embeddings = batch['embeddings'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label']
        rep_ids = batch['rep_id']

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _ = model(embeddings, mask)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
        all_rep_ids.extend(rep_ids)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5

    return auc, dict(zip(all_rep_ids, all_preds))


# =============================================================================
# Task B: Sequence Selection
# =============================================================================
def select_task_b_sequences(train_embeddings: Dict, train_labels: Dict,
                            sequences_dict: Dict, esm_extractor: ESM2HFExtractor) -> Dict[str, List[str]]:
    """Select top 50k sequences per dataset using learned weights"""

    print("\n📊 Task B: Selecting top sequences...", flush=True)

    results = {}

    # Group by dataset
    dataset_reps = defaultdict(list)
    for rep_id in train_embeddings.keys():
        dataset_id = '_'.join(rep_id.split('_')[:3])  # e.g., train_dataset_1
        dataset_reps[dataset_id].append(rep_id)

    for dataset_id, rep_ids in dataset_reps.items():
        print(f"  Processing {dataset_id}...", flush=True)

        # Collect all sequences from this dataset
        all_sequences = []
        all_embeddings = []
        all_labels = []

        for rep_id in rep_ids:
            label = train_labels.get(rep_id, 0)
            seqs = sequences_dict.get(rep_id, [])
            embs = train_embeddings.get(rep_id)

            if embs is not None and len(seqs) == len(embs):
                all_sequences.extend(seqs)
                all_embeddings.append(embs)
                all_labels.extend([label] * len(seqs))

        if not all_sequences:
            print(f"    No sequences found for {dataset_id}", flush=True)
            results[dataset_id] = []
            continue

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)

        # Train LogisticRegression for sequence scoring
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(all_embeddings)

            lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
            lr.fit(X_scaled, all_labels)

            # Score sequences
            scores = lr.predict_proba(X_scaled)[:, 1]

            # Select top 50k
            n_select = min(50000, len(all_sequences))
            top_indices = np.argsort(scores)[-n_select:][::-1]

            results[dataset_id] = [all_sequences[i] for i in top_indices]

            print(f"    Selected {len(results[dataset_id])} sequences", flush=True)

        except Exception as e:
            print(f"    Error: {e}", flush=True)
            # Fallback: random selection
            indices = np.random.choice(len(all_sequences), min(50000, len(all_sequences)), replace=False)
            results[dataset_id] = [all_sequences[i] for i in indices]

    return results


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print_banner()

    config = Config()

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 GPU: {gpu_name}", flush=True)
        print(f"   Memory: {gpu_mem:.1f} GB", flush=True)
    else:
        print("⚠️ No GPU found! Using CPU (will be slow)", flush=True)

    # Discover datasets
    print("\n📁 Discovering datasets...", flush=True)
    datasets = discover_datasets(config)

    print(f"   Training: {len(datasets['train'])} datasets", flush=True)
    print(f"   Test: {len(datasets['test'])} datasets", flush=True)

    # Bug #6 Fix: Strict validation - must have exactly 11 test datasets
    if len(datasets['test']) != 11:
        raise ValueError(f"FATAL: Expected 11 test datasets, found {len(datasets['test'])}. "
                        f"Missing: {set(config.test_patterns) - set(datasets['test'].keys())}")

    # Initialize ESM-2 extractor
    esm_extractor = ESM2HFExtractor(config)

    # ==========================================================================
    # Stage 1: Extract ESM-2 Embeddings for Training Data
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 1: Extracting ESM-2 Embeddings", flush=True)
    print("="*60, flush=True)

    checkpoint_path = Path(config.checkpoint_dir) / 'train_embeddings.pkl'

    # 🛡️ CRASH-PROOF: Incremental checkpointing per dataset
    partial_cache_dir = Path(config.checkpoint_dir) / 'partial'
    partial_cache_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        print(f"📂 Loading cached embeddings from {checkpoint_path}...", flush=True)
        with open(checkpoint_path, 'rb') as f:
            cache = pickle.load(f)
            train_embeddings = cache['embeddings']
            train_labels = cache['labels']
            train_sequences = cache['sequences']
            # Bug #5 Fix: Sync PCA from checkpoint
            if 'pca' in cache:
                esm_extractor.pca = cache['pca']
                print(f"   ✅ PCA loaded from checkpoint", flush=True)
    else:
        train_embeddings = {}
        train_labels = {}
        train_sequences = {}

        for dataset_id, dataset_info in tqdm(datasets['train'].items(), desc="Datasets"):
            # 🛡️ CRASH-PROOF: Check for partial checkpoint
            partial_cache_path = partial_cache_dir / f'{dataset_id}_raw.pkl'
            if partial_cache_path.exists():
                print(f"\n  📂 Loading cached {dataset_id} from partial checkpoint...", flush=True)
                try:
                    with open(partial_cache_path, 'rb') as f:
                        partial_cache = pickle.load(f)
                    train_embeddings.update(partial_cache['embeddings'])
                    train_labels.update(partial_cache['labels'])
                    train_sequences.update(partial_cache['sequences'])
                    print(f"     ✅ Loaded {len(partial_cache['embeddings'])} repertoires", flush=True)
                    continue  # Skip to next dataset
                except Exception as e:
                    print(f"     ⚠️ Failed to load partial cache: {e}, reprocessing...", flush=True)

            metadata = dataset_info['metadata']
            dataset_path = dataset_info['path']
            dataset_embeddings = {}
            dataset_labels = {}
            dataset_sequences = {}

            print(f"\n  Processing {dataset_id}...", flush=True)

            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  {dataset_id}", leave=False):
                rep_id = row['repertoire_id']
                # Bug #1 Fix: Simplified label conversion
                label_val = row['label_positive']
                label = 1 if (label_val is True or str(label_val).lower() == 'true') else 0
                tsv_path = dataset_path / f"{rep_id}.tsv"

                if not tsv_path.exists():
                    continue

                try:
                    # Load repertoire
                    rep_df = load_repertoire_data(tsv_path, n_select=config.primeseq_n_select)
                    sequences = rep_df['junction_aa'].tolist()

                    if len(sequences) < 10:
                        continue

                    # Extract embeddings
                    embeddings = esm_extractor.extract_all(sequences, show_progress=False)

                    # 🛡️ CRASH-PROOF: Check for NaN/Inf in embeddings
                    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

                    dataset_embeddings[rep_id] = embeddings
                    dataset_labels[rep_id] = label
                    dataset_sequences[rep_id] = sequences
                except Exception as e:
                    print(f"     ⚠️ Error processing {rep_id}: {str(e)[:50]}, skipping...", flush=True)
                    continue

            # 🛡️ CRASH-PROOF: Save partial checkpoint immediately after each dataset
            print(f"     💾 Saving partial checkpoint for {dataset_id}...", flush=True)
            with open(partial_cache_path, 'wb') as f:
                pickle.dump({
                    'embeddings': dataset_embeddings,
                    'labels': dataset_labels,
                    'sequences': dataset_sequences
                }, f)
            print(f"     ✅ Saved {len(dataset_embeddings)} repertoires to {partial_cache_path.name}", flush=True)

            # Merge into main dictionaries
            train_embeddings.update(dataset_embeddings)
            train_labels.update(dataset_labels)
            train_sequences.update(dataset_sequences)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Fit PCA on all embeddings
        print("\n  Fitting PCA...", flush=True)
        all_emb = np.vstack(list(train_embeddings.values()))
        esm_extractor.fit_pca(all_emb)

        # Apply PCA
        for rep_id in train_embeddings:
            train_embeddings[rep_id] = esm_extractor.transform_pca(train_embeddings[rep_id])

        # Save final checkpoint
        print(f"\n  Saving final checkpoint to {checkpoint_path}...", flush=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'embeddings': train_embeddings,
                'labels': train_labels,
                'sequences': train_sequences,
                'pca': esm_extractor.pca
            }, f)
        print(f"  ✅ Final checkpoint saved!", flush=True)

    print(f"\n✅ Loaded {len(train_embeddings)} repertoires", flush=True)

    # ==========================================================================
    # Stage 2: LODO Cross-Validation Training
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 2: LODO Cross-Validation Training", flush=True)
    print("="*60, flush=True)

    # Group by dataset
    dataset_reps = defaultdict(list)
    for rep_id in train_embeddings.keys():
        dataset_id = '_'.join(rep_id.split('_')[:3])
        dataset_reps[dataset_id].append(rep_id)

    dataset_ids = sorted(dataset_reps.keys())
    print(f"  Datasets: {dataset_ids}", flush=True)

    fold_models = []
    fold_aucs = []

    for fold_idx, val_dataset in enumerate(dataset_ids):
        print(f"\n  Fold {fold_idx+1}/{len(dataset_ids)}: Validating on {val_dataset}", flush=True)

        # Split
        train_reps = [r for d, reps in dataset_reps.items() for r in reps if d != val_dataset]
        val_reps = dataset_reps[val_dataset]

        print(f"    Train: {len(train_reps)}, Val: {len(val_reps)}", flush=True)

        # Create datasets
        train_emb = {r: train_embeddings[r] for r in train_reps}
        train_lab = {r: train_labels[r] for r in train_reps}
        val_emb = {r: train_embeddings[r] for r in val_reps}
        val_lab = {r: train_labels[r] for r in val_reps}

        train_dataset = RepertoireDataset(train_emb, train_lab)
        val_dataset_obj = RepertoireDataset(val_emb, val_lab)

        train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset_obj, batch_size=config.train_batch_size,
                               shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Initialize model
        model = AttentionMIL(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_lr)
        scaler = GradScaler()

        # Training loop
        best_auc = 0
        best_model_state = None
        patience = 5
        patience_counter = 0

        for epoch in range(config.train_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, config.device)
            val_auc, _ = evaluate(model, val_loader, config.device)

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f} (Best: {best_auc:.4f})", flush=True)

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        fold_models.append(model)
        fold_aucs.append(best_auc)

        print(f"    ✅ Fold {fold_idx+1} Best AUC: {best_auc:.4f}", flush=True)

    mean_auc = np.mean(fold_aucs)
    print(f"\n📊 LODO CV Mean AUC: {mean_auc:.4f} (±{np.std(fold_aucs):.4f})", flush=True)

    # Save LODO models checkpoint
    models_checkpoint_path = Path(config.checkpoint_dir) / 'lodo_models.pkl'
    print(f"\n  Saving LODO models to {models_checkpoint_path}...", flush=True)
    with open(models_checkpoint_path, 'wb') as f:
        pickle.dump({
            'fold_models_state': [m.state_dict() for m in fold_models],
            'fold_aucs': fold_aucs,
            'dataset_ids': dataset_ids,
            'config': {
                'mil_hidden_dim': config.mil_hidden_dim,
                'mil_attention_dim': config.mil_attention_dim,
                'esm_reduced_dim': config.esm_reduced_dim,
            }
        }, f)
    print(f"  ✅ LODO models saved", flush=True)

    # ==========================================================================
    # Stage 3: Test Predictions (Task A)
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 3: Test Predictions (Task A)", flush=True)
    print("="*60, flush=True)

    # Load PCA if needed
    if esm_extractor.pca is None:
        with open(checkpoint_path, 'rb') as f:
            cache = pickle.load(f)
            esm_extractor.pca = cache['pca']

    test_predictions = {}
    # Bug #2/#3 Fix: Build rep_id → dataset mapping during scan
    rep_to_dataset = {}

    # Test embeddings cache directory
    test_cache_dir = Path(config.checkpoint_dir) / 'test_embeddings'
    test_cache_dir.mkdir(parents=True, exist_ok=True)

    for test_name, test_info in datasets['test'].items():
        print(f"\n  Processing {test_name}...", flush=True)
        test_path = test_info['path']

        # Check for cached test embeddings
        test_cache_path = test_cache_dir / f'{test_name}_embeddings.pkl'
        if test_cache_path.exists():
            print(f"    📂 Loading cached embeddings from {test_cache_path}...", flush=True)
            with open(test_cache_path, 'rb') as f:
                cache = pickle.load(f)
                cached_embeddings = cache['embeddings']
                cached_rep_to_dataset = cache['rep_to_dataset']

            # Use cached embeddings for predictions
            for rep_id, embeddings in cached_embeddings.items():
                rep_to_dataset[rep_id] = cached_rep_to_dataset[rep_id]

                if embeddings is None:
                    test_predictions[rep_id] = 0.5
                    continue

                # Create dataset
                test_emb = {rep_id: embeddings}
                test_lab = {rep_id: 0}
                test_dataset = RepertoireDataset(test_emb, test_lab)
                test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

                # Ensemble prediction
                all_probs = []
                for model in fold_models:
                    _, preds = evaluate(model, test_loader, config.device)
                    all_probs.append(preds[rep_id])

                test_predictions[rep_id] = np.mean(all_probs)

            print(f"    ✅ Loaded {len(cached_embeddings)} cached repertoires", flush=True)
            continue

        # No cache - extract embeddings
        tsv_files = list(test_path.glob('*.tsv'))
        print(f"    Found {len(tsv_files)} repertoires", flush=True)

        # Store embeddings for caching
        dataset_embeddings = {}
        dataset_rep_mapping = {}

        for tsv_file in tqdm(tsv_files, desc=f"    {test_name}", leave=False):
            rep_id = tsv_file.stem

            # Bug #2/#3 Fix: Record mapping from directory structure (reliable!)
            rep_to_dataset[rep_id] = test_name
            dataset_rep_mapping[rep_id] = test_name

            # Load and process
            rep_df = load_repertoire_data(tsv_file, n_select=config.primeseq_n_select)
            sequences = rep_df['junction_aa'].tolist()

            if len(sequences) < 10:
                test_predictions[rep_id] = 0.5
                dataset_embeddings[rep_id] = None
                continue

            # Extract embeddings
            embeddings = esm_extractor.extract_all(sequences, show_progress=False)
            embeddings = esm_extractor.transform_pca(embeddings)

            # Store for cache
            dataset_embeddings[rep_id] = embeddings

            # Create dataset
            test_emb = {rep_id: embeddings}
            test_lab = {rep_id: 0}  # Dummy label
            test_dataset = RepertoireDataset(test_emb, test_lab)
            test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

            # Ensemble prediction
            all_probs = []
            for model in fold_models:
                _, preds = evaluate(model, test_loader, config.device)
                all_probs.append(preds[rep_id])

            test_predictions[rep_id] = np.mean(all_probs)

        # Save test embeddings cache for this dataset
        print(f"    💾 Saving embeddings cache to {test_cache_path}...", flush=True)
        with open(test_cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': dataset_embeddings,
                'rep_to_dataset': dataset_rep_mapping
            }, f)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Bug #9 Fix: Validate Task A prediction count = 4,213
    if len(test_predictions) != 4213:
        raise ValueError(f"FATAL: Expected 4,213 Task A predictions, got {len(test_predictions)}. "
                        f"Check test data integrity!")

    print(f"\n✅ Generated {len(test_predictions)} Task A predictions (validated = 4,213)", flush=True)

    # ==========================================================================
    # Stage 4: Task B Sequence Selection
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 4: Task B Sequence Selection", flush=True)
    print("="*60, flush=True)

    task_b_selections = select_task_b_sequences(
        train_embeddings, train_labels, train_sequences, esm_extractor
    )

    # ==========================================================================
    # Stage 5: Generate Submission
    # ==========================================================================
    print("\n" + "="*60, flush=True)
    print("Stage 5: Generating Submission", flush=True)
    print("="*60, flush=True)

    submission_rows = []

    # Task A rows - using the rep_to_dataset mapping (Bug #2/#3 Fix)
    for rep_id, prob in test_predictions.items():
        dataset_name = rep_to_dataset.get(rep_id)
        if dataset_name is None:
            raise ValueError(f"FATAL: No dataset mapping for repertoire {rep_id}")

        submission_rows.append({
            'ID': rep_id,
            'dataset': dataset_name,
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })

    # Task B rows - Bug #4 Fix: Ensure exactly 50,000 sequences per dataset
    for dataset_id, sequences in task_b_selections.items():
        n_sequences = len(sequences)

        if n_sequences < 50000:
            # Bug #4 Fix: Pad with repeated sequences to reach exactly 50k
            print(f"  ⚠️ {dataset_id}: Only {n_sequences} sequences, padding to 50,000", flush=True)
            # Repeat sequences to fill gap
            padded_sequences = sequences.copy()
            while len(padded_sequences) < 50000:
                padded_sequences.extend(sequences[:min(len(sequences), 50000 - len(padded_sequences))])
            sequences = padded_sequences[:50000]

        for i, seq in enumerate(sequences[:50000]):
            submission_rows.append({
                'ID': f"{dataset_id}_seq_top_{i+1}",
                'dataset': dataset_id,
                'label_positive_probability': -999.0,
                'junction_aa': seq,
                'v_call': -999.0,
                'j_call': -999.0
            })

    submission_df = pd.DataFrame(submission_rows)

    # Bug #10 Fix: Strict validation - no padding, must be exactly 404,213
    actual_rows = len(submission_df)
    expected_rows = 404213  # 4,213 Task A + 8 × 50,000 Task B

    print(f"\n  Task A rows: {len(test_predictions)}", flush=True)
    print(f"  Task B rows: {actual_rows - len(test_predictions)}", flush=True)
    print(f"  Total rows: {actual_rows}", flush=True)
    print(f"  Expected: {expected_rows}", flush=True)

    if actual_rows != expected_rows:
        raise ValueError(f"FATAL: Submission has {actual_rows} rows, expected exactly {expected_rows}. "
                        f"Task A: {len(test_predictions)}, Task B: {actual_rows - len(test_predictions)}")

    # Save
    output_path = Path(config.output_dir) / 'submission_esm2_championship.csv'
    submission_df.to_csv(output_path, index=False)

    print(f"\n✅ Submission saved to {output_path}", flush=True)
    print(f"   Total rows: {len(submission_df)} (validated = 404,213)", flush=True)

    # Summary
    print("\n" + "="*60, flush=True)
    print("🏆 CHAMPIONSHIP PIPELINE COMPLETE!", flush=True)
    print("="*60, flush=True)
    print(f"   LODO CV AUC: {mean_auc:.4f}", flush=True)
    print(f"   Task A predictions: {len(test_predictions)}", flush=True)
    print(f"   Task B datasets: {len(task_b_selections)}", flush=True)
    print(f"   Submission: {output_path}", flush=True)


if __name__ == '__main__':
    main()
