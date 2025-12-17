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

🔄 CHECKPOINT SYSTEM:
- Saves progress after each dataset's feature extraction
- Saves progress after each training fold
- Automatically resumes from last checkpoint on restart
"""

import os
import sys
import gc
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Pool, set_start_method
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from queue import Queue
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import psutil
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Checkpoint Configuration (Memory-Optimized)
# ============================================================================
CHECKPOINT_DIR = './checkpoints'
TRAINING_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'training_progress.json')
FEATURE_NAMES_FILE = os.path.join(CHECKPOINT_DIR, 'feature_names.json')
EXTRACTION_STATUS_FILE = os.path.join(CHECKPOINT_DIR, 'extraction_status.json')

# Memory optimization: reduce sequences per repertoire
# 原本 1000 → 降到 500 減少 50% 記憶體使用
MAX_SEQS_PER_REPERTOIRE = 500

# Dataset 8 專用優化參數
DATASET8_GC_FREQUENCY = 30  # 每 30 個 repertoire 執行 GC
DATASET8_PROGRESS_SAVE_FREQUENCY = 100  # 每 100 個 repertoire 保存進度
LARGE_DATASET_THRESHOLD = 500  # 超過此數量的 repertoire 視為大型 dataset

# Set multiprocessing start method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# Checkpoint Functions
# ============================================================================

def ensure_checkpoint_dir():
    """確保 checkpoint 目錄存在"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_dataset_checkpoint_path(dataset_id: int) -> str:
    """獲取特定 dataset 的 checkpoint 文件路徑"""
    return os.path.join(CHECKPOINT_DIR, f'dataset_{dataset_id}.npz')


def save_dataset_checkpoint(dataset_id: int, dataset_data: List[Dict], all_feature_names: List[str]):
    """
    保存單個 dataset 的 feature extraction 結果
    使用 .npz 格式節省空間，每個 dataset 獨立保存
    """
    ensure_checkpoint_dir()

    # 分離 numpy arrays 和 metadata
    esm_embeddings_list = []
    trad_features_list = []
    labels = []
    repertoire_ids = []
    dataset_ids = []

    for item in dataset_data:
        esm_embeddings_list.append(item['esm_embeddings'])
        trad_features_list.append(item['trad_features'])
        labels.append(item['label'])
        repertoire_ids.append(item['repertoire_id'])
        dataset_ids.append(item['dataset_id'])

    # 保存到 .npz (壓縮的 numpy 格式)
    checkpoint_path = get_dataset_checkpoint_path(dataset_id)
    np.savez_compressed(
        checkpoint_path,
        esm_embeddings=np.array(esm_embeddings_list, dtype=object),
        trad_features=np.array(trad_features_list),
        labels=np.array(labels),
        repertoire_ids=np.array(repertoire_ids),
        dataset_ids=np.array(dataset_ids)
    )

    # 更新提取狀態
    status = load_extraction_status()
    status['completed_datasets'].append(dataset_id)
    status['completed_datasets'] = sorted(list(set(status['completed_datasets'])))
    status['timestamp'] = datetime.now().isoformat()
    save_extraction_status(status)

    # 保存 feature names (只需要保存一次)
    if not os.path.exists(FEATURE_NAMES_FILE):
        with open(FEATURE_NAMES_FILE, 'w') as f:
            json.dump(all_feature_names, f)

    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"💾 Dataset {dataset_id} saved: {len(dataset_data)} repertoires, {file_size:.1f} MB")


def load_dataset_checkpoint(dataset_id: int) -> List[Dict]:
    """
    載入單個 dataset 的數據
    支援兩種格式:
    - 標準格式: esm_embeddings, trad_features, labels, repertoire_ids, dataset_ids (分開的陣列)
    - Dataset 8 格式: processed_data (dict 列表)
    """
    checkpoint_path = get_dataset_checkpoint_path(dataset_id)

    if not os.path.exists(checkpoint_path):
        return []

    try:
        data = np.load(checkpoint_path, allow_pickle=True)

        dataset_data = []

        # Check which format we have
        if 'processed_data' in data.keys():
            # Dataset 8 format: list of dicts
            processed_data = data['processed_data']
            for item in processed_data:
                dataset_data.append({
                    'esm_embeddings': item['esm_embeddings'],
                    'trad_features': item['trad_features'],
                    'label': int(item['label']),
                    'repertoire_id': str(item['repertoire_id']),
                    'dataset_id': int(item['dataset_id'])
                })
            print(f"   ✓ Loaded Dataset {dataset_id} (format: processed_data): {len(dataset_data)} repertoires")
        else:
            # Standard format: separate arrays
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
        print(f"⚠️ Error loading dataset {dataset_id} checkpoint: {e}")
        return []


def save_extraction_status(status: Dict):
    """保存提取狀態"""
    ensure_checkpoint_dir()
    with open(EXTRACTION_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def load_extraction_status() -> Dict:
    """載入提取狀態"""
    if not os.path.exists(EXTRACTION_STATUS_FILE):
        return {'completed_datasets': [], 'timestamp': None}

    try:
        with open(EXTRACTION_STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {'completed_datasets': [], 'timestamp': None}


def load_feature_names() -> List[str]:
    """載入 feature names"""
    if not os.path.exists(FEATURE_NAMES_FILE):
        return []
    try:
        with open(FEATURE_NAMES_FILE, 'r') as f:
            return json.load(f)
    except:
        return []


def load_all_datasets_from_checkpoints() -> Tuple[List[Dict], List[str]]:
    """
    從 checkpoints 載入所有已處理的 datasets
    使用 lazy loading，一次只載入需要的數據
    """
    status = load_extraction_status()
    completed = status.get('completed_datasets', [])

    if not completed:
        return [], []

    all_data = []
    for dataset_id in sorted(completed):
        print(f"   Loading dataset {dataset_id} from checkpoint...")
        dataset_data = load_dataset_checkpoint(dataset_id)
        all_data.extend(dataset_data)
        print(f"   ✓ Loaded {len(dataset_data)} repertoires")

    feature_names = load_feature_names()

    return all_data, feature_names


def save_training_checkpoint(completed_folds: List[int], fold_results: List[Dict]):
    """
    保存 training 進度
    在每個 fold 訓練完後調用
    """
    ensure_checkpoint_dir()

    checkpoint = {
        'completed_folds': completed_folds,
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(TRAINING_CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"💾 Training checkpoint saved: Folds {completed_folds} complete")


def load_training_checkpoint() -> Tuple[List[int], List[Dict]]:
    """
    讀取 training checkpoint

    Returns:
        (completed_folds, fold_results)
        如果沒有 checkpoint，返回 ([], [])
    """
    if not os.path.exists(TRAINING_CHECKPOINT_FILE):
        return [], []

    try:
        with open(TRAINING_CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)

        completed_folds = checkpoint['completed_folds']
        fold_results = checkpoint['fold_results']
        timestamp = checkpoint.get('timestamp', 'unknown')

        print(f"📂 Found training checkpoint from {timestamp}")
        print(f"   Completed folds: {completed_folds}")

        return completed_folds, fold_results

    except Exception as e:
        print(f"⚠️ Error loading training checkpoint: {e}")
        return [], []


def clear_checkpoints():
    """清除所有 checkpoint (用於全新開始)"""
    import glob
    files_to_remove = [
        TRAINING_CHECKPOINT_FILE,
        FEATURE_NAMES_FILE,
        EXTRACTION_STATUS_FILE
    ]
    # 加入所有 dataset checkpoint 文件
    files_to_remove.extend(glob.glob(os.path.join(CHECKPOINT_DIR, 'dataset_*.npz')))

    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑️ Removed {f}")


def get_partial_checkpoint_path(dataset_id: int) -> str:
    """獲取 dataset 的部分進度 checkpoint 路徑"""
    return os.path.join(CHECKPOINT_DIR, f'dataset_{dataset_id}_partial.pkl')


def save_partial_progress(dataset_id: int, processed_data: List[Dict], processed_count: int):
    """
    保存大型 dataset 的中間處理進度
    用於 Dataset 8 等超大 dataset 的容錯處理
    """
    ensure_checkpoint_dir()
    partial_path = get_partial_checkpoint_path(dataset_id)

    checkpoint = {
        'dataset_id': dataset_id,
        'processed_count': processed_count,
        'processed_data': processed_data,
        'timestamp': datetime.now().isoformat()
    }

    with open(partial_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"   💾 Partial progress saved: {processed_count} repertoires")


def load_partial_progress(dataset_id: int) -> Tuple[List[Dict], int]:
    """
    載入 dataset 的部分處理進度

    Returns:
        (processed_data, processed_count) 或 ([], 0) 如果沒有部分進度
    """
    partial_path = get_partial_checkpoint_path(dataset_id)

    if not os.path.exists(partial_path):
        return [], 0

    try:
        with open(partial_path, 'rb') as f:
            checkpoint = pickle.load(f)

        processed_data = checkpoint['processed_data']
        processed_count = checkpoint['processed_count']
        timestamp = checkpoint.get('timestamp', 'unknown')

        print(f"   📂 Found partial progress from {timestamp}")
        print(f"   ✓ Resuming from repertoire {processed_count}")

        return processed_data, processed_count
    except Exception as e:
        print(f"   ⚠️ Error loading partial progress: {e}")
        return [], 0


def clear_partial_progress(dataset_id: int):
    """清除 dataset 的部分進度 checkpoint"""
    partial_path = get_partial_checkpoint_path(dataset_id)
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"   🗑️ Cleared partial progress for dataset {dataset_id}")


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

    def extract_embeddings(self, sequences: List[str], batch_size: int = 16, max_seqs: int = 1000) -> np.ndarray:
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

        # Clean sequences: remove invalid characters for ESM-2
        # ESM-2 only accepts standard amino acid letters (ACDEFGHIKLMNPQRSTVWY)
        # plus X for unknown, but NOT * or other special characters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        cleaned_sequences = []
        for seq in sequences:
            # Replace invalid characters with X (unknown amino acid)
            cleaned_seq = ''.join(c if c in valid_aa else 'X' for c in seq.upper())
            if len(cleaned_seq) > 0:
                cleaned_sequences.append(cleaned_seq)

        if len(cleaned_sequences) == 0:
            return np.zeros((1, 1280))  # Return dummy embedding

        sequences = cleaned_sequences

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

    # Drop NaN values
    df_clean = df['junction_aa'].dropna()
    if len(df_clean) == 0:
        return features

    seq_counts = df_clean.value_counts()
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

    # CDR3 length statistics (handle NaN safely)
    lengths = df_clean.str.len()
    features['mean_length'] = lengths.mean() if len(lengths) > 0 else 0.0
    features['std_length'] = lengths.std() if len(lengths) > 1 else 0.0
    features['min_length'] = lengths.min() if len(lengths) > 0 else 0.0
    features['max_length'] = lengths.max() if len(lengths) > 0 else 0.0

    # Top clone frequency
    features['top_clone_freq'] = frequencies[0] if len(frequencies) > 0 else 0.0

    # Replace any NaN values with 0
    features = {k: (0.0 if pd.isna(v) or np.isinf(v) else float(v)) for k, v in features.items()}

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

        # Replace NaN with 0
        all_features = {k: (0.0 if pd.isna(v) else float(v)) for k, v in all_features.items()}

        return all_features, df['junction_aa'].dropna().astype(str).tolist()
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(tsv_path)}: {e}")
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
            val = feature_dict[name]
            # Replace NaN/inf with 0
            if pd.isna(val) or np.isinf(val):
                feature_vector[i] = 0.0
            else:
                feature_vector[i] = float(val)
    return feature_vector


def process_single_repertoire(args):
    """Process a single repertoire - used by multiprocessing pool."""
    tsv_path, repertoire_id, label, all_feature_names, dataset_id = args

    try:
        # Extract traditional features and sequences
        trad_features_dict, sequences = extract_features_from_repertoire(tsv_path)

        if len(sequences) == 0:
            return None

        # Standardize traditional features
        trad_features = standardize_features(trad_features_dict, all_feature_names)

        return {
            'trad_features': trad_features,
            'label': label,
            'repertoire_id': repertoire_id,
            'dataset_id': dataset_id,
            'sequences': sequences[:MAX_SEQS_PER_REPERTOIRE],  # Store for ESM-2 (memory optimized)
            'tsv_path': tsv_path  # Keep path for ESM-2 processing later
        }
    except Exception as e:
        print(f"❌ Error processing {repertoire_id}: {e}")
        return None


def load_dataset(dataset_path: str, dataset_id: int, esm_extractor: ESM2FeatureExtractor,
                 all_feature_names: List[str], num_workers: int = 6) -> List[Dict]:
    """
    Load one dataset and extract all features with parallel processing.

    針對 Dataset 8 等大型 dataset 進行了以下優化：
    1. 更頻繁的 GC（每 30 個 repertoire）
    2. 中間進度保存（每 100 個 repertoire）
    3. 支援從中斷點恢復
    4. 詳細的進度和記憶體日誌
    """
    print(f"\n📂 Loading dataset {dataset_id} from {dataset_path}")

    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    total_repertoires = len(metadata)

    # 判斷是否為大型 dataset
    is_large_dataset = total_repertoires > LARGE_DATASET_THRESHOLD
    gc_frequency = DATASET8_GC_FREQUENCY if is_large_dataset else 50
    save_frequency = DATASET8_PROGRESS_SAVE_FREQUENCY if is_large_dataset else total_repertoires + 1

    if is_large_dataset:
        print(f"   ⚠️ Large dataset detected ({total_repertoires} repertoires)")
        print(f"   📋 GC frequency: every {gc_frequency} repertoires")
        print(f"   💾 Progress save: every {save_frequency} repertoires")

    # 檢查是否有部分進度可以恢復
    partial_data, start_idx = load_partial_progress(dataset_id)

    # Prepare arguments for parallel processing
    args_list = []
    for idx, row in metadata.iterrows():
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = 1 if row['label_positive'] else 0
        tsv_path = os.path.join(dataset_path, filename)

        if os.path.exists(tsv_path):
            args_list.append((tsv_path, repertoire_id, label, all_feature_names, dataset_id))

    # Process traditional features in parallel
    print(f"   Processing {len(args_list)} repertoires with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_repertoire, args_list),
            total=len(args_list),
            desc=f"Dataset {dataset_id} (Traditional)"
        ))

    # Filter out None results
    repertoire_data = [r for r in results if r is not None]
    print(f"   Successfully processed {len(repertoire_data)}/{len(args_list)} repertoires")

    # 強制 GC 在開始 ESM-2 處理前
    gc.collect()
    torch.cuda.empty_cache()

    # 報告當前記憶體狀態
    mem = psutil.virtual_memory()
    print(f"   💾 RAM before ESM-2: {mem.used/1e9:.1f} GB / {mem.total/1e9:.1f} GB ({mem.percent}%)")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   🎮 GPU: {gpu_mem:.1f} GB / {gpu_total:.1f} GB")

    # Now process ESM-2 embeddings (GPU operation with optimized batch size)
    print(f"   Extracting ESM-2 embeddings (max_seqs={MAX_SEQS_PER_REPERTOIRE})...")

    # 如果有部分進度，跳過已處理的部分
    if start_idx > 0:
        print(f"   ⏭️ Skipping first {start_idx} repertoires (already processed)")
        # 將已處理的數據放入結果
        for i in range(start_idx):
            if i < len(partial_data):
                repertoire_data[i]['esm_embeddings'] = partial_data[i]['esm_embeddings']
                if 'sequences' in repertoire_data[i]:
                    del repertoire_data[i]['sequences']
                if 'tsv_path' in repertoire_data[i]:
                    del repertoire_data[i]['tsv_path']

    # 處理 ESM-2 embeddings
    processed_count = start_idx
    for idx in tqdm(range(start_idx, len(repertoire_data)), desc=f"Dataset {dataset_id} (ESM-2)",
                    initial=start_idx, total=len(repertoire_data)):
        rep_data = repertoire_data[idx]

        try:
            sequences = rep_data['sequences']
            # Extract ESM-2 embeddings (batch_size=16, reduced max_seqs for memory)
            esm_embeddings = esm_extractor.extract_embeddings(
                sequences, batch_size=16, max_seqs=MAX_SEQS_PER_REPERTOIRE
            )
            rep_data['esm_embeddings'] = esm_embeddings

            # Remove sequences and tsv_path to free memory (critical for Dataset 7/8)
            if 'sequences' in rep_data:
                del rep_data['sequences']
            if 'tsv_path' in rep_data:
                del rep_data['tsv_path']

            processed_count += 1

        except Exception as e:
            print(f"\n   ❌ Error processing repertoire {idx}: {e}")
            # 保存當前進度並繼續
            if is_large_dataset and processed_count > 0:
                save_partial_progress(dataset_id, repertoire_data[:processed_count], processed_count)
            raise

        # Periodic memory cleanup (更頻繁的 GC)
        if processed_count > 0 and processed_count % gc_frequency == 0:
            gc.collect()
            torch.cuda.empty_cache()

            # 報告進度
            progress_pct = processed_count / len(repertoire_data) * 100
            mem = psutil.virtual_memory()
            print(f"\n   📊 Progress: {processed_count}/{len(repertoire_data)} ({progress_pct:.1f}%)")
            print(f"   💾 RAM: {mem.used/1e9:.1f} GB ({mem.percent}%)")

        # 定期保存進度（僅大型 dataset）
        if is_large_dataset and processed_count > 0 and processed_count % save_frequency == 0:
            save_partial_progress(dataset_id, repertoire_data[:processed_count], processed_count)

    # 處理完成，清除部分進度檔案
    if is_large_dataset:
        clear_partial_progress(dataset_id)

    print(f"✓ Loaded {len(repertoire_data)} repertoires from dataset {dataset_id}")

    # 最終記憶體狀態
    mem = psutil.virtual_memory()
    print(f"   💾 Final RAM: {mem.used/1e9:.1f} GB / {mem.total/1e9:.1f} GB ({mem.percent}%)")

    return repertoire_data


def load_all_training_data(train_root: str, esm_extractor: ESM2FeatureExtractor,
                          resume_from_checkpoint: bool = True) -> Tuple[List[Dict], List[str]]:
    """
    Load all 8 training datasets with memory-optimized checkpoint support.

    Uses per-dataset .npz files to avoid memory explosion with large datasets.

    Args:
        train_root: Path to training data root
        esm_extractor: ESM-2 feature extractor instance
        resume_from_checkpoint: Whether to resume from existing checkpoint

    Returns:
        (all_data, all_feature_names)
    """
    print("\n" + "="*70)
    print("📊 LOADING ALL TRAINING DATA (MEMORY-OPTIMIZED CHECKPOINT)")
    print("="*70)
    print(f"   Max sequences per repertoire: {MAX_SEQS_PER_REPERTOIRE}")
    print(f"   Checkpoint directory: {CHECKPOINT_DIR}")

    # Check for existing extraction status
    all_data = []
    all_feature_names = load_feature_names()
    extraction_status = load_extraction_status()
    completed_datasets = extraction_status.get('completed_datasets', [])

    if resume_from_checkpoint and completed_datasets:
        print(f"\n📂 Found checkpoint: Datasets {completed_datasets} already processed")

        if len(completed_datasets) >= 8:
            # All datasets done, load from checkpoints
            print("✅ All datasets already processed! Loading from checkpoints...")
            all_data, all_feature_names = load_all_datasets_from_checkpoints()
            print(f"✅ Loaded {len(all_data)} repertoires from checkpoints")
            return all_data, all_feature_names

        # Load already processed datasets
        print(f"🔄 Loading completed datasets from checkpoints...")
        for dataset_id in sorted(completed_datasets):
            dataset_data = load_dataset_checkpoint(dataset_id)
            all_data.extend(dataset_data)
            print(f"   ✓ Dataset {dataset_id}: {len(dataset_data)} repertoires")

        print(f"   Total loaded: {len(all_data)} repertoires")

    # First pass: collect all feature names (if not loaded from checkpoint)
    if not all_feature_names:
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

        # Save feature names immediately
        ensure_checkpoint_dir()
        with open(FEATURE_NAMES_FILE, 'w') as f:
            json.dump(all_feature_names, f)
    else:
        print(f"✓ Using cached feature names ({len(all_feature_names)} features)")

    # Second pass: load remaining datasets with standardized features
    remaining_datasets = [d for d in range(1, 9) if d not in completed_datasets]
    if remaining_datasets:
        print(f"\n📥 Phase 2: Processing datasets {remaining_datasets} with ESM-2 extraction...")

    for dataset_id in remaining_datasets:
        print(f"\n{'='*50}")
        print(f"📂 Processing Dataset {dataset_id}/8")
        print(f"{'='*50}")

        dataset_path = os.path.join(train_root, f'train_dataset_{dataset_id}')
        dataset_data = load_dataset(dataset_path, dataset_id, esm_extractor, all_feature_names)

        # 💾 Save checkpoint IMMEDIATELY after each dataset (before extending all_data)
        # This ensures we can recover even if crash happens during memory operations
        save_dataset_checkpoint(dataset_id, dataset_data, all_feature_names)

        all_data.extend(dataset_data)

        # Force garbage collection after each dataset
        gc.collect()
        torch.cuda.empty_cache()

        print(f"✅ Dataset {dataset_id} complete. Total: {len(all_data)} repertoires")

        # Report memory status
        mem = psutil.virtual_memory()
        print(f"   💾 RAM: {mem.used/1e9:.1f} GB / {mem.total/1e9:.1f} GB ({mem.percent}%)")

    print(f"\n{'='*70}")
    print(f"✅ ALL DATASETS LOADED: {len(all_data)} repertoires across 8 datasets")
    print(f"{'='*70}")
    return all_data, all_feature_names


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, use_amp=True):
    """Train for one epoch with mixed precision support."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        esm_emb = batch['esm_embeddings'].to(device)
        trad_feat = batch['trad_features'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(esm_emb, trad_feat, masks)
                logits = logits.squeeze()
                loss = criterion(logits, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            logits, _ = model(esm_emb, trad_feat, masks)
            logits = logits.squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Record metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        # Clear GPU cache periodically
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

    # Reduced batch size to prevent OOM on RTX 5080 16GB
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_repertoires, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                           collate_fn=collate_repertoires, num_workers=2)

    # Initialize model
    model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim,
                                   hidden_dims=[512, 256], dropout=0.3).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler (removed verbose parameter for PyTorch compatibility)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
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
    """Championship training pipeline with checkpoint support."""

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏆 AIRR-ML-25 Championship Deep Learning Pipeline 🏆      ║
    ║                                                              ║
    ║  Target: Beat GROZD (0.81364) → Achieve 0.82+               ║
    ║  Method: ESM-2 + Attention + Hybrid Features                ║
    ║  GPU: RTX 5080 16GB                                         ║
    ║                                                              ║
    ║  🔄 CHECKPOINT SYSTEM ENABLED                               ║
    ║  - Auto-saves after each dataset (feature extraction)       ║
    ║  - Auto-saves after each fold (training)                    ║
    ║  - Auto-resumes from last checkpoint on restart             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Paths
    train_root = './data/train_datasets/train_datasets'
    test_root = './data/test_datasets/test_datasets'

    # Initialize ESM-2 extractor
    print("\n🔧 Initializing ESM-2 Feature Extractor...")
    esm_extractor = ESM2FeatureExtractor(model_name="esm2_t33_650M_UR50D", device=device)

    # Load all training data (with checkpoint support)
    all_data, all_feature_names = load_all_training_data(train_root, esm_extractor, resume_from_checkpoint=True)
    trad_dim = len(all_feature_names)
    print(f"\n📊 Traditional feature dimension: {trad_dim}")

    # Check for existing training checkpoint
    completed_folds, fold_results = load_training_checkpoint()

    # Leave-one-dataset-out cross-validation
    print("\n" + "="*70)
    print("🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION (WITH CHECKPOINT)")
    print("="*70)

    if completed_folds:
        print(f"🔄 Resuming training from fold {max(completed_folds) + 1}")
        print(f"   Already completed folds: {completed_folds}")

    trained_models = []

    for test_dataset_id in range(1, 9):
        # Skip already completed folds
        if test_dataset_id in completed_folds:
            print(f"\n⏭️ Fold {test_dataset_id} already complete, skipping...")
            continue

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

        # Update results
        fold_results.append({'fold': test_dataset_id, 'val_auc': val_auc})
        completed_folds.append(test_dataset_id)
        trained_models.append(model)

        # 💾 Save training checkpoint after each fold
        save_training_checkpoint(completed_folds, fold_results)

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Print CV results
    print("\n" + "="*70)
    print("📈 CROSS-VALIDATION RESULTS")
    print("="*70)
    for result in sorted(fold_results, key=lambda x: x['fold']):
        print(f"Fold {result['fold']}: Val AUC = {result['val_auc']:.4f}")

    mean_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    print(f"\n🎯 Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Save final summary
    ensure_checkpoint_dir()
    summary = {
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'fold_results': fold_results,
        'completed_at': datetime.now().isoformat()
    }
    with open(os.path.join(CHECKPOINT_DIR, 'training_complete.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Training complete! All 8 folds finished.")
    print(f"   Models saved in: ./models/championship_fold*.pt")
    print(f"   Summary saved in: {CHECKPOINT_DIR}/training_complete.json")
    print("   Next step: Run generate_predictions() to create submission file.")


if __name__ == '__main__':
    main()
