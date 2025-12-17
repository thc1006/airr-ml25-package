#!/usr/bin/env python3
"""
🎯 Dataset 8 專用特徵提取腳本
僅處理 Dataset 8，避免 OOM 問題

特點：
- 不載入其他 datasets 的 checkpoints
- 更頻繁的 GC（每 30 個 repertoire）
- 中間進度保存（每 100 個 repertoire）
- 詳細的記憶體監控
"""

import os
import sys
import gc
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import Counter
from scipy.stats import entropy
import psutil
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR = './checkpoints'
FEATURE_NAMES_FILE = os.path.join(CHECKPOINT_DIR, 'feature_names.json')
EXTRACTION_STATUS_FILE = os.path.join(CHECKPOINT_DIR, 'extraction_status.json')

MAX_SEQS_PER_REPERTOIRE = 500
GC_FREQUENCY = 30  # 每 30 個 repertoire 執行 GC
PROGRESS_SAVE_FREQUENCY = 100  # 每 100 個 repertoire 保存進度

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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_dataset_checkpoint_path(dataset_id: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f'dataset_{dataset_id}.npz')


def save_dataset_checkpoint(dataset_id: int, dataset_data: List[Dict], all_feature_names: List[str]):
    ensure_checkpoint_dir()

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

    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"💾 Dataset {dataset_id} saved: {len(dataset_data)} repertoires, {file_size:.1f} MB")


def load_extraction_status() -> Dict:
    if not os.path.exists(EXTRACTION_STATUS_FILE):
        return {'completed_datasets': [], 'timestamp': None}
    try:
        with open(EXTRACTION_STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {'completed_datasets': [], 'timestamp': None}


def save_extraction_status(status: Dict):
    ensure_checkpoint_dir()
    with open(EXTRACTION_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def load_feature_names() -> List[str]:
    if not os.path.exists(FEATURE_NAMES_FILE):
        return []
    try:
        with open(FEATURE_NAMES_FILE, 'r') as f:
            return json.load(f)
    except:
        return []


def get_partial_checkpoint_path(dataset_id: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f'dataset_{dataset_id}_partial.pkl')


def save_partial_progress(dataset_id: int, processed_data: List[Dict], processed_count: int):
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
    partial_path = get_partial_checkpoint_path(dataset_id)
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"   🗑️ Cleared partial progress for dataset {dataset_id}")


# ============================================================================
# ESM-2 Feature Extractor
# ============================================================================

class ESM2FeatureExtractor:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda"):
        self.device = device
        print(f"Loading ESM-2 model: {model_name}...")

        import esm
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print(f"✓ ESM-2 loaded successfully on {device}")

    def extract_embeddings(self, sequences: List[str], batch_size: int = 16, max_seqs: int = 500) -> np.ndarray:
        if len(sequences) > max_seqs:
            np.random.seed(42)
            indices = np.random.choice(len(sequences), max_seqs, replace=False)
            sequences = [sequences[i] for i in sorted(indices)]

        embeddings = []
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        cleaned_sequences = []
        for seq in sequences:
            cleaned_seq = ''.join(c if c in valid_aa else 'X' for c in seq.upper())
            if len(cleaned_seq) > 0:
                cleaned_sequences.append(cleaned_seq)

        if len(cleaned_sequences) == 0:
            return np.zeros((1, 1280))

        sequences = cleaned_sequences

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
# Feature Extraction Functions
# ============================================================================

def extract_vj_features(df: pd.DataFrame, top_n: int = 50) -> Dict[str, float]:
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


def extract_features_from_repertoire(tsv_path: str) -> Tuple[Dict[str, float], List[str]]:
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        vj_features = extract_vj_features(df)
        clonality_features = extract_clonality_features(df)
        all_features = {**vj_features, **clonality_features}
        all_features = {k: (0.0 if pd.isna(v) else float(v)) for k, v in all_features.items()}
        return all_features, df['junction_aa'].dropna().astype(str).tolist()
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(tsv_path)}: {e}")
        return {}, []


def standardize_features(feature_dict: Dict[str, float], all_feature_names: List[str]) -> np.ndarray:
    feature_vector = np.zeros(len(all_feature_names))
    for i, name in enumerate(all_feature_names):
        if name in feature_dict:
            val = feature_dict[name]
            if pd.isna(val) or np.isinf(val):
                feature_vector[i] = 0.0
            else:
                feature_vector[i] = float(val)
    return feature_vector


# ============================================================================
# Main: Process Dataset 8 Only
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🎯 Dataset 8 專用特徵提取腳本                              ║
    ║                                                              ║
    ║  僅處理 Dataset 8，避免 OOM 問題                            ║
    ║  - GC 頻率: 每 30 個 repertoire                             ║
    ║  - 進度保存: 每 100 個 repertoire                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 檢查 Dataset 8 是否已完成
    status = load_extraction_status()
    if 8 in status.get('completed_datasets', []):
        print("✅ Dataset 8 already processed! Nothing to do.")
        return

    # 載入 feature names
    all_feature_names = load_feature_names()
    if not all_feature_names:
        print("❌ Feature names not found! Please run the main pipeline first to extract feature names.")
        return

    print(f"✓ Loaded {len(all_feature_names)} feature names")

    # 初始化 ESM-2
    print("\n🔧 Initializing ESM-2 Feature Extractor...")
    esm_extractor = ESM2FeatureExtractor(model_name="esm2_t33_650M_UR50D", device=device)

    # 報告記憶體狀態
    mem = psutil.virtual_memory()
    print(f"\n💾 RAM: {mem.used/1e9:.1f} GB / {mem.total/1e9:.1f} GB ({mem.percent}%)")

    # 處理 Dataset 8
    dataset_id = 8
    dataset_path = './data/train_datasets/train_datasets/train_dataset_8'

    print(f"\n{'='*70}")
    print(f"📂 Processing Dataset {dataset_id}")
    print(f"{'='*70}")

    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    total_repertoires = len(metadata)

    print(f"   Total repertoires: {total_repertoires}")
    print(f"   GC frequency: every {GC_FREQUENCY} repertoires")
    print(f"   Progress save: every {PROGRESS_SAVE_FREQUENCY} repertoires")

    # 檢查部分進度
    partial_data, start_idx = load_partial_progress(dataset_id)

    # 準備數據
    args_list = []
    for idx, row in metadata.iterrows():
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = 1 if row['label_positive'] else 0
        tsv_path = os.path.join(dataset_path, filename)

        if os.path.exists(tsv_path):
            args_list.append((tsv_path, repertoire_id, label, dataset_id))

    print(f"   Found {len(args_list)} valid repertoires")

    # Phase 1: 提取傳統特徵（串行處理，避免 multiprocessing 記憶體問題）
    print(f"\n   Phase 1: Extracting traditional features...")
    repertoire_data = []

    for idx, (tsv_path, repertoire_id, label, ds_id) in enumerate(tqdm(args_list, desc="Traditional Features")):
        try:
            trad_features_dict, sequences = extract_features_from_repertoire(tsv_path)

            if len(sequences) == 0:
                continue

            trad_features = standardize_features(trad_features_dict, all_feature_names)

            repertoire_data.append({
                'trad_features': trad_features,
                'label': label,
                'repertoire_id': repertoire_id,
                'dataset_id': ds_id,
                'sequences': sequences[:MAX_SEQS_PER_REPERTOIRE]
            })

        except Exception as e:
            print(f"   ❌ Error processing {repertoire_id}: {e}")
            continue

        # 定期 GC
        if (idx + 1) % GC_FREQUENCY == 0:
            gc.collect()

    print(f"   ✓ Processed {len(repertoire_data)}/{len(args_list)} repertoires")

    # Phase 2: ESM-2 embeddings
    print(f"\n   Phase 2: Extracting ESM-2 embeddings...")

    gc.collect()
    torch.cuda.empty_cache()

    mem = psutil.virtual_memory()
    print(f"   💾 RAM before ESM-2: {mem.used/1e9:.1f} GB ({mem.percent}%)")

    processed_count = start_idx
    for idx in tqdm(range(start_idx, len(repertoire_data)), desc="ESM-2",
                    initial=start_idx, total=len(repertoire_data)):
        rep_data = repertoire_data[idx]

        try:
            sequences = rep_data['sequences']
            esm_embeddings = esm_extractor.extract_embeddings(
                sequences, batch_size=16, max_seqs=MAX_SEQS_PER_REPERTOIRE
            )
            rep_data['esm_embeddings'] = esm_embeddings

            if 'sequences' in rep_data:
                del rep_data['sequences']

            processed_count += 1

        except Exception as e:
            print(f"\n   ❌ Error processing repertoire {idx}: {e}")
            if processed_count > 0:
                save_partial_progress(dataset_id, repertoire_data[:processed_count], processed_count)
            raise

        # 定期 GC
        if processed_count > 0 and processed_count % GC_FREQUENCY == 0:
            gc.collect()
            torch.cuda.empty_cache()

            progress_pct = processed_count / len(repertoire_data) * 100
            mem = psutil.virtual_memory()
            print(f"\n   📊 Progress: {processed_count}/{len(repertoire_data)} ({progress_pct:.1f}%)")
            print(f"   💾 RAM: {mem.used/1e9:.1f} GB ({mem.percent}%)")

        # 定期保存進度
        if processed_count > 0 and processed_count % PROGRESS_SAVE_FREQUENCY == 0:
            save_partial_progress(dataset_id, repertoire_data[:processed_count], processed_count)

    # 保存最終 checkpoint
    print(f"\n   Saving final checkpoint...")
    save_dataset_checkpoint(dataset_id, repertoire_data, all_feature_names)

    # 清除部分進度
    clear_partial_progress(dataset_id)

    print(f"\n{'='*70}")
    print(f"✅ Dataset 8 COMPLETE!")
    print(f"   Processed: {len(repertoire_data)} repertoires")
    print(f"{'='*70}")

    mem = psutil.virtual_memory()
    print(f"   💾 Final RAM: {mem.used/1e9:.1f} GB ({mem.percent}%)")


if __name__ == '__main__':
    main()
