#!/usr/bin/env python3
"""
🏆 極致並行系統：2x GPU + 48 CPU 核心全部榨乾
- GPU: 雙進程並行 ESM-2 特徵提取
- CPU: 48 核心並行處理傳統特徵與 AA 性質
- 訓練: DataParallel 跨雙 GPU + 多線程 DataLoader
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
from torch.nn import DataParallel
from torch.multiprocessing import Pool, Queue, Process, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Maximum parallelism
NUM_CPUS = 48  # 全部 48 核心
NUM_GPUS = 2   # 雙 GPU

# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 極致並行系統")
print(f"   CPUs: {NUM_CPUS} cores")
print(f"   GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

# Import ESM-2
try:
    import esm
    print("✓ ESM-2 loaded")
except ImportError:
    os.system("pip install fair-esm")
    import esm

# Load modules
sys.path.insert(0, './src')
from airr_ml25.data import load_all_datasets
from airr_ml25.features.aa_props import compute_seq_summary
from airr_ml25.schema import resolve_metadata_schema


# ============================================================================
# GPU Workers for ESM-2 (2 processes, each on one GPU)
# ============================================================================

def extract_features_gpu_worker(gpu_id, task_queue, result_queue):
    """GPU Worker: Extract ESM-2 embeddings"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"🔧 GPU {gpu_id} Worker started")

    # Load ESM-2
    model_name = "esm2_t33_650M_UR50D"
    layer = 6
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    print(f"✓ GPU {gpu_id} ESM-2 ready")

    processed = 0

    while True:
        task = task_queue.get()

        if task is None:  # Poison pill
            print(f"✓ GPU {gpu_id} Worker finished ({processed} repertoires)")
            break

        rep_id, sequences = task

        # Extract ESM-2
        max_seqs = 1000
        if len(sequences) > max_seqs:
            sequences = np.random.choice(sequences, max_seqs, replace=False).tolist()

        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        clean_seqs = [''.join(c if c in valid_aa else 'X' for c in s.upper()) for s in sequences]
        clean_seqs = [s for s in clean_seqs if len(s) > 0]

        if len(clean_seqs) == 0:
            esm2_emb = np.zeros(1280)
        else:
            embeddings = []
            batch_size = 32

            with torch.no_grad():
                for i in range(0, len(clean_seqs), batch_size):
                    batch = clean_seqs[i:i+batch_size]
                    labels_list = [(f"s{j}", s) for j, s in enumerate(batch)]

                    _, _, tokens = batch_converter(labels_list)
                    tokens = tokens.to(device)

                    results = model(tokens, repr_layers=[layer], return_contacts=False)
                    repr = results["representations"][layer]

                    for j, seq in enumerate(batch):
                        seq_repr = repr[j, 1:len(seq)+1].mean(0)
                        embeddings.append(seq_repr.cpu().numpy())

                    del tokens, results, repr
                    torch.cuda.empty_cache()

            esm2_emb = np.array(embeddings).mean(axis=0)

        result_queue.put({
            'repertoire_id': rep_id,
            'esm2_embedding': esm2_emb
        })

        processed += 1


# ============================================================================
# CPU Workers for Traditional Features (使用 48 核心)
# ============================================================================

def extract_traditional_features_parallel(sequences_df):
    """Extract V/J usage, diversity (CPU-bound)"""
    features = {}

    total = len(sequences_df)
    if total == 0:
        return features

    # V/J gene usage
    if 'v_call' in sequences_df.columns:
        v_counts = sequences_df['v_call'].value_counts().head(30)
        for gene, count in v_counts.items():
            if pd.notna(gene):
                features[f"v_{gene}"] = count / total

    if 'j_call' in sequences_df.columns:
        j_counts = sequences_df['j_call'].value_counts().head(15)
        for gene, count in j_counts.items():
            if pd.notna(gene):
                features[f"j_{gene}"] = count / total

    # Diversity metrics
    if 'junction_aa' in sequences_df.columns:
        seqs = sequences_df['junction_aa'].dropna()
        if len(seqs) > 0:
            unique_ratio = seqs.nunique() / len(seqs)
            features['unique_ratio'] = unique_ratio
            features['n_sequences'] = len(seqs)

            lengths = seqs.str.len()
            features['mean_length'] = lengths.mean()
            features['std_length'] = lengths.std() if len(lengths) > 1 else 0.0

    return features


def extract_aa_property_features_parallel(sequences_df):
    """Extract AA chemical properties (CPU-bound)"""
    features = {}

    if 'junction_aa' not in sequences_df.columns or len(sequences_df) == 0:
        return features

    seqs = sequences_df['junction_aa'].dropna().tolist()

    if len(seqs) == 0:
        return features

    # Parallel AA summary computation
    summaries = [compute_seq_summary(s) for s in seqs[:1000]]
    summary_df = pd.DataFrame(summaries)

    for col in summary_df.columns:
        features[f"aa_{col}_mean"] = summary_df[col].mean()
        features[f"aa_{col}_std"] = summary_df[col].std()

    return features


def process_cpu_features_batch(batch_data):
    """Process a batch of repertoires' traditional + AA features"""
    results = []

    for item in batch_data:
        rep_id = item['repertoire_id']
        seqs_df = item['sequences_df']

        trad_feat = extract_traditional_features_parallel(seqs_df)
        aa_feat = extract_aa_property_features_parallel(seqs_df)

        # Pad to fixed size
        trad_vec = np.zeros(100)
        aa_vec = np.zeros(20)

        for i, (k, v) in enumerate(list(trad_feat.items())[:100]):
            trad_vec[i] = v

        for i, (k, v) in enumerate(list(aa_feat.items())[:20]):
            aa_vec[i] = v

        results.append({
            'repertoire_id': rep_id,
            'traditional_features': trad_vec,
            'aa_features': aa_vec
        })

    return results


# ============================================================================
# Model
# ============================================================================

class RepertoireDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        esm2_emb = item.get('esm2_embedding', np.zeros(1280))
        trad_feat = item.get('traditional_features', np.zeros(100))
        aa_feat = item.get('aa_features', np.zeros(20))
        features = np.concatenate([esm2_emb, trad_feat, aa_feat])
        label = item['label']
        return torch.FloatTensor(features), torch.FloatTensor([label])


class HybridMILModel(nn.Module):
    def __init__(self, esm2_dim=1280, trad_dim=100, aa_dim=20, hidden_dim=256):
        super().__init__()
        input_dim = esm2_dim + trad_dim + aa_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.classifier(h)
        return out


# ============================================================================
# Training with multi-threaded DataLoader
# ============================================================================

def train_fold(fold_id, train_data, val_data, epochs=30):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_id}: Training {len(train_data)} reps, validating {len(val_data)}")
    print(f"{'='*70}")

    train_dataset = RepertoireDataset(train_data)
    val_dataset = RepertoireDataset(val_data)

    # Multi-threaded DataLoader (榨乾 CPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,  # 使用 16 個 worker threads
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # DataParallel across both GPUs
    model = HybridMILModel()
    if torch.cuda.device_count() > 1:
        print(f"   Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_auc = 0.0
    patience_counter = 0
    patience = 8

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                outputs = model(features)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_auc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                'auc': best_auc,
                'fold': fold_id
            }, f'./models/fold_{fold_id}_best.pt')
            patience_counter = 0
            print(f"   ✓ New best: {best_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Fold {fold_id} complete. Best AUC: {best_auc:.4f}")
    return best_auc


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏆 極致並行系統 🏆                                         ║
    ║                                                              ║
    ║  GPU: 2x RTX 6000 Ada (並行 ESM-2 特徵提取)                ║
    ║  CPU: 48 cores (並行傳統特徵與 AA 性質)                    ║
    ║  訓練: DataParallel + Multi-threaded DataLoader            ║
    ║  目標: 奪冠 (beat 0.84590)                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Load datasets
    print("\n📊 Loading datasets...")
    datasets = load_all_datasets(
        './data/train_datasets/train_datasets',
        max_seqs_per_repertoire=1000,
        verbose=True
    )

    # Prepare metadata and tasks
    print("\n🔧 Preparing parallel tasks...")
    gpu_tasks = []
    cpu_tasks = []
    metadata_map = {}

    for ds in datasets:
        metadata = ds['metadata']
        schema = resolve_metadata_schema(metadata)

        for _, row in metadata.iterrows():
            rep_id = row[schema.repertoire_id]
            label = row[schema.label]

            seqs_df = ds['sequences'].get(rep_id)
            if seqs_df is None or len(seqs_df) == 0:
                continue

            sequences = seqs_df['junction_aa'].dropna().tolist()

            # GPU task: ESM-2 extraction
            gpu_tasks.append((rep_id, sequences))

            # CPU task: Traditional + AA features
            cpu_tasks.append({
                'repertoire_id': rep_id,
                'sequences_df': seqs_df
            })

            # Metadata
            metadata_map[rep_id] = {
                'dataset_id': ds['dataset_id'],
                'label': 1.0 if label else 0.0
            }

    print(f"   GPU tasks: {len(gpu_tasks)}")
    print(f"   CPU tasks: {len(cpu_tasks)}")

    # ========================================================================
    # PHASE 1: Parallel GPU extraction (2 GPUs)
    # ========================================================================
    print(f"\n⚡ Phase 1: Parallel ESM-2 extraction (2 GPUs)...")

    mp.set_start_method('spawn', force=True)
    manager = Manager()
    gpu_task_queue = manager.Queue()
    gpu_result_queue = manager.Queue()

    # Fill GPU task queue
    for task in gpu_tasks:
        gpu_task_queue.put(task)

    # Poison pills
    for _ in range(NUM_GPUS):
        gpu_task_queue.put(None)

    # Start GPU workers
    gpu_processes = []
    for gpu_id in range(NUM_GPUS):
        p = Process(target=extract_features_gpu_worker, args=(gpu_id, gpu_task_queue, gpu_result_queue))
        p.start()
        gpu_processes.append(p)

    # Collect GPU results
    gpu_results = {}
    with tqdm(total=len(gpu_tasks), desc="GPU: ESM-2") as pbar:
        for _ in range(len(gpu_tasks)):
            result = gpu_result_queue.get()
            gpu_results[result['repertoire_id']] = result['esm2_embedding']
            pbar.update(1)

    # Wait for GPU workers
    for p in gpu_processes:
        p.join()

    print(f"✓ GPU extraction complete: {len(gpu_results)} repertoires")

    # ========================================================================
    # PHASE 2: Parallel CPU extraction (48 cores)
    # ========================================================================
    print(f"\n⚡ Phase 2: Parallel traditional + AA features (48 CPU cores)...")

    # Split into batches for 48 workers
    batch_size = max(1, len(cpu_tasks) // NUM_CPUS)
    cpu_batches = [cpu_tasks[i:i+batch_size] for i in range(0, len(cpu_tasks), batch_size)]

    cpu_results = {}
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = [executor.submit(process_cpu_features_batch, batch) for batch in cpu_batches]

        with tqdm(total=len(futures), desc="CPU: Traditional+AA") as pbar:
            for future in futures:
                batch_results = future.result()
                for res in batch_results:
                    cpu_results[res['repertoire_id']] = {
                        'traditional_features': res['traditional_features'],
                        'aa_features': res['aa_features']
                    }
                pbar.update(1)

    print(f"✓ CPU extraction complete: {len(cpu_results)} repertoires")

    # ========================================================================
    # PHASE 3: Merge all features
    # ========================================================================
    print(f"\n🔧 Merging features...")
    all_data = []

    for rep_id in gpu_results.keys():
        if rep_id not in cpu_results or rep_id not in metadata_map:
            continue

        all_data.append({
            'dataset_id': metadata_map[rep_id]['dataset_id'],
            'repertoire_id': rep_id,
            'label': metadata_map[rep_id]['label'],
            'esm2_embedding': gpu_results[rep_id],
            'traditional_features': cpu_results[rep_id]['traditional_features'],
            'aa_features': cpu_results[rep_id]['aa_features']
        })

    print(f"✓ Complete features for {len(all_data)} repertoires")

    # ========================================================================
    # PHASE 4: LODO CV with DataParallel training
    # ========================================================================
    print("\n🎓 Leave-One-Dataset-Out CV with DataParallel")
    fold_aucs = []

    for test_ds_id in range(1, 9):
        train_data = [d for d in all_data if d['dataset_id'] != test_ds_id]
        val_data = [d for d in all_data if d['dataset_id'] == test_ds_id]

        if len(val_data) == 0:
            continue

        auc = train_fold(test_ds_id, train_data, val_data, epochs=30)
        fold_aucs.append(auc)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Fold AUCs: {fold_aucs}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    os.makedirs('./models', exist_ok=True)
    main()
