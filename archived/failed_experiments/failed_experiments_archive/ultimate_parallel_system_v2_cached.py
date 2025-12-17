#!/usr/bin/env python3
"""
🏆 極致並行系統 V2（帶完整快取機制）
- 所有特徵提取結果自動保存
- 支援斷點續訓
- 避免重複計算
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.multiprocessing import Process, Manager
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
import multiprocessing as mp
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Import cache manager
sys.path.insert(0, './')
from cache_manager import CacheManager, FeatureCacheManager, save_checkpoint, load_checkpoint

# GPU configuration
NUM_CPUS = 48
NUM_GPUS = 2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize cache managers
cache_mgr = CacheManager(cache_dir='./cache')
feature_cache = FeatureCacheManager(cache_dir='./cache/features')

print(f"🚀 極致並行系統 V2（帶快取）")
print(f"   CPUs: {NUM_CPUS} cores")
print(f"   GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

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
# GPU Workers with caching
# ============================================================================

def extract_features_gpu_worker_cached(gpu_id, task_queue, result_queue, cache_dict):
    """GPU Worker with cache support"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"🔧 GPU {gpu_id} Worker started (with cache)")

    # Load ESM-2
    model_name = "esm2_t33_650M_UR50D"
    layer = 6
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    print(f"✓ GPU {gpu_id} ESM-2 ready")

    processed = 0
    cached_hits = 0

    while True:
        task = task_queue.get()

        if task is None:
            print(f"✓ GPU {gpu_id} finished: {processed} processed, {cached_hits} from cache")
            break

        rep_id, sequences = task

        # Check cache first
        if rep_id in cache_dict:
            result_queue.put(cache_dict[rep_id])
            cached_hits += 1
            processed += 1
            continue

        # Extract ESM-2 (same as before)
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

        result = {
            'repertoire_id': rep_id,
            'esm2_embedding': esm2_emb
        }

        result_queue.put(result)
        processed += 1


# Copy other functions from original (traditional, aa_props, model, etc.)
# ... (keeping them unchanged)

# Import from original file to avoid duplication
from ultimate_parallel_system import (
    extract_traditional_features_parallel,
    extract_aa_property_features_parallel,
    process_cpu_features_batch,
    HybridMILModel,
    RepertoireDataset
)


# ============================================================================
# Training with checkpointing
# ============================================================================

def train_fold_with_checkpointing(fold_id, train_data, val_data, epochs=30, resume=False):
    """Train with checkpoint support"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_id}: Training {len(train_data)} reps, validating {len(val_data)}")
    print(f"{'='*70}")

    # Check for existing checkpoint
    checkpoint_name = f"fold_{fold_id}_checkpoint"
    checkpoint = load_checkpoint(checkpoint_name) if resume else None

    start_epoch = 0
    best_auc = 0.0

    if checkpoint:
        print(f"✓ Resuming from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['best_auc']

    # Create datasets
    train_dataset = RepertoireDataset(train_data)
    val_dataset = RepertoireDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
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

    # Model
    model = HybridMILModel()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    criterion = nn.BCELoss()

    patience_counter = checkpoint['patience_counter'] if checkpoint else 0
    patience = 8

    for epoch in range(start_epoch, epochs):
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

        # Save checkpoint every epoch
        checkpoint_data = {
            'epoch': epoch,
            'fold': fold_id,
            'model_state': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_auc': best_auc,
            'val_auc': val_auc,
            'patience_counter': patience_counter,
            'timestamp': datetime.now().isoformat()
        }
        save_checkpoint(checkpoint_data, checkpoint_name)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
                'auc': best_auc,
                'fold': fold_id,
                'epoch': epoch
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
# Main with full caching
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏆 極致並行系統 V2（帶完整快取）🏆                        ║
    ║                                                              ║
    ║  ✓ 所有特徵自動保存                                         ║
    ║  ✓ 支援斷點續訓                                             ║
    ║  ✓ 避免重複計算                                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Check existing cache
    print("\n📊 Checking cache...")
    cache_mgr.print_stats()

    # Load datasets
    print("\n📊 Loading datasets...")
    datasets = load_all_datasets(
        './data/train_datasets/train_datasets',
        max_seqs_per_repertoire=1000,
        verbose=True
    )

    # Prepare tasks
    print("\n🔧 Preparing tasks...")
    all_repertoire_ids = []
    task_metadata = {}

    for ds in datasets:
        metadata = ds['metadata']
        schema = resolve_metadata_schema(metadata)

        for _, row in metadata.iterrows():
            rep_id = row[schema.repertoire_id]
            label = row[schema.label]

            seqs_df = ds['sequences'].get(rep_id)
            if seqs_df is None or len(seqs_df) == 0:
                continue

            all_repertoire_ids.append(rep_id)
            task_metadata[rep_id] = {
                'dataset_id': ds['dataset_id'],
                'label': label,
                'sequences_df': seqs_df
            }

    print(f"   Total repertoires: {len(all_repertoire_ids)}")

    # Check which features are already cached
    missing_reps = feature_cache.get_missing_repertoires(all_repertoire_ids)
    print(f"   Cached: {len(all_repertoire_ids) - len(missing_reps)}")
    print(f"   Need to compute: {len(missing_reps)}")

    all_features = {}

    if len(missing_reps) > 0:
        # Extract features for missing repertoires only
        print(f"\n⚡ Extracting features for {len(missing_reps)} repertoires...")

        # GPU extraction
        gpu_tasks = [(rid, task_metadata[rid]['sequences_df']['junction_aa'].dropna().tolist())
                    for rid in missing_reps]

        # Load existing cache for GPU workers
        cached_esm2 = {}
        for rid in all_repertoire_ids:
            if rid not in missing_reps:
                cached_features = feature_cache.load_repertoire_features(rid)
                if cached_features and 'esm2_embedding' in cached_features:
                    cached_esm2[rid] = {'repertoire_id': rid, 'esm2_embedding': cached_features['esm2_embedding']}

        mp.set_start_method('spawn', force=True)
        manager = Manager()
        gpu_task_queue = manager.Queue()
        gpu_result_queue = manager.Queue()
        cache_dict_shared = manager.dict(cached_esm2)

        for task in gpu_tasks:
            gpu_task_queue.put(task)

        for _ in range(NUM_GPUS):
            gpu_task_queue.put(None)

        gpu_processes = []
        for gpu_id in range(NUM_GPUS):
            p = Process(target=extract_features_gpu_worker_cached,
                       args=(gpu_id, gpu_task_queue, gpu_result_queue, cache_dict_shared))
            p.start()
            gpu_processes.append(p)

        gpu_results = {}
        with tqdm(total=len(gpu_tasks), desc="GPU: ESM-2") as pbar:
            for _ in range(len(gpu_tasks)):
                result = gpu_result_queue.get()
                gpu_results[result['repertoire_id']] = result['esm2_embedding']
                pbar.update(1)

        for p in gpu_processes:
            p.join()

        print(f"✓ GPU extraction complete")

        # CPU extraction
        print(f"\n⚡ CPU extraction for {len(missing_reps)} repertoires...")
        cpu_tasks = [{'repertoire_id': rid, 'sequences_df': task_metadata[rid]['sequences_df']}
                    for rid in missing_reps]

        batch_size = max(1, len(cpu_tasks) // NUM_CPUS)
        cpu_batches = [cpu_tasks[i:i+batch_size] for i in range(0, len(cpu_tasks), batch_size)]

        cpu_results = {}
        with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
            futures = [executor.submit(process_cpu_features_batch, batch) for batch in cpu_batches]

            with tqdm(total=len(futures), desc="CPU: Features") as pbar:
                for future in futures:
                    batch_results = future.result()
                    for res in batch_results:
                        cpu_results[res['repertoire_id']] = res
                    pbar.update(1)

        print(f"✓ CPU extraction complete")

        # Merge and save to cache
        print(f"\n💾 Saving {len(missing_reps)} features to cache...")
        for rep_id in missing_reps:
            if rep_id in gpu_results and rep_id in cpu_results:
                features = {
                    'esm2_embedding': gpu_results[rep_id],
                    'traditional_features': cpu_results[rep_id]['traditional_features'],
                    'aa_features': cpu_results[rep_id]['aa_features']
                }
                feature_cache.save_repertoire_features(rep_id, features)
                all_features[rep_id] = features

        print(f"✓ Saved to cache")

    # Load all features (from cache + newly computed)
    print(f"\n📦 Loading all features...")
    for rep_id in all_repertoire_ids:
        if rep_id not in all_features:
            cached_features = feature_cache.load_repertoire_features(rep_id)
            if cached_features:
                all_features[rep_id] = cached_features

    # Merge with metadata
    all_data = []
    for rep_id, features in all_features.items():
        all_data.append({
            'dataset_id': task_metadata[rep_id]['dataset_id'],
            'repertoire_id': rep_id,
            'label': 1.0 if task_metadata[rep_id]['label'] else 0.0,
            **features
        })

    print(f"✓ Total features ready: {len(all_data)}")

    # Save complete feature set
    features_backup = Path('./cache/all_features_complete.pkl')
    with open(features_backup, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"✓ Backup saved: {features_backup}")

    # LODO CV
    print("\n🎓 Leave-One-Dataset-Out CV")
    fold_aucs = []

    for test_ds_id in range(1, 9):
        train_data = [d for d in all_data if d['dataset_id'] != test_ds_id]
        val_data = [d for d in all_data if d['dataset_id'] == test_ds_id]

        if len(val_data) == 0:
            continue

        auc = train_fold_with_checkpointing(test_ds_id, train_data, val_data, epochs=30, resume=False)
        fold_aucs.append(auc)

    # Save final results
    results = {
        'fold_aucs': fold_aucs,
        'mean_auc': np.mean(fold_aucs),
        'std_auc': np.std(fold_aucs),
        'timestamp': datetime.now().isoformat()
    }

    with open('./models/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Fold AUCs: {fold_aucs}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"{'='*70}")
    print(f"\n✓ Results saved to ./models/final_results.json")


if __name__ == '__main__':
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)
    main()
