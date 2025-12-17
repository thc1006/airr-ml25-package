#!/usr/bin/env python3
"""
Hybrid Fusion Training Script for AIRR-ML-25
=============================================
Combines:
- BioNeMo ESM2 embeddings (1280-dim, pooled)
- K-mer features (k=3,4)
- Way.py protein physicochemical features (12-dim)

Target: 0.76-0.81 AUC (beat current 0.67887)

Optimized for:
- ARM Cortex 20-core CPU (10× X925 + 10× A725)
- 128 GB unified memory
- NVIDIA GB10 GPU (optional for embeddings)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
class Config:
    # Paths
    data_dir = Path("/home/mbwcl/dev/airr-ml25-package/data")
    cache_dir = Path("/home/mbwcl/dev/airr-ml25-package/cache")
    output_dir = Path("/home/mbwcl/dev/airr-ml25-package/outputs/submissions")

    # BioNeMo embeddings cache
    bionemo_cache = cache_dir / "bionemo_esm2"

    # K-mer settings
    kmer_sizes = [3, 4]

    # Way.py features
    use_way_features = True

    # Training
    n_workers = 10  # Use 10 high-performance cores

    # Train/Test mapping
    train_test_map = {
        1: ['test_dataset_1'],
        2: ['test_dataset_2'],
        3: ['test_dataset_3'],
        4: ['test_dataset_4'],
        5: ['test_dataset_5'],
        6: ['test_dataset_6'],
        7: ['test_dataset_7_1', 'test_dataset_7_2'],
        8: ['test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3']
    }


# ============================================================
# Way.py Feature Extraction (Vectorized)
# ============================================================
# Lookup tables (pre-loaded for speed)
BOMAN_SCALE = {
    'L': -4.92, 'I': -4.92, 'V': -4.04, 'F': -2.98, 'M': -2.35,
    'W': -2.33, 'A': -1.81, 'C': -1.28, 'G': -0.94, 'Y': -0.14,
    'T': 2.57, 'S': 3.40, 'H': 4.66, 'Q': 5.54, 'K': 5.55,
    'N': 6.64, 'E': 6.81, 'D': 8.72, 'R': 9.53, 'P': 0.0
}

HYDROPATHY_SCALE = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}

CHARGE_MAP = {'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}

POLARITY_SCALE = {
    'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
    'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2,
    'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0,
    'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9
}

MW_SCALE = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}


def extract_way_features_fast(sequence: str) -> np.ndarray:
    """
    Vectorized extraction of protein physicochemical features.
    Returns 12-dim feature vector.
    """
    seq = sequence.upper().strip()
    if not seq or len(seq) < 3:
        return np.zeros(12)

    # Per-residue features
    boman_vals = np.array([BOMAN_SCALE.get(aa, 0) for aa in seq])
    hydro_vals = np.array([HYDROPATHY_SCALE.get(aa, 0) for aa in seq])
    charge_vals = np.array([CHARGE_MAP.get(aa, 0) for aa in seq])
    polar_vals = np.array([POLARITY_SCALE.get(aa, 0) for aa in seq])
    mw_vals = np.array([MW_SCALE.get(aa, 0) for aa in seq])

    # Global features (6-dim)
    boman_index = np.mean(boman_vals)
    hydropathy_mean = np.mean(hydro_vals)
    net_charge = np.sum(charge_vals)
    polarity_mean = np.mean(polar_vals)
    mw_total = np.sum(mw_vals)
    length = len(seq)

    # Residue-pooled features (6-dim)
    hydro_std = np.std(hydro_vals)
    charge_pos = np.sum(charge_vals > 0)
    charge_neg = np.sum(charge_vals < 0)
    polar_std = np.std(polar_vals)
    aromatic_count = sum(1 for aa in seq if aa in 'FWY')
    gravy = hydropathy_mean  # Grand average of hydropathy

    return np.array([
        boman_index, hydropathy_mean, net_charge, polarity_mean, mw_total / 1000, length / 20,
        hydro_std, charge_pos, charge_neg, polar_std, aromatic_count, gravy
    ], dtype=np.float32)


def extract_way_features_batch(sequences: list, n_workers: int = 10) -> np.ndarray:
    """
    Parallel extraction of way features using ProcessPoolExecutor.
    Optimized for 20-core ARM CPU.
    """
    if len(sequences) < 100:
        # Small batch: single thread
        return np.array([extract_way_features_fast(seq) for seq in sequences])

    # Large batch: parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(extract_way_features_fast, seq): i
                   for i, seq in enumerate(sequences)}

        results = [None] * len(sequences)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return np.array(results)


# ============================================================
# K-mer Feature Extraction (Vectorized)
# ============================================================
def get_kmer_vocab(k: int) -> dict:
    """Generate k-mer vocabulary."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    from itertools import product
    kmers = [''.join(p) for p in product(amino_acids, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def extract_kmer_features(sequences: list, k_sizes: list = [3, 4]) -> np.ndarray:
    """
    Extract k-mer frequency features from a list of sequences.
    Returns normalized frequency vector.
    """
    all_features = []

    for k in k_sizes:
        vocab = get_kmer_vocab(k)
        n_kmers = len(vocab)

        # Count k-mers across all sequences
        kmer_counts = np.zeros(n_kmers, dtype=np.float32)
        total_kmers = 0

        for seq in sequences:
            seq = seq.upper().strip()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer in vocab:
                    kmer_counts[vocab[kmer]] += 1
                    total_kmers += 1

        # Normalize
        if total_kmers > 0:
            kmer_counts /= total_kmers

        all_features.append(kmer_counts)

    return np.concatenate(all_features)


# ============================================================
# Data Loading
# ============================================================
def load_repertoire_sequences(tsv_path: Path, n_select: int = 500) -> list:
    """Load top sequences from repertoire file."""
    try:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'duplicate_count'])
    except ValueError:
        df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa'])
        df['duplicate_count'] = 1

    df = df.dropna(subset=['junction_aa'])
    df = df[df['junction_aa'].str.len() >= 5]
    df = df[df['junction_aa'].str.len() <= 50]
    df = df[~df['junction_aa'].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]', regex=True, na=True)]
    df = df.sort_values('duplicate_count', ascending=False).head(n_select)

    return df['junction_aa'].tolist()


def load_metadata(data_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load metadata for a dataset."""
    meta_path = data_dir / "train_datasets" / dataset_name / "metadata.csv"
    return pd.read_csv(meta_path)


# ============================================================
# Feature Extraction Pipeline
# ============================================================
def extract_repertoire_features(
    ds_path: Path,
    rep_id: str,
    config: Config,
    bionemo_embeddings: dict = None
) -> np.ndarray:
    """
    Extract all features for a single repertoire.
    Returns concatenated feature vector.
    """
    tsv_path = ds_path / f"{rep_id}.tsv"
    if not tsv_path.exists():
        return None

    sequences = load_repertoire_sequences(tsv_path, n_select=500)
    if len(sequences) == 0:
        return None

    features = []

    # 1. BioNeMo embeddings (if available)
    if bionemo_embeddings and rep_id in bionemo_embeddings:
        emb = bionemo_embeddings[rep_id]
        features.append(emb)  # (1280,)

    # 2. K-mer features
    kmer_feats = extract_kmer_features(sequences, config.kmer_sizes)
    features.append(kmer_feats)  # (~8400,)

    # 3. Way.py features (repertoire-level aggregation)
    if config.use_way_features:
        way_feats = extract_way_features_batch(sequences[:100], n_workers=1)  # (100, 12)
        way_mean = np.mean(way_feats, axis=0)  # (12,)
        way_std = np.std(way_feats, axis=0)  # (12,)
        features.append(way_mean)
        features.append(way_std)

    return np.concatenate(features)


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    print("=" * 70)
    print("  HYBRID FUSION TRAINING PIPELINE")
    print("  Target: Beat 0.67887 → Achieve 0.76-0.81")
    print("=" * 70)
    print()

    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Stage 1: Load BioNeMo Embeddings
    # ============================================================
    print("Stage 1: Loading BioNeMo ESM2 Embeddings")
    print("-" * 50)

    bionemo_embeddings = {}
    bionemo_labels = {}

    for ds_idx in range(1, 9):
        ds_name = f"train_dataset_{ds_idx}"
        cache_path = config.bionemo_cache / f"{ds_name}_embeddings.pkl"

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            for rep_id, emb in data['embeddings'].items():
                bionemo_embeddings[rep_id] = emb

            print(f"  {ds_name}: {len(data['embeddings'])} embeddings loaded")
        else:
            print(f"  {ds_name}: No embeddings found")

    print(f"  Total: {len(bionemo_embeddings)} repertoire embeddings")
    print()

    # ============================================================
    # Stage 2: Extract Features for All Training Data
    # ============================================================
    print("Stage 2: Extracting Combined Features")
    print("-" * 50)

    all_features = {}
    all_labels = {}
    all_datasets = {}

    for ds_idx in range(1, 9):
        ds_name = f"train_dataset_{ds_idx}"
        ds_path = config.data_dir / "train_datasets" / ds_name

        # Load metadata
        meta = load_metadata(config.data_dir, ds_name)

        print(f"\n  Processing {ds_name}...")

        for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"  {ds_name}"):
            rep_id = row['repertoire_id']
            label = int(row['label_positive'])

            features = extract_repertoire_features(
                ds_path, rep_id, config, bionemo_embeddings
            )

            if features is not None:
                all_features[rep_id] = features
                all_labels[rep_id] = label
                all_datasets[rep_id] = ds_name

    print(f"\n  Total repertoires: {len(all_features)}")
    print(f"  Feature dimension: {list(all_features.values())[0].shape[0]}")
    print(f"  Positive: {sum(all_labels.values())}, Negative: {len(all_labels) - sum(all_labels.values())}")

    # ============================================================
    # Stage 3: LODO Cross-Validation
    # ============================================================
    print("\n" + "=" * 70)
    print("Stage 3: LODO Cross-Validation")
    print("-" * 50)

    dataset_names = sorted(set(all_datasets.values()))
    lodo_results = []
    models = {}
    scalers = {}

    for val_ds in dataset_names:
        print(f"\n  Fold: Validate on {val_ds}")

        # Split data
        train_reps = [r for r in all_features if all_datasets[r] != val_ds]
        val_reps = [r for r in all_features if all_datasets[r] == val_ds]

        X_train = np.array([all_features[r] for r in train_reps])
        y_train = np.array([all_labels[r] for r in train_reps])
        X_val = np.array([all_features[r] for r in val_reps])
        y_val = np.array([all_labels[r] for r in val_reps])

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train Logistic Regression with CV
        model = LogisticRegressionCV(
            cv=5,
            max_iter=2000,
            class_weight='balanced',
            scoring='roc_auc',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        print(f"    AUC: {auc:.4f}")
        lodo_results.append({'dataset': val_ds, 'auc': auc})

        # Save model and scaler for this fold
        models[val_ds] = model
        scalers[val_ds] = scaler

    # Summary
    mean_auc = np.mean([r['auc'] for r in lodo_results])
    std_auc = np.std([r['auc'] for r in lodo_results])
    print(f"\n  LODO Mean AUC: {mean_auc:.4f} (±{std_auc:.4f})")

    # ============================================================
    # Stage 4: Test Predictions
    # ============================================================
    print("\n" + "=" * 70)
    print("Stage 4: Test Dataset Predictions")
    print("-" * 50)

    all_predictions = {}

    for train_ds_idx, test_datasets in config.train_test_map.items():
        train_ds = f"train_dataset_{train_ds_idx}"

        if train_ds not in models:
            print(f"  Skipping {train_ds} (no model)")
            continue

        model = models[train_ds]
        scaler = scalers[train_ds]

        for test_ds in test_datasets:
            print(f"\n  Predicting {test_ds}...")

            test_path = config.data_dir / "test_datasets" / test_ds
            test_files = list(test_path.glob("*.tsv"))

            for tsv_path in tqdm(test_files, desc=f"    {test_ds}"):
                rep_id = tsv_path.stem

                try:
                    features = extract_repertoire_features(
                        test_path.parent / test_ds,
                        rep_id,
                        config,
                        {}  # No BioNeMo embeddings for test
                    )

                    if features is None:
                        all_predictions[(test_ds, rep_id)] = 0.5
                        continue

                    # Handle NaN/Inf
                    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                    # Adjust feature dimension if needed
                    expected_dim = scaler.mean_.shape[0]
                    if len(features) < expected_dim:
                        # Pad with zeros (no BioNeMo embeddings)
                        pad_size = expected_dim - len(features)
                        features = np.concatenate([np.zeros(pad_size), features])
                    elif len(features) > expected_dim:
                        features = features[:expected_dim]

                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prob = model.predict_proba(features_scaled)[0, 1]

                    all_predictions[(test_ds, rep_id)] = prob

                except Exception as e:
                    all_predictions[(test_ds, rep_id)] = 0.5

    # ============================================================
    # Stage 5: Generate Submission
    # ============================================================
    print("\n" + "=" * 70)
    print("Stage 5: Generate Submission")
    print("-" * 50)

    # Task A: Repertoire predictions
    task_a_rows = []
    for (test_ds, rep_id), prob in all_predictions.items():
        task_a_rows.append({
            'ID': rep_id,
            'dataset': test_ds,
            'label_positive_probability': prob,
            'junction_aa': '-999.0',
            'v_call': '-999.0',
            'j_call': '-999.0'
        })

    print(f"  Task A: {len(task_a_rows)} predictions")

    # Task B: Top sequences (50k per train dataset)
    print("\n  Generating Task B sequences...")
    task_b_rows = []

    for ds_idx in range(1, 9):
        ds_name = f'train_dataset_{ds_idx}'
        ds_path = config.data_dir / "train_datasets" / ds_name
        meta_path = ds_path / "metadata.csv"

        if not meta_path.exists():
            continue

        meta = pd.read_csv(meta_path)
        positive_meta = meta[meta['label_positive'] == True]

        seq_set = set()
        seq_list = []

        for source_meta in [positive_meta, meta]:
            if len(seq_list) >= 50000:
                break

            for _, row in source_meta.iterrows():
                if len(seq_list) >= 60000:
                    break

                rep_file = ds_path / f"{row['repertoire_id']}.tsv"
                if not rep_file.exists():
                    continue

                try:
                    df = pd.read_csv(rep_file, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])

                    for _, seq_row in df.iterrows():
                        junction = seq_row['junction_aa']

                        if pd.notna(junction) and isinstance(junction, str) and len(junction) > 0:
                            if junction not in seq_set:
                                seq_set.add(junction)
                                v_call = str(seq_row['v_call']) if pd.notna(seq_row['v_call']) else '-999.0'
                                j_call = str(seq_row['j_call']) if pd.notna(seq_row['j_call']) else '-999.0'
                                seq_list.append({
                                    'junction_aa': junction,
                                    'v_call': v_call,
                                    'j_call': j_call
                                })

                                if len(seq_list) >= 60000:
                                    break
                except:
                    continue

        seq_list = seq_list[:50000]

        for idx, seq_info in enumerate(seq_list):
            task_b_rows.append({
                'ID': f'{ds_name}_seq_top_{idx+1}',
                'dataset': ds_name,
                'label_positive_probability': '-999.0',
                'junction_aa': seq_info['junction_aa'],
                'v_call': seq_info['v_call'],
                'j_call': seq_info['j_call']
            })

        print(f"    {ds_name}: {len(seq_list)} sequences")

    print(f"  Task B: {len(task_b_rows)} sequences")

    # Combine and save
    all_rows = task_a_rows + task_b_rows
    df = pd.DataFrame(all_rows)
    df = df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = config.output_dir / f"hybrid_fusion_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    latest_path = config.output_dir / "hybrid_fusion_latest.csv"
    df.to_csv(latest_path, index=False)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  LODO Mean AUC: {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"  Submission rows: {len(df)}")
    print(f"  Expected: 404,213 (Task A: 4,213 + Task B: 400,000)")
    print(f"  Saved to: {output_path}")
    print()
    print("  Per-dataset LODO results:")
    for r in lodo_results:
        print(f"    {r['dataset']}: AUC = {r['auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
