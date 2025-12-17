#!/usr/bin/env python3
"""
Champion V3 - Correct Feature Pipeline
=======================================
Builds features dynamically from actual data format (TRBV not TCRBV).
Uses k-mer features + V/J gene frequencies + diversity metrics.
GPU-accelerated XGBoost training.
"""

import os
import gc
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
TEST_ROOT = Path('./data/test_datasets/test_datasets')
SUBMISSION_DIR = Path('./submissions')
CHECKPOINT_DIR = Path('./checkpoints_v3')

TOP_K = 50000
KMER_K = 4

TEST_TO_TRAIN = {
    'test_dataset_1': 1,
    'test_dataset_2': 2,
    'test_dataset_3': 3,
    'test_dataset_4': 4,
    'test_dataset_5': 5,
    'test_dataset_6': 6,
    'test_dataset_7_1': 7,
    'test_dataset_7_2': 7,
    'test_dataset_8_1': 8,
    'test_dataset_8_2': 8,
    'test_dataset_8_3': 8,
}

# ============================================================================
# Feature Building - Step 1: Collect all unique features from training data
# ============================================================================
def collect_feature_vocabulary(train_datasets: List[int]) -> Dict[str, List[str]]:
    """Collect all unique feature names from training data."""
    print("\n" + "=" * 60)
    print("Phase 1: Collecting Feature Vocabulary")
    print("=" * 60)

    all_v_genes = set()
    all_j_genes = set()
    all_kmers = set()

    for ds_id in train_datasets:
        train_path = TRAIN_ROOT / f'train_dataset_{ds_id}'
        metadata = pd.read_csv(train_path / 'metadata.csv')

        print(f"\n  Dataset {ds_id}: {len(metadata)} repertoires")

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"    Scanning DS{ds_id}"):
            tsv_path = train_path / row['filename']
            try:
                df = pd.read_csv(tsv_path, sep='\t', usecols=['junction_aa', 'v_call', 'j_call'])

                # V genes
                if 'v_call' in df.columns:
                    v_genes = df['v_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).unique()
                    all_v_genes.update(v_genes)

                # J genes
                if 'j_call' in df.columns:
                    j_genes = df['j_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).unique()
                    all_j_genes.update(j_genes)

                # K-mers (sample to avoid memory issues)
                if 'junction_aa' in df.columns:
                    seqs = df['junction_aa'].dropna().astype(str)
                    for seq in seqs.head(1000):  # Sample first 1000 sequences
                        for i in range(len(seq) - KMER_K + 1):
                            kmer = seq[i:i+KMER_K]
                            if kmer.isalpha():  # Only valid amino acid k-mers
                                all_kmers.add(kmer)
            except Exception as e:
                continue

        gc.collect()

    # Create sorted feature lists
    v_features = sorted([f'v_{g}' for g in all_v_genes if g and g != 'nan'])
    j_features = sorted([f'j_{g}' for g in all_j_genes if g and g != 'nan'])
    kmer_features = sorted([f'kmer_{k}' for k in all_kmers])

    # Diversity metrics
    diversity_features = [
        'clonality', 'd50', 'shannon_entropy', 'gini_simpson',
        'mean_length', 'std_length', 'min_length', 'max_length',
        'top_clone_freq', 'unique_clones', 'total_sequences'
    ]

    all_features = diversity_features + v_features + j_features + kmer_features

    print(f"\n  Feature Vocabulary Summary:")
    print(f"    V genes: {len(v_features)}")
    print(f"    J genes: {len(j_features)}")
    print(f"    K-mers: {len(kmer_features)}")
    print(f"    Diversity: {len(diversity_features)}")
    print(f"    TOTAL: {len(all_features)}")

    return {
        'all_features': all_features,
        'v_features': v_features,
        'j_features': j_features,
        'kmer_features': kmer_features,
        'diversity_features': diversity_features
    }


# ============================================================================
# Feature Extraction
# ============================================================================
def extract_features(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """Extract features from a repertoire DataFrame."""
    features = {}
    total = len(df)

    if total == 0:
        return np.zeros(len(feature_names), dtype=np.float32)

    # === V gene usage ===
    if 'v_call' in df.columns:
        v_counts = df['v_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).value_counts()
        for gene, count in v_counts.items():
            if gene and gene != 'nan':
                features[f'v_{gene}'] = count / total

    # === J gene usage ===
    if 'j_call' in df.columns:
        j_counts = df['j_call'].dropna().astype(str).apply(lambda x: x.split('*')[0]).value_counts()
        for gene, count in j_counts.items():
            if gene and gene != 'nan':
                features[f'j_{gene}'] = count / total

    # === K-mer features ===
    if 'junction_aa' in df.columns:
        seqs = df['junction_aa'].dropna().astype(str)
        kmer_counts = Counter()
        total_kmers = 0

        for seq in seqs:
            for i in range(len(seq) - KMER_K + 1):
                kmer = seq[i:i+KMER_K]
                if kmer.isalpha():
                    kmer_counts[kmer] += 1
                    total_kmers += 1

        if total_kmers > 0:
            for kmer, count in kmer_counts.items():
                features[f'kmer_{kmer}'] = count / total_kmers

    # === Diversity metrics ===
    if 'junction_aa' in df.columns:
        seqs = df['junction_aa'].dropna()
        if len(seqs) > 0:
            seq_counts = seqs.value_counts()
            freqs = seq_counts.values / seq_counts.sum()

            # Shannon entropy
            from scipy.stats import entropy
            ent = entropy(freqs)
            features['shannon_entropy'] = ent

            # Gini-Simpson
            features['gini_simpson'] = 1 - np.sum(freqs ** 2)

            # D50
            cumsum = np.cumsum(np.sort(freqs)[::-1])
            features['d50'] = (np.searchsorted(cumsum, 0.5) + 1) / len(freqs)

            # Clonality (normalized entropy)
            max_ent = np.log(len(seq_counts)) if len(seq_counts) > 1 else 1
            features['clonality'] = 1 - (ent / max_ent) if max_ent > 0 else 0

            # Length statistics
            lengths = seqs.str.len()
            features['mean_length'] = lengths.mean()
            features['std_length'] = lengths.std() if len(lengths) > 1 else 0
            features['min_length'] = lengths.min()
            features['max_length'] = lengths.max()

            # Clone statistics
            features['top_clone_freq'] = freqs[0] if len(freqs) > 0 else 0
            features['unique_clones'] = len(seq_counts)
            features['total_sequences'] = len(seqs)

    # === Build feature vector ===
    result = np.zeros(len(feature_names), dtype=np.float32)
    for i, name in enumerate(feature_names):
        if name in features:
            v = features[name]
            if pd.notna(v) and not np.isinf(v):
                result[i] = float(v)

    return result


def extract_dataset(dataset_path: Path, feature_names: List[str],
                   metadata: pd.DataFrame = None, is_test: bool = False) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract features for all repertoires in a dataset."""
    X_list = []
    ids = []
    labels = []

    if is_test:
        # Test dataset: use filenames as IDs
        tsv_files = sorted(dataset_path.glob('*.tsv'))
        for tsv_file in tqdm(tsv_files, desc=f"  {dataset_path.name}", leave=False):
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                feat = extract_features(df, feature_names)
                X_list.append(feat)
                ids.append(tsv_file.stem)
            except Exception as e:
                X_list.append(np.zeros(len(feature_names), dtype=np.float32))
                ids.append(tsv_file.stem)
    else:
        # Training dataset: use metadata
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"  {dataset_path.name}", leave=False):
            tsv_file = dataset_path / row['filename']
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                feat = extract_features(df, feature_names)
                X_list.append(feat)
                ids.append(row['repertoire_id'])
                labels.append(int(row['label_positive']))
            except Exception as e:
                X_list.append(np.zeros(len(feature_names), dtype=np.float32))
                ids.append(row['repertoire_id'])
                labels.append(int(row['label_positive']))

    X = np.vstack(X_list)
    return X, ids, labels


# ============================================================================
# Training
# ============================================================================
def train_models(feature_names: List[str]) -> Dict:
    """Train XGBoost models for all 8 datasets."""
    print("\n" + "=" * 60)
    print("Phase 2: Training XGBoost Models (GPU)")
    print("=" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    models = {}

    for ds_id in range(1, 9):
        print(f"\n--- Dataset {ds_id} ---")
        train_path = TRAIN_ROOT / f'train_dataset_{ds_id}'
        metadata = pd.read_csv(train_path / 'metadata.csv')

        # Extract features
        X, ids, y = extract_dataset(train_path, feature_names, metadata, is_test=False)
        y = np.array(y)

        # Check feature variance
        non_zero = np.sum(np.abs(X).sum(axis=0) > 0)
        print(f"  Samples: {len(y)}, Pos: {y.sum()}, Non-zero features: {non_zero}/{len(feature_names)}")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train XGBoost with GPU
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            device='cuda',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )

        model.fit(X_scaled, y, verbose=False)

        # Validation
        train_pred = model.predict_proba(X_scaled)[:, 1]
        print(f"  Train pred range: [{train_pred.min():.4f}, {train_pred.max():.4f}]")

        # Save checkpoint
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'ds_id': ds_id
        }

        ckpt_path = CHECKPOINT_DIR / f'xgb_ds{ds_id}.pkl'
        with open(ckpt_path, 'wb') as f:
            pickle.dump(model_data, f)

        models[ds_id] = model_data
        print(f"  Saved: {ckpt_path}")

        gc.collect()

    return models


# ============================================================================
# Prediction
# ============================================================================
def predict_test(models: Dict, feature_names: List[str]) -> pd.DataFrame:
    """Generate predictions for all test datasets."""
    print("\n" + "=" * 60)
    print("Phase 3: Task A - Test Predictions")
    print("=" * 60)

    all_predictions = []

    for test_name, train_id in TEST_TO_TRAIN.items():
        test_path = TEST_ROOT / test_name
        if not test_path.exists():
            print(f"  Skipping {test_name}: not found")
            continue

        if train_id not in models:
            print(f"  Skipping {test_name}: no model for train_dataset_{train_id}")
            continue

        model_data = models[train_id]
        model = model_data['model']
        scaler = model_data['scaler']

        # Extract features
        X_test, test_ids, _ = extract_dataset(test_path, feature_names, is_test=True)

        # Predict
        X_scaled = scaler.transform(X_test)
        probs = model.predict_proba(X_scaled)[:, 1]

        # Create predictions
        for rid, prob in zip(test_ids, probs):
            all_predictions.append({
                'ID': rid,
                'dataset': test_name,
                'label_positive_probability': float(prob),
                'junction_aa': -999.0,
                'v_call': -999.0,
                'j_call': -999.0
            })

        print(f"  {test_name}: {len(test_ids)} samples, range=[{probs.min():.4f}, {probs.max():.4f}]")

    return pd.DataFrame(all_predictions)


# ============================================================================
# Sequence Identification (Task B)
# ============================================================================
def identify_sequences() -> pd.DataFrame:
    """Identify top-k sequences from each training dataset."""
    print("\n" + "=" * 60)
    print("Phase 4: Task B - Sequence Identification")
    print("=" * 60)

    all_sequences = []

    for ds_id in range(1, 9):
        ds_name = f'train_dataset_{ds_id}'
        train_path = TRAIN_ROOT / ds_name

        if not train_path.exists():
            print(f"  Skipping {ds_name}: not found")
            continue

        print(f"\n  {ds_name}...")
        metadata = pd.read_csv(train_path / 'metadata.csv')

        # Score sequences based on label association
        seq_scores = defaultdict(lambda: {'score': 0.0, 'v': None, 'j': None})

        pos_count = metadata['label_positive'].sum()
        neg_count = len(metadata) - pos_count

        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="    Scanning"):
            label = row['label_positive']
            weight = 1.0 / pos_count if label == 1 else -0.5 / neg_count

            try:
                df = pd.read_csv(train_path / row['filename'], sep='\t')

                for _, seq_row in df.iterrows():
                    junc = seq_row.get('junction_aa')
                    if pd.isna(junc):
                        continue
                    junc = str(junc)

                    if len(junc) < 8:  # Skip very short sequences
                        continue

                    seq_scores[junc]['score'] += weight

                    # Store V/J genes
                    v = seq_row.get('v_call')
                    j = seq_row.get('j_call')
                    if pd.notna(v) and not seq_scores[junc]['v']:
                        seq_scores[junc]['v'] = str(v).split('*')[0]
                    if pd.notna(j) and not seq_scores[junc]['j']:
                        seq_scores[junc]['j'] = str(j).split('*')[0]
            except:
                continue

        # Sort by score and get top-k
        sorted_seqs = sorted(seq_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:TOP_K]

        for rank, (junc, info) in enumerate(sorted_seqs, 1):
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{rank}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': junc,
                'v_call': info['v'] if info['v'] else 'TRBV20-1',
                'j_call': info['j'] if info['j'] else 'TRBJ2-7'
            })

        # Pad if needed
        current_count = len([s for s in all_sequences if s['dataset'] == ds_name])
        while current_count < TOP_K:
            current_count += 1
            all_sequences.append({
                'ID': f'{ds_name}_seq_top_{current_count}',
                'dataset': ds_name,
                'label_positive_probability': -999.0,
                'junction_aa': f'CASS{"X"*8}',
                'v_call': 'TRBV20-1',
                'j_call': 'TRBJ2-7'
            })

        print(f"    {len(sorted_seqs)} unique sequences identified")
        gc.collect()

    return pd.DataFrame(all_sequences)


# ============================================================================
# Main
# ============================================================================
def main():
    start = datetime.now()
    print(f"""
================================================================================
  AIRR-ML-25 Champion V3 - Correct Feature Pipeline
  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}

  FIX: Using actual gene format (TRBV) instead of wrong format (TCRBV)
  Features: K-mers + V/J genes + Diversity metrics (all dynamically built)
================================================================================
""")

    # Phase 1: Collect feature vocabulary from training data
    vocab = collect_feature_vocabulary(list(range(1, 9)))
    feature_names = vocab['all_features']

    # Save feature names
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(CHECKPOINT_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print(f"\n  Saved feature names to {CHECKPOINT_DIR / 'feature_names.json'}")

    # Phase 2: Train models
    models = train_models(feature_names)

    # Phase 3: Task A predictions
    task_a = predict_test(models, feature_names)

    # Phase 4: Task B sequences
    task_b = identify_sequences()

    # Create submission
    print("\n" + "=" * 60)
    print("Phase 5: Creating Submission")
    print("=" * 60)

    submission = pd.concat([task_a, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    expected = 4213 + 8 * TOP_K
    print(f"  Rows: {len(submission)} (expected: {expected})")

    # Save
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = SUBMISSION_DIR / f'champion_v3_{ts}.csv'
    submission.to_csv(path, index=False)

    print(f"  Saved: {path}")
    print(f"  Size: {os.path.getsize(path) / 1e6:.2f} MB")

    # Stats
    probs = task_a['label_positive_probability']
    print(f"\n  Probability stats:")
    print(f"    Min: {probs.min():.4f}")
    print(f"    Max: {probs.max():.4f}")
    print(f"    Mean: {probs.mean():.4f}")
    print(f"    Std: {probs.std():.4f}")

    # Per-dataset stats
    print(f"\n  Per-dataset prediction ranges:")
    for ds in TEST_TO_TRAIN.keys():
        ds_probs = task_a[task_a['dataset'] == ds]['label_positive_probability']
        if len(ds_probs) > 0:
            print(f"    {ds}: std={ds_probs.std():.4f}, range=[{ds_probs.min():.3f}, {ds_probs.max():.3f}]")

    duration = datetime.now() - start
    print(f"""
================================================================================
  COMPLETE!
  Duration: {duration}
  Submission: {path}
================================================================================
""")

    return str(path)


if __name__ == "__main__":
    main()
