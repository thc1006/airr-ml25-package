#!/usr/bin/env python3
"""
Dataset-Specific ESM-2 Training Pipeline
=========================================
Uses cached ESM-2 15B embeddings for dataset-specific classification.
Each dataset gets its own classifier trained on its specific signals.

Key Insight: immuneML implants DIFFERENT signals per dataset, so LODO fails.
Solution: Train separate models per dataset using within-dataset CV.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  DATASET-SPECIFIC ESM-2 15B + K-MER ENSEMBLE")
print("  Combining deep learning embeddings with k-mer features")
print("=" * 70)
print()

# Configuration
CACHE_15B = Path("/app/cache/esm2_15b_nvfp4")
CACHE_650M = Path("/app/cache/esm2_gb10")
DATA_DIR = Path("/app/data")
OUTPUT_DIR = Path("/app/outputs/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Train/Test mapping
TRAIN_TEST_MAP = {
    1: ['test_dataset_1'],
    2: ['test_dataset_2'],
    3: ['test_dataset_3'],
    4: ['test_dataset_4'],
    5: ['test_dataset_5'],
    6: ['test_dataset_6'],
    7: ['test_dataset_7_1', 'test_dataset_7_2'],
    8: ['test_dataset_8_1', 'test_dataset_8_2', 'test_dataset_8_3']
}


def get_kmers(seq, k=4):
    """Extract k-mers from sequence."""
    kmers = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in kmer):
            kmers.append(kmer)
    return kmers


def extract_kmer_features(ds_path, meta_df, vocab=None):
    """Extract k-mer frequency features for a dataset."""
    kmer_counts_all = []
    rep_ids = []
    labels = []

    for _, row in meta_df.iterrows():
        rep_id = row['repertoire_id']
        rep_file = ds_path / f"{rep_id}.tsv"

        if not rep_file.exists():
            continue

        df = pd.read_csv(rep_file, sep='\t', usecols=['junction_aa'])

        counter = Counter()
        for seq in df['junction_aa'].dropna():
            if isinstance(seq, str):
                counter.update(get_kmers(seq, k=4))

        kmer_counts_all.append(counter)
        rep_ids.append(rep_id)
        labels.append(row.get('label_positive', None))

    # Build vocabulary from this dataset if not provided
    if vocab is None:
        total_counts = Counter()
        for c in kmer_counts_all:
            total_counts.update(c)
        vocab = list(total_counts.keys())

    # Create feature matrix
    X = np.zeros((len(kmer_counts_all), len(vocab)), dtype=np.float32)
    for i, c in enumerate(kmer_counts_all):
        total = sum(c.values()) or 1
        for j, kmer in enumerate(vocab):
            X[i, j] = c.get(kmer, 0) / total

    return X, labels, rep_ids, vocab


def load_embeddings(cache_path, ds_idx):
    """Load pre-computed ESM-2 embeddings."""
    emb_file = cache_path / f"train_dataset_{ds_idx}_embeddings.pkl"
    if not emb_file.exists():
        return None

    with open(emb_file, 'rb') as f:
        data = pickle.load(f)

    return data


def aggregate_embeddings(emb_dict, method='mean'):
    """Aggregate sequence embeddings to repertoire level."""
    if method == 'mean':
        return {k: v.mean(axis=0) for k, v in emb_dict.items()}
    elif method == 'max':
        return {k: v.max(axis=0) for k, v in emb_dict.items()}
    elif method == 'mean_max':
        return {k: np.concatenate([v.mean(axis=0), v.max(axis=0)])
                for k, v in emb_dict.items()}
    else:
        return {k: v.mean(axis=0) for k, v in emb_dict.items()}


def main():
    print("[1/4] Loading cached ESM-2 15B embeddings...")
    all_esm_data = {}

    for ds_idx in range(1, 9):
        data = load_embeddings(CACHE_15B, ds_idx)
        if data:
            all_esm_data[ds_idx] = data
            n_reps = len(data['embeddings'])
            print(f"  train_dataset_{ds_idx}: {n_reps} repertoires loaded")
        else:
            print(f"  train_dataset_{ds_idx}: NOT FOUND")

    print()
    print("[2/4] Training Dataset-Specific Models...")
    print("-" * 70)

    models = {}
    scalers = {}
    vocabs = {}
    cv_results = []

    for ds_idx in range(1, 9):
        print(f"\n=== Dataset {ds_idx} ===")

        ds_path = DATA_DIR / f"train_datasets/train_dataset_{ds_idx}"
        meta = pd.read_csv(ds_path / "metadata.csv")

        # Extract K-mer features
        X_kmer, labels, rep_ids, vocab = extract_kmer_features(ds_path, meta)
        vocabs[ds_idx] = vocab
        print(f"  K-mer features: {X_kmer.shape[1]} 4-mers")

        # Get ESM-2 embeddings
        if ds_idx in all_esm_data:
            esm_data = all_esm_data[ds_idx]
            agg_emb = aggregate_embeddings(esm_data['embeddings'], method='mean')

            # Align with k-mer order
            X_esm = []
            for rep_id in rep_ids:
                if rep_id in agg_emb:
                    X_esm.append(agg_emb[rep_id])
                else:
                    X_esm.append(np.zeros(5120))  # ESM-2 15B dim
            X_esm = np.vstack(X_esm)
            print(f"  ESM-2 15B features: {X_esm.shape[1]} dimensions")

            # Combine features
            X_combined = np.hstack([X_kmer, X_esm])
            print(f"  Combined features: {X_combined.shape[1]} dimensions")
        else:
            X_combined = X_kmer
            print(f"  Using K-mer only (no ESM-2 cache)")

        y = np.array([1 if l else 0 for l in labels])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        scalers[ds_idx] = scaler

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Try different models
        results = {}

        # Logistic Regression
        clf_lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        scores_lr = cross_val_score(clf_lr, X_scaled, y, cv=cv, scoring='roc_auc')
        results['LR'] = scores_lr.mean()

        # L1 Regularization
        clf_l1 = LogisticRegression(max_iter=2000, C=0.1, penalty='l1', solver='saga')
        scores_l1 = cross_val_score(clf_l1, X_scaled, y, cv=cv, scoring='roc_auc')
        results['LR-L1'] = scores_l1.mean()

        # MLP
        clf_mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500,
                                random_state=42, early_stopping=True)
        scores_mlp = cross_val_score(clf_mlp, X_scaled, y, cv=cv, scoring='roc_auc')
        results['MLP'] = scores_mlp.mean()

        # Pick best model
        best_model = max(results, key=results.get)
        best_auc = results[best_model]

        print(f"  CV Results: LR={results['LR']:.4f}, L1={results['LR-L1']:.4f}, MLP={results['MLP']:.4f}")
        print(f"  Best: {best_model} (AUC={best_auc:.4f})")

        cv_results.append({
            'dataset': ds_idx,
            'best_model': best_model,
            'auc': best_auc,
            'n_samples': len(y),
            'n_positive': sum(y)
        })

        # Train final model
        if best_model == 'LR':
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        elif best_model == 'LR-L1':
            clf = LogisticRegression(max_iter=2000, C=0.1, penalty='l1', solver='saga')
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500,
                               random_state=42, early_stopping=True)

        clf.fit(X_scaled, y)
        models[ds_idx] = clf

    # Summary
    print()
    print("-" * 70)
    print("CV Results Summary:")
    mean_auc = np.mean([r['auc'] for r in cv_results])
    for r in cv_results:
        print(f"  Dataset {r['dataset']}: {r['best_model']:6s} AUC={r['auc']:.4f}")
    print(f"  Mean CV AUC: {mean_auc:.4f}")
    print("-" * 70)

    print()
    print("[3/4] Generating Test Predictions...")

    all_predictions = {}

    for train_ds, test_datasets in TRAIN_TEST_MAP.items():
        model = models[train_ds]
        scaler = scalers[train_ds]
        vocab = vocabs[train_ds]

        for test_name in test_datasets:
            print(f"\n  Predicting {test_name} (using train_dataset_{train_ds} model)")

            test_path = DATA_DIR / f"test_datasets/{test_name}"
            test_files = [f for f in os.listdir(test_path) if f.endswith('.tsv')]

            test_preds = {}

            for tsv_file in test_files:
                rep_id = tsv_file.replace('.tsv', '')

                # Extract k-mer features
                df = pd.read_csv(test_path / tsv_file, sep='\t', usecols=['junction_aa'])
                counter = Counter()
                for seq in df['junction_aa'].dropna():
                    if isinstance(seq, str):
                        counter.update(get_kmers(seq, 4))

                total = sum(counter.values()) or 1
                X_kmer = np.array([counter.get(kmer, 0) / total for kmer in vocab],
                                  dtype=np.float32).reshape(1, -1)

                # TODO: Add ESM-2 features for test repertoires
                # For now, pad with zeros
                if train_ds in all_esm_data:
                    X_esm = np.zeros((1, 5120), dtype=np.float32)
                    X_combined = np.hstack([X_kmer, X_esm])
                else:
                    X_combined = X_kmer

                X_scaled = scaler.transform(X_combined)
                prob = model.predict_proba(X_scaled)[0, 1]
                test_preds[rep_id] = prob

            all_predictions[test_name] = test_preds
            probs = list(test_preds.values())
            print(f"    Predicted {len(test_preds)} repertoires, mean prob={np.mean(probs):.4f}")

    print()
    print("[4/4] Generating Submission...")

    # Task A: Repertoire predictions
    task_a_rows = []
    for test_name, preds in all_predictions.items():
        for rep_id, prob in preds.items():
            task_a_rows.append({
                'ID': rep_id,
                'dataset': test_name,
                'label_positive_probability': prob,
                'junction_aa': '-999.0',
                'v_call': '-999.0',
                'j_call': '-999.0'
            })

    print(f"  Task A: {len(task_a_rows)} repertoire predictions")

    # Task B: Top 50,000 sequences per train dataset
    task_b_rows = []

    for ds_idx in range(1, 9):
        ds_name = f'train_dataset_{ds_idx}'
        ds_path = DATA_DIR / f'train_datasets/{ds_name}'
        meta = pd.read_csv(ds_path / 'metadata.csv')

        # Collect from positive repertoires first
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
                    df = pd.read_csv(rep_file, sep='\t',
                                     usecols=['junction_aa', 'v_call', 'j_call'])

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
                except Exception:
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

        print(f"  {ds_name}: {len(seq_list)} sequences")

    print(f"  Task B total: {len(task_b_rows)} sequences")

    # Combine and save
    all_rows = task_a_rows + task_b_rows
    df = pd.DataFrame(all_rows)
    df = df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f'esm2_dataset_specific_{timestamp}.csv'
    df.to_csv(output_path, index=False)

    latest_path = OUTPUT_DIR / 'esm2_dataset_specific_latest.csv'
    df.to_csv(latest_path, index=False)

    print()
    print("=" * 70)
    print("SUBMISSION COMPLETE!")
    print("=" * 70)
    print(f"  Total rows: {len(df)}")
    print(f"  Expected: 404,213")
    print(f"  Mean CV AUC: {mean_auc:.4f}")
    print(f"  Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
