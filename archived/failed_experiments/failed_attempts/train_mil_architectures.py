#!/usr/bin/env python3
"""
Train and compare MIL architectures for AIRR-ML-25.

Usage:
    # In NGC container
    python scripts/train_mil_architectures.py --model eamil --epochs 50
    python scripts/train_mil_architectures.py --model transformer --epochs 50
    python scripts/train_mil_architectures.py --model deepset --epochs 50

Hardware: NVIDIA GB10 Grace Blackwell (optimized)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from airr_ml25.models.mil_architectures import (
    EAMIL,
    DeepSetMIL,
    TransformerMIL,
    enable_gb10_optimizations,
    get_model_memory_footprint,
)


def load_embeddings(cache_dir: Path) -> Tuple[Dict, Dict]:
    """
    Load cached ESM-2 embeddings from per-dataset pickle files.

    Expected format:
        cache_dir/train_dataset_1_embeddings.pkl: {rep_id: array(n_seqs, embed_dim)}
        cache_dir/test_dataset_1_embeddings.pkl: {rep_id: array(n_seqs, embed_dim)}
    """
    print("Loading ESM-2 15B embeddings...")

    train_data = {}  # {rep_id: {'embedding': array, 'label': int, 'dataset': str}}
    test_data = {}   # {dataset_name: {rep_id: array}}

    # Load train datasets
    for train_file in sorted(cache_dir.glob('train_dataset_*_embeddings.pkl')):
        dataset_name = train_file.stem.replace('_embeddings', '')
        print(f"  Loading {dataset_name}...")

        with open(train_file, 'rb') as f:
            embeddings_dict = pickle.load(f)

        # Read labels from original data
        data_dir = Path('/app/data/train_datasets') / dataset_name
        labels_df = pd.read_csv(data_dir / 'metadata.csv')
        label_dict = dict(zip(labels_df['repertoire_id'], labels_df['label']))

        # Combine embeddings + labels
        for rep_id, embedding in embeddings_dict.items():
            train_data[rep_id] = {
                'embedding': embedding,
                'label': label_dict[rep_id],
                'dataset': dataset_name
            }

    # Load test datasets
    for test_file in sorted(cache_dir.glob('test_dataset_*_embeddings.pkl')):
        dataset_name = test_file.stem.replace('_embeddings', '')
        print(f"  Loading {dataset_name}...")

        with open(test_file, 'rb') as f:
            test_data[dataset_name] = pickle.load(f)

    print(f"\n  Train repertoires: {len(train_data)}")
    print(f"  Test datasets: {len(test_data)}")

    return train_data, test_data


def prepare_batch_data(
    repertoire_ids: List[str],
    embeddings_dict: Dict,
    labels_dict: Dict,
    max_instances: int = 200
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare batch of repertoire embeddings with padding.

    Args:
        repertoire_ids: List of repertoire IDs
        embeddings_dict: {rep_id: (n_seqs, embed_dim)}
        labels_dict: {rep_id: label}
        max_instances: Maximum sequences per repertoire

    Returns:
        x: (batch, max_instances, embed_dim) - Padded embeddings
        y: (batch,) - Labels
        mask: (batch, max_instances) - True = padding
    """
    batch_size = len(repertoire_ids)
    embed_dim = next(iter(embeddings_dict.values())).shape[1]

    x = torch.zeros(batch_size, max_instances, embed_dim)
    y = torch.zeros(batch_size)
    mask = torch.ones(batch_size, max_instances, dtype=torch.bool)  # All padding initially

    for i, rep_id in enumerate(repertoire_ids):
        emb = embeddings_dict[rep_id]  # (n_seqs, embed_dim)
        n_seqs = min(len(emb), max_instances)

        x[i, :n_seqs] = torch.from_numpy(emb[:n_seqs])
        y[i] = labels_dict[rep_id]
        mask[i, :n_seqs] = False  # Mark as valid

    return x, y, mask


def train_lodo_cv(
    model_class,
    model_kwargs: dict,
    train_data: dict,
    device: str = 'cuda',
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> Dict:
    """
    Train with Leave-One-Dataset-Out cross-validation.

    Args:
        model_class: MIL model class (EAMIL, TransformerMIL, DeepSetMIL)
        model_kwargs: Model initialization arguments
        train_data: {rep_id: {'embedding': array, 'label': int, 'dataset': str}}
        device: 'cuda' or 'cpu'
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        weight_decay: L2 regularization

    Returns:
        results: {fold: {'auc': float, 'model_state': dict}}
    """
    # Organize data by dataset
    datasets = {}
    for rep_id, data in train_data.items():
        dataset_name = data['dataset']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(rep_id)

    dataset_names = sorted(datasets.keys())
    print(f"\nLODO CV with {len(dataset_names)} folds")
    print(f"Datasets: {dataset_names}")

    results = {}

    # LODO CV
    for fold_idx, val_dataset in enumerate(dataset_names, 1):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx}/{len(dataset_names)}: Validation on {val_dataset}")
        print(f"{'='*70}")

        # Split train/val
        val_rep_ids = datasets[val_dataset]
        train_rep_ids = [
            rep_id for ds, reps in datasets.items()
            if ds != val_dataset for rep_id in reps
        ]

        print(f"  Train repertoires: {len(train_rep_ids)}")
        print(f"  Val repertoires: {len(val_rep_ids)}")

        # Create embeddings dicts
        train_embeddings = {rep_id: train_data[rep_id]['embedding'] for rep_id in train_rep_ids}
        train_labels = {rep_id: train_data[rep_id]['label'] for rep_id in train_rep_ids}

        val_embeddings = {rep_id: train_data[rep_id]['embedding'] for rep_id in val_rep_ids}
        val_labels = {rep_id: train_data[rep_id]['label'] for rep_id in val_rep_ids}

        # Initialize model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_val_auc = 0.0
        best_model_state = None

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_losses = []

            # Shuffle train repertoires
            np.random.shuffle(train_rep_ids)

            # Batch training
            for i in range(0, len(train_rep_ids), batch_size):
                batch_ids = train_rep_ids[i:i+batch_size]
                x, y, mask = prepare_batch_data(batch_ids, train_embeddings, train_labels)
                x, y, mask = x.to(device), y.to(device), mask.to(device)

                # Forward
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits, _ = model(x, mask)
                    loss = criterion(logits.squeeze(), y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for i in range(0, len(val_rep_ids), batch_size):
                    batch_ids = val_rep_ids[i:i+batch_size]
                    x, y, mask = prepare_batch_data(batch_ids, val_embeddings, val_labels)
                    x, y, mask = x.to(device), y.to(device), mask.to(device)

                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits, _ = model(x, mask)

                    val_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    val_true.extend(y.cpu().numpy())

            val_auc = roc_auc_score(val_true, val_preds)

            # Update best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {np.mean(train_losses):.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"Best: {best_val_auc:.4f}")

            scheduler.step()

        print(f"\n  Fold {fold_idx} Best Val AUC: {best_val_auc:.4f}")

        results[val_dataset] = {
            'auc': best_val_auc,
            'model_state': best_model_state,
        }

        # Clear GPU cache
        del model
        torch.cuda.empty_cache()

    # Compute mean CV AUC
    mean_auc = np.mean([res['auc'] for res in results.values()])
    std_auc = np.std([res['auc'] for res in results.values()])

    print(f"\n{'='*70}")
    print(f"LODO CV Results: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"{'='*70}")

    for dataset, res in results.items():
        print(f"  {dataset}: {res['auc']:.4f}")

    results['mean_auc'] = mean_auc
    results['std_auc'] = std_auc

    return results


def generate_submission(
    model_class,
    model_states: Dict,
    train_data: Dict,
    test_data: Dict,
    output_path: Path,
    device: str = 'cuda',
):
    """Generate submission file using ensemble of LODO models."""
    print("\nGenerating submission...")

    # Task A: Predict test repertoires
    test_predictions = {}

    for dataset_name, test_embeddings_dict in test_data.items():
        print(f"  Processing {dataset_name}...")

        rep_ids = list(test_embeddings_dict.keys())
        all_preds = []

        # Ensemble prediction (average over all LODO models)
        for fold_name, result in model_states.items():
            if fold_name in ['mean_auc', 'std_auc']:
                continue

            model = model_class().to(device)
            model.load_state_dict(result['model_state'])
            model.eval()

            preds = []
            with torch.no_grad():
                for rep_id in rep_ids:
                    emb = test_embeddings_dict[rep_id]  # (n_seqs, embed_dim)
                    x = torch.from_numpy(emb).unsqueeze(0).to(device)  # (1, n_seqs, embed_dim)

                    # Create mask (no padding for real data)
                    mask = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)

                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits, _ = model(x, mask)

                    preds.append(torch.sigmoid(logits).cpu().item())

            all_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        # Average predictions
        ensemble_preds = np.mean(all_preds, axis=0)

        for rep_id, pred in zip(rep_ids, ensemble_preds):
            test_predictions[rep_id] = pred

    # Create submission DataFrame
    submission_rows = []

    for rep_id, prob in test_predictions.items():
        submission_rows.append({
            'ID': rep_id,
            'dataset': 'test_dataset_X',  # Will be corrected below
            'label_positive_probability': prob,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0,
        })

    # Task B: Placeholder (k-mer baseline)
    print("\n  Task B: Using k-mer baseline (placeholder)")
    # (User should implement Task B separately)

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to {output_path}")
    print(f"  Total rows: {len(submission_df)}")


def main():
    parser = argparse.ArgumentParser(description='Train MIL architectures for AIRR-ML-25')
    parser.add_argument('--model', type=str, required=True,
                        choices=['eamil', 'transformer', 'deepset'],
                        help='MIL architecture to train')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--cache-dir', type=str,
                        default='/app/cache/esm2_15b_optimized',
                        help='Cache directory for embeddings')
    parser.add_argument('--output-dir', type=str,
                        default='/app/outputs/submissions',
                        help='Output directory')
    args = parser.parse_args()

    # Enable GB10 optimizations
    enable_gb10_optimizations()

    # Paths
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    train_data, test_data = load_embeddings(cache_dir)

    # Model configuration
    model_configs = {
        'eamil': {
            'class': EAMIL,
            'kwargs': {
                'input_dim': 5120,
                'hidden_dim': 1024,
                'attention_dim': 512,
                'dropout': 0.1,
                'use_gated': True,
            }
        },
        'transformer': {
            'class': TransformerMIL,
            'kwargs': {
                'input_dim': 5120,
                'hidden_dim': 1024,
                'num_heads': 8,
                'num_layers': 2,
                'dropout': 0.1,
                'use_flash_attention': True,
            }
        },
        'deepset': {
            'class': DeepSetMIL,
            'kwargs': {
                'input_dim': 5120,
                'hidden_dim': 1024,
                'num_layers': 3,
                'dropout': 0.1,
                'aggregation': 'attention',
            }
        }
    }

    config = model_configs[args.model]
    print(f"\n{'='*70}")
    print(f"Training {args.model.upper()} for AIRR-ML-25")
    print(f"{'='*70}")

    # Train LODO CV
    results = train_lodo_cv(
        model_class=config['class'],
        model_kwargs=config['kwargs'],
        train_data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save results
    results_file = output_dir / f'{args.model}_lodo_results.json'
    with open(results_file, 'w') as f:
        # Remove model states (too large for JSON)
        json_results = {
            k: v for k, v in results.items()
            if k in ['mean_auc', 'std_auc']
        }
        json_results.update({
            k: {'auc': v['auc']} for k, v in results.items()
            if k not in ['mean_auc', 'std_auc']
        })
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Generate submission
    submission_file = output_dir / f'submission_{args.model}.csv'
    generate_submission(
        model_class=config['class'],
        model_states=results,
        train_data=train_data,
        test_data=test_data,
        output_path=submission_file,
    )

    print("\n" + "="*70)
    print(f"Training complete! Mean CV AUC: {results['mean_auc']:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
