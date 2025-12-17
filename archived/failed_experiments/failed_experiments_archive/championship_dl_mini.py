#!/usr/bin/env python3
"""
🏆 Championship Mini - Quick test on 2 datasets
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score

# Import from main module
from championship_dl import (
    ESM2FeatureExtractor,
    ChampionshipClassifier,
    RepertoireDataset,
    collate_repertoires,
    extract_features_from_repertoire,
    standardize_features,
    device
)

def load_dataset_mini(dataset_path: str, dataset_id: int, esm_extractor: ESM2FeatureExtractor,
                     all_feature_names: List[str], max_samples: int = 50) -> List[Dict]:
    """Load one dataset with limited samples for testing."""
    print(f"\n📂 Loading mini dataset {dataset_id} (max {max_samples} samples)")

    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path).head(max_samples)

    repertoire_data = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Dataset {dataset_id}"):
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = 1 if row['label_positive'] else 0

        tsv_path = os.path.join(dataset_path, filename)

        try:
            trad_features_dict, sequences = extract_features_from_repertoire(tsv_path)
            if len(sequences) == 0:
                continue

            # Sample fewer sequences for speed
            esm_embeddings = esm_extractor.extract_embeddings(sequences, batch_size=32, max_seqs=200)
            trad_features = standardize_features(trad_features_dict, all_feature_names)

            repertoire_data.append({
                'esm_embeddings': esm_embeddings,
                'trad_features': trad_features,
                'label': label,
                'repertoire_id': repertoire_id,
                'dataset_id': dataset_id
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print(f"✓ Loaded {len(repertoire_data)} repertoires")
    return repertoire_data


def train_mini():
    """Quick training on 2 datasets."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏆 Championship MINI Test - 2 Datasets                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    train_root = './data/train_datasets/train_datasets'

    # Initialize ESM-2
    print("\n🔧 Initializing ESM-2...")
    esm_extractor = ESM2FeatureExtractor(model_name="esm2_t33_650M_UR50D", device=device)

    # Collect feature names from both datasets
    print("\n🔍 Collecting feature names...")
    all_feature_names_set = set()

    for dataset_id in [1, 2]:
        dataset_path = os.path.join(train_root, f'train_dataset_{dataset_id}')
        metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))

        for filename in metadata['filename'].head(10):
            tsv_path = os.path.join(dataset_path, filename)
            features, _ = extract_features_from_repertoire(tsv_path)
            all_feature_names_set.update(features.keys())

    all_feature_names = sorted(list(all_feature_names_set))
    print(f"✓ Found {len(all_feature_names)} features")

    # Load 2 datasets (50 samples each)
    print("\n📥 Loading datasets...")
    dataset1 = load_dataset_mini(
        os.path.join(train_root, 'train_dataset_1'), 1, esm_extractor, all_feature_names, max_samples=50
    )
    dataset2 = load_dataset_mini(
        os.path.join(train_root, 'train_dataset_2'), 2, esm_extractor, all_feature_names, max_samples=50
    )

    # Train on dataset 1, validate on dataset 2
    print("\n🎯 Training: Dataset 1 → Validate: Dataset 2")
    print(f"Train samples: {len(dataset1)}")
    print(f"Val samples: {len(dataset2)}")

    # Create dataloaders
    train_dataset = RepertoireDataset(dataset1)
    val_dataset = RepertoireDataset(dataset2)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                             collate_fn=collate_repertoires, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                           collate_fn=collate_repertoires, num_workers=2)

    # Initialize model
    trad_dim = len(all_feature_names)
    model = ChampionshipClassifier(esm_dim=1280, trad_dim=trad_dim,
                                   hidden_dims=[256, 128], dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Train for 5 epochs
    print("\n🚀 Starting training (5 epochs)...")

    for epoch in range(5):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            esm_emb = batch['esm_embeddings'].to(device)
            trad_feat = batch['trad_features'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model(esm_emb, trad_feat, masks)
            logits = logits.squeeze()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(probs)
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                esm_emb = batch['esm_embeddings'].to(device)
                trad_feat = batch['trad_features'].to(device)
                masks = batch['masks'].to(device)
                labels = batch['labels'].to(device)

                logits, _ = model(esm_emb, trad_feat, masks)
                logits = logits.squeeze()

                loss = criterion(logits, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} AUC={train_auc:.4f} | Val Loss={val_loss:.4f} AUC={val_auc:.4f}")

    print("\n✅ Mini training complete!")
    print(f"   Final Val AUC: {val_auc:.4f}")
    print("\nIf this looks good, run full training with:")
    print("   ./start_championship_training.sh")


if __name__ == '__main__':
    train_mini()
