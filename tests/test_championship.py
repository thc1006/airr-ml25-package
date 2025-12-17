#!/usr/bin/env python3
"""
Quick test script to verify championship_dl.py components work
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from championship_dl import (
    ESM2FeatureExtractor,
    extract_features_from_repertoire,
    load_dataset,
    ChampionshipClassifier
)

def test_feature_extraction():
    """Test traditional feature extraction on one repertoire."""
    print("=" * 70)
    print("TEST 1: Traditional Feature Extraction")
    print("=" * 70)

    # Test on first file from dataset 1
    dataset_path = './data/train_datasets/train_datasets/train_dataset_1'
    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))

    first_file = metadata.iloc[0]['filename']
    tsv_path = os.path.join(dataset_path, first_file)

    print(f"\nTesting on: {first_file}")
    features, sequences = extract_features_from_repertoire(tsv_path)

    print(f"✓ Extracted {len(features)} features")
    print(f"✓ Found {len(sequences)} sequences")
    print(f"\nSample features:")
    for i, (k, v) in enumerate(list(features.items())[:10]):
        print(f"  {k}: {v:.4f}")
    print(f"\nSample sequences:")
    for seq in sequences[:5]:
        print(f"  {seq}")

    return True


def test_esm2_extraction():
    """Test ESM-2 embedding extraction on sample sequences."""
    print("\n" + "=" * 70)
    print("TEST 2: ESM-2 Embedding Extraction")
    print("=" * 70)

    # Initialize ESM-2
    print("\nInitializing ESM-2 (this may take a minute)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_extractor = ESM2FeatureExtractor(model_name="esm2_t33_650M_UR50D", device=device)

    # Test sequences
    test_sequences = [
        "CASSLDPGLDTDTQYF",
        "CASSLWAGDGYEQFF",
        "CASSQGQAYEQYF",
        "CASSLGQGAPYEQYF"
    ]

    print(f"\nExtracting embeddings for {len(test_sequences)} sequences...")
    embeddings = esm_extractor.extract_embeddings(test_sequences, batch_size=4)

    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  Expected: ({len(test_sequences)}, 1280)")
    print(f"✓ Embedding mean: {embeddings.mean():.4f}")
    print(f"✓ Embedding std: {embeddings.std():.4f}")

    return esm_extractor


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Forward Pass")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy data
    batch_size = 2
    n_sequences = 100
    esm_dim = 1280
    trad_dim = 150

    print(f"\nCreating dummy batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequences per repertoire: {n_sequences}")
    print(f"  ESM dimension: {esm_dim}")
    print(f"  Traditional features: {trad_dim}")

    esm_embeddings = torch.randn(batch_size, n_sequences, esm_dim).to(device)
    trad_features = torch.randn(batch_size, trad_dim).to(device)
    masks = torch.ones(batch_size, n_sequences, dtype=torch.bool).to(device)

    # Initialize model
    print("\nInitializing model...")
    model = ChampionshipClassifier(esm_dim=esm_dim, trad_dim=trad_dim).to(device)

    # Forward pass
    print("Running forward pass...")
    logits, attention_weights = model(esm_embeddings, trad_features, masks)

    print(f"✓ Logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 1)")
    print(f"✓ Attention weights shape: {attention_weights.shape}")

    # Test with sigmoid
    probs = torch.sigmoid(logits)
    print(f"✓ Probabilities: {probs.squeeze().cpu().detach().numpy()}")

    return True


def test_data_loading():
    """Test loading one dataset."""
    print("\n" + "=" * 70)
    print("TEST 4: Dataset Loading (First 10 repertoires)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize ESM-2
    print("\nInitializing ESM-2...")
    esm_extractor = ESM2FeatureExtractor(model_name="esm2_t33_650M_UR50D", device=device)

    # Get feature names from first dataset
    dataset_path = './data/train_datasets/train_datasets/train_dataset_1'
    metadata = pd.read_csv(os.path.join(dataset_path, 'metadata.csv'))

    all_feature_names = set()
    for filename in metadata['filename'].head(10):
        tsv_path = os.path.join(dataset_path, filename)
        features, _ = extract_features_from_repertoire(tsv_path)
        all_feature_names.update(features.keys())

    all_feature_names = sorted(list(all_feature_names))
    print(f"✓ Found {len(all_feature_names)} unique features")

    # Temporarily modify load_dataset to only process 10 samples
    print(f"\nLoading first 10 repertoires from dataset 1...")

    # Manual loading of first 10
    from championship_dl import standardize_features
    repertoire_data = []

    for idx, row in metadata.head(10).iterrows():
        repertoire_id = row['repertoire_id']
        filename = row['filename']
        label = 1 if row['label_positive'] else 0

        tsv_path = os.path.join(dataset_path, filename)
        trad_features_dict, sequences = extract_features_from_repertoire(tsv_path)

        if len(sequences) > 0:
            esm_embeddings = esm_extractor.extract_embeddings(sequences, batch_size=32, max_seqs=100)
            trad_features = standardize_features(trad_features_dict, all_feature_names)

            repertoire_data.append({
                'esm_embeddings': esm_embeddings,
                'trad_features': trad_features,
                'label': label,
                'repertoire_id': repertoire_id,
                'dataset_id': 1
            })

        print(f"  ✓ {idx+1}/10: {repertoire_id} (label={label}, seqs={esm_embeddings.shape[0]})")

    print(f"\n✅ Successfully loaded {len(repertoire_data)} repertoires!")
    print(f"   ESM embedding dim: {repertoire_data[0]['esm_embeddings'].shape[1]}")
    print(f"   Traditional feature dim: {len(all_feature_names)}")

    return True


def main():
    """Run all tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🧪 Championship Pipeline Component Tests                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        # Test 1: Traditional features
        test_feature_extraction()

        # Test 2: ESM-2 embeddings
        test_esm2_extraction()

        # Test 3: Model forward pass
        test_model_forward()

        # Test 4: Data loading
        test_data_loading()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe pipeline is ready for full training!")
        print("Run: ./start_championship_training.sh")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
