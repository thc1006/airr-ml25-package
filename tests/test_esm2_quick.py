#!/usr/bin/env python3
"""Quick test of ESM2 feature extractor"""

import torch
from pathlib import Path
from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config

def main():
    print("="*60)
    print("ESM2 Feature Extractor - Quick Test")
    print("="*60)

    # Check GPU
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize extractor
    config = ESM2Config(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        max_seqs_per_repertoire=50  # Small test
    )

    extractor = ESM2FeatureExtractor(config=config)

    # Test sequence embedding
    test_sequences = [
        "CASSLAPGATNEKLFF",  # Typical TCR CDR3
        "CASSLGQGTEAFF",
        "CASSPGTGGYEQYF"
    ]

    print(f"\nTesting sequence embedding...")
    print(f"Input sequences: {len(test_sequences)}")

    embeddings = extractor.extract_sequence_embeddings(test_sequences)
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Sample values: {embeddings[0, :5]}")

    # Test on real dataset (first dataset, first repertoire)
    TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
    test_dataset = TRAIN_ROOT / "train_dataset_1"

    if test_dataset.exists():
        print(f"\nTesting on real dataset: {test_dataset}")

        import pandas as pd
        metadata = pd.read_csv(test_dataset / "metadata.csv")
        first_rep = metadata.iloc[0]

        print(f"Repertoire ID: {first_rep['repertoire_id']}")
        print(f"Filename: {first_rep['filename']}")

        # Extract features
        tsv_path = test_dataset / first_rep['filename']
        features = extractor.extract_repertoire_features(
            tsv_path,
            first_rep['repertoire_id']
        )

        if features:
            print(f"\nExtracted features: {len(features)}")
            print(f"Feature names (first 5): {list(features.keys())[:5]}")
            print(f"Sample values:")
            for key in list(features.keys())[:3]:
                print(f"  {key}: {features[key]:.6f}")

            # Verify feature count
            expected = config.total_features
            actual = len([k for k in features.keys() if k != 'repertoire_id'])
            print(f"\nExpected features: {expected}")
            print(f"Actual features: {actual}")
            print(f"Match: {expected == actual}")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
