#!/usr/bin/env python3
"""
Test ESM2 feature extraction on a single dataset.
This serves as a quick validation before running on all 8 datasets.
"""

import torch
from pathlib import Path
from champion_v13_esm2 import ESM2FeatureExtractor, ESM2Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
    OUTPUT_DIR = Path("/home/thc1006/dev/airr-ml25-package/checkpoints_esm2")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Test on dataset 1 only
    dataset_id = "train_dataset_1"
    dataset_path = TRAIN_ROOT / dataset_id

    logger.info("="*60)
    logger.info("ESM2 Single Dataset Test")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_id}")
    logger.info(f"Path: {dataset_path}")

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available - using CPU (will be slow)")

    # Initialize extractor with small sample for testing
    config = ESM2Config(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        max_seqs_per_repertoire=100,  # Reduced for quick test
    )

    extractor = ESM2FeatureExtractor(config=config)

    # Extract features
    logger.info("\nExtracting features...")
    features_df = extractor.extract_dataset_features(dataset_path, dataset_id)

    # Report
    logger.info("\n" + "="*60)
    logger.info("Extraction Results")
    logger.info("="*60)
    logger.info(f"Repertoires processed: {len(features_df)}")
    logger.info(f"Total features: {len(features_df.columns) - 1}")  # Exclude repertoire_id
    logger.info(f"Feature shape: {features_df.shape}")

    # Show sample
    logger.info("\nSample features (first 5 columns):")
    print(features_df.iloc[:3, :5])

    # Save checkpoint
    logger.info("\nSaving checkpoint...")
    extractor.save_checkpoint(features_df, dataset_id, OUTPUT_DIR)

    # Verify checkpoint
    loaded_df = extractor.load_checkpoint(dataset_id, OUTPUT_DIR)
    assert loaded_df is not None, "Failed to load checkpoint"
    assert loaded_df.shape == features_df.shape, "Checkpoint shape mismatch"
    logger.info("✓ Checkpoint verified")

    logger.info("\n" + "="*60)
    logger.info("Test completed successfully!")
    logger.info(f"Next step: Run ./extract_esm2_features.sh to process all 8 datasets")
    logger.info("="*60)

if __name__ == "__main__":
    main()
