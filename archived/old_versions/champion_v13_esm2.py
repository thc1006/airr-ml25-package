#!/usr/bin/env python3
"""
Champion V13 - ESM2 Feature Extractor
======================================
Production-grade protein language model feature extraction for AIRR-ML-25.

Features:
- GPU-accelerated ESM2 inference
- Memory-efficient batch processing
- Robust error handling and checkpointing
- Multi-level aggregation statistics
- Compatible with champion_v12 architecture

Performance targets:
- Speed: 30-60 sec/repertoire (500 sequences)
- Memory: <10GB GPU
- Total time: 2-3 hours for all 8 datasets
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from dataclasses import dataclass
from tqdm import tqdm
import logging
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class ESM2Config:
    """ESM2 feature extraction configuration"""
    model_name: str = "facebook/esm2_t6_8M_UR50D"  # 6-layer, 8M params - fastest
    device: str = "cuda"
    batch_size: int = 32
    max_seqs_per_repertoire: int = 500
    max_sequence_length: int = 50
    target_layer: int = 6  # Research shows layer 6 best for TCR/BCR

    # Output dimensions (auto-detected from model)
    embedding_dim: int = 320  # ESM2-t6: 320, ESM2-t12: 480, ESM2-t30: 640, ESM2-t33: 1280

    # Aggregation statistics
    agg_stats: List[str] = None

    def __post_init__(self):
        if self.agg_stats is None:
            self.agg_stats = ['mean', 'std', 'max', 'q75']

    @property
    def total_features(self) -> int:
        """Total features per repertoire"""
        return self.embedding_dim * len(self.agg_stats)


class ESM2FeatureExtractor:
    """
    Production-grade ESM2 feature extractor for immune repertoires.

    Optimized for:
    - Speed: Batch processing with GPU acceleration
    - Memory: Efficient sampling and aggregation
    - Robustness: Error handling and checkpointing

    Args:
        config: ESM2Config instance
        cache_dir: Directory for caching intermediate results

    Example:
        >>> extractor = ESM2FeatureExtractor()
        >>> features = extractor.extract_dataset_features(dataset_path)
        >>> print(features.shape)  # (n_repertoires, 1280)
    """

    def __init__(
        self,
        config: Optional[ESM2Config] = None,
        cache_dir: Optional[Path] = None
    ):
        self.config = config or ESM2Config()
        self.cache_dir = cache_dir or Path("./cache_esm2")
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Model and tokenizer (lazy loading)
        self._model = None
        self._tokenizer = None

        logger.info(f"ESM2FeatureExtractor initialized")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Total features per repertoire: {self.config.total_features}")

    def _load_model(self):
        """Lazy load model and tokenizer"""
        if self._model is not None:
            return

        logger.info("Loading ESM2 model...")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

        # Load model
        self._model = AutoModel.from_pretrained(self.config.model_name)
        self._model.eval()
        self._model.to(self.config.device)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        logger.info(f"Model loaded to {self.config.device}")
        logger.info(f"Model memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _sample_sequences(
        self,
        tsv_path: Path,
        max_seqs: int
    ) -> List[str]:
        """
        Sample sequences from repertoire with frequency weighting.

        Args:
            tsv_path: Path to TSV file
            max_seqs: Maximum number of sequences to sample

        Returns:
            List of sampled sequences
        """
        try:
            # Read TSV
            df = pd.read_csv(tsv_path, sep='\t')

            # Filter valid sequences
            df = df[df['junction_aa'].notna()].copy()
            df = df[df['junction_aa'].str.len() > 0]

            if len(df) == 0:
                return []

            # Check if templates column exists (frequency)
            if 'templates' in df.columns:
                # Weight by frequency
                weights = df['templates'].fillna(1).values
                weights = weights / weights.sum()
            else:
                # Uniform weights
                weights = None

            # Sample
            n_samples = min(max_seqs, len(df))
            if weights is not None:
                sampled_idx = np.random.choice(
                    len(df),
                    size=n_samples,
                    replace=False,
                    p=weights
                )
            else:
                sampled_idx = np.random.choice(
                    len(df),
                    size=n_samples,
                    replace=False
                )

            sequences = df.iloc[sampled_idx]['junction_aa'].tolist()

            # Truncate long sequences
            sequences = [
                seq[:self.config.max_sequence_length]
                for seq in sequences
                if isinstance(seq, str)
            ]

            return sequences

        except Exception as e:
            logger.error(f"Error sampling from {tsv_path}: {e}")
            return []

    def extract_sequence_embeddings(
        self,
        sequences: List[str]
    ) -> np.ndarray:
        """
        Extract ESM2 embeddings for a batch of sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            Array of shape (n_sequences, embedding_dim)
        """
        self._load_model()

        if len(sequences) == 0:
            return np.zeros((0, self.config.embedding_dim))

        try:
            # Tokenize
            inputs = self._tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)

                # Get embeddings from target layer
                # hidden_states: (n_layers, batch_size, seq_len, embedding_dim)
                hidden_states = outputs.hidden_states
                layer_embedding = hidden_states[self.config.target_layer]

                # Use [CLS] token (first token) as sequence representation
                # Shape: (batch_size, embedding_dim)
                cls_embeddings = layer_embedding[:, 0, :].cpu().numpy()

            return cls_embeddings

        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return np.zeros((len(sequences), self.config.embedding_dim))

    def extract_repertoire_features(
        self,
        tsv_path: Path,
        repertoire_id: str
    ) -> Optional[Dict[str, float]]:
        """
        Extract aggregated features for a single repertoire.

        Args:
            tsv_path: Path to repertoire TSV file
            repertoire_id: Unique identifier for this repertoire

        Returns:
            Dictionary of features (esm_mean_0, esm_std_1, ..., esm_q75_319)
            Returns None if extraction fails
        """
        # Check cache
        cache_file = self.cache_dir / f"{repertoire_id}_esm2.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {repertoire_id}: {e}")

        # Sample sequences
        sequences = self._sample_sequences(
            tsv_path,
            self.config.max_seqs_per_repertoire
        )

        if len(sequences) == 0:
            logger.warning(f"No sequences for {repertoire_id}")
            return None

        # Extract embeddings in batches
        all_embeddings = []
        n_batches = int(np.ceil(len(sequences) / self.config.batch_size))

        for i in range(n_batches):
            start_idx = i * self.config.batch_size
            end_idx = min((i + 1) * self.config.batch_size, len(sequences))
            batch = sequences[start_idx:end_idx]

            embeddings = self.extract_sequence_embeddings(batch)
            all_embeddings.append(embeddings)

        # Concatenate
        all_embeddings = np.vstack(all_embeddings)

        # Aggregate statistics
        features = {}

        for stat_name in self.config.agg_stats:
            if stat_name == 'mean':
                values = all_embeddings.mean(axis=0)
            elif stat_name == 'std':
                values = all_embeddings.std(axis=0)
            elif stat_name == 'max':
                values = all_embeddings.max(axis=0)
            elif stat_name == 'q75':
                values = np.percentile(all_embeddings, 75, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {stat_name}")

            # Add to features
            for i, val in enumerate(values):
                features[f'esm_{stat_name}_{i}'] = float(val)

        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.warning(f"Failed to cache {repertoire_id}: {e}")

        return features

    def extract_dataset_features(
        self,
        dataset_path: Path,
        dataset_id: str
    ) -> pd.DataFrame:
        """
        Extract features for all repertoires in a dataset.

        Args:
            dataset_path: Path to dataset directory
            dataset_id: Dataset identifier (e.g., 'train_dataset_1')

        Returns:
            DataFrame with columns: repertoire_id, esm_mean_0, ..., esm_q75_319
        """
        logger.info(f"Extracting ESM2 features for {dataset_id}")

        # Read metadata
        metadata_path = dataset_path / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = pd.read_csv(metadata_path)

        # Extract features for each repertoire
        results = []

        for idx, row in tqdm(
            metadata.iterrows(),
            total=len(metadata),
            desc=f"Processing {dataset_id}"
        ):
            repertoire_id = row['repertoire_id']
            filename = row['filename']
            tsv_path = dataset_path / filename

            if not tsv_path.exists():
                logger.warning(f"File not found: {tsv_path}")
                continue

            features = self.extract_repertoire_features(tsv_path, repertoire_id)

            if features is not None:
                features['repertoire_id'] = repertoire_id
                results.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Reorder columns (repertoire_id first)
        cols = ['repertoire_id'] + [c for c in df.columns if c != 'repertoire_id']
        df = df[cols]

        logger.info(f"Extracted features for {len(df)} repertoires")
        logger.info(f"Feature shape: {df.shape}")

        return df

    def save_checkpoint(
        self,
        features_df: pd.DataFrame,
        dataset_id: str,
        output_dir: Path
    ):
        """
        Save extracted features to checkpoint file.

        Args:
            features_df: DataFrame with extracted features
            dataset_id: Dataset identifier
            output_dir: Directory to save checkpoint
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"esm2_features_{dataset_id}.pkl"

        with open(output_path, 'wb') as f:
            pickle.dump(features_df, f)

        logger.info(f"Checkpoint saved: {output_path}")
        logger.info(f"Size: {output_path.stat().st_size / 1e6:.2f} MB")

    def load_checkpoint(
        self,
        dataset_id: str,
        output_dir: Path
    ) -> Optional[pd.DataFrame]:
        """
        Load features from checkpoint file.

        Args:
            dataset_id: Dataset identifier
            output_dir: Directory containing checkpoint

        Returns:
            DataFrame with features, or None if not found
        """
        checkpoint_path = output_dir / f"esm2_features_{dataset_id}.pkl"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def clear_cache(self):
        """Clear all cached embeddings"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for f in cache_files:
            f.unlink()
        logger.info(f"Cleared {len(cache_files)} cache files")


def main():
    """Main execution for ESM2 feature extraction"""

    # Configuration
    TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
    OUTPUT_DIR = Path("/home/thc1006/dev/airr-ml25-package/checkpoints_esm2")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Dataset IDs
    dataset_ids = [f"train_dataset_{i}" for i in range(1, 9)]

    # Initialize extractor
    config = ESM2Config(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        max_seqs_per_repertoire=500
    )

    extractor = ESM2FeatureExtractor(config=config)

    # Extract features for each dataset
    for dataset_id in dataset_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_id}")
        logger.info(f"{'='*60}")

        dataset_path = TRAIN_ROOT / dataset_id

        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        # Check if checkpoint exists
        existing_features = extractor.load_checkpoint(dataset_id, OUTPUT_DIR)
        if existing_features is not None:
            logger.info(f"Skipping {dataset_id} - checkpoint exists")
            continue

        try:
            # Extract features
            features_df = extractor.extract_dataset_features(
                dataset_path,
                dataset_id
            )

            # Save checkpoint
            extractor.save_checkpoint(features_df, dataset_id, OUTPUT_DIR)

            # GPU memory report
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, "
                          f"Reserved: {memory_reserved:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n" + "="*60)
    logger.info("ESM2 feature extraction complete!")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Summary
    checkpoint_files = list(OUTPUT_DIR.glob("esm2_features_*.pkl"))
    total_size = sum(f.stat().st_size for f in checkpoint_files) / 1e6
    logger.info(f"Total checkpoints: {len(checkpoint_files)}")
    logger.info(f"Total size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()
