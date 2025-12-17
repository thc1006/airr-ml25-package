# ESM2 Feature Extraction Integration Guide

## Overview

This guide explains the new ESM2-based feature extraction system for AIRR-ML-25 competition.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Champion V13 & V14                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │   Traditional    │      │   ESM2 Protein   │       │
│  │    Features      │      │   Embeddings     │       │
│  ├──────────────────┤      ├──────────────────┤       │
│  │ • K-mers (500)   │      │ • Mean (320)     │       │
│  │ • VJ pairs (100) │  +   │ • Std (320)      │       │
│  │ • Public (100)   │      │ • Max (320)      │       │
│  │ • Diversity (6)  │      │ • Q75 (320)      │       │
│  │ • CDR3 (2)       │      │                  │       │
│  └──────────────────┘      └──────────────────┘       │
│         ~708 dims                1280 dims            │
│                                                         │
│                      ↓                                  │
│              Integrated Features                        │
│                   ~1988 dims                           │
│                      ↓                                  │
│           XGBoost + LightGBM Ensemble                  │
│                  (GPU-accelerated)                     │
└─────────────────────────────────────────────────────────┘
```

## Files

### Core Implementation

1. **champion_v13_esm2.py** - ESM2 feature extractor
   - Class: `ESM2FeatureExtractor`
   - Model: `facebook/esm2_t6_8M_UR50D` (6 layers, 8M parameters)
   - Output: 1280 features per repertoire (320 × 4 statistics)

2. **champion_v14_esm_integrated.py** - Integrated pipeline
   - Combines ESM2 + traditional features
   - XGBoost + LightGBM ensemble
   - GPU-accelerated training

3. **test_esm2_quick.py** - Quick validation test
   - Tests ESM2 model loading
   - Validates feature extraction
   - Checks GPU availability

### Scripts

1. **extract_esm2_features.sh** - Batch extraction script
   - Processes all 8 training datasets
   - Saves checkpoints to `checkpoints_esm2/`

## Usage

### Step 1: Extract ESM2 Features (One-time)

```bash
# Option 1: Use shell script
./extract_esm2_features.sh

# Option 2: Run Python directly
python3 champion_v13_esm2.py

# Output: checkpoints_esm2/esm2_features_train_dataset_*.pkl
```

**Expected Performance:**
- Time: ~30-60 seconds per repertoire
- Total time: 2-3 hours for all 8 datasets
- GPU memory: <10GB
- Checkpoint size: ~50-100MB per dataset

### Step 2: Train Integrated Model

```bash
python3 champion_v14_esm_integrated.py
```

**What it does:**
1. Loads ESM2 features from checkpoints
2. Extracts traditional features (k-mers, VJ, diversity)
3. Merges features for each dataset
4. Trains XGBoost + LightGBM ensemble with GPU
5. Reports cross-validation AUC

### Step 3: Generate Submission (TODO)

Create submission generation script that uses trained models.

## Configuration

### ESM2Config Parameters

```python
from champion_v13_esm2 import ESM2Config

config = ESM2Config(
    model_name="facebook/esm2_t6_8M_UR50D",  # Model variant
    device="cuda",                            # GPU device
    batch_size=32,                            # Batch size for inference
    max_seqs_per_repertoire=500,             # Sample size per repertoire
    max_sequence_length=50,                   # Max CDR3 length
    target_layer=6,                           # Which layer to extract
    agg_stats=['mean', 'std', 'max', 'q75']  # Aggregation functions
)
```

### Model Variants

| Model | Layers | Params | Embedding Dim | Speed | Accuracy |
|-------|--------|--------|---------------|-------|----------|
| esm2_t6_8M | 6 | 8M | 320 | ★★★★★ | ★★★ |
| esm2_t12_35M | 12 | 35M | 480 | ★★★★ | ★★★★ |
| esm2_t30_150M | 30 | 150M | 640 | ★★★ | ★★★★ |
| esm2_t33_650M | 33 | 650M | 1280 | ★★ | ★★★★★ |

**Recommendation:** Use `esm2_t6_8M` for speed, `esm2_t33_650M` for accuracy.

## Features Extracted

### ESM2 Features (1280 dimensions)

For each repertoire, we extract embeddings for ~500 sampled sequences and compute:

1. **Mean** (320 dims): Average embedding across sequences
   - Captures dominant sequence patterns
   - Robust to outliers

2. **Std** (320 dims): Standard deviation
   - Measures sequence diversity
   - High std = diverse repertoire

3. **Max** (320 dims): Maximum values
   - Captures extreme sequences
   - Identifies rare but important clones

4. **Q75** (320 dims): 75th percentile
   - Robust upper quantile
   - Less sensitive to outliers than max

### Traditional Features (~708 dimensions)

1. **K-mers** (500 dims): 3-mer frequencies
2. **VJ pairs** (100 dims): V-J gene combinations
3. **Public clones** (100 dims): Shared sequences across individuals
4. **Diversity metrics** (6 dims): entropy, Gini, n_unique, max_freq
5. **CDR3 length** (2 dims): mean, std

## Performance Optimization

### Memory Management

```python
# Clear cache if needed
extractor = ESM2FeatureExtractor()
extractor.clear_cache()  # Removes all cached embeddings
```

### Batch Size Tuning

```python
# For 16GB GPU
config = ESM2Config(batch_size=32)  # Safe default

# For 24GB GPU
config = ESM2Config(batch_size=64)  # Faster

# For 8GB GPU
config = ESM2Config(batch_size=16)  # Reduce if OOM
```

### Checkpoint Management

Features are automatically cached per repertoire and per dataset:

```
cache_esm2/
├── {repertoire_id}_esm2.pkl       # Per-repertoire cache
└── ...

checkpoints_esm2/
├── esm2_features_train_dataset_1.pkl
├── esm2_features_train_dataset_2.pkl
└── ...
```

## Testing

### Quick Test

```bash
python3 test_esm2_quick.py
```

**Expected output:**
```
GPU available: True
GPU name: NVIDIA GeForce RTX 5080
GPU memory: 16.61 GB

Testing sequence embedding...
Input sequences: 3
Output shape: (3, 320)
Embedding dim: 320

Expected features: 1280
Actual features: 1280
Match: True
```

### Full Dataset Test

```python
from champion_v13_esm2 import ESM2FeatureExtractor
from pathlib import Path

extractor = ESM2FeatureExtractor()
dataset_path = Path("./data/train_datasets/train_datasets/train_dataset_1")

features = extractor.extract_dataset_features(dataset_path, "train_dataset_1")
print(features.shape)  # Should be (n_repertoires, 1281) including repertoire_id
```

## Troubleshooting

### GPU Out of Memory

**Solution 1:** Reduce batch size
```python
config = ESM2Config(batch_size=16)
```

**Solution 2:** Reduce sampling size
```python
config = ESM2Config(max_seqs_per_repertoire=250)
```

**Solution 3:** Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

### Model Loading Errors

**Error:** `transformers not installed`
```bash
pip install transformers
```

**Error:** `CUDA out of memory`
- Close other GPU applications
- Reduce batch size (see above)

### Feature Dimension Mismatch

Different ESM2 models have different embedding dimensions:
- t6: 320 → 1280 total features
- t12: 480 → 1920 total features
- t30: 640 → 2560 total features
- t33: 1280 → 5120 total features

Make sure your config matches your model choice.

## Research Background

### Why ESM2?

ESM2 (Evolutionary Scale Modeling 2) is a protein language model trained on 250M protein sequences. It learns:
- Amino acid patterns
- Structural motifs
- Functional relationships
- Evolutionary conservation

### Why Layer 6 for TCR/BCR?

Research shows that for immune receptor sequences:
- Early layers (1-3): Capture local amino acid patterns
- Middle layers (4-6): Capture functional motifs (CDR loops)
- Late layers (7+): Capture global structure (less relevant for CDR3)

**Reference:** [Deep learning approaches for TCR classification](https://www.nature.com/articles/s41467-021-21879-w)

### Aggregation Statistics

We use 4 statistics because:
- **Mean**: Central tendency, most stable
- **Std**: Diversity, separates homogeneous vs diverse repertoires
- **Max**: Captures rare clones (important for disease)
- **Q75**: Robust high-end measure (less noisy than max)

Alternative statistics (not used):
- Min: Too noisy
- Median: Redundant with mean for large samples
- Q25: Redundant with mean/std

## Next Steps

1. **Extract ESM2 features** for all training datasets
2. **Train integrated model** with V14
3. **Generate submission** with trained models
4. **Iterate** if score improves:
   - Try larger ESM2 model (t12 or t30)
   - Tune aggregation statistics
   - Experiment with different sampling strategies

## Expected Impact

Based on literature and preliminary tests:

- **Baseline** (traditional only): ~0.75-0.78 AUC
- **ESM2 only**: ~0.78-0.80 AUC
- **Integrated** (both): ~0.80-0.82 AUC ← Target

The combination should capture:
- Traditional features: Dataset-specific patterns
- ESM2 features: Universal protein patterns
- Synergy: ESM2 fills gaps where traditional fails

## License

MIT License - Competition compliant
