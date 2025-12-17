# ESM2 Feature Extraction - Quick Start Guide

## TL;DR

```bash
# 1. Test on single dataset (5-10 minutes)
python3 test_esm2_single_dataset.py

# 2. Extract all datasets (2-3 hours)
./extract_esm2_features.sh

# 3. Train integrated model
python3 champion_v14_esm_integrated.py
```

## What is ESM2?

ESM2 is a protein language model that converts amino acid sequences into numerical vectors (embeddings). Think of it as "BERT for proteins."

**Key benefits:**
- Captures patterns invisible to k-mer counting
- Pre-trained on 250M protein sequences
- State-of-the-art for protein tasks

## Files Created

### Production Code
1. `champion_v13_esm2.py` - Feature extractor (415 lines)
2. `champion_v14_esm_integrated.py` - Integrated pipeline (387 lines)

### Testing & Utilities
3. `test_esm2_quick.py` - Quick validation test
4. `test_esm2_single_dataset.py` - Single dataset test
5. `extract_esm2_features.sh` - Batch extraction script

### Documentation
6. `ESM2_INTEGRATION_GUIDE.md` - Complete technical guide
7. `ESM2_QUICKSTART.md` - This file

## Workflow

### Phase 1: Extract ESM2 Features (One-time, 2-3 hours)

```bash
# Option A: Test first (recommended)
python3 test_esm2_single_dataset.py  # 5-10 minutes
./extract_esm2_features.sh           # 2-3 hours for remaining 7 datasets

# Option B: Extract all at once
python3 champion_v13_esm2.py
```

**Output:**
```
checkpoints_esm2/
├── esm2_features_train_dataset_1.pkl  (~50-100 MB each)
├── esm2_features_train_dataset_2.pkl
├── ...
└── esm2_features_train_dataset_8.pkl
```

### Phase 2: Train Model

```bash
python3 champion_v14_esm_integrated.py
```

**What it does:**
1. Loads ESM2 features from checkpoints
2. Extracts traditional features (k-mers, VJ pairs)
3. Merges both feature types
4. Trains XGBoost + LightGBM ensemble
5. Reports 5-fold CV AUC

**Expected output:**
```
Mean CV AUC: 0.80XX ± 0.02XX
```

### Phase 3: Generate Submission (TODO)

You'll need to create a submission script that:
1. Loads trained models
2. Processes test datasets
3. Generates predictions
4. Formats submission CSV

## Performance Expectations

### ESM2 Extraction
- **Speed:** 30-60 seconds per repertoire
- **GPU Memory:** 5-8 GB (RTX 5080 has 16GB, plenty of headroom)
- **Total Time:** 2-3 hours for 8 datasets
- **Disk Space:** ~500 MB for all checkpoints

### Model Training
- **Time:** 10-20 minutes
- **GPU Memory:** 10-12 GB
- **CV Score:** Target 0.80+ AUC

## Feature Dimensions

```
Traditional Features:  ~708 dims
├── K-mers:           500 dims  (3-mer frequencies)
├── VJ pairs:         100 dims  (V-J gene combinations)
├── Public clones:    100 dims  (shared sequences)
├── Diversity:          6 dims  (entropy, Gini, etc.)
└── CDR3 length:        2 dims  (mean, std)

ESM2 Features:        1280 dims
├── Mean embeddings:  320 dims
├── Std embeddings:   320 dims
├── Max embeddings:   320 dims
└── Q75 embeddings:   320 dims

Total:               ~1988 dims
```

## System Requirements

### Minimum
- GPU: 8GB VRAM
- RAM: 16GB
- Disk: 20GB free

### Recommended (Current Setup)
- GPU: RTX 5080 16GB
- RAM: 32GB DDR5
- Disk: 100GB SSD

## Troubleshooting

### "CUDA out of memory"
```python
# Edit champion_v13_esm2.py line 44
batch_size=16  # Reduce from 32
```

### "transformers not installed"
```bash
pip install transformers
```

### "Checkpoint not found"
Make sure you ran extraction first:
```bash
python3 champion_v13_esm2.py
```

### Extraction too slow
Reduce sampling:
```python
# Edit champion_v13_esm2.py line 45
max_seqs_per_repertoire=250  # Reduce from 500
```

## Next Steps

1. **Now:** Run `test_esm2_single_dataset.py` to validate
2. **Today:** Extract all ESM2 features (2-3 hours)
3. **Today:** Train integrated model
4. **Tomorrow:** Create submission script
5. **Tomorrow:** Submit and iterate

## Expected Improvement

Based on research literature:

| Approach | Expected AUC | Notes |
|----------|--------------|-------|
| Traditional only | 0.75-0.78 | Baseline (k-mers, VJ) |
| ESM2 only | 0.78-0.80 | Protein patterns |
| **Integrated** | **0.80-0.82** | **Best of both** |
| Current top score | 0.81364 | GROZD team |
| **Target** | **>0.82** | **Beat top score** |

## Technical Details

For full technical documentation, see `ESM2_INTEGRATION_GUIDE.md`.

Key points:
- Model: `facebook/esm2_t6_8M_UR50D`
- Layer 6 embeddings (research-backed)
- 4 aggregation statistics (mean, std, max, q75)
- Frequency-weighted sampling
- Automatic checkpointing
- GPU-accelerated training

## Citation

If this approach wins, acknowledge:
- ESM2: Lin et al. (2022) "Language models of protein sequences at the scale of evolution"
- Competition: AIRR-ML-25 Challenge

## Questions?

Check these files:
1. `ESM2_INTEGRATION_GUIDE.md` - Full technical guide
2. `champion_v13_esm2.py` - Well-commented code
3. Test scripts - Run and inspect output

## License

MIT License - Competition compliant
