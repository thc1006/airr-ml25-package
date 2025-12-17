# 🏆 Championship Deep Learning Pipeline - Execution Guide

## Status: ✅ READY TO TRAIN

Target: Beat GROZD (0.81364) → Achieve 0.82+

---

## Architecture Summary

```
Input Repertoire (25,000 TCR sequences)
    ↓
ESM-2 Embeddings (650M params) → [N, 1280]
    ↓
Multi-Head Attention Aggregation → [1, 1280]
    +
Traditional Features (V/J usage, clonality) → [1, ~150]
    ↓
MLP Classifier [512, 256] → Probability
```

---

## Quick Start

### Option 1: Mini Test (5 epochs, 2 datasets, 100 samples)
```bash
python3 championship_dl_mini.py
```
- **Time**: ~10-15 minutes
- **Purpose**: Verify everything works
- **Output**: Quick AUC score on 2 datasets

### Option 2: Full Training (25 epochs, 8-fold CV, all data)
```bash
./start_championship_training.sh
```
- **Time**: ~6-12 hours (depending on data size)
- **Purpose**: Train championship models
- **Output**: 8 trained models in `./models/`

---

## Full Training Pipeline

### What Happens During Training

1. **Phase 1: Data Loading** (~30-60 min)
   - Loads 8 training datasets
   - Extracts ESM-2 embeddings (1280-dim) for each repertoire
   - Extracts traditional features (V/J usage, clonality)
   - Total samples: ~2,000-4,000 repertoires

2. **Phase 2: Leave-One-Dataset-Out CV** (~5-10 hours)
   - For each of 8 datasets:
     - Train on 7 datasets
     - Validate on 1 held-out dataset
     - Early stopping (patience=5)
     - Save best model
   - Expected per-fold AUC: 0.75-0.85

3. **Phase 3: Model Checkpoints**
   - Saves to `./models/championship_fold{1-8}.pt`
   - Each contains:
     - `model_state_dict`: Trained weights
     - `fold_id`: Which dataset was held out
     - `val_auc`: Best validation AUC
     - `epoch`: Best epoch number

---

## Monitoring Training

### View Progress in Real-Time
```bash
tail -f ./logs/championship_training.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Expected Output Patterns

```
📊 LOADING ALL TRAINING DATA
========================================
🔍 Phase 1: Collecting all feature names...
✓ Found 200-300 unique features

📥 Phase 2: Loading datasets with ESM-2 extraction...
📂 Loading dataset 1 from ./data/train_datasets/...
Dataset 1: 100%|████████| 500/500 [10:00<00:00]
✓ Loaded 500 repertoires from dataset 1

[Repeat for datasets 2-8]

✅ Total loaded: 2000-4000 repertoires

🎓 LEAVE-ONE-DATASET-OUT CROSS-VALIDATION
========================================
🎯 TRAINING FOLD 1/8
Train samples: 1800
Val samples: 200

--- Epoch 1/25 ---
Training: 100%|████████| 225/225 [05:00<00:00]
Train Loss: 0.6234 | Train AUC: 0.6543
Evaluating: 100%|████████| 25/25 [00:30<00:00]
Val Loss: 0.5987 | Val AUC: 0.7123
✓ New best model saved! AUC: 0.7123

--- Epoch 2/25 ---
...

✅ Fold 1 complete. Best Val AUC: 0.8045

[Repeat for folds 2-8]

📈 CROSS-VALIDATION RESULTS
========================================
Fold 1: Val AUC = 0.8045
Fold 2: Val AUC = 0.7823
Fold 3: Val AUC = 0.8156
...
Fold 8: Val AUC = 0.7934

🎯 Mean AUC: 0.8012 ± 0.0156
```

---

## Memory Management

### GPU Memory Usage
- ESM-2 model: ~3-4 GB
- Batch processing: ~2-3 GB
- Training: ~4-5 GB
- **Total**: ~10-12 GB (fits in RTX 5080 16GB)

### Disk Space
- Trained models: ~5 GB per fold × 8 = 40 GB
- Training logs: ~100 MB
- Cached ESM-2 weights: ~2.5 GB

---

## Troubleshooting

### Out of Memory (OOM)
```python
# In championship_dl.py, reduce:
batch_size=8 → batch_size=4  # Line 593, 595
max_seqs=1000 → max_seqs=500  # Line 441
```

### Slow Data Loading
```python
# Reduce workers if CPU bottleneck:
num_workers=4 → num_workers=2  # Line 594, 596
```

### Training Stalls
```bash
# Check if still running:
ps aux | grep championship_dl.py

# Check GPU usage:
nvidia-smi

# If stuck, restart:
pkill -f championship_dl.py
./start_championship_training.sh
```

---

## Next Steps After Training

### 1. Verify Models Exist
```bash
ls -lh ./models/
# Should show:
# championship_fold1.pt
# championship_fold2.pt
# ...
# championship_fold8.pt
```

### 2. Generate Test Predictions
Currently **TODO** - Need to implement:
```python
def generate_predictions(test_root, models, esm_extractor):
    """
    Load 11 test datasets
    For each test repertoire:
      - Extract features
      - Ensemble predict with 8 models
      - Average probabilities
    Generate submissions.csv (404,213 rows)
    """
```

### 3. Submit to Kaggle
```bash
kaggle competitions submit \
  -c adaptive-immune-profiling-challenge-2025 \
  -f championship_submission.csv \
  -m "Deep Learning: ESM-2 + Attention + 8-fold CV"
```

---

## Expected Performance

### Conservative Estimate
- **Train AUC**: 0.85-0.90 (may overfit)
- **Validation AUC**: 0.75-0.82 (more realistic)
- **Public LB**: 0.78-0.83 (with domain shift)

### Target
- **Public LB**: 0.82+ (beat GROZD 0.81364)
- **Private LB**: 0.80+ (more stable)

---

## Files Overview

```
championship_dl.py              # Main training pipeline (fully implemented)
championship_dl_mini.py         # Quick test version (2 datasets, 5 epochs)
test_championship.py            # Component tests (all passing ✅)
start_championship_training.sh  # Background training launcher

./models/                       # Saved model checkpoints
./logs/                        # Training logs
  championship_training.log     # Full output
  championship.pid              # Process ID
```

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| ESM-2 Extraction | ✅ Complete | Tested on sample sequences |
| Traditional Features | ✅ Complete | V/J usage, clonality metrics |
| Attention Aggregation | ✅ Complete | Multi-head attention over sequences |
| Hybrid Classifier | ✅ Complete | ESM + Traditional → Probability |
| Data Loading | ✅ Complete | Handles all 8 datasets |
| Leave-One-Out CV | ✅ Complete | 8-fold training loop |
| Model Checkpointing | ✅ Complete | Saves best models |
| **Test Inference** | ⚠️ TODO | Need to implement prediction generation |
| **Submission Format** | ⚠️ TODO | Need Task A + Task B outputs |

---

## Contact & Support

If training completes successfully:
1. Check `./models/` for 8 checkpoint files
2. Review `./logs/championship_training.log` for final CV results
3. Implement test inference (see Next Steps above)

If issues occur:
1. Check log file for error messages
2. Verify GPU memory with `nvidia-smi`
3. Test mini version first: `python3 championship_dl_mini.py`

---

**Last Updated**: 2025-12-08
**Status**: Ready for training 🚀
