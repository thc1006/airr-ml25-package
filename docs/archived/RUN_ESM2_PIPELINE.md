# ESM2 Pipeline Execution Guide

## Pre-flight Checklist

```bash
# 1. Check GPU
nvidia-smi

# Expected: RTX 5080, 16GB VRAM, ~15GB free

# 2. Check Python environment
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Expected: PyTorch 2.x, CUDA: True

# 3. Check transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Expected: transformers 4.x

# 4. Check data
ls -lh data/train_datasets/train_datasets/

# Expected: train_dataset_1 through train_dataset_8
```

## Execution Steps

### Step 1: Quick Validation (5 minutes)

```bash
python3 test_esm2_quick.py
```

**What to expect:**
- Model downloads (~30MB)
- GPU initialization
- Sample embeddings extracted
- Feature dimensions validated
- ✅ "Test completed successfully!"

**If errors:**
- GPU not available → Check CUDA installation
- Model download fails → Check internet connection
- Import errors → Check requirements.txt

### Step 2: Single Dataset Test (10-15 minutes)

```bash
python3 test_esm2_single_dataset.py
```

**What to expect:**
- Processing ~100-200 repertoires
- Progress bar showing extraction
- Checkpoint saved to `checkpoints_esm2/esm2_features_train_dataset_1.pkl`
- ✅ "Checkpoint verified"

**Monitor GPU:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Expected GPU usage:**
- Memory: 5-8 GB
- Utilization: 80-100%
- Temperature: <80°C

### Step 3: Full Extraction (2-3 hours)

```bash
# Option A: Use script (recommended)
./extract_esm2_features.sh

# Option B: Run Python directly
python3 champion_v13_esm2.py
```

**What to expect:**
- 8 datasets processing sequentially
- ~20-40 minutes per dataset
- Progress bars for each dataset
- Checkpoints saved after each dataset
- Total: 2-3 hours

**Output structure:**
```
checkpoints_esm2/
├── esm2_features_train_dataset_1.pkl  (~50-100 MB)
├── esm2_features_train_dataset_2.pkl
├── esm2_features_train_dataset_3.pkl
├── esm2_features_train_dataset_4.pkl
├── esm2_features_train_dataset_5.pkl
├── esm2_features_train_dataset_6.pkl
├── esm2_features_train_dataset_7.pkl
└── esm2_features_train_dataset_8.pkl
```

**Resumption:**
If interrupted, just re-run - it will skip completed datasets.

### Step 4: Train Integrated Model (15-25 minutes)

```bash
python3 champion_v14_esm_integrated.py
```

**What to expect:**
- Loading ESM2 features from checkpoints
- Extracting traditional features (k-mers, VJ)
- Feature integration and selection
- 5-fold cross-validation
- XGBoost + LightGBM training
- Final CV score: **Target 0.80+**

**Output:**
```
Mean CV AUC: 0.XXXX ± 0.XXXX
```

### Step 5: Generate Submission (TODO)

**Create submission script that:**
1. Loads trained models from Step 4
2. Processes test datasets
3. Generates predictions
4. Formats submission CSV

## Timeline

| Step | Duration | Can Run Unattended? |
|------|----------|---------------------|
| 1. Quick validation | 5 min | No (quick) |
| 2. Single dataset test | 10-15 min | Yes |
| 3. Full extraction | 2-3 hours | **Yes** |
| 4. Model training | 15-25 min | Yes |
| 5. Submission | TBD | TBD |
| **Total** | **~3-4 hours** | **Mostly yes** |

## Best Practices

### Run Order
1. **Morning:** Start Step 1 & 2 (validate everything works)
2. **Afternoon:** Start Step 3 (full extraction, run unattended)
3. **Evening:** Run Step 4 (train model)
4. **Night:** Create and test submission script

### Resource Management
- **Close other GPU applications** (browsers with GPU acceleration, games, etc.)
- **Keep terminal open** or use `screen`/`tmux`
- **Monitor disk space** (~2-3 GB for checkpoints)

### Troubleshooting

**GPU OOM (Out of Memory):**
```python
# Edit champion_v13_esm2.py, line 44
batch_size=16  # Reduce from 32
```

**Extraction too slow:**
```python
# Edit champion_v13_esm2.py, line 45
max_seqs_per_repertoire=250  # Reduce from 500
```

**Checkpoint corrupted:**
```bash
# Delete and re-run for specific dataset
rm checkpoints_esm2/esm2_features_train_dataset_X.pkl
python3 champion_v13_esm2.py
```

## Screen/Tmux Usage (Recommended)

### Using screen
```bash
# Start session
screen -S esm2

# Run extraction
./extract_esm2_features.sh

# Detach: Ctrl+A, then D

# Later, reattach
screen -r esm2

# Kill session
screen -X -S esm2 quit
```

### Using tmux
```bash
# Start session
tmux new -s esm2

# Run extraction
./extract_esm2_features.sh

# Detach: Ctrl+B, then D

# Later, reattach
tmux attach -t esm2

# Kill session
tmux kill-session -t esm2
```

## Progress Monitoring

### Check extraction progress
```bash
# Count completed checkpoints
ls checkpoints_esm2/*.pkl | wc -l

# Check sizes
du -sh checkpoints_esm2/

# View latest log (if using screen/tmux)
tail -f /path/to/logfile
```

### GPU monitoring
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv -l 5 > gpu_log.csv
```

## Verification

### After Step 3 (Full Extraction)
```bash
# Check all checkpoints exist
python3 -c "
from pathlib import Path
import pandas as pd

checkpoint_dir = Path('checkpoints_esm2')
datasets = [f'train_dataset_{i}' for i in range(1, 9)]

for ds in datasets:
    ckpt = checkpoint_dir / f'esm2_features_{ds}.pkl'
    if ckpt.exists():
        # Load and check
        import pickle
        with open(ckpt, 'rb') as f:
            df = pickle.load(f)
        print(f'✓ {ds}: {df.shape[0]} repertoires, {df.shape[1]} features')
    else:
        print(f'✗ {ds}: MISSING')
"
```

### After Step 4 (Training)
- CV AUC should be reported
- Target: >0.80
- If lower, check feature integration

## Expected Results

### Feature Counts
- Traditional features: ~708
- ESM2 features: 1280
- Total: ~1988
- After selection: ~1000

### Performance
- Baseline (V12): ~0.75-0.78 AUC
- **Expected (V14): ~0.80-0.82 AUC**
- Current top: 0.81364

### If Results Below Expectations
1. Check if ESM2 features loaded correctly
2. Verify feature dimensions match
3. Try different feature selection threshold
4. Experiment with ensemble weights

## Next Steps After Success

1. **Create submission script**
   - Based on `champion_v14_esm_integrated.py`
   - Process test datasets
   - Generate predictions

2. **Submit to Kaggle**
   ```bash
   kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \
                             -f submission.csv -m "V14: ESM2 + Traditional"
   ```

3. **Iterate if needed**
   - Try larger ESM2 model (t12, t30)
   - Tune hyperparameters
   - Ensemble with other approaches

## Emergency Recovery

### If extraction fails midway
```bash
# Check which datasets completed
ls checkpoints_esm2/

# Re-run (skips completed)
python3 champion_v13_esm2.py
```

### If GPU crashes
```bash
# Reset GPU
sudo nvidia-smi --gpu-reset

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart extraction
python3 champion_v13_esm2.py
```

### If out of disk space
```bash
# Clear cache
rm -rf cache_esm2/

# Check space
df -h

# Resume
python3 champion_v13_esm2.py
```

## Support

- **Quick Start:** See `ESM2_QUICKSTART.md`
- **Technical Details:** See `ESM2_INTEGRATION_GUIDE.md`
- **Implementation:** See `V13_V14_IMPLEMENTATION_SUMMARY.md`

---

**Ready to start? Run Step 1 now!**

```bash
python3 test_esm2_quick.py
```
