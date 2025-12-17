#!/usr/bin/env python3
"""Debug champion_v6.py to find crash location."""

import sys
import traceback

print("[DEBUG] Starting debug script...")
sys.stdout.flush()

# Step 1: Basic imports
try:
    print("[1] Importing basic modules...")
    import os
    import gc
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from collections import Counter
    from typing import Dict, List, Optional
    print("[1] OK - Basic imports succeeded")
except Exception as e:
    print(f"[1] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: ML imports
try:
    print("[2] Importing ML modules...")
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import entropy, fisher_exact
    from tqdm import tqdm
    from joblib import Parallel, delayed
    print("[2] OK - ML imports succeeded")
except Exception as e:
    print(f"[2] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Check CatBoost
try:
    print("[3] Checking CatBoost...")
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
    print("[3] OK - CatBoost available")
except ImportError:
    HAS_CATBOOST = False
    print("[3] WARNING - CatBoost not available")

# Step 4: Check GPU
try:
    print("[4] Checking GPU...")
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
    use_gpu = result.returncode == 0
    print(f"[4] OK - GPU available: {use_gpu}")
except Exception as e:
    use_gpu = False
    print(f"[4] WARNING - No GPU: {e}")

# Step 5: Import champion_v6 components
try:
    print("[5] Importing champion_v6 components...")
    sys.path.insert(0, '/home/thc1006/dev/airr-ml25-package')

    # Import as module
    import champion_v6
    print(f"[5] OK - champion_v6 module imported")
    print(f"    Config.TRAIN_ROOT: {champion_v6.Config.TRAIN_ROOT}")
    print(f"    Config.K_LIST: {champion_v6.Config.K_LIST}")
except Exception as e:
    print(f"[5] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test dataset access
try:
    print("[6] Testing dataset access...")
    train_sets = sorted([d.name for d in champion_v6.Config.TRAIN_ROOT.glob('train_dataset_*')])
    print(f"[6] OK - Found {len(train_sets)} training datasets: {train_sets[:3]}...")
except Exception as e:
    print(f"[6] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test feature extractor
try:
    print("[7] Testing FeatureExtractorV6...")
    extractor = champion_v6.FeatureExtractorV6(champion_v6.Config.K_LIST)
    print("[7] OK - FeatureExtractorV6 created")
except Exception as e:
    print(f"[7] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 8: Test read_repertoire
try:
    print("[8] Testing read_repertoire...")
    ds_path = champion_v6.Config.TRAIN_ROOT / train_sets[0]
    meta = pd.read_csv(ds_path / 'metadata.csv')
    first_file = meta['filename'].iloc[0]
    df = champion_v6.read_repertoire(ds_path / first_file, champion_v6.Config.MAX_SEQUENCES_PER_FILE)
    print(f"[8] OK - Read repertoire: {len(df)} rows, columns: {list(df.columns)}")
except Exception as e:
    print(f"[8] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 9: Test public clone mining
try:
    print("[9] Testing mine_public_clones_v6...")
    ds_id = 1
    pub_dict = champion_v6.mine_public_clones_v6(
        ds_path,
        max_files=10,  # Reduced for speed
        min_freq=champion_v6.Config.PUB_MIN_FREQ,
        enrichment=champion_v6.Config.PUB_ENRICH,
        p_value_threshold=champion_v6.Config.PUB_P_VALUE,
        top_n=100,  # Reduced for speed
    )
    print(f"[9] OK - Found {len(pub_dict)} public clones")
except Exception as e:
    print(f"[9] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 10: Test single extract_all
try:
    print("[10] Testing extract_all on single file...")
    row = meta.iloc[0]
    feats = extractor.extract_all(df, pub_dict, row, ds_id)
    print(f"[10] OK - Extracted {len(feats)} features")
    print(f"     Sample features: {list(feats.keys())[:5]}...")
except Exception as e:
    print(f"[10] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 11: Test process_file_parallel_v6
try:
    print("[11] Testing process_file_parallel_v6...")
    result = champion_v6.process_file_parallel_v6(row, ds_path, ds_id, pub_dict, extractor)
    if result is None:
        print("[11] WARNING - process_file_parallel_v6 returned None!")
    else:
        print(f"[11] OK - Result has {len(result)} keys")
except Exception as e:
    print(f"[11] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 12: Test Parallel() call on small subset
try:
    print("[12] Testing Parallel() call with 5 files...")
    mini_meta = meta.head(5)

    results = Parallel(n_jobs=4, backend='loky')(
        delayed(champion_v6.process_file_parallel_v6)(row, ds_path, ds_id, pub_dict, extractor)
        for _, row in mini_meta.iterrows()
    )

    valid_results = [r for r in results if r is not None]
    print(f"[12] OK - Parallel returned {len(valid_results)}/{len(results)} valid results")

    if len(valid_results) == 0:
        print("[12] ERROR - All results are None!")
        sys.exit(1)
except Exception as e:
    print(f"[12] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 13: Test DataFrame creation
try:
    print("[13] Testing DataFrame creation...")
    df_train = pd.DataFrame([r for r in results if r is not None])
    print(f"[13] OK - Created DataFrame: {df_train.shape}")
    print(f"     Columns: {list(df_train.columns)[:10]}...")
except Exception as e:
    print(f"[13] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 14: Test EnsembleTrainerV6 creation
try:
    print("[14] Testing EnsembleTrainerV6 creation...")
    trainer = champion_v6.EnsembleTrainerV6(use_gpu=use_gpu, random_state=42)
    print("[14] OK - EnsembleTrainerV6 created")
except Exception as e:
    print(f"[14] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 15: Test training (with mini data)
try:
    print("[15] Testing training with mini data...")
    print("     This may take a moment...")
    sys.stdout.flush()

    # Need enough samples for training
    if len(df_train) < 5:
        print("[15] SKIP - Not enough samples for training test")
    else:
        trainer_result, fcols, score = trainer.train(df_train, ds_id)
        print(f"[15] OK - Training succeeded! CV AUC: {score:.4f}")
        print(f"     Feature columns: {len(fcols)}")
except Exception as e:
    print(f"[15] FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL DEBUG TESTS PASSED!")
print("="*60)
print("\nThe issue might be:")
print("1. Memory exhaustion with full dataset")
print("2. Process being killed by OOM killer")
print("3. GPU memory issues with CatBoost")
print("4. joblib worker crash with full parallelization")
