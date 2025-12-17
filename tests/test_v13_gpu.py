#!/usr/bin/env python3
"""
GPU availability test for Champion V13.
Tests XGBoost, LightGBM, and CatBoost GPU support.
"""

import numpy as np
import sys
from typing import Dict, List


def test_xgboost_gpu() -> bool:
    """Test XGBoost CUDA support."""
    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")

        # Test GPU training
        model = xgb.XGBClassifier(
            device='cuda',
            tree_method='hist',
            n_estimators=10,
            verbosity=0
        )

        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        pred = model.predict_proba(X)

        print("  ✓ XGBoost CUDA: OK")
        return True

    except Exception as e:
        print(f"  ✗ XGBoost CUDA: FAILED - {e}")
        return False


def test_lightgbm_gpu() -> bool:
    """Test LightGBM GPU support."""
    try:
        import lightgbm as lgb
        print(f"LightGBM version: {lgb.__version__}")

        # Test GPU training
        model = lgb.LGBMClassifier(
            device='gpu',
            n_estimators=10,
            verbose=-1
        )

        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        pred = model.predict_proba(X)

        print("  ✓ LightGBM GPU: OK")
        return True

    except Exception as e:
        print(f"  ⚠ LightGBM GPU: FAILED - {e}")
        print("    (LightGBM GPU is optional, will fall back to CPU)")
        return False


def test_catboost_gpu() -> bool:
    """Test CatBoost GPU support."""
    try:
        import catboost as cb
        print(f"CatBoost version: {cb.__version__}")

        # Test GPU training
        model = cb.CatBoostClassifier(
            iterations=10,
            task_type='GPU',
            devices='0',
            verbose=False
        )

        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        pred = model.predict_proba(X)

        print("  ✓ CatBoost GPU: OK")
        return True

    except Exception as e:
        print(f"  ✗ CatBoost GPU: FAILED - {e}")
        return False


def check_nvidia_gpu():
    """Check NVIDIA GPU availability."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("\nGPU Information:")
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 3:
                    print(f"  GPU: {parts[0]}")
                    print(f"  Driver: {parts[1]}")
                    print(f"  Memory: {parts[2]}")
            return True
        else:
            print("\n⚠ nvidia-smi failed")
            return False

    except Exception as e:
        print(f"\n⚠ Could not check GPU: {e}")
        return False


def main():
    """Run all GPU tests."""
    print("=" * 60)
    print("Champion V13 GPU Support Test")
    print("=" * 60)
    print()

    # Check NVIDIA GPU
    has_nvidia = check_nvidia_gpu()
    print()

    # Test each framework
    print("Testing ML Frameworks:")
    print("-" * 60)

    results: Dict[str, bool] = {
        'xgboost': test_xgboost_gpu(),
        'lightgbm': test_lightgbm_gpu(),
        'catboost': test_catboost_gpu(),
    }

    print()
    print("=" * 60)
    print("Summary:")
    print("-" * 60)

    all_passed = all(results.values())
    required_passed = results['xgboost'] and results['catboost']

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:12s}: {status}")

    print("-" * 60)

    if required_passed:
        print("\n✓ All required frameworks (XGBoost + CatBoost) ready!")
        print("  Champion V13 can run with GPU acceleration.")
        return_code = 0
    else:
        print("\n✗ Missing required GPU support!")
        print("  Please check:")
        if not results['xgboost']:
            print("    - XGBoost CUDA installation")
        if not results['catboost']:
            print("    - CatBoost GPU installation")
        print("    - NVIDIA driver and CUDA toolkit")
        return_code = 1

    print("=" * 60)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
