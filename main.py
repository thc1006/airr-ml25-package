#!/usr/bin/env python3
"""
AIRR-ML-25 Competition - Quick Start Entry Point
================================================

This is the main entry point for reproducing our competition results.

Quick Start:
    1. Install dependencies:
       pip install -r requirements.txt

    2. Download competition data from Kaggle:
       kaggle competitions download -c adaptive-immune-profiling-challenge-2025
       unzip to ./data/

    3. Run training and generate submission:
       python main.py

Expected Output:
    - Trained models saved to ./checkpoints_v8/
    - Submission file: ./submissions/v8_submission_<timestamp>.csv
    - Expected score: ~0.73 (public LB), 0.51242 (private LB)

Final Result:
    Private Leaderboard: 0.51242 (Rank #52/~500 teams)
    Public Leaderboard: 0.73029
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def check_environment():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'scipy',
        'xgboost', 'lightgbm', 'catboost', 'tqdm'
    ]

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"   Install with: pip install -r requirements.txt")
        sys.exit(1)

    print("✅ All required packages are installed")

def check_data():
    """Check if competition data exists."""
    data_dirs = [
        Path('./data/train_datasets/train_datasets'),
        Path('./data/test_datasets/test_datasets'),
        Path('./data/sample_submissions.csv')
    ]

    missing = []
    for path in data_dirs:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print(f"❌ Missing competition data:")
        for path in missing:
            print(f"   - {path}")
        print()
        print("Download data from Kaggle:")
        print("  kaggle competitions download -c adaptive-immune-profiling-challenge-2025")
        print("  unzip adaptive-immune-profiling-challenge-2025.zip -d ./data/")
        sys.exit(1)

    print("✅ Competition data found")

def main():
    """Main entry point."""
    print("=" * 70)
    print("AIRR-ML-25 Competition - Champion V8 Pipeline")
    print("=" * 70)
    print()

    # Check environment
    print("Step 1: Checking environment...")
    check_environment()
    print()

    # Auto-detect optimal hardware configuration
    print("Step 2: Detecting hardware configuration...")
    try:
        from src.utils_parallel import (
            get_optimal_n_jobs,
            get_optimal_device,
            print_system_info,
            configure_threading
        )

        # Print system info
        print_system_info()
        print()

        # Get optimal settings
        optimal_n_jobs = get_optimal_n_jobs()
        optimal_device = get_optimal_device()

        # Configure threading for numerical libraries
        configure_threading()

        print(f"✅ Using {optimal_n_jobs} parallel jobs on {optimal_device}")
        print()

    except Exception as e:
        print(f"⚠️  Could not auto-detect hardware: {e}")
        print("   Using default settings (n_jobs=4, device=cpu)")
        optimal_n_jobs = 4
        optimal_device = 'cpu'
        print()

    # Check data
    print("Step 3: Checking competition data...")
    check_data()
    print()

    # Import and run champion v8
    print("Step 4: Running Champion V8 pipeline...")
    print(f"Configuration: {optimal_n_jobs} jobs, {optimal_device} device")
    print("This will take several hours to complete.")
    print()

    try:
        from src.champion_v8 import main as champion_main
        champion_main()
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure you're in the project root directory")
        print("  2. Check that all dependencies are installed")
        print("  3. Verify data directory structure")
        print("  4. Try running: python -m src.champion_v8")
        sys.exit(1)

    print()
    print("=" * 70)
    print("✅ Pipeline completed successfully!")
    print("=" * 70)
    print()
    print("Check ./submissions/ for the generated submission file.")
    print()

if __name__ == '__main__':
    main()
