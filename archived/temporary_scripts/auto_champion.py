#!/usr/bin/env python3
"""
AIRR-ML-25 Championship Auto-Pilot
===================================
全自動監控、驗證、提交腳本
目標: 奪冠 (>0.81364)
"""

import subprocess
import time
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_DIR = Path("/home/thc1006/dev/airr-ml25-package")
LOG_FILE = PROJECT_DIR / "logs" / "auto_champion.log"
SUBMISSION_PATH = PROJECT_DIR / "submissions_fast.csv"
PIPELINE_LOG = PROJECT_DIR / "championship_fast.log"
PIPELINE_PID_CMD = "pgrep -f championship_fast.py"

# Expected submission format
EXPECTED_ROWS = 404213
EXPECTED_COLS = ["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]

def log(msg: str):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def is_pipeline_running() -> bool:
    """Check if championship_fast.py is still running"""
    try:
        result = subprocess.run(PIPELINE_PID_CMD, shell=True, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

def get_pipeline_progress() -> str:
    """Get last few lines of pipeline log"""
    try:
        result = subprocess.run(f"tail -3 {PIPELINE_LOG}", shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Unable to read log"

def validate_submission(path: Path) -> tuple[bool, str]:
    """Validate submission file format"""
    if not path.exists():
        return False, f"File not found: {path}"

    try:
        df = pd.read_csv(path)

        # Check row count
        if len(df) != EXPECTED_ROWS:
            return False, f"Wrong row count: {len(df)} (expected {EXPECTED_ROWS})"

        # Check columns
        missing_cols = set(EXPECTED_COLS) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check for NaN (should use -999.0)
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            return False, f"Contains {nan_count} NaN values (should use -999.0)"

        # Check prediction rows
        pred_rows = df[df['junction_aa'] == -999.0]
        if len(pred_rows) != 4213:
            return False, f"Wrong prediction row count: {len(pred_rows)} (expected 4213)"

        # Check probability range
        probs = pred_rows['label_positive_probability']
        if probs.min() < 0 or probs.max() > 1:
            return False, f"Probabilities out of range: [{probs.min()}, {probs.max()}]"

        # Check sequence rows
        seq_rows = df[df['junction_aa'] != -999.0]
        expected_seq = 8 * 50000
        if len(seq_rows) != expected_seq:
            return False, f"Wrong sequence row count: {len(seq_rows)} (expected {expected_seq})"

        return True, f"Valid! {len(pred_rows)} predictions, {len(seq_rows)} sequences"

    except Exception as e:
        return False, f"Validation error: {e}"

def submit_to_kaggle(path: Path, message: str) -> tuple[bool, str]:
    """Submit to Kaggle competition"""
    cmd = f'kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 -f "{path}" -m "{message}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True, result.stdout + result.stderr
        else:
            return False, f"Submit failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Submit timeout (300s)"
    except Exception as e:
        return False, f"Submit error: {e}"

def check_submission_status() -> str:
    """Check latest Kaggle submission status"""
    try:
        result = subprocess.run(
            "kaggle competitions submissions -c adaptive-immune-profiling-challenge-2025 | head -5",
            shell=True, capture_output=True, text=True, timeout=60
        )
        return result.stdout
    except:
        return "Unable to check status"

def main():
    log("=" * 60)
    log("🏆 AIRR-ML-25 CHAMPIONSHIP AUTO-PILOT STARTED")
    log("=" * 60)
    log(f"Target: Beat GROZD (0.81364) → 0.82+")
    log(f"Monitoring: {PIPELINE_LOG}")
    log(f"Submission: {SUBMISSION_PATH}")
    log("")

    # Phase 1: Wait for pipeline to complete
    log("📡 PHASE 1: Monitoring pipeline...")
    check_interval = 30  # seconds
    last_progress = ""

    while is_pipeline_running():
        progress = get_pipeline_progress()
        if progress != last_progress:
            # Log only first line to avoid spam
            first_line = progress.split('\n')[0][:100]
            log(f"   Progress: {first_line}...")
            last_progress = progress
        time.sleep(check_interval)

    log("✅ Pipeline completed!")
    log("")

    # Wait a bit for file to be fully written
    time.sleep(5)

    # Phase 2: Validate submission
    log("🔍 PHASE 2: Validating submission...")

    # Check both possible paths
    possible_paths = [
        SUBMISSION_PATH,
        PROJECT_DIR / "submission_fast.csv",
        PROJECT_DIR / "submission.csv",
    ]

    valid_path = None
    for path in possible_paths:
        if path.exists():
            log(f"   Found: {path}")
            valid, msg = validate_submission(path)
            log(f"   Validation: {msg}")
            if valid:
                valid_path = path
                break

    if not valid_path:
        log("❌ No valid submission file found!")
        log("   Checking log for errors...")
        log(get_pipeline_progress())
        return 1

    log(f"✅ Valid submission: {valid_path}")
    log("")

    # Phase 3: Submit to Kaggle
    log("🚀 PHASE 3: Submitting to Kaggle...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    message = f"Auto-Champion ESM-2 MLP v1 ({timestamp})"

    success, result = submit_to_kaggle(valid_path, message)
    log(f"   Result: {result}")

    if success:
        log("✅ Submission successful!")
    else:
        log("⚠️ Submission may have issues, but file is ready")

    log("")

    # Phase 4: Check status
    log("📊 PHASE 4: Checking submission status...")
    time.sleep(30)  # Wait for Kaggle to process
    status = check_submission_status()
    log(f"   Status:\n{status}")

    log("")
    log("=" * 60)
    log("🏆 AUTO-PILOT COMPLETE")
    log("=" * 60)
    log(f"Submission file: {valid_path}")
    log(f"Log file: {LOG_FILE}")
    log("")
    log("Next steps if score < 0.82:")
    log("  1. Try attention-based aggregation instead of mean pooling")
    log("  2. Use multi-scale k-mer features")
    log("  3. Ensemble with XGBoost/LightGBM")
    log("")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"Fatal error: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
