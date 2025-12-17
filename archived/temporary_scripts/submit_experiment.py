#!/usr/bin/env python3
"""
統一實驗提交入口
=====================

用法:
    python scripts/submit_experiment.py --exp-name exp_001 --message "描述"

功能:
- 驗證提交檔案格式
- 使用 Kaggle API 提交
- 等待結果（可選）
- 保存結果到實驗目錄
- 更新實驗日誌
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


def validate_submission(submission_path: Path) -> bool:
    """
    驗證提交檔案格式

    規則:
    1. 必須包含 ID, dataset, label_positive_probability, junction_aa, v_call, j_call 欄位
    2. 總行數必須是 404,213
    3. 不能有 NaN (應該用 -999.0)
    """
    print(f"驗證提交檔案: {submission_path}")

    if not submission_path.exists():
        print(f"錯誤: 檔案不存在 {submission_path}")
        return False

    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"錯誤: 無法讀取 CSV 檔案: {e}")
        return False

    # 檢查欄位
    required_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"錯誤: 缺少必要欄位: {missing_cols}")
        return False

    # 檢查行數
    expected_rows = 404213
    if len(df) != expected_rows:
        print(f"警告: 行數不符 (預期: {expected_rows}, 實際: {len(df)})")
        # 不直接返回 False，因為測試時可能用部分數據

    # 檢查 NaN
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"錯誤: 發現 NaN 值:")
        print(null_counts[null_counts > 0])
        return False

    # 檢查 ID 格式
    # Task A: rep_XXX
    # Task B: train_dataset_X_seq_top_Y
    task_a_rows = df[df['ID'].str.startswith('rep_')]
    task_b_rows = df[df['ID'].str.contains('_seq_top_')]

    print(f"Task A 預測數: {len(task_a_rows)}")
    print(f"Task B 序列數: {len(task_b_rows)}")

    # Task A 的 probability 應該在 [0, 1]
    invalid_prob = task_a_rows[
        (task_a_rows['label_positive_probability'] < 0) |
        (task_a_rows['label_positive_probability'] > 1)
    ]
    if len(invalid_prob) > 0:
        print(f"警告: {len(invalid_prob)} 個預測概率超出 [0, 1] 範圍")

    # Task B 的 probability 應該是 -999.0
    invalid_b = task_b_rows[task_b_rows['label_positive_probability'] != -999.0]
    if len(invalid_b) > 0:
        print(f"警告: {len(invalid_b)} 個 Task B 的 probability 不是 -999.0")

    print("驗證通過!")
    return True


def submit_to_kaggle(submission_path: Path, message: str, competition: str) -> bool:
    """
    使用 Kaggle API 提交
    """
    print(f"\n開始提交到 Kaggle...")
    print(f"  競賽: {competition}")
    print(f"  檔案: {submission_path}")
    print(f"  訊息: {message}")

    cmd = [
        'kaggle', 'competitions', 'submit',
        '-c', competition,
        '-f', str(submission_path),
        '-m', message
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"提交失敗: {e}")
        print(e.stderr)
        return False


def get_submission_status(competition: str, max_wait: int = 300) -> dict:
    """
    查詢最新提交的狀態

    Args:
        competition: 競賽名稱
        max_wait: 最長等待時間（秒）

    Returns:
        dict: 包含 status, publicScore 等資訊
    """
    print(f"\n查詢提交狀態 (最多等待 {max_wait} 秒)...")

    start_time = time.time()
    last_status = None

    while time.time() - start_time < max_wait:
        cmd = ['kaggle', 'competitions', 'submissions', '-c', competition, '-v']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')

            if len(lines) < 2:
                print("無法解析提交列表")
                return {}

            # 解析最新提交（第二行）
            header = lines[0].split()
            latest = lines[1].split()

            # 找到 status 和 publicScore 的索引
            status_idx = header.index('status') if 'status' in header else -1
            score_idx = header.index('publicScore') if 'publicScore' in header else -1

            if status_idx == -1:
                print("無法找到 status 欄位")
                return {}

            status = latest[status_idx] if status_idx < len(latest) else 'unknown'

            if status != last_status:
                print(f"  狀態: {status}")
                last_status = status

            # 如果完成了
            if status == 'complete':
                result_dict = {'status': status}

                if score_idx != -1 and score_idx < len(latest):
                    score = latest[score_idx]
                    if score != '':
                        result_dict['publicScore'] = float(score)
                        print(f"  Public Score: {score}")

                return result_dict

            # 等待 10 秒再查詢
            time.sleep(10)

        except subprocess.CalledProcessError as e:
            print(f"查詢失敗: {e}")
            return {}

    print("等待超時")
    return {'status': 'timeout'}


def save_result(exp_dir: Path, result: dict):
    """保存提交結果到實驗目錄"""
    result_path = exp_dir / "kaggle_result.json"

    # 如果已存在，追加到列表
    if result_path.exists():
        with open(result_path, 'r') as f:
            results = json.load(f)
    else:
        results = []

    # 添加時間戳
    result['timestamp'] = datetime.now().isoformat()
    results.append(result)

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n結果已保存到: {result_path}")


def update_experiment_log(exp_name: str, result: dict, message: str):
    """更新實驗日誌"""
    log_path = Path("docs/experiment_log.md")

    # 創建日誌條目
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    score = result.get('publicScore', 'N/A')
    status = result.get('status', 'unknown')

    entry = f"\n### {exp_name} - {timestamp}\n"
    entry += f"- **描述**: {message}\n"
    entry += f"- **狀態**: {status}\n"
    entry += f"- **Public Score**: {score}\n"
    entry += f"- **詳情**: `experiments/{exp_name}/kaggle_result.json`\n"

    # 追加到日誌
    with open(log_path, 'a') as f:
        f.write(entry)

    print(f"實驗日誌已更新: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="統一實驗提交入口")
    parser.add_argument('--exp-name', type=str, required=True, help="實驗名稱")
    parser.add_argument('--message', type=str, required=True, help="提交描述")
    parser.add_argument('--base-dir', type=str, default='experiments', help="實驗基礎目錄")
    parser.add_argument('--competition', type=str,
                       default='adaptive-immune-profiling-challenge-2025',
                       help="Kaggle 競賽名稱")
    parser.add_argument('--no-wait', action='store_true', help="不等待結果")
    parser.add_argument('--submission-file', type=str, default='submission.csv',
                       help="提交檔案名稱")

    args = parser.parse_args()

    # 1. 定位實驗目錄
    exp_dir = Path(args.base_dir) / args.exp_name
    if not exp_dir.exists():
        print(f"錯誤: 實驗目錄不存在 {exp_dir}")
        sys.exit(1)

    # 2. 定位提交檔案
    submission_path = exp_dir / "outputs" / args.submission_file
    if not submission_path.exists():
        # 嘗試根目錄
        submission_path = exp_dir / args.submission_file

    if not submission_path.exists():
        print(f"錯誤: 找不到提交檔案")
        print(f"  嘗試過: {exp_dir / 'outputs' / args.submission_file}")
        print(f"  嘗試過: {exp_dir / args.submission_file}")
        sys.exit(1)

    # 3. 驗證提交檔案
    if not validate_submission(submission_path):
        print("\n驗證失敗，終止提交")
        sys.exit(1)

    # 4. 提交到 Kaggle
    if not submit_to_kaggle(submission_path, args.message, args.competition):
        print("\n提交失敗")
        sys.exit(1)

    # 5. 等待結果（可選）
    result = {}
    if not args.no_wait:
        result = get_submission_status(args.competition)
    else:
        result = {'status': 'submitted', 'note': 'not waiting for result'}

    # 6. 保存結果
    if result:
        save_result(exp_dir, result)
        update_experiment_log(args.exp_name, result, args.message)

    print("\n完成!")


if __name__ == "__main__":
    main()
