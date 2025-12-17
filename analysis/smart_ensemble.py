#!/usr/bin/env python3
"""
AIRR-ML-25 Smart Ensemble Strategy
====================================
基於深度分析創建的智慧 ensemble 策略。

分析發現：
1. V5 (0.74006) 是最佳版本
2. Dataset 8 翻轉率最高 (19.7%)，是關鍵優化點
3. Dataset 5,6 非常穩定，不需要改變
4. V8 (0.73029) 與 V5 相關性 0.89，有一定差異性

策略：
- 保守 ensemble：以 V5 為主，少量混入 V8
- 按 dataset 加權：不同 dataset 用不同權重
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_submissions():
    """載入各版本的提交文件"""
    submissions = {
        'v5': pd.read_csv('submissions/v5_fixed_genes.csv'),  # 0.74006 - 最佳
        'v8': pd.read_csv('submissions/v8_submission_fixed.csv'),  # 0.73029
        # 'v7': pd.read_csv('submissions/v7_using_v5_format.csv'),  # 0.69294 - 太差不用
    }
    return submissions

def extract_task_a(df):
    """提取 Task A 預測"""
    return df[df['label_positive_probability'] != -999.0].copy()

def extract_task_b(df):
    """提取 Task B 序列"""
    return df[df['label_positive_probability'] == -999.0].copy()

def global_ensemble(v5, v8, weight_v5=0.9):
    """全局加權 ensemble"""
    weight_v8 = 1 - weight_v5

    v5_a = extract_task_a(v5)
    v8_a = extract_task_a(v8)

    # 合併
    merged = v5_a[['ID', 'dataset', 'label_positive_probability']].copy()
    merged = merged.rename(columns={'label_positive_probability': 'v5_prob'})
    merged = merged.merge(
        v8_a[['ID', 'label_positive_probability']].rename(columns={'label_positive_probability': 'v8_prob'}),
        on='ID'
    )

    # 加權平均
    merged['label_positive_probability'] = merged['v5_prob'] * weight_v5 + merged['v8_prob'] * weight_v8

    # 準備輸出
    result = merged[['ID', 'dataset', 'label_positive_probability']].copy()
    result['junction_aa'] = '-999.0'
    result['v_call'] = '-999.0'
    result['j_call'] = '-999.0'

    return result

def dataset_weighted_ensemble(v5, v8):
    """
    按 dataset 加權的 ensemble

    基於分析結果：
    - Dataset 5,6: 非常穩定 (corr > 0.99)，純用 V5
    - Dataset 7: 最不穩定 (corr 0.35)，保守混合
    - Dataset 8: 翻轉最多 (19.7%)，關鍵優化點
    - Dataset 1-4: 中等穩定，輕微混合
    """
    # 權重設定 (V5 權重)
    dataset_weights = {
        1: 0.90,  # 中等穩定
        2: 0.90,
        3: 0.90,
        4: 0.90,
        5: 1.00,  # 非常穩定，純 V5
        6: 1.00,  # 非常穩定，純 V5
        7: 0.85,  # 最不穩定，需要修正
        8: 0.85,  # 翻轉最多，關鍵點
    }

    v5_a = extract_task_a(v5)
    v8_a = extract_task_a(v8)

    # 合併
    merged = v5_a[['ID', 'dataset', 'label_positive_probability']].copy()
    merged = merged.rename(columns={'label_positive_probability': 'v5_prob'})
    merged = merged.merge(
        v8_a[['ID', 'label_positive_probability']].rename(columns={'label_positive_probability': 'v8_prob'}),
        on='ID'
    )

    # 提取 dataset ID
    merged['dataset_id'] = merged['dataset'].str.extract(r'(\d+)').astype(int)

    # 按 dataset 加權
    def apply_weight(row):
        w_v5 = dataset_weights.get(row['dataset_id'], 0.9)
        w_v8 = 1 - w_v5
        return row['v5_prob'] * w_v5 + row['v8_prob'] * w_v8

    merged['label_positive_probability'] = merged.apply(apply_weight, axis=1)

    # 準備輸出
    result = merged[['ID', 'dataset', 'label_positive_probability']].copy()
    result['junction_aa'] = '-999.0'
    result['v_call'] = '-999.0'
    result['j_call'] = '-999.0'

    return result

def create_ensemble_submission(task_a_result, task_b_source, output_name):
    """創建完整的提交文件"""
    task_b = extract_task_b(task_b_source)

    submission = pd.concat([task_a_result, task_b], ignore_index=True)
    submission = submission[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

    # 驗證
    expected_rows = 404213
    if len(submission) != expected_rows:
        print(f"WARNING: Expected {expected_rows} rows, got {len(submission)}")

    # 儲存
    output_path = Path('submissions') / output_name
    submission.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return submission

def main():
    print("=" * 70)
    print("AIRR-ML-25 Smart Ensemble")
    print("=" * 70)

    # 載入提交
    subs = load_submissions()
    v5 = subs['v5']
    v8 = subs['v8']

    print("\n載入的提交:")
    print(f"  V5: {len(v5)} rows (LB: 0.74006)")
    print(f"  V8: {len(v8)} rows (LB: 0.73029)")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 策略 1: 全局 ensemble (90% V5 + 10% V8)
    print("\n=== 策略 1: 全局 Ensemble (90% V5 + 10% V8) ===")
    task_a_global = global_ensemble(v5, v8, weight_v5=0.9)
    sub_global = create_ensemble_submission(
        task_a_global,
        v5,  # 使用 V5 的 Task B
        f'ensemble_global_90_10_{timestamp}.csv'
    )

    # 策略 2: 全局 ensemble (85% V5 + 15% V8)
    print("\n=== 策略 2: 全局 Ensemble (85% V5 + 15% V8) ===")
    task_a_global_85 = global_ensemble(v5, v8, weight_v5=0.85)
    sub_global_85 = create_ensemble_submission(
        task_a_global_85,
        v5,
        f'ensemble_global_85_15_{timestamp}.csv'
    )

    # 策略 3: 按 Dataset 加權
    print("\n=== 策略 3: Dataset 加權 Ensemble ===")
    task_a_ds = dataset_weighted_ensemble(v5, v8)
    sub_ds = create_ensemble_submission(
        task_a_ds,
        v5,
        f'ensemble_dataset_weighted_{timestamp}.csv'
    )

    # 分析差異
    print("\n=== 預測差異分析 ===")
    v5_a = extract_task_a(v5)

    for name, task_a in [('global_90_10', task_a_global), ('global_85_15', task_a_global_85), ('dataset_weighted', task_a_ds)]:
        merged = v5_a[['ID', 'label_positive_probability']].merge(
            task_a[['ID', 'label_positive_probability']],
            on='ID',
            suffixes=('_v5', '_ensemble')
        )
        diff = (merged['label_positive_probability_v5'] - merged['label_positive_probability_ensemble']).abs()
        print(f"\n{name}:")
        print(f"  平均差異: {diff.mean():.6f}")
        print(f"  最大差異: {diff.max():.6f}")
        print(f"  差異 > 0.05 的樣本: {(diff > 0.05).sum()}")

    print("\n" + "=" * 70)
    print("完成！請選擇一個 ensemble 提交。")
    print("建議：先提交 global_90_10 (最保守)")
    print("=" * 70)

if __name__ == '__main__':
    main()
