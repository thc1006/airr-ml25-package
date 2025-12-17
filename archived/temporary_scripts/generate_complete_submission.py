#!/usr/bin/env python3
"""
完整 Submission 生成器 - 補全 Dataset 8 所有缺失的預測
"""
import pandas as pd
from pathlib import Path

results_dir = Path("./results_k4")
out_path = Path("./submission.csv")

# 載入已完成的 7 個 datasets 的結果
predictions = []
sequences = []

for i in range(1, 8):  # Datasets 1-7
    pred_file = results_dir / f"train_dataset_{i}_test_predictions.tsv"
    seq_file = results_dir / f"train_dataset_{i}_important_sequences.tsv"

    if pred_file.exists():
        df = pd.read_csv(pred_file, sep='\t')
        predictions.append(df)
        print(f"✅ Dataset {i} predictions: {len(df)} rows")

    if seq_file.exists():
        df = pd.read_csv(seq_file, sep='\t')
        sequences.append(df)
        print(f"✅ Dataset {i} sequences: {len(df)} rows")

# 合併現有結果
all_preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
all_seqs = pd.concat(sequences, ignore_index=True) if sequences else pd.DataFrame()

print(f"\n現有預測總數: {len(all_preds)}")
print(f"現有序列總數: {len(all_seqs)}")

# Dataset 8: 從 sample_submissions.csv 取得所有 test_dataset_8 的模板
sample = pd.read_csv("./data/sample_submissions.csv")

# 取得所有 Dataset 8 的預測（test_dataset_8_1, 8_2, 8_3）
dataset8_preds = sample[
    (sample['dataset'].str.contains('test_dataset_8')) &
    (sample['label_positive_probability'] != -999.0)
].copy()

# 使用 baseline 0.5 作為預測值
dataset8_preds['label_positive_probability'] = 0.5

# 取得 Dataset 8 的序列（train_dataset_8）
dataset8_seqs = sample[
    (sample['dataset'] == 'train_dataset_8') &
    (sample['junction_aa'] != -999.0)
].copy()

print(f"\nDataset 8 補完:")
print(f"  預測行數: {len(dataset8_preds)}")
print(f"    test_dataset_8_1: {len(dataset8_preds[dataset8_preds['dataset'] == 'test_dataset_8_1'])}")
print(f"    test_dataset_8_2: {len(dataset8_preds[dataset8_preds['dataset'] == 'test_dataset_8_2'])}")
print(f"    test_dataset_8_3: {len(dataset8_preds[dataset8_preds['dataset'] == 'test_dataset_8_3'])}")
print(f"  序列行數: {len(dataset8_seqs)}")

# 最終合併
final_df = pd.concat([all_preds, dataset8_preds, all_seqs, dataset8_seqs], ignore_index=True)

# 確保欄位順序和格式正確
final_df = final_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

# 將 -999.0 轉換為字串格式（Kaggle 可能需要）
final_df = final_df.astype(str)

# 保存
final_df.to_csv(out_path, index=False)

print(f"\n✅ 完整 Submission 已保存: {out_path}")
print(f"   總行數: {len(final_df)}")
print(f"   預期行數: 404,213")
print(f"   差異: {len(final_df) - 404213}")

# 驗證格式
print("\n=== 格式驗證 ===")
print(f"欄位: {list(final_df.columns)}")
print(f"預測行數 (label_positive_probability != '-999.0'): {len(final_df[final_df['label_positive_probability'] != '-999.0'])}")
print(f"序列行數 (junction_aa != '-999.0'): {len(final_df[final_df['junction_aa'] != '-999.0'])}")
