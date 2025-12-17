#!/usr/bin/env python3
"""
生成完整 Submission - 使用 results_k4/ 的所有正確結果
包含正確訓練的 Dataset 8 (使用 LogReg 係數方法)
"""
import pandas as pd
from pathlib import Path

results_dir = Path("./results_k4")
out_path = Path("./submission_corrected.csv")

print("=== 生成完整 Submission (使用正確的 LogReg 方法) ===\n")

# 載入所有 8 個 datasets 的結果
predictions = []
sequences = []

for i in range(1, 9):  # Datasets 1-8 (包含 Dataset 8!)
    pred_file = results_dir / f"train_dataset_{i}_test_predictions.tsv"
    seq_file = results_dir / f"train_dataset_{i}_important_sequences.tsv"

    if pred_file.exists():
        df = pd.read_csv(pred_file, sep='\t')
        predictions.append(df)
        print(f"✅ Dataset {i} predictions: {len(df)} rows")
    else:
        print(f"❌ Dataset {i} predictions: NOT FOUND")

    if seq_file.exists():
        df = pd.read_csv(seq_file, sep='\t')
        sequences.append(df)
        print(f"✅ Dataset {i} sequences: {len(df)} rows")
    else:
        print(f"❌ Dataset {i} sequences: NOT FOUND")

# 合併所有結果
if not predictions or not sequences:
    raise ValueError("Missing predictions or sequences files!")

all_preds = pd.concat(predictions, ignore_index=True)
all_seqs = pd.concat(sequences, ignore_index=True)

print(f"\n=== 統計資訊 ===")
print(f"預測總數: {len(all_preds)}")
print(f"序列總數: {len(all_seqs)}")

# 檢查 Dataset 8
dataset8_preds = all_preds[all_preds['dataset'].str.contains('test_dataset_8')]
dataset8_seqs = all_seqs[all_seqs['dataset'] == 'train_dataset_8']

print(f"\nDataset 8 詳情:")
print(f"  預測行數: {len(dataset8_preds)}")
for ds in dataset8_preds['dataset'].unique():
    count = len(dataset8_preds[dataset8_preds['dataset'] == ds])
    print(f"    {ds}: {count} rows")
print(f"  序列行數: {len(dataset8_seqs)}")

# 合併最終結果
final_df = pd.concat([all_preds, all_seqs], ignore_index=True)

# 確保欄位順序正確
final_df = final_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

# 驗證格式
print(f"\n=== 格式驗證 ===")
print(f"總行數: {len(final_df)}")
print(f"預期行數: 404,213")
print(f"差異: {len(final_df) - 404213}")

pred_rows = len(final_df[final_df['label_positive_probability'] != -999.0])
seq_rows = len(final_df[final_df['junction_aa'] != -999.0])
print(f"\n預測行數: {pred_rows} (應為 4,213)")
print(f"序列行數: {seq_rows} (應為 400,000)")

# 檢查是否有缺失值 (除了 -999.0 placeholder)
print(f"\n=== 數據質量檢查 ===")
print(f"ID 缺失: {final_df['ID'].isna().sum()}")
print(f"dataset 缺失: {final_df['dataset'].isna().sum()}")
print(f"label_positive_probability 缺失: {final_df['label_positive_probability'].isna().sum()}")

# Dataset 8 預測值分佈
ds8_pred_values = final_df[
    final_df['dataset'].str.contains('test_dataset_8') &
    (final_df['label_positive_probability'] != -999.0)
]['label_positive_probability']

if len(ds8_pred_values) > 0:
    print(f"\nDataset 8 預測值分佈:")
    print(f"  Mean: {ds8_pred_values.mean():.4f}")
    print(f"  Median: {ds8_pred_values.median():.4f}")
    print(f"  Min: {ds8_pred_values.min():.4f}")
    print(f"  Max: {ds8_pred_values.max():.4f}")
    print(f"  Std: {ds8_pred_values.std():.4f}")

# 保存
final_df.to_csv(out_path, index=False)

print(f"\n✅ 完整 Submission 已保存: {out_path}")
print(f"   使用正確的 LogReg 係數方法")
print(f"   包含正確訓練的 Dataset 8 結果")
print(f"\n下一步: 提交到 Kaggle 並檢查分數是否提升！")
