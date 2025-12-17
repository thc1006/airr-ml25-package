#!/usr/bin/env python3
"""
快速生成submission - 使用現有7個完整結果 + 補完 Dataset 8
"""
import pandas as pd
from pathlib import Path

results_dir = Path("./results_k4")
out_path = Path("./quick_submission.csv")

# 載入已完成的 7 個 datasets
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

# 合併
all_preds = pd.concat(predictions, ignore_index=True)
all_seqs = pd.concat(sequences, ignore_index=True)

print(f"\nTotal predictions: {len(all_preds)}")
print(f"Total sequences: {len(all_seqs)}")

# Dataset 8: 需要補完 - 使用 sample_submissions.csv 中的 Dataset 8 資料作為模板
sample = pd.read_csv("./data/sample_submissions.csv")
dataset8_preds = sample[sample['dataset'] == 'test_dataset_8_3'].copy()
dataset8_preds['label_positive_probability'] = 0.5  # 使用baseline 0.5

dataset8_seqs = sample[sample['dataset'] == 'train_dataset_8'].head(50000).copy()

print(f"\n Dataset 8 (補完):")
print(f"  Predictions: {len(dataset8_preds)} rows")
print(f"  Sequences: {len(dataset8_seqs)} rows")

# 最終合併
final_df = pd.concat([all_preds, dataset8_preds, all_seqs, dataset8_seqs], ignore_index=True)

# 確保欄位順序正確
final_df = final_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

# 保存
final_df.to_csv(out_path, index=False)

print(f"\n✅ Submission saved: {out_path}")
print(f"   Total rows: {len(final_df)}")
print(f"   Expected: 404,213")
print(f"   Difference: {len(final_df) - 404213}")
