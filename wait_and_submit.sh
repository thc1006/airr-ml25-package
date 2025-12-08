#!/bin/bash
# 等待 Dataset 8 完成後自動生成完整 submission

echo "=== 等待 Dataset 8 訓練完成 ==="
while true; do
    if [ -f ./results_k4/train_dataset_8_test_predictions.tsv ] && [ -f ./results_k4/train_dataset_8_important_sequences.tsv ]; then
        echo "✅ Dataset 8 訓練完成！"
        break
    fi
    echo "$(date +%H:%M:%S) - 等待中... (檢查檔案: train_dataset_8_*.tsv)"
    sleep 15
done

echo ""
echo "=== 生成完整 submission ==="
python3 <<'EOF'
import pandas as pd
from pathlib import Path

results_dir = Path("./results_k4")
out_path = Path("./submission_complete.csv")

# 載入所有 8 個 datasets 的結果
predictions = []
sequences = []

for i in range(1, 9):  # Datasets 1-8
    pred_file = results_dir / f"train_dataset_{i}_test_predictions.tsv"
    seq_file = results_dir / f"train_dataset_{i}_important_sequences.tsv"

    if pred_file.exists():
        df = pd.read_csv(pred_file, sep='\t')
        predictions.append(df)
        print(f"✅ Dataset {i} predictions: {len(df)} rows")
    else:
        print(f"❌ Dataset {i} predictions: MISSING")

    if seq_file.exists():
        df = pd.read_csv(seq_file, sep='\t')
        sequences.append(df)
        print(f"✅ Dataset {i} sequences: {len(df)} rows")
    else:
        print(f"❌ Dataset {i} sequences: MISSING")

# 合併
all_preds = pd.concat(predictions, ignore_index=True)
all_seqs = pd.concat(sequences, ignore_index=True)

print(f"\n總預測數: {len(all_preds)}")
print(f"總序列數: {len(all_seqs)}")

# 最終合併
final_df = pd.concat([all_preds, all_seqs], ignore_index=True)

# 確保欄位順序正確
final_df = final_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

# 保存
final_df.to_csv(out_path, index=False)

print(f"\n✅ 完整 Submission 已保存: {out_path}")
print(f"   總行數: {len(final_df):,}")
print(f"   預期行數: 404,213")
print(f"   差異: {len(final_df) - 404213:+,}")
EOF

echo ""
echo "=== 驗證 submission 格式 ==="
python3 -c "
import pandas as pd
df = pd.read_csv('submission_complete.csv')
print(f'Total rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Null values: {df.isnull().sum().sum()}')
print(f'Duplicate IDs: {df[\"ID\"].duplicated().sum()}')
"

echo ""
echo "✅ 全部完成！可以提交 submission_complete.csv"
