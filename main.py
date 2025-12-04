"""
main.py
============

這個程式提供了一個簡化的 baseline 實作，示範如何依照競賽要求讀取訓練與測試資料、提取簡單的 k‑mer 特徵、使用 L1 正則化的邏輯迴歸模型進行訓練，並輸出預測結果與重要序列列表。

程式假定每個訓練資料夾內有一個 `metadata.csv` 檔，包含 `repertoire_id`、`filename`、`label_positive` 欄位；訓練與測試資料檔為 TSV 格式，具備 `junction_aa`、`v_call` 與 `j_call` 欄位。您可以依照官方資料集格式調整此程式。

此 baseline 主要用於教學，並未針對比賽最佳化。建議參賽者在此基礎上加入更複雜的特徵工程與模型，例如 Transformer‑based 表示、XGBoost 等。
"""

import argparse
import os
import pandas as pd
import numpy as np
from collections import Counter
from itertools import islice
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def load_metadata(train_dir: str) -> pd.DataFrame:
    """讀取訓練目錄中的 metadata.csv。

    Args:
        train_dir: 訓練資料目錄路徑。
    Returns:
        pandas DataFrame，含有 repertoire_id, filename, label_positive 欄位。
    """
    metadata_path = os.path.join(train_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {train_dir}")
    return pd.read_csv(metadata_path)


def load_repertoire(filepath: str) -> pd.DataFrame:
    """讀取單一 TSV 檔，回傳 DataFrame。
    """
    return pd.read_csv(filepath, sep='\t')


def extract_kmer_counts(sequence: str, k: int = 3) -> Counter:
    """將蛋白質序列切割為 k‑mer 並計算頻率。
    Args:
        sequence: junction_aa 字串。
        k: k‑mer 長度。
    Returns:
        collections.Counter，鍵為 k‑mer，值為出現次數。
    """
    return Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))


def build_feature_matrix(file_list: List[Tuple[str, bool]], k: int = 3) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[str, int]]]:
    """為給定的 (filepath, label) 列表建立 k‑mer 特徵矩陣。

    Args:
        file_list: 每一項為 (filepath, label_positive)。
        k: k‑mer 長度。
    Returns:
        X: 特徵 DataFrame；列為每個 repertoire，欄為 k‑mer，值為該 k‑mer 出現次數的總和。
        y: 目標 Series。
        seq_counts: 每個 repertoire 對應到的序列頻率 Counter，用於之後的重要序列分析。
    """
    all_kmers: Counter = Counter()
    repertoire_features = []
    labels = []
    seq_counts_per_rep: Dict[str, Dict[str, int]] = {}
    for filepath, label in file_list:
        df = load_repertoire(filepath)
        # 聚合 repertoire 中所有 junction_aa 的 k‑mer
        repertoire_counter: Counter = Counter()
        for seq in df['junction_aa'].astype(str).fillna(''):
            repertoire_counter.update(extract_kmer_counts(seq, k=k))
        all_kmers.update(repertoire_counter.keys())
        seq_counts_per_rep[filepath] = repertoire_counter
        repertoire_features.append(repertoire_counter)
        labels.append(int(label))
    # 建立矩陣
    kmers_sorted = sorted(all_kmers)
    data = np.zeros((len(repertoire_features), len(kmers_sorted)), dtype=float)
    for i, counter in enumerate(repertoire_features):
        for j, kmer in enumerate(kmers_sorted):
            data[i, j] = counter.get(kmer, 0)
    X = pd.DataFrame(data, columns=kmers_sorted)
    y = pd.Series(labels)
    return X, y, seq_counts_per_rep


def train_model(X: pd.DataFrame, y: pd.Series, n_jobs: int) -> Pipeline:
    """使用 L1 正則化邏輯迴歸進行模型訓練並選擇最佳 C。

    Args:
        X: 特徵矩陣。
        y: 標籤向量。
        n_jobs: 並行數目（傳遞給 GridSearchCV）。
    Returns:
        已擬合的 sklearn Pipeline。
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))
    ])
    param_grid = {'clf__C': [1.0, 0.5, 0.1, 0.05]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=n_jobs)
    grid.fit(X, y)
    return grid.best_estimator_


def predict_proba(model: Pipeline, seq_counts_per_rep: Dict[str, Dict[str, int]], kmers: List[str]) -> Dict[str, float]:
    """為每個 repertoire 預測為陽性的機率。

    Args:
        model: 已訓練的模型。
        seq_counts_per_rep: 每個 repertoire 的 k‑mer 計數。
        kmers: 特徵列表。
    Returns:
        字典：repertoire 路徑 -> 預測機率。
    """
    data = []
    keys = []
    for path, counter in seq_counts_per_rep.items():
        row = [counter.get(k, 0) for k in kmers]
        data.append(row)
        keys.append(path)
    X = pd.DataFrame(data, columns=kmers)
    probs = model.predict_proba(X)[:, 1]
    return dict(zip(keys, probs))


def rank_important_sequences(train_counts: Dict[str, Dict[str, int]], labels: pd.Series, n_top: int = 50000) -> pd.DataFrame:
    """簡單地依照序列在陽性樣本中的出現次數減去陰性樣本中的出現次數排序。

    Args:
        train_counts: 訓練資料各 repertoire 的序列計數映射；每個 repertoire 對應到 Counter(junction_aa->次數)。
        labels: 訓練資料標籤向量，順序與 train_counts 對應。
        n_top: 選取的前幾個序列。
    Returns:
        DataFrame，含有 `junction_aa`、`importance_score` 欄位。
    """
    # 建立總表：序列 -> 正樣本次數與負樣本次數
    pos_counts: Counter = Counter()
    neg_counts: Counter = Counter()
    for (path, counter), label in zip(train_counts.items(), labels):
        if label:
            pos_counts.update(counter)
        else:
            neg_counts.update(counter)
    importance = {}
    for seq in set(list(pos_counts.keys()) + list(neg_counts.keys())):
        importance_score = pos_counts.get(seq, 0) - neg_counts.get(seq, 0)
        importance[seq] = importance_score
    # 取絕對值排序
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n_top]
    return pd.DataFrame(ranked, columns=['junction_aa', 'importance_score'])


def save_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def process_dataset(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int) -> None:
    # 讀取 metadata 並組合訓練檔案與標籤
    metadata = load_metadata(train_dir)
    train_file_list = []
    for _, row in metadata.iterrows():
        file_path = os.path.join(train_dir, row['filename'])
        train_file_list.append((file_path, bool(row['label_positive'])))
    X_train, y_train, train_counts = build_feature_matrix(train_file_list, k=3)
    model = train_model(X_train, y_train, n_jobs=n_jobs)
    # 預測測試資料
    for test_dir in test_dirs:
        # 讀取所有 TSV
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.tsv')])
        test_file_list = []
        for f_name in test_files:
            file_path = os.path.join(test_dir, f_name)
            test_file_list.append((file_path, None))
        # 建立特徵
        kmers = list(X_train.columns)
        # 對每個 test repertoire 取得 k‑mer counter
        test_counts = {}
        for file_path, _ in test_file_list:
            df = load_repertoire(file_path)
            rep_counter: Counter = Counter()
            for seq in df['junction_aa'].astype(str).fillna(''):
                rep_counter.update(extract_kmer_counts(seq, k=3))
            test_counts[file_path] = rep_counter
        probs = predict_proba(model, test_counts, kmers)
        # 儲存預測
        pred_rows = []
        for path, prob in probs.items():
            rep_id = os.path.basename(path)
            pred_rows.append({'ID': rep_id, 'dataset': os.path.basename(test_dir), 'label_positive_probability': prob})
        pred_df = pd.DataFrame(pred_rows)
        out_pred_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
        save_tsv(pred_df, out_pred_path)
    # 儲存重要序列
    imp_df = rank_important_sequences(train_counts, y_train)
    imp_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(imp_df, imp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline trainer for AIRR‑ML‑25 challenge.")
    parser.add_argument('--train_dir', type=str, required=True, help='訓練資料目錄 (包含 metadata.csv)')
    parser.add_argument('--test_dirs', type=str, required=True, nargs='+', help='一或多個測試資料目錄')
    parser.add_argument('--out_dir', type=str, required=True, help='輸出結果目錄')
    parser.add_argument('--n_jobs', type=int, default=1, help='並行使用的 CPU 核心數')
    parser.add_argument('--device', type=str, default='cpu', help='運算裝置 (未使用)')
    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs)


if __name__ == '__main__':
    main()