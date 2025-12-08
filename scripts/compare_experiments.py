#!/usr/bin/env python3
"""
實驗比較工具
==============

用法:
    python scripts/compare_experiments.py --exps exp_001 exp_002 exp_003

功能:
- 比較多個實驗的指標
- 生成對比表格
- 視覺化結果（可選）
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def load_experiment_results(exp_name: str, base_dir: str = "experiments") -> Dict[str, Any]:
    """載入實驗結果"""
    exp_dir = Path(base_dir) / exp_name

    if not exp_dir.exists():
        print(f"警告: 實驗目錄不存在 {exp_dir}")
        return {}

    result = {
        'name': exp_name,
        'exists': True,
    }

    # 載入配置
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            result['config'] = yaml.safe_load(f)

    # 載入指標
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            result['metrics'] = json.load(f)

    # 載入 Kaggle 結果
    kaggle_path = exp_dir / "kaggle_result.json"
    if kaggle_path.exists():
        with open(kaggle_path, 'r') as f:
            kaggle_results = json.load(f)
            if kaggle_results:
                # 取最新的結果
                latest = kaggle_results[-1]
                result['public_score'] = latest.get('publicScore')
                result['status'] = latest.get('status')
                result['timestamp'] = latest.get('timestamp')

    return result


def compare_metrics(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """比較實驗指標"""
    rows = []

    for exp in experiments:
        row = {
            'Experiment': exp['name'],
            'Status': exp.get('status', 'N/A'),
        }

        # Local CV AUC
        if 'metrics' in exp:
            metrics = exp['metrics']
            if isinstance(metrics, dict):
                # 計算平均 AUC
                aucs = []
                for key, value in metrics.items():
                    if 'auc' in key.lower() and isinstance(value, (int, float)):
                        aucs.append(value)

                if aucs:
                    row['Local CV (mean)'] = f"{sum(aucs)/len(aucs):.5f}"
                    row['Local CV (std)'] = f"{pd.Series(aucs).std():.5f}"

        # Public LB
        if 'public_score' in exp:
            row['Public LB'] = f"{exp['public_score']:.5f}"
        else:
            row['Public LB'] = 'N/A'

        # Timestamp
        if 'timestamp' in exp:
            row['Timestamp'] = exp['timestamp'][:19]  # 只取日期和時間
        else:
            row['Timestamp'] = 'N/A'

        rows.append(row)

    return pd.DataFrame(rows)


def compare_configs(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """比較實驗配置"""
    rows = []

    for exp in experiments:
        row = {'Experiment': exp['name']}

        if 'config' not in exp:
            rows.append(row)
            continue

        config = exp['config']

        # Features
        if 'features' in config:
            features = config['features']

            if 'kmer' in features:
                kmer = features['kmer']
                row['k-values'] = str(kmer.get('k_values', 'N/A'))
                row['TF-IDF'] = str(kmer.get('use_tfidf', False))

            if 'vj_gene' in features:
                row['V/J gene'] = 'Yes'
            else:
                row['V/J gene'] = 'No'

            if 'diversity' in features:
                row['Diversity'] = 'Yes'
            else:
                row['Diversity'] = 'No'

        # Model
        if 'model' in config:
            model = config['model']
            row['Model'] = model.get('type', 'N/A')

            # 超參數（簡化）
            model_config = model.get(model.get('type', ''), {})
            if 'n_estimators' in model_config:
                row['n_estimators'] = model_config['n_estimators']
            if 'max_depth' in model_config:
                row['max_depth'] = model_config['max_depth']
            if 'learning_rate' in model_config:
                row['learning_rate'] = model_config['learning_rate']

        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame, title: str):
    """美化列印表格"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def calculate_improvements(experiments: List[Dict[str, Any]], baseline_score: float):
    """計算相對於 baseline 的改進"""
    print("\n" + "=" * 80)
    print("相對改進分析 (vs Baseline)")
    print("=" * 80)
    print(f"Baseline Score: {baseline_score:.5f}\n")

    for exp in experiments:
        name = exp['name']
        public_score = exp.get('public_score')

        if public_score is None:
            print(f"{name:20s} : N/A")
            continue

        delta = public_score - baseline_score
        delta_pct = (delta / baseline_score) * 100

        status = ""
        if delta > 0:
            status = "+"
        elif delta < 0:
            status = "-"
        else:
            status = " "

        print(f"{name:20s} : {public_score:.5f}  ({status}{abs(delta):.5f}, {delta_pct:+.2f}%)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="實驗比較工具")
    parser.add_argument('--exps', nargs='+', required=True, help="實驗名稱列表")
    parser.add_argument('--base-dir', type=str, default='experiments', help="實驗基礎目錄")
    parser.add_argument('--baseline-score', type=float, default=0.66987,
                       help="Baseline 分數（用於計算改進）")
    parser.add_argument('--output', type=str, help="輸出檔案路徑（可選，CSV 或 HTML）")

    args = parser.parse_args()

    # 載入所有實驗
    print(f"載入 {len(args.exps)} 個實驗...")
    experiments = []
    for exp_name in args.exps:
        exp_data = load_experiment_results(exp_name, args.base_dir)
        if exp_data:
            experiments.append(exp_data)

    if not experiments:
        print("錯誤: 沒有找到任何實驗")
        return

    # 比較指標
    metrics_df = compare_metrics(experiments)
    print_comparison_table(metrics_df, "實驗指標比較")

    # 比較配置
    config_df = compare_configs(experiments)
    print_comparison_table(config_df, "實驗配置比較")

    # 計算改進
    calculate_improvements(experiments, args.baseline_score)

    # 輸出到檔案（可選）
    if args.output:
        output_path = Path(args.output)

        if output_path.suffix == '.csv':
            # CSV 格式
            combined_df = pd.merge(metrics_df, config_df, on='Experiment', how='outer')
            combined_df.to_csv(output_path, index=False)
            print(f"\n結果已保存到: {output_path}")

        elif output_path.suffix == '.html':
            # HTML 格式
            html = f"""
            <html>
            <head>
                <title>實驗比較</title>
                <style>
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    h2 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>實驗比較報告</h1>
                <h2>實驗指標</h2>
                {metrics_df.to_html(index=False)}
                <h2>實驗配置</h2>
                {config_df.to_html(index=False)}
            </body>
            </html>
            """

            with open(output_path, 'w') as f:
                f.write(html)

            print(f"\n結果已保存到: {output_path}")


if __name__ == "__main__":
    main()
