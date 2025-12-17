#!/usr/bin/env python3
"""
統一實驗訓練入口
=====================

用法:
    python scripts/train_experiment.py --config configs/exp_001.yaml --exp-name exp_001

功能:
- 自動創建實驗目錄
- 複製配置檔案
- 初始化日誌
- 訓練模型並保存
- 生成提交檔案
"""

import argparse
import json
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def setup_experiment_dir(exp_name: str, base_dir: str = "experiments") -> Path:
    """創建實驗目錄結構"""
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 創建子目錄
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "outputs").mkdir(exist_ok=True)

    return exp_dir


def setup_logging(exp_dir: Path) -> logging.Logger:
    """設置日誌"""
    log_file = exp_dir / "logs" / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """載入配置檔案"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], exp_dir: Path):
    """保存配置到實驗目錄"""
    config_path = exp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def train_model(config: Dict[str, Any], exp_dir: Path, logger: logging.Logger):
    """
    訓練模型的主函數

    這裡需要根據 config 中的模型類型調用對應的訓練函數
    """
    logger.info("=" * 60)
    logger.info("開始訓練實驗")
    logger.info("=" * 60)

    model_type = config.get('model', {}).get('type', 'xgboost')
    logger.info(f"模型類型: {model_type}")

    # 動態導入對應的訓練模組
    if model_type == 'xgboost':
        from src.airr_ml25.models.xgboost_model import train_xgboost
        results = train_xgboost(config, exp_dir, logger)

    elif model_type == 'lightgbm':
        from src.airr_ml25.models.lightgbm_model import train_lightgbm
        results = train_lightgbm(config, exp_dir, logger)

    elif model_type == 'ensemble':
        from src.airr_ml25.models.ensemble import train_ensemble
        results = train_ensemble(config, exp_dir, logger)

    elif model_type == 'deep':
        from src.airr_ml25.models.deep_models import train_deep_model
        results = train_deep_model(config, exp_dir, logger)

    else:
        raise ValueError(f"不支援的模型類型: {model_type}")

    return results


def save_metrics(metrics: Dict[str, Any], exp_dir: Path):
    """保存評估指標"""
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="統一實驗訓練入口")
    parser.add_argument('--config', type=str, required=True, help="配置檔案路徑 (YAML)")
    parser.add_argument('--exp-name', type=str, required=True, help="實驗名稱")
    parser.add_argument('--base-dir', type=str, default='experiments', help="實驗基礎目錄")
    parser.add_argument('--resume', action='store_true', help="從檢查點恢復訓練")

    args = parser.parse_args()

    # 1. 設置實驗目錄
    exp_dir = setup_experiment_dir(args.exp_name, args.base_dir)

    # 2. 設置日誌
    logger = setup_logging(exp_dir)
    logger.info(f"實驗目錄: {exp_dir}")

    # 3. 載入配置
    config = load_config(args.config)
    logger.info(f"載入配置: {args.config}")

    # 4. 保存配置到實驗目錄
    save_config(config, exp_dir)

    # 5. 訓練模型
    try:
        results = train_model(config, exp_dir, logger)

        # 6. 保存評估指標
        if 'metrics' in results:
            save_metrics(results['metrics'], exp_dir)
            logger.info(f"評估指標已保存到: {exp_dir / 'metrics.json'}")

        # 7. 檢查提交檔案
        submission_path = exp_dir / "outputs" / "submission.csv"
        if submission_path.exists():
            # 驗證提交檔案格式
            df = pd.read_csv(submission_path)
            logger.info(f"提交檔案: {submission_path}")
            logger.info(f"  - 行數: {len(df)}")
            logger.info(f"  - 欄位: {df.columns.tolist()}")

            # 檢查是否有缺失值
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"發現缺失值: {null_counts[null_counts > 0].to_dict()}")

        logger.info("=" * 60)
        logger.info("訓練完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"訓練失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
