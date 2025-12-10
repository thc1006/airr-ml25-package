# 🏆 AIRR-ML-25 Championship 训练指南

## 📋 目录
- [快速开始](#快速开始)
- [故障排除历史](#故障排除历史)
- [监控训练](#监控训练)
- [技术细节](#技术细节)
- [预期结果](#预期结果)

---

## 🚀 快速开始

### 方法1：全自动训练（推荐）

```bash
# 启动训练并自动监控
python3 auto_train_championship.py
```

训练将在后台运行，自动监控GPU状态。

### 方法2：后台运行

```bash
# 后台启动
nohup python3 championship_dl.py > logs/train.log 2>&1 &

# 查看进度
tail -f logs/train.log
```

---

## 🔍 监控训练

### 实时监控面板

```bash
./monitor_training.sh
```

显示内容：
- ✅ 训练进程状态（PID、CPU、内存）
- 🎮 GPU 状态（利用率、内存、温度、功耗）
- 📊 训练进度（最近的日志输出）
- 💡 快速命令提示

### 手动监控命令

```bash
# 查看GPU状态
nvidia-smi

# 实时GPU监控
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/auto_train_*.log

# 检查进程
ps aux | grep championship_dl.py
```

---

## 🐛 故障排除历史

### ❌ 问题1：TypeError with ReduceLROnPlateau

**症状**:
```python
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

**原因**: PyTorch 新版本不再支持 `verbose` 参数

**修复**:
```python
# 旧代码 (Line 607-609)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# 新代码
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
```

### ⚠️ 问题2：潜在的 OOM (Out of Memory)

**预防措施**:

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| **ESM-2 batch_size** | 32 | 16 | 减少序列编码批处理大小 |
| **DataLoader batch_size** | 8 | 4 | 减少模型训练批处理大小 |
| **num_workers** | 4 | 2 | 减少数据加载并行度 |
| **混合精度训练** | ❌ | ✅ | FP16/FP32 自动切换，节省40%内存 |

---

## 🔧 技术细节

### 架构概览

```
🏗️ Championship Pipeline 架构
├── ESM-2 (650M 参数) - 蛋白质语言模型
│   ├── 批处理: 16 sequences
│   ├── 最大序列: 1000 per repertoire
│   └── 输出维度: 1280-dim embeddings
│
├── Attention Aggregator - 序列聚合
│   ├── Multi-head attention (4 heads)
│   ├── 输入: (batch, n_sequences, 1280)
│   └── 输出: (batch, 1280)
│
├── Traditional Features - 传统特征
│   ├── V/J gene usage (90+ features)
│   ├── Clonality metrics (4 features)
│   ├── CDR3 length stats (7 features)
│   └── 总计: ~389 dimensions
│
└── MLP Classifier - 分类器
    ├── 输入: 1280 + 389 = 1669-dim
    ├── 隐藏层: [512, 256]
    ├── Dropout: 0.3
    └── 输出: 1 (binary classification)
```

### 训练配置

```python
优化器: AdamW
  - learning_rate: 1e-4
  - weight_decay: 0.01

学习率调度:
  - ReduceLROnPlateau
  - mode: 'max' (监控AUC)
  - factor: 0.5
  - patience: 3

损失函数: BCEWithLogitsLoss

早停:
  - patience: 5 epochs
  - 监控指标: Validation AUC

交叉验证:
  - Leave-one-dataset-out CV
  - 8 folds (每个dataset作为验证集一次)

混合精度训练:
  - torch.cuda.amp.autocast()
  - torch.cuda.amp.GradScaler()
```

### 数据加载流程

```
Phase 1: 特征名称收集
  └─ 采样10个repertoires per dataset
  └─ 收集所有唯一的V/J/VJ特征名称
  └─ 结果: 389 unique features

Phase 2: 完整数据加载 (with ESM-2)
  ├─ Dataset 1 (400 repertoires) → ~14 min
  ├─ Dataset 2 (400 repertoires) → ~14 min
  ├─ Dataset 3 (400 repertoires) → ~13 min
  ├─ Dataset 4 (400 repertoires) → ~14 min
  ├─ Dataset 5 (400 repertoires) → ~14 min
  ├─ Dataset 6 (397 repertoires) → ~13 min
  ├─ Dataset 7 (302 repertoires) → ~12 min
  └─ Dataset 8 (308 repertoires) → ~12 min

总加载时间: ~1.5-2 hours
总样本数: 3,607 repertoires
```

### GPU 内存占用

```
组件内存占用 (RTX 5080 16GB):

ESM-2 模型: ~2.5 GB
数据批处理: ~500 MB (batch_size=4)
分类模型: ~100 MB
梯度/优化器: ~200 MB
------------------
总计: ~3.3 GB (峰值可能达到 ~4-5 GB)

剩余内存: ~12 GB (充足)
```

---

## 📊 预期结果

### 时间估算

```
数据加载: ~1.5-2 hours
训练 (8 folds × 25 epochs):
  └─ 每个epoch: ~10-15 min
  └─ 早停预计: ~10-15 epochs per fold
  └─ 单个fold: ~2-3 hours
  └─ 总计: ~16-24 hours

总训练时间: 18-26 hours
```

### 性能指标

```
目标:
  └─ Cross-Val AUC: > 0.80
  └─ Public Leaderboard: > 0.78 (Top 10)
  └─ Private Leaderboard: > 0.81364 (击败GROZD)

当前基线 (Logistic Regression):
  └─ CV AUC: ~0.67
  └─ 预期提升: +0.13-0.15

预期 Championship Pipeline:
  └─ CV AUC: 0.80-0.83
  └─ Leaderboard: Top 1-3
```

---

## 📝 检查点

训练完成后，检查以下文件：

```bash
# 模型文件
ls -lh ./models/championship_fold*.pt

# 每个fold应该包含:
# - championship_fold1.pt  (~350 MB)
# - championship_fold2.pt
# - ...
# - championship_fold8.pt

# 日志文件
ls -lh ./logs/

# 验证最佳AUC
grep "Best Val AUC" logs/auto_train_*.log
```

---

## 🎯 下一步

训练完成后：

1. **生成预测**:
   ```bash
   python generate_predictions.py
   ```

2. **创建提交文件**:
   ```bash
   # 检查格式
   python -c "import pandas as pd; df = pd.read_csv('submission.csv'); print(df.shape)"

   # 预期: (404213, 6) rows
   ```

3. **提交到 Kaggle**:
   ```bash
   kaggle competitions submit -c adaptive-immune-profiling-challenge-2025 \
                              -f submission.csv \
                              -m "Championship DL: ESM-2 + Attention (CV AUC: 0.82)"
   ```

---

## 🆘 紧急情况

### 停止训练

```bash
# 找到PID
ps aux | grep championship_dl.py

# 优雅停止
kill <PID>

# 强制停止
kill -9 <PID>
```

### GPU 温度过高 (>85°C)

```bash
# 检查温度
nvidia-smi

# 如果温度过高:
# 1. 停止训练
# 2. 等待GPU冷却
# 3. 检查风扇设置
python set_fan_speed.py 100  # 设置风扇100%

# 4. 重启训练
```

### OOM 错误

如果遇到 CUDA OOM:

```python
# 进一步减小批处理大小 (championship_dl.py)
train_loader = DataLoader(..., batch_size=2, ...)  # 从4降到2
```

---

## 📚 参考资料

- [ESM-2 Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902)
- [DeepRC Architecture](https://www.nature.com/articles/s42256-020-0202-0)
- [Competition Overview](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)

---

*最后更新: 2025-12-09 12:57 UTC*
*训练状态: 🟢 运行中*
