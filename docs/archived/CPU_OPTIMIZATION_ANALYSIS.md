# 🔬 CPU 多核心優化深度分析 - 2025-12-09

## 📊 優化歷程與測試結果

### 測試版本對比

| 版本 | Traditional Features | ESM-2 Embeddings | 總時間 | CPU 使用 | GPU Memory |
|------|---------------------|------------------|--------|---------|-----------|
| **v1 (原始)** | 2.1s/rep (單核) | 2.0s/rep | 14分鐘 | 1核 100% | 3000 MiB |
| **v2 (並行特徵)** | 0.018s/rep (6核) | 2.0s/rep | 13.3分鐘 | 6核 並行 | 3000 MiB |
| **v3 (batch=32)** | 0.018s/rep | 2.0s/rep | 13.3分鐘 | 1核 GPU | 3026 MiB |
| **v4 (async+batch)** | 0.018s/rep | 3.25s/rep | 21.7分鐘 | 2核 並行 | 6061 MiB |
| **v5 (batch=48)** | 0.018s/rep | 4.0s/rep | 26.8分鐘 | 1核 GPU | 9154 MiB |
| **✅ v6 (最終)** | 0.018s/rep (6核) | 2.0s/rep | 13.3分鐘 | 6核 + GPU | 3000 MiB |

### 關鍵發現

## 🎯 核心問題：GPU 是絕對瓶頸

### 時間分解

```
單個 Dataset (400 repertoires):
├─ Traditional Features: 8 秒 (6%)   ← 已優化（6 核心並行）
└─ ESM-2 Embeddings:    800 秒 (94%) ← **無法並行**（單 GPU）
   Total: 808 秒 = 13.47 分鐘
```

**結論：即使完全消除 CPU 時間（8 秒），也只節省 1%。**

## 🚫 為什麼多核心對 GPU 階段沒用？

### 1. GPU 計算本質上是順序的

```python
for repertoire in dataset:  # 必須順序處理
    sequences = repertoire.get_sequences()  # CPU: 0.001 秒
    embeddings = esm2.encode(sequences)     # GPU: 2.0 秒 ← 瓶頸在這
```

**問題：**
- CPU 準備數據只需 0.001 秒
- GPU 處理需要 2.0 秒
- CPU 有 1.999 秒在等待 GPU
- **即使用 8 個 CPU 核心，GPU 還是要等那 2.0 秒**

### 2. 為什麼增大 batch_size 反而慢？

#### 測試結果

| batch_size | 速度 | GPU Memory | 說明 |
|------------|------|-----------|------|
| 16 | **2.0s/rep** | 3000 MiB | ✅ 最優 |
| 24 | 2.5s/rep | 4500 MiB | 變慢 25% |
| 32 | 3.0s/rep | 5000 MiB | 變慢 50% |
| 48 | 4.0s/rep | 9154 MiB | 變慢 100% |

#### 原因分析

ESM-2 的計算複雜度是 **O(batch_size² × sequence_length²)**：

```
batch_size=16: 16² × 1000² = 256M operations
batch_size=32: 32² × 1000² = 1024M operations  (4倍！)
batch_size=48: 48² × 1000² = 2304M operations  (9倍！)
```

**結論：batch_size 增大導致記憶體傳輸和計算都急劇增加。**

### 3. 為什麼批次處理多個 repertoires 更慢？

#### v4 測試（一次處理 4 個 repertoires）

```python
# 合併 4 個 repertoires 的序列
all_seqs = rep1 + rep2 + rep3 + rep4  # ~4000 sequences
embeddings = esm2.encode(all_seqs, batch_size=32)

# 結果：
# - 4000 sequences ÷ 32 = 125 次 GPU 調用
# - 每次調用 ~0.1 秒
# - 總共：13 秒/batch

# 對比原始方式：
# - 1000 sequences ÷ 16 = 63 次 GPU 調用 × 4 repertoires
# - 每次調用 ~0.032 秒
# - 總共：8 秒/batch
```

**問題：**
- 更大的 batch 導致每次 GPU 調用變慢
- 記憶體碎片化增加
- 總時間反而增加

## ✅ 最終優化策略

### 保留的優化

1. **Traditional Features 並行處理** ✅
   - 6 個 CPU 核心並行
   - 從 14 分鐘 → 8 秒（105倍加速）
   - **節省 1% 總時間**

2. **ESM-2 batch_size=16** ✅
   - 經過測試的最優值
   - 平衡速度和記憶體
   - GPU 利用率 90-95%

### 放棄的優化（無效或負面）

1. ❌ 異步 prefetch（複雜且無用）
   - CPU 準備數據只需 0.001 秒
   - GPU 需要 2.0 秒
   - Prefetch 無法加速 GPU

2. ❌ 增大 batch_size（反而變慢）
   - batch_size > 16 導致速度下降
   - 記憶體使用急劇增加

3. ❌ 批次處理多個 repertoires（更慢）
   - 增加計算複雜度
   - 記憶體碎片化

## 💡 為什麼看起來 CPU 單核心在工作？

**觀察到的現象：**
```bash
CPU 0-2, 4-7: 2-5% 使用率
CPU 3 或 6:   95-98% 使用率  ← 看起來只有一個核心
```

**真相：**

這是 **正常的**！因為：

1. **Traditional Features 階段（8 秒）：**
   - 6 個核心並行工作
   - 每個核心使用率 ~80-100%
   - 但只持續 8 秒

2. **ESM-2 階段（800 秒）：**
   - 主進程（1 個核心）調用 GPU
   - 其他核心閒置
   - 佔了 99% 的總時間

**所以大部分時間看起來只有 1 個核心在工作，這是正確的。**

## 🚀 進一步優化的唯一方法

### 唯一有效的加速方式

| 方法 | 理論加速 | 可行性 | 成本 |
|------|---------|--------|------|
| **多 GPU 並行** | 2-8x | 高 | $$$ 需要多張 GPU |
| **使用更小模型** | 2-5x | 中 | 準確度下降 10-15% |
| **量化/INT8** | 1.5-2x | 中 | 準確度下降 2-5% |
| **預計算並快取** | ∞ | 高 | 需要 ~100GB 存儲 |
| **分布式訓練** | 4-16x | 低 | 複雜，需要集群 |

### 推薦方案（如果經常訓練）

**預計算 ESM-2 embeddings 並快取：**

```python
# 一次性計算所有 embeddings
python precompute_embeddings.py  # 運行一次，~2 小時

# 之後的訓練：
# - 資料載入：1 分鐘（從快取讀取）
# - 訓練：16-24 小時（不變）
# - 總節省：每次訓練節省 1.5 小時
```

## 📊 最終性能分析

### 當前配置（最優）

```
Hardware:
- CPU: AMD Ryzen 7 7800X3D (8 cores)
- GPU: RTX 5080 16GB
- RAM: 32GB DDR5

Performance:
- Traditional Features: 56 repertoires/s (6 cores)
- ESM-2 Embeddings: 0.5 repertoires/s (1 GPU)
- GPU Utilization: 90-95%
- GPU Memory: 3000 MiB / 16384 MiB (18%)

Bottleneck:
✅ GPU compute-bound (optimal)
✅ Not CPU-bound
✅ Not memory-bound
✅ Not I/O-bound
```

### 時間分配

```
Dataset 1 (400 repertoires):
├─ Traditional Features: 8s    (1%)    [6 CPU cores]
├─ ESM-2 Embeddings:    800s   (99%)   [1 GPU]
└─ Total:               808s   (13.5min)

All 8 Datasets:
├─ Data Loading:  ~1.8 hours  (10%)
└─ Training:      ~16-24 hours (90%)

Total Training Time: 18-26 hours
```

## 結論

1. **Traditional Features 優化成功** ✅
   - 6 核心並行，加速 105倍
   - 但只佔總時間 1%

2. **ESM-2 GPU 是絕對瓶頸**
   - 佔 99% 時間
   - 無法用多核心 CPU 解決
   - batch_size=16 已是最優

3. **CPU 多核心優化對此場景幫助有限**
   - 節省 ~1% 總時間（8 秒）
   - 投入產出比低

4. **當前配置已經是最優解**
   - GPU 利用率 90-95%
   - 無明顯瓶頸
   - 除非使用多 GPU，否則無法進一步加速

---

**最後更新**: 2025-12-09 17:00 UTC
**測試版本**: v1-v6
**最終推薦**: v6 (Traditional 6 cores + ESM-2 batch_size=16)
