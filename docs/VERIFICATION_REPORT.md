# 奪冠計畫驗證報告

> **驗證日期**: 2025-12-07
> **驗證目的**: 交叉驗證前一版奪冠計畫的正確性

---

## 驗證摘要

| 驗證項目 | 結果 | 說明 |
|----------|------|------|
| Task B 問題診斷 | **部分正確** ⚠️ | 問題存在但描述不夠精確 |
| 0.72866 分數來源 | **錯誤** ❌ | 不是官方 baseline，是參賽者用 k=4 的結果 |
| 排行榜資訊 | **過時** ⚠️ | 領先者已從 GROZD 變為 SajayR |
| 文獻引用 | **正確** ✅ | DeepRC、ESM-2 引用均已驗證 |
| 分數預測 | **需調整** ⚠️ | 部分預測依據有誤 |

---

## 詳細驗證結果

### 1. Task B 問題診斷

#### 原始聲明 (CHAMPIONSHIP_WINNING_PLAN_v2.md)
> "我們的 Task B 實現有根本性錯誤：
> 1. 使用 k-mer 頻率而非二進制
> 2. 使用 ensemble 重要性而非 LogReg 係數"

#### 驗證結果

**聲明 1: "使用 k-mer 頻率而非二進制" - 錯誤** ❌

經檢查 `champion_pipeline.py` 第 862-870 行：
```python
seen_kmers = set()
for k in self.config.k_values:
    if len(seq) >= k:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_key = f"k{k}_{kmer}"
            if kmer_key not in seen_kmers and kmer_key in kmer_importances:
                seen_kmers.add(kmer_key)  # 只加一次！這是二進制！
                score += kmer_importances[kmer_key]
```

**結論**: Champion pipeline 使用 `seen_kmers` set 防止重複計算，實際上是二進制方式。原始聲明錯誤。

---

**聲明 2: "使用 ensemble 重要性而非 LogReg 係數" - 正確** ✅

經檢查 `champion_pipeline.py` 第 689-718 行的 `get_feature_importance()` 方法：
```python
def get_feature_importance(self) -> Dict[str, float]:
    importances = defaultdict(float)

    # XGBoost feature_importances_ * 0.30
    # LightGBM feature_importances_ * 0.30
    # CatBoost feature_importances_ * 0.25
    # np.abs(LogReg coef_) * 0.15  # 使用絕對值！失去正負號！
```

**結論**: 確實使用了 ensemble 的加權特徵重要性，且 LogReg 係數取了絕對值（失去正負號資訊）。

---

**附加發現：main.py 有正確實現！**

`main.py` 第 360-388 行完全符合官方 baseline：
```python
def score_all_sequences(self, sequences_df, sequence_col='junction_aa'):
    coefficients = coefficients / scaler.scale_  # 反標準化
    counts[kmer_to_index[kmer]] = 1  # 二進制
    scores.append(np.dot(counts, coefficients))  # LogReg 係數
```

**結論**: 如果我們使用 main.py 而非 champion_pipeline.py，Task B 應該會有正確的實現。

---

### 2. 官方 Baseline 分數 0.72866

#### 原始聲明
> "官方 baseline 使用 k=3 k-mers + LogReg 達到 0.72866"

#### 驗證結果 - 錯誤 ❌

從 Tavily 搜索結果：
```
"follow_the_signs: Logistic regression on 4-mer frequencies similar to
example baseline predictor provided... 0.72866"
```

**事實**:
- 0.72866 來自參賽者 "follow_the_signs"
- 他們使用 **k=4** (4-mer)，不是 k=3
- 這不是官方 baseline 的分數

**重要發現**: k=4 可能比 k=3 效果更好！官方 baseline 用 k=3，follow_the_signs 用 k=4 達到 0.72866。

---

### 3. 排行榜資訊

#### 原始資料 (06_Leaderboard.md - 2025-12-04)
| Rank | Team | Score |
|------|------|-------|
| 1 | GROZD | 0.81364 |
| 2 | WoLongFengChu | 0.80509 |
| 3 | GoBlue | 0.80412 |
| 4 | SajayR | 0.79463 |

#### 最新資料 (2025-12-07 Tavily 搜索)
| Rank | Team | Score |
|------|------|-------|
| 1 | **SajayR** | **0.82518** ✨ |
| 2 | GROZD | 0.81998 |
| 3 | GoBlue | 0.80992 |

**變化分析**:
- SajayR 從第 4 名躍升至第 1 名
- SajayR 分數提升約 0.03 (0.79463 → 0.82518)
- 前三名分數都有提升

**新目標**: 要打敗 SajayR，需要超過 0.82518

---

### 4. 文獻引用驗證

#### DeepRC / Modern Hopfield Networks - 正確 ✅

**論文資訊**:
- 標題: "Modern Hopfield Networks and Attention for Immune Repertoire Classification"
- 會議: NeurIPS 2020
- 作者包括: Victor Greiff, Geir Kjetil Sandve
- 連結: https://proceedings.neurips.cc/paper/2020/hash/da4902cb0bc38210839714ebdcf0efc3-Abstract.html

**重要發現**: Victor Greiff 和 Geir Kjetil Sandve 同時是：
1. DeepRC 論文作者
2. AIRR-ML-25 競賽主辦方

**結論**: DeepRC 方法很可能對此競賽有效！

---

#### ESM-2 蛋白質語言模型 - 正確 ✅

從 Scholar Gateway 搜索驗證：
- ESM-2 (Lin et al., 2023) 確實存在
- 被廣泛用於蛋白質/肽序列嵌入
- 33 層模型生成 1280 維嵌入
- 有多篇論文展示其在免疫相關任務的應用

---

### 5. 分數預測調整

#### 原始預測
| 組件 | 預期提升 | 累計分數 |
|------|---------|---------|
| 基線 | - | 0.65 |
| **修復 Task B** | **+0.10~0.15** | **0.75~0.80** |
| V/J 基因特徵 | +0.03~0.05 | 0.78~0.85 |

#### 調整後預測

由於發現：
1. main.py 已有正確的 Task B 實現
2. champion_pipeline.py 的問題不是二進制 vs 頻率，而是係數來源
3. k=4 可能比 k=3 更好

**新預測**:

| 組件 | 預期提升 | 累計分數 | 信心度 |
|------|---------|---------|--------|
| 基線 (k=3) | - | 0.65 | 高 |
| 使用 k=4 | +0.05~0.08 | 0.70~0.73 | 中-高 |
| 修復 Task B 係數來源 | +0.03~0.05 | 0.73~0.78 | 中 |
| 加入 V/J 基因特徵 | +0.02~0.04 | 0.75~0.82 | 中 |
| 多尺度 k-mer (k=3,4,5) | +0.01~0.03 | 0.76~0.85 | 低-中 |
| Ensemble stacking | +0.02~0.04 | 0.78~0.87 | 中 |
| DeepRC 注意力 | +0.02~0.05 | 0.80~0.90 | 低 |

**關鍵洞察**:
- 最快的提升可能來自改用 k=4
- Task B 的問題較預期輕微（已是二進制，只需改係數來源）

---

## 修正後的行動建議

### P0 (立即行動) - 預期提升 0.05-0.08

1. **使用 k=4 而非 k=3**
   - follow_the_signs 用 k=4 達到 0.72866
   - 我們用 k=3 只有 ~0.65
   - 簡單修改 `load_and_encode_kmers(train_dir_path, k=4)`

2. **確認使用 main.py 的 Task B 實現**
   - main.py 的 `score_all_sequences` 是正確的
   - 確保 submission 使用此方法

### P1 (第二優先) - 預期提升 0.03-0.05

3. **加入 V/J 基因特徵**
   - 使用 champion_pipeline.py 的 V/J 特徵提取
   - 但保持 main.py 的 Task B 方法

### P2 (如有時間) - 預期提升 0.02-0.04

4. **Multi-scale k-mers (k=3,4,5)**
   - 合併多個 k 值的特徵

5. **Ensemble 優化**
   - XGBoost + LightGBM + LogReg 堆疊

---

## 結論

原始奪冠計畫有以下問題：

| 問題 | 影響程度 |
|------|---------|
| 0.72866 分數來源錯誤 | 中 - 誤導預期 |
| Task B "頻率 vs 二進制" 聲明錯誤 | 低 - 不影響修復方向 |
| 排行榜資訊過時 | 中 - 目標需調整 |
| k 值重要性被低估 | **高** - k=4 可能是關鍵改進 |

**最重要的修正**: 優先嘗試 **k=4**，這可能是最簡單且最有效的改進。

---

*驗證報告生成時間: 2025-12-07*
*下次行動: 使用 k=4 重新訓練並提交*
