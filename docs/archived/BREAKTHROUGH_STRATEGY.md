# 🚀 突破策略分析報告 (UltraThink)
**基於最新實驗結果的深度分析**

---

## 📊 實驗結果分析

### 當前數據

```
train_dataset_1:
  - XGBoost:    CV AUC = 0.6282 ✅ (最佳)
  - LightGBM:   CV AUC = 0.6177
  - L1 C=0.01:  CV AUC = 0.6083
  - L1 C=0.1:   CV AUC = 0.6137

train_dataset_2:
  - LightGBM:   CV AUC = 0.7092 🔥 (非常好！)
  - (XGBoost 結果未知)
```

---

## 🔑 關鍵發現

### 發現 1: Dataset-Specific 模型選擇至關重要

| Dataset | 最佳模型 | CV AUC | 差距 |
|---------|---------|--------|------|
| Dataset 1 | **XGBoost** | 0.6282 | - |
| Dataset 1 | LightGBM | 0.6177 | -1.05% |
| Dataset 2 | **LightGBM** | 0.7092 | - |
| Dataset 2 | XGBoost? | ? | ? |

**洞察**:
- 不同 dataset **確實需要不同模型**！
- **不應該對所有 dataset 都使用同一模型**
- Dataset 2 的 0.7092 遠高於 Dataset 1 的 0.6282 **(+12.9%)**

### 發現 2: Dataset 2 異常優秀

```
Dataset 2: 0.7092 CV AUC (LightGBM)

這是目前為止最好的單一 dataset 結果！
遠高於：
  - Dataset 1: 0.6282 (差距 +12.9%)
  - 之前的 auto_monitor.log Dataset 2: 0.6549 (差距 +8.3%)
```

**可能原因**:
1. Dataset 2 的疾病信號更強
2. LightGBM 特別適合 Dataset 2 的特徵分佈
3. k-mer 特徵在 Dataset 2 上特別有效
4. Dataset 2 的樣本質量更高

### 發現 3: L1 正則化確認過強

```
L1 C=0.01:  0.6083 (underfitting)
L1 C=0.1:   0.6137 (underfitting)
XGBoost:    0.6282 (+2.4% vs C=0.1)
```

**結論**: L1 LogisticRegression **確實不如樹模型**

---

## 🎯 立即突破方向

### 策略 1: Per-Dataset 最佳模型選擇 (HIGHEST PRIORITY)

**目標**: 為每個 dataset 找出最佳模型

```python
# 系統化測試所有 dataset
models = ['XGBoost', 'LightGBM', 'CatBoost']

results = {}
for dataset_id in range(1, 9):
    best_model = None
    best_auc = 0

    for model_name in models:
        auc = train_and_evaluate(dataset_id, model_name)
        if auc > best_auc:
            best_auc = auc
            best_model = model_name

    results[dataset_id] = {
        'model': best_model,
        'auc': best_auc
    }
```

**預期**:
- 如果 Dataset 2 的 0.7092 是代表性的
- 其他 dataset 也可能有類似提升
- **估計整體提升: +3-5%** → **0.70-0.72**

### 策略 2: Heterogeneous Ensemble (HIGH PRIORITY)

**當前問題**: 只用單一模型 (XGBoost)

**改進方案**: 每個 dataset 用多個模型 ensemble

```python
# 對每個 dataset
dataset_predictions = {}

for dataset_id in range(1, 9):
    # 訓練 3 個模型
    xgb_pred = train_xgboost(dataset_id)
    lgb_pred = train_lightgbm(dataset_id)
    cat_pred = train_catboost(dataset_id)

    # Weighted ensemble (根據 CV AUC)
    weights = optimize_weights([xgb_pred, lgb_pred, cat_pred])
    final_pred = weighted_average([xgb_pred, lgb_pred, cat_pred], weights)

    dataset_predictions[dataset_id] = final_pred
```

**預期提升**: +1-2% (ensemble diversity)

### 策略 3: 深度分析 Dataset 2 的成功因素 (MEDIUM PRIORITY)

**問題**: 為何 Dataset 2 特別好？

**分析方向**:

1. **特徵分佈分析**
```python
# 比較 Dataset 1 vs Dataset 2 的特徵統計
features_d1 = extract_features(dataset_1)
features_d2 = extract_features(dataset_2)

# 找出差異
feature_importance_d1 = xgb_d1.feature_importances_
feature_importance_d2 = lgb_d2.feature_importances_

# 哪些特徵在 D2 特別重要？
top_features_d2 = get_top_features(feature_importance_d2, top_k=50)
```

2. **樣本質量分析**
```python
# Dataset 2 的序列統計
- Average sequences per repertoire
- Average CDR3 length
- V/J gene diversity
- Public clone frequency
```

3. **疾病信號強度**
```python
# 比較 positive vs negative 的 k-mer 差異
kmer_diff_d1 = compute_kmer_divergence(dataset_1)
kmer_diff_d2 = compute_kmer_divergence(dataset_2)

# Dataset 2 的 k-mer 差異更大？
```

**目標**: 將 Dataset 2 的成功經驗應用到其他 dataset

---

## 🔬 進階突破方向

### 策略 4: k-mer 範圍擴展 + Per-Dataset 選擇

**假設**: 不同 dataset 可能需要不同的 k 值

```python
# 測試所有組合
k_configs = [
    [3],
    [4],
    [5],
    [3, 4],
    [4, 5],
    [3, 5],
    [3, 4, 5],
    [4, 5, 6],
]

for dataset_id in range(1, 9):
    best_k_config = None
    best_auc = 0

    for k_list in k_configs:
        features = extract_kmer_features(dataset_id, k_list)
        auc = train_evaluate(features)

        if auc > best_auc:
            best_auc = auc
            best_k_config = k_list

    print(f"Dataset {dataset_id}: Best k={best_k_config}, AUC={best_auc}")
```

**預期**:
- Dataset 1 可能最佳 k=[3,4]
- Dataset 2 可能最佳 k=[4,5] 或 [3,4,5]
- **提升: +1-2%**

### 策略 5: Feature Selection per Dataset

**當前**: 使用所有 k-mer 特徵

**改進**: 根據 feature importance 選擇最佳特徵子集

```python
# 對每個 dataset
for dataset_id in range(1, 9):
    # 1. 訓練初始模型
    model = train_model(dataset_id, all_features)

    # 2. 獲取 feature importance
    importance = model.feature_importances_

    # 3. 選擇 top K features
    top_features = select_top_k(importance, k=1000)

    # 4. 用選定特徵重新訓練
    model_final = train_model(dataset_id, top_features)
```

**理由**:
- 減少 noise features
- 降低過擬合風險
- 提升泛化能力

**預期**: +0.5-1%

### 策略 6: Stacking Ensemble (ADVANCED)

**兩層模型**:

```python
# Level 0: Base models (per dataset)
base_models = {
    'xgboost': XGBClassifier(...),
    'lightgbm': LGBMClassifier(...),
    'catboost': CatBoostClassifier(...),
}

# Level 1: Meta-model
meta_model = LogisticRegression(C=10)  # 用較大的 C

# Training
for dataset_id in range(1, 9):
    # Level 0 predictions (5-fold CV)
    base_predictions = []
    for model_name, model in base_models.items():
        preds = cross_val_predict(model, X, y, cv=5)
        base_predictions.append(preds)

    # Stack predictions as new features
    X_meta = np.column_stack(base_predictions)

    # Train meta-model
    meta_model.fit(X_meta, y)
```

**預期**: +1-2% (但風險較高)

### 策略 7: Hyperparameter Optimization (Auto-tuning)

**使用 Optuna 自動調參**:

```python
import optuna

def objective(trial, dataset_id):
    # XGBoost parameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = XGBClassifier(**params)
    cv_score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

    return cv_score

# 對每個 dataset 優化
for dataset_id in range(1, 9):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset_id), n_trials=100)

    best_params = study.best_params
    print(f"Dataset {dataset_id}: {best_params}")
```

**預期**: +1-3% per dataset

---

## 📈 預期提升路徑 (Updated)

### 保守估計 (低風險)

```
目前最佳:                0.67887

+ Per-dataset 模型選擇:   +0.025 → 0.70387  (基於 D2 的 0.7092)
+ Heterogeneous ensemble: +0.010 → 0.71387
+ k-mer 範圍優化:        +0.010 → 0.72387

預期: 0.724 (+6.6%)
```

### 樂觀估計 (中風險)

```
目前最佳:                0.67887

+ Per-dataset 模型選擇:   +0.030 → 0.70887  (如果其他 dataset 也有提升)
+ Heterogeneous ensemble: +0.015 → 0.72387
+ Feature selection:      +0.010 → 0.73387
+ Hyperparameter tuning:  +0.015 → 0.74887

預期: 0.749 (+10.3%)
```

### 激進估計 (高風險)

```
目前最佳:                0.67887

+ Per-dataset 模型選擇:   +0.035 → 0.71387
+ Heterogeneous ensemble: +0.020 → 0.73387
+ Stacking:              +0.020 → 0.75387
+ k-mer + feature eng:   +0.020 → 0.77387
+ Auto-tuning:           +0.020 → 0.79387

預期: 0.794 (+16.9%)
```

**與目標的差距**:
```
目前:         0.67887
激進估計:     0.79387
GROZD:        0.81364
差距:         -0.01977 (-2.0%)  ← 非常接近！
```

---

## 🎯 立即行動計劃 (優先級排序)

### Phase 1: 快速驗證 (1-2 hours)

```bash
# 1. 測試 Dataset 1 的 XGBoost vs LightGBM vs CatBoost
python quick_compare_models.py --dataset 1

# 2. 測試 Dataset 2 的 XGBoost (對比 LightGBM 0.7092)
python quick_compare_models.py --dataset 2

# 3. 測試所有 8 個 dataset 的最佳模型
for i in {1..8}; do
    python quick_compare_models.py --dataset $i
done
```

**預期輸出**:
```
Dataset 1: XGBoost=0.6282, LightGBM=0.6177, CatBoost=?
Dataset 2: XGBoost=?, LightGBM=0.7092, CatBoost=?
Dataset 3-8: ...
```

### Phase 2: Per-Dataset 最佳化 (2-4 hours)

```python
# per_dataset_optimization.py
results = {}

for dataset_id in range(1, 9):
    print(f"\n=== Dataset {dataset_id} ===")

    # Test 3 models
    xgb_auc = train_xgboost(dataset_id)
    lgb_auc = train_lightgbm(dataset_id)
    cat_auc = train_catboost(dataset_id)

    # Select best
    best_model = max([
        ('XGBoost', xgb_auc),
        ('LightGBM', lgb_auc),
        ('CatBoost', cat_auc)
    ], key=lambda x: x[1])

    results[dataset_id] = {
        'best_model': best_model[0],
        'best_auc': best_model[1],
        'xgb': xgb_auc,
        'lgb': lgb_auc,
        'cat': cat_auc,
    }

    print(f"Best: {best_model[0]} = {best_model[1]:.4f}")

# Save results
with open('per_dataset_best_models.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Phase 3: Heterogeneous Ensemble (2-3 hours)

```python
# heterogeneous_ensemble.py
for dataset_id in range(1, 9):
    # Train all 3 models
    xgb_model = train_xgboost(dataset_id)
    lgb_model = train_lightgbm(dataset_id)
    cat_model = train_catboost(dataset_id)

    # Get predictions
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    cat_pred = cat_model.predict_proba(X_test)[:, 1]

    # Optimize weights (grid search)
    best_weights = None
    best_auc = 0

    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1-w1, 11):
            w3 = 1 - w1 - w2

            ensemble_pred = w1*xgb_pred + w2*lgb_pred + w3*cat_pred
            auc = roc_auc_score(y_test, ensemble_pred)

            if auc > best_auc:
                best_auc = auc
                best_weights = (w1, w2, w3)

    print(f"Dataset {dataset_id}: Weights={best_weights}, AUC={best_auc:.4f}")
```

### Phase 4: k-mer 擴展測試 (1-2 hours)

```python
# kmer_expansion_test.py
k_configs = [[3], [4], [5], [3,4], [4,5], [3,5], [3,4,5]]

for dataset_id in range(1, 9):
    best_k = None
    best_auc = 0

    for k_list in k_configs:
        features = extract_kmer_features(dataset_id, k_list)
        model = train_best_model(dataset_id, features)  # 用 Phase 2 找出的最佳模型
        auc = evaluate(model, features)

        if auc > best_auc:
            best_auc = auc
            best_k = k_list

    print(f"Dataset {dataset_id}: Best k={best_k}, AUC={best_auc:.4f}")
```

---

## 🔍 深度分析：為何 Dataset 2 特別好？

### 需要回答的問題

1. **Dataset 2 的樣本統計**
```python
# 分析 metadata
meta = pd.read_csv('data/train_datasets/train_dataset_2/metadata.csv')

print(f"Total repertoires: {len(meta)}")
print(f"Positive: {meta['label_positive'].sum()}")
print(f"Negative: {(~meta['label_positive']).sum()}")
print(f"Balance: {meta['label_positive'].mean():.2%}")
```

2. **序列特徵分析**
```python
# 分析序列統計
seq_stats = analyze_sequences(dataset_2)

print(f"Avg sequences per repertoire: {seq_stats['avg_seqs']}")
print(f"Avg CDR3 length: {seq_stats['avg_length']}")
print(f"V gene diversity: {seq_stats['v_diversity']}")
print(f"J gene diversity: {seq_stats['j_diversity']}")
```

3. **k-mer 信號強度**
```python
# 比較 positive vs negative 的 k-mer 分佈
pos_kmers = extract_kmers(positive_samples)
neg_kmers = extract_kmers(negative_samples)

# Jensen-Shannon divergence (信號強度指標)
jsd = compute_jsd(pos_kmers, neg_kmers)
print(f"Dataset 2 JSD: {jsd:.4f}")
```

4. **LightGBM vs XGBoost 差異**
```python
# 為何 LightGBM 在 D2 特別好？
# 可能原因：
# - Leaf-wise growth (vs level-wise in XGBoost)
# - Better handling of categorical features
# - Faster training → more iterations
```

---

## 💡 創新想法

### 想法 1: Dataset Clustering

**假設**: 相似的 dataset 可能需要相似的配置

```python
# 1. 計算 dataset 之間的相似度
similarities = compute_dataset_similarity(datasets)

# 2. 聚類
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=3).fit(similarities)

# 3. 對每個 cluster 使用相同配置
for cluster_id in range(3):
    datasets_in_cluster = [i for i, c in enumerate(clusters.labels_) if c == cluster_id]
    best_config = optimize_config(datasets_in_cluster)

    for dataset_id in datasets_in_cluster:
        apply_config(dataset_id, best_config)
```

### 想法 2: Transfer Learning Between Datasets

**如果 Dataset 2 表現特別好，能否遷移到其他 dataset？**

```python
# 1. 在 Dataset 2 訓練一個強模型
model_d2 = train_best_model(dataset_2)

# 2. 獲取 feature importance
importance_d2 = model_d2.feature_importances_

# 3. 在其他 dataset 上使用相同的 top features
top_features_d2 = get_top_k_features(importance_d2, k=1000)

for dataset_id in [1, 3, 4, 5, 6, 7, 8]:
    # 只使用 Dataset 2 的 top features
    X_subset = X_all[:, top_features_d2]
    model = train_model(dataset_id, X_subset)
```

### 想法 3: Meta-Learning (Model Selection)

**訓練一個模型來選擇最佳模型**:

```python
# 1. 計算 dataset 的 meta-features
meta_features = {
    'n_samples': len(dataset),
    'n_positive': sum(labels),
    'avg_seq_length': ...,
    'kmer_diversity': ...,
    'v_gene_entropy': ...,
}

# 2. 訓練 meta-model
# Input: meta-features
# Output: best model (XGBoost/LightGBM/CatBoost)

# 3. 對新 dataset，先計算 meta-features，再選擇模型
```

---

## 📋 實驗檢查清單

### 今天必須完成 (Critical)

- [ ] **測試所有 8 個 dataset 的 XGBoost vs LightGBM vs CatBoost**
- [ ] **找出每個 dataset 的最佳模型**
- [ ] **生成 per-dataset 最佳模型的提交檔案**
- [ ] **提交到 Leaderboard 驗證**

### 明天完成 (High Priority)

- [ ] Heterogeneous ensemble (3 模型加權平均)
- [ ] k-mer 範圍擴展測試 (k=5, k=3-5)
- [ ] Dataset 2 深度分析報告

### 本週完成 (Medium Priority)

- [ ] Feature selection per dataset
- [ ] Hyperparameter auto-tuning (Optuna)
- [ ] Stacking ensemble

### 如果有時間 (Low Priority)

- [ ] Dataset clustering
- [ ] Meta-learning model selection
- [ ] Transfer learning between datasets

---

## 🎯 成功指標

### 短期目標 (1-2 天)

```
Current LB:    0.67887
Target:        0.72+ (保守估計)
Success:       +4% improvement
```

### 中期目標 (3-5 天)

```
Current LB:    0.67887
Target:        0.75+ (樂觀估計)
Success:       +7% improvement
```

### 終極目標 (1 週)

```
Current LB:    0.67887
Target:        0.79+ (激進估計)
Success:       +11% improvement
GROZD:         0.81364
Gap:           -2% (可接受！)
```

---

## 🚨 風險管理

### 已知風險

1. **CV vs LB 差距**
   - Dataset 2 的 0.7092 CV 可能不等於 LB
   - 需要 LB 驗證

2. **過擬合風險**
   - Ensemble 太複雜可能過擬合
   - 需要保守測試

3. **時間壓力**
   - 比賽截止時間：2025-12-17 06:59 UTC
   - 剩餘時間：< 24 小時？

### 緩解策略

1. **每次改動都 LB 驗證**
   - 不要信任 CV
   - 保留提交次數（5/day）

2. **保守優先**
   - 先做低風險高回報的
   - Per-dataset 模型選擇優先
   - Stacking 最後做

3. **時間分配**
   - Phase 1-2: 4 小時
   - Phase 3-4: 3 小時
   - 提交和驗證: 2 小時

---

## 💎 最終建議

### 立即執行（接下來 4 小時）

1. **✅ 運行完整的模型比較**
```bash
python compare_all_models.py --datasets 1-8 --models xgboost,lightgbm,catboost
```

2. **✅ 生成 per-dataset 最佳提交**
```bash
python generate_submission.py --strategy per_dataset_best --output submission_per_dataset.csv
```

3. **✅ 提交到 Leaderboard**
```bash
kaggle competitions submit -f submission_per_dataset.csv -m "Per-dataset best models"
```

### 根據 LB 結果決定下一步

**如果 LB > 0.72**:
→ 繼續 heterogeneous ensemble

**如果 LB 0.69-0.72**:
→ 測試 k-mer 擴展

**如果 LB < 0.69**:
→ 檢查 CV vs LB 洩漏問題

---

**結論**: Dataset 2 的 0.7092 是一個**重大突破信號**！
如果能將這個成功複製到其他 dataset，我們有機會達到 **0.75-0.79**，
非常接近 GROZD 的 0.81364！

**下一步**: 立即執行 Phase 1-2，找出所有 dataset 的最佳模型！
