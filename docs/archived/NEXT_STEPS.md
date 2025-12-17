# AIRR-ML-25 下一步行動計劃 (基於研究調研)

## 當前狀態
- **最佳分數**: 0.74006 (v5)
- **目標**: 0.84590 (1st place: SajayR)
- **差距**: 10.6%
- **截止時間**: December 17, 2025 06:59 UTC

---

## 研究發現摘要

| 方法 | 來源 | 性能 | 關鍵技術 |
|------|------|------|----------|
| **Mal-ID** | Science 2025 | AUROC 0.986 | LR Ensemble + BCR+TCR |
| **FACTS** | 2024 | ROC AUC 0.99 | k-mer + LightGBM |
| **DeepRC** | NeurIPS 2020 | SOTA | Modern Hopfield + Attention MIL |
| **EAMIL** | 2024 | Top | ESM encoding + MIL |

---

## 優先實施計劃

### Phase 1: 快速增益 (預估 +3-5%)

#### 1.1 氨基酸理化性質編碼 (2-3 小時)

**科學依據**: Mal-ID (Science 2025), FACTS 都使用了類似方法

```python
# Atchley factors - 5 個因子代表不同理化性質
ATCHLEY_FACTORS = {
    'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
    'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
    'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
    'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
    'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
    'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
    'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
    'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
    'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
    'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
    'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
    'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
    'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
    'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
    'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
    'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
    'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
    'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512],
}
```

#### 1.2 改進 Public Clone Mining (1-2 小時)

```python
from scipy.stats import fisher_exact

def significant_public_clones(pos_reps, neg_reps, p_threshold=0.01):
    """使用 Fisher's exact test (Emerson 2017 方法)"""
    significant = {}
    for seq in all_public_seqs:
        pos_count = sum(seq in rep for rep in pos_reps)
        neg_count = sum(seq in rep for rep in neg_reps)
        table = [[pos_count, len(pos_reps) - pos_count],
                 [neg_count, len(neg_reps) - neg_count]]
        odds, p = fisher_exact(table)
        if p < p_threshold:
            significant[seq] = -np.log10(p) * np.sign(np.log(odds + 1e-10))
    return significant
```

#### 1.3 多樣性指標 (1 小時)

```python
def diversity_features(clone_counts):
    freqs = clone_counts / clone_counts.sum()
    return {
        'shannon': -np.sum(freqs * np.log2(freqs + 1e-10)),
        'simpson': np.sum(freqs ** 2),
        'gini': gini_coefficient(freqs),
        'd50': d50_index(freqs),
        'clonality': 1 - shannon / np.log2(len(freqs)),
    }
```

---

### Phase 2: 中等增益 (預估 +3-5%)

#### 2.1 Attention-based MIL (4-6 小時)

**關鍵**: attention weights 直接對應 Task B 的序列重要性！

```python
class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        h = self.encoder(x)
        a = self.attention(h)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1), -1e9)
        a = F.softmax(a, dim=1)
        z = (a * h).sum(dim=1)
        return self.classifier(z), a.squeeze(-1)
```

#### 2.2 Task B 優化 (2-3 小時)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif

def task_b_scoring(sequences, labels, model_importance):
    # TF-IDF + Mutual Information + Model importance
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
    X_tfidf = tfidf.fit_transform(sequences)
    mi_scores = mutual_info_classif(X_tfidf, labels)

    final_score = (
        0.4 * normalized(mi_scores) +
        0.3 * normalized(model_importance) +
        0.3 * normalized(tfidf_scores)
    )
    return final_score
```

---

### Phase 3: 進階優化 (預估 +2-3%)

#### 3.1 ESM2 嵌入 (如果時間允許)

```python
# 使用小模型節省時間
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

def extract_esm_features(sequences, sample_size=300):
    sampled = random.sample(sequences, min(sample_size, len(sequences)))
    embeddings = []
    for seq in sampled:
        inputs = tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.numpy())
    return np.mean(embeddings, axis=0)
```

---

## 實施時間線

| 時間 | 任務 | 預期增益 |
|------|------|----------|
| 第 1-3 小時 | Atchley 編碼 + 多樣性指標 | +1-2% |
| 第 4-6 小時 | 改進 Public Clone + Fisher test | +2-3% |
| 第 7-12 小時 | Attention MIL + Task B | +2-3% |
| 第 13-18 小時 | ESM2 嵌入 (可選) | +1-2% |
| 第 19-24 小時 | 最終 Ensemble + 提交 | +1% |

---

## 代碼架構

```
champion_v9/
├── __init__.py
├── config.py           # 所有超參數
├── features/
│   ├── kmer.py         # k-mer 特徵 (保留 v5)
│   ├── atchley.py      # 氨基酸理化性質
│   ├── diversity.py    # 多樣性指標
│   └── public_clone.py # 改進版 public clone
├── models/
│   ├── attention_mil.py # MIL 模型
│   └── ensemble.py      # XGBoost + LightGBM
├── task_a.py           # Task A 預測
├── task_b.py           # Task B 序列選擇
└── main.py             # 主程序
```

---

## 風險評估

| 風險 | 影響 | 緩解措施 |
|------|------|----------|
| 過擬合 | 高 | 使用 LODO CV |
| 時間不足 | 高 | 優先實施 Phase 1 |
| GPU 記憶體 | 中 | 使用小模型/批處理 |
| 提交錯誤 | 高 | 格式驗證腳本 |

---

*更新時間: 2025-12-16*
*基於 Science 2025 Mal-ID, NeurIPS 2020 DeepRC, FACTS 2024 等研究*
