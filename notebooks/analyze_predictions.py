#!/usr/bin/env python3
"""
Comprehensive analysis of AIRR-ML-25 model predictions
Author: Data Scientist Agent
Date: 2025-12-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Paths
RESULTS_DIR = Path("/home/thc1006/dev/airr-ml25-package/results_k4")
DATA_DIR = Path("/home/thc1006/dev/airr-ml25-package/data")
MODEL_PATH = RESULTS_DIR / "model_k3_gpu.json"

print("=" * 80)
print("AIRR-ML-25 Model Performance Analysis")
print("=" * 80)
print(f"Current Score: 0.66987 (Target: 0.82+, Gap: 0.15013)")
print("=" * 80)

# ============================================================================
# 1. PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[1/4] ANALYZING PREDICTION DISTRIBUTIONS...")

prediction_files = sorted(RESULTS_DIR.glob("*_test_predictions.tsv"))
all_predictions = []
dataset_stats = []

for pred_file in prediction_files:
    dataset_name = pred_file.stem.replace("_test_predictions", "")
    df = pd.read_csv(pred_file, sep='\t')

    # Calculate statistics
    stats = {
        'dataset': dataset_name,
        'n_repertoires': len(df),
        'mean_prob': df['label_positive_probability'].mean(),
        'std_prob': df['label_positive_probability'].std(),
        'min_prob': df['label_positive_probability'].min(),
        'max_prob': df['label_positive_probability'].max(),
        'median_prob': df['label_positive_probability'].median(),
        'q25_prob': df['label_positive_probability'].quantile(0.25),
        'q75_prob': df['label_positive_probability'].quantile(0.75),
        'low_confidence': (df['label_positive_probability'].between(0.4, 0.6)).sum(),
        'low_conf_pct': (df['label_positive_probability'].between(0.4, 0.6)).sum() / len(df) * 100
    }
    dataset_stats.append(stats)
    all_predictions.append(df.assign(dataset=dataset_name))

df_stats = pd.DataFrame(dataset_stats)
df_all_preds = pd.concat(all_predictions, ignore_index=True)

print("\n--- Dataset-wise Prediction Statistics ---")
print(df_stats.to_string(index=False))

print(f"\n--- Overall Statistics ---")
print(f"Total test repertoires: {len(df_all_preds):,}")
print(f"Mean probability: {df_all_preds['label_positive_probability'].mean():.4f}")
print(f"Std probability: {df_all_preds['label_positive_probability'].std():.4f}")
print(f"Low confidence predictions (0.4-0.6): {(df_all_preds['label_positive_probability'].between(0.4, 0.6)).sum()} ({(df_all_preds['label_positive_probability'].between(0.4, 0.6)).mean()*100:.1f}%)")

# Visualizations
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, (_, row) in enumerate(df_stats.iterrows()):
    dataset = row['dataset']
    df_dataset = df_all_preds[df_all_preds['dataset'] == dataset]

    axes[i].hist(df_dataset['label_positive_probability'], bins=50, alpha=0.7, edgecolor='black')
    axes[i].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    axes[i].axvspan(0.4, 0.6, alpha=0.2, color='yellow', label=f'Low confidence ({row["low_conf_pct"]:.1f}%)')
    axes[i].set_title(f'{dataset}\n(n={row["n_repertoires"]}, mean={row["mean_prob"]:.3f})', fontsize=10)
    axes[i].set_xlabel('Predicted Probability')
    axes[i].set_ylabel('Count')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'prediction_distributions.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {RESULTS_DIR / 'prediction_distributions.png'}")

# ============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[2/4] ANALYZING FEATURE IMPORTANCE...")

with open(MODEL_PATH, 'r') as f:
    model_data = json.load(f)

# Extract feature importance
if 'feature_importances' in model_data:
    feature_importance = model_data['feature_importances']

    # Convert to DataFrame
    df_importance = pd.DataFrame([
        {'feature': fname, 'importance': score}
        for fname, score in feature_importance.items()
    ]).sort_values('importance', ascending=False)

    print(f"\n--- Top 50 Most Important Features ---")
    print(df_importance.head(50).to_string(index=False))

    # Analyze feature types
    kmer_features = df_importance[df_importance['feature'].str.startswith('kmer_')]
    vj_features = df_importance[df_importance['feature'].str.contains('_v_|_j_')]
    other_features = df_importance[~(df_importance['feature'].str.startswith('kmer_') |
                                     df_importance['feature'].str.contains('_v_|_j_'))]

    print(f"\n--- Feature Type Distribution ---")
    print(f"K-mer features: {len(kmer_features)} (total importance: {kmer_features['importance'].sum():.4f})")
    print(f"V/J gene features: {len(vj_features)} (total importance: {vj_features['importance'].sum():.4f})")
    print(f"Other features: {len(other_features)} (total importance: {other_features['importance'].sum():.4f})")

    # Extract k-mer sequences from top features
    top_kmers = kmer_features.head(100).copy()
    top_kmers['kmer_sequence'] = top_kmers['feature'].str.replace('kmer_', '')
    top_kmers['kmer_length'] = top_kmers['kmer_sequence'].str.len()

    print(f"\n--- Top 30 Most Important K-mers ---")
    print(top_kmers[['kmer_sequence', 'importance', 'kmer_length']].head(30).to_string(index=False))

    # K-mer length distribution in top features
    kmer_length_dist = top_kmers['kmer_length'].value_counts().sort_index()
    print(f"\n--- K-mer Length Distribution (Top 100) ---")
    for k, count in kmer_length_dist.items():
        print(f"k={k}: {count} features")

    # Visualize feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 30 features
    top_30 = df_importance.head(30)
    axes[0].barh(range(len(top_30)), top_30['importance'].values)
    axes[0].set_yticks(range(len(top_30)))
    axes[0].set_yticklabels(top_30['feature'].values, fontsize=8)
    axes[0].set_xlabel('Importance Score')
    axes[0].set_title('Top 30 Most Important Features')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)

    # Feature type comparison
    type_importance = pd.DataFrame([
        {'Type': 'K-mer', 'Total Importance': kmer_features['importance'].sum(), 'Count': len(kmer_features)},
        {'Type': 'V/J Gene', 'Total Importance': vj_features['importance'].sum(), 'Count': len(vj_features)},
        {'Type': 'Other', 'Total Importance': other_features['importance'].sum(), 'Count': len(other_features)}
    ])

    x = np.arange(len(type_importance))
    width = 0.35
    axes[1].bar(x - width/2, type_importance['Total Importance'], width, label='Total Importance')
    axes[1].bar(x + width/2, type_importance['Count']/type_importance['Count'].max(), width, label='Count (normalized)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(type_importance['Type'])
    axes[1].set_ylabel('Value')
    axes[1].set_title('Feature Type Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {RESULTS_DIR / 'feature_importance.png'}")

    # Save detailed feature importance
    df_importance.to_csv(RESULTS_DIR / 'feature_importance_full.csv', index=False)
    print(f"✓ Saved: {RESULTS_DIR / 'feature_importance_full.csv'}")

else:
    print("⚠ Feature importance not found in model file")

# ============================================================================
# 3. DATASET CHARACTERISTICS ANALYSIS
# ============================================================================
print("\n[3/4] ANALYZING DATASET CHARACTERISTICS...")

train_datasets_dir = DATA_DIR / "train_datasets" / "train_datasets"
dataset_characteristics = []

for i in range(1, 9):
    dataset_path = train_datasets_dir / f"train_dataset_{i}"
    metadata_path = dataset_path / "metadata.csv"

    if not metadata_path.exists():
        print(f"⚠ Metadata not found for dataset {i}")
        continue

    metadata = pd.read_csv(metadata_path)

    # Count positive/negative samples
    n_positive = metadata['label_positive'].sum()
    n_negative = len(metadata) - n_positive

    # Sample a few repertoire files to estimate average size
    sample_files = list(dataset_path.glob("*.tsv"))[:5]
    avg_sequences = 0
    if sample_files:
        seq_counts = []
        for f in sample_files:
            try:
                df_seq = pd.read_csv(f, sep='\t', nrows=100000)  # Limit rows for speed
                seq_counts.append(len(df_seq))
            except:
                pass
        if seq_counts:
            avg_sequences = int(np.mean(seq_counts))

    char = {
        'dataset': f'train_dataset_{i}',
        'n_repertoires': len(metadata),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'pos_ratio': n_positive / len(metadata) if len(metadata) > 0 else 0,
        'avg_sequences': avg_sequences
    }
    dataset_characteristics.append(char)

df_char = pd.DataFrame(dataset_characteristics)

print("\n--- Training Dataset Characteristics ---")
print(df_char.to_string(index=False))

print(f"\n--- Key Observations ---")
print(f"1. Dataset 8 is largest: {df_char[df_char['dataset']=='train_dataset_8']['n_repertoires'].values[0]} repertoires")
print(f"2. Class balance range: {df_char['pos_ratio'].min():.3f} to {df_char['pos_ratio'].max():.3f}")
print(f"3. Total training repertoires: {df_char['n_repertoires'].sum()}")

# Visualize dataset characteristics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Repertoire counts
axes[0].bar(df_char['dataset'].str.replace('train_dataset_', 'D'), df_char['n_repertoires'])
axes[0].set_xlabel('Dataset')
axes[0].set_ylabel('Number of Repertoires')
axes[0].set_title('Training Dataset Sizes')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Class balance
x = np.arange(len(df_char))
width = 0.35
axes[1].bar(x - width/2, df_char['n_positive'], width, label='Positive', color='green', alpha=0.7)
axes[1].bar(x + width/2, df_char['n_negative'], width, label='Negative', color='red', alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(df_char['dataset'].str.replace('train_dataset_', 'D'))
axes[1].set_xlabel('Dataset')
axes[1].set_ylabel('Count')
axes[1].set_title('Class Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# Positive ratio
axes[2].bar(df_char['dataset'].str.replace('train_dataset_', 'D'), df_char['pos_ratio'], color='blue', alpha=0.7)
axes[2].axhline(0.5, color='red', linestyle='--', linewidth=2, label='Perfect balance')
axes[2].set_xlabel('Dataset')
axes[2].set_ylabel('Positive Ratio')
axes[2].set_title('Class Balance Ratio')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'dataset_characteristics.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {RESULTS_DIR / 'dataset_characteristics.png'}")

# ============================================================================
# 4. GENERATE IMPROVEMENT RECOMMENDATIONS
# ============================================================================
print("\n[4/4] GENERATING IMPROVEMENT RECOMMENDATIONS...")

print("\n" + "=" * 80)
print("IMPROVEMENT RECOMMENDATIONS (Prioritized)")
print("=" * 80)

recommendations = []

# R1: Low confidence predictions
low_conf_total = (df_all_preds['label_positive_probability'].between(0.4, 0.6)).sum()
low_conf_pct = low_conf_total / len(df_all_preds) * 100

recommendations.append({
    'priority': 1,
    'category': 'Model Calibration',
    'issue': f'{low_conf_total} predictions ({low_conf_pct:.1f}%) fall in low-confidence zone (0.4-0.6)',
    'impact': 'High - directly affects ROC-AUC score',
    'actions': [
        'Implement calibration methods (Platt scaling, isotonic regression)',
        'Try different probability thresholds per dataset',
        'Use ensemble methods to improve confidence',
        'Add temperature scaling to XGBoost predictions'
    ],
    'estimated_gain': '+0.03-0.05'
})

# R2: Dataset 8 special treatment
dataset_8_stats = df_stats[df_stats['dataset'] == 'train_dataset_8']
if len(dataset_8_stats) > 0:
    d8_size = dataset_8_stats['n_repertoires'].values[0]
    d8_mean = dataset_8_stats['mean_prob'].values[0]

    recommendations.append({
        'priority': 2,
        'category': 'Dataset-Specific Modeling',
        'issue': f'Dataset 8 is largest ({d8_size} repertoires) with different k-mer strategy (k=3 vs k=4)',
        'impact': 'Medium-High - accounts for significant portion of test data',
        'actions': [
            'Train separate ensemble for Dataset 8 (k=3, k=4, k=5)',
            'Analyze why k=3 was chosen for Dataset 8',
            'Try adaptive k-mer selection per dataset',
            'Use dataset-specific feature engineering'
        ],
        'estimated_gain': '+0.02-0.04'
    })

# R3: Feature engineering
recommendations.append({
    'priority': 3,
    'category': 'Feature Engineering',
    'issue': 'Current features limited to k-mers and V/J genes',
    'impact': 'High - new features could capture biological patterns',
    'actions': [
        'Add CDR3 length distribution statistics (mean, std, skewness)',
        'Add clonality metrics (Shannon entropy, Gini coefficient, D50)',
        'Add VJ pair combination features',
        'Add public clonotype features (sequences shared across repertoires)',
        'Try multi-scale k-mers (k=3,4,5 combined)',
        'Add amino acid physicochemical properties'
    ],
    'estimated_gain': '+0.04-0.08'
})

# R4: Model architecture
recommendations.append({
    'priority': 4,
    'category': 'Model Architecture',
    'issue': 'Single XGBoost model may not capture all patterns',
    'impact': 'Medium-High - ensemble diversity improves robustness',
    'actions': [
        'Build ensemble: XGBoost + LightGBM + CatBoost',
        'Try different hyperparameters (tree depth, learning rate, subsample)',
        'Use stacking with logistic regression meta-learner',
        'Implement per-dataset models with averaging',
        'Try TabPFN for small datasets (if applicable)'
    ],
    'estimated_gain': '+0.02-0.05'
})

# R5: Cross-validation strategy
recommendations.append({
    'priority': 5,
    'category': 'Validation Strategy',
    'issue': 'Need to ensure generalization across datasets',
    'impact': 'Medium - prevents overfitting to public leaderboard',
    'actions': [
        'Implement leave-one-dataset-out cross-validation',
        'Track per-dataset performance separately',
        'Use stratified splits to maintain class balance',
        'Monitor validation curves to detect overfitting'
    ],
    'estimated_gain': '+0.01-0.03 (prevents overfitting)'
})

# R6: Sequence identification (Task B)
recommendations.append({
    'priority': 6,
    'category': 'Task B Optimization',
    'issue': 'Important sequences selection may not be optimal',
    'impact': 'Medium - affects Jaccard similarity score',
    'actions': [
        'Use SHAP values for sequence-level importance',
        'Cluster similar sequences and select representatives',
        'Weight sequences by frequency and model importance',
        'Try different top-k selection strategies'
    ],
    'estimated_gain': '+0.01-0.03'
})

# Print recommendations
for i, rec in enumerate(recommendations, 1):
    print(f"\n{'─' * 80}")
    print(f"RECOMMENDATION #{i} [Priority: {rec['priority']}] - {rec['category']}")
    print(f"{'─' * 80}")
    print(f"Issue: {rec['issue']}")
    print(f"Impact: {rec['impact']}")
    print(f"Estimated Score Gain: {rec['estimated_gain']}")
    print(f"\nAction Items:")
    for j, action in enumerate(rec['actions'], 1):
        print(f"  {j}. {action}")

# Save recommendations to file
with open(RESULTS_DIR / 'improvement_recommendations.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("AIRR-ML-25 MODEL PERFORMANCE ANALYSIS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Current Score: 0.66987 (Target: 0.82+, Gap: 0.15013)\n")
    f.write(f"Analysis Date: 2025-12-08\n")
    f.write("=" * 80 + "\n\n")

    for i, rec in enumerate(recommendations, 1):
        f.write(f"\n{'─' * 80}\n")
        f.write(f"RECOMMENDATION #{i} [Priority: {rec['priority']}] - {rec['category']}\n")
        f.write(f"{'─' * 80}\n")
        f.write(f"Issue: {rec['issue']}\n")
        f.write(f"Impact: {rec['impact']}\n")
        f.write(f"Estimated Score Gain: {rec['estimated_gain']}\n")
        f.write(f"\nAction Items:\n")
        for j, action in enumerate(rec['actions'], 1):
            f.write(f"  {j}. {action}\n")

print(f"\n✓ Saved: {RESULTS_DIR / 'improvement_recommendations.txt'}")

# Final summary
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY")
print("=" * 80)
print(f"Total test repertoires analyzed: {len(df_all_preds):,}")
print(f"Low-confidence predictions: {low_conf_total} ({low_conf_pct:.1f}%)")
print(f"Top feature type: K-mers ({kmer_features['importance'].sum():.4f} total importance)")
print(f"Dataset 8 size: {d8_size} repertoires (largest)")
print(f"\nEstimated total score gain from all recommendations: +0.13 to +0.28")
print(f"Target score: 0.82, Estimated achievable: 0.80-0.95")
print("=" * 80)

print("\n✓ Analysis complete. Check results_k4/ for detailed outputs.")
