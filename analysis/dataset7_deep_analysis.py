#!/usr/bin/env python3
"""
Dataset 7 Deep Analysis
目標：找出為什麼 V9 和 V5 在 Dataset 7 上的預測相關性極低
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Paths
TRAIN_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/train_datasets/train_datasets")
TEST_ROOT = Path("/home/thc1006/dev/airr-ml25-package/data/test_datasets/test_datasets")

def load_metadata(dataset_path):
    """Load metadata from dataset"""
    metadata_file = dataset_path / "metadata.csv"
    if metadata_file.exists():
        return pd.read_csv(metadata_file)
    return None

def analyze_repertoire(tsv_file):
    """Analyze a single repertoire file"""
    try:
        df = pd.read_csv(tsv_file, sep='\t')

        stats = {
            'n_sequences': len(df),
            'unique_junctions': df['junction_aa'].nunique(),
            'unique_v_genes': df['v_call'].nunique() if 'v_call' in df.columns else 0,
            'unique_j_genes': df['j_call'].nunique() if 'j_call' in df.columns else 0,
            'has_templates': 'templates' in df.columns,
            'mean_cdr3_length': df['junction_aa'].str.len().mean(),
            'median_cdr3_length': df['junction_aa'].str.len().median(),
        }

        if 'templates' in df.columns:
            stats['mean_templates'] = df['templates'].mean()
            stats['median_templates'] = df['templates'].median()
            stats['clonality_index'] = (df['templates'].sum() - len(df)) / (df['templates'].sum() - 1) if df['templates'].sum() > 1 else 0

        # Top V genes
        if 'v_call' in df.columns:
            v_counts = df['v_call'].value_counts()
            stats['top_v_gene'] = v_counts.index[0] if len(v_counts) > 0 else None
            stats['top_v_freq'] = v_counts.iloc[0] / len(df) if len(v_counts) > 0 else 0

        # Top J genes
        if 'j_call' in df.columns:
            j_counts = df['j_call'].value_counts()
            stats['top_j_gene'] = j_counts.index[0] if len(j_counts) > 0 else None
            stats['top_j_freq'] = j_counts.iloc[0] / len(df) if len(j_counts) > 0 else 0

        return stats
    except Exception as e:
        print(f"Error analyzing {tsv_file.name}: {e}")
        return None

def analyze_dataset(dataset_num):
    """Analyze entire dataset"""
    print(f"\n{'='*80}")
    print(f"DATASET {dataset_num} ANALYSIS")
    print(f"{'='*80}")

    # Load metadata
    train_path = TRAIN_ROOT / f"train_dataset_{dataset_num}"
    metadata = load_metadata(train_path)

    if metadata is None:
        print(f"No metadata found for dataset {dataset_num}")
        return None

    print(f"\n1. METADATA SUMMARY")
    print(f"-" * 40)
    print(f"Total repertoires: {len(metadata)}")
    print(f"Positive cases: {metadata['label_positive'].sum()} ({metadata['label_positive'].mean()*100:.1f}%)")
    print(f"Negative cases: {(~metadata['label_positive']).sum()} ({(~metadata['label_positive']).mean()*100:.1f}%)")

    # Demographics
    if 'age' in metadata.columns:
        print(f"\nAge: {metadata['age'].mean():.1f} ± {metadata['age'].std():.1f} (range: {metadata['age'].min()}-{metadata['age'].max()})")
    if 'sex' in metadata.columns:
        print(f"Sex distribution: {metadata['sex'].value_counts().to_dict()}")
    if 'race' in metadata.columns:
        print(f"Race distribution: {metadata['race'].value_counts().to_dict()}")
    if 'sequencing_run_id' in metadata.columns:
        print(f"Batches: {metadata['sequencing_run_id'].nunique()} - {metadata['sequencing_run_id'].value_counts().to_dict()}")

    # Analyze repertoires
    print(f"\n2. REPERTOIRE CHARACTERISTICS")
    print(f"-" * 40)

    repertoire_stats = []
    for idx, row in metadata.iterrows():
        tsv_file = train_path / row['filename']
        if tsv_file.exists():
            stats = analyze_repertoire(tsv_file)
            if stats:
                stats['repertoire_id'] = row['repertoire_id']
                stats['label_positive'] = row['label_positive']
                if 'sequencing_run_id' in row:
                    stats['batch'] = row['sequencing_run_id']
                repertoire_stats.append(stats)

    rep_df = pd.DataFrame(repertoire_stats)

    # Overall statistics
    print(f"\nRepertoire size:")
    print(f"  Mean: {rep_df['n_sequences'].mean():.0f}")
    print(f"  Median: {rep_df['n_sequences'].median():.0f}")
    print(f"  Range: {rep_df['n_sequences'].min():.0f} - {rep_df['n_sequences'].max():.0f}")

    print(f"\nUnique sequences per repertoire:")
    print(f"  Mean: {rep_df['unique_junctions'].mean():.0f}")
    print(f"  Median: {rep_df['unique_junctions'].median():.0f}")

    print(f"\nCDR3 length:")
    print(f"  Mean: {rep_df['mean_cdr3_length'].mean():.2f}")
    print(f"  Median: {rep_df['median_cdr3_length'].median():.2f}")

    if 'clonality_index' in rep_df.columns:
        print(f"\nClonality index:")
        print(f"  Mean: {rep_df['clonality_index'].mean():.4f}")
        print(f"  Median: {rep_df['clonality_index'].median():.4f}")

    # Compare positive vs negative
    print(f"\n3. POSITIVE vs NEGATIVE COMPARISON")
    print(f"-" * 40)

    pos = rep_df[rep_df['label_positive']]
    neg = rep_df[~rep_df['label_positive']]

    features_to_compare = ['n_sequences', 'unique_junctions', 'mean_cdr3_length', 'clonality_index']

    for feature in features_to_compare:
        if feature in rep_df.columns:
            pos_mean = pos[feature].mean()
            neg_mean = neg[feature].mean()

            # T-test
            t_stat, p_val = ttest_ind(pos[feature].dropna(), neg[feature].dropna())

            print(f"\n{feature}:")
            print(f"  Positive: {pos_mean:.3f}")
            print(f"  Negative: {neg_mean:.3f}")
            print(f"  Difference: {pos_mean - neg_mean:.3f} ({(pos_mean/neg_mean - 1)*100:+.1f}%)")
            print(f"  T-test p-value: {p_val:.4e}")

    # V/J gene usage comparison
    print(f"\n4. V/J GENE USAGE PATTERNS")
    print(f"-" * 40)

    if 'top_v_gene' in rep_df.columns:
        pos_v = pos['top_v_gene'].value_counts().head(10)
        neg_v = neg['top_v_gene'].value_counts().head(10)

        print(f"\nTop V genes in POSITIVE cases:")
        for gene, count in pos_v.items():
            print(f"  {gene}: {count} ({count/len(pos)*100:.1f}%)")

        print(f"\nTop V genes in NEGATIVE cases:")
        for gene, count in neg_v.items():
            print(f"  {gene}: {count} ({count/len(neg)*100:.1f}%)")

    # Batch effects
    if 'batch' in rep_df.columns:
        print(f"\n5. BATCH EFFECTS")
        print(f"-" * 40)

        batch_label = pd.crosstab(rep_df['batch'], rep_df['label_positive'], normalize='index')
        print(f"\nLabel distribution by batch:")
        print(batch_label)

        chi2, p_val = chi2_contingency(pd.crosstab(rep_df['batch'], rep_df['label_positive']))[:2]
        print(f"\nChi-square test p-value: {p_val:.4e}")

    return rep_df

def compare_datasets():
    """Compare Dataset 7 with other datasets"""
    print(f"\n{'='*80}")
    print(f"CROSS-DATASET COMPARISON")
    print(f"{'='*80}")

    all_datasets = []

    for dataset_num in range(1, 9):
        train_path = TRAIN_ROOT / f"train_dataset_{dataset_num}"
        metadata = load_metadata(train_path)

        if metadata is None:
            continue

        dataset_stats = {
            'dataset': dataset_num,
            'n_repertoires': len(metadata),
            'n_positive': metadata['label_positive'].sum(),
            'positive_rate': metadata['label_positive'].mean(),
        }

        # Quick repertoire stats
        repertoire_sizes = []
        for idx, row in metadata.iterrows():
            tsv_file = train_path / row['filename']
            if tsv_file.exists():
                try:
                    df = pd.read_csv(tsv_file, sep='\t')
                    repertoire_sizes.append(len(df))
                except:
                    pass

        if repertoire_sizes:
            dataset_stats['mean_repertoire_size'] = np.mean(repertoire_sizes)
            dataset_stats['median_repertoire_size'] = np.median(repertoire_sizes)

        all_datasets.append(dataset_stats)

    comparison_df = pd.DataFrame(all_datasets)
    print("\nDataset Comparison:")
    print(comparison_df.to_string(index=False))

    # Highlight Dataset 7
    print(f"\n{'='*80}")
    print(f"DATASET 7 UNIQUENESS")
    print(f"{'='*80}")

    ds7 = comparison_df[comparison_df['dataset'] == 7].iloc[0]
    others = comparison_df[comparison_df['dataset'] != 7]

    print(f"\nDataset 7 vs Others:")
    print(f"  Positive rate: {ds7['positive_rate']:.3f} vs {others['positive_rate'].mean():.3f} (mean of others)")
    print(f"  N repertoires: {ds7['n_repertoires']:.0f} vs {others['n_repertoires'].mean():.0f} (mean of others)")

    if 'mean_repertoire_size' in ds7:
        print(f"  Mean repertoire size: {ds7['mean_repertoire_size']:.0f} vs {others['mean_repertoire_size'].mean():.0f} (mean of others)")

    return comparison_df

def analyze_public_clonotypes():
    """Analyze public clonotypes in Dataset 7"""
    print(f"\n{'='*80}")
    print(f"PUBLIC CLONOTYPE ANALYSIS")
    print(f"{'='*80}")

    train_path = TRAIN_ROOT / f"train_dataset_7"
    metadata = load_metadata(train_path)

    if metadata is None:
        return

    # Collect all CDR3 sequences
    all_sequences_pos = []
    all_sequences_neg = []

    for idx, row in metadata.iterrows():
        tsv_file = train_path / row['filename']
        if not tsv_file.exists():
            continue

        try:
            df = pd.read_csv(tsv_file, sep='\t')
            sequences = df['junction_aa'].tolist()

            if row['label_positive']:
                all_sequences_pos.extend(sequences)
            else:
                all_sequences_neg.extend(sequences)
        except:
            pass

    # Count sequences
    pos_counter = Counter(all_sequences_pos)
    neg_counter = Counter(all_sequences_neg)

    # Find public clonotypes (appearing in multiple individuals)
    print(f"\nTotal unique sequences:")
    print(f"  Positive: {len(pos_counter):,}")
    print(f"  Negative: {len(neg_counter):,}")

    # Find sequences appearing in both groups
    shared_sequences = set(pos_counter.keys()) & set(neg_counter.keys())
    print(f"\nShared sequences between groups: {len(shared_sequences):,}")

    # Find highly frequent public clonotypes
    public_threshold = 5  # Appears at least 5 times

    public_pos = {seq: count for seq, count in pos_counter.items() if count >= public_threshold}
    public_neg = {seq: count for seq, count in neg_counter.items() if count >= public_threshold}

    print(f"\nPublic clonotypes (count >= {public_threshold}):")
    print(f"  Positive: {len(public_pos):,}")
    print(f"  Negative: {len(public_neg):,}")

    # Top public clonotypes
    print(f"\nTop 20 public clonotypes in POSITIVE:")
    for seq, count in sorted(public_pos.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {seq}: {count}")

    print(f"\nTop 20 public clonotypes in NEGATIVE:")
    for seq, count in sorted(public_neg.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {seq}: {count}")

def main():
    print("Dataset 7 Deep Analysis Report")
    print("=" * 80)

    # 1. Analyze Dataset 7 in detail
    dataset7_stats = analyze_dataset(7)

    # 2. Compare with other datasets
    comparison = compare_datasets()

    # 3. Analyze public clonotypes
    analyze_public_clonotypes()

    # 4. Summary and recommendations
    print(f"\n{'='*80}")
    print(f"SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*80}")

    print("""
    Key Findings to Investigate:

    1. CHECK LABEL BALANCE:
       - Is Dataset 7's positive/negative ratio significantly different?
       - Could this cause different prediction strategies?

    2. CHECK BATCH EFFECTS:
       - Are positive/negative cases distributed differently across batches?
       - Could V9 be more sensitive to batch effects than V5?

    3. CHECK REPERTOIRE SIZE:
       - Are Dataset 7 repertoires unusually large/small?
       - Could this affect feature extraction differently in V9 vs V5?

    4. CHECK PUBLIC CLONOTYPE PATTERNS:
       - 733 significant public clones - is this normal or exceptional?
       - Are these clones actually predictive or noise?

    5. CHECK TEST SET CHARACTERISTICS:
       - How many test repertoires in 7_1 vs 7_2?
       - Are they from the same distribution as training data?

    HYPOTHESIS:
    If V9 and V5 have correlation 0.07-0.20, but both have reasonable AUC,
    it suggests they're using COMPLETELY DIFFERENT signals:
    - V5 might be using simple frequency patterns
    - V9 might be using complex deep learning features that don't correlate

    This isn't necessarily bad - it could mean V9 found orthogonal information.
    The question is: which one is actually correct for the test set?

    RECOMMENDED ACTIONS:
    1. Check validation AUC for Dataset 7 specifically in both V5 and V9
    2. Analyze which model has better calibration on Dataset 7
    3. Consider ensemble with different weights for Dataset 7
    4. Investigate if V9's attention mechanism is focusing on noise in Dataset 7
    """)

if __name__ == "__main__":
    main()
