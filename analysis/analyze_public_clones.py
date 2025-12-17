#!/usr/bin/env python3
"""
Public Clone Analysis Script
Compares v5 enrichment vs v9 Fisher exact test approaches
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
TRAIN_ROOT = Path('./data/train_datasets/train_datasets')
OUTPUT_DIR = Path('./analysis_results')
OUTPUT_DIR.mkdir(exist_ok=True)


def mine_v5_style(dataset_path, max_files=20, min_freq=0.15, enrichment=5.0):
    """V5 enrichment ratio approach"""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()[:max_files]
    neg_files = meta[meta['label_positive'] == False]['filename'].tolist()[:max_files]

    pos_c = Counter()
    neg_c = Counter()

    for f in pos_files:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            pos_c.update(df['junction_aa'].dropna().unique())
        except:
            pass

    for f in neg_files:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            neg_c.update(df['junction_aa'].dropna().unique())
        except:
            pass

    n_pos = max(1, len(pos_files))
    n_neg = max(1, len(neg_files))

    scored = []
    for seq, count in pos_c.items():
        pf = count / n_pos
        nf = neg_c.get(seq, 0) / n_neg
        if pf >= min_freq and pf > nf * enrichment:
            score = float(np.log((pf + 1e-6) / (nf + 1e-6)))
            scored.append({'seq': seq, 'score': score, 'method': 'v5_enrichment',
                          'pos_freq': pf, 'neg_freq': nf})

    return scored


def mine_v9_style(dataset_path, max_files=40, p_threshold=0.01, min_freq=0.10):
    """V9 Fisher exact test approach"""
    meta = pd.read_csv(dataset_path / 'metadata.csv')
    pos_files = meta[meta['label_positive'] == True]['filename'].tolist()[:max_files]
    neg_files = meta[meta['label_positive'] == False]['filename'].tolist()[:max_files]

    n_pos = len(pos_files)
    n_neg = len(neg_files)

    pos_counts = Counter()
    neg_counts = Counter()

    for f in pos_files:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            unique_seqs = set(df['junction_aa'].dropna().astype(str).unique())
            pos_counts.update(unique_seqs)
        except:
            pass

    for f in neg_files:
        try:
            df = pd.read_csv(dataset_path / f, sep='\t', usecols=['junction_aa'])
            unique_seqs = set(df['junction_aa'].dropna().astype(str).unique())
            neg_counts.update(unique_seqs)
        except:
            pass

    all_seqs = set(pos_counts.keys()) | set(neg_counts.keys())
    significant = []

    for seq in all_seqs:
        if len(seq) < 8:
            continue

        pos_c = pos_counts.get(seq, 0)
        neg_c = neg_counts.get(seq, 0)

        pos_freq = pos_c / n_pos if n_pos > 0 else 0
        neg_freq = neg_c / n_neg if n_neg > 0 else 0

        if pos_freq < min_freq and neg_freq < min_freq:
            continue

        table = [[pos_c, n_pos - pos_c],
                 [neg_c, n_neg - neg_c]]
        try:
            odds_ratio, p_value = fisher_exact(table)
        except:
            continue

        if p_value < p_threshold:
            log_odds = np.log(odds_ratio + 1e-10)
            score = -np.log10(p_value + 1e-10) * np.sign(log_odds)

            significant.append({
                'seq': seq, 'score': score, 'method': 'v9_fisher',
                'p_value': p_value, 'odds_ratio': odds_ratio,
                'pos_freq': pos_freq, 'neg_freq': neg_freq,
                'direction': 'positive' if odds_ratio > 1 else 'negative'
            })

    return significant


def analyze_dataset(ds_name, ds_path):
    """Compare v5 and v9 for a single dataset"""
    print(f"\nAnalyzing {ds_name}...")

    # Mine with both methods
    v5_results = mine_v5_style(ds_path)
    v9_results = mine_v9_style(ds_path)

    print(f"  v5: Found {len(v5_results)} sequences")
    print(f"  v9: Found {len(v9_results)} sequences")

    # Convert to dataframes
    v5_df = pd.DataFrame(v5_results) if v5_results else pd.DataFrame()
    v9_df = pd.DataFrame(v9_results) if v9_results else pd.DataFrame()

    # Calculate overlap
    if not v5_df.empty and not v9_df.empty:
        v5_seqs = set(v5_df['seq'])
        v9_seqs = set(v9_df['seq'])
        overlap = len(v5_seqs & v9_seqs)
        jaccard = overlap / len(v5_seqs | v9_seqs) if (v5_seqs | v9_seqs) else 0
        print(f"  Overlap: {overlap} sequences (Jaccard: {jaccard:.3f})")

        # Sequences unique to each method
        v5_only = len(v5_seqs - v9_seqs)
        v9_only = len(v9_seqs - v5_seqs)
        print(f"  v5 only: {v5_only}, v9 only: {v9_only}")
    else:
        overlap = jaccard = v5_only = v9_only = 0

    return {
        'dataset': ds_name,
        'v5_count': len(v5_results),
        'v9_count': len(v9_results),
        'overlap': overlap,
        'jaccard': jaccard,
        'v5_only': v5_only,
        'v9_only': v9_only,
        'v5_df': v5_df,
        'v9_df': v9_df
    }


def plot_comparison(results_list):
    """Create comparison plots"""

    # Summary statistics
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['v5_df', 'v9_df']}
        for r in results_list
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Public clone counts
    ax = axes[0, 0]
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary['v5_count'], width, label='v5 Enrichment', alpha=0.7)
    ax.bar(x + width/2, summary['v9_count'], width, label='v9 Fisher', alpha=0.7)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Public Clones')
    ax.set_title('Public Clone Count Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['dataset'].str.replace('train_dataset_', 'DS'))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Overlap analysis
    ax = axes[0, 1]
    ax.bar(x, summary['overlap'], alpha=0.7, color='green')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Overlap Count')
    ax.set_title('Method Overlap (Shared Sequences)')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['dataset'].str.replace('train_dataset_', 'DS'))
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Jaccard similarity
    ax = axes[1, 0]
    ax.bar(x, summary['jaccard'], alpha=0.7, color='purple')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('Method Agreement (Jaccard Index)')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['dataset'].str.replace('train_dataset_', 'DS'))
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Unique sequences per method
    ax = axes[1, 1]
    ax.bar(x - width/2, summary['v5_only'], width, label='v5 unique', alpha=0.7)
    ax.bar(x + width/2, summary['v9_only'], width, label='v9 unique', alpha=0.7)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Unique Sequence Count')
    ax.set_title('Method-Specific Discoveries')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['dataset'].str.replace('train_dataset_', 'DS'))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'public_clone_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {OUTPUT_DIR / 'public_clone_comparison.png'}")

    # Save summary table
    summary.to_csv(OUTPUT_DIR / 'public_clone_summary.csv', index=False)
    print(f"Saved summary table to {OUTPUT_DIR / 'public_clone_summary.csv'}")


def plot_score_distributions(results_list):
    """Plot score distributions for each method"""

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, result in enumerate(results_list):
        if idx >= 8:
            break

        ax = axes[idx]
        v5_df = result['v5_df']
        v9_df = result['v9_df']

        if not v5_df.empty:
            ax.hist(v5_df['score'], bins=30, alpha=0.5, label='v5', color='blue')
        if not v9_df.empty:
            ax.hist(v9_df['score'], bins=30, alpha=0.5, label='v9', color='red')

        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title(result['dataset'].replace('train_dataset_', 'DS'))
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'score_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved score distributions to {OUTPUT_DIR / 'score_distributions.png'}")


def main():
    print("="*70)
    print("Public Clone Mining Analysis: v5 vs v9")
    print("="*70)

    if not TRAIN_ROOT.exists():
        print(f"ERROR: Training data not found at {TRAIN_ROOT}")
        print("Please ensure data is downloaded to ./data/train_datasets/train_datasets/")
        return

    # Find all training datasets
    datasets = sorted([d for d in TRAIN_ROOT.glob('train_dataset_*') if d.is_dir()])

    if not datasets:
        print("ERROR: No training datasets found")
        return

    print(f"Found {len(datasets)} training datasets")

    # Analyze each dataset
    results = []
    for ds in datasets:
        try:
            result = analyze_dataset(ds.name, ds)
            results.append(result)
        except Exception as e:
            print(f"  ERROR analyzing {ds.name}: {e}")
            continue

    if not results:
        print("ERROR: No successful analyses")
        return

    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    plot_score_distributions(results)

    # Print detailed summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  v5: {result['v5_count']} sequences")
        print(f"  v9: {result['v9_count']} sequences")
        print(f"  Overlap: {result['overlap']} (Jaccard: {result['jaccard']:.3f})")
        print(f"  v5 unique: {result['v5_only']}, v9 unique: {result['v9_only']}")

    # Overall statistics
    total_v5 = sum(r['v5_count'] for r in results)
    total_v9 = sum(r['v9_count'] for r in results)
    avg_jaccard = np.mean([r['jaccard'] for r in results if r['jaccard'] > 0])

    print(f"\nOverall:")
    print(f"  Total v5 sequences: {total_v5}")
    print(f"  Total v9 sequences: {total_v9}")
    print(f"  Average Jaccard similarity: {avg_jaccard:.3f}")
    print(f"  v9 finds {(total_v9/total_v5 - 1)*100:.1f}% more sequences than v5")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
1. v9 Fisher test typically finds MORE sequences than v5 enrichment
   - Reason: Lower min_freq threshold (0.10 vs 0.15)
   - Reason: Larger sample size (40 vs 20 files)
   - Reason: Statistical test catches weaker signals

2. Jaccard similarity indicates method agreement:
   - > 0.7: High agreement (methods find similar sequences)
   - 0.4-0.7: Moderate agreement (some differences)
   - < 0.4: Low agreement (methods capture different signals)

3. v5-unique sequences: High enrichment but not statistically significant
   v9-unique sequences: Statistically significant but lower enrichment

4. Biological interpretation:
   - v5 is stricter (fewer false positives, more false negatives)
   - v9 is more sensitive (fewer false negatives, slightly more false positives)
   - v9 preferred for machine learning (more features, statistical rigor)
    """)

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("\nRecommendation: Use v9 Fisher exact test for production")


if __name__ == '__main__':
    main()
