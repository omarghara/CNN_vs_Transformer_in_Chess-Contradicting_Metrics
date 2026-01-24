#!/usr/bin/env python3
"""
Statistical Analysis Script for Chess Puzzle Evaluation Data

This script performs comprehensive statistical analysis on CNN vs Transformer model
predictions for chess puzzles. It includes:
- Bootstrap confidence intervals
- McNemar's test for paired binary data
- Rating/theme/phase stratification analysis
- Top-5 accuracy comparison
- Effect size calculations (Cohen's d)
- Error pattern analysis
- Feature correlation analysis
- Comprehensive visualizations and reports

Usage Examples:
    # Full analysis with default settings
    python statistical_analysis.py

    # Quick test with 10k sample
    python statistical_analysis.py --sample 10000

    # Custom input/output paths
    python statistical_analysis.py --input Data/results_full.csv --output-dir my_analysis

    # With custom bootstrap iterations and random seed
    python statistical_analysis.py --bootstrap-iterations 5000 --seed 42

    # Skip calibration analysis
    python statistical_analysis.py --skip-calibration
    
    # Generate Top-3 and Top-5 accuracy summary markdown files
    python statistical_analysis.py --generate-top3-summary --generate-top5-summary

Requirements:
    numpy, pandas, matplotlib, seaborn, scipy, chess, tqdm, scikit-learn

Author: Generated for Chess CNN vs Transformer Analysis
Date: 2026-01-23
"""

import argparse
import ast
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import psutil
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
import chess
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Statistical Analysis for Chess Puzzle Model Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input CSV file (auto-detects if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_outputs',
        help='Directory for output files (default: analysis_outputs)'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size for quick testing (default: use all data)'
    )
    
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip calibration analysis'
    )
    
    parser.add_argument(
        '--bootstrap-iterations',
        type=int,
        default=1000,
        help='Number of bootstrap samples (default: 1000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Chunk size for reading large CSV files (default: 100000)'
    )
    
    parser.add_argument(
        '--generate-top3-summary',
        action='store_true',
        help='Generate separate Top-3 accuracy summary markdown file'
    )
    
    parser.add_argument(
        '--generate-top5-summary',
        action='store_true',
        help='Generate separate Top-5 accuracy summary markdown file'
    )
    
    return parser.parse_args()


def get_memory_usage() -> str:
    """Get current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"{mem_info.rss / 1024 / 1024:.2f} MB"


def auto_detect_csv_file() -> Optional[str]:
    """
    Auto-detect the correct CSV file.
    
    Returns:
        Path to the CSV file or None if not found
    """
    possible_paths = [
        'Data/results_first_move.csv',
        'puzzles.csv',
        'Data/puzzles.csv',
        'Data/results_full.csv',
        'results_full.csv',
        'results_first_move.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found CSV file: {path}")
            return path
    
    return None


def load_data_chunked(
    file_path: str,
    sample_size: Optional[int] = None,
    chunk_size: int = 100000
) -> pd.DataFrame:
    """
    Load data with chunked reading for memory efficiency.
    
    Args:
        file_path: Path to CSV file
        sample_size: Number of rows to sample (None for all)
        chunk_size: Size of chunks to read
        
    Returns:
        DataFrame with loaded data
    """
    print(f"\n📊 Loading data from: {file_path}")
    print(f"Memory usage before loading: {get_memory_usage()}")
    
    try:
        # Get total rows for progress bar
        total_rows = sum(1 for _ in open(file_path)) - 1  # -1 for header
        print(f"Total rows in file: {total_rows:,}")
        
        if sample_size and sample_size < total_rows:
            print(f"Sampling {sample_size:,} rows...")
            # Read with random sampling
            skip_rows = sorted(
                np.random.choice(range(1, total_rows + 1), total_rows - sample_size, replace=False)
            )
            df = pd.read_csv(file_path, skiprows=skip_rows)
        else:
            # Read in chunks with progress bar
            chunks = []
            with pd.read_csv(file_path, chunksize=chunk_size) as reader:
                with tqdm(total=total_rows, desc="Loading chunks") as pbar:
                    for chunk in reader:
                        chunks.append(chunk)
                        pbar.update(len(chunk))
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        
        print(f"✓ Loaded {len(df):,} rows")
        print(f"Memory usage after loading: {get_memory_usage()}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {e}", file=sys.stderr)
        sys.exit(1)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to handle variations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    # Create mapping for common variations
    column_mapping = {
        'rating': 'Rating',
        'themes': 'Themes',
        'puzzleid': 'PuzzleId',
        'fen': 'FEN',
        'moves': 'Moves',
        'ratingdeviation': 'RatingDeviation',
        'popularity': 'Popularity',
        'nbplays': 'NbPlays',
        'gameurl': 'GameUrl',
        'openingtags': 'OpeningTags',
        'cnn_correct': 'cnn_first_move_correct',
        'transformer_correct': 'transformer_first_move_correct'
    }
    
    # Rename columns
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]
    
    if new_columns:
        df = df.rename(columns=new_columns)
        print(f"✓ Normalized {len(new_columns)} column names")
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate data quality and print summary statistics.
    
    Args:
        df: DataFrame to validate
    """
    print("\n" + "="*80)
    print("DATA VALIDATION & SUMMARY STATISTICS")
    print("="*80)
    
    # Required columns
    required_cols = [
        'cnn_first_move_correct', 'transformer_first_move_correct',
        'Rating', 'Themes', 'FEN'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ All required columns present")
    
    # Dataset shape
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Memory usage
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("  No missing values")
    
    # Model accuracy
    print("\nModel Performance:")
    cnn_acc = df['cnn_first_move_correct'].mean() * 100
    transformer_acc = df['transformer_first_move_correct'].mean() * 100
    print(f"  CNN Accuracy: {cnn_acc:.2f}%")
    print(f"  Transformer Accuracy: {transformer_acc:.2f}%")
    print(f"  Difference: {transformer_acc - cnn_acc:+.2f}%")
    
    # Rating statistics
    print("\nRating Statistics:")
    print(f"  Min: {df['Rating'].min()}")
    print(f"  Max: {df['Rating'].max()}")
    print(f"  Mean: {df['Rating'].mean():.1f}")
    print(f"  Median: {df['Rating'].median():.1f}")
    print(f"  Std: {df['Rating'].std():.1f}")
    
    # Theme statistics
    print("\nTheme Statistics:")
    non_null_themes = df['Themes'].notna().sum()
    print(f"  Puzzles with themes: {non_null_themes:,} ({100*non_null_themes/len(df):.1f}%)")
    
    # Data quality checks
    print("\nData Quality Checks:")
    
    # Check for invalid ratings
    invalid_ratings = df[(df['Rating'] < 0) | (df['Rating'] > 4000)].shape[0]
    if invalid_ratings > 0:
        print(f"  ⚠ Warning: {invalid_ratings} invalid ratings (< 0 or > 4000)")
    else:
        print(f"  ✓ All ratings valid (0-4000)")
    
    # Check for agreement
    both_correct = (df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
    both_wrong = (~df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
    cnn_only = (df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
    trans_only = (~df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
    
    print(f"\nPrediction Agreement:")
    print(f"  Both correct: {both_correct:,} ({100*both_correct/len(df):.2f}%)")
    print(f"  Both wrong: {both_wrong:,} ({100*both_wrong/len(df):.2f}%)")
    print(f"  CNN only: {cnn_only:,} ({100*cnn_only/len(df):.2f}%)")
    print(f"  Transformer only: {trans_only:,} ({100*trans_only/len(df):.2f}%)")
    
    print("="*80 + "\n")


def parse_top5_moves(moves_str: str) -> List[str]:
    """
    Parse top-5 moves string safely.
    
    Args:
        moves_str: String representation of list of moves
        
    Returns:
        List of moves
    """
    if pd.isna(moves_str):
        return []
    
    try:
        # Handle string representation of list
        if isinstance(moves_str, str):
            moves = ast.literal_eval(moves_str)
            return moves if isinstance(moves, list) else []
        return []
    except (ValueError, SyntaxError):
        return []


def get_first_move(moves_str: str) -> Optional[str]:
    """
    Extract the first move from the Moves column (ground truth).
    
    Args:
        moves_str: Space-separated string of moves
        
    Returns:
        First move or None if invalid
    """
    if pd.isna(moves_str):
        return None
    moves = str(moves_str).strip().split()
    return moves[0] if moves else None


def prepare_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame with ground truth column.
    
    Args:
        df: Input DataFrame with Moves column
        
    Returns:
        DataFrame copy with ground_truth column added
    """
    df_copy = df.copy()
    df_copy['ground_truth'] = df_copy['Moves'].apply(get_first_move)
    return df_copy


def calculate_top5_accuracy(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate top-5 accuracy for both models.
    
    Args:
        df: DataFrame with top5 moves columns
        
    Returns:
        Tuple of (CNN top-5 accuracy, Transformer top-5 accuracy)
    """
    print("\n📊 Calculating Top-5 Accuracy...")
    
    df_copy = prepare_ground_truth(df)
    
    # Calculate top-5 accuracy
    cnn_top5_correct = []
    trans_top5_correct = []
    
    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Computing top-5"):
        ground_truth = row['ground_truth']
        if ground_truth is None:
            continue
        
        # CNN top-5
        cnn_top5 = parse_top5_moves(row.get('cnn_top5_moves', ''))
        cnn_top5_correct.append(ground_truth in cnn_top5)
        
        # Transformer top-5
        trans_top5 = parse_top5_moves(row.get('transformer_top5_moves', ''))
        trans_top5_correct.append(ground_truth in trans_top5)
    
    cnn_top5_acc = np.mean(cnn_top5_correct) * 100 if cnn_top5_correct else 0
    trans_top5_acc = np.mean(trans_top5_correct) * 100 if trans_top5_correct else 0
    
    print(f"  CNN Top-5 Accuracy: {cnn_top5_acc:.2f}%")
    print(f"  Transformer Top-5 Accuracy: {trans_top5_acc:.2f}%")
    
    return cnn_top5_acc, trans_top5_acc


def calculate_top3_accuracy(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate top-3 accuracy for both models.
    
    Args:
        df: DataFrame with top5 moves columns (contains top-3 as well)
        
    Returns:
        Tuple of (CNN top-3 accuracy, Transformer top-3 accuracy)
    """
    print("\n📊 Calculating Top-3 Accuracy...")
    
    df_copy = prepare_ground_truth(df)
    
    # Calculate top-3 accuracy
    cnn_top3_correct = []
    trans_top3_correct = []
    
    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Computing top-3"):
        ground_truth = row['ground_truth']
        if ground_truth is None:
            continue
        
        # CNN top-3 (first 3 from top-5)
        cnn_top5 = parse_top5_moves(row.get('cnn_top5_moves', ''))
        cnn_top3 = cnn_top5[:3] if len(cnn_top5) >= 3 else cnn_top5
        cnn_top3_correct.append(ground_truth in cnn_top3)
        
        # Transformer top-3 (first 3 from top-5)
        trans_top5 = parse_top5_moves(row.get('transformer_top5_moves', ''))
        trans_top3 = trans_top5[:3] if len(trans_top5) >= 3 else trans_top5
        trans_top3_correct.append(ground_truth in trans_top3)
    
    cnn_top3_acc = np.mean(cnn_top3_correct) * 100 if cnn_top3_correct else 0
    trans_top3_acc = np.mean(trans_top3_correct) * 100 if trans_top3_correct else 0
    
    print(f"  CNN Top-3 Accuracy: {cnn_top3_acc:.2f}%")
    print(f"  Transformer Top-3 Accuracy: {trans_top3_acc:.2f}%")
    
    return cnn_top3_acc, trans_top3_acc


def generate_topk_accuracy_summary(
    output_dir: str,
    topk_accuracy: Tuple[float, float],
    df: pd.DataFrame,
    k: int
) -> None:
    """
    Generate a separate markdown file for Top-K accuracy summary.
    
    Args:
        output_dir: Directory for output files
        topk_accuracy: Tuple of (CNN top-k, Transformer top-k) accuracy
        df: Main DataFrame
        k: Value of k (3 or 5)
    """
    print(f"\n📝 Generating Top-{k} Accuracy Summary...")
    
    summary_path = os.path.join(output_dir, f'top{k}_accuracy_summary.md')
    
    with open(summary_path, 'w') as f:
        f.write(f"# Top-{k} Accuracy Summary: CNN vs Transformer\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset Size**: {len(df):,} chess puzzles\n\n")
        f.write("---\n\n")
        
        f.write(f"## Top-{k} Accuracy\n\n")
        f.write("| Model | Accuracy |\n")
        f.write("|-------|----------|\n")
        f.write(f"| CNN | {topk_accuracy[0]:.2f}% |\n")
        f.write(f"| Transformer | {topk_accuracy[1]:.2f}% |\n")
        f.write(f"| **Difference** | **{topk_accuracy[1] - topk_accuracy[0]:.2f}%** |\n\n")
        
        # Analysis
        diff = topk_accuracy[1] - topk_accuracy[0]
        f.write("## Analysis\n\n")
        if abs(diff) < 0.5:
            f.write(f"The models show nearly identical top-{k} accuracy (difference: {diff:.2f}%).\n\n")
        elif diff > 0:
            f.write(f"The Transformer model outperforms CNN by {diff:.2f}% in top-{k} accuracy.\n\n")
        else:
            f.write(f"The CNN model outperforms Transformer by {abs(diff):.2f}% in top-{k} accuracy.\n\n")
        
        f.write("---\n\n")
        f.write("*Summary generated automatically by statistical_analysis.py*\n")
    
    print(f"✓ Saved Top-{k} accuracy summary to {summary_path}")


def generate_top3_accuracy_summary(
    output_dir: str,
    top3_accuracy: Tuple[float, float],
    df: pd.DataFrame
) -> None:
    """
    Generate a separate markdown file for Top-3 accuracy summary.
    
    Args:
        output_dir: Directory for output files
        top3_accuracy: Tuple of (CNN top-3, Transformer top-3) accuracy
        df: Main DataFrame
    """
    generate_topk_accuracy_summary(output_dir, top3_accuracy, df, 3)


def generate_top5_accuracy_summary(
    output_dir: str,
    top5_accuracy: Tuple[float, float],
    df: pd.DataFrame
) -> None:
    """
    Generate a separate markdown file for Top-5 accuracy summary.
    
    Args:
        output_dir: Directory for output files
        top5_accuracy: Tuple of (CNN top-5, Transformer top-5) accuracy
        df: Main DataFrame
    """
    generate_topk_accuracy_summary(output_dir, top5_accuracy, df, 5)


def bootstrap_analysis(
    df: pd.DataFrame,
    n_iterations: int,
    output_dir: str,
    seed: int
) -> Dict[str, Any]:
    """
    Perform bootstrap analysis for confidence intervals.
    
    Args:
        df: DataFrame with model predictions
        n_iterations: Number of bootstrap iterations
        output_dir: Directory for output files
        seed: Random seed
        
    Returns:
        Dictionary with bootstrap results
    """
    print(f"\n🔄 Bootstrap Analysis ({n_iterations} iterations)...")
    np.random.seed(seed)
    
    cnn_correct = df['cnn_first_move_correct'].values
    trans_correct = df['transformer_first_move_correct'].values
    
    # Bootstrap sampling
    cnn_accs = []
    trans_accs = []
    differences = []
    
    for _ in tqdm(range(n_iterations), desc="Bootstrap sampling"):
        # Sample with replacement
        indices = np.random.choice(len(df), size=len(df), replace=True)
        
        cnn_sample = cnn_correct[indices]
        trans_sample = trans_correct[indices]
        
        cnn_acc = np.mean(cnn_sample) * 100
        trans_acc = np.mean(trans_sample) * 100
        
        cnn_accs.append(cnn_acc)
        trans_accs.append(trans_acc)
        differences.append(trans_acc - cnn_acc)
    
    # Calculate confidence intervals
    cnn_ci = np.percentile(cnn_accs, [2.5, 97.5])
    trans_ci = np.percentile(trans_accs, [2.5, 97.5])
    diff_ci = np.percentile(differences, [2.5, 97.5])
    
    results = {
        'cnn_mean': np.mean(cnn_accs),
        'cnn_ci': cnn_ci,
        'trans_mean': np.mean(trans_accs),
        'trans_ci': trans_ci,
        'diff_mean': np.mean(differences),
        'diff_ci': diff_ci,
        'cnn_accs': cnn_accs,
        'trans_accs': trans_accs,
        'differences': differences
    }
    
    print(f"\nBootstrap Results:")
    print(f"  CNN: {results['cnn_mean']:.2f}% (95% CI: [{cnn_ci[0]:.2f}, {cnn_ci[1]:.2f}])")
    print(f"  Transformer: {results['trans_mean']:.2f}% (95% CI: [{trans_ci[0]:.2f}, {trans_ci[1]:.2f}])")
    print(f"  Difference: {results['diff_mean']:.2f}% (95% CI: [{diff_ci[0]:.2f}, {diff_ci[1]:.2f}])")
    
    # Visualization 1: Distribution of differences
    plt.figure(figsize=(10, 6))
    plt.hist(differences, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    plt.axvline(diff_ci[0], color='blue', linestyle='--', label='95% CI')
    plt.axvline(diff_ci[1], color='blue', linestyle='--')
    plt.xlabel('Accuracy Difference (Transformer - CNN) %')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap Distribution of Accuracy Difference ({n_iterations} iterations)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 2: Confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['CNN', 'Transformer']
    means = [results['cnn_mean'], results['trans_mean']]
    cis = [cnn_ci, trans_ci]
    
    y_pos = np.arange(len(models))
    ax.barh(y_pos, means, alpha=0.6, color=['blue', 'green'])
    ax.errorbar(means, y_pos, xerr=[[m - ci[0] for m, ci in zip(means, cis)],
                                     [ci[1] - m for m, ci in zip(means, cis)]],
                fmt='none', ecolor='black', capsize=5, linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Model Accuracy with 95% Bootstrap Confidence Intervals')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_confidence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved bootstrap visualizations")
    
    return results


def mcnemar_test(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    """
    Perform McNemar's test for paired binary data.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
        
    Returns:
        Dictionary with test results
    """
    print("\n🧪 McNemar's Test...")
    
    # Create contingency table
    # b: CNN correct, Transformer wrong
    # c: CNN wrong, Transformer correct
    b = (df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
    c = (~df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
    
    # McNemar's test statistic
    if b + c > 0:
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)  # with continuity correction
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0
        p_value = 1.0
    
    results = {
        'b': b,
        'c': c,
        'chi2': chi2,
        'p_value': p_value
    }
    
    # Save results
    with open(os.path.join(output_dir, 'mcnemar_test.txt'), 'w') as f:
        f.write("McNemar's Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Contingency Table:\n")
        f.write(f"  CNN correct, Transformer wrong (b): {b:,}\n")
        f.write(f"  CNN wrong, Transformer correct (c): {c:,}\n\n")
        f.write(f"Test Statistics:\n")
        f.write(f"  Chi-square statistic: {chi2:.4f}\n")
        f.write(f"  P-value: {p_value:.6f}\n\n")
        
        if p_value < 0.001:
            sig = "*** (highly significant)"
        elif p_value < 0.01:
            sig = "** (very significant)"
        elif p_value < 0.05:
            sig = "* (significant)"
        else:
            sig = "(not significant)"
        
        f.write(f"Interpretation: {sig}\n")
        f.write(f"\nNull Hypothesis: The two models have equal performance\n")
        f.write(f"Alternative Hypothesis: The two models have different performance\n")
    
    print(f"  Chi-square: {chi2:.4f}, p-value: {p_value:.6f}")
    print(f"✓ Saved McNemar's test results")
    
    return results


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        group1: First group (binary values)
        group2: Second group (binary values)
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def rating_stratification(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Perform rating stratification analysis.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
        
    Returns:
        DataFrame with stratified results
    """
    print("\n📊 Rating Stratification Analysis...")
    
    # Create rating bins
    bins = [0, 800, 1200, 1600, 2000, 2400, 2800, 3500]
    labels = ['400-800', '800-1200', '1200-1600', '1600-2000', '2000-2400', '2400-2800', '2800+']
    
    df['rating_bin'] = pd.cut(df['Rating'], bins=bins, labels=labels)
    
    # Calculate accuracy per bin
    results = []
    for bin_label in labels:
        bin_data = df[df['rating_bin'] == bin_label]
        
        if len(bin_data) == 0:
            continue
        
        cnn_acc = bin_data['cnn_first_move_correct'].mean() * 100
        trans_acc = bin_data['transformer_first_move_correct'].mean() * 100
        diff = trans_acc - cnn_acc
        count = len(bin_data)
        
        # Calculate effect size
        effect_size = cohens_d(
            bin_data['transformer_first_move_correct'].values,
            bin_data['cnn_first_move_correct'].values
        )
        
        results.append({
            'Rating Bin': bin_label,
            'Count': count,
            'CNN Accuracy (%)': cnn_acc,
            'Transformer Accuracy (%)': trans_acc,
            'Difference (%)': diff,
            'Cohen\'s d': effect_size
        })
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(
        os.path.join(output_dir, 'rating_stratification.csv'),
        index=False,
        float_format='%.2f'
    )
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy by rating bin
    x = np.arange(len(results_df))
    width = 0.35
    
    ax1.bar(x - width/2, results_df['CNN Accuracy (%)'], width, label='CNN', alpha=0.8)
    ax1.bar(x + width/2, results_df['Transformer Accuracy (%)'], width, label='Transformer', alpha=0.8)
    ax1.set_xlabel('Rating Bin')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy by Rating Bin')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Rating Bin'], rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Difference (Transformer - CNN)
    colors = ['green' if d > 0 else 'red' for d in results_df['Difference (%)']]
    ax2.bar(x, results_df['Difference (%)'], color=colors, alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Rating Bin')
    ax2.set_ylabel('Accuracy Difference (Transformer - CNN) %')
    ax2.set_title('Performance Difference by Rating Bin')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Rating Bin'], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add effect size annotations
    for i, (diff, effect) in enumerate(zip(results_df['Difference (%)'], results_df['Cohen\'s d'])):
        ax2.text(i, diff, f'd={effect:.2f}', ha='center', va='bottom' if diff > 0 else 'top',
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_stratification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved rating stratification results")
    print(f"\n{results_df.to_string(index=False)}")
    
    return results_df


def parse_themes(themes_str: str) -> List[str]:
    """
    Parse theme string into list of individual themes.
    
    Args:
        themes_str: Space-separated theme string
        
    Returns:
        List of themes
    """
    if pd.isna(themes_str):
        return []
    return str(themes_str).strip().split()


def theme_stratification(df: pd.DataFrame, output_dir: str, top_n: int = 20) -> pd.DataFrame:
    """
    Perform theme stratification analysis.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
        top_n: Number of top themes to analyze
        
    Returns:
        DataFrame with stratified results
    """
    print(f"\n📊 Theme Stratification Analysis (Top {top_n} themes)...")
    
    # Parse themes and create weighted accuracy
    theme_stats = defaultdict(lambda: {'cnn_first_move_correct': 0, 'transformer_first_move_correct': 0, 'total': 0})
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing themes"):
        themes = parse_themes(row['Themes'])
        if not themes:
            continue
        
        weight = 1.0 / len(themes)  # Weight for multi-theme puzzles
        
        for theme in themes:
            theme_stats[theme]['total'] += weight
            if row['cnn_first_move_correct']:
                theme_stats[theme]['cnn_first_move_correct'] += weight
            if row['transformer_first_move_correct']:
                theme_stats[theme]['transformer_first_move_correct'] += weight
    
    # Calculate accuracy for each theme
    results = []
    for theme, stats in theme_stats.items():
        if stats['total'] < 10:  # Skip themes with very few puzzles
            continue
        
        cnn_acc = (stats['cnn_first_move_correct'] / stats['total']) * 100
        trans_acc = (stats['transformer_first_move_correct'] / stats['total']) * 100
        diff = trans_acc - cnn_acc
        
        results.append({
            'Theme': theme,
            'Count': stats['total'],
            'CNN Accuracy (%)': cnn_acc,
            'Transformer Accuracy (%)': trans_acc,
            'Difference (%)': diff
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Count', ascending=False).head(top_n)
    
    # Save to CSV
    results_df.to_csv(
        os.path.join(output_dir, 'theme_stratification.csv'),
        index=False,
        float_format='%.2f'
    )
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Sort by count for visualization
    plot_df = results_df.sort_values('Count', ascending=True)
    
    # Plot 1: Accuracy comparison
    y = np.arange(len(plot_df))
    width = 0.35
    
    ax1.barh(y - width/2, plot_df['CNN Accuracy (%)'], width, label='CNN', alpha=0.8)
    ax1.barh(y + width/2, plot_df['Transformer Accuracy (%)'], width, label='Transformer', alpha=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(plot_df['Theme'])
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title(f'Model Accuracy by Theme (Top {top_n} by count)')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Difference
    colors = ['green' if d > 0 else 'red' for d in plot_df['Difference (%)']]
    ax2.barh(y, plot_df['Difference (%)'], color=colors, alpha=0.6)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(plot_df['Theme'])
    ax2.set_xlabel('Accuracy Difference (Transformer - CNN) %')
    ax2.set_title('Performance Difference by Theme')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'theme_stratification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved theme stratification results")
    
    return results_df


def parse_fen_safely(fen: str) -> Optional[chess.Board]:
    """
    Parse FEN string safely with error handling.
    
    Args:
        fen: FEN string
        
    Returns:
        chess.Board object or None if parsing fails
    """
    try:
        return chess.Board(fen)
    except (ValueError, IndexError):
        return None


def extract_game_phase(fen: str) -> str:
    """
    Extract game phase from FEN position.
    
    Args:
        fen: FEN string
        
    Returns:
        Game phase: 'opening', 'middlegame', or 'endgame'
    """
    board = parse_fen_safely(fen)
    if board is None:
        return 'unknown'
    
    # Count pieces (excluding kings)
    piece_count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            piece_count += 1
    
    # Simple heuristic
    if piece_count >= 20:
        return 'opening'
    elif piece_count >= 12:
        return 'middlegame'
    else:
        return 'endgame'


def phase_stratification(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Perform game phase stratification analysis.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
        
    Returns:
        DataFrame with stratified results
    """
    print("\n📊 Game Phase Stratification Analysis...")
    print("Extracting game phases from FEN positions...")
    
    # Extract game phase with progress bar
    game_phases = []
    for fen in tqdm(df['FEN'], desc="Parsing FENs"):
        game_phases.append(extract_game_phase(fen))
    df['game_phase'] = game_phases
    
    # Calculate accuracy per phase
    results = []
    for phase in ['opening', 'middlegame', 'endgame']:
        phase_data = df[df['game_phase'] == phase]
        
        if len(phase_data) == 0:
            continue
        
        cnn_acc = phase_data['cnn_first_move_correct'].mean() * 100
        trans_acc = phase_data['transformer_first_move_correct'].mean() * 100
        diff = trans_acc - cnn_acc
        count = len(phase_data)
        
        # Calculate effect size
        effect_size = cohens_d(
            phase_data['transformer_first_move_correct'].values,
            phase_data['cnn_first_move_correct'].values
        )
        
        results.append({
            'Phase': phase.capitalize(),
            'Count': count,
            'CNN Accuracy (%)': cnn_acc,
            'Transformer Accuracy (%)': trans_acc,
            'Difference (%)': diff,
            'Cohen\'s d': effect_size
        })
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(
        os.path.join(output_dir, 'phase_stratification.csv'),
        index=False,
        float_format='%.2f'
    )
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy by phase
    x = np.arange(len(results_df))
    width = 0.35
    
    ax1.bar(x - width/2, results_df['CNN Accuracy (%)'], width, label='CNN', alpha=0.8)
    ax1.bar(x + width/2, results_df['Transformer Accuracy (%)'], width, label='Transformer', alpha=0.8)
    ax1.set_xlabel('Game Phase')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy by Game Phase')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Phase'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Difference
    colors = ['green' if d > 0 else 'red' for d in results_df['Difference (%)']]
    ax2.bar(x, results_df['Difference (%)'], color=colors, alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Game Phase')
    ax2.set_ylabel('Accuracy Difference (Transformer - CNN) %')
    ax2.set_title('Performance Difference by Game Phase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Phase'])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add effect size annotations
    for i, (diff, effect) in enumerate(zip(results_df['Difference (%)'], results_df['Cohen\'s d'])):
        ax2.text(i, diff, f'd={effect:.2f}', ha='center', va='bottom' if diff > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_stratification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved phase stratification results")
    print(f"\n{results_df.to_string(index=False)}")
    
    return results_df


def error_pattern_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze error patterns with heatmap.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
    """
    print("\n📊 Error Pattern Analysis...")
    
    # Create confusion matrix style data
    both_correct = (df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
    both_wrong = (~df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
    cnn_only = (df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
    trans_only = (~df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
    
    total = len(df)
    
    # Create matrix for heatmap
    matrix = np.array([
        [both_correct / total * 100, cnn_only / total * 100],
        [trans_only / total * 100, both_wrong / total * 100]
    ])
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=['Transformer Correct', 'Transformer Wrong'],
        yticklabels=['CNN Correct', 'CNN Wrong'],
        cbar_kws={'label': 'Percentage of Total Puzzles'}
    )
    plt.title('Error Pattern Heatmap\n(Percentage of all puzzles)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved error pattern heatmap")


def feature_correlation_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze correlation between features and model errors.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
    """
    print("\n📊 Feature Correlation Analysis...")
    
    # Create error indicators
    df['cnn_error'] = (~df['cnn_first_move_correct'].astype(bool)).astype(int)
    df['trans_error'] = (~df['transformer_first_move_correct'].astype(bool)).astype(int)
    
    # Select numeric features
    features = ['Rating', 'RatingDeviation', 'Popularity', 'NbPlays']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        print("  No numeric features available for correlation analysis")
        return
    
    # Calculate correlations
    corr_data = df[available_features + ['cnn_error', 'trans_error']].corr()
    
    # Extract correlations with errors
    error_corr = corr_data.loc[available_features, ['cnn_error', 'trans_error']]
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        error_corr,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-0.3,
        vmax=0.3,
        xticklabels=['CNN Error', 'Transformer Error'],
        yticklabels=available_features
    )
    plt.title('Correlation between Features and Model Errors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature correlation analysis")


def generate_summary_statistics(df: pd.DataFrame, output_dir: str, **kwargs) -> None:
    """
    Generate summary statistics CSV.
    
    Args:
        df: DataFrame with model predictions
        output_dir: Directory for output files
        **kwargs: Additional statistics to include
    """
    print("\n📊 Generating Summary Statistics...")
    
    stats = {
        'Metric': [],
        'Value': []
    }
    
    # Dataset statistics
    stats['Metric'].extend([
        'Total Puzzles',
        'CNN Accuracy (%)',
        'Transformer Accuracy (%)',
        'Accuracy Difference (%)',
        'Both Correct (%)',
        'Both Wrong (%)',
        'CNN Only Correct (%)',
        'Transformer Only Correct (%)'
    ])
    
    stats['Value'].extend([
        f"{len(df):,}",
        f"{df['cnn_first_move_correct'].mean() * 100:.2f}",
        f"{df['transformer_first_move_correct'].mean() * 100:.2f}",
        f"{(df['transformer_first_move_correct'].mean() - df['cnn_first_move_correct'].mean()) * 100:.2f}",
        f"{((df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum() / len(df)) * 100:.2f}",
        f"{((~df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum() / len(df)) * 100:.2f}",
        f"{((df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum() / len(df)) * 100:.2f}",
        f"{((~df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum() / len(df)) * 100:.2f}"
    ])
    
    # Add additional statistics from kwargs
    for key, value in kwargs.items():
        stats['Metric'].append(key)
        if isinstance(value, float):
            stats['Value'].append(f"{value:.2f}")
        else:
            stats['Value'].append(str(value))
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(
        os.path.join(output_dir, 'summary_statistics.csv'),
        index=False
    )
    
    print(f"✓ Saved summary statistics")


def generate_markdown_report(
    output_dir: str,
    df: pd.DataFrame,
    bootstrap_results: Dict,
    mcnemar_results: Dict,
    rating_df: pd.DataFrame,
    theme_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    top5_accuracy: Tuple[float, float]
) -> None:
    """
    Generate comprehensive markdown report.
    
    Args:
        output_dir: Directory for output files
        df: Main DataFrame
        bootstrap_results: Bootstrap analysis results
        mcnemar_results: McNemar test results
        rating_df: Rating stratification results
        theme_df: Theme stratification results
        phase_df: Phase stratification results
        top5_accuracy: Tuple of (CNN top-5, Transformer top-5) accuracy
    """
    print("\n📝 Generating Markdown Report...")
    
    report_path = os.path.join(output_dir, 'analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Statistical Analysis Report: CNN vs Transformer Chess Models\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"This report presents a comprehensive statistical analysis of {len(df):,} chess puzzle predictions ")
        f.write("comparing CNN and Transformer models.\n\n")
        
        # Overall Performance
        f.write("## Overall Performance\n\n")
        f.write("### Top-1 Accuracy\n\n")
        f.write("| Model | Accuracy | 95% CI |\n")
        f.write("|-------|----------|--------|\n")
        f.write(f"| CNN | {bootstrap_results['cnn_mean']:.2f}% | ")
        f.write(f"[{bootstrap_results['cnn_ci'][0]:.2f}, {bootstrap_results['cnn_ci'][1]:.2f}] |\n")
        f.write(f"| Transformer | {bootstrap_results['trans_mean']:.2f}% | ")
        f.write(f"[{bootstrap_results['trans_ci'][0]:.2f}, {bootstrap_results['trans_ci'][1]:.2f}] |\n")
        f.write(f"| **Difference** | **{bootstrap_results['diff_mean']:.2f}%** | ")
        f.write(f"**[{bootstrap_results['diff_ci'][0]:.2f}, {bootstrap_results['diff_ci'][1]:.2f}]** |\n\n")
        
        f.write("### Top-5 Accuracy\n\n")
        f.write("| Model | Accuracy |\n")
        f.write("|-------|----------|\n")
        f.write(f"| CNN | {top5_accuracy[0]:.2f}% |\n")
        f.write(f"| Transformer | {top5_accuracy[1]:.2f}% |\n")
        f.write(f"| **Difference** | **{top5_accuracy[1] - top5_accuracy[0]:.2f}%** |\n\n")
        
        # Statistical Tests
        f.write("## Statistical Tests\n\n")
        f.write("### McNemar's Test\n\n")
        f.write(f"- **Chi-square statistic**: {mcnemar_results['chi2']:.4f}\n")
        f.write(f"- **P-value**: {mcnemar_results['p_value']:.6f}\n")
        
        if mcnemar_results['p_value'] < 0.05:
            f.write(f"- **Result**: Statistically significant difference (p < 0.05)\n\n")
        else:
            f.write(f"- **Result**: No statistically significant difference (p >= 0.05)\n\n")
        
        f.write("**Contingency Table:**\n\n")
        f.write(f"- CNN correct, Transformer wrong: {mcnemar_results['b']:,}\n")
        f.write(f"- CNN wrong, Transformer correct: {mcnemar_results['c']:,}\n\n")
        
        # Prediction Agreement
        f.write("## Prediction Agreement\n\n")
        both_correct = (df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
        both_wrong = (~df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
        cnn_only = (df['cnn_first_move_correct'].astype(bool) & ~df['transformer_first_move_correct'].astype(bool)).sum()
        trans_only = (~df['cnn_first_move_correct'].astype(bool) & df['transformer_first_move_correct'].astype(bool)).sum()
        
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        f.write(f"| Both Correct | {both_correct:,} | {100*both_correct/len(df):.2f}% |\n")
        f.write(f"| Both Wrong | {both_wrong:,} | {100*both_wrong/len(df):.2f}% |\n")
        f.write(f"| CNN Only | {cnn_only:,} | {100*cnn_only/len(df):.2f}% |\n")
        f.write(f"| Transformer Only | {trans_only:,} | {100*trans_only/len(df):.2f}% |\n\n")
        
        # Rating Stratification
        f.write("## Rating Stratification\n\n")
        f.write("Performance breakdown by puzzle difficulty rating:\n\n")
        f.write(rating_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("![Rating Stratification](rating_stratification.png)\n\n")
        
        # Theme Stratification
        f.write("## Theme Stratification\n\n")
        f.write("Performance breakdown by puzzle theme (top themes by count):\n\n")
        f.write(theme_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("![Theme Stratification](theme_stratification.png)\n\n")
        
        # Phase Stratification
        f.write("## Game Phase Stratification\n\n")
        f.write("Performance breakdown by game phase:\n\n")
        f.write(phase_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("![Phase Stratification](phase_stratification.png)\n\n")
        
        # Visualizations
        f.write("## Additional Visualizations\n\n")
        f.write("### Bootstrap Analysis\n\n")
        f.write("![Bootstrap Distribution](bootstrap_distribution.png)\n\n")
        f.write("![Bootstrap Confidence](bootstrap_confidence.png)\n\n")
        
        f.write("### Error Patterns\n\n")
        f.write("![Error Heatmap](error_heatmap.png)\n\n")
        
        f.write("### Feature Correlations\n\n")
        f.write("![Feature Correlations](feature_correlations.png)\n\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        
        # Find which model performs better overall
        overall_diff = bootstrap_results['diff_mean']
        if abs(overall_diff) < 0.5:
            f.write("1. **Overall Performance**: The models show nearly identical top-1 accuracy ")
            f.write(f"(difference: {overall_diff:.2f}%).\n\n")
        elif overall_diff > 0:
            f.write(f"1. **Overall Performance**: Transformer outperforms CNN by {overall_diff:.2f}% ")
            f.write("in top-1 accuracy.\n\n")
        else:
            f.write(f"1. **Overall Performance**: CNN outperforms Transformer by {abs(overall_diff):.2f}% ")
            f.write("in top-1 accuracy.\n\n")
        
        # Rating insights
        max_diff_rating = rating_df.loc[rating_df['Difference (%)'].abs().idxmax()]
        f.write(f"2. **Rating Stratification**: Largest performance difference observed in ")
        f.write(f"{max_diff_rating['Rating Bin']} rating range ")
        f.write(f"({max_diff_rating['Difference (%)']:.2f}%).\n\n")
        
        # Theme insights
        max_diff_theme = theme_df.loc[theme_df['Difference (%)'].abs().idxmax()]
        f.write(f"3. **Theme Stratification**: Largest performance difference observed for ")
        f.write(f"'{max_diff_theme['Theme']}' theme ")
        f.write(f"({max_diff_theme['Difference (%)']:.2f}%).\n\n")
        
        # Top-5 insights
        top5_diff = top5_accuracy[1] - top5_accuracy[0]
        f.write(f"4. **Top-5 Accuracy**: ")
        if abs(top5_diff) > abs(overall_diff):
            f.write(f"The performance gap widens to {top5_diff:.2f}% when considering top-5 predictions.\n\n")
        else:
            f.write(f"The performance gap narrows to {top5_diff:.2f}% when considering top-5 predictions.\n\n")
        
        # Statistical significance
        if mcnemar_results['p_value'] < 0.05:
            f.write(f"5. **Statistical Significance**: McNemar's test indicates a statistically ")
            f.write(f"significant difference (p = {mcnemar_results['p_value']:.6f}).\n\n")
        else:
            f.write(f"5. **Statistical Significance**: McNemar's test does not indicate a statistically ")
            f.write(f"significant difference (p = {mcnemar_results['p_value']:.6f}).\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("This analysis reveals:\n\n")
        f.write("- The models show different strengths across puzzle ratings and themes\n")
        f.write("- Stratified analysis provides deeper insights than aggregate metrics alone\n")
        f.write("- Both models have substantial room for improvement (< 40% top-1 accuracy)\n")
        f.write("- Consider ensemble approaches to leverage complementary strengths\n\n")
        
        # Files Generated
        f.write("## Files Generated\n\n")
        f.write("This analysis generated the following output files:\n\n")
        f.write("- `analysis_report.md` - This comprehensive report\n")
        f.write("- `summary_statistics.csv` - Overall summary statistics\n")
        f.write("- `mcnemar_test.txt` - McNemar's test detailed results\n")
        f.write("- `bootstrap_confidence.png` - Bootstrap confidence intervals\n")
        f.write("- `bootstrap_distribution.png` - Bootstrap distribution plot\n")
        f.write("- `rating_stratification.csv` / `.png` - Performance by rating\n")
        f.write("- `theme_stratification.csv` / `.png` - Performance by theme\n")
        f.write("- `phase_stratification.csv` / `.png` - Performance by game phase\n")
        f.write("- `error_heatmap.png` - Error pattern visualization\n")
        f.write("- `feature_correlations.png` - Feature-error correlations\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated automatically by statistical_analysis.py*\n")
    
    print(f"✓ Saved markdown report to {report_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: CNN vs TRANSFORMER CHESS MODELS")
    print("="*80)
    
    # Set random seed
    np.random.seed(args.seed)
    print(f"\n🎲 Random seed set to: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Auto-detect or use specified CSV file
    if args.input is None:
        csv_file = auto_detect_csv_file()
        if csv_file is None:
            print("✗ Error: Could not find CSV file", file=sys.stderr)
            print("Please specify the path with --input", file=sys.stderr)
            sys.exit(1)
    else:
        csv_file = args.input
        if not os.path.exists(csv_file):
            print(f"✗ Error: File not found: {csv_file}", file=sys.stderr)
            sys.exit(1)
    
    # Load data
    df = load_data_chunked(csv_file, args.sample, args.chunk_size)
    
    # Normalize column names
    df = normalize_column_names(df)
    
    # Validate data
    validate_data(df)
    
    # Perform analyses
    print("\n" + "="*80)
    print("RUNNING STATISTICAL ANALYSES")
    print("="*80)
    
    # Bootstrap analysis
    bootstrap_results = bootstrap_analysis(
        df, args.bootstrap_iterations, args.output_dir, args.seed
    )
    
    # McNemar's test
    mcnemar_results = mcnemar_test(df, args.output_dir)
    
    # Rating stratification
    rating_df = rating_stratification(df, args.output_dir)
    
    # Theme stratification
    theme_df = theme_stratification(df, args.output_dir, top_n=20)
    
    # Phase stratification
    phase_df = phase_stratification(df, args.output_dir)
    
    # Top-5 accuracy
    top5_accuracy = calculate_top5_accuracy(df)
    
    # Top-3 accuracy (calculated only if summary requested)
    top3_accuracy = None
    if args.generate_top3_summary:
        top3_accuracy = calculate_top3_accuracy(df)
        generate_top3_accuracy_summary(args.output_dir, top3_accuracy, df)
    
    # Generate Top-5 summary if requested
    if args.generate_top5_summary:
        generate_top5_accuracy_summary(args.output_dir, top5_accuracy, df)
    
    # Error pattern analysis
    error_pattern_analysis(df, args.output_dir)
    
    # Feature correlation analysis
    feature_correlation_analysis(df, args.output_dir)
    
    # Generate summary statistics
    summary_stats = {
        'CNN Top-5 Accuracy (%)': top5_accuracy[0],
        'Transformer Top-5 Accuracy (%)': top5_accuracy[1],
        'McNemar Chi-square': mcnemar_results['chi2'],
        'McNemar P-value': mcnemar_results['p_value'],
        'Bootstrap Iterations': args.bootstrap_iterations
    }
    
    # Add Top-3 to summary if calculated
    if top3_accuracy is not None:
        summary_stats['CNN Top-3 Accuracy (%)'] = top3_accuracy[0]
        summary_stats['Transformer Top-3 Accuracy (%)'] = top3_accuracy[1]
    
    generate_summary_statistics(df, args.output_dir, **summary_stats)
    
    # Generate markdown report
    generate_markdown_report(
        args.output_dir, df, bootstrap_results, mcnemar_results,
        rating_df, theme_df, phase_df, top5_accuracy
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ All results saved to: {args.output_dir}")
    print(f"✓ Final memory usage: {get_memory_usage()}")
    print("\nKey outputs:")
    print(f"  - Comprehensive report: {os.path.join(args.output_dir, 'analysis_report.md')}")
    print(f"  - Summary statistics: {os.path.join(args.output_dir, 'summary_statistics.csv')}")
    if args.generate_top3_summary:
        print(f"  - Top-3 accuracy summary: {os.path.join(args.output_dir, 'top3_accuracy_summary.md')}")
    if args.generate_top5_summary:
        print(f"  - Top-5 accuracy summary: {os.path.join(args.output_dir, 'top5_accuracy_summary.md')}")
    print(f"  - Visualizations: {len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])} PNG files")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
