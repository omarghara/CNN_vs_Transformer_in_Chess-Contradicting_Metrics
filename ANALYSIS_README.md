# Statistical Analysis Script Usage Guide

## Overview

The `statistical_analysis.py` script provides comprehensive statistical analysis for comparing CNN and Transformer models on chess puzzle predictions. It generates detailed reports, visualizations, and statistical tests.

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy, pandas, matplotlib, seaborn, scipy
- python-chess
- tqdm (progress bars)
- scikit-learn, statsmodels
- tabulate (for markdown tables)
- psutil (memory monitoring)

## Quick Start

### 1. Full Analysis (All Data)

```bash
python statistical_analysis.py
```

This will:
- Auto-detect the CSV file in `Data/results_full.csv`
- Analyze all ~5.6M puzzles
- Save outputs to `analysis_outputs/` directory
- Takes approximately 15-30 minutes depending on hardware

### 2. Quick Test (Sampled Data)

```bash
python statistical_analysis.py --sample 10000
```

Recommended for testing - analyzes 10,000 random puzzles (takes ~2-3 minutes).

### 3. Custom Configuration

```bash
python statistical_analysis.py \
  --input Data/results_full.csv \
  --output-dir my_analysis \
  --sample 50000 \
  --bootstrap-iterations 5000 \
  --seed 42
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input CSV file | Auto-detect |
| `--output-dir` | Directory for output files | `analysis_outputs` |
| `--sample` | Sample size for testing | All data |
| `--skip-calibration` | Skip calibration analysis | False |
| `--bootstrap-iterations` | Number of bootstrap samples | 1000 |
| `--seed` | Random seed for reproducibility | 42 |
| `--chunk-size` | Chunk size for reading CSV | 100000 |

## Expected Input Format

The script expects a CSV file with the following columns:

- `PuzzleId`: Unique puzzle identifier
- `FEN`: Chess position in Forsyth-Edwards Notation
- `Moves`: Ground truth solution moves
- `Rating`: Puzzle difficulty rating (e.g., 399-3395)
- `RatingDeviation`: Rating uncertainty
- `Popularity`: Puzzle popularity score
- `NbPlays`: Number of plays
- `Themes`: Space-separated puzzle themes
- `GameUrl`: URL to original game
- `OpeningTags`: Chess opening tags
- `cnn_predicted_move`: CNN's top-1 prediction (UCI format)
- `transformer_predicted_move`: Transformer's top-1 prediction
- `cnn_top5_moves`: CNN's top-5 predictions (list string)
- `transformer_top5_moves`: Transformer's top-5 predictions
- `cnn_correct`: Boolean (0/1) - CNN correctness
- `transformer_correct`: Boolean (0/1) - Transformer correctness

## Generated Outputs

All outputs are saved to the specified output directory (default: `analysis_outputs/`):

### Reports
- **`analysis_report.md`**: Comprehensive markdown report with all findings
- **`summary_statistics.csv`**: Overall summary statistics
- **`mcnemar_test.txt`**: McNemar's test detailed results

### Visualizations
- **`bootstrap_confidence.png`**: Bootstrap confidence intervals
- **`bootstrap_distribution.png`**: Distribution of accuracy differences
- **`rating_stratification.png`**: Performance by rating bins
- **`theme_stratification.png`**: Performance by puzzle themes
- **`phase_stratification.png`**: Performance by game phase (opening/middlegame/endgame)
- **`error_heatmap.png`**: Error pattern visualization
- **`feature_correlations.png`**: Feature-error correlations

### Data Tables
- **`rating_stratification.csv`**: Numeric results by rating
- **`theme_stratification.csv`**: Numeric results by theme
- **`phase_stratification.csv`**: Numeric results by game phase

## Analyses Performed

### 1. **Bootstrap Analysis**
- Computes 95% confidence intervals for model accuracies
- Uses resampling with replacement
- Provides distribution of accuracy differences

### 2. **McNemar's Test**
- Paired statistical test for binary classification
- Tests if models have significantly different performance
- Reports chi-square statistic and p-value

### 3. **Rating Stratification**
- Breaks down performance by puzzle difficulty
- Rating bins: 400-800, 800-1200, ..., 2800+
- Calculates Cohen's d effect sizes

### 4. **Theme Stratification**
- Analyzes performance across different tactical themes
- Handles multi-theme puzzles with weighted approach
- Shows top 20 themes by frequency

### 5. **Game Phase Analysis**
- Classifies puzzles as opening/middlegame/endgame
- Based on piece count in FEN position
- Compares model performance per phase

### 6. **Top-5 Accuracy**
- Measures if correct move is in top-5 predictions
- Broader metric than top-1 accuracy
- Useful for understanding near-miss patterns

### 7. **Error Pattern Analysis**
- Creates confusion matrix-style heatmap
- Shows agreement and disagreement patterns
- Identifies complementary strengths

### 8. **Feature Correlation**
- Correlates puzzle features with model errors
- Features: Rating, RatingDeviation, Popularity, NbPlays
- Identifies which features predict errors

## Performance & Memory

### Memory Usage
- Full dataset (~5.6M rows): ~1-2 GB RAM
- Chunked reading minimizes memory footprint
- Progress bars show memory usage

### Runtime
- Full analysis (5.6M puzzles, 1000 bootstrap): ~20-30 minutes
- Sample (10k puzzles, 100 bootstrap): ~2-3 minutes
- FEN parsing is the slowest operation

### Optimization Tips
1. Use `--sample` for development/testing
2. Reduce `--bootstrap-iterations` for faster testing
3. Increase `--chunk-size` if you have more RAM
4. Use SSD for faster CSV reading

## Interpreting Results

### Key Metrics

1. **Top-1 Accuracy**: Percentage where the model's top prediction is correct
2. **Top-5 Accuracy**: Percentage where correct move is in model's top 5
3. **Cohen's d**: Effect size measure (0.2=small, 0.5=medium, 0.8=large)
4. **McNemar p-value**: < 0.05 indicates significant difference

### Understanding Stratification

- **Positive difference**: Transformer outperforms CNN
- **Negative difference**: CNN outperforms Transformer
- **Near-zero difference**: Similar performance

Look for:
- Simpson's Paradox: Overall trend reverses in subgroups
- Specialization: Models excel in different puzzle types
- Consistency: Performance variation across strata

### Report Sections

1. **Executive Summary**: High-level findings
2. **Overall Performance**: Aggregate metrics
3. **Statistical Tests**: Significance testing
4. **Stratification**: Detailed breakdowns
5. **Key Findings**: Automated insights
6. **Conclusions**: Summary and recommendations

## Troubleshooting

### Common Issues

**Issue**: CSV file not found
- **Solution**: Use `--input` to specify the path, or place CSV in `Data/results_full.csv`

**Issue**: Out of memory
- **Solution**: Use `--sample` to analyze a subset, or increase system RAM

**Issue**: Slow FEN parsing
- **Solution**: Use `--sample` for testing, or wait for full analysis (one-time cost)

**Issue**: Missing columns error
- **Solution**: Verify CSV has all required columns (see Input Format section)

### Data Quality Warnings

The script validates data and reports:
- Missing values per column
- Invalid ratings (< 0 or > 4000)
- Unexpected data types
- Inconsistent puzzle counts

Review the "DATA VALIDATION & SUMMARY STATISTICS" section for any warnings.

## Examples

### Example 1: Quick Exploratory Analysis

```bash
# Test with 5,000 puzzles
python statistical_analysis.py --sample 5000 --bootstrap-iterations 100
```

### Example 2: Production Analysis with Custom Seed

```bash
# Full analysis with reproducible results
python statistical_analysis.py --seed 123 --output-dir production_results
```

### Example 3: Memory-Constrained Environment

```bash
# Use smaller chunks and sample
python statistical_analysis.py --sample 100000 --chunk-size 50000
```

## Citation

If you use this analysis script in your research, please cite:

```
Statistical Analysis Script for Chess Model Comparison
Author: Generated for CNN vs Transformer Analysis
Repository: omarghara/CNN_vs_Transformer_in_Chess-Contradicting_Metrics
Year: 2026
```

## License

This script is part of the CNN vs Transformer Chess Analysis project.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for similar problems
- Review the comprehensive docstrings in the code

---

**Last Updated**: 2026-01-23
