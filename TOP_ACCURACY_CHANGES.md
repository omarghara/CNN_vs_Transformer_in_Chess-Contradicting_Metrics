# Top-K Accuracy Calculation Changes

## Summary of Changes

This document describes the changes made to fix the Top-5 accuracy calculation and add support for Top-3 accuracy summary generation.

## Issues Fixed

### 1. Column Name Normalization
**Problem**: The CSV file uses column names `cnn_correct` and `transformer_correct`, but the code expected `cnn_first_move_correct` and `transformer_first_move_correct`.

**Solution**: Added column name mapping in the `normalize_column_names()` function to automatically rename these columns during data loading:
```python
'cnn_correct': 'cnn_first_move_correct',
'transformer_correct': 'transformer_first_move_correct'
```

### 2. Top-5 Accuracy Calculation
**Problem**: Top-5 accuracy was showing 0.00% due to the column name mismatch.

**Solution**: With the column normalization fix, the existing `calculate_top5_accuracy()` function now works correctly and produces accurate results.

## New Features

### 1. Top-3 Accuracy Calculation
Added a new function `calculate_top3_accuracy()` that:
- Extracts the first 3 moves from each model's top-5 predictions
- Compares them against the ground truth (first move in the Moves column)
- Calculates and returns accuracy percentages for both models

### 2. Top-3 Accuracy Summary Markdown
Added `generate_top3_accuracy_summary()` function that creates a dedicated markdown file with:
- Top-3 accuracy table for both models
- Performance difference analysis
- Timestamp and dataset size information

### 3. Top-5 Accuracy Summary Markdown
Added `generate_top5_accuracy_summary()` function that creates a dedicated markdown file with:
- Top-5 accuracy table for both models
- Performance difference analysis
- Timestamp and dataset size information

### 4. Command-Line Flags
Added two new optional flags to control summary generation:
- `--generate-top3-summary`: Generates `top3_accuracy_summary.md`
- `--generate-top5-summary`: Generates `top5_accuracy_summary.md`

## Usage Examples

### Basic Usage (without summaries)
```bash
python statistical_analysis.py --input Data/results_full.csv
```

### Generate Top-5 Summary Only
```bash
python statistical_analysis.py --input Data/results_full.csv --generate-top5-summary
```

### Generate Both Top-3 and Top-5 Summaries
```bash
python statistical_analysis.py --input Data/results_full.csv --generate-top3-summary --generate-top5-summary
```

### With Sample Data for Testing
```bash
python statistical_analysis.py --input Data/results_full.csv --sample 1000 --generate-top3-summary --generate-top5-summary
```

## Output Files

When the flags are used, the following new files are generated:

1. **`top3_accuracy_summary.md`** (when `--generate-top3-summary` is used)
   - Standalone markdown file with Top-3 accuracy results
   - Includes model comparison table and analysis

2. **`top5_accuracy_summary.md`** (when `--generate-top5-summary` is used)
   - Standalone markdown file with Top-5 accuracy results
   - Includes model comparison table and analysis

## Changes to Existing Files

### Modified Functions
1. **`normalize_column_names()`**: Added column name mappings for correctness columns
2. **`parse_arguments()`**: Added two new command-line arguments
3. **`main()`**: 
   - Added Top-3 calculation (conditional on flag)
   - Added calls to summary generation functions (conditional on flags)
   - Updated summary statistics to include Top-3 when calculated
   - Updated final output messages to mention new summary files

### Documentation Updates
- Updated module docstring with new usage examples
- Added help text for new command-line flags

## Testing

The changes were tested with a sample of 1,000 puzzles from the full dataset:
- Top-3 Accuracy: CNN 62.90%, Transformer 64.10% (difference: 1.20%)
- Top-5 Accuracy: CNN 73.70%, Transformer 76.00% (difference: 2.30%)

Both accuracy calculations now work correctly and produce non-zero values.

## Backward Compatibility

All changes are backward compatible:
- The script runs normally without the new flags (default behavior unchanged)
- Column name normalization handles both old and new column naming conventions
- No existing functionality was removed or modified in a breaking way
