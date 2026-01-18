# 🔍 CNN vs Transformer in Chess: Contradicting Metrics

A comprehensive analysis investigating an unexpected paradox in chess move prediction - why does a CNN model appear to perform as well as (or better than) a Transformer model in aggregate metrics, despite the Transformer winning 48% vs 30% in head-to-head games?

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [The Central Paradox](#-the-central-paradox)
- [Repository Structure](#-repository-structure)
- [Dataset Description](#-dataset-description)
- [What Has Been Completed](#-what-has-been-completed)
- [What Remains (TODOs)](#-what-remains-todos)
- [Methodology](#-methodology)
- [Technologies Used](#-technologies-used)
- [Expected Research Outcomes](#-expected-research-outcomes)
- [Quick Stats](#-quick-stats)
- [How to Use](#-how-to-use)

## 🎯 Project Overview

This project investigates a chess move prediction paradox using a dataset of **5.6 million Lichess puzzles**. Two models (CNN and Transformer) were evaluated for their ability to predict the correct first move in chess puzzles. While head-to-head gameplay suggests the Transformer significantly outperforms the CNN (48% vs 30% win rate), aggregate Top-1 accuracy metrics tell a different story.

The analysis employs **causal inference** and **Simpson's Paradox** frameworks to understand how confounding variables (rating, themes, opening tags) may explain this apparent contradiction.

## 🔥 The Central Paradox

### Unexpected Finding

```
CNN Top-1 Accuracy:         38.20%
Transformer Top-1 Accuracy: 37.98%
```

**The Paradox**: Despite the Transformer winning ~48% of head-to-head games versus the CNN's ~30%, the aggregate Top-1 accuracy suggests the CNN performs slightly *better* or at least equally well!

### Hypothesis

This paradox likely arises from **Simpson's Paradox** - a phenomenon where trends in different groups disappear or reverse when groups are combined. Confounding variables such as:
- **Rating**: Puzzle difficulty ratings (399-3395)
- **Themes**: Tactical patterns (mating, endgame, sacrifice, etc.)
- **Opening Tags**: Chess opening classifications

...may affect both model performance and the distribution of puzzles, creating misleading aggregate statistics.

## 📁 Repository Structure

```
CNN_vs_Transformer_in_Chess-Contradicting_Metrics/
├── chess_policy_comparison.ipynb    # Main analysis notebook
├── Data/
│   └── results_full.csv            # Full dataset (5.6M puzzles)
├── .gitattributes
└── README.md                       # This file
```

## 📊 Dataset Description

### Source
- **Dataset**: Lichess Puzzles
- **Size**: 5,600,086 puzzles
- **After Filtering**: 5,234,621 puzzles (93.5% retained)
- **Rating Range**: 399 - 3395

### Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `PuzzleId` | string | Unique puzzle identifier |
| `FEN` | string | Chess position in Forsyth-Edwards Notation |
| `Moves` | string | Ground truth solution moves (first move is target) |
| `Rating` | int | Puzzle difficulty rating (399-3395) |
| `RatingDeviation` | int | Uncertainty measure for the rating |
| `Popularity` | int | Puzzle popularity score |
| `NbPlays` | int | Number of times the puzzle has been played |
| `Themes` | string | Puzzle themes/categories (e.g., "crushing hangingPiece long middlegame") |
| `GameUrl` | string | URL to the original game |
| `OpeningTags` | string | Chess opening classification tags |
| `cnn_predicted_move` | string | Top-1 predicted move from CNN model (UCI format) |
| `transformer_predicted_move` | string | Top-1 predicted move from Transformer model (UCI format) |
| `cnn_top5_moves` | string | Top-5 predicted moves from CNN model (UCI format, string list) |
| `transformer_top5_moves` | string | Top-5 predicted moves from Transformer model (UCI format, string list) |
| `cnn_correct` | boolean | Whether CNN's top-1 prediction is correct |
| `transformer_correct` | boolean | Whether Transformer's top-1 prediction is correct |

## ✅ What Has Been Completed

### Section 1: Setup and Data Loading ✅
- Imported necessary libraries (pandas, numpy, matplotlib, seaborn, ast, json)
- Loaded the full dataset from `Data/results_full.csv`
- Performed initial data inspection
- Verified dataset structure and column types

**Key Statistics:**
- Total puzzles: 5,600,086
- 16 columns total
- Memory usage: ~683.6 MB

### Section 2: Reproducing the Paradox ✅
- Computed Top-1 accuracy for both models
- Visualized the paradox with comparison plots
- Confirmed unexpected finding:
  - **CNN**: 38.20% accuracy
  - **Transformer**: 37.98% accuracy

### Section 3: Data Cleaning and Filling ✅
- Handled missing values in dataset
- Applied data quality filters
- Retained 93.5% of original data (5,234,621 puzzles)
- Created cleaned dataset for analysis

### Section 3.2: Outlier Detection and Handling ✅
- Created boxplots for Rating, RatingDeviation, Popularity, and NbPlays
- Identified and analyzed outliers
- Applied appropriate filtering strategies
- Documented outlier characteristics

### Section 4: Causal Framing - DAG ✅
- Defined causal Directed Acyclic Graph (DAG)
- Identified confounding variables:
  - Rating (puzzle difficulty)
  - Themes (tactical patterns)
  - Opening Tags (chess openings)
- Established causal framework for analysis

**DAG Structure:**
```
         Model Type
              |
              v
      Prediction Accuracy
              ^
         /    |    \
        /     |     \
    Rating  Themes  [other confounders]
       |       |
       +-------+ (potential relationship)
```

## 🔨 What Remains (TODOs)

### Section 5: Confounding/Simpson: Rating ❌

**Goal**: Investigate whether Simpson's Paradox occurs when stratifying by rating bins.

**Tasks:**
- [ ] Create rating bins (e.g., [400-800], [800-1200], ..., [2800-3400])
- [ ] Compute Top-1 accuracy per rating bin for CNN
- [ ] Compute Top-1 accuracy per rating bin for Transformer
- [ ] Create comparison plots showing accuracy vs rating bin
- [ ] Generate comparison table
- [ ] **Key Question**: Does Transformer outperform CNN within individual rating bins, even though aggregate shows otherwise?

### Section 6: Confounding/Simpson: Themes ❌

**Goal**: Analyze model performance across different tactical themes.

**Tasks:**
- [ ] Parse theme strings into individual themes
- [ ] Handle multi-theme puzzles (weighted approach: weight = 1/num_themes)
- [ ] Compute weighted Top-1 accuracy per theme for both models
- [ ] Create theme-level comparison table
- [ ] Plot accuracy by theme
- [ ] Identify themes where models show different strengths
- [ ] Discuss theme-level performance differences

### Section 7: Confounding/Simpson: Opening Tags ❌

**Goal**: Investigate if opening types affect model performance differently.

**Tasks:**
- [ ] Parse opening tag strings
- [ ] Handle multi-tag puzzles appropriately
- [ ] Compute accuracy per opening tag for both models
- [ ] Create opening-level comparison visualizations
- [ ] Analyze if certain openings favor one model over another

### Section 8: Distribution Divergence Analysis ❌

**Goal**: Quantify how differently the models distribute their errors across puzzle types using KL divergence.

**Tasks:**
- [ ] Define disagreement sets:
  - Set A: CNN correct, Transformer wrong
  - Set B: CNN wrong, Transformer correct
- [ ] For each set, compute distribution over rating bins
- [ ] For each set, compute distribution over themes
- [ ] Apply epsilon smoothing for KL divergence calculation
- [ ] Compute KL divergence between Set A and Set B distributions
- [ ] Display distributions and KL values
- [ ] **Interpret**: Which puzzle types show strongest specialization patterns?

### Section 9: Joint Analysis ❌

**Goal**: Perform comprehensive multi-variable analysis.

**Tasks:**
- [ ] Split dataset into:
  - Split A (discovery set): Identify patterns
  - Split B (evaluation set): Validate findings
- [ ] **On Split A only:**
  - [ ] Identify disagreement sets
  - [ ] Compute KL contribution per theme
  - [ ] Compute KL contribution per rating bin
  - [ ] Select "important" themes/bins (top K or top 30% KL mass)
  - [ ] Store important themes/bins
- [ ] **On Split B only:**
  - [ ] Classify puzzles as "important" vs "non-important"
  - [ ] Compute Top-1 accuracy for both models on each group
  - [ ] Create comparison plots/tables

### Section 10: Top-5 Accuracy Analysis ❌

**Goal**: Evaluate if different metrics tell different stories.

**Tasks:**
- [ ] Compute Top-5 accuracy for both models
- [ ] Define and compute "near-miss" metric:
  - Correct move in Top-5 but not Top-1
- [ ] Create metric comparison plots
- [ ] Generate comprehensive metric comparison table
- [ ] **Discussion**: How does metric choice affect our conclusions?

### Section 11: Error Analysis ❌

**Goal**: Deep dive into specific error patterns.

**Tasks:**
- [ ] Identify common error patterns for each model
- [ ] Analyze error severity (how wrong were wrong predictions?)
- [ ] Compare error distributions
- [ ] Examine specific puzzle examples where models disagree
- [ ] Document qualitative differences in error types

### Section 12: Statistical Testing ❌

**Goal**: Validate findings with rigorous statistical tests.

**Tasks:**
- [ ] Perform McNemar's test for paired binary data
- [ ] Conduct bootstrap hypothesis testing
- [ ] Calculate confidence intervals for accuracy differences
- [ ] Test significance of observed patterns
- [ ] Report p-values and effect sizes

### Section 13: Summary and Conclusions ❌

**Goal**: Synthesize findings and explain the paradox.

**Tasks:**
- [ ] Create summary table of all findings
- [ ] Explain how the paradox was resolved
- [ ] Document key lessons from the analysis
- [ ] Discuss Simpson's Paradox implications
- [ ] Highlight limitations of the study
- [ ] Provide recommendations for model evaluation
- [ ] Suggest future research directions

## 🧪 Methodology

### Causal Framework

The analysis uses **causal inference** to understand the relationship between model type and prediction accuracy. The key insight is that confounding variables may create misleading aggregate statistics.

### Simpson's Paradox

**Simpson's Paradox** occurs when a trend appears in different groups of data but disappears or reverses when the groups are combined. This project investigates whether:

1. **Stratification by Rating**: When we split puzzles by difficulty (rating), does the Transformer consistently outperform the CNN within each difficulty level?

2. **Stratification by Themes**: Do different tactical themes reveal different relative strengths between the models?

3. **Distribution Divergence**: Do the models specialize in different types of puzzles, making aggregate comparisons misleading?

### Directed Acyclic Graph (DAG)

The causal DAG helps visualize the hypothesized relationships:

- **Model Type** → **Prediction Accuracy** (direct effect)
- **Confounders** (Rating, Themes, Openings) → **Prediction Accuracy**
- Confounders may be correlated with each other
- Confounders affect which puzzles are in the dataset

### Analysis Approach

1. **Descriptive Statistics**: Understand the data distribution
2. **Stratification**: Break down performance by confounding variables
3. **Divergence Analysis**: Quantify specialization using KL divergence
4. **Cross-Validation**: Use discovery/evaluation splits
5. **Statistical Testing**: Validate findings rigorously

## 💻 Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **ast**: Parse string representations of Python data structures
- **json**: JSON data handling

### Analysis Tools
- Jupyter Notebook
- Python 3.x

### Statistical Methods
- Simpson's Paradox detection
- Kullback-Leibler (KL) divergence
- Weighted accuracy metrics
- Bootstrap hypothesis testing
- McNemar's test

## 🎓 Expected Research Outcomes

### Primary Goals

1. **Resolve the Paradox**: Determine why aggregate metrics contradict head-to-head performance
2. **Identify Confounders**: Pinpoint which variables (rating, themes, openings) drive the paradox
3. **Quantify Specialization**: Measure how models specialize in different puzzle types
4. **Metric Sensitivity**: Understand how metric choice affects conclusions

### Secondary Goals

1. Document a clear case study of Simpson's Paradox in ML evaluation
2. Provide guidelines for chess AI model comparison
3. Demonstrate importance of stratified analysis in ML
4. Create reproducible analysis workflow

### Potential Findings

- **Hypothesis 1**: Transformer outperforms CNN within each rating bin, but dataset composition creates misleading aggregate
- **Hypothesis 2**: Models specialize in different themes, making overall comparison inappropriate
- **Hypothesis 3**: Top-5 accuracy and near-miss metrics tell a different story than Top-1
- **Hypothesis 4**: Statistical tests reveal no significant difference in aggregate, but significant differences in strata

## 📈 Quick Stats

### Dataset
- **Total Puzzles**: 5,600,086
- **After Filtering**: 5,234,621 (93.5% retained)
- **Rating Range**: 399 - 3395
- **Median Rating**: ~1483
- **Features**: 16 columns

### Model Performance (Aggregate)
```
┌─────────────┬──────────┐
│ Model       │ Top-1 %  │
├─────────────┼──────────┤
│ CNN         │  38.20%  │
│ Transformer │  37.98%  │
└─────────────┴──────────┘
```

### Head-to-Head Game Performance
```
┌─────────────┬──────────┐
│ Model       │ Win %    │
├─────────────┼──────────┤
│ CNN         │  ~30%    │
│ Transformer │  ~48%    │
└─────────────┴──────────┘
```

### Progress
- **Completed Sections**: 1-4 (Setup, Paradox, Cleaning, DAG)
- **Remaining Sections**: 5-13 (Analysis, Testing, Conclusions)
- **Overall Progress**: ~31% (4/13 sections)

## 🚀 How to Use

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn jupyter
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/omarghara/CNN_vs_Transformer_in_Chess-Contradicting_Metrics.git
cd CNN_vs_Transformer_in_Chess-Contradicting_Metrics
```

2. **Verify data file**
```bash
# Ensure Data/results_full.csv exists
ls -lh Data/results_full.csv
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook chess_policy_comparison.ipynb
```

### Running the Analysis

1. **Execute completed sections** (Sections 1-4):
   - Run all cells sequentially
   - Verify output matches expected results
   - Check that CNN accuracy ≈ 38.20%, Transformer ≈ 37.98%

2. **Continue with TODO sections** (Sections 5-13):
   - Implement analysis following TODO comments
   - Each section builds on previous findings
   - Follow the causal framework established in Section 4

### Data Format

Expected CSV structure in `Data/results_full.csv`:
```csv
PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,cnn_predicted_move,transformer_predicted_move,cnn_top5_moves,transformer_top5_moves,cnn_correct,transformer_correct
00008,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,1877,76,95,8786,crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#48,,b2b1,b2b1,"[""b2b1"", ""f2g3"", ""e7e6"", ""b2a3"", ""b2c2"]","[""b2b1"", ""f2g3"", ""b2c2"", ""e7e6"", ""b2a1"]",0,0
```

### Key Metrics to Monitor

- **Aggregate Accuracy**: Overall Top-1 performance
- **Stratified Accuracy**: Performance within rating bins/themes
- **KL Divergence**: Specialization patterns
- **Top-5 Accuracy**: Broader performance view
- **Near-Miss Rate**: Almost-correct predictions

### Expected Runtime

- **Data Loading**: < 1 minute
- **Sections 1-4**: 2-5 minutes
- **Full Analysis (when complete)**: ~15-30 minutes
  (depends on hardware and data size)

## 📝 Notes

- The dataset is large (~1.6 GB CSV file)
- Some analyses require significant memory (recommend ≥16GB RAM)
- Results may vary slightly due to floating-point arithmetic
- The analysis prioritizes interpretability over raw performance

## 🤝 Contributing

This is a research project. Contributions to complete the TODO sections are welcome! Please:
1. Follow the established causal framework
2. Document findings thoroughly
3. Include visualizations for key results
4. Update this README with new findings

## 📄 License

[Specify license if applicable]

## 👤 Author

Omar Ghara

## 🙏 Acknowledgments

- Lichess for providing the puzzle database
- The chess AI community for model development
- Contributors to the pandas, numpy, and matplotlib ecosystems

---

**Status**: 🚧 Work in Progress - Sections 5-13 need completion

**Last Updated**: 2026-01-18
