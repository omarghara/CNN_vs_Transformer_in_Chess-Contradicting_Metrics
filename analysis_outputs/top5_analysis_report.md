# Top-5 Statistical Analysis Report: CNN vs Transformer Chess Models

**Analysis Date**: 2026-01-24 13:30:45

---

## Executive Summary

This report presents a comprehensive statistical analysis of 5,600,086 chess puzzle predictions comparing CNN and Transformer models using Top-5 accuracy metrics.

## Overall Performance

### Top-5 Accuracy

| Model | Accuracy | 95% CI |
|-------|----------|--------|
| CNN | 89.64% | [89.62, 89.67] |
| Transformer | 93.37% | [93.35, 93.39] |
| **Difference** | **3.72%** | **[3.70, 3.74]** |

## Statistical Tests

### McNemar's Test

- **Chi-square statistic**: 106974.7541
- **P-value**: 0.000000
- **Result**: Statistically significant difference (p < 0.05)

**Contingency Table:**

- CNN correct, Transformer wrong: 98,933
- CNN wrong, Transformer correct: 307,430

## Prediction Agreement

| Category | Count | Percentage |
|----------|-------|------------|
| Both Correct | 4,921,197 | 87.88% |
| Both Wrong | 272,526 | 4.87% |
| CNN Only | 98,933 | 1.77% |
| Transformer Only | 307,430 | 5.49% |

## Rating Stratification

Performance breakdown by puzzle difficulty rating:

| Rating Bin   |   Count |   CNN Accuracy (%) |   Transformer Accuracy (%) |   Difference (%) |   Cohen's d |
|:-------------|--------:|-------------------:|---------------------------:|-----------------:|------------:|
| 400-800      |  519803 |            98.5612 |                    99.3499 |          0.78876 |   0.0776446 |
| 800-1200     | 1480817 |            94.5457 |                    97.2331 |          2.68737 |   0.135671  |
| 1200-1600    | 1349092 |            89.3935 |                    94.0201 |          4.62659 |   0.168357  |
| 1600-2000    | 1148221 |            85.5247 |                    90.5569 |          5.03213 |   0.155549  |
| 2000-2400    |  779902 |            83.8521 |                    88.0849 |          4.23284 |   0.122101  |
| 2400-2800    |  305270 |            82.6164 |                    86.1739 |          3.55751 |   0.0981473 |
| 2800+        |   16981 |            79.9541 |                    83.0929 |          3.1388  |   0.0809385 |

![Rating Stratification](top5_rating_stratification.png)

## Theme Stratification

Performance breakdown by puzzle theme (top themes by count):

| Theme            |    Count |   CNN Accuracy (%) |   Transformer Accuracy (%) |   Difference (%) |
|:-----------------|---------:|-------------------:|---------------------------:|-----------------:|
| short            | 693709   |            89.2061 |                    93.667  |         4.46086  |
| endgame          | 652728   |            92.0886 |                    95.2313 |         3.14273  |
| middlegame       | 624683   |            87.8104 |                    92.1054 |         4.295    |
| crushing         | 517224   |            88.6693 |                    92.817  |         4.14768  |
| advantage        | 418061   |            85.8491 |                    91.157  |         5.30791  |
| mate             | 361345   |            96.3867 |                    97.7496 |         1.36289  |
| long             | 337220   |            86.9355 |                    90.9489 |         4.01346  |
| oneMove          | 172690   |            99.2631 |                    99.8696 |         0.606511 |
| mateIn1          | 171943   |            99.283  |                    99.8838 |         0.600791 |
| fork             | 159836   |            89.7991 |                    93.7738 |         3.97472  |
| master           | 155849   |            88.376  |                    92.615  |         4.23894  |
| mateIn2          | 151316   |            95.4459 |                    97.1807 |         1.73477  |
| veryLong         |  96455.9 |            88.6606 |                    91.9542 |         3.29355  |
| kingsideAttack   |  92353.4 |            88.4069 |                    92.0081 |         3.60122  |
| sacrifice        |  72333   |            65.0331 |                    71.981  |         6.9479   |
| pin              |  70053.6 |            84.0808 |                    90.793  |         6.71219  |
| opening          |  68912.5 |            89.5964 |                    93.5601 |         3.96366  |
| defensiveMove    |  68626.1 |            94.0578 |                    95.7843 |         1.72643  |
| advancedPawn     |  62247   |            91.8425 |                    94.3964 |         2.55392  |
| discoveredAttack |  59818.2 |            67.1278 |                    79.1668 |        12.039    |

![Theme Stratification](top5_theme_stratification.png)

## Game Phase Stratification

Performance breakdown by game phase:

| Phase      |   Count |   CNN Accuracy (%) |   Transformer Accuracy (%) |   Difference (%) |   Cohen's d |
|:-----------|--------:|-------------------:|---------------------------:|-----------------:|------------:|
| Opening    | 1910446 |            87.4901 |                    91.6531 |          4.16301 |    0.136528 |
| Middlegame | 2593450 |            89.2554 |                    93.2775 |          4.02213 |    0.142827 |
| Endgame    | 1096190 |            94.3162 |                    96.5652 |          2.24897 |    0.107969 |

![Phase Stratification](top5_phase_stratification.png)

## Additional Visualizations

### Bootstrap Analysis

![Bootstrap Distribution](top5_bootstrap_distribution.png)

![Bootstrap Confidence](top5_bootstrap_confidence.png)

## Key Findings

1. **Overall Performance**: Transformer outperforms CNN by 3.72% in top-5 accuracy.

2. **Rating Stratification**: Largest performance difference observed in 1600-2000 rating range (5.03%).

3. **Theme Stratification**: Largest performance difference observed for 'discoveredAttack' theme (12.04%).

4. **Statistical Significance**: McNemar's test indicates a statistically significant difference (p = 0.000000).

## Conclusions

This analysis reveals:

- The models show different strengths across puzzle ratings and themes
- Stratified analysis provides deeper insights than aggregate metrics alone
- Top-5 accuracy shows how often the correct move is within the top 5 predictions
- Consider ensemble approaches to leverage complementary strengths

## Files Generated

This analysis generated the following output files:

- `top5_analysis_report.md` - This comprehensive report
- `top5_mcnemar_test.txt` - McNemar's test detailed results
- `top5_bootstrap_confidence.png` - Bootstrap confidence intervals
- `top5_bootstrap_distribution.png` - Bootstrap distribution plot
- `top5_rating_stratification.csv` / `.png` - Performance by rating
- `top5_theme_stratification.csv` / `.png` - Performance by theme
- `top5_phase_stratification.csv` / `.png` - Performance by game phase

---

*Report generated automatically by statistical_analysis.py*
