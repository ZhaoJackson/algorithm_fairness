# Algorithmic Fairness on COMPAS

This repository implements two fairness approaches on the COMPAS dataset, combining fairness-aware feature selection with conditional-fairness interventions (Local Massaging and Preferential Sampling). The core workflow lives in the notebook `Fairness_Algorithms.ipynb`, and results are saved under `output/`.

## Repository Structure

```
algorithm_fairness/
├── Fairness_Algorithms.ipynb
├── README.md
├── data/
│   ├── compas-scores-two-years.csv
│   ├── data_cleaned.csv
│   └── data_renamed.csv
├── output/
│   ├── age/
│   │   ├── local_massaging/
│   │   │   └── lm_full.csv
│   │   └── preferential_sample/
│   │       └── full_results_Preferential_Sampling.csv
│   └── race/
│       ├── local_massaging/
│       │   └── local_massaging.csv
│       └── preferential_sample/
│           └── preferential_sampling.csv
└── reference/
    ├── Fairness-aware Feature Selection.pdf
    ├── Handling Conditional Discrimination.pdf
    ├── Handling Conditional Discrimination and Information Theoretic Measures for Fairness-Aware Feature Selection.pdf
    ├── Info Theoretic Measures for Fairness-aware Feature selection.pdf
    └── Report.pdf
```

## Data

- `data/compas-scores-two-years.csv`: Original COMPAS dataset (~7.2k rows)
- `data/data_cleaned.csv`: Minimal cleaned copy produced by the notebook
- `data/data_renamed.csv`: Alternative column naming for reference

Target and sensitive variables
- Target: `two_year_recid` (binary)
- Sensitive: Race (African-American vs Caucasian), Sex
- Conditional axis: Age categories `age_cat ∈ {Less than 25, 25 - 45, Greater than 45}`

## Methods

### 1) Fairness-aware Feature Selection (reference: fairness-aware feature selection and information-theoretic measures)
- Compute mutual information (MI) between features and the target.
- Penalize features highly associated with sensitive attributes (demo: penalize sex-related one-hots), then select top-k.
- Purpose: reduce sensitive leakage before any fairness intervention.

### 2) Handling Conditional Discrimination

Local Massaging (age-stratified, sex fairness)
- Partition data by `age_cat`.
- Train a model per stratum to score near the decision boundary.
- Flip a minimal set of labels (female 0→1, male 1→0) to reduce within-age-group disparity.

Preferential Sampling (age-stratified, sex fairness)
- Partition by `age_cat` and score samples.
- Delete or duplicate borderline cases to rebalance outcomes without changing labels.

Race-focused variants
- The notebook also contains race-based local massaging and preferential sampling to reduce disparities between African-American and Caucasian groups.

## How to Run

1) Open `Fairness_Algorithms.ipynb` and run all cells.
2) The notebook will:
   - Load and clean data (`data_cleaned.csv`)
   - Perform MI-based feature selection with sensitive penalty
   - Apply Local Massaging and Preferential Sampling (age-stratified)
   - Train baselines (scaled Logistic Regression, RandomForest, XGBoost, LightGBM)
   - Evaluate accuracy/F1 and fairness metrics (dp_gap, eo_gap)
   - Save outputs under `output/`

## Outputs

Age-stratified (sex fairness within age groups)
- `output/age/local_massaging/lm_full.csv`
- `output/age/preferential_sample/full_results_Preferential_Sampling.csv`

Race-focused
- `output/race/local_massaging/local_massaging.csv`
- `output/race/preferential_sample/preferential_sampling.csv`

## Evaluation

Metrics
- Accuracy, F1
- Demographic parity gap (dp_gap): |P(ŷ=1|group=0) − P(ŷ=1|group=1)|
- Equal opportunity gap (eo_gap): |TPR(group=0) − TPR(group=1)|

Reported in the notebook
- Baseline: accuracy/F1, race calibration difference, and dp_gap/eo_gap (race) on test.
- Local Massaging / Preferential Sampling: accuracy/F1 and dp_gap/eo_gap from the modified data.
- Consolidated summary comparing: Baseline (race) vs Local Massaging (sex|age) vs Preferential Sampling (sex|age).

Interpretation
- Smaller dp_gap/eo_gap indicates better parity and equal opportunity.
- Compare baseline vs interventions to understand fairness–accuracy trade-offs.

## Practical Notes

- Run feature selection before interventions to limit sensitive leakage.
- For conditional fairness, operate within each `age_cat` to avoid Simpson’s paradox effects.
- Inspect CSVs in `output/age/` and `output/race/` to audit group rates, label flips, and sampling edits.

## Implications & Limitations

- Local Massaging minimally changes labels; it can reduce measured gaps but may diverge from recorded outcomes.
- Preferential Sampling changes class balance; it may impact calibration and variance.
- Conditional fairness by age can reduce sex disparities within age groups, while other axes (e.g., race) may still show gaps.
- Feature selection reduces—but may not eliminate—proxy leakage; audit selected features.
- Reassess calibration and performance after interventions to confirm acceptable trade-offs.

## References (see `reference/`)

- Fairness-aware Feature Selection.pdf
- Handling Conditional Discrimination.pdf
- Handling Conditional Discrimination and Information Theoretic Measures for Fairness-Aware Feature Selection.pdf
- Info Theoretic Measures for Fairness-aware Feature selection.pdf
- Report.pdf
