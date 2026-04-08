# Predicting Time to Clearance of Sport-Related Concussions Using Machine Learning

> **Paper:** Tran M, Holler J, Moran B, Schilaty ND, Templeton JM. *Predicting Time to Clearance of Sport-Related Concussions Using Machine Learning.* Digital Health (in review).
>
> **IRB:** USF STUDY003514 · **Funding:** Florida Department of State Center for Neuromusculoskeletal Research

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Data](#data)
5. [Full ML Pipeline — Step-by-Step](#full-ml-pipeline--step-by-step)
   - [Stage 1 · Data Preprocessing](#stage-1--data-preprocessing)
   - [Stage 2 · Feature Engineering](#stage-2--feature-engineering)
   - [Stage 3 · Hyperparameter Tuning](#stage-3--hyperparameter-tuning)
   - [Stage 4 · Leave-One-Out Cross-Validation (LOOCV)](#stage-4--leave-one-out-cross-validation-loocv)
   - [Stage 5 · Evaluation & Feature Importance](#stage-5--evaluation--feature-importance)
6. [Pipeline Sequence Diagram](#pipeline-sequence-diagram)
7. [Manuscript–Code Alignment](#manuscriptcode-alignment)
8. [Reproducing Results](#reproducing-results)
9. [Citation](#citation)

---

## Overview

This repository contains the complete preprocessing and machine learning (ML) code for a retrospective cohort study of 217 athletes with sport-related concussions (SRC) seen at the USF Concussion Center (2021–2025). Six ML classifiers are trained and evaluated for binary classification of recovery duration into **normal recovery (< 30 days, Class 0)** and **prolonged recovery (≥ 30 days, Class 1)** using clinical assessment data collected at two time points.

**Key findings:**
- Adding Visit 2 features improved accuracy in 66% of models.
- XGBoost achieved the highest Visit 2 accuracy (0.84, +5% over Visit 1).
- VOR Vertical Headache and its inter-visit change score were the most consistent predictors of prolonged recovery (present in 81% and 100% of models, respectively).
- Treatment presence between visits was the strongest predictor of normal recovery.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── preprocessing/
│   └── data_preprocessing.ipynb          # Full cleaning pipeline (all visits)
├── machineLearning_visit1_REVISED-orginalset.ipynb  # ML pipeline — Visit 1 features (n=48)
├── machineLearning_visit2_REVISED-originalExperimentSet.ipynb  # ML pipeline — Visit 2 features (n=95)
└── figures/                              # Output figures (confusion matrices, FI plots)
```

> **Note on data:** The USF Concussion Center dataset is not publicly available (clinical REDCap database). The preprocessing notebook documents every cleaning step so the pipeline can be applied to similarly structured clinical datasets. Contact the corresponding author for data access inquiries.

---

## Environment Setup

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version tested | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5 | Data loading and manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `scikit-learn` | ≥ 1.2 | DT, RF, SVC, Ridge, metrics, `ParameterSampler` |
| `lightgbm` | ≥ 3.3 | LightGBM classifier |
| `xgboost` | ≥ 1.7 | XGBoost classifier |
| `matplotlib` | ≥ 3.6 | Plotting |
| `seaborn` | ≥ 0.12 | Confusion matrix heatmaps |

---

## Data

The original dataset contained **3,038 unique patient records** diagnosed with concussion at USF facilities (2017–2026). After preprocessing (described below), **217 patients** with sports-related concussions and at least two clinical visits were retained for ML analysis.

| Attribute | Value |
|---|---|
| Final N | 217 |
| Male / Female | 80 (36.9%) / 137 (63.1%) |
| Mean age | 26.94 years |
| Class 0 (normal recovery, < 30 days) | 41 patients |
| Class 1 (prolonged recovery, ≥ 30 days) | 176 patients |
| Visit 1 feature count | 48 |
| Visit 2 feature count | 95 (includes base features + difference variants + treatment presence) |

**Target variable:** `buckets` — binary label derived from `days_2_clearance` (0 = < 30 days, 1 = ≥ 30 days).

**Non-predictor columns explicitly excluded from X:**
`buckets`, `record_id`, `injury_mechanism`, `days_2_clearance`, `days_2_firstvisit`, `high_total_sx_severity`, `redcap_repeat_instance` (and `high_total_sx_severity_diff` for Visit 2).

---

## Full ML Pipeline — Step-by-Step

The pipeline is implemented across three notebooks in the following order:

```
data_preprocessing.ipynb  →  machineLearning_visit1_REVISED-orginalset.ipynb
                          →  machineLearning_visit2_REVISED-originalExperimentSet.ipynb
```

### Stage 1 · Data Preprocessing

> **Notebook:** `preprocessing/data_preprocessing.ipynb`  
> **Manuscript section:** *Methodology → Data Preprocessing*

The preprocessing pipeline applies iterative, threshold-based missingness filtering before any ML is performed. No data imputation is used so that the cleaned dataset represents the most realistic clinical population.

**Steps in order:**

1. **Load raw data** — Import the full REDCap export (N = 3,038 rows).

2. **Initial null filtering** — Remove columns with missingness ≥ 0.9 and rows with missingness ≥ 0.8.

3. **Iterative threshold reduction** — Decrement both thresholds by 0.1 and repeat null filtering. Continue until the dataset stabilizes. This produces an intermediate dataset of **2,338 rows × 49 columns**.

4. **Outlier filtering (clinical timeline)** — Retain only records where:
   - Days from injury to first visit: 0 ≤ days ≤ 365
   - Days to clearance: ≥ 1 day
   
   This yields **1,865 rows × 49 columns**.

5. **Visit filtering** — Retain only patients with exactly two clinical visits recorded → **1,201 patients**.

6. **Mechanism of injury filtering** — Retain only sports-related concussions → **217 patients**.

7. **No imputation** — Missing values are retained as-is; no mean/mode/kNN/MICE imputation is applied.

> ⚠️ **EPV reporting :** The event-per-variable (EPV) ratio is computed after feature selection and reported in both notebooks. EPV = minority class N ÷ number of features. EPV for visit 1 and 2 fall below the recommended threshold of 10, which is explicitly noted as a limitation in the manuscript.

---

### Stage 2 · Feature Engineering

> **Notebooks:** both ML notebooks (Cells 4–6)  
> **Manuscript section:** *Methodology → Data Preprocessing (feature engineering paragraph)*

Feature engineering is performed **after** preprocessing and **before** the ML loop — it is not nested inside cross-validation folds.

**Visit 1 additions (48 total features):**
- `prev_head_injury` — binary flag for prior head injury history
- `hx_mood_disorder` — binary flag for history of mood disorder

**Visit 2 additions (95 total features):**
- All Visit 1 features (48)
- `treatment_present` — binary indicator of any treatment administered between visits
- **Difference features (Δ):** For every shared continuous base feature, a `_diff` variant is created:
  ```
  feature_diff = visit2_value − visit1_value
  ```
  Difference features are computed for all shared base features **except** `prev_head_injury`, `hx_mood_disorder`, and `treatment_present`.

> ⚠️ **Limitation acknowledged:** Running feature engineering outside of the LOOCV folds introduces a risk of data leakage. Future work will nest variable selection within the validation process. See the *Limitations* and *Future Work* sections of the manuscript.

---

### Stage 3 · Hyperparameter Tuning

> **Notebooks:** both ML notebooks (one cell per model, immediately before the LOOCV loop)  
> **Manuscript section:** *Methodology → ML Model Hyperparameter Tuning*

Hyperparameter tuning is performed **once** using an **80/20 train–test split** (before LOOCV), and the resulting best parameters are then fixed for use inside the LOOCV loop. This design choice is explicitly acknowledged as a source of potential optimism in the manuscript.

**Tuning procedure by model:**

#### LightGBM
```python
# scikit-learn ParameterSampler, n_iter=50, random_state=42
param_dist = {
    "num_leaves":        [31, 50, 70],
    "max_depth":         [-1, 10, 20, 30],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "n_estimators":      [100, 200, 500, 1000],
    "min_child_samples": [10, 20, 30, 50],
    "subsample":         [0.6, 0.8, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
}
```
Selection criterion: highest validation accuracy on the 20% holdout.

#### Decision Tree
```python
param_dist = {
    "criterion":         ["gini", "entropy", "log_loss"],
    "max_depth":         [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 5, 10],
    "max_features":      [None, "sqrt", "log2"],
    "splitter":          ["best", "random"],
}
```

#### Random Forest
```python
param_dist = {
    "criterion":         ["gini", "entropy"],
    "max_depth":         [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 5, 10],
    "max_features":      [None, "sqrt", "log2"],
}
```

#### XGBoost
```python
param_dist = {
    "max_depth":          [3, 5, 7],
    "learning_rate":      [0.05, 0.1, 0.2],
    "n_estimators":       [50, 100, 150],
    "subsample":          [0.7, 0.85, 1.0],
    "colsample_bytree":   [0.7, 0.85, 1.0],
    "gamma":              [0, 0.1, 0.3],
    "min_child_weight":   [1, 3, 5],
    "reg_alpha":          [0, 0.1, 0.5],
    "reg_lambda":         [0.5, 1.0, 1.5],
}
```

#### Support Vector Classifier (SVC)
```python
# Restricted to linear kernel for feature importance extraction
param_dist = {
    "C":     [0.1, 10, 100],
    "gamma": [1, 0.1, 0.01],
}
# 27 combinations evaluated (limited by computational resources)
```

#### Ridge Classifier
```python
# k-fold cross-validation on full training set (no ParameterSampler)
alpha_candidates = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20]
# Best alpha selected by highest mean CV accuracy; model re-fit on full data.
```

**Class weighting:** All models use `class_weight="balanced"` (or equivalent, e.g., `scale_pos_weight` for XGBoost) to address the 176:41 class imbalance.

---

### Stage 4 · Leave-One-Out Cross-Validation (LOOCV)

> **Notebooks:** both ML notebooks (one LOOCV loop per model)  
> **Manuscript section:** *Methodology → Study Design*

LOOCV is used as the primary validation strategy due to the small dataset size (N = 217). For each iteration *i* (i = 1 … N):

```
1.  Hold out observation i as the test instance.
2.  Train the model (using best_params from Stage 3) on the remaining N–1 observations.
3.  Accumulate feature importances (summed across folds, averaged at the end).
4.  Predict class label and probability for observation i.
5.  Store (y_true_i, y_pred_i, y_prob_i).
```

After all N iterations, aggregate predictions are used to compute all reported metrics. A final model is refitted on the **complete dataset** (all N observations) using the tuned hyperparameters; this final model is used exclusively for plotting feature importance and average effect figures.

**Feature importance accumulation by model type:**

| Model | FI method |
|---|---|
| LightGBM | `booster_.feature_importance(importance_type="gain")` summed across folds |
| XGBoost | Gain-based importance, summed across folds |
| Decision Tree | `feature_importances_` (Gini reduction), summed across folds |
| Random Forest | `feature_importances_` (Gini reduction), summed across folds |
| SVC (linear) | Absolute magnitude of learned coefficients from final refitted model |
| Ridge | Permutation importance from final refitted model |

---

### Stage 5 · Evaluation & Feature Importance

> **Notebooks:** both ML notebooks (metric and plotting cells after each LOOCV loop)  
> **Manuscript section:** *Results*

**Metrics computed** (with 95% bootstrap CIs, B = 1,000 resamples):

| Metric | Symbol | Definition |
|---|---|---|
| Accuracy | Acc. | (TP + TN) / N |
| Balanced Accuracy | B.Acc | Mean of sensitivity and specificity |
| Precision | Prec. | TP / (TP + FP) |
| Recall / Sensitivity | Rec. | TP / (TP + FN) |
| F1 Score | F1 | 2 · (Prec · Rec) / (Prec + Rec) |
| Specificity | Spec. | TN / (TN + FP) |
| Matthews Correlation Coefficient | MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) |
| Brier Score | Brier | Mean squared error of predicted probabilities |

**Output per model:** confusion matrix, top-20 feature importance table, and average effect bar chart (blue = Class 0 / normal recovery; red = Class 1 / prolonged recovery).

---

## Pipeline Sequence Diagram

```
RAW DATA (N=3,038)
       │
       ▼
┌─────────────────────────────────┐
│  STAGE 1: DATA PREPROCESSING    │
│  • Iterative null filtering     │
│  • Clinical timeline outliers   │
│  • Visit & mechanism filtering  │
│  • No imputation                │
└─────────────────────────────────┘
       │
       ▼  N=217, sports-related concussion patients with 2 visits
       │
       ├──────────────────────────┬──────────────────────────────┐
       ▼                          ▼                              │
  VISIT 1 CSV               VISIT 2 CSV                         │
  (48 features)             (95 features:                       │
                             base + Δ + treatment_present)      │
       │                          │                             │
       ▼                          ▼                             │
┌─────────────────────────────────────────┐                     │
│  STAGE 2: FEATURE ENGINEERING           │                     │
│  • Add prev_head_injury, hx_mood_disorder (V1 & V2)          │
│  • Add treatment_present (V2 only)      │                     │
│  • Compute _diff features (V2 only)     │                     │
└─────────────────────────────────────────┘                     │
       │                                                        │
       ▼                                                        │
┌─────────────────────────────────────────┐                     │
│  STAGE 3: HYPERPARAMETER TUNING         │                     │
│  • 80/20 stratified split               │                     │
│  • ParameterSampler, n_iter=50          │                     │
│  • class_weight="balanced"              │                     │
│  • Best params selected by val accuracy │                     │
└─────────────────────────────────────────┘                     │
       │  best_params (fixed)                                   │
       ▼                                                        │
┌─────────────────────────────────────────┐                     │
│  STAGE 4: LOOCV (N=217 folds)           │                     │
│  For i = 1 … 217:                       │                     │
│    train on N-1 samples                 │                     │
│    predict sample i                     │                     │
│    accumulate FI scores                 │                     │
└─────────────────────────────────────────┘                     │
       │  y_true, y_pred, y_prob (all N)                        │
       ▼                                                        │
┌─────────────────────────────────────────┐                     │
│  STAGE 5: EVALUATION & FI ANALYSIS      │                     │
│  • Metrics + 95% bootstrap CIs          │                     │
│  • Confusion matrices                   │                     │
│  • Top-20 FI tables                     │                     │
│  • Average effect plots                 │                     │
│  • Final model refit on full N          │◄───────────────────┘
└─────────────────────────────────────────┘
```

---

## Manuscript–Code Alignment

The table below maps every key methodological decision in the manuscript to its corresponding notebook cell(s) to facilitate direct verification.

| Manuscript Section | Notebook | Cell Description |
|---|---|---|
| *Cohort* — N=217 SRC patients | `visit1_REVISED-orginalset-orginalset` | Load CSV, define X and y |
| *Data Preprocessing* — iterative null filtering | `preprocessing` notebook | Missingness loop |
| *Data Preprocessing* — EPV ratio | Both ML notebooks | EPV computation & warning |
| *Feature Engineering* — Visit 1 | `visit1_REVISED-orginalset` | Drop non-predictors, add `prev_head_injury`, `hx_mood_disorder` |
| *Feature Engineering* — Visit 2 Δ features | `visit2_REVISED-originalExperimentSet`| Drop non-predictors, include `_diff` columns and `treatment_present` |
| *Hyperparameter Tuning* — LightGBM | `visit1_REVISED-orginalset` | `ParameterSampler`, 80/20 split |
| *Hyperparameter Tuning* — Decision Tree | `visit1_REVISED-orginalset`| `ParameterSampler`, 80/20 split |
| *Hyperparameter Tuning* — Random Forest | `visit1_REVISED-orginalset` | `ParameterSampler`, 80/20 split |
| *Hyperparameter Tuning* — XGBoost | `visit1_REVISED-orginalset`, Cell 23 | `ParameterSampler`, 80/20 split |
| *Hyperparameter Tuning* — SVC | `visit1_REVISED-orginalset`, Cell 27 | Linear kernel, 27 combinations |
| *Hyperparameter Tuning* — Ridge | `visit1_REVISED-orginalset`, Cell 31 | k-fold CV over alpha candidates |
| *LOOCV* — LightGBM | `visit1_REVISED-orginalset`, Cell 12 | LOOCV loop, gain-based FI |
| *LOOCV* — Decision Tree | `visit1_REVISED-orginalset`, Cell 16 | LOOCV loop, Gini FI |
| *LOOCV* — Random Forest | `visit1_REVISED-orginalset`, Cell 20 | LOOCV loop, Gini FI |
| *LOOCV* — XGBoost | `visit1_REVISED-orginalset`, Cell 24 | LOOCV loop, gain-based FI |
| *LOOCV* — SVC | `visit1_REVISED-orginalset`, Cell 28 | LOOCV loop, linear coefficients |
| *LOOCV* — Ridge | `visit1_REVISED-orginalset`, Cell 32 | LOOCV loop, permutation importance |
| *Results* — metrics + 95% CI | Both ML notebooks, plotting cells | `compute_metrics()` with bootstrap |
| *Results* — confusion matrices | Both ML notebooks, plotting cells | `plot_confusion()` |
| *Results* — top-20 FI tables | Both ML notebooks, plotting cells | `plot_feature_importance()` |
| *Results* — average effect plots | Both ML notebooks, plotting cells | `plot_effect_direction()` |
| *Results* — feature frequency (Tables 11–12) | Both ML notebooks, final summary cells | Cross-model FI aggregation |

---

## Reproducing Results

```bash
# 1. Run preprocessing
jupyter nbconvert --to notebook --execute preprocessing/data_preprocessing.ipynb

# 2. Run Visit 1 ML pipeline
jupyter nbconvert --to notebook --execute machineLearning_visit1_REVISED-orginalset.ipynb

# 3. Run Visit 2 ML pipeline
jupyter nbconvert --to notebook --execute machineLearning_visit2_REVISED-originalExperimentSet.ipynb
```

All random operations use `random_state=42`. LOOCV results are fully deterministic given the same input data and best hyperparameters.

> **Expected runtimes:** Visit 1 LOOCV ≈ 10–20 min per model on a modern CPU; Visit 2 LOOCV is longer due to the larger feature set (95 features). Total end-to-end runtime for all six models per visit is approximately 2–4 hours without GPU acceleration.

---

## Citation

```bibtex
@article{tran2025src,
  title   = {Predicting Time to Clearance of Sport-Related Concussions Using Machine Learning},
  author  = {Tran, Megan and Holler, Jessica and Moran, Byron and Schilaty, Nathan D. and Templeton, John Michael},
  journal = {Digital Health},
  year    = {2025},
  note    = {Under review}
}
```

**Corresponding author:** John M. Templeton — jtemplet@usf.edu  
**Code author:** Megan Tran — [GitHub](https://github.com/MeganTran6023/Sport-Related-Concussions_Machine-Learning)
