# Eisenhower Estimator

ML system that predicts whether a task is **important** and/or **urgent** — the two axes of the Eisenhower Matrix — and **how long it will take**, using data from a personal task manager.

---

## What It Does

Given a new task (description, project, creation time, due date), it predicts:

| Model | Target | Type | Imbalance / Distribution | Dataset | Status |
|---|---|---|---|---|---|
| **Model 1** | `important` (binary) | Classification | 3.2:1 | 4,744 rows | Trained (v2026-04-06) |
| **Model 2** | `urgent` (binary) | Classification | 17.4:1 | 4,744 rows | Trained (v2026-04-06) — F1-urgent=0.719, ROC-AUC=0.987 |
| **Model 3** | `duration_minutes` (continuous) | Regression | Right-skewed (log1p transform) | 1,537 rows | Training pipeline ready |
| **Model 3b** | `duration_bucket` (6 ordinal classes) | Classification | ≤2 min (15%), 3–5 min (35%), 6–10 min (37%), 11–30 min (10%), 31–60 min (2%), >60 min (2%) | 1,537 rows | **Trained (v2026-04-06)** — F1-macro=0.494 (test), Accuracy=73.4% |

All models are trained with Optuna hyperparameter search, tracked with MLflow, and registered in the MLflow Model Registry. Model 1 also includes a **rule-based baseline** and a **hybrid classifier** (rule gates + ML) evaluated on the same holdout set for direct comparison.

---

## Architecture

```
eisenhower_estimator/
├── src/
│   ├── data/
│   │   └── extractor.py                    # SQLite → parquet pipeline
│   ├── evaluation/                         # Comparable evaluation harness (NEW)
│   │   ├── __init__.py
│   │   ├── rule_based.py                   # RuleBasedClassifier — pure rule engine
│   │   ├── hybrid.py                       # HybridClassifier — rules + ML
│   │   └── evaluator.py                    # ModelEvaluator — side-by-side metrics
│   ├── transformers/                       # Reusable sklearn transformers
│   │   ├── cyclical_encoder.py             # sin/cos for hour, day_of_week, month
│   │   ├── target_encoder.py               # Target encoding (per-fold, leak-safe)
│   │   ├── text_vectorizer.py              # TF-IDF / BoW + optional LSA
│   │   ├── scaler_selector.py              # Configurable scaler
│   │   ├── resampler.py                    # SMOTE / ADASYN / none
│   │   ├── feature_selector.py             # SelectKBest (f_classif / mutual_info)
│   │   └── dimensionality_reducer.py       # PCA / LDA / none
│   └── training/
│       ├── base_objective.py               # Abstract BaseObjective + shared CV utilities
│       ├── pipeline_builder.py             # ColumnTransformer + classifier (Model 1)
│       ├── pipeline_builder_urgent.py      # ColumnTransformer + classifier (Model 2)
│       ├── pipeline_builder_time_spent.py  # ColumnTransformer + regressor (Model 3) + TaskCVImputer
│       ├── pipeline_builder_time_bucket.py # ColumnTransformer + classifier (Model 3b)
│       ├── optuna_objective.py             # Optuna search space — Model 1
│       ├── optuna_objective_urgent.py      # Optuna search space — Model 2
│       ├── optuna_objective_time_spent.py  # Optuna search space — Model 3 (regression)
│       ├── optuna_objective_time_bucket.py # Optuna search space — Model 3b (multiclass)
│       ├── trainer.py                      # ModelTrainer — Model 1 end-to-end (+ system eval)
│       ├── urgent_trainer.py               # UrgentModelTrainer — Model 2 end-to-end
│       ├── time_spent_trainer.py           # TimeSpentTrainer — Model 3 end-to-end
│       └── time_bucket_trainer.py          # TimeBucketTrainer — Model 3b end-to-end
├── notebooks/
│   ├── eda_important.py                    # EDA — Model 1 (# %% cells, generates versioned report)
│   ├── eda_urgent.py                       # EDA — Model 2 (# %% cells)
│   ├── eda_time_spent.py                   # EDA — Model 3 & 3b (# %% cells)
│   ├── train_important_model.py            # Training entry point — Model 1
│   ├── train_urgent_model.py               # Training entry point — Model 2
│   ├── train_time_spent_model.py           # Training entry point — Model 3
│   └── train_time_bucket_model.py          # Training entry point — Model 3b
├── configs/
│   ├── config.yaml                         # Global config (paths, MLflow, data.version, EDA settings)
│   ├── model1_config.yaml                  # Model 1 Optuna search space and settings
│   ├── model2_config.yaml                  # Model 2 Optuna search space and settings
│   ├── model3_config.yaml                  # Model 3 Optuna search space and settings
│   └── model3b_config.yaml                 # Model 3b Optuna search space and settings
├── data/
│   ├── raw/todo_tasks.parquet              # Source data from SQLite (4,744 rows as of 2026-04-06)
│   └── processed/
│       ├── eda_important_features.parquet  # 4,744 × 17 — Model 1 input
│       ├── eda_urgent_features.parquet     # 4,744 × 16 — Model 2 input
│       └── eda_time_spent_features.parquet # 1,307 × 26 — Model 3 & 3b input
├── docs/
│   ├── eda_important_report.md             # Latest EDA report (canonical)
│   ├── eda_important_report_2026-04-06.md  # Versioned copy (date-stamped)
│   ├── eda_urgent_report.md
│   ├── eda_time_spent_report.md
│   └── figures/
├── models/                                 # Local model artifacts (.pkl)
├── mlruns/                                 # MLflow artifact store
└── mlflow.db                               # MLflow backend (SQLite)
```

---

## Versioning

Every MLflow run is tagged with three provenance fields so any result can be reproduced exactly:

| MLflow Tag | Value | Purpose |
|---|---|---|
| `data_version` | ISO date string (e.g. `2026-04-06`) | Identifies the data snapshot — set in `configs/config.yaml` under `data.version` |
| `data_sha256` | SHA-256 hex digest of the processed parquet | Detects silent data overwrites |
| `git_commit` | Short 8-char commit hash | Pins the code state |
| `config_version` | ISO date string from model config | Tracks search space changes (Model 3b) |

For Model 3b, the extractor also writes `data/raw/time_spent_tasks_metadata.json` — a JSON sidecar with extraction timestamp, row count, date range, project distribution, and raw parquet SHA-256. This is the lineage anchor: every downstream artifact can reference it to prove data provenance.

To compare runs across data versions in the MLflow UI, filter by `data_version` tag in the experiment view.

### How to reproduce a specific run

```bash
# 1. Check out the git commit from the run's git_commit tag
git checkout <git_commit>

# 2. Verify the parquet hash matches data_sha256
python -c "import hashlib; print(hashlib.sha256(open('data/processed/eda_important_features.parquet','rb').read()).hexdigest())"

# 3. Re-run training with the same config
python notebooks/train_important_model.py
```

### Versioned EDA reports

Each run of `notebooks/eda_important.py` writes two report files:
- `docs/eda_important_report.md` — canonical (always latest)
- `docs/eda_important_report_<data_version>.md` — date-stamped copy (never overwritten)

---

## System Evaluation — Rule-Based vs ML vs Hybrid

Model 1 is evaluated across three approaches on the **same holdout test set** for direct comparison:

| Approach | Description |
|---|---|
| **Rule-based** | Domain-knowledge rules (e.g. "early-morning work tasks are important"). Fully explainable, zero training. |
| **ML model** | Best Optuna trial pipeline (LightGBM / XGBoost / RF / LR + preprocessor). |
| **Hybrid** | Rules fire first; undecided rows are passed to the ML model. Combines explainability + accuracy. |

Results from the quick-run validation (5 Optuna trials, data v2026-04-06):

| Metric | Rule-Based | ML Model | Hybrid |
|---|---|---|---|
| F1-macro | 0.4324 | 0.9781 | **0.9810** |
| F1 (not important) | 0.0000 | 0.9666 | **0.9709** |
| F1 (important) | 0.8648 | 0.9896 | **0.9910** |
| ROC-AUC | 0.508 | 0.997 | **0.998** |
| Rule coverage | 1.2% | N/A | 1.2% |

The hybrid approach wins on all metrics. Rule coverage is low (1.2%) because the current rules are conservative (they only fire when confidence is very high). This is by design — see `src/evaluation/rule_based.py` to add more rules.

Every evaluation run is logged as a nested MLflow child run under the parent training run, with confusion matrices and classification reports per approach.

---

## Key Features

### Custom Reusable sklearn Transformers
All transformers in `src/transformers/` implement `BaseEstimator` + `TransformerMixin` and work with any sklearn Pipeline:

| Transformer | Purpose |
|---|---|
| `CyclicalEncoder` | Encodes periodic features as sin/cos pairs (configurable period: 24h, 7d, 12m) |
| `SafeTargetEncoder` | Target-encodes categoricals with per-fold fitting and unseen-category fallback to global mean |
| `TextVectorizerTransformer` | TF-IDF or BoW + optional TruncatedSVD (LSA) to reduce memorisation risk |
| `ScalerSelector` | Swappable scaler: receives the scaler name — `standard` / `robust` / `minmax` / `none` |
| `ResamplerTransformer` | SMOTE / ADASYN / none — applied ONLY to training folds, never test data |
| `FeatureSelectorTransformer` | SelectKBest (f_classif or mutual_info) or passthrough |
| `DimensionalityReducerTransformer` | PCA / LDA / none — post-selector reduction |

### TaskCVImputer (Model 3 & 3b)
Custom sklearn transformer that handles leakage-prone per-task features (`task_cv`, `task_median_duration`). During cross-validation, it recomputes per-task statistics from **only the training fold** to prevent data leakage. It uses a 3-tier fallback: task_description → task_type mean → global mean.

### Multi-Model Optuna Search
Optuna (TPE sampler) searches across 7 model types simultaneously:

**Models 1 & 2 (Classification):**

| Model type | Notes |
|---|---|
| `lgbm` | LightGBM — fast, handles mixed types natively |
| `xgboost` | XGBoost — strong baseline |
| `random_forest` | Sklearn RF with balanced class weights |
| `logistic` | Logistic Regression — linear baseline |
| `voting_lgbm_rf` | Soft VotingClassifier: LGBM + RF |
| `voting_lgbm_lr` | Soft VotingClassifier: LGBM + LR |
| `voting_all` | Soft VotingClassifier: LGBM + XGB + RF + LR |

**Model 3 (Regression):**

| Model type | Notes |
|---|---|
| `lgbm` | LightGBM regressor |
| `xgboost` | XGBoost regressor |
| `random_forest` | Sklearn RF regressor |
| `ridge` | Ridge Regression — linear baseline |
| `voting_lgbm_rf` | Voting: LGBM + RF |
| `voting_lgbm_ridge` | Voting: LGBM + Ridge |
| `voting_all` | Voting: LGBM + XGB + RF + Ridge |

**Model 3b (Multiclass Classification):**
Same 7 classifier types as Models 1 & 2, with multiclass support (`multi:softprob` for XGBoost). Supports SMOTE resampling and LDA dimensionality reduction.

### MLflow Experiment Tracking
Every Optuna trial is a **nested MLflow child run** under a parent training run:

```
Experiment: "important-classifier"  /  "urgent-classifier"  /  "time-spent-regressor"  /  "time-bucket-classifier"
└── Parent run: training_<timestamp>
    ├── trial_0000  (lgbm, smote, tfidf+lsa)
    ├── trial_0001  (voting_all, none, bow)
    ...
    └── trial_0049
```

Each run logs: params, CV metrics, confusion matrix (classifiers) or residual plots (regressor), ROC curve (binary), feature importances, Optuna visualisations, and best hyperparameters JSON.

**Model 3 specific:** Logs RMSE (log scale), MAE (minutes), Median AE, R² (log scale), predicted vs actual scatter, residual distribution, segmented error analysis.

**Model 3b specific:** Logs macro/weighted F1, per-class F1, confusion matrix heatmap, per-class F1 bar chart, classification report.

### Leakage Guards
| Guard | Enforcement |
|---|---|
| `is_overdue` excluded from both models | Not available at task creation time |
| `urgent` excluded from Model 1 | Target leakage |
| `important` excluded from Model 2 | Target leakage |
| `SafeTargetEncoder` fit per CV fold | Prevents project-code encoding leakage |
| `ResamplerTransformer` applied train-fold only | SMOTE never sees validation data |
| `StratifiedGroupKFold` for Models 2, 3, 3b | Tests generalisation to unseen projects |
| `TaskCVImputer` recomputes per train-fold (Model 3 & 3b) | `task_cv` and `task_median_duration` never leak test-fold statistics |

### Model-Specific Design Choices

| Aspect | Model 1 (Important) | Model 2 (Urgent) | Model 3 (Time Spent) | Model 3b (Time Bucket) |
|---|---|---|---|---|
| Task type | Binary classification | Binary classification | Regression | Multiclass classification (6 classes) |
| Target | `important` | `urgent` | `log1p(duration_minutes)` | `duration_bucket` (ordinal) |
| Imbalance | 3:1 | 21.36:1 | N/A (continuous) | Highly imbalanced (37% / 2% extremes) |
| Optimisation metric | F1-macro | F1-macro | RMSE (log scale) | F1-macro |
| CV strategy | `StratifiedKFold(5)` | `StratifiedGroupKFold(5)` | `StratifiedGroupKFold(5)` | `StratifiedGroupKFold(5)` |
| Dataset | 4,744 × 17 | 4,744 × 16 | 1,537 × 26 | 1,537 × 26 |
| Features | 8 engineered | 11 engineered | 14 engineered | 14 engineered (same as Model 3) |
| Resampling | SMOTE/ADASYN/none | SMOTE/ADASYN/none | N/A | SMOTE/none |
| Dim. reduction | PCA/LDA/none | PCA/LDA/none | PCA/none | PCA/LDA/none |
| ICC ceiling | N/A | N/A | 0.680 | N/A |

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo>
cd eisenhower_estimator
python -m venv eisen_mat_env

# Activate (Windows)
source eisen_mat_env/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Train

### Model 1 — Important Classifier

```bash
source eisen_mat_env/Scripts/activate
cd eisenhower_estimator

# Step 1: extract fresh data from SQLite
python src/data/extractor.py configs/config.yaml

# Step 2: run EDA (generates processed parquet + versioned report)
python notebooks/eda_important.py

# Step 3: train (quick run = 5 trials; set QUICK_RUN=False for full 50-trial search)
python notebooks/train_important_model.py
```

Or open `notebooks/train_important_model.py` in VSCode and run `# %%` cells interactively.

**Quick smoke test** (5 trials): set `QUICK_RUN = True` in the notebook (default).  
**Full training** (50 trials): set `QUICK_RUN = False`.

Training automatically produces a **3-way system comparison** (rule-based / ML / hybrid) logged as MLflow child runs.

### Model 2 — Urgent Classifier

```bash
source eisen_mat_env/Scripts/activate
cd eisenhower_estimator

# Step 1: extract fresh data from SQLite
python src/data/extractor.py configs/config.yaml

# Step 2: run EDA (generates processed parquet + versioned report + versioned figures)
python notebooks/eda_urgent.py

# Step 3: train (quick run = 5 trials; set QUICK_RUN=False for full 50-trial search)
python notebooks/train_urgent_model.py
```

Or open `notebooks/train_urgent_model.py` in VSCode and run `# %%` cells interactively.

**Quick smoke test** (5 trials): set `QUICK_RUN = True` in the script (default).  
**Full training** (50 trials): set `QUICK_RUN = False`.

Each run is tagged with `data_version`, `data_sha256` (SHA-256 of processed parquet), and `git_commit` for full traceability. Versioned EDA reports are saved to `docs/eda_urgent_report_<data_version>.md` (never overwritten); the canonical report is always `docs/eda_urgent_report.md`.

> **Note**: Model 2 uses `StratifiedGroupKFold(group=project_code)`. Some folds may have very few
> urgent samples (17.4:1 imbalance, 258 urgent out of 4,744). This is expected — the trainer handles
> it gracefully with neutral scores and a warning log. Watch `test_f1_1` (urgent class F1) — not just
> `test_f1_macro`. A model predicting all "not urgent" achieves `test_f1_macro ≈ 0.48` but
> `test_f1_1 = 0.0` — failure mode. Target: `test_f1_1 > 0.5`.

### Model 3 — Time Spent Regressor

```bash
source eisen_mat_env/Scripts/activate
cd eisenhower_estimator
python notebooks/train_time_spent_model.py
```

Or open `notebooks/train_time_spent_model.py` in VSCode and run `# %%` cells interactively.

**Quick smoke test** (5 trials): set `QUICK_RUN = True` in the script.

> **Note**: Model 3 predicts `log1p(duration_minutes)` and uses `StratifiedGroupKFold(group=project_code)`.
> The ICC ceiling for this target is ≈ 0.695 — the same task description done by the same user shows
> high variance in actual duration, so perfect prediction is not possible. Focus on **RMSE (log scale)**
> and **Median Absolute Error (minutes)** rather than raw MAE, which is heavily influenced by outliers.

### Model 3b — Time Bucket Classifier

```bash
source eisen_mat_env/Scripts/activate
cd eisenhower_estimator

# Step 1: extract fresh data from SQLite (writes parquet + metadata JSON with SHA-256)
python src/data/extractor_time_spent.py configs/config.yaml

# Step 2: run EDA (generates processed parquet, versioned report, versioned figures)
python notebooks/eda_time_spent.py

# Step 3: train (quick run = 5 trials; set QUICK_RUN=False for full 100-trial search)
python notebooks/train_time_bucket_model.py
```

Or open `notebooks/train_time_bucket_model.py` in VSCode and run `# %%` cells interactively.

**Quick smoke test** (5 trials): set `QUICK_RUN = True` in the script.  
**Full training** (100 trials): set `QUICK_RUN = False`.

Each run is tagged with `data_version`, `data_sha256` (SHA-256 of processed parquet), `config_version`, and `git_commit` for full traceability. The extractor also writes `data/raw/time_spent_tasks_metadata.json` with extraction timestamp, row count, date range, and raw parquet SHA-256 as the lineage anchor. Versioned EDA reports are saved to `docs/eda_time_spent_report_<data_version>.md` (never overwritten); the canonical report is always `docs/eda_time_spent_report.md`.

> **Note**: Model 3b classifies tasks into 6 duration buckets (≤2 min, 3–5 min, 6–10 min, 11–30 min,
> 31–60 min, >60 min). The last two buckets have very few samples (≈2% each). Watch **per-class F1**
> closely — the model struggles with the two heaviest tails. SMOTE resampling and `class_weight='balanced'`
> are explored in the search space. The CV F1-macro (~0.24) is lower than holdout (~0.49) due to
> `StratifiedGroupKFold` leaving entire projects out of each fold, which is the correct evaluation strategy.

#### Model 3b — Training Results (v2026-04-06)

| Metric | Value |
|---|---|
| MLflow Run ID | `24fc0cee0566469584def74c3f23fdb5` |
| Best trial | #24 |
| Best CV F1-macro (StratifiedGroupKFold) | 0.2392 |
| **Test F1-macro** | **0.4938** |
| Test F1-weighted | 0.7333 |
| Test Precision (macro) | 0.5021 |
| Test Recall (macro) | 0.4919 |
| Test Accuracy | 73.4% |
| Best model type | `voting_lgbm_lr` (LightGBM + LogReg ensemble) |
| Resampling | SMOTE |
| Data | 1,537 rows (v2026-04-06) |

Per-class F1 on holdout test set:

| Bucket | F1 | Note |
|---|---|---|
| ≤2 min | 0.691 | Exercise tasks — well separated |
| 3–5 min | 0.759 | Short admin/updates |
| 6–10 min | 0.860 | Most common — strong signal |
| 11–30 min | 0.471 | Mixed types — harder |
| 31–60 min | 0.182 | Few samples, high variance |
| >60 min | 0.000 | Only 21 train / 5 test samples — insufficient data |

**Quality note**: The model learns the 3 dominant buckets (≤2, 3–5, 6–10 min) well (F1 > 0.69). The two minority buckets (31–60 min, >60 min) are structurally limited by sample size — the ICC of 0.68 sets a theoretical ceiling and the project-grouped CV ensures these results are honest out-of-distribution estimates.

---

## How to View Results in MLflow UI

This project stores **all run metadata in a SQLite file** (`mlflow.db`) located at the project root. You must point the UI at this file, not at the `mlruns/` folder:

```bash
# From the project root folder
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

> **Important**: Running `mlflow ui` without `--backend-store-uri sqlite:///mlflow.db` will open an empty UI.
> The `mlruns/` folder only holds large binary artifacts (plots, models). All metrics, params, and run metadata live in `mlflow.db`.

Open [http://localhost:5000](http://localhost:5000)

You will see four experiments:
- **`important-classifier`** — Model 1 runs
- **`urgent-classifier`** — Model 2 runs
- **`time-spent-regressor`** — Model 3 runs
- **`time-bucket-classifier`** — Model 3b runs

Each experiment shows:
- All Optuna trials ranked by the primary metric (F1-macro or RMSE)
- Confusion matrix (classifiers) or scatter/residual plots (regressor) for the best model
- Optuna optimization history and parameter importance plots
- Best hyperparameters JSON
- (Model 2 only) Class distribution plot (train vs test)
- (Model 3 only) Segmented error analysis, predicted vs actual scatter
- (Model 3b only) Per-class F1 bar chart

---

## How to Load the Trained Models

### Important Classifier (Model 1)

```python
import mlflow.sklearn

# Load from MLflow registry (after training)
model = mlflow.sklearn.load_model("models:/important-classifier/Production")

# Or load from local file
import joblib
model = joblib.load("models/important_classifier_best.pkl")

# Predict on new tasks
import pandas as pd
X_new = pd.DataFrame([{
    "hour_created": 9,
    "day_of_week_created": 1,          # Tuesday
    "project_code": "SBM",
    "project_category": "Work",
    "days_until_due": 14,
    "desc_word_count": 8,
    "desc_char_len": 45,
    "desc_clean": "revisar relatorio mensal producao",
}])

prediction = model.predict(X_new)          # 0 = not important, 1 = important
probability = model.predict_proba(X_new)   # [[prob_0, prob_1]]
```

**Note**: New `project_code` values not seen during training fall back to the global target mean.

### Urgent Classifier (Model 2)

```python
import mlflow.sklearn

# Load from MLflow registry (after training)
model = mlflow.sklearn.load_model("models:/urgent-classifier/Production")

# Or load from local file
import joblib
model = joblib.load("models/urgent_classifier_best.pkl")

# Predict on new tasks
import pandas as pd
X_new = pd.DataFrame([{
    "hour_created": 14,
    "day_of_week_created": 3,            # Wednesday
    "month_created": 3,                  # March
    "project_code": "PROJ_XYZ",
    "project_category": "Work",
    "lead_time_bucket": "short",         # one of: overdue_long, overdue_short, very_short, short, medium, long
    "lead_time_days": 5,                 # can be negative for retroactive tasks
    "lead_time_hours": 120,              # can be negative
    "desc_word_count": 6,
    "desc_char_len": 35,
    "desc_clean": "enviar relatorio para reunião",
}])

prediction = model.predict(X_new)          # 0 = not urgent, 1 = urgent
probability = model.predict_proba(X_new)   # [[prob_0, prob_1]]
```

**Notes:**
- `project_category` must be `"Work"` or `"Personal"` as a string
- If `project_category = "Personal"`, the model will almost certainly predict `not_urgent` (no Personal tasks were ever urgent in training data)
- `lead_time_days` and `lead_time_hours` can be negative (retroactive/recurring tasks) — expected
- New `project_code` values fall back to the global target mean

### Time Spent Regressor (Model 3)

```python
import mlflow.sklearn

# Load from MLflow registry (after training)
model = mlflow.sklearn.load_model("models:/time-spent-regressor/Production")

# Or load from local file
import joblib
model = joblib.load("models/time_spent_regressor_best.pkl")

# Predict on new tasks
import numpy as np
import pandas as pd
X_new = pd.DataFrame([{
    "day_of_week": 1,                    # Tuesday
    "project_code": "SBM",
    "task_type": "admin_update",
    "has_number": 0,
    "is_repeated_task": 1,
    "task_freq": 12,
    "task_cv": 0.35,                     # NaN if unknown — TaskCVImputer handles it
    "desc_char_len": 40,
    "desc_clean": "atualizar planilha mensal",
    "task_type_x_repeated": "admin_update_1",
    "is_long_project": 0,
    "log_task_freq": 2.485,              # log(task_freq)
    "desc_has_time_ref": 0,
    "task_median_duration": np.nan,       # NaN — TaskCVImputer handles it
}])

prediction_log = model.predict(X_new)                      # log1p scale
prediction_minutes = np.expm1(prediction_log)               # convert to minutes
print(f"Estimated time: {prediction_minutes[0]:.1f} minutes")
```

**Notes:**
- The model predicts on `log1p(duration_minutes)` scale — use `np.expm1()` to convert back
- `task_cv` and `task_median_duration` can be `NaN` — the `TaskCVImputer` in the pipeline handles imputation
- `task_type_x_repeated` is the interaction between `task_type` and `is_repeated_task` (e.g. `"admin_update_1"`)
- The ICC ceiling for this target is ≈ 0.695, so even a perfect model would still have significant error

### Time Bucket Classifier (Model 3b)

```python
import mlflow.sklearn

# Load from MLflow registry (after training)
model = mlflow.sklearn.load_model("models:/time-bucket-classifier/Production")

# Or load from local file
import joblib
model = joblib.load("models/time_bucket_classifier_best.pkl")

# Predict on new tasks — uses the same 14 features as Model 3
import pandas as pd
X_new = pd.DataFrame([{
    "day_of_week": 3,                    # Wednesday
    "project_code": "IAMZ",
    "task_type": "execute_task",
    "has_number": 0,
    "is_repeated_task": 0,
    "task_freq": 1,
    "task_cv": None,                     # NaN — imputed by TaskCVImputer
    "desc_char_len": 55,
    "desc_clean": "preparar apresentacao reuniao",
    "task_type_x_repeated": "execute_task_0",
    "is_long_project": 1,
    "log_task_freq": 0.0,
    "desc_has_time_ref": 0,
    "task_median_duration": None,         # NaN — imputed by TaskCVImputer
}])

prediction = model.predict(X_new)          # 0–5 (ordinal bucket index)
probability = model.predict_proba(X_new)   # [[prob_0, ..., prob_5]]

bucket_labels = ["≤2 min", "3–5 min", "6–10 min", "11–30 min", "31–60 min", ">60 min"]
print(f"Predicted bucket: {bucket_labels[prediction[0]]}")
```

---

## Configuration

### `configs/config.yaml`
Global settings: data paths, MLflow tracking URI, experiment names, training defaults (CV folds, test size, random seed, Optuna trials).

### `configs/model1_config.yaml`
Model 1 specific settings: Optuna search space ranges, allowed model types, resampling options.

### `configs/model2_config.yaml`
Model 2 specific settings: `scale_pos_weight` range (10–30), `StratifiedGroupKFold` notes, lead_time handling, resampling recommendations for 21.36:1 imbalance.

### `configs/model3_config.yaml`
Model 3 specific settings: regression search space, feature selection strategies (`selectkbest_f_reg`, `selectkbest_mi_reg`), PCA dimensionality reduction. No resampling (continuous target).

### `configs/model3b_config.yaml`
Model 3b specific settings: multiclass classification search space, SMOTE resampling search, LDA dimensionality reduction (classification-only), `class_weight` options for handling 6 imbalanced buckets.

To change the number of Optuna trials:
```yaml
# configs/model3_config.yaml  or  model3b_config.yaml
training:
  optuna_trials: 100
  optuna_timeout_seconds: 7200
```

---

## EDA Reports

| Report | Key Finding |
|---|---|
| [eda_important_report.md](docs/eda_important_report.md) | `hour_created` (V=0.80) is the strongest predictor. 75/25 class split. |
| [eda_urgent_report.md](docs/eda_urgent_report.md) | `project_code` (V=0.635) is strongest. 95.5/4.5 severe imbalance. Urgent = Work only. |
| [eda_time_spent_report.md](docs/eda_time_spent_report.md) | `is_repeated_task` (r=-0.320 with error) is strongest. ICC ceiling=0.695. Right-skewed distribution with median 5 min. |

---

## Data Pipeline

```
SQLite DB (personal_organizer.db)
    ↓  src/data/extractor.py
data/raw/todo_tasks.parquet
    ↓  notebooks/eda_important.py  /  notebooks/eda_urgent.py  /  notebooks/eda_time_spent.py
data/processed/eda_important_features.parquet   (3,689 × 17)
data/processed/eda_urgent_features.parquet      (3,689 × 16)
data/processed/eda_time_spent_features.parquet  (1,307 × 26)
    ↓  notebooks/train_*.py
models/important_classifier_best.pkl
models/urgent_classifier_best.pkl
models/time_spent_regressor_best.pkl
models/time_bucket_classifier_best.pkl
mlruns/   (MLflow artifacts + model registry)
```

> **Note**: Model 3 & 3b use a smaller dataset (1,307 rows) because only completed tasks with
> recorded `duration_minutes` can be used for time estimation. The parquet includes 14 engineered
> features: 9 original + 5 iteration-2 features from error analysis (`task_type_x_repeated`,
> `is_long_project`, `log_task_freq`, `desc_has_time_ref`, `task_median_duration`).

---

## Prioritization Algorithm — Using the Models Together

The ultimate goal is a **MCDM-style (Multi-Criteria Decision Making) prioritization score** that combines the four model outputs into a single actionable rank for each task:

```
Priority Score = f( important, urgent, estimated_duration )
```

For example, a weighted formula:

```python
score = w1 * P(important=1)
      + w2 * P(urgent=1)
      - w3 * estimated_duration_penalty
```

where `estimated_duration_penalty` captures that a task taking 60 min should be considered "heavier" than one taking 2 min — and therefore scored lower all else being equal (unless urgency is high).

The key open question is: **which time prediction model should feed this formula?**

---

### Model 3 — Continuous Regressor

**Output**: a continuous estimate in minutes (after `np.expm1()` conversion)

**Strengths:**
- Naturally integrates into a numeric formula — can express `30 min` vs `31 min`
- Directly interpretable as a penalty: `penalty = log1p(duration_minutes)` avoids extreme sensitivity to outliers
- Median AE ≈ 0.73 min — for short tasks (the majority), point estimates are quite close

**Error patterns and flaws:**
- **Systematic under-prediction for long tasks**: The model collapses toward 10–15 min for tasks that actually take 30–120+ min. A 60 min task may be predicted as 12 min, meaning it will be ranked *too favourably* in the priority score
- **High variance for non-repeated tasks**: MAE for repeated tasks is ~1.7 min; for non-repeated (freq=1) tasks it explodes to ~22 min — these are exactly the "unknown" tasks that are hardest to estimate
- **Outlier sensitivity**: A handful of extreme tasks (SBM project, 200–750 min actual) completely dominate raw MAE. The log1p target helps during training but predictions are still poor in those regimes
- **ICC ceiling of 0.695**: Even a perfect model would struggle here — the same task description done on different days has inherently high time variance. R² on the raw minute scale is near zero (R²≈-0.006)
- **Implication for scoring**: The continuous penalty is noise-heavy for rare long tasks. If you use the raw prediction to penalize priority, long-but-unknown tasks get unfairly high scores

---

### Model 3b — Bucket Classifier

**Output**: predicted class in `{≤2 min, 3–5 min, 6–10 min, 11–30 min, 31–60 min, >60 min}` (6 buckets), plus class probabilities

**Strengths:**
- **Works well for the common case**: 87% of tasks fall in the first three buckets (≤10 min), which are predicted with F1 > 0.60. Weighted F1 = 0.677 reflects this
- **Buckets match human intuition**: It's more useful to know a task is "~5 min" vs "~45 min" than to know it's "6 min vs 7 min" — categories are adequate for scoring
- **Robustness to outliers**: A task that takes 750 min doesn't derail the classifier — it either correctly lands in `>60 min` or is mis-classified by one bucket, not off by 700 min
- **Probability output**: `predict_proba()` gives a soft signal — e.g. 60% `6–10 min` and 35% `11–30 min` is more honest than a false-precision point estimate

**Error patterns and flaws:**
- **Macro F1 = 0.42 is misleading**: It is dragged down by the two rarest classes (`31–60 min` and `>60 min`, 2% each). The model rarely predicts these correctly — it tends to predict the adjacent bucket
- **Adjacent-bucket errors dominate**: Most errors are ±1 bucket (e.g. predicting `3–5 min` for a `6–10 min` task). This is tolerable for scoring since the penalty difference between adjacent buckets is small
- **Minority bucket collapse**: The `31–60 min` and `>60 min` buckets are almost always missed. If the scoring formula gives a large penalty to these buckets, a 45 min task will routinely escape with a 10 min penalty
- **Same root cause as regressor**: Repeated/known tasks are well-classified; novel one-off tasks remain unpredictable regardless of the model form

---

### Recommendation

**Use the Bucket Classifier (Model 3b) as the primary time input to the scoring formula**, with the following rationale:

1. **Buckets are the right granularity for prioritization.** The difference between 8 min and 12 min is irrelevant when ranking 50 tasks. The difference between `≤5 min` and `>30 min` is meaningful. Use ordinal bucket indices (0–5) or midpoint values (`[1, 4, 8, 20, 45, 90]` min) as the penalty input.

2. **The regressor's false precision misleads the scoring formula.** A continuous estimate of 13.2 min for a task that actually took 60 min quietly inflates that task's priority. A bucket mis-classification of `6–10 min` instead of `11–30 min` is a smaller and more bounded error.

3. **Use class probabilities for a soft penalty.** Instead of a hard bucket index, compute the expected duration as a weighted sum of bucket midpoints:
   ```python
   midpoints = [1, 4, 8, 20, 45, 90]   # minutes per bucket
   proba = model_3b.predict_proba(X_new)[0]   # shape: (6,)
   expected_minutes = sum(p * m for p, m in zip(proba, midpoints))
   ```
   This retains uncertainty information and degrades more gracefully for ambiguous tasks.

4. **For tasks with very high urgency or importance, the time estimate matters less.** Apply the duration penalty only when `P(urgent) < threshold` — a truly urgent task should be done immediately regardless of its estimated duration.

5. **Acknowledge the ICC ceiling.** Even with the best model, ~30% of variance in task duration is fundamentally unpredictable from description alone. Design the scoring formula so that the time penalty has a lower weight than importance/urgency — the classifier provides signal but not certainty.

**Suggested scoring formula:**

```python
import numpy as np

def prioritization_score(
    p_important: float,   # P(important=1) from Model 1
    p_urgent: float,      # P(urgent=1) from Model 2
    bucket_proba: list,   # predict_proba output from Model 3b (6 values)
    w_important: float = 0.40,
    w_urgent: float = 0.45,
    w_duration: float = 0.15,
) -> float:
    """
    Higher score = higher priority.
    Duration penalty is log-scaled so a 90 min task doesn't dominate.
    """
    midpoints = [1, 4, 8, 20, 45, 90]
    expected_min = sum(p * m for p, m in zip(bucket_proba, midpoints))
    duration_penalty = np.log1p(expected_min) / np.log1p(90)  # normalised 0–1

    return (
        w_important * p_important
        + w_urgent * p_urgent
        - w_duration * duration_penalty
    )
```

> The weights above are a starting point. Calibrate them on your own task history by asking: "which tasks do I wish I had done first?"

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | Pipelines, ColumnTransformer, base estimators |
| `lightgbm` | Gradient boosting classifier |
| `xgboost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE, ADASYN |
| `optuna` | Bayesian hyperparameter search (TPE) |
| `mlflow` | Experiment tracking + model registry |
| `category-encoders` | Target encoding utilities |
| `pandas` / `pyarrow` | Data loading (parquet) |
| `loguru` | Structured logging |
| `matplotlib` / `seaborn` | EDA visualisations and training plots |
