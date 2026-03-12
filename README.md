# Eisenhower Estimator

ML system that predicts whether a task is **important** and/or **urgent** — the two axes of the Eisenhower Matrix — using data from a personal task manager.

---

## What It Does

Given a new task (description, project, creation time, due date), it predicts:

| Model | Target | Imbalance | Status |
|---|---|---|---|
| **Model 1** | `important` (binary) | 3:1 | Training pipeline complete |
| **Model 2** | `urgent` (binary) | 21.36:1 | Training pipeline complete |

Both models are trained with Optuna hyperparameter search, tracked with MLflow, and registered in the MLflow Model Registry for consumption by downstream applications.

---

## Architecture

```
eisenhower_estimator/
├── src/
│   ├── data/
│   │   └── extractor.py                    # SQLite → parquet pipeline
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
│       ├── optuna_objective.py             # Optuna search space — Model 1
│       ├── optuna_objective_urgent.py      # Optuna search space — Model 2
│       ├── trainer.py                      # ModelTrainer — Model 1 end-to-end
│       └── urgent_trainer.py               # UrgentModelTrainer — Model 2 end-to-end
├── notebooks/
│   ├── eda_important.py                    # EDA — Model 1 (# %% cells)
│   ├── eda_urgent.py                       # EDA — Model 2 (# %% cells)
│   ├── train_important_model.py            # Training entry point — Model 1
│   └── train_urgent_model.py               # Training entry point — Model 2
├── configs/
│   ├── config.yaml                         # Global config (paths, MLflow, training defaults)
│   ├── model1_config.yaml                  # Model 1 Optuna search space and settings
│   └── model2_config.yaml                  # Model 2 Optuna search space and settings
├── data/
│   ├── raw/todo_tasks.parquet              # Source data from SQLite
│   └── processed/
│       ├── eda_important_features.parquet  # 3,689 × 17 — Model 1 input
│       └── eda_urgent_features.parquet     # 3,689 × 16 — Model 2 input
├── docs/
│   ├── eda_important_report.md
│   ├── eda_urgent_report.md
│   └── figures/
├── models/                                 # Local model artifacts (.pkl)
└── mlruns/                                 # MLflow tracking data (auto-created)
```

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

### Multi-Model Optuna Search
Optuna (TPE sampler) searches across 7 model types simultaneously:

| Model type | Notes |
|---|---|
| `lgbm` | LightGBM — fast, handles mixed types natively |
| `xgboost` | XGBoost — strong baseline |
| `random_forest` | Sklearn RF with balanced class weights |
| `logistic` | Logistic Regression — linear baseline |
| `voting_lgbm_rf` | Soft VotingClassifier: LGBM + RF |
| `voting_lgbm_lr` | Soft VotingClassifier: LGBM + LR |
| `voting_all` | Soft VotingClassifier: LGBM + XGB + RF + LR |

### MLflow Experiment Tracking
Every Optuna trial is a **nested MLflow child run** under a parent training run:

```
Experiment: "important-classifier"  /  "urgent-classifier"
└── Parent run: training_<timestamp>
    ├── trial_0000  (lgbm, smote, tfidf+lsa)
    ├── trial_0001  (voting_all, none, bow)
    ...
    └── trial_0049
```

Each run logs: params, CV metrics (F1-macro, F1-class-0, F1-class-1, ROC-AUC), confusion matrix, ROC curve, feature importances, Optuna visualisations, classification report, and best hyperparameters JSON.

### Leakage Guards
| Guard | Enforcement |
|---|---|
| `is_overdue` excluded from both models | Not available at task creation time |
| `urgent` excluded from Model 1 | Target leakage |
| `important` excluded from Model 2 | Target leakage |
| `SafeTargetEncoder` fit per CV fold | Prevents project-code encoding leakage |
| `ResamplerTransformer` applied train-fold only | SMOTE never sees validation data |
| `StratifiedGroupKFold` for Model 2 | Tests generalisation to unseen projects |

### Model-Specific Design Choices

| Aspect | Model 1 (Important) | Model 2 (Urgent) |
|---|---|---|
| Imbalance | 3:1 | 21.36:1 |
| `scale_pos_weight` search | 1–5 | 10–30 |
| CV strategy | `StratifiedKFold(5)` | `StratifiedGroupKFold(5, group=project_code)` |
| `month_created` | Excluded (p=0.43 in EDA) | Included (V=0.326) |
| Lead-time features | `days_until_due` (log+scaled) | `lead_time_days/hours/bucket` (scale only — can be negative) |
| Primary signal | `hour_created` (V=0.80) | `project_code` (V=0.635) |

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
python notebooks/train_important_model.py
```

Or open `notebooks/train_important_model.py` in VSCode and run `# %%` cells interactively.

**Quick smoke test** (5 trials): set `QUICK_RUN = True` in the script.

### Model 2 — Urgent Classifier

```bash
source eisen_mat_env/Scripts/activate
cd eisenhower_estimator
python notebooks/train_urgent_model.py
```

Or open `notebooks/train_urgent_model.py` in VSCode and run `# %%` cells interactively.

**Quick smoke test** (5 trials): set `QUICK_RUN = True` in the script.

> **Note**: Model 2 uses `StratifiedGroupKFold(group=project_code)`. Some folds may have very few
> urgent samples (21.36:1 imbalance). This is expected — the trainer handles it gracefully with
> neutral scores and a warning log. Watch `test_f1_1` (urgent class F1) — not just `test_f1_macro`.
> A model predicting all "not urgent" achieves `test_f1_macro ≈ 0.48` but `test_f1_1 = 0.0` — failure mode.

---

## How to View Results in MLflow UI

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Open [http://localhost:5000](http://localhost:5000)

You will see two experiments:
- **`important-classifier`** — Model 1 runs
- **`urgent-classifier`** — Model 2 runs

Each experiment shows:
- All Optuna trials ranked by F1-macro
- Confusion matrix and ROC curve for the best model
- Optuna optimization history and parameter importance plots
- Classification report and best hyperparameters JSON
- (Model 2 only) Class distribution plot (train vs test)

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

---

## Configuration

### `configs/config.yaml`
Global settings: data paths, MLflow tracking URI, experiment names, training defaults (CV folds, test size, random seed, Optuna trials).

### `configs/model1_config.yaml`
Model 1 specific settings: Optuna search space ranges, allowed model types, resampling options.

### `configs/model2_config.yaml`
Model 2 specific settings: `scale_pos_weight` range (10–30), `StratifiedGroupKFold` notes, lead_time handling, resampling recommendations for 21.36:1 imbalance.

To change the number of Optuna trials:
```yaml
# configs/model2_config.yaml
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

---

## Data Pipeline

```
SQLite DB (personal_organizer.db)
    ↓  src/data/extractor.py
data/raw/todo_tasks.parquet
    ↓  notebooks/eda_important.py  /  notebooks/eda_urgent.py
data/processed/eda_important_features.parquet   (3,689 × 17)
data/processed/eda_urgent_features.parquet      (3,689 × 16)
    ↓  notebooks/train_important_model.py  /  notebooks/train_urgent_model.py
models/important_classifier_best.pkl
models/urgent_classifier_best.pkl
mlruns/   (MLflow artifacts + model registry)
```

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
