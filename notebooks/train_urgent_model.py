# %% [markdown]
# # Model 2 — Urgent Classifier Training
#
# This script trains and evaluates the **Urgent/Not Urgent** binary classifier
# using LightGBM, XGBoost, RandomForest, LogisticRegression and VotingClassifier variants,
# with Optuna hyperparameter search and MLflow experiment tracking.
#
# **Key facts about this model:**
# - Target: `urgent` (binary)
# - Imbalance: 95.5% not urgent, 4.5% urgent (21.36:1 ratio — severe)
# - CV strategy: `StratifiedGroupKFold(group=project_code)` — tests generalisation to new projects
# - Primary signal: `project_code` (Cramér's V = 0.635)
# - NOT included: `is_overdue` (unavailable at task creation), `important` (leakage)
#
# **Run from the project root** (eisenhower_estimator/):
# ```bash
# source eisen_mat_env/Scripts/activate
# python notebooks/train_urgent_model.py
# ```
#
# Or run interactively in VSCode / Jupyter (each `# %%` is a cell).
#
# **MLflow UI**: after training, start the UI with:
# ```bash
# mlflow ui --backend-store-uri mlruns --port 5000
# ```

# %% [markdown]
# ## 0. Environment check

# %%
import sys
from pathlib import Path

# Ensure project root is in path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

logger.info(f"Python: {sys.version}")
logger.info(f"Project root: {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Quick data check

# %%
import pandas as pd

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "eda_urgent_features.parquet"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed data not found: {DATA_PATH}\n"
        f"Run the EDA script first: notebooks/eda_urgent.py"
    )

df = pd.read_parquet(DATA_PATH)
logger.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
logger.info(
    f"Target distribution:\n{df['urgent'].value_counts(normalize=True).round(3)}\n"
    f"Absolute counts:\n{df['urgent'].value_counts()}"
)
logger.info(f"Columns: {df.columns.tolist()}")

# Sanity check: forbidden columns
FORBIDDEN = {"important", "is_overdue", "days_until_due", "has_due_date"}
present_forbidden = FORBIDDEN & set(df.columns)
if present_forbidden:
    logger.warning(f"Forbidden columns present (will be dropped by trainer): {present_forbidden}")

# Check lead_time_days range (can be negative — verify this is expected)
if "lead_time_days" in df.columns:
    negative_count = (df["lead_time_days"] < 0).sum()
    logger.info(
        f"lead_time_days: min={df['lead_time_days'].min():.0f}, "
        f"max={df['lead_time_days'].max():.0f}, "
        f"negative={negative_count} ({negative_count / len(df):.1%}) — expected for retroactive tasks"
    )

# Check lead_time_bucket distribution
if "lead_time_bucket" in df.columns:
    logger.info(f"lead_time_bucket distribution:\n{df['lead_time_bucket'].value_counts()}")

# %% [markdown]
# ## 2. Verify dependencies

# %%
def check_dependencies():
    missing = []
    packages = {
        "mlflow": "mlflow",
        "optuna": "optuna",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "imblearn": "imbalanced-learn",
        "category_encoders": "category-encoders",
        "joblib": "joblib",
    }
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        raise ImportError(
            f"Missing packages: {missing}\n"
            f"Install with: pip install {' '.join(missing)}"
        )
    logger.info("All dependencies available.")

check_dependencies()

# %% [markdown]
# ## 3. Configure training

# %%
# Training configuration overrides (edit here for quick experiments)
# These override values in configs/model2_config.yaml

QUICK_RUN = False          # Set True to run only 5 trials for a quick smoke test
N_TRIALS_OVERRIDE = None   # Set an int to override config (e.g. 10 for fast test)
TIMEOUT_OVERRIDE = None    # Set seconds to override config timeout

# %% [markdown]
# ## 4. Run training

# %%
import os
os.chdir(str(PROJECT_ROOT))   # Ensure relative paths work

from src.training.urgent_trainer import UrgentModelTrainer

trainer = UrgentModelTrainer(
    config_path="configs/config.yaml",
    model_config_path="configs/model2_config.yaml",
    data_path=DATA_PATH,
)

# Apply quick-run overrides
if QUICK_RUN or N_TRIALS_OVERRIDE is not None:
    n_trials = N_TRIALS_OVERRIDE or 5
    trainer.model_config.setdefault("training", {})["optuna_trials"] = n_trials
    logger.info(f"Quick run mode: n_trials={n_trials}")

if TIMEOUT_OVERRIDE is not None:
    trainer.model_config.setdefault("training", {})["optuna_timeout_seconds"] = TIMEOUT_OVERRIDE

result = trainer.train()

# %% [markdown]
# ## 5. Results summary

# %%
import json

logger.info("=" * 60)
logger.info("TRAINING SUMMARY — URGENT CLASSIFIER")
logger.info("=" * 60)
logger.info(f"MLflow Run ID   : {result['run_id']}")
logger.info(f"Best trial      : #{result['best_trial']}")
logger.info(f"Best CV F1-macro: {result['best_cv_f1_macro']:.4f}")
logger.info(f"Test F1-macro   : {result['test_metrics']['f1_macro']:.4f}")
logger.info(f"Test F1-class-0 : {result['test_metrics']['f1_0']:.4f}  (Not Urgent)")
logger.info(f"Test F1-class-1 : {result['test_metrics']['f1_1']:.4f}  (Urgent)  ← key metric")
if "roc_auc" in result["test_metrics"]:
    logger.info(f"Test ROC-AUC    : {result['test_metrics']['roc_auc']:.4f}")
logger.info(f"Model URI       : {result['model_uri']}")
logger.info(f"Best config     :\n{json.dumps(result['best_config'], indent=2, default=str)}")

# Interpretation guidance
f1_urgent = result["test_metrics"]["f1_1"]
if f1_urgent >= 0.6:
    logger.info("Model quality: GOOD — urgent class F1 >= 0.6")
elif f1_urgent >= 0.4:
    logger.warning(
        "Model quality: MODERATE — urgent class F1 between 0.4 and 0.6. "
        "Consider: more trials, higher scale_pos_weight, or SMOTE."
    )
else:
    logger.error(
        f"Model quality: POOR — urgent class F1 = {f1_urgent:.4f}. "
        f"The model is likely predicting mostly 'not urgent'. "
        f"Try: QUICK_RUN=False with 100+ trials, or adjust scale_pos_weight range in model2_config.yaml."
    )

# %% [markdown]
# ## 6. Load and test the registered model

# %%
import mlflow.sklearn
import numpy as np

try:
    loaded_model = mlflow.sklearn.load_model(result["model_uri"])

    # Test inference on 3 samples — one from each project_category
    # Note: project_category must be "Work" or "Personal" as a string
    feature_cols = [
        "hour_created", "day_of_week_created", "month_created",
        "project_code", "project_category",
        "lead_time_bucket", "lead_time_days", "lead_time_hours",
        "desc_word_count", "desc_char_len", "desc_clean",
    ]
    sample = df[feature_cols].head(3)
    preds = loaded_model.predict(sample)
    probas = loaded_model.predict_proba(sample)
    logger.info("Inference test on 3 samples:")
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        logger.info(
            f"  Sample {i}: pred={pred} ({'urgent' if pred else 'not_urgent'}), "
            f"proba_urgent={proba[1]:.3f}"
        )
except Exception as exc:
    logger.error(f"Could not load/test registered model: {exc}")

# %% [markdown]
# ## 7. Open MLflow UI
#
# Run in your terminal:
# ```bash
# mlflow ui --backend-store-uri mlruns --port 5000
# ```
# Then open: http://localhost:5000
#
# The experiment **"urgent-classifier"** will show:
# - All Optuna trials ranked by F1-macro
# - Confusion matrix and ROC curve for the best model
# - Optuna optimization history and parameter importance plots
# - Classification report and best hyperparameters JSON
# - Class distribution plot (train vs test)
#
# **Important metric to watch**: `test_f1_1` (urgent class F1)
# A model predicting all "not urgent" achieves test_f1_macro ≈ 0.48 but test_f1_1 = 0.0 — failure mode.
