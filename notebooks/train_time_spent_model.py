# %% [markdown]
# # Model 3 — Time Spent Regressor Training
#
# This script trains and evaluates the **Time Spent Regression** model
# using LightGBM, XGBoost, RandomForest, Ridge and VotingRegressor variants,
# with Optuna hyperparameter search and MLflow experiment tracking.
#
# **Target**: `log1p(duration_minutes)` — continuous, right-skewed.
# Predictions are converted back to minutes via `np.expm1()`.
#
# **Run from the project root** (eisenhower_estimator/):
# ```bash
# source eisen_mat_env/Scripts/activate
# python notebooks/train_time_spent_model.py
# ```
#
# Or run interactively in VSCode / Jupyter (each `# %%` is a cell).
#
# **MLflow UI**: after training, start the UI with:
# ```bash
# mlflow ui --backend-store-uri sqlite:///mlflow.db
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
import numpy as np

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "eda_time_spent_features.parquet"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed data not found: {DATA_PATH}\n"
        f"Run the EDA script first: notebooks/eda_time_spent.py"
    )

df = pd.read_parquet(DATA_PATH)
logger.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
logger.info(f"Target (log_duration) stats:\n{df['log_duration'].describe()}")
logger.info(f"Original target (duration_minutes) stats:\n{df['duration_minutes'].describe()}")
logger.info(f"Columns: {df.columns.tolist()}")

# Duration bucket distribution
logger.info(f"Duration bucket distribution:\n{df['duration_bucket'].value_counts().sort_index()}")

# Task type distribution
logger.info(f"Task type distribution:\n{df['task_type'].value_counts()}")

# Project distribution
logger.info(f"Project distribution:\n{df['project_code'].value_counts()}")

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
        "joblib": "joblib",
        "yaml": "pyyaml",
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
# These override values in configs/model3_config.yaml

QUICK_RUN = False          # Set True to run only 5 trials for a quick smoke test
N_TRIALS_OVERRIDE = None   # Set an int to override config (e.g. 10 for fast test)
TIMEOUT_OVERRIDE = None    # Set seconds to override config timeout

# %% [markdown]
# ## 4. Run training

# %%
import os
os.chdir(str(PROJECT_ROOT))   # Ensure relative paths work

from src.training.time_spent_trainer import TimeSpentTrainer

trainer = TimeSpentTrainer(
    config_path="configs/config.yaml",
    model_config_path="configs/model3_config.yaml",
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
logger.info("TRAINING SUMMARY")
logger.info("=" * 60)
logger.info(f"MLflow Run ID      : {result['run_id']}")
logger.info(f"Best trial         : #{result['best_trial']}")
logger.info(f"Best CV RMSE (log) : {result['best_cv_rmse_log']:.4f}")
logger.info(f"Test RMSE (log)    : {result['test_metrics']['rmse_log']:.4f}")
logger.info(f"Test MAE (log)     : {result['test_metrics']['mae_log']:.4f}")
logger.info(f"Test R² (log)      : {result['test_metrics']['r2_log']:.4f}")
logger.info(f"Test MAE (minutes) : {result['test_metrics']['mae_minutes']:.2f}")
logger.info(f"Test RMSE (minutes): {result['test_metrics']['rmse_minutes']:.2f}")
logger.info(f"Test R² (minutes)  : {result['test_metrics']['r2_minutes']:.4f}")
logger.info(f"Test MedAE (min)   : {result['test_metrics']['median_ae_minutes']:.2f}")
logger.info(f"Model URI          : {result['model_uri']}")
logger.info(f"Best config        :\n{json.dumps(result['best_config'], indent=2, default=str)}")

# %% [markdown]
# ## 6. Critical analysis of results

# %%
# Interpret results against EDA baseline expectations
metrics = result["test_metrics"]

logger.info("=" * 60)
logger.info("CRITICAL ANALYSIS")
logger.info("=" * 60)

# ICC = 0.695 means 69.5% of variance is predictable
# Theoretical best R² ≈ 0.695 on log scale
r2_log = metrics["r2_log"]
logger.info(f"R² (log) = {r2_log:.4f} vs theoretical ICC ceiling = 0.695")
if r2_log > 0.70:
    logger.warning(
        "R² exceeds ICC ceiling — potential overfitting or data leakage. "
        "Investigate feature importance and check for target-derived features."
    )
elif r2_log > 0.50:
    logger.info(
        "R² is in a good range — model captures most of the systematic "
        "between-task variance without overfitting."
    )
elif r2_log > 0.30:
    logger.info(
        "R² is moderate — model captures some signal but misses complexity "
        "in task duration estimation. Consider adding more features."
    )
else:
    logger.warning(
        "R² is low — model struggles to predict duration. "
        "Check if text features are being used effectively."
    )

# MAE in minutes — most interpretable metric
mae_min = metrics["mae_minutes"]
median_ae = metrics["median_ae_minutes"]
logger.info(f"MAE = {mae_min:.2f} min (average prediction error)")
logger.info(f"Median AE = {median_ae:.2f} min (typical prediction error)")

# Context: 86.4% of tasks are 2, 5, or 10 min
# A naive "always predict 5 min" baseline would have MAE ≈ 5-6 min
if mae_min < 3:
    logger.info("MAE < 3 min — excellent performance for this dataset.")
elif mae_min < 6:
    logger.info("MAE < 6 min — good; competitive with 'always predict 5' baseline.")
elif mae_min < 10:
    logger.info(
        "MAE < 10 min — moderate. SBM outliers likely inflate this. "
        "Check per-project MAE for diagnosis."
    )
else:
    logger.warning(
        f"MAE = {mae_min:.1f} min is high. "
        f"SBM long tasks (up to 750 min) are likely dominating error."
    )

# %% [markdown]
# ## 7. Load and test the registered model

# %%
import mlflow.sklearn

try:
    loaded_model = mlflow.sklearn.load_model(result["model_uri"])
    sample = df[["day_of_week", "project_code", "task_type", "has_number",
                 "is_repeated_task", "task_freq", "task_cv", "desc_char_len",
                 "desc_clean"]].head(5)

    preds_log = loaded_model.predict(sample)
    preds_min = np.expm1(preds_log)

    logger.info("Inference test on 5 samples:")
    for i, (pred_log, pred_min) in enumerate(zip(preds_log, preds_min)):
        actual_min = df['duration_minutes'].iloc[i]
        logger.info(
            f"  Sample {i}: pred={pred_min:.1f} min "
            f"(log={pred_log:.3f}), actual={actual_min} min"
        )
except Exception as exc:
    logger.error(f"Could not load/test registered model: {exc}")

# %% [markdown]
# ## 8. Open MLflow UI
#
# Run in your terminal:
# ```bash
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# ```
# Then open: http://localhost:5000
#
# The experiment **"time-spent-regressor"** will show:
# - All Optuna trials as child runs
# - Best model params, CV metrics, test metrics
# - Plots: predicted vs actual, residuals, error distribution,
#   segmented analysis, feature importance
# - Optuna visualisations: optimization history, param importances
# - Artifacts: regression report, best hyperparameters JSON
