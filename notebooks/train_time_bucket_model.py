# %% [markdown]
# # Model 3b — Time Bucket Classifier Training
#
# This script trains and evaluates the **Time Bucket Classifier** model,
# which predicts which duration bucket a task falls into:
#   ≤2 min | 3–5 min | 6–10 min | 11–30 min | 31–60 min | >60 min
#
# Uses the same features as Model 3 (time spent regressor) but treats
# prediction as a 6-class classification problem.
#
# **Run from the project root** (eisenhower_estimator/):
# ```bash
# source eisen_mat_env/Scripts/activate
# python notebooks/train_time_bucket_model.py
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

# Class distribution
logger.info(f"Duration bucket distribution:\n{df['duration_bucket'].value_counts().sort_index()}")
logger.info(f"Task type distribution:\n{df['task_type'].value_counts()}")
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
QUICK_RUN = False          # Set True for 5-trial smoke test
N_TRIALS_OVERRIDE = None   # Set int to override config
TIMEOUT_OVERRIDE = None    # Set seconds to override config

# %% [markdown]
# ## 4. Run training

# %%
import os
os.chdir(str(PROJECT_ROOT))

from src.training.time_bucket_trainer import TimeBucketTrainer

trainer = TimeBucketTrainer(
    config_path="configs/config.yaml",
    model_config_path="configs/model3b_config.yaml",
    data_path=DATA_PATH,
)

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
logger.info("TRAINING SUMMARY — Time Bucket Classifier")
logger.info("=" * 60)
logger.info(f"MLflow Run ID       : {result['run_id']}")
logger.info(f"Best trial          : #{result['best_trial']}")
logger.info(f"Best CV F1 (macro)  : {result['best_cv_f1_macro']:.4f}")
logger.info(f"Test F1 (macro)     : {result['test_metrics']['f1_macro']:.4f}")
logger.info(f"Test F1 (weighted)  : {result['test_metrics']['f1_weighted']:.4f}")
logger.info(f"Test Precision      : {result['test_metrics']['precision_macro']:.4f}")
logger.info(f"Test Recall         : {result['test_metrics']['recall_macro']:.4f}")
logger.info(f"Test Accuracy       : {result['test_metrics']['accuracy']:.4f}")
logger.info(f"Model URI           : {result['model_uri']}")

# Per-class F1
from src.training.pipeline_builder_time_bucket import BUCKET_ORDER
for bucket in BUCKET_ORDER:
    key = f"f1_{bucket}"
    if key in result["test_metrics"]:
        logger.info(f"  F1 ({bucket:>10s}) : {result['test_metrics'][key]:.4f}")

logger.info(f"Best config:\n{json.dumps(result['best_config'], indent=2, default=str)}")

# %% [markdown]
# ## 6. Load and test the model

# %%
try:
    import joblib
    model_path = PROJECT_ROOT / "models" / "time_bucket_classifier_best.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")

        from src.training.pipeline_builder_time_bucket import BUCKET_ORDER

        sample = df.head(5)
        from src.training.optuna_objective_time_bucket import FEATURE_COLS
        X_sample = sample[FEATURE_COLS]
        preds = model.predict(X_sample)
        pred_labels = [BUCKET_ORDER[p] for p in preds]

        logger.info("Sample predictions:")
        for i, (actual, predicted) in enumerate(
            zip(sample["duration_bucket"].values, pred_labels)
        ):
            logger.info(f"  Task {i}: actual={actual}, predicted={predicted}")
    else:
        logger.warning(f"Model file not found: {model_path}")
except Exception as exc:
    logger.error(f"Could not load model: {exc}")
