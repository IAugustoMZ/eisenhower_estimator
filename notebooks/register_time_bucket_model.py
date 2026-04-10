"""
register_time_bucket_model.py
─────────────────────────────
Registers the pre-trained time-bucket classifier pkl into the local MLflow
Model Registry without re-training.

Run from the project root:
    source eisen_mat_env/Scripts/activate
    python notebooks/register_time_bucket_model.py

What it does:
  1. Loads models/time_bucket_classifier_best.pkl
  2. Creates/reuses the "time-bucket-classifier" MLflow experiment
  3. Logs the model as a new run with a minimal input example
  4. Registers it in the Model Registry under "time-bucket-classifier"

After running:
  - Restart the Docker stack so mlflow-init syncs the updated mlruns/ directory.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI      = "sqlite:///mlflow.db"
EXPERIMENT_NAME   = "time-bucket-classifier"
REGISTERED_NAME   = "time-bucket-classifier"
MODEL_PATH        = PROJECT_ROOT / "models" / "time_bucket_classifier_best.pkl"

# Feature columns the model expects (from optuna_objective_time_bucket.py)
FEATURE_COLS = [
    "day_of_week",
    "project_code",
    "task_type",
    "has_number",
    "is_repeated_task",
    "task_freq",
    "task_cv",
    "desc_char_len",
    "desc_clean",
    "task_type_x_repeated",
    "is_long_project",
    "log_task_freq",
    "desc_has_time_ref",
    "task_median_duration",
]

# ── Validate ──────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"Train the model first with: python notebooks/train_time_bucket_model.py"
    )

# ── Build a minimal input example ─────────────────────────────────────────────
input_example = pd.DataFrame([{
    "day_of_week":          2,
    "project_code":         "PESSOAL",
    "task_type":            "development",
    "has_number":           0,
    "is_repeated_task":     0,
    "task_freq":            1.0,
    "task_cv":              0.3,
    "desc_char_len":        25,
    "desc_clean":           "implement feature",
    "task_type_x_repeated": "development_0",
    "is_long_project":      0,
    "log_task_freq":        0.0,
    "desc_has_time_ref":    0,
    "task_median_duration": 20.0,
}])

# ── Register ──────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

logger.info(f"Loading model from: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

logger.info("Starting MLflow registration run...")

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
try:
    with mlflow.start_run(run_name="registration_from_pkl") as run:
        mlflow.set_tag("model", REGISTERED_NAME)
        mlflow.set_tag("stage", "registration")
        mlflow.set_tag("source", "pre-trained pkl")
        mlflow.log_param("model_file", str(MODEL_PATH))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=REGISTERED_NAME,
        )
        logger.info(f"Model logged and registered: {model_info.model_uri}")
        logger.info(f"Run ID: {run.info.run_id}")
finally:
    os.environ.pop("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", None)

logger.info("Done. Restart the Docker stack to sync the updated mlruns/ into the container.")
