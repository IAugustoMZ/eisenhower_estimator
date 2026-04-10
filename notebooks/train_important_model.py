# %% [markdown]
# # Model 1 — Important Classifier Training (Versioned)
#
# End-to-end training pipeline for the **Important/Not Important** binary classifier.
# Covers three approaches with comparable metrics on the same holdout set:
#   - Rule-based baseline (domain knowledge)
#   - Pure ML (LightGBM / XGBoost / RF / LR + Optuna)
#   - Hybrid (rules as hard overrides + ML for undecided cases)
#
# Every MLflow run is tagged with `data_version`, `data_sha256`, and `git_commit`
# so any run can be reproduced exactly from code + data.
#
# **Run from the project root**:
# ```bash
# source eisen_mat_env/Scripts/activate
# python notebooks/train_important_model.py
# ```
#
# **MLflow UI** (after training):
# ```bash
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
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
# ## 1. Verify dependencies

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
# ## 2. Data check

# %%
import yaml
import pandas as pd

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "eda_important_features.parquet"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed data not found: {DATA_PATH}\n"
        f"Run the EDA script first: python notebooks/eda_important.py"
    )

df = pd.read_parquet(DATA_PATH)
logger.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
logger.info(f"Target distribution:\n{df['important'].value_counts(normalize=True).round(3)}")

# Load and display data version from config
with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
    _cfg = yaml.safe_load(f)
_data_version = _cfg.get("data", {}).get("version", "unknown")
logger.info(f"Data version: {_data_version}")

FORBIDDEN = {"urgent", "is_overdue", "month_created", "has_due_date"}
present_forbidden = FORBIDDEN & set(df.columns)
if present_forbidden:
    logger.warning(f"Forbidden columns present (will be dropped by trainer): {present_forbidden}")

# %% [markdown]
# ## 3. Configure training
#
# Set `QUICK_RUN = True` for a smoke test (5 trials, validates the full pipeline).
# Set `QUICK_RUN = False` for the full Optuna search (50 trials by default).

# %%
# -- Training configuration overrides ----------------------------------------
QUICK_RUN = False           # Set False for the full 50-trial Optuna run
N_TRIALS_OVERRIDE = None    # Override trial count explicitly (None = use config)
TIMEOUT_OVERRIDE = None     # Override timeout in seconds (None = use config)

# %% [markdown]
# ## 4. Run training

# %%
import os
os.chdir(str(PROJECT_ROOT))

from src.training.trainer import ModelTrainer

trainer = ModelTrainer(
    config_path="configs/config.yaml",
    model_config_path="configs/model1_config.yaml",
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
# ## 5. ML training summary

# %%
import json

logger.info("=" * 60)
logger.info("ML TRAINING SUMMARY")
logger.info("=" * 60)
logger.info(f"MLflow Run ID    : {result['run_id']}")
logger.info(f"Data version     : {_data_version}")
logger.info(f"Best trial       : #{result['best_trial']}")
logger.info(f"Best CV F1-macro : {result['best_cv_f1_macro']:.4f}")
logger.info(f"Test F1-macro    : {result['test_metrics']['f1_macro']:.4f}")
logger.info(f"Test F1-class-0  : {result['test_metrics']['f1_0']:.4f}  (Not Important)")
logger.info(f"Test F1-class-1  : {result['test_metrics']['f1_1']:.4f}  (Important)")
if "roc_auc" in result["test_metrics"]:
    logger.info(f"Test ROC-AUC     : {result['test_metrics']['roc_auc']:.4f}")
logger.info(f"Model URI        : {result['model_uri']}")
logger.info(f"Best config      :\n{json.dumps(result['best_config'], indent=2, default=str)}")

# %% [markdown]
# ## 6. System evaluation — comparable metrics (rule-based vs ML vs hybrid)

# %%
if result.get("comparison_report"):
    from src.evaluation import RuleBasedClassifier, HybridClassifier, ModelEvaluator

    report = result["comparison_report"]
    logger.info("")
    logger.info("=" * 60)
    logger.info("SYSTEM EVALUATION — Rule-Based vs ML vs Hybrid")
    logger.info("=" * 60)

    # Re-instantiate evaluator just for the print (metrics already logged to MLflow)
    from sklearn.model_selection import train_test_split
    import numpy as np

    X = df[["hour_created", "day_of_week_created", "project_code",
            "project_category", "days_until_due", "desc_word_count",
            "desc_char_len", "desc_clean"]].copy()
    y = df["important"].to_numpy()

    # Use same random_state and test_size as trainer to get the identical split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=_cfg["training"]["test_size"],
        stratify=y,
        random_state=_cfg["training"]["random_seed"],
    )

    import mlflow.sklearn
    ml_pipeline = mlflow.sklearn.load_model(result["model_uri"])

    rule_clf = RuleBasedClassifier()
    hybrid_clf = HybridClassifier(rule_clf, ml_pipeline)
    evaluator = ModelEvaluator(rule_clf, ml_pipeline, hybrid_clf)

    # Print comparison table (metrics already in MLflow, just display locally)
    evaluator.print_comparison_table(report)

    # Show rule coverage
    cov = rule_clf.coverage(X_test)
    logger.info(f"\nRule coverage on test set: {cov['fired_ratio']:.1%} of rows decided by rules")
    for rule_name, ratio in cov.get("per_rule_coverage", {}).items():
        logger.info(f"  {rule_name}: {ratio:.1%}")
else:
    logger.warning("Comparison report not available (system evaluation may have failed).")

# %% [markdown]
# ## 7. Inference test on 3 samples

# %%
import mlflow.sklearn
import numpy as np

try:
    loaded_model = mlflow.sklearn.load_model(result["model_uri"])
    sample = df[["hour_created", "day_of_week_created", "project_code",
                 "project_category", "days_until_due", "desc_word_count",
                 "desc_char_len", "desc_clean"]].head(3)
    preds = loaded_model.predict(sample)
    probas = loaded_model.predict_proba(sample)
    logger.info("Inference test on 3 samples:")
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        logger.info(f"  Sample {i}: pred={pred} ({'important' if pred else 'not_important'}), "
                    f"P(important)={proba[1]:.3f}")
except Exception as exc:
    logger.error(f"Could not load/test registered model: {exc}")

# %% [markdown]
# ## 8. View results in MLflow UI
#
# ```bash
# mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# ```
# Open: http://localhost:5000
#
# In the **"important-classifier"** experiment you will see:
# - Parent run tagged with `data_version`, `data_sha256`, `git_commit`
# - Child runs: one per Optuna trial + three evaluation runs (rule/ml/hybrid)
# - Evaluation child runs: confusion matrix, classification report per approach
# - Artifacts: system_comparison_report.md (side-by-side metrics)
# - Plots: confusion matrix, ROC curve, feature importance, Optuna visualisations
