"""
OptunaObjective — Time Spent Regressor (Model 3)
Defines the Optuna search space for the duration regression model.

Covers:
  - Preprocessor: scaler, text vectorizer, LSA settings
  - Model selection: lgbm / xgboost / random_forest / ridge / voting variants
  - Feature selection: f_regression / mutual_info_regression / none

Each Optuna trial:
  1. Samples a hyperparameter config
  2. Applies TaskCVImputer to impute task_cv for unseen tasks
  3. Runs StratifiedGroupKFold CV (group=project_code, stratify=duration_bucket)
  4. Returns mean RMSE on log scale as the objective value (minimise)

IMPORTANT:
  - SafeTargetEncoder is fit per fold (target=log_duration) to prevent leakage.
  - LDA is excluded (classification only).
  - No resampling — this is regression, not classification.
  - task_cv imputation uses task_type-level mean for unseen tasks.
"""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline

from src.training.base_objective import BaseObjective, _sample_model_config
from src.training.pipeline_builder_time_spent import (
    TaskCVImputer,
    build_full_pipeline_time_spent,
)

# Silence Optuna's per-trial INFO logs (we log our own)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature columns used in this model ───────────────────────────────────────
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
    # Iteration-2 features (from error analysis)
    "task_type_x_repeated",
    "is_long_project",
    "log_task_freq",
    "desc_has_time_ref",
    "task_median_duration",
]
TARGET_COL = "log_duration"
ORIGINAL_TARGET_COL = "duration_minutes"
GROUP_COL = "project_code"
STRATIFY_COL = "duration_bucket"

# Explicitly forbidden features (leakage / not significant)
FORBIDDEN_COLS = {
    "duration_minutes",     # raw target — log_duration is the training target
    "project_category",     # p=0.31
    "is_weekend",           # p=0.31
    "month",                # p=0.070
    "quarter",              # not significant
    "week_of_year",         # borderline noise
    "desc_word_count",      # p=0.156
    "project_name",         # redundant with project_code
    "date",                 # temporal leakage risk
    "duration_bucket",      # derived from target
}


def _validate_feature_cols(df: pd.DataFrame) -> None:
    """Raise early if required columns are missing."""
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"TimeSpentObjective: required feature columns missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TimeSpentObjective: target column '{TARGET_COL}' not found in DataFrame."
        )


def _sample_regressor_config(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    """
    Sample model-specific hyperparameters for regression models.

    Regression models don't have scale_pos_weight or class_weight.
    Shares lgbm/xgboost/random_forest core sampling from base_objective
    but excludes classification-only params.
    """
    config: dict[str, Any] = {}

    if model_type in ("lgbm", "voting_lgbm_rf", "voting_lgbm_ridge", "voting_all"):
        config["lgbm_n_estimators"] = trial.suggest_int("lgbm_n_estimators", 100, 600, step=50)
        config["lgbm_max_depth"] = trial.suggest_int("lgbm_max_depth", -1, 12)
        config["lgbm_learning_rate"] = trial.suggest_float("lgbm_learning_rate", 0.01, 0.3, log=True)
        config["lgbm_num_leaves"] = trial.suggest_int("lgbm_num_leaves", 15, 127)
        config["lgbm_min_child_samples"] = trial.suggest_int("lgbm_min_child_samples", 5, 50)
        config["lgbm_subsample"] = trial.suggest_float("lgbm_subsample", 0.5, 1.0)
        config["lgbm_colsample_bytree"] = trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0)
        config["lgbm_reg_alpha"] = trial.suggest_float("lgbm_reg_alpha", 1e-8, 1.0, log=True)
        config["lgbm_reg_lambda"] = trial.suggest_float("lgbm_reg_lambda", 1e-8, 1.0, log=True)

    if model_type in ("xgboost", "voting_all"):
        config["xgb_n_estimators"] = trial.suggest_int("xgb_n_estimators", 100, 600, step=50)
        config["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 12)
        config["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
        config["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        config["xgb_colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0)
        config["xgb_reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-8, 1.0, log=True)
        config["xgb_reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 1e-8, 1.0, log=True)

    if model_type in ("random_forest", "voting_lgbm_rf", "voting_all"):
        config["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 100, 500, step=50)
        config["rf_max_depth"] = trial.suggest_categorical("rf_max_depth", [None, 5, 10, 15, 20])
        config["rf_min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 10)
        config["rf_min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 5)
        config["rf_max_features"] = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5])

    if model_type in ("ridge", "voting_lgbm_ridge", "voting_all"):
        config["ridge_alpha"] = trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True)

    return config


def _sample_config(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    """Sample a full hyperparameter config from the Optuna search space for Model 3."""
    model_type = trial.suggest_categorical(
        "model_type",
        ["lgbm", "xgboost", "random_forest", "ridge",
         "voting_lgbm_rf", "voting_lgbm_ridge", "voting_all"],
    )

    config: dict[str, Any] = {
        "model_type": model_type,
        "random_state": random_state,
        # ── Preprocessor ──
        "scaler_name": trial.suggest_categorical(
            "scaler_name", ["standard", "robust", "minmax", "none"]
        ),
        "include_text": trial.suggest_categorical("include_text", [True, False]),
        "target_enc_smoothing": trial.suggest_float(
            "target_enc_smoothing", 0.1, 10.0, log=True
        ),
        # ── Feature selection (regression-specific scoring functions) ──
        "selector_strategy": trial.suggest_categorical(
            "selector_strategy", ["selectkbest_f_reg", "selectkbest_mi_reg", "none"]
        ),
    }

    # selector_k: only tune when a selection strategy is active
    if config["selector_strategy"] != "none":
        config["selector_k"] = trial.suggest_int("selector_k", 5, 40)
    else:
        config["selector_k"] = "all"

    # ── Dimensionality reduction (PCA or none — no LDA for regression) ──
    config["reducer_method"] = trial.suggest_categorical(
        "reducer_method", ["pca", "none"]
    )
    if config["reducer_method"] == "pca":
        config["reducer_n_components"] = trial.suggest_int(
            "reducer_n_components", 5, 40, step=5
        )
    else:
        config["reducer_n_components"] = 1

    # Text hyperparams (only sample if text is included)
    if config["include_text"]:
        config["vectorizer_name"] = trial.suggest_categorical(
            "vectorizer_name", ["tfidf", "bow"]
        )
        config["text_max_features"] = trial.suggest_int(
            "text_max_features", 100, 1000, step=100
        )
        config["text_ngram_max"] = trial.suggest_int("text_ngram_max", 1, 2)
        config["text_use_lsa"] = trial.suggest_categorical("text_use_lsa", [True, False])
        config["text_n_components"] = trial.suggest_int(
            "text_n_components", 10, 60, step=10
        )
        config["text_min_df"] = trial.suggest_int("text_min_df", 1, 5)
    else:
        config["vectorizer_name"] = "tfidf"
        config["text_use_lsa"] = True

    # ── Model-specific hyperparameters ──
    config.update(_sample_regressor_config(trial, model_type))

    return config


# ── Regression cross-validation ──────────────────────────────────────────────

def cross_validate_regression(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    groups: np.ndarray | None = None,
    stratify_bins: np.ndarray | None = None,
    cv_imputer: TaskCVImputer | None = None,
) -> dict[str, float]:
    """
    Run regression CV manually so we can:
      1. Apply TaskCVImputer per fold (fit on train, transform both)
      2. Stratify on duration_bucket while grouping by project_code
      3. Compute regression metrics (RMSE, MAE, R²) on both log and original scales

    Parameters
    ----------
    pipeline          : Unfitted sklearn Pipeline (will be cloned per fold)
    X                 : Feature DataFrame
    y                 : Target array (log1p(duration_minutes))
    n_splits          : Number of folds
    random_state      : RNG seed
    groups            : Group labels for StratifiedGroupKFold (project_code)
    stratify_bins     : Bin labels for stratification (duration_bucket)
    cv_imputer        : TaskCVImputer instance (will be cloned and fit per fold)

    Returns
    -------
    dict with mean and std of: rmse_log, mae_minutes, rmse_minutes, r2_log
    """
    # Build splitter: StratifiedGroupKFold if we have groups+bins, else GroupKFold
    if groups is not None and stratify_bins is not None:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            # Validate that all groups have enough members
            split_kwargs = {"X": X, "y": stratify_bins, "groups": groups}
            # Test split to detect errors early
            list(splitter.split(**split_kwargs))
        except ValueError as exc:
            logger.warning(
                f"StratifiedGroupKFold failed ({exc}). "
                f"Falling back to GroupKFold (no stratification)."
            )
            splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
            split_kwargs = {"X": X, "y": y, "groups": groups}
    elif groups is not None:
        n_groups = len(np.unique(groups))
        effective_splits = min(n_splits, n_groups)
        if effective_splits < n_splits:
            logger.warning(
                f"Only {n_groups} unique groups — reducing CV folds "
                f"from {n_splits} to {effective_splits}."
            )
        splitter = GroupKFold(n_splits=effective_splits)
        split_kwargs = {"X": X, "y": y, "groups": groups}
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_kwargs = {"X": X, "y": y}

    fold_metrics: dict[str, list[float]] = {
        "rmse_log": [], "mae_minutes": [], "rmse_minutes": [], "r2_log": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(**split_kwargs)):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Guard: skip fold if it has <2 unique values (can't regress)
        if len(np.unique(y_train_fold)) < 2:
            logger.warning(
                f"  Fold {fold_idx + 1}/{n_splits}: only one unique target value "
                f"in train fold — skipping with neutral metrics."
            )
            fold_metrics["rmse_log"].append(float("inf"))
            fold_metrics["mae_minutes"].append(float("inf"))
            fold_metrics["rmse_minutes"].append(float("inf"))
            fold_metrics["r2_log"].append(0.0)
            continue

        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)

        # Apply TaskCVImputer per fold (fit on train, transform both)
        if cv_imputer is not None:
            imputer_clone = clone(cv_imputer)
            # Temporarily add log_duration to train fold so the imputer can
            # recompute task_median_duration in a leakage-safe manner
            X_train_with_target = X_train_fold.copy()
            X_train_with_target["log_duration"] = y_train_fold
            imputer_clone.fit(X_train_with_target)
            X_train_fold = imputer_clone.transform(X_train_fold)
            X_val_fold = imputer_clone.transform(X_val_fold)

        pipeline_clone = clone(pipeline)

        try:
            pipeline_clone.fit(X_train_fold, y_train_fold)
        except Exception as exc:
            logger.warning(
                f"  Fold {fold_idx + 1}/{n_splits}: fit failed ({exc}). "
                f"Skipping with neutral metrics."
            )
            fold_metrics["rmse_log"].append(float("inf"))
            fold_metrics["mae_minutes"].append(float("inf"))
            fold_metrics["rmse_minutes"].append(float("inf"))
            fold_metrics["r2_log"].append(0.0)
            continue

        y_pred_log = pipeline_clone.predict(X_val_fold)

        # Metrics on log scale
        rmse_log = float(np.sqrt(mean_squared_error(y_val_fold, y_pred_log)))
        r2_log = float(r2_score(y_val_fold, y_pred_log))

        # Convert to original minutes for interpretable metrics
        y_val_minutes = np.expm1(y_val_fold)
        y_pred_minutes = np.expm1(np.clip(y_pred_log, 0, 10))  # clip to prevent exp overflow

        mae_minutes = float(mean_absolute_error(y_val_minutes, y_pred_minutes))
        rmse_minutes = float(np.sqrt(mean_squared_error(y_val_minutes, y_pred_minutes)))

        fold_metrics["rmse_log"].append(rmse_log)
        fold_metrics["mae_minutes"].append(mae_minutes)
        fold_metrics["rmse_minutes"].append(rmse_minutes)
        fold_metrics["r2_log"].append(r2_log)

        logger.debug(
            f"  Fold {fold_idx + 1}/{n_splits}: "
            f"rmse_log={rmse_log:.4f}, mae_min={mae_minutes:.2f}, "
            f"r2={r2_log:.4f}"
        )

    # Filter out inf folds for mean/std computation
    valid_mask = [v != float("inf") for v in fold_metrics["rmse_log"]]
    n_valid = sum(valid_mask)
    if n_valid == 0:
        logger.error("All CV folds failed — returning worst-case metrics.")
        return {
            "rmse_log_mean": float("inf"), "rmse_log_std": 0.0,
            "mae_minutes_mean": float("inf"), "mae_minutes_std": 0.0,
            "rmse_minutes_mean": float("inf"), "rmse_minutes_std": 0.0,
            "r2_log_mean": 0.0, "r2_log_std": 0.0,
            "n_valid_folds": 0,
        }

    result: dict[str, float] = {"n_valid_folds": float(n_valid)}
    for k, v in fold_metrics.items():
        valid_values = [val for val, ok in zip(v, valid_mask) if ok]
        result[f"{k}_mean"] = float(np.mean(valid_values))
        result[f"{k}_std"] = float(np.std(valid_values))

    return result


# ── Optuna Objective ──────────────────────────────────────────────────────────

class TimeSpentRegressorObjective(BaseObjective):
    """
    Callable Optuna objective for the Time Spent Regressor.

    Uses StratifiedGroupKFold (group=project_code, stratify=duration_bucket)
    to prevent project-level leakage, especially for SBM bimodality.

    Minimises RMSE on log scale.

    Usage
    -----
    objective = TimeSpentRegressorObjective(df, cv_folds=5, random_state=42)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        _validate_feature_cols(df)
        self.X = df[FEATURE_COLS].copy()
        self.y = df[TARGET_COL].to_numpy()
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Groups for StratifiedGroupKFold
        self.groups = df[GROUP_COL].to_numpy()

        # Stratification bins (duration_bucket or synthetic bins)
        if STRATIFY_COL in df.columns:
            self.stratify_bins = df[STRATIFY_COL].to_numpy()
        else:
            # Fallback: create bins from the target
            self.stratify_bins = pd.qcut(
                df[TARGET_COL], q=5, labels=False, duplicates="drop"
            ).to_numpy()
            logger.warning(
                f"TimeSpentRegressorObjective: '{STRATIFY_COL}' not found. "
                f"Using quantile bins for stratification."
            )

        # TaskCVImputer for per-fold imputation
        self.cv_imputer = TaskCVImputer()

    def __call__(self, trial: optuna.Trial) -> float:
        config = _sample_config(trial, self.random_state)

        try:
            pipeline = build_full_pipeline_time_spent(config)
            cv_results = cross_validate_regression(
                pipeline=pipeline,
                X=self.X,
                y=self.y,
                n_splits=self.cv_folds,
                random_state=self.random_state,
                groups=self.groups,
                stratify_bins=self.stratify_bins,
                cv_imputer=self.cv_imputer,
            )

            for k, v in cv_results.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("model_type", config.get("model_type"))

            objective_value = cv_results["rmse_log_mean"]

            # Guard: if all folds failed, prune this trial
            if objective_value == float("inf"):
                logger.warning(
                    f"Trial {trial.number}: all CV folds failed — pruning."
                )
                raise optuna.exceptions.TrialPruned()

            logger.info(
                f"Trial {trial.number} | model={config.get('model_type')} | "
                f"rmse_log={objective_value:.4f} | "
                f"mae_min={cv_results['mae_minutes_mean']:.2f} | "
                f"r2={cv_results['r2_log_mean']:.4f} | "
                f"valid_folds={int(cv_results['n_valid_folds'])}"
            )
            return objective_value

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            logger.warning(
                f"Trial {trial.number} FAILED: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            raise optuna.exceptions.TrialPruned()
