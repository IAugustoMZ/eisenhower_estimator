"""
OptunaObjective — Time Bucket Classifier (Model 3b)
Defines the Optuna search space for the duration bucket classification model.

Target: duration_bucket encoded as integers (0–5, ordered by duration).
Metric: macro F1 (maximise) — balances across all 6 classes.

Each Optuna trial:
  1. Samples a hyperparameter config (preprocessor + classifier)
  2. Applies TaskCVImputer per fold to impute task_cv and task_median_duration
  3. Runs StratifiedGroupKFold CV (group=project_code, stratify=duration_bucket_int)
  4. Applies optional SMOTE/ADASYN resampling on train folds
  5. Returns mean macro F1 as the objective value (maximise)

IMPORTANT:
  - TaskCVImputer is fit per fold with log_duration for leakage-safe imputation.
  - Resampling is applied ONLY after the preprocessor on train folds.
  - Classification strategies (selectkbest_f, selectkbest_mi) are used,
    not regression variants.
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
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

from src.training.base_objective import BaseObjective, _sample_model_config
from src.training.pipeline_builder_time_spent import TaskCVImputer
from src.training.pipeline_builder_time_bucket import (
    BUCKET_TO_INT,
    build_full_pipeline_time_bucket,
)
from src.transformers import ResamplerTransformer

# Silence Optuna's per-trial INFO logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature columns (same as Model 3 regressor) ─────────────────────────────
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
    # Iteration-2 features
    "task_type_x_repeated",
    "is_long_project",
    "log_task_freq",
    "desc_has_time_ref",
    "task_median_duration",
]
TARGET_COL = "duration_bucket"
LOG_DURATION_COL = "log_duration"
GROUP_COL = "project_code"

# Explicitly forbidden features (leakage / not significant)
FORBIDDEN_COLS = {
    "duration_minutes",     # raw target (duration)
    "log_duration",         # log of target
    "project_category",     # p=0.31
    "is_weekend",           # p=0.31
    "month",                # p=0.070
    "quarter",              # not significant
    "week_of_year",         # borderline noise
    "desc_word_count",      # p=0.156
    "project_name",         # redundant with project_code
    "date",                 # temporal leakage risk
}


def _validate_feature_cols(df: pd.DataFrame) -> None:
    """Raise early if required columns are missing."""
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"TimeBucketObjective: required feature columns missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TimeBucketObjective: target column '{TARGET_COL}' not found."
        )


def _encode_buckets(series: pd.Series) -> np.ndarray:
    """Convert string bucket labels to ordered integers (0–5)."""
    encoded = series.map(BUCKET_TO_INT)
    unmapped = encoded.isna()
    if unmapped.any():
        unknown = series[unmapped].unique().tolist()
        raise ValueError(
            f"Unknown duration buckets: {unknown}. "
            f"Expected one of: {list(BUCKET_TO_INT.keys())}"
        )
    return encoded.astype(int).to_numpy()


# ── Search space sampling ────────────────────────────────────────────────────

def _sample_bucket_config(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    """Sample a full hyperparameter config for the bucket classifier."""
    model_type = trial.suggest_categorical(
        "model_type",
        ["lgbm", "xgboost", "random_forest", "logistic",
         "voting_lgbm_rf", "voting_lgbm_lr", "voting_all"],
    )

    config: dict[str, Any] = {
        "model_type": model_type,
        "random_state": random_state,
        # ── Preprocessor (shared with regressor) ──
        "scaler_name": trial.suggest_categorical(
            "scaler_name", ["standard", "robust", "minmax", "none"]
        ),
        "include_text": trial.suggest_categorical("include_text", [True, False]),
        "target_enc_smoothing": trial.suggest_float(
            "target_enc_smoothing", 0.1, 10.0, log=True
        ),
        # ── Feature selection (classification-specific) ──
        "selector_strategy": trial.suggest_categorical(
            "selector_strategy", ["selectkbest_f", "selectkbest_mi", "none"]
        ),
        # ── Resampling (for class imbalance) ──
        "resampler_strategy": trial.suggest_categorical(
            "resampler_strategy", ["smote", "none"]
        ),
    }

    # selector_k: only tune when a selection strategy is active
    if config["selector_strategy"] != "none":
        config["selector_k"] = trial.suggest_int("selector_k", 5, 40)
    else:
        config["selector_k"] = "all"

    # ── Dimensionality reduction (PCA / LDA / none) ──
    config["reducer_method"] = trial.suggest_categorical(
        "reducer_method", ["pca", "lda", "none"]
    )
    if config["reducer_method"] in ("pca", "lda"):
        config["reducer_n_components"] = trial.suggest_int(
            "reducer_n_components", 3, 40, step=1
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
    config.update(_sample_model_config(trial, model_type))

    return config


# ── Multiclass cross-validation ──────────────────────────────────────────────

def cross_validate_bucket(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    resampler_strategy: str = "none",
    n_splits: int = 5,
    random_state: int = 42,
    groups: np.ndarray | None = None,
    cv_imputer: TaskCVImputer | None = None,
    y_log_duration: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Run multiclass CV with per-fold TaskCVImputer and optional resampling.

    Parameters
    ----------
    pipeline           : Unfitted sklearn Pipeline (cloned per fold)
    X                  : Feature DataFrame
    y                  : Target array (integer-encoded bucket labels, 0-5)
    resampler_strategy : "smote" / "adasyn" / "none"
    n_splits           : Number of folds
    random_state       : RNG seed
    groups             : Group labels for StratifiedGroupKFold
    cv_imputer         : TaskCVImputer (cloned and fit per fold)
    y_log_duration     : log_duration values for leakage-safe task_median_duration

    Returns
    -------
    dict with mean and std of: f1_macro, precision_macro, recall_macro, accuracy
    """
    # Build splitter
    if groups is not None:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_kwargs = {"X": X, "y": y, "groups": groups}
            list(splitter.split(**split_kwargs))  # test split
        except ValueError as exc:
            logger.warning(
                f"StratifiedGroupKFold failed ({exc}). "
                f"Falling back to GroupKFold."
            )
            n_groups = len(np.unique(groups))
            splitter = GroupKFold(n_splits=min(n_splits, n_groups))
            split_kwargs = {"X": X, "y": y, "groups": groups}
    else:
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_kwargs = {"X": X, "y": y}

    resampler = ResamplerTransformer(
        strategy=resampler_strategy, random_state=random_state
    )

    fold_metrics: dict[str, list[float]] = {
        "f1_macro": [], "precision_macro": [], "recall_macro": [], "accuracy": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(**split_kwargs)):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Guard: skip fold if it has fewer than 2 unique classes
        if len(np.unique(y_train_fold)) < 2:
            logger.warning(
                f"  Fold {fold_idx + 1}/{n_splits}: only one class in train — "
                f"skipping with neutral metrics."
            )
            fold_metrics["f1_macro"].append(0.0)
            fold_metrics["precision_macro"].append(0.0)
            fold_metrics["recall_macro"].append(0.0)
            fold_metrics["accuracy"].append(0.0)
            continue

        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)

        # Apply TaskCVImputer per fold
        if cv_imputer is not None:
            imputer_clone = clone(cv_imputer)
            X_train_with_target = X_train_fold.copy()
            if y_log_duration is not None:
                X_train_with_target["log_duration"] = y_log_duration[train_idx]
            imputer_clone.fit(X_train_with_target)
            X_train_fold = imputer_clone.transform(X_train_fold)
            X_val_fold = imputer_clone.transform(X_val_fold)

        pipeline_clone = clone(pipeline)

        try:
            if resampler_strategy != "none":
                # Preprocessor → resample → classifier
                preprocessor = pipeline_clone.named_steps["preprocessor"]
                classifier = pipeline_clone.named_steps["classifier"]
                X_transformed = preprocessor.fit_transform(X_train_fold, y_train_fold)
                X_resampled, y_resampled = resampler.fit_resample(
                    X_transformed, y_train_fold
                )

                # Fit feature selector + dim reducer + classifier on resampled data
                rest_pipeline = pipeline_clone[1:]  # feature_selector → dim_reducer → classifier
                rest_pipeline.fit(X_resampled, y_resampled)
            else:
                pipeline_clone.fit(X_train_fold, y_train_fold)
        except Exception as exc:
            logger.warning(
                f"  Fold {fold_idx + 1}/{n_splits}: fit failed ({exc}). "
                f"Skipping with neutral metrics."
            )
            fold_metrics["f1_macro"].append(0.0)
            fold_metrics["precision_macro"].append(0.0)
            fold_metrics["recall_macro"].append(0.0)
            fold_metrics["accuracy"].append(0.0)
            continue

        y_pred = pipeline_clone.predict(X_val_fold)

        fold_metrics["f1_macro"].append(
            f1_score(y_val_fold, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["precision_macro"].append(
            precision_score(y_val_fold, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["recall_macro"].append(
            recall_score(y_val_fold, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["accuracy"].append(
            float(np.mean(y_val_fold == y_pred))
        )

        logger.debug(
            f"  Fold {fold_idx + 1}/{n_splits}: "
            f"f1_macro={fold_metrics['f1_macro'][-1]:.4f}, "
            f"acc={fold_metrics['accuracy'][-1]:.4f}"
        )

    # Aggregate
    n_valid = sum(1 for v in fold_metrics["f1_macro"] if v > 0.0)
    if n_valid == 0:
        logger.error("All CV folds failed — returning worst-case metrics.")
        return {
            "f1_macro_mean": 0.0, "f1_macro_std": 0.0,
            "precision_macro_mean": 0.0, "precision_macro_std": 0.0,
            "recall_macro_mean": 0.0, "recall_macro_std": 0.0,
            "accuracy_mean": 0.0, "accuracy_std": 0.0,
            "n_valid_folds": 0.0,
        }

    result: dict[str, float] = {"n_valid_folds": float(n_valid)}
    for k, v in fold_metrics.items():
        result[f"{k}_mean"] = float(np.mean(v))
        result[f"{k}_std"] = float(np.std(v))
    return result


# ── Optuna Objective ──────────────────────────────────────────────────────────

class TimeBucketClassifierObjective(BaseObjective):
    """
    Callable Optuna objective for the Time Bucket Classifier.

    Uses StratifiedGroupKFold (group=project_code, stratify=bucket_int).
    Maximises macro F1 across 6 duration bucket classes.

    Usage
    -----
    objective = TimeBucketClassifierObjective(df, cv_folds=5, random_state=42)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        _validate_feature_cols(df)
        self.X = df[FEATURE_COLS].copy()
        self.y = _encode_buckets(df[TARGET_COL])
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Groups for StratifiedGroupKFold
        self.groups = df[GROUP_COL].to_numpy()

        # log_duration for leakage-safe task_median_duration computation
        if LOG_DURATION_COL in df.columns:
            self.y_log_duration = df[LOG_DURATION_COL].to_numpy()
        else:
            self.y_log_duration = None

        # TaskCVImputer for per-fold imputation
        self.cv_imputer = TaskCVImputer()

    def __call__(self, trial: optuna.Trial) -> float:
        config = _sample_bucket_config(trial, self.random_state)
        resampler_strategy = config.pop("resampler_strategy", "none")

        try:
            pipeline = build_full_pipeline_time_bucket(config)
            cv_results = cross_validate_bucket(
                pipeline=pipeline,
                X=self.X,
                y=self.y,
                resampler_strategy=resampler_strategy,
                n_splits=self.cv_folds,
                random_state=self.random_state,
                groups=self.groups,
                cv_imputer=self.cv_imputer,
                y_log_duration=self.y_log_duration,
            )

            for k, v in cv_results.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("model_type", config.get("model_type"))
            trial.set_user_attr("resampler_strategy", resampler_strategy)

            objective_value = cv_results["f1_macro_mean"]

            if objective_value == 0.0 and cv_results.get("n_valid_folds", 0) == 0:
                logger.warning(
                    f"Trial {trial.number}: all CV folds failed — pruning."
                )
                raise optuna.exceptions.TrialPruned()

            logger.info(
                f"Trial {trial.number} | model={config.get('model_type')} | "
                f"f1_macro={objective_value:.4f} | "
                f"acc={cv_results['accuracy_mean']:.4f} | "
                f"resampler={resampler_strategy} | "
                f"valid_folds={int(cv_results.get('n_valid_folds', 0))}"
            )
            return objective_value

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            logger.warning(
                f"Trial {trial.number} failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
            raise optuna.exceptions.TrialPruned()
