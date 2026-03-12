"""
BaseObjective
Shared utilities and abstract base class for all Optuna objectives.

Subclasses (ImportantClassifierObjective, UrgentClassifierObjective) inherit
from BaseObjective and only need to define:
  - self.X, self.y, self.cv_folds, self.random_state
  - __call__() to sample a model-specific config and invoke cross_validate_pipeline()

Shared utilities (module-level functions):
  - cross_validate_pipeline()  — generic CV loop with resampling on train folds only
  - _fit_with_resampling()     — preprocessor → resample → classifier
  - _clone_pipeline()          — sklearn clone wrapper
  - _safe_predict_proba()      — safe probability extraction
  - _sample_model_config()     — shared model hyperparameter sampling (lgbm, xgb, rf, lr)
"""
from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
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
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.transformers import ResamplerTransformer


# ── Shared model hyperparameter sampling ──────────────────────────────────────

def _sample_model_config(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    """
    Sample model-specific hyperparameters shared across all objectives.

    Each model type is guarded so parameters are only sampled when needed.
    Returns a partial config dict to be merged into the caller's config.
    """
    config: dict[str, Any] = {}

    if model_type in ("lgbm", "voting_lgbm_rf", "voting_lgbm_lr", "voting_all"):
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
        config["rf_class_weight"] = trial.suggest_categorical(
            "rf_class_weight", ["balanced", "balanced_subsample"]
        )

    if model_type in ("logistic", "voting_lgbm_lr", "voting_all"):
        config["lr_C"] = trial.suggest_float("lr_C", 1e-4, 100.0, log=True)
        config["lr_penalty"] = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        config["lr_solver"] = "saga"    # supports both l1 and l2
        config["lr_max_iter"] = 2000
        config["lr_class_weight"] = trial.suggest_categorical("lr_class_weight", ["balanced", None])

    return config


# ── Cross-validation loop ─────────────────────────────────────────────────────

def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    resampler_strategy: str,
    n_splits: int = 5,
    random_state: int = 42,
    splitter: BaseCrossValidator | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Run CV manually (not sklearn cross_val_score) so we can:
      1. Apply resampling ONLY to training folds (never validation)
      2. Fit SafeTargetEncoder per fold (no leakage)
      3. Support any sklearn splitter (StratifiedKFold, StratifiedGroupKFold, etc.)
      4. Gracefully fall back to StratifiedKFold if a fold has no positive samples
         (edge case for severe imbalance + grouped splits)

    Parameters
    ----------
    pipeline          : Unfitted sklearn Pipeline (will be cloned per fold)
    X                 : Feature DataFrame (reset_index before passing)
    y                 : Target array (numpy, 0-based index)
    resampler_strategy: "smote" | "adasyn" | "none"
    n_splits          : Number of folds
    random_state      : RNG seed for reproducibility
    splitter          : CV splitter instance. If None, defaults to StratifiedKFold.
    groups            : Group labels for group-aware splitters (e.g. StratifiedGroupKFold)

    Returns
    -------
    dict with mean and std of: f1_macro, f1_0, f1_1, roc_auc, precision_macro, recall_macro
    """
    if splitter is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    resampler = ResamplerTransformer(strategy=resampler_strategy, random_state=random_state)

    fold_metrics: dict[str, list[float]] = {
        "f1_macro": [], "f1_0": [], "f1_1": [],
        "roc_auc": [], "precision_macro": [], "recall_macro": [],
    }

    split_kwargs: dict[str, Any] = {"X": X, "y": y}
    if groups is not None:
        split_kwargs["groups"] = groups

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(**split_kwargs)):
        # Guard: skip fold if it has no positive samples (edge case — severe imbalance)
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        if len(np.unique(y_train_fold)) < 2:
            logger.warning(
                f"  Fold {fold_idx + 1}/{n_splits}: only one class in train fold — skipping. "
                f"Falling back to neutral metrics."
            )
            fold_metrics["f1_macro"].append(0.0)
            fold_metrics["f1_0"].append(0.0)
            fold_metrics["f1_1"].append(0.0)
            fold_metrics["roc_auc"].append(0.5)
            fold_metrics["precision_macro"].append(0.0)
            fold_metrics["recall_macro"].append(0.0)
            continue

        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)

        pipeline_clone = _clone_pipeline(pipeline)

        if resampler_strategy != "none":
            pipeline_clone = _fit_with_resampling(
                pipeline, X_train_fold, y_train_fold, resampler
            )
        else:
            pipeline_clone.fit(X_train_fold, y_train_fold)

        y_pred = pipeline_clone.predict(X_val_fold)
        y_proba = _safe_predict_proba(pipeline_clone, X_val_fold)

        f1_per_class = f1_score(y_val_fold, y_pred, average=None, zero_division=0)
        fold_metrics["f1_macro"].append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
        fold_metrics["f1_0"].append(float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0)
        fold_metrics["f1_1"].append(float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0)
        fold_metrics["precision_macro"].append(
            precision_score(y_val_fold, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["recall_macro"].append(
            recall_score(y_val_fold, y_pred, average="macro", zero_division=0)
        )

        if y_proba is not None:
            try:
                fold_metrics["roc_auc"].append(roc_auc_score(y_val_fold, y_proba))
            except ValueError:
                fold_metrics["roc_auc"].append(0.5)
        else:
            fold_metrics["roc_auc"].append(0.5)

        logger.debug(
            f"  Fold {fold_idx + 1}/{n_splits}: "
            f"f1_macro={fold_metrics['f1_macro'][-1]:.4f}, "
            f"f1_0={fold_metrics['f1_0'][-1]:.4f}, "
            f"f1_1={fold_metrics['f1_1'][-1]:.4f}"
        )

    return {
        f"{k}_mean": float(np.mean(v)) for k, v in fold_metrics.items()
    } | {
        f"{k}_std": float(np.std(v)) for k, v in fold_metrics.items()
    }


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _fit_with_resampling(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    resampler: ResamplerTransformer,
) -> Pipeline:
    """
    Fit preprocessor, apply resampling on transformed features, then fit classifier.

    Resampling is applied ONLY after the preprocessor so that:
      - SafeTargetEncoder sees only the real training distribution
      - SMOTE/ADASYN operates in the encoded feature space, not raw categoricals

    Returns a new fitted pipeline.
    """
    pipeline_clone = _clone_pipeline(pipeline)
    preprocessor = pipeline_clone.named_steps["preprocessor"]
    classifier = pipeline_clone.named_steps["classifier"]

    X_transformed = preprocessor.fit_transform(X_train, y_train)
    X_resampled, y_resampled = resampler.fit_resample(X_transformed, y_train)
    classifier.fit(X_resampled, y_resampled)
    return pipeline_clone


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    """Return a fresh unfitted clone of the pipeline."""
    return clone(pipeline)


def _safe_predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Return probability of positive class, or None if not supported."""
    try:
        proba = pipeline.predict_proba(X)
        return proba[:, 1]
    except (AttributeError, NotImplementedError):
        return None


# ── Abstract base class ───────────────────────────────────────────────────────

class BaseObjective(ABC):
    """
    Abstract base class for all Optuna objectives.

    Subclasses must define:
      - self.X  (pd.DataFrame, feature matrix)
      - self.y  (np.ndarray, target array)
      - self.cv_folds  (int)
      - self.random_state (int)
      - __call__(trial) -> float  (samples config, runs CV, returns objective value)

    The __call__ method should follow this pattern:
      1. Sample a config using model-specific _sample_config()
      2. Pop "resampler_strategy" from config
      3. Call build_full_pipeline(config)
      4. Call cross_validate_pipeline(...)
      5. Store user_attrs on trial
      6. Return cv_results["f1_macro_mean"]
      7. Catch exceptions → raise optuna.exceptions.TrialPruned()
    """

    @abstractmethod
    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective function. Must return a scalar to maximize."""
        ...
