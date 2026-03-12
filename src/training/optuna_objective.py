"""
OptunaObjective
Defines the Optuna search space for Model 1 (important classifier).

Covers:
  - Preprocessor: scaler, text vectorizer, LSA settings
  - Model selection: lgbm / xgboost / random_forest / logistic / voting variants
  - Resampling: smote / adasyn / none
  - Class imbalance weighting

Each Optuna trial:
  1. Samples a hyperparameter config
  2. Applies resampling to training folds only
  3. Runs StratifiedKFold CV
  4. Returns mean F1-macro as the objective value (maximise)

IMPORTANT: SafeTargetEncoder is fit per fold inside CV to prevent leakage.
"""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.training.base_objective import (
    BaseObjective,
    _clone_pipeline,
    _fit_with_resampling,
    _safe_predict_proba,
    _sample_model_config,
    cross_validate_pipeline,
)
from src.training.pipeline_builder import build_full_pipeline
from src.transformers import ResamplerTransformer

# Silence Optuna's per-trial INFO logs (we log our own)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature columns used in this model ───────────────────────────────────────
FEATURE_COLS = [
    "hour_created",
    "day_of_week_created",
    "project_code",
    "project_category",
    "days_until_due",
    "desc_word_count",
    "desc_char_len",
    "desc_clean",
]
TARGET_COL = "important"

# Explicitly forbidden features (leakage guards)
FORBIDDEN_COLS = {"urgent", "is_overdue", "month_created", "has_due_date", "days_since_created"}


def _validate_feature_cols(df: pd.DataFrame) -> None:
    """Raise early if any forbidden column sneaked in, or required columns are missing."""
    present_forbidden = set(df.columns) & FORBIDDEN_COLS
    if present_forbidden:
        raise ValueError(
            f"OptunaObjective: forbidden columns detected in DataFrame: {present_forbidden}. "
            f"These must be removed before training to prevent leakage."
        )
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"OptunaObjective: required feature columns missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(f"OptunaObjective: target column '{TARGET_COL}' not found in DataFrame.")


def _sample_config(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    """Sample a full hyperparameter config from the Optuna search space for Model 1."""
    model_type = trial.suggest_categorical(
        "model_type",
        ["lgbm", "xgboost", "random_forest", "logistic", "voting_lgbm_rf", "voting_lgbm_lr", "voting_all"],
    )

    config: dict[str, Any] = {
        "model_type": model_type,
        "random_state": random_state,
        # ── Preprocessor ──
        "scaler_name": trial.suggest_categorical("scaler_name", ["standard", "robust", "minmax", "none"]),
        "include_text": trial.suggest_categorical("include_text", [True, False]),
        "target_enc_smoothing": trial.suggest_float("target_enc_smoothing", 0.1, 10.0, log=True),
        # ── Resampling ──
        "resampler_strategy": trial.suggest_categorical("resampler_strategy", ["smote", "adasyn", "none"]),
        # ── Feature selection (post-preprocessor, supervised) ──
        "selector_strategy": trial.suggest_categorical(
            "selector_strategy", ["selectkbest_f", "selectkbest_mi", "none"]
        ),
    }

    # selector_k: only tune when a selection strategy is active
    if config["selector_strategy"] != "none":
        config["selector_k"] = trial.suggest_int("selector_k", 5, 40)
    else:
        config["selector_k"] = "all"

    # ── Dimensionality reduction (post-selector) ──
    config["reducer_method"] = trial.suggest_categorical("reducer_method", ["pca", "lda", "none"])
    if config["reducer_method"] == "pca":
        config["reducer_n_components"] = trial.suggest_int("reducer_n_components", 5, 40, step=5)
    else:
        config["reducer_n_components"] = 1

    # Text hyperparams (only sample if text is included)
    if config["include_text"]:
        config["vectorizer_name"] = trial.suggest_categorical("vectorizer_name", ["tfidf", "bow"])
        config["text_max_features"] = trial.suggest_int("text_max_features", 100, 1000, step=100)
        config["text_ngram_max"] = trial.suggest_int("text_ngram_max", 1, 2)
        config["text_use_lsa"] = trial.suggest_categorical("text_use_lsa", [True, False])
        config["text_n_components"] = trial.suggest_int("text_n_components", 10, 60, step=10)
        config["text_min_df"] = trial.suggest_int("text_min_df", 1, 5)
    else:
        config["vectorizer_name"] = "tfidf"
        config["text_use_lsa"] = True

    # ── Model-specific hyperparameters ─────────────────────────────────────
    # scale_pos_weight applies to lgbm / xgboost (3:1 imbalance → ~3.0)
    if model_type in ("lgbm", "xgboost", "voting_lgbm_rf", "voting_lgbm_lr", "voting_all"):
        config["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 5.0)

    config.update(_sample_model_config(trial, model_type))

    return config


class ImportantClassifierObjective(BaseObjective):
    """
    Callable Optuna objective for the Important classifier.

    Uses StratifiedKFold (not grouped) since project_code is not the primary
    signal and generalization to unseen projects is not the primary concern.

    Usage
    -----
    objective = ImportantClassifierObjective(df, cv_folds=5, random_state=42)
    study = optuna.create_study(direction="maximize")
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

    def __call__(self, trial: optuna.Trial) -> float:
        config = _sample_config(trial, self.random_state)
        resampler_strategy = config.pop("resampler_strategy")

        try:
            pipeline = build_full_pipeline(config)
            cv_results = cross_validate_pipeline(
                pipeline=pipeline,
                X=self.X,
                y=self.y,
                resampler_strategy=resampler_strategy,
                n_splits=self.cv_folds,
                random_state=self.random_state,
                splitter=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
            )

            for k, v in cv_results.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("resampler_strategy", resampler_strategy)
            trial.set_user_attr("model_type", config.get("model_type"))

            objective_value = cv_results["f1_macro_mean"]
            logger.info(
                f"Trial {trial.number} | model={config.get('model_type')} | "
                f"f1_macro={objective_value:.4f} | "
                f"f1_0={cv_results['f1_0_mean']:.4f} | "
                f"f1_1={cv_results['f1_1_mean']:.4f} | "
                f"roc_auc={cv_results['roc_auc_mean']:.4f}"
            )
            return objective_value

        except Exception as exc:
            logger.warning(
                f"Trial {trial.number} FAILED: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            raise optuna.exceptions.TrialPruned()
