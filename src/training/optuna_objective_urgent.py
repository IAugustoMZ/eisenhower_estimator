"""
OptunaObjectiveUrgent
Defines the Optuna search space for Model 2 (urgent classifier).

Covers:
  - Preprocessor: scaler, text vectorizer, LSA, lead_time encoding
  - Model selection: lgbm / xgboost / random_forest / logistic / voting variants
  - Resampling: smote / adasyn / none (CRITICAL for 21.36:1 imbalance)
  - Class imbalance weighting (scale_pos_weight 15–25)

CV Strategy: StratifiedGroupKFold (group=project_code)
  Rationale: urgent is almost exclusively a Work project phenomenon. Grouping by
  project_code tests whether the model generalises to *new projects not seen
  during training*, which is the real-world deployment scenario.
  Fallback: if any fold has zero positive samples, that fold is given neutral
  scores (f1_macro=0, roc_auc=0.5) — the mean across folds will be penalised
  naturally. The trainer logs a warning.

IMPORTANT: SafeTargetEncoder is fit per fold inside CV to prevent leakage.
NEVER include is_overdue — it is not available at task creation time.
"""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold

from src.training.base_objective import (
    BaseObjective,
    _sample_model_config,
    cross_validate_pipeline,
)
from src.training.pipeline_builder_urgent import build_full_pipeline_urgent

# Silence Optuna's per-trial INFO logs (we log our own)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature columns used in Model 2 ──────────────────────────────────────────
FEATURE_COLS = [
    "hour_created",
    "day_of_week_created",
    "month_created",
    "project_code",
    "project_category",
    "lead_time_bucket",
    "lead_time_days",
    "lead_time_hours",
    "desc_word_count",
    "desc_char_len",
    "desc_clean",
]
TARGET_COL = "urgent"
GROUP_COL = "project_code"   # StratifiedGroupKFold groups by project

# Explicitly forbidden features (leakage guards for Model 2)
FORBIDDEN_COLS = {
    "important",      # target leakage for Model 2
    "is_overdue",     # NOT available at task creation time
    "days_until_due", # not a Model 2 feature — lead_time_* replace it
    "has_due_date",   # constant
}


def _validate_feature_cols(df: pd.DataFrame) -> None:
    """Raise early if any forbidden column sneaked in, or required columns are missing."""
    present_forbidden = set(df.columns) & FORBIDDEN_COLS
    if present_forbidden:
        raise ValueError(
            f"UrgentObjective: forbidden columns detected in DataFrame: {present_forbidden}. "
            f"These must be removed before training to prevent leakage."
        )
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"UrgentObjective: required feature columns missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(f"UrgentObjective: target column '{TARGET_COL}' not found in DataFrame.")


def _sample_config(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    """
    Sample a full hyperparameter config from the Optuna search space for Model 2.

    Key differences vs Model 1:
      - scale_pos_weight range: 10–30 (for 21.36:1 imbalance, vs 1–5 for Model 1)
      - month_created included (V=0.326 — significant for urgency)
      - lead_time features replace days_until_due
    """
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
        # ── Resampling — critical for 21.36:1 imbalance ──
        "resampler_strategy": trial.suggest_categorical("resampler_strategy", ["smote", "adasyn", "none"]),
        # ── Feature selection (post-preprocessor, supervised) ──
        "selector_strategy": trial.suggest_categorical(
            "selector_strategy", ["selectkbest_f", "selectkbest_mi", "none"]
        ),
    }

    # selector_k: only tune when a selection strategy is active
    if config["selector_strategy"] != "none":
        config["selector_k"] = trial.suggest_int("selector_k", 5, 50)
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
    # scale_pos_weight: 10–30 range for 21.36:1 imbalance
    # The EDA recommends ~21; allow search to explore the neighbourhood.
    if model_type in ("lgbm", "xgboost", "voting_lgbm_rf", "voting_lgbm_lr", "voting_all"):
        config["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 10.0, 30.0)

    config.update(_sample_model_config(trial, model_type))

    return config


class UrgentClassifierObjective(BaseObjective):
    """
    Callable Optuna objective for the Urgent classifier.

    Uses StratifiedGroupKFold (group=project_code) to test generalisation
    to unseen projects — the real-world deployment scenario.

    Fallback: folds with zero positive samples receive neutral scores (f1=0,
    roc_auc=0.5) to avoid crashes while penalising bad splits naturally.

    Usage
    -----
    objective = UrgentClassifierObjective(df, cv_folds=5, random_state=42)
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
        self.groups = df[GROUP_COL].to_numpy()   # group vector for StratifiedGroupKFold
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Log class distribution for visibility
        pos_rate = self.y.mean()
        logger.info(
            f"UrgentClassifierObjective: {len(self.y)} samples, "
            f"{pos_rate:.1%} positive (urgent), "
            f"imbalance ratio ≈ {(1 - pos_rate) / pos_rate:.1f}:1"
        )

    def __call__(self, trial: optuna.Trial) -> float:
        config = _sample_config(trial, self.random_state)
        resampler_strategy = config.pop("resampler_strategy")

        try:
            pipeline = build_full_pipeline_urgent(config)

            # StratifiedGroupKFold: stratified on y, grouped on project_code
            # This ensures each fold tests on projects unseen during training.
            splitter = StratifiedGroupKFold(n_splits=self.cv_folds)

            cv_results = cross_validate_pipeline(
                pipeline=pipeline,
                X=self.X,
                y=self.y,
                resampler_strategy=resampler_strategy,
                n_splits=self.cv_folds,
                random_state=self.random_state,
                splitter=splitter,
                groups=self.groups,
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
