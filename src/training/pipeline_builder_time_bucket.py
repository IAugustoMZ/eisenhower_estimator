"""
PipelineBuilder — Time Bucket Classifier (Model 3b)
Constructs a full sklearn Pipeline for duration bucket classification:

  ColumnTransformer (reuses Model 3 preprocessor from pipeline_builder_time_spent)
      ↓
  FeatureSelectorTransformer   ← SelectKBest (f_classif / mutual_info_classif) or none
      ↓
  DimensionalityReducerTransformer  ← PCA / LDA / none
      ↓
  Classifier (LGBMClassifier / XGBClassifier / RF / LogisticRegression / voting)

Target: duration_bucket (6 ordered categories)
  ≤2 min → 0, 3–5 min → 1, 6–10 min → 2, 11–30 min → 3, 31–60 min → 4, >60 min → 5

Uses the same preprocessor as Model 3 (time spent regressor) since both share
identical feature engineering. Only the final estimator differs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.training.pipeline_builder_time_spent import (
    build_preprocessor_time_spent,
    TaskCVImputer,
)
from src.transformers import (
    DimensionalityReducerTransformer,
    FeatureSelectorTransformer,
)

# ── Bucket ordering (ascending by duration) ──────────────────────────────────
BUCKET_ORDER = [
    "≤2 min",       # 0
    "3–5 min",      # 1
    "6–10 min",     # 2
    "11–30 min",    # 3
    "31–60 min",    # 4
    ">60 min",      # 5
]
BUCKET_TO_INT = {b: i for i, b in enumerate(BUCKET_ORDER)}
N_CLASSES = len(BUCKET_ORDER)


# ── Feature selector (classification variants) ──────────────────────────────

def build_feature_selector_bucket(config: dict[str, Any]) -> FeatureSelectorTransformer:
    """Build a post-preprocessor feature selector for classification."""
    strategy = config.get("selector_strategy", "selectkbest_f")

    # Guard: remap regression strategies if accidentally passed
    _REMAP = {
        "selectkbest_f_reg": "selectkbest_f",
        "selectkbest_mi_reg": "selectkbest_mi",
    }
    if strategy in _REMAP:
        logger.warning(
            f"build_feature_selector_bucket: regression strategy '{strategy}' "
            f"remapped to '{_REMAP[strategy]}' for classification."
        )
        strategy = _REMAP[strategy]

    return FeatureSelectorTransformer(
        strategy=strategy,
        k=config.get("selector_k", "all"),
    )


def build_dim_reducer_bucket(config: dict[str, Any]) -> DimensionalityReducerTransformer:
    """Build the dimensionality reducer (PCA / LDA / none)."""
    method = config.get("reducer_method", "none")
    n_components = config.get("reducer_n_components", 20)

    # LDA max components = min(n_features, n_classes - 1) = 5
    if method == "lda":
        n_components = min(n_components, N_CLASSES - 1)

    return DimensionalityReducerTransformer(
        method=method,
        n_components=n_components,
        random_state=config.get("random_state", 42),
    )


# ── Classifier builders ──────────────────────────────────────────────────────

def build_classifier(config: dict[str, Any]):
    """
    Instantiate the multiclass classifier from config.

    model_type: lgbm / xgboost / random_forest / logistic /
                voting_lgbm_rf / voting_lgbm_lr / voting_all
    """
    model_type = config.get("model_type", "lgbm")
    random_state = config.get("random_state", 42)
    logger.info(f"PipelineBuilder (time_bucket): classifier='{model_type}'")

    if model_type == "lgbm":
        return _build_lgbm_classifier(config, random_state)
    if model_type == "xgboost":
        return _build_xgb_classifier(config, random_state)
    if model_type == "random_forest":
        return _build_rf_classifier(config, random_state)
    if model_type == "logistic":
        return _build_logistic(config, random_state)
    if model_type == "voting_lgbm_rf":
        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lgbm_classifier(config, random_state)),
                ("rf", _build_rf_classifier(config, random_state)),
            ],
            voting="soft",
        )
    if model_type == "voting_lgbm_lr":
        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lgbm_classifier(config, random_state)),
                ("lr", _build_logistic(config, random_state)),
            ],
            voting="soft",
        )
    if model_type == "voting_all":
        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lgbm_classifier(config, random_state)),
                ("xgb", _build_xgb_classifier(config, random_state)),
                ("rf", _build_rf_classifier(config, random_state)),
                ("lr", _build_logistic(config, random_state)),
            ],
            voting="soft",
        )

    raise ValueError(
        f"build_classifier: unknown model_type='{model_type}'. "
        f"Supported: lgbm, xgboost, random_forest, logistic, "
        f"voting_lgbm_rf, voting_lgbm_lr, voting_all"
    )


def _build_lgbm_classifier(config: dict, random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required. Run: pip install lightgbm") from exc

    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=config.get("lgbm_n_estimators", 300),
        max_depth=config.get("lgbm_max_depth", -1),
        learning_rate=config.get("lgbm_learning_rate", 0.05),
        num_leaves=config.get("lgbm_num_leaves", 31),
        min_child_samples=config.get("lgbm_min_child_samples", 20),
        subsample=config.get("lgbm_subsample", 0.8),
        colsample_bytree=config.get("lgbm_colsample_bytree", 0.8),
        reg_alpha=config.get("lgbm_reg_alpha", 0.0),
        reg_lambda=config.get("lgbm_reg_lambda", 1.0),
        class_weight=config.get("lgbm_class_weight", "balanced"),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgb_classifier(config: dict, random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required. Run: pip install xgboost") from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        n_estimators=config.get("xgb_n_estimators", 300),
        max_depth=config.get("xgb_max_depth", 6),
        learning_rate=config.get("xgb_learning_rate", 0.05),
        subsample=config.get("xgb_subsample", 0.8),
        colsample_bytree=config.get("xgb_colsample_bytree", 0.8),
        reg_alpha=config.get("xgb_reg_alpha", 0.0),
        reg_lambda=config.get("xgb_reg_lambda", 1.0),
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mlogloss",
    )


def _build_rf_classifier(config: dict, random_state: int):
    return RandomForestClassifier(
        n_estimators=config.get("rf_n_estimators", 300),
        max_depth=config.get("rf_max_depth", None),
        min_samples_split=config.get("rf_min_samples_split", 2),
        min_samples_leaf=config.get("rf_min_samples_leaf", 1),
        max_features=config.get("rf_max_features", "sqrt"),
        class_weight=config.get("rf_class_weight", "balanced"),
        random_state=random_state,
        n_jobs=-1,
    )


def _build_logistic(config: dict, random_state: int):
    return LogisticRegression(
        C=config.get("lr_C", 1.0),
        penalty=config.get("lr_penalty", "l2"),
        solver=config.get("lr_solver", "saga"),
        max_iter=config.get("lr_max_iter", 2000),
        class_weight=config.get("lr_class_weight", "balanced"),
        random_state=random_state,
        n_jobs=-1,
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_full_pipeline_time_bucket(config: dict[str, Any]) -> Pipeline:
    """
    Assemble the full sklearn Pipeline for time bucket classification:
      preprocessor → feature_selector → dim_reducer → classifier

    Reuses build_preprocessor_time_spent() for identical feature processing.
    """
    preprocessor = build_preprocessor_time_spent(config)
    feature_selector = build_feature_selector_bucket(config)
    dim_reducer = build_dim_reducer_bucket(config)
    classifier = build_classifier(config)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector),
        ("dim_reducer", dim_reducer),
        ("classifier", classifier),
    ])
