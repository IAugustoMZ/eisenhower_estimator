"""
PipelineBuilder
Constructs a full sklearn Pipeline from a config dict:

  ColumnTransformer (per-feature-group encoding)
      ↓
  FeatureSelectorTransformer   ← SelectKBest (f_classif / mutual_info) or none
      ↓
  DimensionalityReducerTransformer  ← PCA / LDA / none
      ↓
  Classifier

Feature groups in ColumnTransformer:
  - Cyclical temporal  : hour_created, day_of_week_created
  - Target-encoded cat : project_code
  - Ordinal-encoded cat: project_category ("Personal"/"Work" → 0/1)
  - Numeric (scaled)   : days_until_due, desc_word_count, desc_char_len
  - Text               : desc_clean (TF-IDF or BoW + optional LSA)

Features intentionally EXCLUDED:
  - is_overdue        — not available at inference time
  - month_created     — not significant (p=0.43 in EDA)
  - has_due_date      — constant feature
  - days_since_created — redundant with days_until_due
  - urgent            — target leakage guard for Model 1
"""
from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.transformers import (
    CyclicalEncoder,
    DimensionalityReducerTransformer,
    FeatureSelectorTransformer,
    SafeTargetEncoder,
    ScalerSelector,
    TextVectorizerTransformer,
)

# ── Feature column constants ──────────────────────────────────────────────────
HOUR_COL = "hour_created"
DOW_COL = "day_of_week_created"
PROJECT_CODE_COL = "project_code"
PROJECT_CATEGORY_COL = "project_category"
NUMERIC_COLS = ["days_until_due", "desc_word_count", "desc_char_len"]
TEXT_COL = "desc_clean"

# project_category is "Personal" / "Work" string — encode as 0/1
CATEGORY_COLS = [PROJECT_CATEGORY_COL]


class _LogThenScale(BaseEstimator, TransformerMixin):
    """
    Applies log1p (after clipping negatives to 0) then delegates to ScalerSelector.
    Used for numeric columns (days_until_due can be legitimately negative for overdue tasks).
    Implemented as a single transformer instead of nested Pipeline to avoid
    sklearn ColumnTransformer clone/routing issues with sub-pipelines.
    """

    def __init__(self, scaler_name: str = "robust", columns: list | None = None):
        self.scaler_name = scaler_name
        self.columns = columns

    def fit(self, X, y=None):
        arr = self._prepare(X)
        self._scaler = ScalerSelector(scaler_name=self.scaler_name, columns=self.columns)
        self._scaler.fit(arr, y)
        return self

    def transform(self, X, y=None):
        arr = self._prepare(X)
        return self._scaler.transform(arr)

    def get_feature_names_out(self, input_features=None):
        return self._scaler.get_feature_names_out(input_features)

    @staticmethod
    def _prepare(X) -> np.ndarray:
        import pandas as pd
        arr = X.to_numpy(dtype=float) if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X, dtype=float)
        arr = np.clip(arr, 0, None)
        return np.log1p(arr)


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """
    Build the ColumnTransformer from a hyperparameter config dict.

    Config keys (all optional with defaults):
    ┌────────────────────────┬──────────────────────────────────┬──────────┐
    │ Key                    │ Values                           │ Default  │
    ├────────────────────────┼──────────────────────────────────┼──────────┤
    │ scaler_name            │ standard/robust/minmax/none      │ robust   │
    │ vectorizer_name        │ tfidf/bow                        │ tfidf    │
    │ text_max_features      │ int                              │ 500      │
    │ text_ngram_max         │ 1 or 2                           │ 2        │
    │ text_use_lsa           │ bool                             │ True     │
    │ text_n_components      │ int                              │ 30       │
    │ text_min_df            │ int                              │ 2        │
    │ target_enc_smoothing   │ float                            │ 1.0      │
    │ include_text           │ bool                             │ True     │
    └────────────────────────┴──────────────────────────────────┴──────────┘
    """
    scaler_name = config.get("scaler_name", "robust")
    vectorizer_name = config.get("vectorizer_name", "tfidf")
    text_max_features = config.get("text_max_features", 500)
    text_ngram_max = config.get("text_ngram_max", 2)
    text_use_lsa = config.get("text_use_lsa", True)
    text_n_components = config.get("text_n_components", 30)
    text_min_df = config.get("text_min_df", 2)
    target_enc_smoothing = config.get("target_enc_smoothing", 1.0)
    include_text = config.get("include_text", True)

    log_then_scale = _LogThenScale(scaler_name=scaler_name, columns=NUMERIC_COLS)

    transformers = [
        (
            "hour_cyclical",
            CyclicalEncoder(column=HOUR_COL, period=24),
            [HOUR_COL],
        ),
        (
            "dow_cyclical",
            CyclicalEncoder(column=DOW_COL, period=7),
            [DOW_COL],
        ),
        (
            "project_code_target",
            SafeTargetEncoder(column=PROJECT_CODE_COL, smoothing=target_enc_smoothing),
            [PROJECT_CODE_COL],
        ),
        (
            "category_binary",
            OrdinalEncoder(
                categories=[["Personal", "Work"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            CATEGORY_COLS,
        ),
        (
            "numeric_scaled",
            log_then_scale,
            NUMERIC_COLS,
        ),
    ]

    if include_text:
        transformers.append((
            "text_features",
            TextVectorizerTransformer(
                column=TEXT_COL,
                vectorizer_name=vectorizer_name,
                max_features=text_max_features,
                ngram_range=(1, text_ngram_max),
                use_lsa=text_use_lsa,
                n_components=text_n_components,
                min_df=text_min_df,
            ),
            [TEXT_COL],
        ))
        logger.debug(
            f"PipelineBuilder: text ENABLED "
            f"(vectorizer={vectorizer_name}, lsa={text_use_lsa}, n_components={text_n_components})"
        )
    else:
        logger.debug("PipelineBuilder: text DISABLED for this trial.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_selector(config: dict[str, Any]) -> FeatureSelectorTransformer:
    """
    Build the post-preprocessor feature selector.

    Config keys:
    ┌────────────────────────┬────────────────────────────────────┬──────────────┐
    │ Key                    │ Values                             │ Default      │
    ├────────────────────────┼────────────────────────────────────┼──────────────┤
    │ selector_strategy      │ selectkbest_f/selectkbest_mi/none  │ selectkbest_f│
    │ selector_k             │ int or "all"                       │ "all"        │
    └────────────────────────┴────────────────────────────────────┴──────────────┘

    Recommendation: 'selectkbest_f' is the default — fast, supervised, generalises
    well for mixed numeric+encoded features. 'selectkbest_mi' is slower but captures
    non-linear dependencies. 'none' is safest when feature count is small.
    """
    return FeatureSelectorTransformer(
        strategy=config.get("selector_strategy", "selectkbest_f"),
        k=config.get("selector_k", "all"),
    )


def build_dim_reducer(config: dict[str, Any]) -> DimensionalityReducerTransformer:
    """
    Build the post-selector dimensionality reducer.

    Config keys:
    ┌─────────────────────┬────────────────────┬──────────┐
    │ Key                 │ Values             │ Default  │
    ├─────────────────────┼────────────────────┼──────────┤
    │ reducer_method      │ pca / lda / none   │ none     │
    │ reducer_n_components│ int                │ 20       │
    └─────────────────────┴────────────────────┴──────────┘

    Recommendation: 'pca' when text features are included (decorrelates LSA dims
    from cyclical/numeric features). 'lda' produces exactly 1 component for binary
    classification — only use as an aggressive regularization experiment.
    'none' is the recommended default when text is excluded.
    """
    return DimensionalityReducerTransformer(
        method=config.get("reducer_method", "none"),
        n_components=config.get("reducer_n_components", 20),
        random_state=config.get("random_state", 42),
    )


def build_classifier(config: dict[str, Any]):
    """
    Instantiate the classifier from config.

    model_type options: lgbm / xgboost / random_forest / logistic /
                        voting_lgbm_rf / voting_lgbm_lr / voting_all
    """
    model_type = config.get("model_type", "lgbm")
    random_state = config.get("random_state", 42)
    logger.info(f"PipelineBuilder: classifier='{model_type}'")

    if model_type == "lgbm":
        return _build_lgbm(config, random_state)
    if model_type == "xgboost":
        return _build_xgboost(config, random_state)
    if model_type == "random_forest":
        return _build_random_forest(config, random_state)
    if model_type == "logistic":
        return _build_logistic(config, random_state)
    if model_type == "voting_lgbm_rf":
        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lgbm(config, random_state)),
                ("rf", _build_random_forest(config, random_state)),
            ],
            voting="soft",
        )
    if model_type == "voting_lgbm_lr":
        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lgbm(config, random_state)),
                ("lr", _build_logistic(config, random_state)),
            ],
            voting="soft",
        )
    if model_type == "voting_all":
        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lgbm(config, random_state)),
                ("xgb", _build_xgboost(config, random_state)),
                ("rf", _build_random_forest(config, random_state)),
                ("lr", _build_logistic(config, random_state)),
            ],
            voting="soft",
        )

    raise ValueError(
        f"build_classifier: unknown model_type='{model_type}'. "
        f"Supported: lgbm, xgboost, random_forest, logistic, "
        f"voting_lgbm_rf, voting_lgbm_lr, voting_all"
    )


def _build_lgbm(config: dict, random_state: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required. Run: pip install lightgbm") from exc

    return LGBMClassifier(
        n_estimators=config.get("lgbm_n_estimators", 300),
        max_depth=config.get("lgbm_max_depth", -1),
        learning_rate=config.get("lgbm_learning_rate", 0.05),
        num_leaves=config.get("lgbm_num_leaves", 31),
        min_child_samples=config.get("lgbm_min_child_samples", 20),
        subsample=config.get("lgbm_subsample", 0.8),
        colsample_bytree=config.get("lgbm_colsample_bytree", 0.8),
        reg_alpha=config.get("lgbm_reg_alpha", 0.0),
        reg_lambda=config.get("lgbm_reg_lambda", 1.0),
        scale_pos_weight=config.get("scale_pos_weight", 1.0),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgboost(config: dict, random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required. Run: pip install xgboost") from exc

    return XGBClassifier(
        n_estimators=config.get("xgb_n_estimators", 300),
        max_depth=config.get("xgb_max_depth", 6),
        learning_rate=config.get("xgb_learning_rate", 0.05),
        subsample=config.get("xgb_subsample", 0.8),
        colsample_bytree=config.get("xgb_colsample_bytree", 0.8),
        reg_alpha=config.get("xgb_reg_alpha", 0.0),
        reg_lambda=config.get("xgb_reg_lambda", 1.0),
        scale_pos_weight=config.get("scale_pos_weight", 1.0),
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )


def _build_random_forest(config: dict, random_state: int):
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


def build_full_pipeline(config: dict[str, Any]) -> Pipeline:
    """
    Assemble the full sklearn Pipeline:
      preprocessor → feature_selector → dim_reducer → classifier

    Parameters
    ----------
    config : dict
        Combined hyperparameter dict (preprocessor + selector + reducer + classifier).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = build_preprocessor(config)
    feature_selector = build_feature_selector(config)
    dim_reducer = build_dim_reducer(config)
    classifier = build_classifier(config)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector),
        ("dim_reducer", dim_reducer),
        ("classifier", classifier),
    ])
