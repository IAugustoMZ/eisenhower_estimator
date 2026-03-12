"""
PipelineBuilderUrgent
Constructs the full sklearn Pipeline for Model 2 (urgent classifier).

Feature groups in ColumnTransformer:
  - Cyclical temporal  : hour_created, day_of_week_created, month_created
  - Target-encoded cat : project_code
  - Ordinal-encoded cat: project_category ("Personal"/"Work" → 0/1)
  - Ordinal-encoded cat: lead_time_bucket (ordered bucket label → int)
  - Numeric (scaled)   : lead_time_days, lead_time_hours, desc_word_count, desc_char_len
  - Text (optional)    : desc_clean (TF-IDF / BoW + optional LSA)

Key differences from Model 1 pipeline:
  - lead_time_days / lead_time_hours: CAN be negative (retroactive tasks) — NOT log-transformed
  - month_created: included here (V=0.326 for urgency — significant unlike Model 1)
  - lead_time_bucket: ordinal categorical with a fixed ordered mapping
  - days_until_due: EXCLUDED (not used in Model 2 — use lead_time_* instead)
  - is_overdue: EXCLUDED (not available at task creation time)

Features intentionally EXCLUDED:
  - important        — target leakage guard for Model 2
  - is_overdue       — not available at inference time
  - days_until_due   — not a Model 2 feature (lead_time_* replace it)
  - has_due_date     — constant (all tasks have due dates)
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
MONTH_COL = "month_created"
PROJECT_CODE_COL = "project_code"
PROJECT_CATEGORY_COL = "project_category"
LEAD_TIME_BUCKET_COL = "lead_time_bucket"

# Numeric cols that CAN be negative — scaled only (no log1p)
NUMERIC_COLS = ["lead_time_days", "lead_time_hours", "desc_word_count", "desc_char_len"]

TEXT_COL = "desc_clean"

# Lead time bucket ordering (from EDA: short overdue → very short → short → medium → long)
# These are the expected string labels in the data — encode as ordinal ints
LEAD_TIME_BUCKET_ORDER = [
    "overdue_long",
    "overdue_short",
    "very_short",
    "short",
    "medium",
    "long",
]


class _ScaleOnly(BaseEstimator, TransformerMixin):
    """
    Applies ScalerSelector directly without log1p.

    Used for lead_time_days / lead_time_hours which can be legitimately
    negative (retroactive tasks created after their due date). Log-transforming
    negative values would produce NaN. RobustScaler is recommended here as it
    is insensitive to outliers and handles negative values.
    """

    def __init__(self, scaler_name: str = "robust", columns: list | None = None):
        self.scaler_name = scaler_name
        self.columns = columns

    def fit(self, X, y=None):
        arr = self._to_array(X)
        self._scaler = ScalerSelector(scaler_name=self.scaler_name, columns=self.columns)
        self._scaler.fit(arr, y)
        return self

    def transform(self, X, y=None):
        arr = self._to_array(X)
        return self._scaler.transform(arr)

    def get_feature_names_out(self, input_features=None):
        return self._scaler.get_feature_names_out(input_features)

    @staticmethod
    def _to_array(X) -> np.ndarray:
        import pandas as pd
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


def build_preprocessor_urgent(config: dict[str, Any]) -> ColumnTransformer:
    """
    Build the ColumnTransformer for Model 2 (Urgent classifier).

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

    scale_only = _ScaleOnly(scaler_name=scaler_name, columns=NUMERIC_COLS)

    # Determine known lead_time_bucket categories from the fixed ordering
    # Unknown values at inference will be encoded as -1 (handle_unknown="use_encoded_value")
    lead_time_bucket_known = [LEAD_TIME_BUCKET_ORDER]

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
            "month_cyclical",
            CyclicalEncoder(column=MONTH_COL, period=12),
            [MONTH_COL],
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
            [PROJECT_CATEGORY_COL],
        ),
        (
            "lead_time_bucket_ordinal",
            OrdinalEncoder(
                categories=lead_time_bucket_known,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            [LEAD_TIME_BUCKET_COL],
        ),
        (
            "numeric_scaled",
            scale_only,
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
            f"PipelineBuilderUrgent: text ENABLED "
            f"(vectorizer={vectorizer_name}, lsa={text_use_lsa}, n_components={text_n_components})"
        )
    else:
        logger.debug("PipelineBuilderUrgent: text DISABLED for this trial.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifier_urgent(config: dict[str, Any]):
    """
    Instantiate the classifier for Model 2. Delegates to pipeline_builder for
    shared model types. scale_pos_weight is tuned for 21.36:1 imbalance.

    model_type options: lgbm / xgboost / random_forest / logistic /
                        voting_lgbm_rf / voting_lgbm_lr / voting_all
    """
    # Import shared builder helpers to avoid code duplication
    from src.training.pipeline_builder import build_classifier
    return build_classifier(config)


def build_full_pipeline_urgent(config: dict[str, Any]) -> Pipeline:
    """
    Assemble the full sklearn Pipeline for Model 2:
      preprocessor_urgent → feature_selector → dim_reducer → classifier

    Parameters
    ----------
    config : dict
        Combined hyperparameter dict (preprocessor + selector + reducer + classifier).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from src.training.pipeline_builder import build_feature_selector, build_dim_reducer

    preprocessor = build_preprocessor_urgent(config)
    feature_selector = build_feature_selector(config)
    dim_reducer = build_dim_reducer(config)
    classifier = build_classifier_urgent(config)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector),
        ("dim_reducer", dim_reducer),
        ("classifier", classifier),
    ])
