"""
PipelineBuilder — Time Spent Regressor (Model 3)
Constructs a full sklearn Pipeline from a config dict:

  ColumnTransformer (per-feature-group encoding)
      ↓
  FeatureSelectorTransformer   ← SelectKBest (f_regression / mutual_info_regression) or none
      ↓
  DimensionalityReducerTransformer  ← PCA / none (LDA excluded — classification only)
      ↓
  Regressor (LGBMRegressor / XGBRegressor / RFRegressor / Ridge / voting variants)

Feature groups in ColumnTransformer:
  - Cyclical temporal  : day_of_week
  - Target-encoded cat : project_code (target = log_duration, continuous)
  - Ordinal-encoded cat: task_type (ordered by median duration)
  - Binary passthrough : has_number, is_repeated_task
  - Numeric (log+scale): task_freq, desc_char_len
  - Numeric (scale only): task_cv (imputed for unseen tasks by TaskCVImputer)
  - Text               : desc_clean (TF-IDF or BoW + optional LSA)

Iteration-2 features (from error analysis):
  - Interaction cat    : task_type_x_repeated (ordinal, 12 categories)
  - Binary passthrough : is_long_project, desc_has_time_ref
  - Numeric (log+scale): log_task_freq
  - Numeric (scale only): task_median_duration (imputed by TaskCVImputer per fold)

Features intentionally EXCLUDED (per EDA §11.4):
  - project_category   — p=0.31, subsumed by project_code
  - is_weekend         — p=0.31, not significant
  - month              — p=0.070, not significant
  - quarter, week_of_year, desc_word_count — not significant
  - project_name       — redundant with project_code
  - date               — temporal leakage risk
  - duration_bucket    — derived from target
  - log_duration       — is the target itself
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from src.transformers import (
    CyclicalEncoder,
    DimensionalityReducerTransformer,
    FeatureSelectorTransformer,
    SafeTargetEncoder,
    ScalerSelector,
    TextVectorizerTransformer,
)

# ── Feature column constants ──────────────────────────────────────────────────
DOW_COL = "day_of_week"
PROJECT_CODE_COL = "project_code"
TASK_TYPE_COL = "task_type"
HAS_NUMBER_COL = "has_number"
IS_REPEATED_COL = "is_repeated_task"
TASK_FREQ_COL = "task_freq"
TASK_CV_COL = "task_cv"
DESC_CHAR_LEN_COL = "desc_char_len"
TEXT_COL = "desc_clean"
# Iteration-2 feature columns (from error analysis)
TASK_TYPE_X_REPEATED_COL = "task_type_x_repeated"
IS_LONG_PROJECT_COL = "is_long_project"
LOG_TASK_FREQ_COL = "log_task_freq"
DESC_HAS_TIME_REF_COL = "desc_has_time_ref"
TASK_MEDIAN_DUR_COL = "task_median_duration"

# Ordered by median duration (ascending) — EDA §6.3
TASK_TYPE_ORDER = [
    "exercise",        # median=5, mean=5.5
    "admin_update",    # median=5, mean=7.8
    "communication",   # median=5, mean=14.1
    "devotional",      # median=10, mean=10.7
    "study_review",    # median=10, mean=20.2
    "other",           # median=10, mean=27.3
    "development",     # median=15, mean=49.6
]

# Ordered by median duration: task_type × is_repeated interaction
# 12 categories = 7 task_types × {rep, new} minus missing combos
TASK_TYPE_X_REPEATED_ORDER = [
    "exercise_rep",
    "exercise_new",
    "admin_update_rep",
    "devotional_rep",
    "communication_rep",
    "admin_update_new",
    "study_review_rep",
    "devotional_new",
    "other_rep",
    "communication_new",
    "study_review_new",
    "other_new",
    "development_rep",
    "development_new",
]

LOG_SCALE_COLS = [TASK_FREQ_COL, DESC_CHAR_LEN_COL]
BINARY_PASSTHROUGH_COLS = [HAS_NUMBER_COL, IS_REPEATED_COL, IS_LONG_PROJECT_COL, DESC_HAS_TIME_REF_COL]


# ── Custom Transformers ──────────────────────────────────────────────────────

class _LogThenScale(BaseEstimator, TransformerMixin):
    """
    Applies log1p (after clipping negatives to 0) then delegates to ScalerSelector.
    Used for non-negative numeric columns (task_freq, desc_char_len).
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
        arr = X.to_numpy(dtype=float) if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X, dtype=float)
        arr = np.clip(arr, 0, None)
        return np.log1p(arr)


class _ScaleOnly(BaseEstimator, TransformerMixin):
    """
    Scales without log-transform. Used for task_cv which is already
    a ratio [0, ~1] and doesn't need log transformation.
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
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy(dtype=float).reshape(-1, 1) if isinstance(X, pd.Series) else X.to_numpy(dtype=float)
        arr = np.asarray(X, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr


class TaskCVImputer(BaseEstimator, TransformerMixin):
    """
    Imputes leakage-prone per-task features for unseen tasks.

    Handles two columns (both require per-fold computation to prevent leakage):
      1. task_cv: coefficient of variation per task_description
      2. task_median_duration: median log_duration per task_description

    At training time, learns mappings:
      - task_description → task_cv / task_median_duration
      - task_type → mean_cv / mean_median_duration  (fallback)
      - global mean (ultimate fallback)

    At inference time, imputes using the 3-tier fallback:
      task_description → task_type → global mean

    IMPORTANT: Applied BEFORE the ColumnTransformer as a preprocessing step
    (in the objective / trainer), not inside the pipeline.
    """

    def __init__(self) -> None:
        pass

    def fit(self, df: pd.DataFrame, y=None) -> "TaskCVImputer":
        """
        Learn task_cv and task_median_duration mappings from training data.

        Parameters
        ----------
        df : DataFrame with columns: task_cv, task_type, desc_clean,
             and optionally task_median_duration + log_duration (for target).
        """
        required_cols = {TASK_CV_COL, TASK_TYPE_COL, TEXT_COL}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"TaskCVImputer: missing columns {missing} in input DataFrame.")

        # ── task_cv mappings ──
        self._global_mean_cv = float(df[TASK_CV_COL].mean())
        self._task_type_cv_map: dict[str, float] = (
            df.groupby(TASK_TYPE_COL)[TASK_CV_COL].mean().to_dict()
        )
        self._task_desc_cv_map: dict[str, float] = (
            df.groupby(TEXT_COL)[TASK_CV_COL].first().to_dict()
        )

        # ── task_median_duration mappings (leakage-safe: fit on train fold only) ──
        self._has_median_dur = TASK_MEDIAN_DUR_COL in df.columns
        if self._has_median_dur:
            # Recompute from training fold's log_duration to prevent leakage
            if "log_duration" in df.columns:
                # Use actual target values from train fold
                self._task_desc_median_map: dict[str, float] = (
                    df.groupby(TEXT_COL)["log_duration"].median().to_dict()
                )
                self._task_type_median_map: dict[str, float] = (
                    df.groupby(TASK_TYPE_COL)["log_duration"].median().to_dict()
                )
                self._global_mean_median = float(df["log_duration"].median())
            else:
                # Fallback: use the pre-computed column values
                self._task_desc_median_map = (
                    df.groupby(TEXT_COL)[TASK_MEDIAN_DUR_COL].first().to_dict()
                )
                self._task_type_median_map = (
                    df.groupby(TASK_TYPE_COL)[TASK_MEDIAN_DUR_COL].mean().to_dict()
                )
                self._global_mean_median = float(df[TASK_MEDIAN_DUR_COL].mean())

        self._is_fitted = True
        logger.debug(
            f"TaskCVImputer fitted: {len(self._task_desc_cv_map)} known tasks, "
            f"{len(self._task_type_cv_map)} task types, "
            f"global_mean_cv={self._global_mean_cv:.4f}"
            + (f", median_dur_mapped={self._has_median_dur}" if self._has_median_dur else "")
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute task_cv and task_median_duration for unseen tasks.

        Returns a copy of the DataFrame with imputed values.
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "_is_fitted")

        result = df.copy()
        for idx in result.index:
            desc = result.at[idx, TEXT_COL]
            task_type = result.at[idx, TASK_TYPE_COL]

            # ── Impute task_cv ──
            if desc in self._task_desc_cv_map:
                result.at[idx, TASK_CV_COL] = self._task_desc_cv_map[desc]
            elif task_type in self._task_type_cv_map:
                result.at[idx, TASK_CV_COL] = self._task_type_cv_map[task_type]
            else:
                result.at[idx, TASK_CV_COL] = self._global_mean_cv

            # ── Impute task_median_duration ──
            if self._has_median_dur and TASK_MEDIAN_DUR_COL in result.columns:
                if desc in self._task_desc_median_map:
                    result.at[idx, TASK_MEDIAN_DUR_COL] = self._task_desc_median_map[desc]
                elif task_type in self._task_type_median_map:
                    result.at[idx, TASK_MEDIAN_DUR_COL] = self._task_type_median_map[task_type]
                else:
                    result.at[idx, TASK_MEDIAN_DUR_COL] = self._global_mean_median

        return result


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_preprocessor_time_spent(config: dict[str, Any]) -> ColumnTransformer:
    """
    Build the ColumnTransformer for the time spent regression model.

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
    include_text = config.get("include_text", True)
    target_enc_smoothing = config.get("target_enc_smoothing", 1.0)

    transformers = [
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
            "task_type_ordinal",
            OrdinalEncoder(
                categories=[TASK_TYPE_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            [TASK_TYPE_COL],
        ),
        (
            "binary_passthrough",
            FunctionTransformer(func=None, feature_names_out="one-to-one"),
            BINARY_PASSTHROUGH_COLS,
        ),
        (
            "numeric_log_scaled",
            _LogThenScale(scaler_name=scaler_name, columns=LOG_SCALE_COLS),
            LOG_SCALE_COLS,
        ),
        (
            "task_cv_scaled",
            _ScaleOnly(scaler_name=scaler_name, columns=[TASK_CV_COL]),
            [TASK_CV_COL],
        ),
        # Iteration-2: task_type × is_repeated interaction (ordinal encoded)
        (
            "task_type_x_repeated_ordinal",
            OrdinalEncoder(
                categories=[TASK_TYPE_X_REPEATED_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            [TASK_TYPE_X_REPEATED_COL],
        ),
        # Iteration-2: log_task_freq (already log1p, just scale)
        (
            "log_task_freq_scaled",
            _ScaleOnly(scaler_name=scaler_name, columns=[LOG_TASK_FREQ_COL]),
            [LOG_TASK_FREQ_COL],
        ),
        # Iteration-2: task_median_duration (imputed per fold by TaskCVImputer)
        (
            "task_median_dur_scaled",
            _ScaleOnly(scaler_name=scaler_name, columns=[TASK_MEDIAN_DUR_COL]),
            [TASK_MEDIAN_DUR_COL],
        ),
    ]

    if include_text:
        vectorizer_name = config.get("vectorizer_name", "tfidf")
        transformers.append((
            "text_features",
            TextVectorizerTransformer(
                column=TEXT_COL,
                vectorizer_name=vectorizer_name,
                max_features=config.get("text_max_features", 500),
                ngram_range=(1, config.get("text_ngram_max", 2)),
                use_lsa=config.get("text_use_lsa", True),
                n_components=config.get("text_n_components", 30),
                min_df=config.get("text_min_df", 2),
            ),
            [TEXT_COL],
        ))
        logger.debug(
            f"PipelineBuilder (time_spent): text ENABLED "
            f"(vectorizer={vectorizer_name}, lsa={config.get('text_use_lsa', True)})"
        )
    else:
        logger.debug("PipelineBuilder (time_spent): text DISABLED for this trial.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_selector_regression(config: dict[str, Any]) -> FeatureSelectorTransformer:
    """
    Build the post-preprocessor feature selector for regression.

    Uses f_regression / mutual_info_regression instead of classification variants.
    """
    strategy = config.get("selector_strategy", "selectkbest_f_reg")

    # Guard: if a classification strategy was accidentally passed, remap to regression
    _REMAP = {
        "selectkbest_f": "selectkbest_f_reg",
        "selectkbest_mi": "selectkbest_mi_reg",
    }
    if strategy in _REMAP:
        logger.warning(
            f"build_feature_selector_regression: classification strategy '{strategy}' "
            f"remapped to '{_REMAP[strategy]}' for regression."
        )
        strategy = _REMAP[strategy]

    return FeatureSelectorTransformer(
        strategy=strategy,
        k=config.get("selector_k", "all"),
    )


def build_dim_reducer_time_spent(config: dict[str, Any]) -> DimensionalityReducerTransformer:
    """
    Build the post-selector dimensionality reducer for regression.

    LDA is excluded (classification only). If 'lda' is passed, falls back to 'none'.
    """
    method = config.get("reducer_method", "none")

    if method == "lda":
        logger.warning(
            "build_dim_reducer_time_spent: LDA is classification-only. "
            "Falling back to 'none' for regression."
        )
        method = "none"

    return DimensionalityReducerTransformer(
        method=method,
        n_components=config.get("reducer_n_components", 20),
        random_state=config.get("random_state", 42),
    )


def build_regressor(config: dict[str, Any]):
    """
    Instantiate the regressor from config.

    model_type options: lgbm / xgboost / random_forest / ridge /
                        voting_lgbm_rf / voting_lgbm_ridge / voting_all
    """
    model_type = config.get("model_type", "lgbm")
    random_state = config.get("random_state", 42)
    logger.info(f"PipelineBuilder (time_spent): regressor='{model_type}'")

    if model_type == "lgbm":
        return _build_lgbm_regressor(config, random_state)
    if model_type == "xgboost":
        return _build_xgb_regressor(config, random_state)
    if model_type == "random_forest":
        return _build_rf_regressor(config, random_state)
    if model_type == "ridge":
        return _build_ridge(config, random_state)
    if model_type == "voting_lgbm_rf":
        return VotingRegressor(
            estimators=[
                ("lgbm", _build_lgbm_regressor(config, random_state)),
                ("rf", _build_rf_regressor(config, random_state)),
            ],
        )
    if model_type == "voting_lgbm_ridge":
        return VotingRegressor(
            estimators=[
                ("lgbm", _build_lgbm_regressor(config, random_state)),
                ("ridge", _build_ridge(config, random_state)),
            ],
        )
    if model_type == "voting_all":
        return VotingRegressor(
            estimators=[
                ("lgbm", _build_lgbm_regressor(config, random_state)),
                ("xgb", _build_xgb_regressor(config, random_state)),
                ("rf", _build_rf_regressor(config, random_state)),
                ("ridge", _build_ridge(config, random_state)),
            ],
        )

    raise ValueError(
        f"build_regressor: unknown model_type='{model_type}'. "
        f"Supported: lgbm, xgboost, random_forest, ridge, "
        f"voting_lgbm_rf, voting_lgbm_ridge, voting_all"
    )


def _build_lgbm_regressor(config: dict, random_state: int):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is required. Run: pip install lightgbm") from exc

    return LGBMRegressor(
        n_estimators=config.get("lgbm_n_estimators", 300),
        max_depth=config.get("lgbm_max_depth", -1),
        learning_rate=config.get("lgbm_learning_rate", 0.05),
        num_leaves=config.get("lgbm_num_leaves", 31),
        min_child_samples=config.get("lgbm_min_child_samples", 20),
        subsample=config.get("lgbm_subsample", 0.8),
        colsample_bytree=config.get("lgbm_colsample_bytree", 0.8),
        reg_alpha=config.get("lgbm_reg_alpha", 0.0),
        reg_lambda=config.get("lgbm_reg_lambda", 1.0),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgb_regressor(config: dict, random_state: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required. Run: pip install xgboost") from exc

    return XGBRegressor(
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
    )


def _build_rf_regressor(config: dict, random_state: int):
    return RandomForestRegressor(
        n_estimators=config.get("rf_n_estimators", 300),
        max_depth=config.get("rf_max_depth", None),
        min_samples_split=config.get("rf_min_samples_split", 2),
        min_samples_leaf=config.get("rf_min_samples_leaf", 1),
        max_features=config.get("rf_max_features", "sqrt"),
        random_state=random_state,
        n_jobs=-1,
    )


def _build_ridge(config: dict, random_state: int):
    return Ridge(
        alpha=config.get("ridge_alpha", 1.0),
        random_state=random_state,
    )


def build_full_pipeline_time_spent(config: dict[str, Any]) -> Pipeline:
    """
    Assemble the full sklearn Pipeline for time spent regression:
      preprocessor → feature_selector → dim_reducer → regressor

    Parameters
    ----------
    config : dict
        Combined hyperparameter dict.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = build_preprocessor_time_spent(config)
    feature_selector = build_feature_selector_regression(config)
    dim_reducer = build_dim_reducer_time_spent(config)
    regressor = build_regressor(config)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector),
        ("dim_reducer", dim_reducer),
        ("regressor", regressor),
    ])
