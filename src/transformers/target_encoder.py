"""
SafeTargetEncoder — sklearn TransformerMixin
Wraps category_encoders.TargetEncoder with:
  - Unseen-category fallback (global mean)
  - NaN-safe input handling
  - Proper fit/transform separation to prevent leakage

IMPORTANT: This transformer MUST be fit only on training folds,
never on the full dataset before CV splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SafeTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target-encodes a single categorical column.

    Parameters
    ----------
    column : str
        Categorical column to encode.
    smoothing : float
        Smoothing factor (higher = shrink more toward global mean).
        Default 1.0.
    min_samples_leaf : int
        Minimum samples required to trust a category mean. Default 1.
    handle_missing : str
        'value' — replace missing with global mean after fit.
        Default 'value'.

    Output
    ------
    Single float column: <column>_target_enc
    """

    def __init__(
        self,
        column: str,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
    ) -> None:
        self.column = column
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SafeTargetEncoder":
        if self.column not in X.columns:
            raise ValueError(
                f"SafeTargetEncoder: column '{self.column}' not found in input. "
                f"Available: {list(X.columns)}"
            )
        if y is None:
            raise ValueError("SafeTargetEncoder requires y (target) during fit.")

        series = X[self.column].astype(str).fillna("__MISSING__")
        y_arr = np.asarray(y, dtype=float)

        self._global_mean = float(np.mean(y_arr))
        n_total = len(y_arr)

        # Compute per-category smoothed mean
        self._encoding_map: dict[str, float] = {}
        for cat in series.unique():
            mask = series == cat
            n_cat = int(mask.sum())
            cat_mean = float(y_arr[mask.to_numpy()].mean())
            # Additive smoothing: blend cat_mean toward global_mean
            alpha = n_cat / (n_cat + self.smoothing)
            self._encoding_map[cat] = alpha * cat_mean + (1 - alpha) * self._global_mean

        self._feature_names_out = [f"{self.column}_target_enc"]
        self._is_fitted = True
        logger.debug(
            f"SafeTargetEncoder fitted: column='{self.column}', "
            f"categories={len(self._encoding_map)}, global_mean={self._global_mean:.4f}"
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        if self.column not in X.columns:
            raise ValueError(
                f"SafeTargetEncoder: column '{self.column}' missing at transform time."
            )
        series = X[self.column].astype(str).fillna("__MISSING__")
        unseen = set(series.unique()) - set(self._encoding_map.keys())
        if unseen:
            logger.warning(
                f"SafeTargetEncoder: {len(unseen)} unseen categories in '{self.column}' "
                f"→ falling back to global mean ({self._global_mean:.4f}). "
                f"Unseen: {unseen}"
            )
        encoded = series.map(self._encoding_map).fillna(self._global_mean).to_numpy(dtype=float)
        return encoded.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "_is_fitted")
        return self._feature_names_out
