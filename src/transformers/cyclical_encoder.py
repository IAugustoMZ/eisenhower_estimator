"""
CyclicalEncoder — sklearn TransformerMixin
Encodes integer periodic features (e.g. hour_created, day_of_week_created)
as sin/cos pairs so the model understands the cyclic nature (23h ≈ 0h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Transforms a single integer column into two columns: sin and cos of
    (2π * value / period).

    Parameters
    ----------
    column : str
        Name of the column to encode (must be present in DataFrame input).
    period : float
        The full cycle length (e.g. 24 for hours, 7 for day-of-week).

    Output columns
    --------------
    <column>_sin, <column>_cos
    """

    def __init__(self, column: str, period: float) -> None:
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self.column = column
        self.period = period

    def fit(self, X: pd.DataFrame, y=None) -> "CyclicalEncoder":
        if self.column not in X.columns:
            raise ValueError(
                f"CyclicalEncoder: column '{self.column}' not found in input. "
                f"Available columns: {list(X.columns)}"
            )
        self._feature_names_out = [f"{self.column}_sin", f"{self.column}_cos"]
        self._is_fitted = True
        logger.debug(f"CyclicalEncoder fitted: column='{self.column}', period={self.period}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        if self.column not in X.columns:
            raise ValueError(
                f"CyclicalEncoder: column '{self.column}' missing at transform time."
            )
        values = X[self.column].to_numpy(dtype=float)
        angle = 2.0 * np.pi * values / self.period
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "_is_fitted")
        return self._feature_names_out
