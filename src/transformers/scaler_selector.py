"""
ScalerSelector — sklearn TransformerMixin
Selects and applies one of three scalers by name:
  - 'standard'  → StandardScaler
  - 'robust'    → RobustScaler  (preferred for skewed features)
  - 'minmax'    → MinMaxScaler
  - 'none'      → passthrough (identity transform)

Designed to be used inside ColumnTransformer so Optuna can tune scaler choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

SUPPORTED_SCALERS = ("standard", "robust", "minmax", "none")


class ScalerSelector(BaseEstimator, TransformerMixin):
    """
    Wraps a configurable scaler so it can be swapped via Optuna trials.

    Parameters
    ----------
    scaler_name : str
        One of 'standard', 'robust', 'minmax', 'none'.
    columns : list[str] | None
        Column names for output feature naming. If None, names are positional.
    """

    def __init__(self, scaler_name: str = "robust", columns: list[str] | None = None) -> None:
        if scaler_name not in SUPPORTED_SCALERS:
            raise ValueError(
                f"scaler_name must be one of {SUPPORTED_SCALERS}, got '{scaler_name}'"
            )
        self.scaler_name = scaler_name
        self.columns = columns

    def _build_scaler(self):
        mapping = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
            "none": None,
        }
        return mapping[self.scaler_name]

    def fit(self, X, y=None) -> "ScalerSelector":
        arr = self._to_array(X)
        self._n_features = arr.shape[1]
        self._scaler = self._build_scaler()
        if self._scaler is not None:
            self._scaler.fit(arr)
        self._is_fitted = True
        logger.debug(f"ScalerSelector fitted: scaler='{self.scaler_name}', n_features={self._n_features}")
        return self

    def transform(self, X, y=None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        arr = self._to_array(X)
        if arr.shape[1] != self._n_features:
            raise ValueError(
                f"ScalerSelector: expected {self._n_features} features, got {arr.shape[1]}"
            )
        if self._scaler is not None:
            return self._scaler.transform(arr)
        return arr

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "_is_fitted")
        if self.columns is not None:
            return [f"{c}_scaled" for c in self.columns]
        if input_features is not None:
            return [f"{f}_scaled" for f in input_features]
        return [f"scaled_{i}" for i in range(self._n_features)]

    @staticmethod
    def _to_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        if isinstance(X, pd.Series):
            return X.to_numpy(dtype=float).reshape(-1, 1)
        return np.asarray(X, dtype=float)
