"""
FeatureSelectorTransformer — sklearn TransformerMixin
Applies supervised feature selection after the ColumnTransformer, on the full
assembled feature matrix. Operates on numpy arrays (post-preprocessing output).

Supported strategies:
  - 'selectkbest_f'  → SelectKBest with f_classif (ANOVA F-statistic)
  - 'selectkbest_mi' → SelectKBest with mutual_info_classif
  - 'none'           → passthrough (no selection)

Recommended: 'selectkbest_f' — fast, supervised, well-suited for mixed
numeric + encoded features. mutual_info is slower but captures non-linear deps.

Placement: AFTER ColumnTransformer, BEFORE DimensionalityReducerTransformer.

IMPORTANT: This transformer is supervised — it requires y during fit.
It is correctly placed inside a sklearn Pipeline so Pipeline.fit(X, y)
propagates y to each step's fit call.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.utils.validation import check_is_fitted

SUPPORTED_STRATEGIES = ("selectkbest_f", "selectkbest_mi", "none")


class FeatureSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Configurable supervised feature selector.

    Parameters
    ----------
    strategy : str
        One of 'selectkbest_f', 'selectkbest_mi', 'none'.
    k : int | str
        Number of top features to keep. Use 'all' to keep all (equivalent to 'none').
        Optuna should tune this between 5 and n_features.
        Will be auto-capped to n_features if k > n_features.

    Attributes
    ----------
    n_features_in_ : int
        Total features before selection.
    n_features_selected_ : int
        Features kept after selection.
    selected_mask_ : ndarray of bool
        Boolean mask of selected feature indices.
    """

    def __init__(self, strategy: str = "selectkbest_f", k: int | str = "all") -> None:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"FeatureSelectorTransformer: strategy must be one of "
                f"{SUPPORTED_STRATEGIES}, got '{strategy}'"
            )
        self.strategy = strategy
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelectorTransformer":
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        if self.strategy == "none" or self.k == "all":
            self._selector = None
            self.n_features_selected_ = n_features
            self.selected_mask_ = np.ones(n_features, dtype=bool)
            self._is_fitted = True
            logger.debug(
                f"FeatureSelectorTransformer: strategy='none' — keeping all {n_features} features."
            )
            return self

        # Cap k to available features
        effective_k = self.k
        if isinstance(effective_k, int) and effective_k > n_features:
            logger.warning(
                f"FeatureSelectorTransformer: k={effective_k} > n_features={n_features}. "
                f"Capping k to {n_features}."
            )
            effective_k = n_features

        if self.strategy == "selectkbest_f":
            score_func = f_classif
        else:
            score_func = mutual_info_classif

        self._selector = SelectKBest(score_func=score_func, k=effective_k)
        self._selector.fit(X_arr, y)
        self.selected_mask_ = self._selector.get_support()
        self.n_features_selected_ = int(self.selected_mask_.sum())

        logger.info(
            f"FeatureSelectorTransformer: strategy='{self.strategy}', "
            f"k={effective_k} -> selected {self.n_features_selected_}/{n_features} features."
        )
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        X_arr = self._to_array(X)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"FeatureSelectorTransformer: expected {self.n_features_in_} features "
                f"at transform time, got {X_arr.shape[1]}."
            )

        if self._selector is None:
            return X_arr

        return self._selector.transform(X_arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "_is_fitted")
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(self.n_features_in_)]
        input_features = list(input_features)
        if self._selector is None:
            return input_features
        return [f for f, keep in zip(input_features, self.selected_mask_) if keep]

    @staticmethod
    def _to_array(X) -> np.ndarray:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)
