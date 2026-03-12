"""
DimensionalityReducerTransformer — sklearn TransformerMixin
Applies dimensionality reduction after FeatureSelectorTransformer,
on the full assembled + selected feature matrix.

Supported methods:
  - 'pca'  → PCA (unsupervised, n_components tunable)
  - 'lda'  → LinearDiscriminantAnalysis (supervised, binary → always 1 component)
  - 'none' → passthrough

RECOMMENDATIONS (documented here for Optuna config reference):
  - 'pca'  — recommended when text features are included (LSA dims + cyclical
              features can be collinear; PCA decorrelates them).
              n_components: tune 5–40.
  - 'lda'  — aggressive reduction (binary → 1D). Useful as a regularization
              experiment, but loses most signal. Use sparingly.
  - 'none' — recommended default when feature count is already low
              (without text: ~9 features, no reduction needed).

Placement: AFTER FeatureSelectorTransformer, BEFORE classifier.

IMPORTANT: LDA is supervised — requires y during fit.
Pipeline.fit(X, y) correctly propagates y to each step.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted

SUPPORTED_METHODS = ("pca", "lda", "none")


class DimensionalityReducerTransformer(BaseEstimator, TransformerMixin):
    """
    Configurable dimensionality reducer.

    Parameters
    ----------
    method : str
        One of 'pca', 'lda', 'none'.
    n_components : int | None
        Components to keep.
        - PCA: tunable (auto-capped to min(n_samples, n_features) - 1).
        - LDA: ignored for binary classification — always 1.
        - none: ignored.
    random_state : int
        Random seed (used by PCA).

    Attributes
    ----------
    n_features_in_ : int
    n_components_out_ : int
        Actual number of output dimensions after reduction.
    """

    def __init__(
        self,
        method: str = "none",
        n_components: int = 20,
        random_state: int = 42,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"DimensionalityReducerTransformer: method must be one of "
                f"{SUPPORTED_METHODS}, got '{method}'"
            )
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "DimensionalityReducerTransformer":
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.method == "none":
            self._reducer = None
            self.n_components_out_ = n_features
            self._is_fitted = True
            logger.debug(
                f"DimensionalityReducerTransformer: method='none' — passthrough "
                f"({n_features} features)."
            )
            return self

        if self.method == "pca":
            # Cap n_components to a safe maximum
            max_components = min(n_samples - 1, n_features)
            effective_n = min(self.n_components, max_components)
            if effective_n < self.n_components:
                logger.warning(
                    f"DimensionalityReducerTransformer: PCA n_components capped "
                    f"from {self.n_components} to {effective_n} "
                    f"(n_samples={n_samples}, n_features={n_features})."
                )
            self._reducer = PCA(n_components=max(1, effective_n), random_state=self.random_state)
            self._reducer.fit(X_arr)
            explained = self._reducer.explained_variance_ratio_.sum()
            self.n_components_out_ = self._reducer.n_components_
            logger.info(
                f"DimensionalityReducerTransformer: PCA({self.n_components_out_}) "
                f"explains {explained:.1%} variance."
            )

        elif self.method == "lda":
            if y is None:
                raise ValueError(
                    "DimensionalityReducerTransformer: LDA requires y during fit. "
                    "Ensure this transformer is inside a sklearn Pipeline so y is propagated."
                )
            n_classes = len(np.unique(y))
            # LDA max components = min(n_features, n_classes - 1)
            lda_max = min(n_features, n_classes - 1)
            if lda_max < 1:
                logger.warning(
                    f"DimensionalityReducerTransformer: LDA n_components would be 0 "
                    f"(n_classes={n_classes}). Falling back to passthrough."
                )
                self._reducer = None
                self.n_components_out_ = n_features
                self._is_fitted = True
                return self

            self._reducer = LinearDiscriminantAnalysis(n_components=lda_max)
            self._reducer.fit(X_arr, y)
            self.n_components_out_ = lda_max
            logger.info(
                f"DimensionalityReducerTransformer: LDA -> {lda_max} component(s) "
                f"(binary classification always produces 1 component)."
            )

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        X_arr = self._to_array(X)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"DimensionalityReducerTransformer: expected {self.n_features_in_} "
                f"features at transform time, got {X_arr.shape[1]}."
            )

        if self._reducer is None:
            return X_arr

        return self._reducer.transform(X_arr)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "_is_fitted")
        return [f"{self.method}_component_{i}" for i in range(self.n_components_out_)]

    @staticmethod
    def _to_array(X) -> np.ndarray:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)
