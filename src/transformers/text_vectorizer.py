"""
TextVectorizerTransformer — sklearn TransformerMixin
Supports BoW (CountVectorizer) or TF-IDF, followed by optional TruncatedSVD (LSA)
for dimensionality reduction.

Designed for Portuguese text (desc_clean column).
LSA is strongly recommended to mitigate memorization of recurring task names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

SUPPORTED_VECTORIZERS = ("tfidf", "bow")


class TextVectorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Vectorizes a text column and optionally reduces with LSA (TruncatedSVD).

    Parameters
    ----------
    column : str
        Name of the text column (e.g. 'desc_clean').
    vectorizer_name : str
        'tfidf' or 'bow' (CountVectorizer).
    max_features : int | None
        Max vocabulary size. None = unlimited.
    ngram_range : tuple
        e.g. (1, 2) for unigrams + bigrams.
    use_lsa : bool
        If True, apply TruncatedSVD after vectorization (reduces memorization risk).
    n_components : int
        Number of LSA components (only used if use_lsa=True).
    min_df : int | float
        Minimum document frequency for vocabulary inclusion.
    sublinear_tf : bool
        Apply sublinear TF scaling in TF-IDF (log(1+tf)). Ignored for BoW.
    """

    def __init__(
        self,
        column: str = "desc_clean",
        vectorizer_name: str = "tfidf",
        max_features: int | None = 500,
        ngram_range: tuple[int, int] = (1, 2),
        use_lsa: bool = True,
        n_components: int = 30,
        min_df: int = 2,
        sublinear_tf: bool = True,
    ) -> None:
        if vectorizer_name not in SUPPORTED_VECTORIZERS:
            raise ValueError(
                f"vectorizer_name must be one of {SUPPORTED_VECTORIZERS}, got '{vectorizer_name}'"
            )
        if use_lsa and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        self.column = column
        self.vectorizer_name = vectorizer_name
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_lsa = use_lsa
        self.n_components = n_components
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

    def _build_vectorizer(self):
        common = dict(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            strip_accents="unicode",
            analyzer="word",
        )
        if self.vectorizer_name == "tfidf":
            return TfidfVectorizer(sublinear_tf=self.sublinear_tf, **common)
        return CountVectorizer(**common)

    def fit(self, X: pd.DataFrame, y=None) -> "TextVectorizerTransformer":
        if self.column not in X.columns:
            raise ValueError(
                f"TextVectorizerTransformer: column '{self.column}' not found. "
                f"Available: {list(X.columns)}"
            )
        texts = X[self.column].fillna("").astype(str).tolist()

        self._vectorizer = self._build_vectorizer()
        matrix = self._vectorizer.fit_transform(texts)

        if self.use_lsa:
            # Cap n_components to vocab size - 1 to avoid SVD errors
            max_comp = min(self.n_components, matrix.shape[1] - 1)
            if max_comp < self.n_components:
                logger.warning(
                    f"TextVectorizerTransformer: vocab size ({matrix.shape[1]}) < "
                    f"n_components ({self.n_components}). Capping to {max_comp}."
                )
            self._svd = TruncatedSVD(n_components=max(1, max_comp), random_state=42)
            self._svd.fit(matrix)
            explained = self._svd.explained_variance_ratio_.sum()
            logger.info(
                f"TextVectorizerTransformer LSA: {max_comp} components explain "
                f"{explained:.1%} of variance."
            )
            self._output_dim = max(1, max_comp)
        else:
            self._svd = None
            self._output_dim = matrix.shape[1]

        self._feature_names_out = [
            f"text_{self.vectorizer_name}_{'lsa' if self.use_lsa else 'raw'}_{i}"
            for i in range(self._output_dim)
        ]
        self._is_fitted = True
        logger.debug(
            f"TextVectorizerTransformer fitted: vectorizer={self.vectorizer_name}, "
            f"use_lsa={self.use_lsa}, output_dim={self._output_dim}"
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        texts = X[self.column].fillna("").astype(str).tolist()
        matrix = self._vectorizer.transform(texts)
        if self._svd is not None:
            return self._svd.transform(matrix)
        return matrix.toarray()

    def get_feature_names_out(self, input_features=None) -> list[str]:
        check_is_fitted(self, "_is_fitted")
        return self._feature_names_out
