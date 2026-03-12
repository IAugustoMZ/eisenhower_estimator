"""
ResamplerTransformer
Wraps SMOTE / ADASYN / passthrough so resampling strategy can be tuned by Optuna.

CRITICAL: This is NOT used inside a sklearn Pipeline — it must be applied
manually ONLY to training folds, never to validation/test data.
Use ResamplerTransformer.fit_resample(X_train, y_train) directly.

Supported strategies:
  - 'smote'    → SMOTE (Synthetic Minority Over-sampling Technique)
  - 'adasyn'   → ADASYN (Adaptive Synthetic Sampling)
  - 'none'     → no resampling (passthrough)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

SUPPORTED_STRATEGIES = ("smote", "adasyn", "none")


class ResamplerTransformer:
    """
    Resampling wrapper. Not a sklearn transformer (no fit/transform API)
    to make it explicit that it must NOT be placed inside a Pipeline.

    Parameters
    ----------
    strategy : str
        One of 'smote', 'adasyn', 'none'.
    random_state : int
        Random seed for reproducibility.
    k_neighbors : int
        Number of nearest neighbors for SMOTE/ADASYN. Default 5.
        Will be auto-reduced if minority class has fewer samples.
    """

    def __init__(
        self,
        strategy: str = "smote",
        random_state: int = 42,
        k_neighbors: int = 5,
    ) -> None:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {SUPPORTED_STRATEGIES}, got '{strategy}'"
            )
        self.strategy = strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply resampling to training data only.

        Returns
        -------
        X_resampled, y_resampled
        """
        if self.strategy == "none":
            logger.debug("ResamplerTransformer: strategy='none', skipping resampling.")
            return X, y

        classes, counts = np.unique(y, return_counts=True)
        minority_count = int(counts.min())
        logger.info(
            f"ResamplerTransformer: strategy='{self.strategy}', "
            f"class distribution before = {dict(zip(classes.tolist(), counts.tolist()))}"
        )

        # Auto-reduce k_neighbors if minority class is too small
        effective_k = min(self.k_neighbors, minority_count - 1)
        if effective_k < 1:
            logger.warning(
                f"ResamplerTransformer: minority class has only {minority_count} samples. "
                f"Cannot apply {self.strategy}. Falling back to passthrough."
            )
            return X, y

        if effective_k < self.k_neighbors:
            logger.warning(
                f"ResamplerTransformer: k_neighbors reduced from {self.k_neighbors} "
                f"to {effective_k} due to small minority class ({minority_count} samples)."
            )

        try:
            if self.strategy == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=self.random_state, k_neighbors=effective_k)
            else:  # adasyn
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(random_state=self.random_state, n_neighbors=effective_k)

            X_res, y_res = sampler.fit_resample(X, y)
            new_classes, new_counts = np.unique(y_res, return_counts=True)
            logger.info(
                f"ResamplerTransformer: class distribution after = "
                f"{dict(zip(new_classes.tolist(), new_counts.tolist()))}"
            )
            return X_res, y_res

        except Exception as exc:
            logger.error(
                f"ResamplerTransformer: {self.strategy} failed with error: {exc}. "
                f"Falling back to passthrough."
            )
            return X, y
