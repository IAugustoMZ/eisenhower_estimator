"""
HybridClassifier — rule engine as hard overrides, ML model for undecided cases.

Architecture
------------
  Input row
     │
     ▼
  RuleBasedClassifier.predict_with_trace()
     │
     ├── Rule fired  →  use rule label directly (high-confidence, explainable)
     │
     └── Abstained   →  delegate to ml_model.predict() / predict_proba()
                        (handles the ambiguous middle ground)

This design guarantees:
  1. Rules always win when they fire — business logic is never overridden by ML.
  2. ML handles the hard cases where rules deliberately abstain.
  3. Both modes are fully auditable via predict_with_trace().

The HybridClassifier is sklearn-compatible (implements predict / predict_proba)
so it plugs directly into the ModelEvaluator without special-casing.

Parameters
----------
rule_classifier : RuleBasedClassifier
    Fitted rule engine. Rules fire before the ML model is consulted.
ml_model : sklearn-compatible estimator
    Fitted ML pipeline (e.g. the best Optuna pipeline from ModelTrainer).
    Must implement predict() and optionally predict_proba().
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.rule_based import RuleBasedClassifier, ABSTAIN


class HybridClassifier:
    """
    Combines a RuleBasedClassifier with any sklearn-compatible ML model.

    The rule engine acts as a hard gate: rows it decides are never passed
    to ML. Only abstained rows reach the ML model.

    Parameters
    ----------
    rule_classifier : RuleBasedClassifier
    ml_model        : sklearn estimator with predict() and predict_proba()
    """

    def __init__(
        self,
        rule_classifier: RuleBasedClassifier,
        ml_model,
    ) -> None:
        if not isinstance(rule_classifier, RuleBasedClassifier):
            raise TypeError(
                "HybridClassifier: rule_classifier must be a RuleBasedClassifier instance."
            )
        if not hasattr(ml_model, "predict"):
            raise TypeError(
                "HybridClassifier: ml_model must implement predict()."
            )
        self.rule_classifier = rule_classifier
        self.ml_model = ml_model

    # ── sklearn-compatible API ────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return final predictions, combining rules and ML.

        For rows where a rule fired: use rule label.
        For rows where no rule fired (abstained): use ml_model.predict().
        """
        return self._combine(X)[0]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return soft probabilities of shape (n, 2).

        Rule-decided rows: probability 1.0 for the chosen class.
        ML-decided rows: probability from ml_model.predict_proba().
        Falls back to predict() if ml_model has no predict_proba.
        """
        _, proba = self._combine(X)
        return proba

    def predict_with_trace(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        """
        Return (predictions, trace) where trace[i] is one of:
          - rule name that fired (e.g. "early_morning_work_task")
          - "ml_model" for rows decided by the ML model
        """
        preds, proba = self._combine(X)
        _, rule_traces = self.rule_classifier.predict_with_trace(X)
        traces = [
            t if t != "default" else "ml_model"
            for t in rule_traces
        ]
        return preds, traces

    def coverage_report(self, X: pd.DataFrame) -> dict:
        """Return rule coverage stats + the fraction delegated to ML."""
        coverage = self.rule_classifier.coverage(X)
        ml_ratio = 1.0 - coverage["fired_ratio"]
        coverage["ml_delegated_ratio"] = ml_ratio
        logger.info(
            f"HybridClassifier coverage: rules fired on {coverage['fired_ratio']:.1%} "
            f"of rows; {ml_ratio:.1%} delegated to ML."
        )
        return coverage

    # ── Private ───────────────────────────────────────────────────────────────

    def _combine(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Core combination logic. Returns (predictions, probabilities).

        Indexes are preserved so rule and ML decisions align with input rows.
        """
        n = len(X)
        predictions = np.full(n, -1, dtype=int)
        probabilities = np.full((n, 2), -1.0, dtype=float)

        # Step 1: apply rules, get per-row labels and traces
        rule_labels, rule_traces = self.rule_classifier._apply_rules(X)

        # Identify which rows were decided by rules vs need ML
        rule_decided_idx = [i for i, lbl in enumerate(rule_labels) if lbl is not ABSTAIN]
        ml_needed_idx = [i for i, lbl in enumerate(rule_labels) if lbl is ABSTAIN]

        # Step 2: fill rule-decided rows
        for i in rule_decided_idx:
            lbl = rule_labels[i]
            predictions[i] = lbl
            if lbl == 1:
                probabilities[i] = [0.0, 1.0]
            else:
                probabilities[i] = [1.0, 0.0]

        # Step 3: delegate undecided rows to ML model
        if ml_needed_idx:
            X_ml = X.iloc[ml_needed_idx].reset_index(drop=True)
            try:
                ml_preds = self.ml_model.predict(X_ml)
                predictions[ml_needed_idx] = ml_preds
            except Exception as exc:
                logger.error(
                    f"HybridClassifier: ML model predict() failed: {exc}. "
                    f"Falling back to majority class (1=important)."
                )
                predictions[ml_needed_idx] = 1

            try:
                ml_proba = self.ml_model.predict_proba(X_ml)
                probabilities[ml_needed_idx] = ml_proba
            except (AttributeError, NotImplementedError, Exception) as exc:
                logger.warning(
                    f"HybridClassifier: ML model predict_proba() unavailable: {exc}. "
                    f"Using hard probabilities from predictions."
                )
                for rel_i, abs_i in enumerate(ml_needed_idx):
                    p = int(predictions[abs_i])
                    probabilities[abs_i] = [1.0 - p, float(p)]

        if ml_needed_idx:
            logger.debug(
                f"HybridClassifier: {len(rule_decided_idx)} rule-decided, "
                f"{len(ml_needed_idx)} delegated to ML."
            )

        return predictions, probabilities
