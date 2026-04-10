"""
ModelEvaluator — unified evaluation harness producing comparable metrics
across pure-rule, pure-ML, and hybrid classifiers on the same holdout set.

All three approaches are evaluated on an identical test split using the same
metrics so results are directly comparable. Metrics are logged to MLflow as
separate child runs under the active parent run (if one is active), or as
standalone runs otherwise.

Comparable metric set
---------------------
  f1_macro        — primary optimization target (handles class imbalance)
  f1_0            — F1 for "not important" class (minority-class recall proxy)
  f1_1            — F1 for "important" class
  roc_auc         — threshold-independent discrimination
  precision_macro — precision averaged over both classes
  recall_macro    — recall averaged over both classes
  coverage        — fraction of rows decided by rules (hybrid/rule only)

Usage
-----
    evaluator = ModelEvaluator(
        rule_clf=RuleBasedClassifier(),
        ml_clf=trained_pipeline,
        hybrid_clf=HybridClassifier(rule_clf, ml_clf),
    )
    report = evaluator.evaluate(X_test, y_test, log_to_mlflow=True)
    evaluator.print_comparison_table(report)
"""
from __future__ import annotations

import json
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class ModelEvaluator:
    """
    Evaluates multiple classifiers on the same holdout set and produces
    a side-by-side comparison report with identical metrics for each approach.

    Parameters
    ----------
    rule_clf    : RuleBasedClassifier — pure rule engine baseline
    ml_clf      : sklearn-compatible pipeline — pure ML baseline
    hybrid_clf  : HybridClassifier — rule gates + ML for hard cases
    """

    # Canonical model keys — used as keys in the report dict and MLflow run names
    RULE_KEY = "rule_based"
    ML_KEY = "ml_model"
    HYBRID_KEY = "hybrid"

    def __init__(
        self,
        rule_clf,
        ml_clf,
        hybrid_clf,
    ) -> None:
        self._classifiers = {
            self.RULE_KEY: rule_clf,
            self.ML_KEY: ml_clf,
            self.HYBRID_KEY: hybrid_clf,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        log_to_mlflow: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Run all three classifiers on (X_test, y_test) and return a report dict.

        Parameters
        ----------
        X_test         : Feature DataFrame (same split used for ML training)
        y_test         : Ground-truth labels (numpy array)
        log_to_mlflow  : If True, log per-classifier metrics as MLflow child runs

        Returns
        -------
        report : dict mapping classifier key -> metrics dict
        """
        report: dict[str, dict[str, Any]] = {}

        for key, clf in self._classifiers.items():
            logger.info(f"ModelEvaluator: evaluating '{key}'...")
            try:
                metrics = self._compute_metrics(clf, X_test, y_test, key)
                report[key] = metrics
                logger.info(
                    f"  {key}: f1_macro={metrics['f1_macro']:.4f} | "
                    f"f1_0={metrics['f1_0']:.4f} | f1_1={metrics['f1_1']:.4f} | "
                    f"roc_auc={metrics.get('roc_auc', float('nan')):.4f}"
                )
            except Exception as exc:
                logger.error(f"ModelEvaluator: evaluation of '{key}' failed: {exc}")
                report[key] = {"error": str(exc)}

        if log_to_mlflow and _MLFLOW_AVAILABLE:
            self._log_to_mlflow(report, X_test, y_test)

        return report

    def print_comparison_table(self, report: dict[str, dict[str, Any]]) -> None:
        """Print a formatted side-by-side comparison table to stdout."""
        metrics_to_show = ["f1_macro", "f1_0", "f1_1", "roc_auc", "precision_macro", "recall_macro", "coverage"]
        header = f"{'Metric':<22}" + "".join(f"{k:>14}" for k in [self.RULE_KEY, self.ML_KEY, self.HYBRID_KEY])
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)
        for metric in metrics_to_show:
            row = f"{metric:<22}"
            for key in [self.RULE_KEY, self.ML_KEY, self.HYBRID_KEY]:
                val = report.get(key, {}).get(metric)
                if val is None:
                    row += f"{'N/A':>14}"
                elif isinstance(val, float):
                    row += f"{val:>14.4f}"
                else:
                    row += f"{str(val):>14}"
            print(row)
        print(separator)

    def generate_report_text(self, report: dict[str, dict[str, Any]]) -> str:
        """Return a markdown-formatted comparison report string."""
        lines = [
            "## Model Comparison Report — Important Classifier",
            "",
            "| Metric | Rule-Based | ML Model | Hybrid |",
            "|---|---|---|---|",
        ]
        metrics_to_show = ["f1_macro", "f1_0", "f1_1", "roc_auc", "precision_macro", "recall_macro", "coverage"]
        for metric in metrics_to_show:
            vals = []
            for key in [self.RULE_KEY, self.ML_KEY, self.HYBRID_KEY]:
                v = report.get(key, {}).get(metric)
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v or "N/A"))
            lines.append(f"| {metric} | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append("")
        lines.append("### Per-classifier classification reports")
        for key in [self.RULE_KEY, self.ML_KEY, self.HYBRID_KEY]:
            lines.append(f"\n#### {key}")
            clf_report = report.get(key, {}).get("classification_report", "N/A")
            lines.append(f"```\n{clf_report}\n```")
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        clf,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        key: str,
    ) -> dict[str, Any]:
        """Compute the canonical metric set for one classifier."""
        y_pred = clf.predict(X_test)

        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        metrics: dict[str, Any] = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "f1_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=["not_important", "important"],
                zero_division=0,
            ),
        }

        # ROC-AUC — requires predict_proba
        try:
            proba = clf.predict_proba(X_test)
            proba_pos = proba[:, 1] if proba.ndim == 2 else proba
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba_pos))
        except (AttributeError, NotImplementedError, ValueError) as exc:
            logger.debug(f"ROC-AUC unavailable for '{key}': {exc}")
            metrics["roc_auc"] = float("nan")

        # Coverage — only meaningful for rule/hybrid classifiers
        if hasattr(clf, "coverage"):
            try:
                cov = clf.coverage(X_test)
                metrics["coverage"] = float(cov.get("fired_ratio", float("nan")))
            except Exception:
                metrics["coverage"] = float("nan")
        elif hasattr(clf, "coverage_report"):
            try:
                cov = clf.coverage_report(X_test)
                metrics["coverage"] = float(cov.get("fired_ratio", float("nan")))
            except Exception:
                metrics["coverage"] = float("nan")
        else:
            metrics["coverage"] = float("nan")

        return metrics

    def _log_to_mlflow(
        self,
        report: dict[str, dict[str, Any]],
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        """Log per-classifier metrics as nested MLflow child runs under the active parent."""
        if not _MLFLOW_AVAILABLE:
            return

        active_run = mlflow.active_run()
        if active_run is None:
            logger.warning(
                "ModelEvaluator: no active MLflow run found. "
                "Evaluation metrics will not be logged. "
                "Wrap evaluate() inside a mlflow.start_run() context."
            )
            return

        for key, metrics in report.items():
            if "error" in metrics:
                continue
            with mlflow.start_run(run_name=f"eval_{key}", nested=True):
                mlflow.set_tag("evaluator_type", key)
                scalar_keys = ["f1_macro", "f1_0", "f1_1", "roc_auc",
                               "precision_macro", "recall_macro", "coverage"]
                for k in scalar_keys:
                    v = metrics.get(k)
                    if isinstance(v, float) and not (v != v):  # skip NaN
                        try:
                            mlflow.log_metric(f"eval_{k}", v)
                        except Exception:
                            pass

                # Classification report as text artifact
                clf_report = metrics.get("classification_report", "")
                if clf_report:
                    with tempfile.NamedTemporaryFile(
                        suffix=f"_clf_report_{key}.txt",
                        delete=False, mode="w", encoding="utf-8"
                    ) as f:
                        f.write(clf_report)
                        mlflow.log_artifact(f.name, artifact_path=f"evaluation/{key}")

                # Confusion matrix plot
                if _MATPLOTLIB_AVAILABLE:
                    self._log_confusion_matrix(key, X_test, y_test)

        # Log the full comparison report JSON
        try:
            serializable = {
                k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
                for k, v in report.items()
                if "error" not in v
            }
            with tempfile.NamedTemporaryFile(
                suffix="_comparison_report.json",
                delete=False, mode="w", encoding="utf-8"
            ) as f:
                json.dump(serializable, f, indent=2, default=str)
                mlflow.log_artifact(f.name, artifact_path="evaluation")
        except Exception as exc:
            logger.warning(f"Could not log comparison report JSON: {exc}")

    def _log_confusion_matrix(
        self, key: str, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> None:
        clf = self._classifiers.get(key)
        if clf is None:
            return
        try:
            y_pred = clf.predict(X_test)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred,
                display_labels=["Not Important", "Important"],
                ax=ax, colorbar=False,
            )
            ax.set_title(f"Confusion Matrix — {key}")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path=f"evaluation/{key}")
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"Could not log confusion matrix for '{key}': {exc}")
