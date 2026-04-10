"""
ModelTrainer — orchestrates end-to-end training with Optuna + MLflow.

Responsibilities:
  1. Load processed parquet data
  2. Split train / holdout test set (stratified)
  3. Run Optuna study — each trial is a nested MLflow run
  4. Log best trial metrics, params, plots to parent MLflow run
  5. Retrain best pipeline on full train set
  6. Evaluate on holdout test set
  7. Register final model in MLflow Model Registry
  8. Save model artifact to models/ directory

MLflow structure:
  Experiment: "important-classifier"
  Parent run : training_<timestamp>
    ↳ Child runs: trial_0, trial_1, ... trial_N  (nested)
  Registered model: "important-classifier" → version tagged "Production"
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.evaluation import HybridClassifier, ModelEvaluator, RuleBasedClassifier
from src.training.optuna_objective import (
    FEATURE_COLS,
    TARGET_COL,
    ImportantClassifierObjective,
    _fit_with_resampling,
    _safe_predict_proba,
    _validate_feature_cols,
    cross_validate_pipeline,
)
from src.training.pipeline_builder import build_full_pipeline
from src.transformers import ResamplerTransformer

# Lazily imported for plotting — only available in training env
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plots will be skipped.")


class ModelTrainer:
    """
    High-level trainer that wires Optuna, MLflow and sklearn Pipeline together.

    Parameters
    ----------
    config_path : str | Path
        Path to the global config.yaml.
    model_config_path : str | Path
        Path to model1_config.yaml (search space overrides and training settings).
    data_path : str | Path | None
        Override for the processed parquet file path. If None, uses config value.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        model_config_path: str | Path = "configs/model1_config.yaml",
        data_path: str | Path | None = None,
    ) -> None:
        self.config = self._load_yaml(config_path)
        self.model_config = self._load_yaml(model_config_path)
        self.data_path = Path(
            data_path
            or Path(self.config["data"]["processed_dir"])
            / "eda_important_features.parquet"
        )
        self._setup_mlflow()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """
        Full training pipeline. Returns dict with best trial info and test metrics.
        """
        logger.info("=" * 70)
        logger.info("ModelTrainer: starting Model 1 — Important Classifier")
        logger.info("=" * 70)

        df = self._load_data()
        X_train, X_test, y_train, y_test = self._split_data(df)

        training_cfg = self.model_config.get("training", {})
        n_trials = training_cfg.get("optuna_trials", self.config["training"]["optuna_trials"])
        timeout = training_cfg.get("optuna_timeout_seconds", self.config["training"]["optuna_timeout_seconds"])
        cv_folds = training_cfg.get("cv_folds", self.config["training"]["cv_folds"])
        random_state = training_cfg.get("random_state", self.config["training"]["random_seed"])

        experiment_name = self.config["mlflow"]["experiment_name_model1"]
        timestamp = int(time.time())
        parent_run_name = f"training_{timestamp}"

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            logger.info(f"MLflow parent run: {parent_run.info.run_id}")
            mlflow.set_tag("model", "important-classifier")
            mlflow.set_tag("stage", "training")
            # ── Versioning tags — tie every run to its data + code snapshot ──
            mlflow.set_tag("data_version", self.config.get("data", {}).get("version", "unknown"))
            mlflow.set_tag("data_sha256", self._compute_file_sha256(self.data_path))
            mlflow.set_tag("git_commit", self._get_git_commit())
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            # ── Optuna study ──────────────────────────────────────────────────
            study = self._run_optuna_study(
                X_train=X_train,
                y_train=y_train,
                n_trials=n_trials,
                timeout=timeout,
                cv_folds=cv_folds,
                random_state=random_state,
                parent_run_id=parent_run.info.run_id,
            )

            best_trial = study.best_trial
            best_config = dict(best_trial.params)
            best_config["random_state"] = random_state
            resampler_strategy = best_trial.user_attrs.get("resampler_strategy", "none")

            logger.info(f"Best trial #{best_trial.number}: f1_macro={best_trial.value:.4f}")
            logger.info(f"Best config: {json.dumps(best_config, indent=2, default=str)}")

            mlflow.log_param("best_trial_number", best_trial.number)
            mlflow.log_param("best_model_type", best_config.get("model_type"))
            mlflow.log_param("best_resampler", resampler_strategy)
            self._safe_log_metric("best_cv_f1_macro", best_trial.value)
            for k, v in best_trial.user_attrs.items():
                if isinstance(v, (int, float)):
                    self._safe_log_metric(f"best_cv_{k}", v)

            # ── Retrain on full train set ─────────────────────────────────────
            logger.info("Retraining best pipeline on full training set...")
            resampler_strategy_cfg = best_config.pop("resampler_strategy", resampler_strategy)
            final_pipeline = build_full_pipeline(best_config)
            best_config["resampler_strategy"] = resampler_strategy_cfg  # restore

            if resampler_strategy_cfg != "none":
                resampler = ResamplerTransformer(strategy=resampler_strategy_cfg, random_state=random_state)
                final_pipeline = _fit_with_resampling(final_pipeline, X_train, y_train, resampler)
            else:
                final_pipeline.fit(X_train, y_train)

            # ── Evaluate on holdout test set ──────────────────────────────────
            test_metrics = self._evaluate_on_test(final_pipeline, X_test, y_test)
            logger.info(f"Holdout test metrics: {json.dumps(test_metrics, indent=2)}")
            for k, v in test_metrics.items():
                self._safe_log_metric(f"test_{k}", v)

            # ── 3-way comparison: rule-based / ML / hybrid ────────────────────
            comparison_report = self._run_system_evaluation(
                final_pipeline, X_test, y_test
            )
            result_comparison = comparison_report  # stored for return value

            # ── Log plots ─────────────────────────────────────────────────────
            if _MATPLOTLIB_AVAILABLE:
                self._log_confusion_matrix(final_pipeline, X_test, y_test)
                self._log_roc_curve(final_pipeline, X_test, y_test)
                self._log_feature_importance(final_pipeline, best_config.get("model_type"))
                self._log_optuna_plots(study)

            # ── Log classification report as artifact ─────────────────────────
            report = classification_report(
                y_test,
                final_pipeline.predict(X_test),
                target_names=["not_important", "important"],
                zero_division=0,
            )
            self._log_text_artifact(report, "classification_report.txt")
            self._log_text_artifact(
                json.dumps(best_config, indent=2, default=str),
                "best_hyperparameters.json",
            )

            # ── Register model in MLflow registry ─────────────────────────────
            registered_model_name = self.config["mlflow"]["registered_model_name_model1"]
            model_uri = self._log_and_register_model(
                final_pipeline, X_test, registered_model_name
            )
            logger.info(f"Model registered: {model_uri}")

            result = {
                "run_id": parent_run.info.run_id,
                "best_trial": best_trial.number,
                "best_cv_f1_macro": best_trial.value,
                "test_metrics": test_metrics,
                "comparison_report": result_comparison,
                "model_uri": model_uri,
                "best_config": best_config,
            }

        logger.info("Training complete.")
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _setup_mlflow(self) -> None:
        tracking_uri = self.config["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = self.config["mlflow"]["experiment_name_model1"]
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {tracking_uri} | Experiment: {experiment_name}")

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"ModelTrainer: processed data not found at '{self.data_path}'. "
                f"Run the EDA feature engineering script first."
            )
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} cols from '{self.data_path}'")

        # Guard: remove forbidden columns if they snuck in (must stay in sync with FORBIDDEN_COLS)
        from src.training.optuna_objective import FORBIDDEN_COLS
        forbidden_present = set(df.columns) & FORBIDDEN_COLS
        if forbidden_present:
            logger.warning(
                f"ModelTrainer: dropping forbidden columns from input data: {forbidden_present}"
            )
            df = df.drop(columns=list(forbidden_present))

        _validate_feature_cols(df)
        return df

    def _split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        test_size = self.config["training"]["test_size"]
        random_state = self.config["training"]["random_seed"]

        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info(
            f"Train/test split: train={len(X_train)} ({y_train.mean():.1%} positive), "
            f"test={len(X_test)} ({y_test.mean():.1%} positive)"
        )
        return X_train, X_test, y_train, y_test

    def _run_optuna_study(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_trials: int,
        timeout: int,
        cv_folds: int,
        random_state: int,
        parent_run_id: str,
    ) -> optuna.Study:
        # Reset index so pd.concat aligns correctly — X_train has a
        # non-contiguous index after train_test_split, y_train is a plain
        # numpy array (0-based), so alignment would produce NaN without this.
        train_df = X_train.reset_index(drop=True).copy()
        train_df[TARGET_COL] = y_train
        objective = ImportantClassifierObjective(
            df=train_df,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        # Wrap objective to log each trial as a nested MLflow child run
        def mlflow_objective(trial: optuna.Trial) -> float:
            with mlflow.start_run(
                run_name=f"trial_{trial.number:04d}",
                nested=True,
            ):
                mlflow.set_tag("trial_number", trial.number)
                value = objective(trial)
                # Log trial params and metrics
                for k, v in trial.params.items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        pass
                for k, v in trial.user_attrs.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_metric("cv_f1_macro", value)
            return value

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="important-classifier",
        )

        logger.info(f"Starting Optuna: n_trials={n_trials}, timeout={timeout}s")
        study.optimize(
            mlflow_objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        if completed == 0:
            raise RuntimeError(
                f"Optuna study finished with 0 completed trials "
                f"({pruned} pruned, {failed} failed). "
                f"Check trial error logs above for the root cause. "
                f"Common causes: data issues, missing columns, import errors."
            )

        logger.info(
            f"Optuna finished: {completed} complete, {pruned} pruned, {failed} failed. "
            f"Best f1_macro={study.best_value:.4f}"
        )
        return study

    def _run_system_evaluation(
        self,
        ml_pipeline,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> dict:
        """
        Run the 3-way comparison (rule-based vs ML vs hybrid) on the holdout set.

        All three classifiers receive the identical test split so metrics are
        directly comparable. Results are logged as nested MLflow child runs
        under the active parent run.
        """
        try:
            rule_clf = RuleBasedClassifier()
            hybrid_clf = HybridClassifier(
                rule_classifier=rule_clf,
                ml_model=ml_pipeline,
            )
            evaluator = ModelEvaluator(
                rule_clf=rule_clf,
                ml_clf=ml_pipeline,
                hybrid_clf=hybrid_clf,
            )
            report = evaluator.evaluate(X_test, y_test, log_to_mlflow=True)
            # Print to console for immediate visibility
            logger.info("=== System Evaluation — Comparable Metrics ===")
            evaluator.print_comparison_table(report)
            # Log markdown report as artifact
            md_report = evaluator.generate_report_text(report)
            self._log_text_artifact(md_report, "system_comparison_report.md")
            return report
        except Exception as exc:
            logger.error(
                f"ModelTrainer: 3-way system evaluation failed: {exc}. "
                f"Training result is still valid — this is a non-fatal error."
            )
            return {}

    def _evaluate_on_test(
        self, pipeline, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> dict[str, float]:
        y_pred = pipeline.predict(X_test)
        y_proba = _safe_predict_proba(pipeline, X_test)

        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        metrics = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "f1_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                metrics["roc_auc"] = 0.5
        return metrics

    def _log_confusion_matrix(self, pipeline, X_test, y_test) -> None:
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                pipeline.predict(X_test),
                display_labels=["Not Important", "Important"],
                ax=ax,
                colorbar=False,
            )
            ax.set_title("Confusion Matrix — Holdout Test Set")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Confusion matrix logged to MLflow.")
        except Exception as exc:
            logger.warning(f"Could not log confusion matrix: {exc}")

    def _log_roc_curve(self, pipeline, X_test, y_test) -> None:
        try:
            y_proba = _safe_predict_proba(pipeline, X_test)
            if y_proba is None:
                return
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve — Holdout Test Set")
            ax.legend()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("ROC curve logged to MLflow.")
        except Exception as exc:
            logger.warning(f"Could not log ROC curve: {exc}")

    def _log_feature_importance(self, pipeline, model_type: str | None) -> None:
        """Log feature importances for tree-based models."""
        try:
            classifier = pipeline.named_steps.get("classifier")
            preprocessor = pipeline.named_steps.get("preprocessor")

            # VotingClassifier: try first estimator
            if hasattr(classifier, "estimators_"):
                classifier = classifier.estimators_[0]

            if not hasattr(classifier, "feature_importances_"):
                logger.debug(f"Model type '{model_type}' has no feature_importances_. Skipping.")
                return

            importances = classifier.feature_importances_
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            if len(importances) != len(feature_names):
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            n_top = min(25, len(importances))
            top_idx = np.argsort(importances)[-n_top:][::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                [feature_names[i] for i in top_idx][::-1],
                importances[top_idx][::-1],
            )
            ax.set_title(f"Top {n_top} Feature Importances")
            ax.set_xlabel("Importance")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Feature importance plot logged to MLflow.")
        except Exception as exc:
            logger.warning(f"Could not log feature importance: {exc}")

    def _log_optuna_plots(self, study: optuna.Study) -> None:
        """Log Optuna visualisations (optimization history, param importance)."""
        try:
            import optuna.visualization as vis

            plots = {
                "optuna_optimization_history.html": vis.plot_optimization_history(study),
                "optuna_param_importances.html": vis.plot_param_importances(study),
                "optuna_parallel_coordinate.html": vis.plot_parallel_coordinate(study),
            }
            for filename, fig in plots.items():
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                    f.write(fig.to_html())
                    mlflow.log_artifact(f.name, artifact_path="optuna_plots")
            logger.debug("Optuna visualisations logged to MLflow.")
        except Exception as exc:
            logger.warning(f"Could not log Optuna plots: {exc}")

    def _log_and_register_model(
        self, pipeline, X_example: pd.DataFrame, registered_model_name: str
    ) -> str:
        """
        Log model to MLflow and register in the Model Registry.

        Newer MLflow versions (2.10+) call log_model_metrics_for_step inside
        log_model, which tries to re-insert metrics already present in the run
        (e.g. best_cv_roc_auc with is_nan=1), triggering a SQLite UNIQUE
        constraint violation. We suppress this via env var and catch any
        duplicate-metric errors as a fallback.
        """
        import joblib
        import os

        input_example = X_example.head(5)

        original_env = os.environ.get("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING")
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=registered_model_name,
            )
        except Exception as exc:
            if "UNIQUE constraint failed: metrics" not in str(exc):
                raise
            logger.warning(
                f"MLflow log_model raised a duplicate-metric error (known SQLite issue). "
                f"Model artifact was saved; registry registration may be incomplete. "
                f"Error: {exc}"
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model_info = type("_ModelInfo", (), {"model_uri": model_uri})()
        finally:
            if original_env is None:
                os.environ.pop("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", None)
            else:
                os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = original_env

        # Save a local copy to models/
        models_dir = Path(self.config.get("models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        local_path = models_dir / "important_classifier_best.pkl"
        joblib.dump(pipeline, local_path)
        logger.info(f"Model saved locally: {local_path}")

        return model_info.model_uri

    @staticmethod
    def _safe_log_metric(key: str, value: float) -> None:
        """
        Log a metric to MLflow, skipping NaN/inf values and catching duplicate errors.

        MLflow's SQLite backend raises IntegrityError when the same metric key+timestamp
        combination is inserted twice. This helper makes all metric logging idempotent.
        """
        import math
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            logger.debug(f"Skipping metric '{key}' — value is NaN/inf.")
            return
        try:
            mlflow.log_metric(key, value)
        except Exception as exc:
            if "UNIQUE constraint failed: metrics" in str(exc):
                logger.debug(f"Metric '{key}' already logged — skipping duplicate.")
            else:
                logger.warning(f"Could not log metric '{key}={value}': {exc}")

    @staticmethod
    def _compute_file_sha256(path: Path) -> str:
        """
        Compute SHA-256 digest of a file so every MLflow run is pinned to an
        exact data snapshot — catching silent parquet overwrites.
        """
        try:
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception as exc:
            logger.warning(f"Could not compute SHA-256 for '{path}': {exc}")
            return "unknown"

    @staticmethod
    def _get_git_commit() -> str:
        """
        Return the current HEAD git commit hash (short 8-char form).
        Falls back to 'unknown' when not in a git repo or git is unavailable.
        Best practice: every MLflow run should be reproducible from code + data;
        the commit hash makes that lookup possible.
        """
        try:
            result = subprocess.check_output(
                ["git", "rev-parse", "--short=8", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return result.decode().strip()
        except Exception:
            return "unknown"

    def _log_text_artifact(self, content: str, filename: str) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=f"_{filename}", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(content)
            mlflow.log_artifact(f.name, artifact_path="reports")

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: '{path}'")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
