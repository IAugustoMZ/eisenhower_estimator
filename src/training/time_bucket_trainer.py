"""
TimeBucketTrainer — orchestrates end-to-end training for the
Time Bucket Classifier (Model 3b).

Classifies tasks into 6 duration buckets:
  ≤2 min | 3–5 min | 6–10 min | 11–30 min | 31–60 min | >60 min

Uses the same features as Model 3 (time spent regressor) but treats
the prediction as a multiclass classification problem.

MLflow structure:
  Experiment: "time-bucket-classifier"
  Parent run : training_<timestamp>
    ↳ Child runs: trial_0, trial_1, ... trial_N  (nested)
  Registered model: "time-bucket-classifier" → version tagged "Production"
"""
from __future__ import annotations

import json
import os
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
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.training.optuna_objective_time_bucket import (
    FEATURE_COLS,
    TARGET_COL,
    LOG_DURATION_COL,
    GROUP_COL,
    FORBIDDEN_COLS,
    TimeBucketClassifierObjective,
    _validate_feature_cols,
    _encode_buckets,
)
from src.training.pipeline_builder_time_bucket import (
    BUCKET_ORDER,
    BUCKET_TO_INT,
    build_full_pipeline_time_bucket,
)
from src.training.pipeline_builder_time_spent import TaskCVImputer
from src.transformers import ResamplerTransformer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plots will be skipped.")


class TimeBucketTrainer:
    """
    High-level trainer for the Time Bucket Classifier (Model 3b).

    Parameters
    ----------
    config_path : str | Path
        Path to the global config.yaml.
    model_config_path : str | Path
        Path to model3b_config.yaml.
    data_path : str | Path | None
        Override for the processed parquet file path.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        model_config_path: str | Path = "configs/model3b_config.yaml",
        data_path: str | Path | None = None,
    ) -> None:
        self.config = self._load_yaml(config_path)
        self.model_config = self._load_yaml(model_config_path)
        self.data_path = Path(
            data_path
            or Path(self.config["data"]["processed_dir"])
            / "eda_time_spent_features.parquet"
        )
        self._setup_mlflow()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """
        Full training pipeline. Returns dict with best trial info and test metrics.
        """
        logger.info("=" * 70)
        logger.info(
            "TimeBucketTrainer: starting Model 3b — Time Bucket Classifier"
        )
        logger.info("=" * 70)

        df = self._load_data()
        X_train, X_test, y_train, y_test, meta_train, meta_test = (
            self._split_data(df)
        )

        training_cfg = self.model_config.get("training", {})
        n_trials = training_cfg.get(
            "optuna_trials",
            self.config["training"]["optuna_trials"],
        )
        timeout = training_cfg.get(
            "optuna_timeout_seconds",
            self.config["training"]["optuna_timeout_seconds"],
        )
        cv_folds = training_cfg.get(
            "cv_folds", self.config["training"]["cv_folds"]
        )
        random_state = training_cfg.get(
            "random_state", self.config["training"]["random_seed"]
        )

        experiment_name = self.config["mlflow"].get(
            "experiment_name_model3b", "time-bucket-classifier"
        )
        timestamp = int(time.time())
        parent_run_name = f"training_{timestamp}"

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            logger.info(f"MLflow parent run: {parent_run.info.run_id}")
            mlflow.set_tag("model", "time-bucket-classifier")
            mlflow.set_tag("stage", "training")
            # ── Versioning tags ───────────────────────────────────────────
            data_version = self.config["data"].get("version", "unknown")
            mlflow.set_tag("data_version", data_version)
            mlflow.set_tag(
                "config_version",
                self.model_config.get("config_version", "unknown"),
            )
            # SHA-256 of processed parquet (lineage anchor)
            if self.data_path.exists():
                import hashlib
                h = hashlib.sha256()
                with open(self.data_path, "rb") as _f:
                    for _chunk in iter(lambda: _f.read(65536), b""):
                        h.update(_chunk)
                mlflow.set_tag("data_sha256", h.hexdigest())
            # Short git commit hash
            try:
                import subprocess
                _git_hash = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                mlflow.set_tag("git_commit", _git_hash)
            except Exception:
                mlflow.set_tag("git_commit", "unknown")
            # ─────────────────────────────────────────────────────────────
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_classes", len(BUCKET_ORDER))

            # ── Optuna study ──────────────────────────────────────────────
            study = self._run_optuna_study(
                X_train=X_train,
                y_train=y_train,
                meta_train=meta_train,
                n_trials=n_trials,
                timeout=timeout,
                cv_folds=cv_folds,
                random_state=random_state,
                parent_run_id=parent_run.info.run_id,
            )

            best_trial = study.best_trial
            best_config = dict(best_trial.params)
            best_config["random_state"] = random_state
            best_resampler = best_trial.user_attrs.get(
                "resampler_strategy", "none"
            )

            logger.info(
                f"Best trial #{best_trial.number}: "
                f"f1_macro={best_trial.value:.4f}"
            )
            logger.info(
                f"Best config: {json.dumps(best_config, indent=2, default=str)}"
            )

            mlflow.log_param("best_trial_number", best_trial.number)
            mlflow.log_param("best_model_type", best_config.get("model_type"))
            mlflow.log_param("best_resampler", best_resampler)
            self._safe_log_metric("best_cv_f1_macro", best_trial.value)
            for k, v in best_trial.user_attrs.items():
                if isinstance(v, (int, float)):
                    self._safe_log_metric(f"best_cv_{k}", v)

            # ── Retrain on full train set ─────────────────────────────────
            logger.info("Retraining best pipeline on full training set...")

            # Fit TaskCVImputer
            cv_imputer = TaskCVImputer()
            X_train_with_target = X_train.copy()
            if LOG_DURATION_COL in meta_train.columns:
                X_train_with_target["log_duration"] = meta_train[
                    LOG_DURATION_COL
                ].values
            cv_imputer.fit(X_train_with_target)
            X_train_imputed = cv_imputer.transform(X_train)
            X_test_imputed = cv_imputer.transform(X_test)

            final_pipeline = build_full_pipeline_time_bucket(best_config)

            # Apply resampling if the best trial used it
            if best_resampler != "none":
                resampler = ResamplerTransformer(
                    strategy=best_resampler,
                    random_state=random_state,
                )
                preprocessor = final_pipeline.named_steps["preprocessor"]
                X_transformed = preprocessor.fit_transform(
                    X_train_imputed, y_train
                )
                X_resampled, y_resampled = resampler.fit_resample(
                    X_transformed, y_train
                )
                rest_pipeline = final_pipeline[1:]
                rest_pipeline.fit(X_resampled, y_resampled)
            else:
                final_pipeline.fit(X_train_imputed, y_train)

            # ── Evaluate on holdout test set ──────────────────────────────
            test_metrics = self._evaluate_on_test(
                final_pipeline, X_test_imputed, y_test
            )
            logger.info(
                f"Holdout test metrics:\n{json.dumps(test_metrics, indent=2)}"
            )
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    # Skip human-readable Unicode keys — MLflow only gets the safe keys
                    if any(c in k for c in ("≤", "–", ">", "–")):
                        continue
                    self._safe_log_metric(f"test_{k}", v)

            # ── Log plots and reports ─────────────────────────────────────
            if _MATPLOTLIB_AVAILABLE:
                self._log_confusion_matrix(
                    final_pipeline, X_test_imputed, y_test
                )
                self._log_per_class_f1(
                    final_pipeline, X_test_imputed, y_test
                )
                self._log_optuna_plots(study)

            report = self._build_classification_report(
                final_pipeline, X_test_imputed, y_test
            )
            self._log_text_artifact(report, "classification_report.txt")
            self._log_text_artifact(
                json.dumps(best_config, indent=2, default=str),
                "best_hyperparameters.json",
            )

            # ── Register model ────────────────────────────────────────────
            registered_model_name = self.config["mlflow"].get(
                "registered_model_name_model3b", "time-bucket-classifier"
            )
            model_uri = self._log_and_register_model(
                final_pipeline, X_test_imputed, registered_model_name
            )
            logger.info(f"Model registered: {model_uri}")

            result = {
                "run_id": parent_run.info.run_id,
                "best_trial": best_trial.number,
                "best_cv_f1_macro": best_trial.value,
                "test_metrics": test_metrics,
                "model_uri": model_uri,
                "best_config": best_config,
            }

        logger.info("Training complete.")
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _setup_mlflow(self) -> None:
        tracking_uri = self.config["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = self.config["mlflow"].get(
            "experiment_name_model3b", "time-bucket-classifier"
        )
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow tracking URI: {tracking_uri} | "
            f"Experiment: {experiment_name}"
        )

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"TimeBucketTrainer: data not found at '{self.data_path}'. "
                f"Run the EDA script first: notebooks/eda_time_spent.py"
            )
        df = pd.read_parquet(self.data_path)
        logger.info(
            f"Loaded data: {df.shape[0]} rows, {df.shape[1]} cols "
            f"from '{self.data_path}'"
        )

        if TARGET_COL not in df.columns:
            raise ValueError(
                f"TimeBucketTrainer: target column '{TARGET_COL}' not found. "
                f"Available: {list(df.columns)}"
            )

        # Log class distribution
        logger.info(
            f"Target distribution:\n"
            f"{df[TARGET_COL].value_counts().sort_index().to_string()}"
        )

        _validate_feature_cols(df)
        return df

    def _split_data(
        self, df: pd.DataFrame
    ) -> tuple[
        pd.DataFrame, pd.DataFrame,
        np.ndarray, np.ndarray,
        pd.DataFrame, pd.DataFrame,
    ]:
        """
        Stratified train/test split.

        Returns X_train, X_test, y_train (int-encoded), y_test (int-encoded),
        meta_train, meta_test.
        """
        test_size = self.config["training"]["test_size"]
        random_state = self.config["training"]["random_seed"]

        X = df[FEATURE_COLS].copy()
        y = _encode_buckets(df[TARGET_COL])

        # Metadata for analysis
        meta_cols = [GROUP_COL, "task_type"]
        if LOG_DURATION_COL in df.columns:
            meta_cols.append(LOG_DURATION_COL)
        if "duration_minutes" in df.columns:
            meta_cols.append("duration_minutes")
        meta = df[meta_cols].copy()

        X_train, X_test, y_train, y_test, meta_train, meta_test = (
            train_test_split(
                X, y, meta,
                test_size=test_size,
                stratify=y,
                random_state=random_state,
            )
        )

        logger.info(
            f"Train/test split: train={len(X_train)}, test={len(X_test)}"
        )

        # Log train class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        dist_str = ", ".join(
            f"{BUCKET_ORDER[u]}={c}" for u, c in zip(unique, counts)
        )
        logger.info(f"Train class distribution: {dist_str}")

        return X_train, X_test, y_train, y_test, meta_train, meta_test

    def _run_optuna_study(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        meta_train: pd.DataFrame,
        n_trials: int,
        timeout: int,
        cv_folds: int,
        random_state: int,
        parent_run_id: str,
    ) -> optuna.Study:
        """Run Optuna — each trial as a nested MLflow child run."""
        # Reconstruct full DataFrame for the objective (CLAUDE.md rule #10)
        train_df = X_train.reset_index(drop=True).copy()
        meta_reset = meta_train.reset_index(drop=True)

        # Add target column (string bucket labels for encoding)
        int_to_bucket = {v: k for k, v in BUCKET_TO_INT.items()}
        train_df[TARGET_COL] = pd.Series(y_train).map(int_to_bucket).values

        # Add log_duration for leakage-safe imputation
        if LOG_DURATION_COL in meta_reset.columns:
            train_df[LOG_DURATION_COL] = meta_reset[LOG_DURATION_COL].values

        objective = TimeBucketClassifierObjective(
            df=train_df,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        def mlflow_objective(trial: optuna.Trial) -> float:
            with mlflow.start_run(
                run_name=f"trial_{trial.number:04d}",
                nested=True,
            ):
                mlflow.set_tag("trial_number", trial.number)
                value = objective(trial)
                for k, v in trial.params.items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        pass
                for k, v in trial.user_attrs.items():
                    if isinstance(v, (int, float)):
                        self._safe_log_metric(k, v)
                self._safe_log_metric("cv_f1_macro", value)
            return value

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(
            direction="maximize",  # Maximise F1 macro
            sampler=sampler,
            study_name="time-bucket-classifier",
        )

        logger.info(
            f"Starting Optuna: n_trials={n_trials}, timeout={timeout}s"
        )
        study.optimize(
            mlflow_objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,),
            show_progress_bar=False,
        )

        completed = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        pruned = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ])
        failed = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.FAIL
        ])

        if completed == 0:
            raise RuntimeError(
                f"Optuna study finished with 0 completed trials "
                f"({pruned} pruned, {failed} failed)."
            )

        logger.info(
            f"Optuna finished: {completed} complete, {pruned} pruned, "
            f"{failed} failed. Best f1_macro={study.best_value:.4f}"
        )
        return study

    def _evaluate_on_test(
        self, pipeline, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate on holdout test set."""
        y_pred = pipeline.predict(X_test)

        metrics: dict[str, float] = {
            "f1_macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        # Per-class F1 — use MLflow-safe metric keys (alphanumeric + _ only)
        # BUCKET_ORDER labels contain Unicode (≤, –, >) which MLflow rejects.
        _safe_bucket_keys = [
            "le2min", "3to5min", "6to10min", "11to30min", "31to60min", "gt60min"
        ]
        f1_per_class = f1_score(
            y_test, y_pred, average=None, zero_division=0
        )
        for i, f1_val in enumerate(f1_per_class):
            safe_key = _safe_bucket_keys[i] if i < len(_safe_bucket_keys) else f"class_{i}"
            label = BUCKET_ORDER[i] if i < len(BUCKET_ORDER) else f"class_{i}"
            # Keep human-readable label in the dict for display, safe key for MLflow
            metrics[f"f1_{label}"] = float(f1_val)
            metrics[f"f1_mlflow_{safe_key}"] = float(f1_val)

        return metrics

    # ── Plotting helpers ──────────────────────────────────────────────────────

    def _log_confusion_matrix(
        self, pipeline, X_test, y_test
    ) -> None:
        """Log a confusion matrix heatmap as an MLflow artifact."""
        try:
            y_pred = pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax)

            labels = [
                BUCKET_ORDER[i] if i < len(BUCKET_ORDER) else str(i)
                for i in range(cm.shape[0])
            ]
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=labels,
                yticklabels=labels,
                ylabel="Actual",
                xlabel="Predicted",
                title="Confusion Matrix — Time Bucket Classifier",
            )
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor",
            )

            # Add text annotations
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10,
                    )

            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Confusion matrix plot logged.")
        except Exception as exc:
            logger.warning(f"Could not log confusion matrix: {exc}")

    def _log_per_class_f1(self, pipeline, X_test, y_test) -> None:
        """Log a bar chart of per-class F1 scores."""
        try:
            y_pred = pipeline.predict(X_test)
            f1_per_class = f1_score(
                y_test, y_pred, average=None, zero_division=0
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [
                BUCKET_ORDER[i] if i < len(BUCKET_ORDER) else f"class_{i}"
                for i in range(len(f1_per_class))
            ]
            bars = ax.bar(labels, f1_per_class, color="steelblue", edgecolor="black")
            ax.set_ylabel("F1 Score")
            ax.set_title("Per-Class F1 — Time Bucket Classifier")
            ax.set_ylim(0, 1.05)

            for bar, val in zip(bars, f1_per_class):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                )

            plt.setp(
                ax.get_xticklabels(), rotation=30, ha="right",
                rotation_mode="anchor",
            )
            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Per-class F1 bar chart logged.")
        except Exception as exc:
            logger.warning(f"Could not log per-class F1 chart: {exc}")

    def _log_optuna_plots(self, study: optuna.Study) -> None:
        """Log Optuna visualisation plots."""
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            fig = plot_optimization_history(study)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.figure.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig.figure)

            fig = plot_param_importances(study)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.figure.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig.figure)

            logger.debug("Optuna visualisation plots logged.")
        except Exception as exc:
            logger.warning(f"Could not log Optuna plots: {exc}")

    def _build_classification_report(
        self, pipeline, X_test, y_test
    ) -> str:
        """Build a text classification report."""
        y_pred = pipeline.predict(X_test)
        labels = list(range(len(BUCKET_ORDER)))
        target_names = BUCKET_ORDER

        report_lines = [
            "=" * 70,
            "Time Bucket Classifier — Classification Report",
            "=" * 70,
            "",
            classification_report(
                y_test, y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0,
            ),
            "",
            "Confusion Matrix:",
            str(confusion_matrix(y_test, y_pred)),
        ]
        return "\n".join(report_lines)

    # ── Model registration ────────────────────────────────────────────────────

    def _log_and_register_model(
        self, pipeline, X_test, registered_model_name: str
    ) -> str:
        """Log the model to MLflow and register it."""
        input_example = X_test.head(1)

        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=registered_model_name,
            )
        finally:
            os.environ.pop("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", None)

        # Also save locally
        models_dir = Path(self.config.get("models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        local_path = models_dir / "time_bucket_classifier_best.pkl"
        try:
            import joblib
            joblib.dump(pipeline, local_path)
            logger.info(f"Model saved locally: {local_path}")
        except Exception as exc:
            logger.warning(f"Could not save model locally: {exc}")

        return model_info.model_uri

    # ── Utility methods ───────────────────────────────────────────────────────

    def _safe_log_metric(self, key: str, value: float) -> None:
        """Log metric, skipping NaN/inf and catching duplicate-key errors."""
        import math
        if not isinstance(value, (int, float)):
            return
        if math.isnan(value) or math.isinf(value):
            logger.debug(f"Skipping metric '{key}': value is {value}")
            return
        try:
            mlflow.log_metric(key, value)
        except Exception as exc:
            if "UNIQUE constraint" in str(exc):
                logger.debug(f"Metric '{key}' already logged — skipping.")
            else:
                logger.warning(f"Failed to log metric '{key}': {exc}")

    def _log_text_artifact(self, text: str, filename: str) -> None:
        """Write text content to a temp file and log as MLflow artifact."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="reports")
            logger.debug(f"Artifact logged: {filename}")
        except Exception as exc:
            logger.warning(f"Could not log artifact '{filename}': {exc}")

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        """Load a YAML configuration file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
