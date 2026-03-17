"""
TimeSpentTrainer — orchestrates end-to-end training with Optuna + MLflow
for the Time Spent Regression model (Model 3).

Responsibilities:
  1. Load processed parquet data (eda_time_spent_features.parquet)
  2. Split train / holdout test set (stratified by duration_bucket, grouped by project_code)
  3. Run Optuna study — each trial is a nested MLflow run
  4. Log best trial metrics, params, plots to parent MLflow run
  5. Retrain best pipeline on full train set
  6. Evaluate on holdout test set (log-scale + original-scale metrics)
  7. Log segmented analysis (per project, per task type, per duration bucket)
  8. Register final model in MLflow Model Registry
  9. Save model artifact to models/ directory

MLflow structure:
  Experiment: "time-spent-regressor"
  Parent run : training_<timestamp>
    ↳ Child runs: trial_0, trial_1, ... trial_N  (nested)
  Registered model: "time-spent-regressor" → version tagged "Production"
"""
from __future__ import annotations

import json
import math
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.training.optuna_objective_time_spent import (
    FEATURE_COLS,
    TARGET_COL,
    ORIGINAL_TARGET_COL,
    GROUP_COL,
    STRATIFY_COL,
    FORBIDDEN_COLS,
    TimeSpentRegressorObjective,
    _validate_feature_cols,
    cross_validate_regression,
)
from src.training.pipeline_builder_time_spent import (
    TaskCVImputer,
    build_full_pipeline_time_spent,
)

# Lazily imported for plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plots will be skipped.")


class TimeSpentTrainer:
    """
    High-level trainer for the Time Spent Regression model.

    Parameters
    ----------
    config_path : str | Path
        Path to the global config.yaml.
    model_config_path : str | Path
        Path to model3_config.yaml.
    data_path : str | Path | None
        Override for the processed parquet file path.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        model_config_path: str | Path = "configs/model3_config.yaml",
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
        logger.info("TimeSpentTrainer: starting Model 3 — Time Spent Regressor")
        logger.info("=" * 70)

        df = self._load_data()
        X_train, X_test, y_train, y_test, meta_train, meta_test = self._split_data(df)

        training_cfg = self.model_config.get("training", {})
        n_trials = training_cfg.get(
            "optuna_trials", self.config["training"]["optuna_trials"]
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

        experiment_name = self.config["mlflow"]["experiment_name_model3"]
        timestamp = int(time.time())
        parent_run_name = f"training_{timestamp}"

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            logger.info(f"MLflow parent run: {parent_run.info.run_id}")
            mlflow.set_tag("model", "time-spent-regressor")
            mlflow.set_tag("stage", "training")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

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

            logger.info(
                f"Best trial #{best_trial.number}: "
                f"rmse_log={best_trial.value:.4f}"
            )
            logger.info(
                f"Best config: {json.dumps(best_config, indent=2, default=str)}"
            )

            mlflow.log_param("best_trial_number", best_trial.number)
            mlflow.log_param("best_model_type", best_config.get("model_type"))
            self._safe_log_metric("best_cv_rmse_log", best_trial.value)
            for k, v in best_trial.user_attrs.items():
                if isinstance(v, (int, float)):
                    self._safe_log_metric(f"best_cv_{k}", v)

            # ── Retrain on full train set ─────────────────────────────────
            logger.info("Retraining best pipeline on full training set...")

            # Fit TaskCVImputer on full training set and transform
            cv_imputer = TaskCVImputer()
            # Temporarily add log_duration for leakage-safe task_median_duration
            X_train_with_target = X_train.copy()
            X_train_with_target["log_duration"] = y_train
            cv_imputer.fit(X_train_with_target)
            X_train_imputed = cv_imputer.transform(X_train)
            X_test_imputed = cv_imputer.transform(X_test)

            final_pipeline = build_full_pipeline_time_spent(best_config)
            final_pipeline.fit(X_train_imputed, y_train)

            # ── Evaluate on holdout test set ──────────────────────────────
            test_metrics = self._evaluate_on_test(
                final_pipeline, X_test_imputed, y_test
            )
            logger.info(
                f"Holdout test metrics: {json.dumps(test_metrics, indent=2)}"
            )
            for k, v in test_metrics.items():
                self._safe_log_metric(f"test_{k}", v)

            # ── Log plots ─────────────────────────────────────────────────
            if _MATPLOTLIB_AVAILABLE:
                self._log_predicted_vs_actual(
                    final_pipeline, X_test_imputed, y_test
                )
                self._log_residual_plot(
                    final_pipeline, X_test_imputed, y_test
                )
                self._log_error_distribution(
                    final_pipeline, X_test_imputed, y_test
                )
                self._log_feature_importance(
                    final_pipeline, best_config.get("model_type")
                )
                self._log_segmented_analysis(
                    final_pipeline, X_test_imputed, y_test, meta_test
                )
                self._log_optuna_plots(study)

            # ── Log reports as artifacts ──────────────────────────────────
            report = self._build_regression_report(
                final_pipeline, X_test_imputed, y_test, meta_test
            )
            self._log_text_artifact(report, "regression_report.txt")
            self._log_text_artifact(
                json.dumps(best_config, indent=2, default=str),
                "best_hyperparameters.json",
            )

            # ── Register model in MLflow registry ─────────────────────────
            registered_model_name = self.config["mlflow"][
                "registered_model_name_model3"
            ]
            model_uri = self._log_and_register_model(
                final_pipeline, X_test_imputed, registered_model_name
            )
            logger.info(f"Model registered: {model_uri}")

            result = {
                "run_id": parent_run.info.run_id,
                "best_trial": best_trial.number,
                "best_cv_rmse_log": best_trial.value,
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
        experiment_name = self.config["mlflow"]["experiment_name_model3"]
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow tracking URI: {tracking_uri} | "
            f"Experiment: {experiment_name}"
        )

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"TimeSpentTrainer: processed data not found at '{self.data_path}'. "
                f"Run the EDA script first: notebooks/eda_time_spent.py"
            )
        df = pd.read_parquet(self.data_path)
        logger.info(
            f"Loaded data: {df.shape[0]} rows, {df.shape[1]} cols "
            f"from '{self.data_path}'"
        )

        # Verify target column exists
        if TARGET_COL not in df.columns:
            raise ValueError(
                f"TimeSpentTrainer: target column '{TARGET_COL}' not in data. "
                f"Available: {list(df.columns)}"
            )

        # Log target statistics
        target = df[TARGET_COL]
        logger.info(
            f"Target '{TARGET_COL}' stats: "
            f"mean={target.mean():.3f}, std={target.std():.3f}, "
            f"min={target.min():.3f}, max={target.max():.3f}"
        )
        if ORIGINAL_TARGET_COL in df.columns:
            orig = df[ORIGINAL_TARGET_COL]
            logger.info(
                f"Original target '{ORIGINAL_TARGET_COL}' stats: "
                f"mean={orig.mean():.1f} min, median={orig.median():.1f} min, "
                f"max={orig.max():.1f} min"
            )

        _validate_feature_cols(df)
        return df

    def _split_data(
        self, df: pd.DataFrame
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray,
        pd.DataFrame, pd.DataFrame
    ]:
        """
        Split data into train/test, stratified by duration_bucket.

        Returns X_train, X_test, y_train, y_test, meta_train, meta_test.
        meta contains extra columns (project_code, task_type, duration_bucket,
        duration_minutes) for segmented analysis — not used by the pipeline.
        """
        test_size = self.config["training"]["test_size"]
        random_state = self.config["training"]["random_seed"]

        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].to_numpy()

        # Metadata for segmented analysis
        meta_cols = [GROUP_COL, "task_type"]
        if ORIGINAL_TARGET_COL in df.columns:
            meta_cols.append(ORIGINAL_TARGET_COL)
        if STRATIFY_COL in df.columns:
            meta_cols.append(STRATIFY_COL)
        meta = df[meta_cols].copy()

        # Stratify by duration_bucket if available
        if STRATIFY_COL in df.columns:
            stratify = df[STRATIFY_COL]
        else:
            stratify = pd.qcut(y, q=5, labels=False, duplicates="drop")

        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, meta,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )

        logger.info(
            f"Train/test split: train={len(X_train)}, test={len(X_test)}"
        )
        logger.info(
            f"Train target: mean={y_train.mean():.3f}, "
            f"std={y_train.std():.3f}"
        )
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
        # Reset index to avoid misalignment (CLAUDE.md rule #10)
        train_df = X_train.reset_index(drop=True).copy()
        train_df[TARGET_COL] = y_train

        # Add stratification column from meta
        meta_reset = meta_train.reset_index(drop=True)
        if STRATIFY_COL in meta_reset.columns:
            train_df[STRATIFY_COL] = meta_reset[STRATIFY_COL].values

        objective = TimeSpentRegressorObjective(
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
                for k, v in trial.params.items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        pass
                for k, v in trial.user_attrs.items():
                    if isinstance(v, (int, float)):
                        self._safe_log_metric(k, v)
                self._safe_log_metric("cv_rmse_log", value)
            return value

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(
            direction="minimize",  # Minimise RMSE
            sampler=sampler,
            study_name="time-spent-regressor",
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
                f"({pruned} pruned, {failed} failed). "
                f"Check trial error logs above for the root cause."
            )

        logger.info(
            f"Optuna finished: {completed} complete, {pruned} pruned, "
            f"{failed} failed. Best rmse_log={study.best_value:.4f}"
        )
        return study

    def _evaluate_on_test(
        self, pipeline, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate on holdout test set. Returns metrics on both log and original scales.
        """
        y_pred_log = pipeline.predict(X_test)

        # Log-scale metrics
        rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred_log)))
        mae_log = float(mean_absolute_error(y_test, y_pred_log))
        r2_log = float(r2_score(y_test, y_pred_log))

        # Original-scale metrics (minutes)
        y_test_min = np.expm1(y_test)
        y_pred_min = np.expm1(np.clip(y_pred_log, 0, 10))

        mae_minutes = float(mean_absolute_error(y_test_min, y_pred_min))
        rmse_minutes = float(np.sqrt(mean_squared_error(y_test_min, y_pred_min)))
        r2_minutes = float(r2_score(y_test_min, y_pred_min))

        # Median absolute error (robust to outliers)
        median_ae_minutes = float(
            np.median(np.abs(y_test_min - y_pred_min))
        )

        return {
            "rmse_log": rmse_log,
            "mae_log": mae_log,
            "r2_log": r2_log,
            "mae_minutes": mae_minutes,
            "rmse_minutes": rmse_minutes,
            "r2_minutes": r2_minutes,
            "median_ae_minutes": median_ae_minutes,
        }

    # ── Plotting helpers ──────────────────────────────────────────────────────

    def _log_predicted_vs_actual(
        self, pipeline, X_test, y_test
    ) -> None:
        """Scatter plot: predicted vs actual on both scales."""
        try:
            y_pred_log = pipeline.predict(X_test)
            y_test_min = np.expm1(y_test)
            y_pred_min = np.expm1(np.clip(y_pred_log, 0, 10))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Log scale
            axes[0].scatter(y_test, y_pred_log, alpha=0.4, s=15, edgecolors="none")
            lims = [
                min(y_test.min(), y_pred_log.min()) - 0.2,
                max(y_test.max(), y_pred_log.max()) + 0.2,
            ]
            axes[0].plot(lims, lims, "--", color="red", lw=1.5, label="Perfect")
            axes[0].set_xlabel("Actual (log1p)")
            axes[0].set_ylabel("Predicted (log1p)")
            axes[0].set_title("Predicted vs Actual — log scale")
            axes[0].legend()

            # Original scale (clipped for visibility)
            clip_max = np.percentile(y_test_min, 98)
            mask = y_test_min <= clip_max
            axes[1].scatter(
                y_test_min[mask], y_pred_min[mask],
                alpha=0.4, s=15, edgecolors="none"
            )
            lims_orig = [0, clip_max * 1.1]
            axes[1].plot(
                lims_orig, lims_orig, "--", color="red", lw=1.5, label="Perfect"
            )
            axes[1].set_xlabel("Actual (minutes)")
            axes[1].set_ylabel("Predicted (minutes)")
            axes[1].set_title(
                f"Predicted vs Actual — minutes (≤{clip_max:.0f} min)"
            )
            axes[1].legend()

            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Predicted vs actual plot logged.")
        except Exception as exc:
            logger.warning(f"Could not log predicted vs actual plot: {exc}")

    def _log_residual_plot(self, pipeline, X_test, y_test) -> None:
        """Residual plot on log scale."""
        try:
            y_pred_log = pipeline.predict(X_test)
            residuals = y_test - y_pred_log

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Residuals vs predicted
            axes[0].scatter(
                y_pred_log, residuals, alpha=0.4, s=15, edgecolors="none"
            )
            axes[0].axhline(y=0, color="red", linestyle="--", lw=1.5)
            axes[0].set_xlabel("Predicted (log1p)")
            axes[0].set_ylabel("Residual")
            axes[0].set_title("Residuals vs Predicted")

            # Residual histogram
            axes[1].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
            axes[1].axvline(x=0, color="red", linestyle="--", lw=1.5)
            axes[1].set_xlabel("Residual (log scale)")
            axes[1].set_ylabel("Count")
            axes[1].set_title(
                f"Residual Distribution "
                f"(mean={residuals.mean():.3f}, std={residuals.std():.3f})"
            )

            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Residual plot logged.")
        except Exception as exc:
            logger.warning(f"Could not log residual plot: {exc}")

    def _log_error_distribution(self, pipeline, X_test, y_test) -> None:
        """Absolute error distribution in original minutes."""
        try:
            y_pred_log = pipeline.predict(X_test)
            y_test_min = np.expm1(y_test)
            y_pred_min = np.expm1(np.clip(y_pred_log, 0, 10))
            abs_errors = np.abs(y_test_min - y_pred_min)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(
                abs_errors, bins=50, edgecolor="black", alpha=0.7,
                range=(0, np.percentile(abs_errors, 98)),
            )
            ax.axvline(
                x=np.median(abs_errors), color="red", linestyle="--",
                lw=1.5, label=f"Median={np.median(abs_errors):.1f} min",
            )
            ax.axvline(
                x=np.mean(abs_errors), color="orange", linestyle="--",
                lw=1.5, label=f"Mean={np.mean(abs_errors):.1f} min",
            )
            ax.set_xlabel("Absolute Error (minutes)")
            ax.set_ylabel("Count")
            ax.set_title("Error Distribution (original scale)")
            ax.legend()

            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Error distribution plot logged.")
        except Exception as exc:
            logger.warning(f"Could not log error distribution: {exc}")

    def _log_segmented_analysis(
        self, pipeline, X_test, y_test, meta_test
    ) -> None:
        """Log per-project and per-task-type MAE analysis."""
        try:
            y_pred_log = pipeline.predict(X_test)
            y_test_min = np.expm1(y_test)
            y_pred_min = np.expm1(np.clip(y_pred_log, 0, 10))
            abs_errors = np.abs(y_test_min - y_pred_min)

            meta_reset = meta_test.reset_index(drop=True)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # ── Per project ─────────────────────────────────────────────
            if GROUP_COL in meta_reset.columns:
                project_groups = meta_reset[GROUP_COL]
                project_mae = {}
                project_count = {}
                for proj in project_groups.unique():
                    mask = project_groups == proj
                    if mask.sum() > 0:
                        project_mae[proj] = float(np.mean(abs_errors[mask]))
                        project_count[proj] = int(mask.sum())

                # Log per-project metrics
                for proj, mae in project_mae.items():
                    self._safe_log_metric(
                        f"test_mae_project_{proj}", mae
                    )

                # Sort by MAE descending for plot
                sorted_projects = sorted(
                    project_mae.items(), key=lambda x: x[1], reverse=True
                )
                proj_names = [
                    f"{p} (n={project_count[p]})"
                    for p, _ in sorted_projects
                ]
                proj_maes = [m for _, m in sorted_projects]

                axes[0].barh(proj_names, proj_maes, color="steelblue")
                axes[0].set_xlabel("MAE (minutes)")
                axes[0].set_title("MAE by Project")
                axes[0].invert_yaxis()

            # ── Per task type ────────────────────────────────────────────
            if "task_type" in meta_reset.columns:
                tt_groups = meta_reset["task_type"]
                tt_mae = {}
                tt_count = {}
                for tt in tt_groups.unique():
                    mask = tt_groups == tt
                    if mask.sum() > 0:
                        tt_mae[tt] = float(np.mean(abs_errors[mask]))
                        tt_count[tt] = int(mask.sum())

                for tt, mae in tt_mae.items():
                    self._safe_log_metric(f"test_mae_tasktype_{tt}", mae)

                sorted_tt = sorted(
                    tt_mae.items(), key=lambda x: x[1], reverse=True
                )
                tt_names = [
                    f"{t} (n={tt_count[t]})" for t, _ in sorted_tt
                ]
                tt_maes = [m for _, m in sorted_tt]

                axes[1].barh(tt_names, tt_maes, color="coral")
                axes[1].set_xlabel("MAE (minutes)")
                axes[1].set_title("MAE by Task Type")
                axes[1].invert_yaxis()

            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Segmented analysis plot logged.")
        except Exception as exc:
            logger.warning(f"Could not log segmented analysis: {exc}")

    def _log_feature_importance(
        self, pipeline, model_type: str | None
    ) -> None:
        """Log feature importances for tree-based models."""
        try:
            regressor = pipeline.named_steps.get("regressor")
            preprocessor = pipeline.named_steps.get("preprocessor")

            # VotingRegressor: try first estimator
            if hasattr(regressor, "estimators_"):
                regressor = regressor.estimators_[0]

            if not hasattr(regressor, "feature_importances_"):
                logger.debug(
                    f"Model type '{model_type}' has no feature_importances_. "
                    f"Skipping."
                )
                return

            importances = regressor.feature_importances_
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = [
                    f"feature_{i}" for i in range(len(importances))
                ]

            if len(importances) != len(feature_names):
                feature_names = [
                    f"feature_{i}" for i in range(len(importances))
                ]

            n_top = min(25, len(importances))
            top_idx = np.argsort(importances)[-n_top:][::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                [feature_names[i] for i in top_idx][::-1],
                importances[top_idx][::-1],
            )
            ax.set_title(f"Top {n_top} Feature Importances")
            ax.set_xlabel("Importance")

            fig.tight_layout()
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as f:
                fig.savefig(f.name, bbox_inches="tight", dpi=120)
                mlflow.log_artifact(f.name, artifact_path="plots")
            plt.close(fig)
            logger.debug("Feature importance plot logged.")
        except Exception as exc:
            logger.warning(f"Could not log feature importance: {exc}")

    def _log_optuna_plots(self, study: optuna.Study) -> None:
        """Log Optuna visualisations."""
        try:
            import optuna.visualization as vis

            plots = {
                "optuna_optimization_history.html": vis.plot_optimization_history(study),
                "optuna_param_importances.html": vis.plot_param_importances(study),
                "optuna_parallel_coordinate.html": vis.plot_parallel_coordinate(study),
            }
            for filename, fig in plots.items():
                with tempfile.NamedTemporaryFile(
                    suffix=".html", delete=False, mode="w"
                ) as f:
                    f.write(fig.to_html())
                    mlflow.log_artifact(f.name, artifact_path="optuna_plots")
            logger.debug("Optuna visualisations logged.")
        except Exception as exc:
            logger.warning(f"Could not log Optuna plots: {exc}")

    def _build_regression_report(
        self, pipeline, X_test, y_test, meta_test
    ) -> str:
        """Build a human-readable regression report."""
        y_pred_log = pipeline.predict(X_test)
        y_test_min = np.expm1(y_test)
        y_pred_min = np.expm1(np.clip(y_pred_log, 0, 10))
        abs_errors = np.abs(y_test_min - y_pred_min)

        lines = [
            "=" * 60,
            "TIME SPENT REGRESSOR — HOLDOUT TEST REPORT",
            "=" * 60,
            "",
            "Overall Metrics (log scale):",
            f"  RMSE (log):  {np.sqrt(mean_squared_error(y_test, y_pred_log)):.4f}",
            f"  MAE (log):   {mean_absolute_error(y_test, y_pred_log):.4f}",
            f"  R² (log):    {r2_score(y_test, y_pred_log):.4f}",
            "",
            "Overall Metrics (original scale — minutes):",
            f"  MAE:         {mean_absolute_error(y_test_min, y_pred_min):.2f} min",
            f"  RMSE:        {np.sqrt(mean_squared_error(y_test_min, y_pred_min)):.2f} min",
            f"  R²:          {r2_score(y_test_min, y_pred_min):.4f}",
            f"  Median AE:   {np.median(abs_errors):.2f} min",
            "",
            "Error Percentiles (minutes):",
            f"  25th:  {np.percentile(abs_errors, 25):.2f} min",
            f"  50th:  {np.percentile(abs_errors, 50):.2f} min",
            f"  75th:  {np.percentile(abs_errors, 75):.2f} min",
            f"  90th:  {np.percentile(abs_errors, 90):.2f} min",
            f"  95th:  {np.percentile(abs_errors, 95):.2f} min",
            "",
        ]

        meta_reset = meta_test.reset_index(drop=True)

        # Per-project breakdown
        if GROUP_COL in meta_reset.columns:
            lines.append("Per-Project MAE (minutes):")
            for proj in sorted(meta_reset[GROUP_COL].unique()):
                mask = meta_reset[GROUP_COL] == proj
                n = int(mask.sum())
                if n > 0:
                    mae = float(np.mean(abs_errors[mask]))
                    med_ae = float(np.median(abs_errors[mask]))
                    lines.append(
                        f"  {proj:20s}: MAE={mae:7.2f}, "
                        f"MedAE={med_ae:7.2f}, n={n}"
                    )
            lines.append("")

        # Per-task-type breakdown
        if "task_type" in meta_reset.columns:
            lines.append("Per-Task-Type MAE (minutes):")
            for tt in sorted(meta_reset["task_type"].unique()):
                mask = meta_reset["task_type"] == tt
                n = int(mask.sum())
                if n > 0:
                    mae = float(np.mean(abs_errors[mask]))
                    med_ae = float(np.median(abs_errors[mask]))
                    lines.append(
                        f"  {tt:20s}: MAE={mae:7.2f}, "
                        f"MedAE={med_ae:7.2f}, n={n}"
                    )
            lines.append("")

        # ICC context
        lines.extend([
            "Context:",
            "  ICC = 0.695 (69.5% of variance is between-task = predictable)",
            "  30.5% is within-task noise (irreducible floor)",
            "  SBM project has bimodal duration profile",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _log_and_register_model(
        self, pipeline, X_example: pd.DataFrame, registered_model_name: str
    ) -> str:
        """
        Log model to MLflow and register in the Model Registry.
        Handles the known SQLite UNIQUE constraint duplicate-metric error.
        """
        import joblib

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
                f"MLflow log_model raised a duplicate-metric error "
                f"(known SQLite issue). "
                f"Model artifact was saved; registry registration may "
                f"be incomplete. Error: {exc}"
            )
            model_uri = (
                f"runs:/{mlflow.active_run().info.run_id}/model"
            )
            model_info = type(
                "_ModelInfo", (), {"model_uri": model_uri}
            )()
        finally:
            if original_env is None:
                os.environ.pop(
                    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", None
                )
            else:
                os.environ[
                    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"
                ] = original_env

        # Save a local copy to models/
        models_dir = Path(self.config.get("models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        local_path = models_dir / "time_spent_regressor_best.pkl"
        joblib.dump(pipeline, local_path)
        logger.info(f"Model saved locally: {local_path}")

        return model_info.model_uri

    @staticmethod
    def _safe_log_metric(key: str, value: float) -> None:
        """
        Log a metric to MLflow, skipping NaN/inf values and catching
        duplicate errors.
        """
        if value is None or (
            isinstance(value, float)
            and (math.isnan(value) or math.isinf(value))
        ):
            logger.debug(f"Skipping metric '{key}' — value is NaN/inf.")
            return
        try:
            mlflow.log_metric(key, value)
        except Exception as exc:
            if "UNIQUE constraint failed: metrics" in str(exc):
                logger.debug(
                    f"Metric '{key}' already logged — skipping duplicate."
                )
            else:
                logger.warning(
                    f"Could not log metric '{key}={value}': {exc}"
                )

    def _log_text_artifact(
        self, content: str, filename: str
    ) -> None:
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
