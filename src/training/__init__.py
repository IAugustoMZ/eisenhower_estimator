"""
src.training — training orchestration layer.

Exports
-------
ModelTrainer                  — Model 1 end-to-end training with Optuna + MLflow
UrgentModelTrainer            — Model 2 end-to-end training with Optuna + MLflow
TimeSpentTrainer              — Model 3 end-to-end training with Optuna + MLflow
TimeBucketTrainer             — Model 3b end-to-end training with Optuna + MLflow
ImportantClassifierObjective  — Optuna objective for Model 1 (reusable)
UrgentClassifierObjective     — Optuna objective for Model 2 (reusable)
TimeSpentRegressorObjective   — Optuna objective for Model 3 (reusable)
TimeBucketClassifierObjective — Optuna objective for Model 3b (reusable)
build_full_pipeline           — assembles ColumnTransformer + classifier (Model 1)
build_full_pipeline_urgent    — assembles ColumnTransformer + classifier (Model 2)
build_full_pipeline_time_spent  — assembles ColumnTransformer + regressor (Model 3)
build_full_pipeline_time_bucket — assembles ColumnTransformer + classifier (Model 3b)
BaseObjective                 — abstract base class for all Optuna objectives
"""
from .trainer import ModelTrainer
from .urgent_trainer import UrgentModelTrainer
from .time_spent_trainer import TimeSpentTrainer
from .time_bucket_trainer import TimeBucketTrainer
from .optuna_objective import ImportantClassifierObjective
from .optuna_objective_urgent import UrgentClassifierObjective
from .optuna_objective_time_spent import TimeSpentRegressorObjective
from .optuna_objective_time_bucket import TimeBucketClassifierObjective
from .pipeline_builder import build_full_pipeline
from .pipeline_builder_urgent import build_full_pipeline_urgent
from .pipeline_builder_time_spent import build_full_pipeline_time_spent
from .pipeline_builder_time_bucket import build_full_pipeline_time_bucket
from .base_objective import BaseObjective

__all__ = [
    "ModelTrainer",
    "UrgentModelTrainer",
    "TimeSpentTrainer",
    "TimeBucketTrainer",
    "ImportantClassifierObjective",
    "UrgentClassifierObjective",
    "TimeSpentRegressorObjective",
    "TimeBucketClassifierObjective",
    "build_full_pipeline",
    "build_full_pipeline_urgent",
    "build_full_pipeline_time_spent",
    "build_full_pipeline_time_bucket",
    "BaseObjective",
]
