"""
src.training — training orchestration layer.

Exports
-------
ModelTrainer                 — Model 1 end-to-end training with Optuna + MLflow
UrgentModelTrainer           — Model 2 end-to-end training with Optuna + MLflow
ImportantClassifierObjective — Optuna objective for Model 1 (reusable)
UrgentClassifierObjective    — Optuna objective for Model 2 (reusable)
build_full_pipeline          — assembles ColumnTransformer + classifier (Model 1)
build_full_pipeline_urgent   — assembles ColumnTransformer + classifier (Model 2)
BaseObjective                — abstract base class for all Optuna objectives
"""
from .trainer import ModelTrainer
from .urgent_trainer import UrgentModelTrainer
from .optuna_objective import ImportantClassifierObjective
from .optuna_objective_urgent import UrgentClassifierObjective
from .pipeline_builder import build_full_pipeline
from .pipeline_builder_urgent import build_full_pipeline_urgent
from .base_objective import BaseObjective

__all__ = [
    "ModelTrainer",
    "UrgentModelTrainer",
    "ImportantClassifierObjective",
    "UrgentClassifierObjective",
    "build_full_pipeline",
    "build_full_pipeline_urgent",
    "BaseObjective",
]
