"""
Evaluation package — comparable metrics across rule-based, ML, and hybrid classifiers.
"""
from src.evaluation.rule_based import RuleBasedClassifier
from src.evaluation.hybrid import HybridClassifier
from src.evaluation.evaluator import ModelEvaluator

__all__ = ["RuleBasedClassifier", "HybridClassifier", "ModelEvaluator"]
