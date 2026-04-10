"""
RuleBasedClassifier — pure rule engine for the 'important' label.

Design rationale
----------------
Rules capture domain knowledge that is hard for ML to learn from sparse signals
(e.g. "tasks created at 6 AM on weekdays are always important" derives from a
single user's personal habits). They also act as a useful ablation baseline:
if ML cannot beat rules, something is wrong.

Rules are encoded as a priority-ordered list of (predicate, label) tuples.
Each predicate receives a single-row pandas Series and returns True/False.
The first matching rule wins; if no rule fires the classifier returns None
(which the HybridClassifier passes to the ML model).

To add a rule:
    1. Write a predicate function (row: pd.Series) -> bool
    2. Register it via RuleBasedClassifier.add_rule(predicate, label, name)

All rule decisions are recorded in rule_trace so you can audit which rule
fired on each sample — critical for debugging and explaining predictions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Canonical label values
IMPORTANT = 1
NOT_IMPORTANT = 0
ABSTAIN = None  # rule did not fire — pass to downstream model


@dataclass
class Rule:
    """A named, ordered rule entry."""
    name: str
    predicate: Callable[[pd.Series], bool]
    label: Optional[int]   # IMPORTANT=1, NOT_IMPORTANT=0, ABSTAIN=None


class RuleBasedClassifier:
    """
    Priority-ordered rule engine for the 'important' binary classifier.

    Rules are evaluated in insertion order. The first rule whose predicate
    returns True determines the output. If no rule fires, predict() returns
    ABSTAIN (None) per row, and predict_hard() returns a configurable default.

    Parameters
    ----------
    default_label : int, optional
        Label to use when no rule fires in predict_hard(). Default = IMPORTANT (1)
        because the class prior is 75% important.
    """

    def __init__(self, default_label: int = IMPORTANT) -> None:
        self._rules: list[Rule] = []
        self.default_label = default_label
        self._register_default_rules()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_rule(
        self,
        predicate: Callable[[pd.Series], bool],
        label: Optional[int],
        name: str = "",
    ) -> "RuleBasedClassifier":
        """
        Append a rule to the end of the priority list.

        Parameters
        ----------
        predicate : Callable[[pd.Series], bool]
            Function receiving one row as a Series; returns True when rule fires.
        label     : int | None
            Predicted label when rule fires (IMPORTANT=1, NOT_IMPORTANT=0).
            Pass None to explicitly abstain (rarely needed; just don't add a rule).
        name      : str
            Human-readable rule name for audit/logging.
        """
        self._rules.append(Rule(name=name or f"rule_{len(self._rules)}", predicate=predicate, label=label))
        return self  # chainable

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return per-row predictions as int array.
        Rows where no rule fires use self.default_label.
        """
        labels, _ = self._apply_rules(X)
        return np.array(
            [lbl if lbl is not None else self.default_label for lbl in labels],
            dtype=int,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return soft probability array of shape (n, 2).
        Rule-fired rows get probability 1.0 for the decided class.
        Abstained rows get the class prior (0.25, 0.75).
        """
        labels, _ = self._apply_rules(X)
        proba = np.zeros((len(labels), 2), dtype=float)
        for i, lbl in enumerate(labels):
            if lbl == NOT_IMPORTANT:
                proba[i] = [1.0, 0.0]
            elif lbl == IMPORTANT:
                proba[i] = [0.0, 1.0]
            else:
                # Abstained: return class prior (75% important based on EDA)
                proba[i] = [0.25, 0.75]
        return proba

    def predict_with_trace(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Return (predictions, rule_trace) where rule_trace[i] is the name of
        the rule that fired on row i, or 'default' if none fired.
        """
        labels, traces = self._apply_rules(X)
        preds = np.array(
            [lbl if lbl is not None else self.default_label for lbl in labels],
            dtype=int,
        )
        final_traces = [t if t else "default" for t in traces]
        return preds, final_traces

    def coverage(self, X: pd.DataFrame) -> dict[str, float]:
        """
        Return rule coverage statistics:
        - fired_ratio: fraction of rows where at least one rule fired
        - per_rule_coverage: fraction of rows each rule fired on (first-match only)
        """
        _, traces = self._apply_rules(X)
        n = len(X)
        fired = [t for t in traces if t]
        counts: dict[str, int] = {}
        for name in fired:
            counts[name] = counts.get(name, 0) + 1
        return {
            "fired_ratio": len(fired) / n if n > 0 else 0.0,
            "per_rule_coverage": {k: v / n for k, v in counts.items()},
        }

    def list_rules(self) -> list[dict]:
        """Return human-readable list of registered rules."""
        return [{"priority": i, "name": r.name, "label": r.label} for i, r in enumerate(self._rules)]

    # ── Private ───────────────────────────────────────────────────────────────

    def _apply_rules(self, X: pd.DataFrame) -> tuple[list[Optional[int]], list[str]]:
        """
        Apply all rules row-by-row. Returns (labels, rule_names) parallel lists.
        """
        labels: list[Optional[int]] = []
        traces: list[str] = []
        for _, row in X.iterrows():
            matched_label: Optional[int] = None
            matched_name: str = ""
            for rule in self._rules:
                try:
                    if rule.predicate(row):
                        matched_label = rule.label
                        matched_name = rule.name
                        break
                except Exception as exc:
                    logger.warning(f"Rule '{rule.name}' raised an exception on row: {exc}. Skipping.")
            labels.append(matched_label)
            traces.append(matched_name)
        return labels, traces

    def _register_default_rules(self) -> None:
        """
        Register the domain-knowledge rules derived from EDA findings.

        Rule ordering follows effect-size priority (strongest signal first).
        Each rule must be conservative — only fire when confidence is high.

        EDA-derived signals used here:
          - hour_created (Cramér's V=0.80): tasks at 6-8 AM on weekdays = almost always important
          - project_category: Personal tasks are 70% important; Work tasks 80%+ important
          - desc_word_count: very long descriptions (>30 words) lean important
          - days_until_due: tasks due within 24h tend to be important
        """

        # ── Rule 1: Early-morning Work tasks ─────────────────────────────────
        # EDA: hour_created is the strongest predictor (V=0.80). Tasks created
        # at 5-8 AM on weekdays are almost exclusively important.
        def _early_morning_work(row: pd.Series) -> bool:
            hour = row.get("hour_created", -1)
            dow = row.get("day_of_week_created", -1)
            category = row.get("project_category", "")
            return (
                5 <= hour <= 8
                and 0 <= dow <= 4  # Monday–Friday
                and str(category).lower() == "work"
            )

        self.add_rule(_early_morning_work, IMPORTANT, "early_morning_work_task")

        # ── Rule 2: Late-night tasks (non-important signal) ───────────────────
        # EDA: tasks created between 22:00 and 03:00 are overwhelmingly not important
        # (personal reminders, newsletter subscriptions, etc.)
        def _late_night_personal(row: pd.Series) -> bool:
            hour = row.get("hour_created", -1)
            category = row.get("project_category", "")
            return (
                (hour >= 22 or hour <= 3)
                and str(category).lower() == "personal"
            )

        self.add_rule(_late_night_personal, NOT_IMPORTANT, "late_night_personal_task")

        # ── Rule 3: Very short personal descriptions (noise/reminders) ───────
        # EDA: desc_word_count r≈0.08; single-word personal tasks (e.g. "newsletter")
        # are strongly not-important based on TF-IDF analysis.
        def _single_word_personal(row: pd.Series) -> bool:
            word_count = row.get("desc_word_count", 99)
            category = row.get("project_category", "")
            return (
                word_count <= 1
                and str(category).lower() == "personal"
            )

        self.add_rule(_single_word_personal, NOT_IMPORTANT, "single_word_personal_task")

        # ── Rule 4: Imminent due date — high-importance signal ────────────────
        # Tasks due in 0 days (due today) are very likely important regardless
        # of other features. Conservative: only fire for Work tasks.
        def _due_today_work(row: pd.Series) -> bool:
            days = row.get("days_until_due", 999)
            category = row.get("project_category", "")
            return (
                0 <= days <= 0
                and str(category).lower() == "work"
            )

        self.add_rule(_due_today_work, IMPORTANT, "due_today_work_task")

        logger.debug(f"RuleBasedClassifier: registered {len(self._rules)} default rules.")
