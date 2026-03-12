# Agent Skills — Eisenhower Estimator

This file describes specialized agent roles for this MLOps project.
Each skill maps to a stage in the ML pipeline.

---

## eda-analyst

**Trigger**: User asks to explore data, understand feature distributions, or run exploratory analysis.

**Responsibilities**:
- Load raw parquet from `data/raw/`
- Engineer temporal, categorical, and text features
- Run statistical tests (Chi2, Mann-Whitney, point-biserial, Cramér's V)
- Produce visualizations saved to `docs/figures/`
- Save processed data to `data/processed/` as parquet
- Write conclusions to `docs/eda_<target>_report.md`

**Key files**:
- `notebooks/eda_important.py` — EDA for the `important` target
- `notebooks/eda_urgent.py` — EDA for the `urgent` target
- `docs/eda_important_report.md` — conclusions and feature recommendations
- `docs/eda_urgent_report.md` — conclusions and feature recommendations
- `data/processed/eda_important_features.parquet`
- `data/processed/eda_urgent_features.parquet`

---

## feature-engineer

**Trigger**: User asks to build, refactor, or extend the feature engineering pipeline.

**Responsibilities**:
- Implement or extend reusable sklearn transformers in `src/transformers/`
- All transformers must extend `TransformerMixin` and `BaseEstimator`
- Must implement: `fit`, `transform`, `get_feature_names_out`
- Ensure fit-only-on-train semantics (no leakage)
- `SafeTargetEncoder`: fit per fold, not globally

**Key files**:
- `src/transformers/cyclical_encoder.py` — sin/cos for hour, day_of_week
- `src/transformers/target_encoder.py` — target encoding with unseen-category fallback
- `src/transformers/text_vectorizer.py` — TF-IDF / BoW + optional LSA
- `src/transformers/scaler_selector.py` — configurable scaler
- `src/transformers/resampler.py` — SMOTE / ADASYN / passthrough (outside pipeline)

**Leakage rules**:
- `is_overdue` MUST NOT be used in Model 1 (not available at inference time)
- `urgent` MUST NOT be a feature in Model 1 (target for Model 2)
- `is_overdue` MUST NOT be used in Model 2 either (not available at task creation)
- `important` MUST NOT be a feature in Model 2

---

## model-trainer

**Trigger**: User asks to train, tune, or run an experiment for a classification model.

**Responsibilities**:
- Load processed parquet from `data/processed/`
- Build `ColumnTransformer` + classifier `Pipeline` via `PipelineBuilder`
- Run Optuna study with MLflow nested run logging
- Each trial: StratifiedKFold (k=5) CV, resampling ONLY on train folds
- Supported models: LightGBM, XGBoost, RandomForest, LogisticRegression, VotingClassifier
- Primary metric: F1-macro. Secondary: F1-class-0, F1-class-1, ROC-AUC
- Retrain best pipeline on full train set
- Evaluate on holdout test set (stratified split, never seen during Optuna)
- Register final model in MLflow Model Registry

**Entry points**:
- `notebooks/train_important_model.py` — Model 1 (important)
- `notebooks/train_urgent_model.py` — Model 2 (urgent, to be created)

**Key files**:
- `src/training/trainer.py` — `ModelTrainer` orchestrator
- `src/training/pipeline_builder.py` — `build_full_pipeline(config)`
- `src/training/optuna_objective.py` — `ImportantClassifierObjective`
- `configs/model1_config.yaml` — search space configuration

**How to run**:
```bash
source eisen_mat_env/Scripts/activate
python notebooks/train_important_model.py
```

---

## model-evaluator

**Trigger**: User asks to evaluate a trained model, generate explanations, or compare model versions.

**Responsibilities**:
- Load registered model from MLflow Model Registry:
  `mlflow.sklearn.load_model("models:/important-classifier/Production")`
- Compute: classification report, confusion matrix, ROC-AUC, PR-AUC
- Generate SHAP feature importance plots (for tree-based models)
- Compare model versions via MLflow run comparison
- Save evaluation artifacts to `docs/figures/` and log to MLflow

**Key artifacts in MLflow**:
- `plots/` — confusion matrix, ROC curve, feature importances
- `optuna_plots/` — optimization history, param importances, parallel coordinate
- `reports/classification_report.txt`
- `reports/best_hyperparameters.json`

**MLflow UI**:
```bash
mlflow ui --backend-store-uri mlruns --port 5000
# Open: http://localhost:5000
```

---

## data-extractor

**Trigger**: User asks to refresh the raw dataset from the SQLite source.

**Responsibilities**:
- Run `src/data/extractor.py` to pull from the personal organizer database
- Validate schema and label distributions
- Save output to `data/raw/todo_tasks.parquet`

**Key files**:
- `src/data/extractor.py`
- `configs/config.yaml`

---

## mlops-engineer

**Trigger**: User asks about MLflow setup, model registry, experiment tracking, or deployment.

**Responsibilities**:
- Configure MLflow tracking URI and experiment names in `configs/config.yaml`
- Manage model versions in the MLflow Model Registry
- Transition model stages: None → Staging → Production
- Load production model for inference:
  ```python
  import mlflow.sklearn
  model = mlflow.sklearn.load_model("models:/important-classifier/Production")
  predictions = model.predict(X_new)
  probabilities = model.predict_proba(X_new)[:, 1]
  ```
- Monitor model drift by comparing new data distributions to training data
- All models saved locally to `models/` as joblib `.pkl` files

**MLflow experiment structure**:
```
Experiment: "important-classifier"
Parent run : training_<timestamp>
  ↳ Child run: trial_0000
  ↳ Child run: trial_0001
  ...
  ↳ Child run: trial_N
Registered model: "important-classifier"
  ↳ Version 1 (Production)
  ↳ Version 2 (Staging)
```
