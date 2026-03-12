1. "Before writing any code, describe your approach and wait for approval. Always ask clarifying questions before writing any code if requirements are ambiguous."

2. "If a task requires changes to more than 3 files, stop and break it into smaller tasks first."

3. "After writing code, list what could break and suggest tests to cover it."

4. "When there’s a bug, start by writing a test that reproduces it, then fix it until the test passes."

5. "Every time I correct you, add a new rule to the CLAUDE .md file so it never happens again."

6. "Always use extensively error and exception handling, so the user has the highest clarity on what's going on. Always implement fallbacks to edge cases you identify"

7. "Use the best coding practices, clean architecture, design patterns and SOLID principles. Think in scalability, simplicity over sophistication and maaintainability"
8. "The backend uses Pydantic v2 (2.10.x). Always use v2 API: `field_validator` (not `validator`), `model_config = ConfigDict(from_attributes=True)` (not `Config.orm_mode`), `model_validator` (not `root_validator`). Never use Pydantic v1-only features."

9. "The custom `useForm` hook uses positional parameters: `useForm(initialValues, validationSchema, onSubmit, options)`. Never pass an object with named properties—always use positional arguments. Example: `useForm({ name: '' }, schema, async (values) => {...})`"

10. "When joining a numpy array (y) with a DataFrame (X) using pd.concat, always reset the DataFrame index first and assign the array as a column directly — never use pd.Series(numpy_array) in pd.concat along axis=1 with a non-contiguous-index DataFrame, as index misalignment silently introduces NaN."

11. "MLflow's SQLite backend raises a UNIQUE constraint error when the same metric key is logged twice in the same run (e.g. when mlflow.sklearn.log_model internally calls log_model_metrics_for_step on metrics already present). Always: (a) use a _safe_log_metric() helper that skips NaN/inf values and catches duplicate-key IntegrityErrors, and (b) set os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false' around mlflow.sklearn.log_model() calls to suppress auto-metric re-logging."
