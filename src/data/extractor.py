"""
Data extractor for Eisenhower Estimator.

Reads todo_task data from the SQLite database (timesheet_appointment project)
and saves it as a raw Parquet file for downstream ML processing.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML config. Raises FileNotFoundError with a clear message if missing."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Make sure you are running this script from the project root."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Config loaded from '{config_path}'.")
    return config


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection. Raises FileNotFoundError if the DB is missing."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at '{db_path}'. "
            "Check the 'database.path' entry in configs/config.yaml."
        )
    conn = sqlite3.connect(str(db_path))
    logger.info(f"Connected to database at '{db_path}'.")
    return conn


def _validate_tables(conn: sqlite3.Connection, required: list[str]) -> None:
    """Raise ValueError if any required table is absent from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row[0] for row in cursor.fetchall()}
    missing = [t for t in required if t not in existing]
    if missing:
        raise ValueError(
            f"Required table(s) not found in the database: {missing}. "
            f"Available tables: {sorted(existing)}"
        )
    logger.debug(f"Tables validated: {required}")


def _validate_columns(conn: sqlite3.Connection, table: str, required: list[str]) -> None:
    """Raise ValueError if any required column is absent from the table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    missing = [c for c in required if c not in existing]
    if missing:
        raise ValueError(
            f"Required column(s) not found in '{table}': {missing}. "
            f"Available columns: {sorted(existing)}"
        )
    logger.debug(f"Columns validated for '{table}': {required}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(config: dict) -> pd.DataFrame:
    """
    Query todo_task joined with project, return a DataFrame with all
    feature and label columns defined in the config.
    """
    db_path = config["database"]["path"]
    todo_table = config["extraction"]["table"]
    project_table = config["extraction"]["project_table"]
    feature_cols = config["extraction"]["feature_columns"]
    label_cols = config["extraction"]["label_columns"]
    required_cols = feature_cols + label_cols

    conn = _get_connection(db_path)

    try:
        _validate_tables(conn, [todo_table, project_table])
        _validate_columns(conn, todo_table, required_cols)

        query = f"""
            SELECT
                t.id,
                t.description,
                t.comments,
                t.project_id,
                p.code  AS project_code,
                p.name  AS project_name,
                t.due_date,
                t.created_at,
                t.updated_at,
                t.register_timesheet,
                t.completed_date,
                t.important,
                t.urgent
            FROM {todo_table} t
            LEFT JOIN {project_table} p ON t.project_id = p.id
        """

        df = pd.read_sql_query(query, conn)
        logger.info(f"Extracted {len(df):,} rows from '{todo_table}'.")
        return df

    finally:
        conn.close()
        logger.debug("Database connection closed.")


# ---------------------------------------------------------------------------
# Validation & audit
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, label_cols: list[str]) -> None:
    """
    Perform basic data quality checks and log a clear audit report.
    Raises ValueError on critical issues (missing labels).
    """
    logger.info("--- Data Quality Audit ---")

    # Shape
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Null counts
    null_counts = df.isnull().sum()
    null_report = null_counts[null_counts > 0]
    if null_report.empty:
        logger.info("Nulls: none detected.")
    else:
        logger.warning(f"Null values detected:\n{null_report.to_string()}")

    # Label integrity — labels must never be null
    for col in label_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            raise ValueError(
                f"Label column '{col}' has {n_null} null value(s). "
                "All tasks must have labels before training."
            )

    # Label distribution
    for col in label_cols:
        dist = df[col].value_counts().sort_index()
        pct = (dist / len(df) * 100).round(1)
        logger.info(f"Label '{col}' distribution:\n{pd.concat([dist, pct], axis=1, keys=['count', '%']).to_string()}")

    logger.info("--- End Audit ---")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, config: dict) -> Path:
    """
    Save the DataFrame as a Parquet file to data/raw/.
    Returns the path of the saved file.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / config["data"]["raw_filename"]
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.success(f"Raw data saved to '{output_path}' ({output_path.stat().st_size / 1024:.1f} KB).")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: str | Path = "configs/config.yaml") -> None:
    """Full extraction pipeline: load config → extract → validate → save."""
    logger.info("=== Eisenhower Estimator — Data Extraction ===")

    try:
        config = load_config(config_path)
        df = extract(config)
        validate_dataframe(df, config["extraction"]["label_columns"])
        output_path = save_parquet(df, config)
        logger.success(f"Extraction complete. Output: '{output_path}'")

    except FileNotFoundError as e:
        logger.error(f"[FileNotFoundError] {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"[ValidationError] {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"[UnexpectedError] {e}")
        sys.exit(1)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    run(config_path)
