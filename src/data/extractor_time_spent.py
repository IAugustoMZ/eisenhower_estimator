"""
Data extractor for the Time Spent ML model.

Reads timesheet_entry data from the SQLite database (timesheet_appointment project),
filters entries whose description starts with "To Do Completed:", joins with the
project table, and saves the result as a raw Parquet file.

Target column: duration_minutes (time spent on the to-do task).
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TODO_COMPLETED_PREFIX = "To Do Completed:"
REQUIRED_TABLES = ["timesheet_entry", "project"]
REQUIRED_ENTRY_COLUMNS = ["date", "project_id", "duration_minutes", "description"]
OUTPUT_FILENAME = "time_spent_tasks.parquet"


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
    """Open a read-only SQLite connection. Raises FileNotFoundError if DB is missing."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at '{db_path}'. "
            "Check the 'database.path' entry in configs/config.yaml."
        )
    # uri=True + ?mode=ro opens the database in read-only mode
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    logger.info(f"Connected (read-only) to database at '{db_path}'.")
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

def extract(db_path: str | Path) -> pd.DataFrame:
    """
    Query timesheet_entry JOIN project, filter to-do completed entries,
    and return a clean DataFrame ready for parquet persistence.
    """
    conn = _get_connection(db_path)

    try:
        _validate_tables(conn, REQUIRED_TABLES)
        _validate_columns(conn, "timesheet_entry", REQUIRED_ENTRY_COLUMNS)

        query = f"""
            SELECT
                t.date,
                t.description,
                t.project_id,
                p.code           AS project_code,
                p.name           AS project_name,
                t.duration_minutes
            FROM timesheet_entry t
            LEFT JOIN project p ON t.project_id = p.id
            WHERE t.description LIKE '{TODO_COMPLETED_PREFIX}%'
        """

        df = pd.read_sql_query(query, conn)
        logger.info(
            f"Extracted {len(df):,} rows matching prefix '{TODO_COMPLETED_PREFIX}'."
        )
        return df

    finally:
        conn.close()
        logger.debug("Database connection closed.")


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight transformations:
    - Parse `date` as datetime.
    - Derive `task_description` by stripping the 'To Do Completed:' prefix.
    - Drop rows where duration_minutes is null or non-positive (corrupt entries).
    """
    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isnull().sum()
    if n_bad_dates > 0:
        logger.warning(
            f"{n_bad_dates} row(s) had unparseable dates and will be dropped."
        )
        df = df.dropna(subset=["date"])

    # Derive clean task description
    df["task_description"] = (
        df["description"]
        .str.removeprefix(TODO_COMPLETED_PREFIX)
        .str.strip()
    )

    # Guard: drop rows with null or non-positive duration (target integrity)
    n_before = len(df)
    df = df[df["duration_minutes"].notnull() & (df["duration_minutes"] > 0)]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            f"{n_dropped} row(s) with null or non-positive duration_minutes were dropped."
        )

    # Reorder columns for readability
    df = df[
        ["date", "project_id", "project_code", "project_name",
         "description", "task_description", "duration_minutes"]
    ]

    logger.info(f"Transformation complete. Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation & audit
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """Log a data quality audit and raise on critical issues."""
    logger.info("--- Data Quality Audit ---")
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Null report
    null_counts = df.isnull().sum()
    null_report = null_counts[null_counts > 0]
    if null_report.empty:
        logger.info("Nulls: none detected.")
    else:
        logger.warning(f"Null values detected:\n{null_report.to_string()}")

    # Target integrity
    n_null_target = df["duration_minutes"].isnull().sum()
    if n_null_target > 0:
        raise ValueError(
            f"Target column 'duration_minutes' has {n_null_target} null value(s). "
            "All rows must have a valid target before training."
        )

    # Target distribution
    desc = df["duration_minutes"].describe()
    logger.info(
        f"Target 'duration_minutes' stats:\n"
        f"  count={desc['count']:.0f}  mean={desc['mean']:.1f}  "
        f"std={desc['std']:.1f}  min={desc['min']:.0f}  "
        f"p50={desc['50%']:.0f}  max={desc['max']:.0f}"
    )

    # Project distribution
    project_dist = df["project_code"].value_counts()
    logger.info(f"Project distribution (top 5):\n{project_dist.head(5).to_string()}")

    logger.info("--- End Audit ---")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, raw_dir: str | Path) -> Path:
    """Save DataFrame as Parquet to data/raw/. Returns the saved file path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / OUTPUT_FILENAME
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.success(
        f"Dataset saved to '{output_path}' "
        f"({output_path.stat().st_size / 1024:.1f} KB)."
    )
    return output_path


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_metadata(df: pd.DataFrame, parquet_path: Path) -> Path:
    """
    Write a JSON sidecar next to the parquet file with lineage information:
      - extraction_timestamp (UTC ISO-8601)
      - row_count
      - date_range (earliest and latest 'date' in the dataset)
      - project_distribution (counts per project_code)
      - sha256 of the parquet file

    This file is the lineage anchor: every downstream artifact (EDA report,
    trained model, MLflow run) can reference it to prove data provenance.
    """
    metadata = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "date_range": {
            "min": str(df["date"].min().date()),
            "max": str(df["date"].max().date()),
        },
        "project_distribution": df["project_code"].value_counts().to_dict(),
        "sha256": _sha256(parquet_path),
    }

    meta_path = parquet_path.with_name(
        parquet_path.stem + "_metadata.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.success(f"Metadata written to '{meta_path}'.")
    logger.info(
        f"SHA-256: {metadata['sha256'][:16]}...  |  "
        f"Rows: {metadata['row_count']}  |  "
        f"Date range: {metadata['date_range']['min']} → {metadata['date_range']['max']}"
    )
    return meta_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: str | Path = "configs/config.yaml") -> None:
    """Full pipeline: load config → extract → transform → validate → save."""
    logger.info("=== Time Spent Extractor — Data Extraction ===")

    try:
        config = load_config(config_path)
        db_path = config["database"]["path"]
        raw_dir = config["data"]["raw_dir"]

        df_raw = extract(db_path)
        df = transform(df_raw)
        validate(df)
        output_path = save_parquet(df, raw_dir)
        save_metadata(df, output_path)

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
