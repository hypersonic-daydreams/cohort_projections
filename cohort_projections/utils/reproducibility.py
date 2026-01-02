"""
Reproducibility Utilities
=========================

Provides tools for logging execution history to the 'cohort_projections_meta' database.
Supports Type II SCD and scientific reproducibility requirements.
"""

import datetime
import json
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

import psycopg2

DB_NAME = "cohort_projections_meta"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_db_connection():
    try:
        return psycopg2.connect(dbname=DB_NAME)
    except psycopg2.OperationalError:
        return None


def get_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=PROJECT_ROOT
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def get_script_id(cur, script_path):
    """Resolve script path to DB ID."""
    rel_path = str(Path(script_path).resolve().relative_to(PROJECT_ROOT))
    cur.execute("SELECT id FROM code_inventory WHERE filepath = %s", (rel_path,))
    result = cur.fetchone()
    if result:
        return result[0]
    return None


@contextmanager
def log_execution(script_path, parameters=None, inputs=None, outputs=None):
    """
    Context manager to log script execution.

    Usage:
        with log_execution(__file__, params={"year": 2024}) as run_id:
            main()

    Args:
        script_path: Path to the running script (usually __file__)
        parameters: Dict of run parameters
        inputs: List of input file paths (optional)
        outputs: List of output file paths (optional)
    """
    conn = get_db_connection()
    if not conn:
        yield None
        return

    cur = conn.cursor()
    run_id = str(uuid.uuid4())
    start_time = datetime.datetime.now(datetime.UTC)
    git_commit = get_git_commit()
    script_id = get_script_id(cur, script_path)

    # Convert lists to JSON-compatible format
    params_json = json.dumps(parameters or {})
    inputs_json = json.dumps([str(p) for p in (inputs or [])])
    outputs_json = json.dumps([str(p) for p in (outputs or [])])

    try:
        # Initial Insert
        cur.execute(
            """
            INSERT INTO run_history
            (id, script_id, git_commit, start_time, status, parameters, input_manifest, output_manifest)
            VALUES (%s, %s, %s, %s, 'running', %s, %s, %s)
        """,
            (run_id, script_id, git_commit, start_time, params_json, inputs_json, outputs_json),
        )
        conn.commit()

        yield run_id

        # Success Update
        end_time = datetime.datetime.now(datetime.UTC)
        cur.execute(
            """
            UPDATE run_history
            SET status = 'success', end_time = %s
            WHERE id = %s
        """,
            (end_time, run_id),
        )
        conn.commit()

    except Exception as e:
        # Failure Update
        end_time = datetime.datetime.now(datetime.UTC)
        # error_params would be used here if we had a plan for it
        # error_params = json.dumps({"error": str(e)})
        # Ideally we'd merge error into params or have an error column,
        # but for now we append to params if key doesn't exist

        try:
            cur.execute(
                """
                UPDATE run_history
                SET status = 'failed', end_time = %s
                WHERE id = %s
            """,
                (end_time, run_id),
            )
            conn.commit()
        except Exception:
            pass

        raise e
    finally:
        conn.close()
