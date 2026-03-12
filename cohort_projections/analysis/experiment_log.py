"""Experiment log: append-only journal of benchmark experiments."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "analysis" / "experiments" / "experiment_log.csv"

# Column order — must match config/experiment_log_schema.yaml
LOG_COLUMNS = [
    "experiment_id",
    "run_date",
    "hypothesis",
    "base_method",
    "config_delta_summary",
    "run_id",
    "outcome",
    "key_metrics_summary",
    "interpretation",
    "next_action",
    "agent_or_human",
    "spec_path",
]

# Valid outcome values
VALID_OUTCOMES = frozenset({
    "passed_all_gates",
    "failed_hard_gate",
    "needs_human_review",
    "inconclusive",
    "not_run",
})

# Valid next_action values
VALID_NEXT_ACTIONS = frozenset({
    "proceed_to_next",
    "flag_for_review",
    "promote_candidate",
    "abandon_line",
})


def append_experiment_entry(
    entry: dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a single experiment entry to the CSV log.

    Parameters
    ----------
    entry : dict[str, Any]
        Must contain all keys in LOG_COLUMNS.
    log_path : Path
        Path to the CSV log file.

    Raises
    ------
    ValueError
        If required columns are missing, or ``outcome`` / ``next_action``
        values are not in the allowed sets.
    """
    # Validate all required columns are present
    missing = set(LOG_COLUMNS) - set(entry.keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Validate outcome
    if entry["outcome"] not in VALID_OUTCOMES:
        raise ValueError(
            f"Invalid outcome {entry['outcome']!r}. "
            f"Must be one of: {sorted(VALID_OUTCOMES)}"
        )

    # Validate next_action
    if entry["next_action"] not in VALID_NEXT_ACTIONS:
        raise ValueError(
            f"Invalid next_action {entry['next_action']!r}. "
            f"Must be one of: {sorted(VALID_NEXT_ACTIONS)}"
        )

    # Create parent directories if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file doesn't exist
    write_header = not log_path.exists()

    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def read_experiment_log(
    log_path: Path = DEFAULT_LOG_PATH,
) -> pd.DataFrame:
    """Read the full experiment log as a DataFrame.

    Returns an empty DataFrame with LOG_COLUMNS if the file does not exist
    or contains only a header row.
    """
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    df = pd.read_csv(log_path, dtype=str)

    if df.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)

    return df


def get_tested_hypotheses(
    log_path: Path = DEFAULT_LOG_PATH,
) -> set[str]:
    """Return the set of experiment_ids already logged.

    Useful for dedup checks before running an experiment.
    Returns an empty set if the log file does not exist.
    """
    if not log_path.exists():
        return set()

    df = read_experiment_log(log_path)
    if df.empty:
        return set()

    return set(df["experiment_id"].dropna())
