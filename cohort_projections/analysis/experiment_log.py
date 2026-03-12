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


# ---------------------------------------------------------------------------
# Shared parameter-level dedup utilities
# ---------------------------------------------------------------------------


def config_delta_summary(config_delta: dict[str, Any]) -> str:
    """Build a canonical human-readable summary of a config_delta dict.

    This mirrors the format used in experiment_log.csv's
    ``config_delta_summary`` column for cross-referencing.

    Parameters
    ----------
    config_delta : dict[str, Any]
        The config_delta mapping (parameter name -> value).

    Returns
    -------
    str
        A summary string like ``"college_blend_factor=0.7"`` or
        ``"boom_period_dampening: {2005-2010=0.5, 2010-2015=0.3}"``.
    """
    parts: list[str] = []
    for key, val in config_delta.items():
        if isinstance(val, dict):
            inner = ", ".join(f"{k}={v}" for k, v in val.items())
            parts.append(f"{key}: {{{inner}}}")
        elif isinstance(val, list):
            parts.append(f"{key}={val}")
        else:
            parts.append(f"{key}={val}")
    return "; ".join(parts)


def _match_config_delta(
    log_summary: str,
    candidate_delta: dict[str, Any],
) -> bool:
    """Check if a log entry's config_delta_summary matches a candidate config_delta.

    Uses parameter-level matching: for each key in the candidate, checks
    whether the log summary contains that key=value pair. This allows
    matching across different experiment IDs that tested the same parameter
    at the same value.

    Parameters
    ----------
    log_summary : str
        The ``config_delta_summary`` string from the experiment log.
    candidate_delta : dict[str, Any]
        The config_delta dict to match against.

    Returns
    -------
    bool
        True if all parameters in the candidate appear in the log summary.
    """
    if not isinstance(log_summary, str):
        return False

    candidate_summary = config_delta_summary(candidate_delta)

    # Direct match covers the common case
    if candidate_summary == log_summary:
        return True

    # Fallback: check each individual key=value fragment
    for key, val in candidate_delta.items():
        if isinstance(val, dict):
            inner = ", ".join(f"{k}={v}" for k, v in val.items())
            fragment = f"{key}: {{{inner}}}"
        elif isinstance(val, list):
            fragment = f"{key}={val}"
        else:
            fragment = f"{key}={val}"

        if fragment not in log_summary:
            return False

    return True


def is_config_delta_tested(
    config_delta: dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
) -> bool:
    """Check if a config delta has been tested by matching against experiment log entries.

    Performs parameter-level matching against the ``config_delta_summary``
    column of the experiment log CSV.

    Parameters
    ----------
    config_delta : dict[str, Any]
        The config delta to check.
    log_path : Path
        Path to the experiment log CSV file.

    Returns
    -------
    bool
        True if the config delta matches any entry in the experiment log.
    """
    if not log_path.exists():
        return False

    df = read_experiment_log(log_path)
    if df.empty or "config_delta_summary" not in df.columns:
        return False

    summaries = df["config_delta_summary"].dropna().tolist()
    return any(_match_config_delta(s, config_delta) for s in summaries)
