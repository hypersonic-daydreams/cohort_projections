"""Utilities for versioned benchmarking and method promotion."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "config" / "method_profiles"
DEFAULT_ALIAS_PATH = DEFAULT_PROFILE_DIR / "aliases.yaml"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "data" / "analysis" / "benchmark_history"
DEFAULT_PROMOTION_HISTORY = DEFAULT_HISTORY_DIR / "promotion_history.csv"
BENCHMARK_CONTRACT_VERSION = "1.0"

BAKKEN_FIPS = {"38105", "38053", "38061", "38025", "38089"}
RESERVATION_FIPS = {"38005", "38085", "38079"}
URBAN_COLLEGE_FIPS = {"38017", "38015", "38035", "38101"}
RECENT_ORIGINS = {2015, 2020}
SENSITIVITY_SWING_ALERT = 5.0
SENTINEL_COUNTIES = {
    "38017": "sentinel_cass_mape",
    "38035": "sentinel_grand_forks_mape",
    "38101": "sentinel_ward_mape",
    "38015": "sentinel_burleigh_mape",
    "38105": "sentinel_williams_mape",
    "38053": "sentinel_mckenzie_mape",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    return _sha256_bytes(path.read_bytes())


def get_git_commit(project_root: Path = PROJECT_ROOT) -> str:
    """Return the current git commit hash, or ``unknown`` if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_dirty(project_root: Path = PROJECT_ROOT) -> bool:
    """Return whether the working tree has uncommitted changes."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        ).decode()
        return bool(status.strip())
    except subprocess.CalledProcessError:
        return False


def build_run_id(
    primary_method_id: str,
    git_commit: str,
    now: dt.datetime | None = None,
) -> str:
    """Build a benchmark run identifier."""
    timestamp = (now or dt.datetime.now(dt.UTC)).strftime("%Y%m%d-%H%M%S")
    short_git = git_commit[:7] if git_commit and git_commit != "unknown" else "unknown"
    return f"br-{timestamp}-{primary_method_id}-{short_git}"


def _profile_path(profile_dir: Path, method_id: str, config_id: str) -> Path:
    return profile_dir / f"{method_id}__{config_id}.yaml"


def load_method_profile(
    method_id: str,
    config_id: str,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
) -> dict[str, Any]:
    """Load and validate a method profile YAML."""
    profile_path = _profile_path(profile_dir, method_id, config_id)
    if not profile_path.exists():
        raise FileNotFoundError(f"Method profile not found: {profile_path}")

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid method profile format: {profile_path}")
    if profile.get("method_id") != method_id:
        raise ValueError(f"Profile method_id mismatch in {profile_path}")
    if profile.get("config_id") != config_id:
        raise ValueError(f"Profile config_id mismatch in {profile_path}")
    if "resolved_config" not in profile:
        raise ValueError(f"Profile missing resolved_config: {profile_path}")

    profile["profile_path"] = str(profile_path)
    profile["profile_hash"] = sha256_file(profile_path)
    return profile


def load_aliases(alias_path: Path = DEFAULT_ALIAS_PATH) -> dict[str, dict[str, str]]:
    """Load alias mappings from YAML."""
    if not alias_path.exists():
        return {}
    aliases = yaml.safe_load(alias_path.read_text(encoding="utf-8")) or {}
    if not isinstance(aliases, dict):
        raise ValueError(f"Alias file must contain a mapping: {alias_path}")
    return aliases


def update_alias_mapping(
    alias_name: str,
    method_id: str,
    config_id: str,
    alias_path: Path = DEFAULT_ALIAS_PATH,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Update a single alias mapping and write the YAML file."""
    aliases = load_aliases(alias_path)
    prior = aliases.get(alias_name)
    aliases[alias_name] = {"method_id": method_id, "config_id": config_id}
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_text(yaml.safe_dump(aliases, sort_keys=False), encoding="utf-8")
    return prior, aliases[alias_name]


def with_county_categories(annual_county: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark category labels to annual county-level results."""
    df = annual_county.copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["category"] = df["county_fips"].map(assign_county_category)
    return df


def assign_county_category(fips: str) -> str:
    """Return the canonical county category used in benchmark summaries."""
    if fips in BAKKEN_FIPS:
        return "Bakken"
    if fips in RESERVATION_FIPS:
        return "Reservation"
    if fips in URBAN_COLLEGE_FIPS:
        return "Urban/College"
    return "Rural"


def compute_prediction_intervals_generic(
    annual_county: pd.DataFrame,
    annual_state: pd.DataFrame,
) -> pd.DataFrame:
    """Compute empirical error intervals for all methods present in the data."""
    records: list[dict[str, Any]] = []

    for level_name, frame in [("county", annual_county), ("state", annual_state)]:
        for method in sorted(frame["method"].unique()):
            method_frame = frame[frame["method"] == method]
            for horizon in sorted(method_frame["horizon"].unique()):
                pct_errors = method_frame.loc[
                    method_frame["horizon"] == horizon, "pct_error"
                ].dropna()
                if pct_errors.empty:
                    continue

                row: dict[str, Any] = {
                    "level": level_name,
                    "method": method,
                    "horizon": int(horizon),
                    "n_obs": int(len(pct_errors)),
                    "mean": round(float(pct_errors.mean()), 6),
                    "std": round(float(pct_errors.std(ddof=0)), 6),
                }
                for percentile in [5, 10, 25, 50, 75, 90, 95]:
                    row[f"p{percentile}"] = round(
                        float(pct_errors.quantile(percentile / 100.0)),
                        6,
                    )
                records.append(row)

    return pd.DataFrame(records)


def _mean_abs_pct_error(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame["pct_error"].abs().mean())


def _mean_pct_error(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame["pct_error"].mean())


def _aggregation_violations(annual_county: pd.DataFrame, annual_state: pd.DataFrame) -> int:
    grouped = (
        annual_county.groupby(["origin_year", "method", "validation_year"], as_index=False)
        .agg(projected=("projected", "sum"), actual=("actual", "sum"))
    )
    merged = grouped.merge(
        annual_state[
            ["origin_year", "method", "validation_year", "projected_state", "actual_state"]
        ],
        on=["origin_year", "method", "validation_year"],
        how="left",
    )
    # County totals are stored at 0.1 precision while state totals are stored as
    # whole numbers, so sub-person drift up to 1.0 can arise from rounding alone.
    projected_mismatch = (merged["projected"] - merged["projected_state"]).abs() > 1.0
    actual_mismatch = (merged["actual"] - merged["actual_state"]).abs() > 1.0
    return int((projected_mismatch | actual_mismatch).sum())


def build_summary_scorecard(
    annual_state: pd.DataFrame,
    annual_county: pd.DataFrame,
    sensitivity_tornado: pd.DataFrame,
    method_profiles: dict[str, dict[str, Any]],
    scope: str,
    run_id: str,
) -> pd.DataFrame:
    """Build the canonical benchmark scorecard."""
    county_df = with_county_categories(annual_county)
    records: list[dict[str, Any]] = []

    for method in sorted(annual_state["method"].unique()):
        profile = method_profiles[method]
        state_method = annual_state[annual_state["method"] == method]
        county_method = county_df[county_df["method"] == method]
        recent_state = state_method[state_method["origin_year"].isin(RECENT_ORIGINS)]
        recent_short = recent_state[recent_state["horizon"] <= 5]
        recent_medium = recent_state[
            (recent_state["horizon"] > 5) & (recent_state["horizon"] <= 10)
        ]
        tornado_method = sensitivity_tornado[sensitivity_tornado["method"] == method]
        aggregation_violations = _aggregation_violations(county_method, state_method)

        row: dict[str, Any] = {
            "run_id": run_id,
            "method_id": method,
            "config_id": profile["config_id"],
            "scope": scope,
            "status_at_run": profile["status"],
            "state_ape_recent_short": round(_mean_abs_pct_error(recent_short), 6),
            "state_ape_recent_medium": round(_mean_abs_pct_error(recent_medium), 6),
            "state_signed_bias_recent": round(_mean_pct_error(recent_state), 6),
            "county_mape_overall": round(_mean_abs_pct_error(county_method), 6),
            "county_mape_urban_college": round(
                _mean_abs_pct_error(county_method[county_method["category"] == "Urban/College"]),
                6,
            ),
            "county_mape_rural": round(
                _mean_abs_pct_error(county_method[county_method["category"] == "Rural"]),
                6,
            ),
            "county_mape_bakken": round(
                _mean_abs_pct_error(county_method[county_method["category"] == "Bakken"]),
                6,
            ),
            "negative_population_violations": int((county_method["projected"] < 0).sum()),
            "scenario_order_violations": 0,
            "aggregation_violations": aggregation_violations,
            "sensitivity_instability_flag": bool(
                not tornado_method.empty
                and (
                    tornado_method["swing_state_error"].abs().max() >= SENSITIVITY_SWING_ALERT
                    or tornado_method["mape_swing"].abs().max() >= SENSITIVITY_SWING_ALERT
                )
            ),
        }

        for sentinel_fips, column in SENTINEL_COUNTIES.items():
            sentinel_frame = county_method[county_method["county_fips"] == sentinel_fips]
            row[column] = round(_mean_abs_pct_error(sentinel_frame), 6)

        records.append(row)

    return pd.DataFrame(records)


def build_comparison_to_champion(
    scorecard: pd.DataFrame,
    champion_method_id: str,
) -> dict[str, Any]:
    """Build challenger deltas relative to the champion."""
    champion_rows = scorecard[scorecard["method_id"] == champion_method_id]
    if champion_rows.empty:
        raise ValueError(f"Champion method not found in scorecard: {champion_method_id}")
    champion = champion_rows.iloc[0].to_dict()

    comparison_metrics = [
        "state_ape_recent_short",
        "state_ape_recent_medium",
        "state_signed_bias_recent",
        "county_mape_overall",
        "county_mape_urban_college",
        "county_mape_rural",
        "county_mape_bakken",
    ]

    challengers: list[dict[str, Any]] = []
    for _, row in scorecard.iterrows():
        method_id = str(row["method_id"])
        if method_id == champion_method_id:
            continue
        challenger = row.to_dict()
        deltas = {
            metric: round(float(challenger[metric]) - float(champion[metric]), 6)
            for metric in comparison_metrics
        }
        challengers.append(
            {
                "method_id": method_id,
                "config_id": challenger["config_id"],
                "deltas": deltas,
                "hard_constraint_regression": bool(
                    challenger["negative_population_violations"] > champion["negative_population_violations"]
                    or challenger["aggregation_violations"] > champion["aggregation_violations"]
                    or challenger["scenario_order_violations"] > champion["scenario_order_violations"]
                ),
            }
        )

    return {
        "champion_method_id": champion_method_id,
        "champion_config_id": champion["config_id"],
        "challengers": challengers,
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write the run manifest as JSON."""
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def append_benchmark_index(
    index_path: Path,
    scorecard: pd.DataFrame,
    manifest_path: Path,
    benchmark_label: str,
    benchmark_contract_version: str,
    git_commit: str,
    champion_method_id: str,
    decision_id: str | None = None,
    decision_status: str = "pending",
) -> None:
    """Append one row per method to the benchmark history index."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_date",
        "method_id",
        "config_id",
        "scope",
        "benchmark_label",
        "benchmark_contract_version",
        "git_commit",
        "decision_id",
        "decision_status",
        "is_champion_at_run",
        "summary_scorecard_path",
        "manifest_path",
    ]
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for _, row in scorecard.iterrows():
            writer.writerow(
                {
                    "run_id": row["run_id"],
                    "run_date": str(row["run_id"]).split("-")[1],
                    "method_id": row["method_id"],
                    "config_id": row["config_id"],
                    "scope": row["scope"],
                    "benchmark_label": benchmark_label,
                    "benchmark_contract_version": benchmark_contract_version,
                    "git_commit": git_commit,
                    "decision_id": decision_id or "",
                    "decision_status": decision_status,
                    "is_champion_at_run": str(row["method_id"] == champion_method_id).lower(),
                    "summary_scorecard_path": str(manifest_path.parent / "summary_scorecard.csv"),
                    "manifest_path": str(manifest_path),
                }
            )


def render_benchmark_decision_record(
    manifest: dict[str, Any],
    scorecard: pd.DataFrame,
    comparison: dict[str, Any],
) -> str:
    """Render a draft benchmark decision record in Markdown."""
    champion_method = comparison["champion_method_id"]
    champion_row = scorecard[scorecard["method_id"] == champion_method].iloc[0]
    challenger = comparison["challengers"][0]
    challenger_row = scorecard[scorecard["method_id"] == challenger["method_id"]].iloc[0]
    decision_id = f"{manifest['run_date']}-{challenger['method_id']}-vs-{champion_method}"

    metrics = [
        ("Recent-origin state APE (short)", "state_ape_recent_short"),
        ("Recent-origin state APE (medium)", "state_ape_recent_medium"),
        ("Recent-origin signed bias", "state_signed_bias_recent"),
        ("County MAPE overall", "county_mape_overall"),
        ("County MAPE urban/college", "county_mape_urban_college"),
        ("County MAPE rural", "county_mape_rural"),
        ("County MAPE Bakken", "county_mape_bakken"),
    ]
    lines = [
        "# Benchmark Decision Record",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Decision ID | {decision_id} |",
        f"| Date | {manifest['run_date']} |",
        f"| Scope | {manifest['scope']} |",
        f"| Champion Method | `{champion_method}` |",
        f"| Champion Config | `{champion_row['config_id']}` |",
        f"| Challenger Method | `{challenger['method_id']}` |",
        f"| Challenger Config | `{challenger_row['config_id']}` |",
        f"| Benchmark Run ID | `{manifest['run_id']}` |",
        "| Reviewer |  |",
        "| Status | Draft |",
        "",
        "## Context",
        "",
        f"Benchmark label: `{manifest['benchmark_label']}`.",
        "",
        "## Primary Metrics",
        "",
        "| Metric | Champion | Challenger | Delta |",
        "|--------|----------|------------|-------|",
    ]
    for label, column in metrics:
        delta = float(challenger_row[column]) - float(champion_row[column])
        lines.append(
            f"| {label} | {champion_row[column]:.6f} | {challenger_row[column]:.6f} | {delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Hard Constraints and Risks",
            "",
            f"- Champion negative population violations: {int(champion_row['negative_population_violations'])}",
            f"- Challenger negative population violations: {int(challenger_row['negative_population_violations'])}",
            f"- Champion aggregation violations: {int(champion_row['aggregation_violations'])}",
            f"- Challenger aggregation violations: {int(challenger_row['aggregation_violations'])}",
            f"- Challenger hard constraint regression: {challenger['hard_constraint_regression']}",
            "",
            "## Decision",
            "",
            "- `promote`",
            "- `accept_but_do_not_promote`",
            "- `retain_champion`",
            "- `reject`",
            "- `needs_more_segmentation`",
            "",
            "Decision rationale:",
            "",
            "## Reversion Plan",
            "",
            f"Restore alias to `{champion_method}` / `{champion_row['config_id']}` if the challenger is later reverted.",
            "",
        ]
    )
    return "\n".join(lines)


def decision_file_is_approved(decision_path: Path) -> bool:
    """Return whether a decision record is explicitly marked approved."""
    if not decision_path.exists():
        return False
    text = decision_path.read_text(encoding="utf-8")
    return "| Status | Approved |" in text


def append_promotion_history(
    history_path: Path,
    alias_name: str,
    prior_mapping: dict[str, Any] | None,
    new_mapping: dict[str, Any],
    decision_id: str,
) -> None:
    """Append a promotion event to CSV history."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "promoted_at_utc",
        "alias_name",
        "prior_method_id",
        "prior_config_id",
        "new_method_id",
        "new_config_id",
        "decision_id",
    ]
    write_header = not history_path.exists()
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "promoted_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "alias_name": alias_name,
                "prior_method_id": "" if prior_mapping is None else prior_mapping.get("method_id", ""),
                "prior_config_id": "" if prior_mapping is None else prior_mapping.get("config_id", ""),
                "new_method_id": new_mapping["method_id"],
                "new_config_id": new_mapping["config_id"],
                "decision_id": decision_id,
            }
        )
