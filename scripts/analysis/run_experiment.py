#!/usr/bin/env python3
"""Experiment orchestrator — single-command pipeline from spec to classification.

Reads an experiment spec YAML, derives method/config identifiers, creates an
immutable method profile, runs the benchmark suite as a subprocess, evaluates
the results against the evaluation policy, and writes the outcome to the
experiment log.

Usage
-----
    python scripts/analysis/run_experiment.py --spec <path-to-spec-yaml>

Optional flags:
    --dry-run        Validate spec and print resolved config without running.
    --policy <path>  Override the default evaluation policy path.
    --profile-dir    Override the default method profile directory.

Design invariants:
    - The orchestrator NEVER promotes — it only classifies.
    - Method profiles are write-once (skip if file already exists).
    - The experiment log is append-only.
    - If anything fails mid-run, an ``inconclusive`` entry is still logged.

See also:
    - docs/plans/benchmarking-process-improvement-roadmap.md (BM-001-04)
    - config/benchmark_evaluation_policy.yaml
    - cohort_projections/analysis/evaluation_policy.py
    - cohort_projections/analysis/experiment_log.py
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))

from cohort_projections.analysis.benchmarking import (  # noqa: E402
    DEFAULT_PROFILE_DIR,
    load_aliases,
    load_method_profile,
)
from cohort_projections.analysis.evaluation_policy import (  # noqa: E402
    evaluate_scorecard,
    load_policy,
)
from cohort_projections.analysis.experiment_log import (  # noqa: E402
    append_experiment_entry,
)
from cohort_projections.utils.reproducibility import log_execution  # noqa: E402

# Directories for pending and completed specs
DEFAULT_PENDING_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "pending"
DEFAULT_COMPLETED_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "completed"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "data" / "analysis" / "benchmark_history"

# Required top-level fields in an experiment spec
REQUIRED_SPEC_FIELDS = frozenset(
    {
        "experiment_id",
        "hypothesis",
        "base_method",
        "base_config",
        "config_delta",
        "scope",
        "benchmark_label",
        "requested_by",
    }
)

# Map evaluation classification to next_action for the experiment log
_CLASSIFICATION_TO_ACTION: dict[str, str] = {
    "passed_all_gates": "proceed_to_next",
    "failed_hard_gate": "flag_for_review",
    "needs_human_review": "flag_for_review",
    "inconclusive": "flag_for_review",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the experiment orchestrator from a spec YAML."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to experiment spec YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate spec and print resolved config without running.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Override default evaluation policy path.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="Override default method profile directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Spec loading and validation
# ---------------------------------------------------------------------------


def _load_spec(spec_path: Path) -> dict[str, Any]:
    """Load and validate an experiment spec YAML."""
    if not spec_path.exists():
        raise FileNotFoundError(f"Experiment spec not found: {spec_path}")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"Spec must be a YAML mapping: {spec_path}")

    missing = REQUIRED_SPEC_FIELDS - set(spec.keys())
    if missing:
        raise ValueError(
            f"Spec is missing required fields: {sorted(missing)} in {spec_path}"
        )

    if spec["scope"] != "county":
        raise ValueError(
            f"Only scope='county' is supported. Got: {spec['scope']!r}"
        )

    return spec


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


def _derive_method_id(spec: dict[str, Any]) -> str:
    """Derive method_id from spec, using override if provided."""
    if spec.get("method_id_override"):
        return str(spec["method_id_override"])
    return str(spec["base_method"])


def _derive_config_id(spec: dict[str, Any]) -> str:
    """Derive config_id from spec, using override or auto-generating."""
    if spec.get("config_id_override"):
        return str(spec["config_id_override"])
    today = dt.date.today().strftime("%Y%m%d")
    experiment_id = str(spec["experiment_id"])
    # Strip "exp-YYYYMMDD-" prefix to create slug
    parts = experiment_id.split("-", 3)
    if len(parts) >= 4:
        slug = parts[3]
    elif len(parts) == 3:
        slug = parts[2]
    else:
        slug = experiment_id
    return f"cfg-{today}-{slug}"


def _check_method_dispatch(method_id: str) -> bool:
    """Check if method_id is registered in METHOD_DISPATCH."""
    import walk_forward_validation as wfv  # noqa: E402

    return method_id in wfv.METHOD_DISPATCH


# ---------------------------------------------------------------------------
# Method profile creation
# ---------------------------------------------------------------------------


def _create_method_profile(
    spec: dict[str, Any],
    method_id: str,
    config_id: str,
    profile_dir: Path,
) -> Path:
    """Create an immutable method profile by deep-copying the base and applying config_delta.

    Returns the path to the profile YAML. If the file already exists, skips creation
    (idempotent).
    """
    profile_path = profile_dir / f"{method_id}__{config_id}.yaml"
    if profile_path.exists():
        print(f"Profile already exists, skipping creation: {profile_path}")
        return profile_path

    # Load the base profile
    base_profile = load_method_profile(
        spec["base_method"],
        spec["base_config"],
        profile_dir=profile_dir,
    )

    # Deep-copy and apply config_delta
    new_profile = copy.deepcopy(base_profile)
    config_delta = spec.get("config_delta", {})
    if isinstance(config_delta, dict) and config_delta:
        _deep_merge(new_profile.setdefault("resolved_config", {}), config_delta)

    # Set experiment metadata
    new_profile["method_id"] = method_id
    new_profile["config_id"] = config_id
    new_profile["status"] = "experiment"
    new_profile["created_date"] = dt.date.today().isoformat()

    # Remove transient keys that should not be persisted
    new_profile.pop("profile_path", None)
    new_profile.pop("profile_hash", None)

    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        yaml.safe_dump(new_profile, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Created method profile: {profile_path}")
    return profile_path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Benchmark subprocess
# ---------------------------------------------------------------------------


def _run_benchmark_subprocess(
    scope: str,
    champion_method: str,
    champion_config: str,
    challenger_method: str,
    challenger_config: str,
    benchmark_label: str,
    profile_dir: Path,
) -> tuple[bool, str, str, str]:
    """Run the benchmark suite as a subprocess.

    Returns (success, run_id, stdout, stderr).
    """
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_benchmark_suite.py"),
        "--scope", scope,
        "--champion-method", champion_method,
        "--champion-config", champion_config,
        "--challenger-method", challenger_method,
        "--challenger-config", challenger_config,
        "--benchmark-label", benchmark_label,
        "--profile-dir", str(profile_dir),
    ]
    print(f"Running benchmark suite: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        print(f"Benchmark suite failed (rc={result.returncode})")
        if stderr:
            print(f"stderr: {stderr}")
        return False, "", stdout, stderr

    # Parse run_id from the last line of stdout
    # Expected format: "Benchmark run complete: data/analysis/benchmark_history/<run_id>"
    run_id = ""
    for line in reversed(stdout.splitlines()):
        if "Benchmark run complete:" in line:
            # Extract the run_id from the path
            path_part = line.split("Benchmark run complete:")[-1].strip()
            run_id = Path(path_part).name
            break

    if not run_id:
        print("Could not parse run_id from benchmark output")
        return False, "", stdout, stderr

    return True, run_id, stdout, stderr


# ---------------------------------------------------------------------------
# Result evaluation
# ---------------------------------------------------------------------------


def _evaluate_results(
    run_id: str,
    challenger_method: str,
    champion_method: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Load the scorecard and evaluate challenger against champion."""
    scorecard_path = DEFAULT_HISTORY_DIR / run_id / "summary_scorecard.csv"
    if not scorecard_path.exists():
        return {
            "classification": "inconclusive",
            "reasons": [f"Scorecard not found: {scorecard_path}"],
            "hard_gate_results": {},
            "tradeoff_results": {},
            "sensitivity_flag": False,
        }

    scorecard = pd.read_csv(scorecard_path)

    challenger_rows = scorecard[scorecard["method_id"] == challenger_method]
    champion_rows = scorecard[scorecard["method_id"] == champion_method]

    if challenger_rows.empty:
        return {
            "classification": "inconclusive",
            "reasons": [
                f"Challenger '{challenger_method}' not found in scorecard"
            ],
            "hard_gate_results": {},
            "tradeoff_results": {},
            "sensitivity_flag": False,
        }
    if champion_rows.empty:
        return {
            "classification": "inconclusive",
            "reasons": [
                f"Champion '{champion_method}' not found in scorecard"
            ],
            "hard_gate_results": {},
            "tradeoff_results": {},
            "sensitivity_flag": False,
        }

    challenger_row = challenger_rows.iloc[0].to_dict()
    champion_row = champion_rows.iloc[0].to_dict()

    return evaluate_scorecard(challenger_row, champion_row, policy)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _config_delta_summary(config_delta: dict[str, Any]) -> str:
    """Build a one-line summary of config_delta keys and values."""
    if not config_delta:
        return "no changes"
    parts = []
    for key, value in config_delta.items():
        if isinstance(value, dict):
            sub_keys = ", ".join(f"{k}={v}" for k, v in value.items())
            parts.append(f"{key}: {{{sub_keys}}}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def _key_metrics_summary(evaluation: dict[str, Any]) -> str:
    """Format key metric deltas from the evaluation result."""
    tradeoff = evaluation.get("tradeoff_results", {})
    if not tradeoff:
        return "no tradeoff metrics"
    parts = []
    for metric, detail in tradeoff.items():
        delta = detail.get("delta", 0.0)
        parts.append(f"{metric}: {delta:+.4f}")
    return "; ".join(parts)


def _interpretation(
    classification: str, reasons: list[str]
) -> str:
    """Auto-generate an interpretation from classification and reasons."""
    summary_map = {
        "passed_all_gates": "Challenger passed all hard gates and tradeoff thresholds.",
        "needs_human_review": "Hard gates passed but one or more tradeoff thresholds were breached.",
        "failed_hard_gate": "One or more hard constraint gates were violated.",
        "inconclusive": "Unable to fully evaluate — benchmark did not complete or data is missing.",
    }
    base = summary_map.get(classification, f"Classification: {classification}")
    failed_reasons = [r for r in reasons if "FAILED" in r or "BREACHED" in r]
    if failed_reasons:
        return f"{base} Issues: {'; '.join(failed_reasons)}"
    return base


# ---------------------------------------------------------------------------
# Experiment log entry
# ---------------------------------------------------------------------------


def _write_experiment_log(
    spec: dict[str, Any],
    run_id: str,
    evaluation: dict[str, Any],
    spec_dest_path: Path,
) -> None:
    """Build and append the experiment log entry."""
    classification = evaluation["classification"]
    reasons = evaluation.get("reasons", [])
    config_delta = spec.get("config_delta", {})

    entry = {
        "experiment_id": spec["experiment_id"],
        "run_date": dt.date.today().isoformat(),
        "hypothesis": spec["hypothesis"],
        "base_method": spec["base_method"],
        "config_delta_summary": _config_delta_summary(config_delta),
        "run_id": run_id or "not_run",
        "outcome": classification,
        "key_metrics_summary": _key_metrics_summary(evaluation),
        "interpretation": _interpretation(classification, reasons),
        "next_action": _CLASSIFICATION_TO_ACTION.get(classification, "flag_for_review"),
        "agent_or_human": spec.get("requested_by", "unknown"),
        "spec_path": str(spec_dest_path.relative_to(PROJECT_ROOT)),
    }

    append_experiment_entry(entry)
    print(f"Experiment log entry written for {spec['experiment_id']}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # --- Step 1: Parse and validate spec ---
    spec = _load_spec(args.spec)
    experiment_id = spec["experiment_id"]
    scope = spec["scope"]
    benchmark_label = spec["benchmark_label"]
    config_delta = spec.get("config_delta", {})

    print(f"Experiment: {experiment_id}")
    print(f"Hypothesis: {spec['hypothesis']}")

    # --- Step 2: Derive method_id and config_id ---
    method_id = _derive_method_id(spec)
    config_id = _derive_config_id(spec)
    print(f"Derived method_id: {method_id}")
    print(f"Derived config_id: {config_id}")

    # --- Step 3: Check METHOD_DISPATCH ---
    if not _check_method_dispatch(method_id):
        print(
            f"Method '{method_id}' is not registered in METHOD_DISPATCH. "
            "This experiment requires code changes — classifying as inconclusive."
        )
        evaluation: dict[str, Any] = {
            "classification": "inconclusive",
            "reasons": [
                f"Method '{method_id}' not in METHOD_DISPATCH — code changes required"
            ],
            "hard_gate_results": {},
            "tradeoff_results": {},
            "sensitivity_flag": False,
        }
        completed_dir = DEFAULT_COMPLETED_DIR
        completed_dir.mkdir(parents=True, exist_ok=True)
        dest = completed_dir / args.spec.name
        shutil.move(str(args.spec), str(dest))
        _write_experiment_log(spec, run_id="", evaluation=evaluation, spec_dest_path=dest)
        result_json = {
            "outcome": "inconclusive",
            "run_id": None,
            "experiment_id": experiment_id,
            "classification_details": evaluation,
        }
        print(json.dumps(result_json, indent=2))
        return

    # --- Dry run: stop here ---
    if args.dry_run:
        # Resolve champion for display
        aliases = load_aliases()
        champion_alias = aliases.get(f"{scope}_champion", {})
        print("\nResolved experiment inputs:")
        print(f"  Scope: {scope}")
        print(f"  Champion: {champion_alias.get('method_id', 'N/A')} / {champion_alias.get('config_id', 'N/A')}")
        print(f"  Challenger: {method_id} / {config_id}")
        print(f"  Base method: {spec['base_method']} / {spec['base_config']}")
        print(f"  Config delta: {_config_delta_summary(config_delta)}")
        print(f"  Benchmark label: {benchmark_label}")
        print(f"  Requested by: {spec['requested_by']}")
        print("\nDry run complete — no benchmark executed.")
        return

    # Wrap the rest in error handling so we always log an outcome
    run_id = ""
    evaluation = {}
    spec_dest_path = args.spec  # default; updated after move

    parameters = {
        "experiment_id": experiment_id,
        "scope": scope,
        "benchmark_label": benchmark_label,
        "method_id": method_id,
        "config_id": config_id,
    }
    output_paths: list[Path] = []

    try:
        with log_execution(__file__, parameters=parameters, outputs=output_paths):
            # --- Step 4: Create immutable method profile ---
            _create_method_profile(spec, method_id, config_id, args.profile_dir)

            # --- Step 5: Run benchmark suite ---
            aliases = load_aliases()
            champion_alias = aliases.get(f"{scope}_champion", {})
            if not champion_alias:
                raise ValueError(
                    f"No champion alias found for '{scope}_champion' in aliases.yaml"
                )
            champion_method = champion_alias["method_id"]
            champion_config = champion_alias["config_id"]

            print(f"Champion: {champion_method} / {champion_config}")
            print(f"Challenger: {method_id} / {config_id}")

            success, run_id, stdout, stderr = _run_benchmark_subprocess(
                scope=scope,
                champion_method=champion_method,
                champion_config=champion_config,
                challenger_method=method_id,
                challenger_config=config_id,
                benchmark_label=benchmark_label,
                profile_dir=args.profile_dir,
            )

            if not success:
                evaluation = {
                    "classification": "inconclusive",
                    "reasons": [
                        "Benchmark suite failed",
                        stderr[:500] if stderr else "no stderr",
                    ],
                    "hard_gate_results": {},
                    "tradeoff_results": {},
                    "sensitivity_flag": False,
                }
            else:
                # --- Step 6: Evaluate results against policy ---
                policy_path = args.policy
                policy = load_policy(policy_path) if policy_path else load_policy()
                evaluation = _evaluate_results(
                    run_id, method_id, champion_method, policy
                )

    except Exception:
        # Catch-all: log inconclusive and re-print traceback
        tb = traceback.format_exc()
        print(f"Error during experiment execution:\n{tb}")
        if not evaluation:
            evaluation = {
                "classification": "inconclusive",
                "reasons": [f"Unhandled error: {tb[:500]}"],
                "hard_gate_results": {},
                "tradeoff_results": {},
                "sensitivity_flag": False,
            }

    # --- Step 7: Write experiment log entry ---
    # --- Step 8: Move spec from pending to completed ---
    completed_dir = DEFAULT_COMPLETED_DIR
    completed_dir.mkdir(parents=True, exist_ok=True)
    spec_dest_path = completed_dir / args.spec.name
    try:
        shutil.move(str(args.spec), str(spec_dest_path))
    except Exception as move_err:
        print(f"Warning: could not move spec to completed: {move_err}")
        spec_dest_path = args.spec

    _write_experiment_log(spec, run_id, evaluation, spec_dest_path)

    # --- Step 9: Print structured JSON result ---
    classification = evaluation.get("classification", "inconclusive")
    result_json = {
        "outcome": classification,
        "run_id": run_id or None,
        "experiment_id": experiment_id,
        "classification_details": evaluation,
    }
    print(json.dumps(result_json, indent=2))


if __name__ == "__main__":
    main()
