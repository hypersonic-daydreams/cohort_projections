#!/usr/bin/env python3
"""Run the canonical benchmark suite for a champion vs challenger comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import qc_diagnostics as qcd  # noqa: E402
import sensitivity_analysis as sa  # noqa: E402
import walk_forward_validation as wfv  # noqa: E402

from cohort_projections.analysis.benchmarking import (  # noqa: E402
    BENCHMARK_CONTRACT_VERSION,
    DEFAULT_HISTORY_DIR,
    DEFAULT_PROFILE_DIR,
    append_benchmark_index,
    build_comparison_to_champion,
    build_run_id,
    build_summary_scorecard,
    compute_prediction_intervals_generic,
    get_git_commit,
    get_git_dirty,
    load_method_profile,
    with_county_categories,
    write_manifest,
)
from cohort_projections.utils.reproducibility import log_execution  # noqa: E402

# ---------------------------------------------------------------------------
# Config injection: bridge method profiles → METHOD_DISPATCH
# ---------------------------------------------------------------------------


# Keys in resolved_config that should be converted from YAML lists to Python sets
_SET_KEYS = frozenset({"bakken_fips", "college_fips", "college_age_groups"})


def _yaml_config_to_method_config(resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Convert a profile's resolved_config (YAML types) to METHOD_DISPATCH types.

    YAML serialisation loses Python sets (→ list) and tuple-keyed dicts
    (→ string-keyed dicts).  This function reverses those conversions so the
    config is compatible with walk_forward_validation's MethodConfig contract.
    """
    cfg: dict[str, Any] = {}
    for key, value in resolved_config.items():
        if key in _SET_KEYS and isinstance(value, list):
            cfg[key] = set(value)
        elif key == "boom_period_dampening" and isinstance(value, dict):
            # "2005-2010" → (2005, 2010)
            converted: dict[tuple[int, int], float] = {}
            for period_str, factor in value.items():
                parts = str(period_str).split("-")
                converted[(int(parts[0]), int(parts[1]))] = float(factor)
            cfg[key] = converted
        else:
            cfg[key] = value
    return cfg


def _inject_profile_configs(
    profile_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Temporarily override METHOD_DISPATCH configs with profile resolved_configs.

    For each method in *profile_map*, if the profile contains a ``resolved_config``,
    convert it to MethodConfig-compatible types and patch
    ``METHOD_DISPATCH[method_id]["config"]``.

    Returns a dict mapping method_id → original config (or None if not patched),
    so the caller can restore the originals after the run.
    """
    originals: dict[str, dict[str, Any] | None] = {}
    for method_id, profile in profile_map.items():
        resolved = profile.get("resolved_config")
        if not resolved or method_id not in wfv.METHOD_DISPATCH:
            originals[method_id] = None
            continue
        originals[method_id] = wfv.METHOD_DISPATCH[method_id]["config"]  # type: ignore[index]
        patched = _yaml_config_to_method_config(resolved)
        wfv.METHOD_DISPATCH[method_id]["config"] = patched  # type: ignore[index]
        print(f"  Injected profile config for '{method_id}' ({len(patched)} keys)")
    return originals


def _restore_configs(originals: dict[str, dict[str, Any] | None]) -> None:
    """Restore METHOD_DISPATCH configs to their pre-injection state."""
    for method_id, original in originals.items():
        if original is not None and method_id in wfv.METHOD_DISPATCH:
            wfv.METHOD_DISPATCH[method_id]["config"] = original  # type: ignore[index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the versioned benchmark suite.")
    parser.add_argument("--scope", default="county", help="Benchmark scope. Only 'county' is supported today.")
    parser.add_argument("--champion-method", required=True, help="Champion method ID.")
    parser.add_argument("--champion-config", required=True, help="Champion config ID.")
    parser.add_argument("--challenger-method", required=True, help="Challenger method ID.")
    parser.add_argument("--challenger-config", required=True, help="Challenger config ID.")
    parser.add_argument("--benchmark-label", required=True, help="Human-readable benchmark label.")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="Directory containing immutable method profiles.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Directory where benchmark bundles are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print resolved method/config pairs without running the suite.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for walk-forward validation and "
            "sensitivity analysis. "
            "0 = auto-detect min(len(ORIGIN_YEARS), cpu_count). "
            "1 = sequential (default)."
        ),
    )
    return parser.parse_args()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_qc_summary(
    bias_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    report_cards: pd.DataFrame,
) -> dict[str, Any]:
    methods = sorted(report_cards["method"].unique())
    per_method: dict[str, Any] = {}
    for method in methods:
        method_cards = report_cards[report_cards["method"] == method]
        method_residual = residual_df[residual_df["method"] == method]
        per_method[method] = {
            "grade_a_count": int((method_cards["grade"] == "A").sum()),
            "grade_d_count": int((method_cards["grade"] == "D").sum()),
            "mean_mape": round(float(method_cards["mape"].mean()), 6),
            "outlier_count": int((outlier_df["method"] == method).sum()) if not outlier_df.empty else 0,
            "mean_autocorr_lag1": round(
                float(method_residual["mean_autocorr_lag1"].mean()),
                6,
            )
            if not method_residual.empty
            else 0.0,
        }
    return {
        "methods": methods,
        "bias_rows": int(len(bias_df)),
        "residual_rows": int(len(residual_df)),
        "outlier_rows": int(len(outlier_df)),
        "report_card_rows": int(len(report_cards)),
        "per_method": per_method,
    }


def main() -> None:
    args = _parse_args()
    if args.scope != "county":
        raise ValueError("Only scope='county' is implemented in P0.")
    if args.champion_method == args.challenger_method and args.champion_config == args.challenger_config:
        raise ValueError("Champion and challenger must differ.")

    champion_profile = load_method_profile(
        args.champion_method,
        args.champion_config,
        profile_dir=args.profile_dir,
    )
    challenger_profile = load_method_profile(
        args.challenger_method,
        args.challenger_config,
        profile_dir=args.profile_dir,
    )

    for method_id in [args.champion_method, args.challenger_method]:
        if method_id not in wfv.METHOD_DISPATCH:
            raise ValueError(
                f"Method '{method_id}' is not registered in walk_forward_validation.METHOD_DISPATCH"
            )

    if args.dry_run:
        print("Resolved benchmark inputs:")
        print(f"  Scope: {args.scope}")
        print(f"  Champion: {args.champion_method} / {args.champion_config}")
        print(f"  Challenger: {args.challenger_method} / {args.challenger_config}")
        print(f"  Benchmark label: {args.benchmark_label}")
        return

    methods = [args.champion_method, args.challenger_method]
    profile_map = {args.champion_method: champion_profile, args.challenger_method: challenger_profile}
    git_commit = get_git_commit(PROJECT_ROOT)
    git_dirty = get_git_dirty(PROJECT_ROOT)
    run_id = build_run_id(args.challenger_method, git_commit)
    run_dir = args.history_dir / run_id
    resolved_dir = run_dir / "resolved_configs"
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_dir.mkdir(parents=True, exist_ok=False)

    for profile in [champion_profile, challenger_profile]:
        resolved_path = resolved_dir / f"{profile['method_id']}__{profile['config_id']}.yaml"
        _write_yaml(resolved_path, profile)

    start = time.perf_counter()
    parameters = {
        "scope": args.scope,
        "benchmark_label": args.benchmark_label,
        "champion_method": args.champion_method,
        "champion_config": args.champion_config,
        "challenger_method": args.challenger_method,
        "challenger_config": args.challenger_config,
    }
    output_paths = [
        run_dir / "manifest.json",
        run_dir / "summary_scorecard.csv",
        run_dir / "summary_scorecard.json",
    ]

    with log_execution(__file__, parameters=parameters, outputs=output_paths) as execution_log_run_id:
        # Inject profile resolved_configs into METHOD_DISPATCH so that
        # config-only experiments (config_delta from experiment specs)
        # actually affect the computation.  Restore originals when done.
        print("Injecting profile configs into METHOD_DISPATCH...")
        saved_configs = _inject_profile_configs(profile_map)

        print("Loading shared benchmark inputs...")
        snapshots = wfv.load_all_snapshots()
        mig_raw = wfv.load_migration_rates_raw()
        survival = wfv.load_survival_rates()
        fertility = wfv.load_fertility_rates()

        print("Running annual walk-forward validation...")
        annual_state, annual_county, projection_curves = wfv.run_annual_validation(
            snapshots,
            mig_raw,
            survival,
            fertility,
            methods=methods,
            workers=args.workers,
        )
        annual_horizon = wfv.compute_annual_horizon_summary(annual_state, annual_county)
        annual_comparison = wfv.compute_annual_method_comparison(annual_state, annual_county)

        print("Running sensitivity analysis...")
        sensitivity_results = sa.run_sensitivity_analysis(
            snapshots,
            mig_raw,
            survival,
            fertility,
            methods=methods,
            workers=args.workers,
        )
        sensitivity_tornado = sa.compute_tornado_data(sensitivity_results)

        print("Computing QC summaries...")
        county_with_categories = with_county_categories(annual_county)
        bias_df = qcd.compute_bias_analysis(county_with_categories)
        residual_df = qcd.compute_residual_diagnostics(county_with_categories)
        outlier_df = qcd.detect_outliers(county_with_categories)
        report_cards = qcd.compute_county_report_cards(county_with_categories)
        qc_summary = _build_qc_summary(bias_df, residual_df, outlier_df, report_cards)

        print("Computing uncertainty summaries...")
        prediction_intervals = compute_prediction_intervals_generic(
            county_with_categories,
            annual_state,
        )

        print("Building scorecard and manifest...")
        scorecard = build_summary_scorecard(
            annual_state=annual_state,
            annual_county=annual_county,
            sensitivity_tornado=sensitivity_tornado,
            method_profiles=profile_map,
            scope=args.scope,
            run_id=run_id,
        )
        comparison = build_comparison_to_champion(scorecard, args.champion_method)

        annual_state.to_csv(run_dir / "state_metrics.csv", index=False)
        county_with_categories.to_csv(run_dir / "county_metrics.csv", index=False)
        annual_horizon.to_csv(run_dir / "annual_horizon_summary.csv", index=False)
        annual_comparison.to_csv(run_dir / "annual_method_comparison.csv", index=False)
        projection_curves.to_csv(run_dir / "projection_curves.csv", index=False)
        sensitivity_results.to_csv(run_dir / "sensitivity_results.csv", index=False)
        sensitivity_tornado.to_csv(run_dir / "sensitivity_summary.csv", index=False)
        bias_df.to_csv(run_dir / "bias_analysis.csv", index=False)
        residual_df.to_csv(run_dir / "residual_diagnostics.csv", index=False)
        outlier_df.to_csv(run_dir / "outlier_flags.csv", index=False)
        report_cards.to_csv(run_dir / "county_report_cards.csv", index=False)
        prediction_intervals.to_csv(run_dir / "uncertainty_summary.csv", index=False)
        scorecard.to_csv(run_dir / "summary_scorecard.csv", index=False)
        (run_dir / "summary_scorecard.json").write_text(
            scorecard.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
        (run_dir / "comparison_to_champion.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "qc_summary.json").write_text(
            json.dumps(qc_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        duration_seconds = round(time.perf_counter() - start, 3)
        run_date = run_id.split("-")[1]
        manifest = {
            "run_id": run_id,
            "run_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}",
            "benchmark_label": args.benchmark_label,
            "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "command": " ".join(sys.argv),
            "scope": args.scope,
            "methods": [
                {
                    "method_id": args.champion_method,
                    "config_id": args.champion_config,
                    "profile_path": champion_profile["profile_path"],
                    "profile_hash": champion_profile["profile_hash"],
                },
                {
                    "method_id": args.challenger_method,
                    "config_id": args.challenger_config,
                    "profile_path": challenger_profile["profile_path"],
                    "profile_hash": challenger_profile["profile_hash"],
                },
            ],
            "champion_method_id": args.champion_method,
            "challenger_method_ids": [args.challenger_method],
            "input_artifacts": [
                {
                    "path": champion_profile["profile_path"],
                    "sha256": champion_profile["profile_hash"],
                },
                {
                    "path": challenger_profile["profile_path"],
                    "sha256": challenger_profile["profile_hash"],
                },
            ],
            "output_artifacts": sorted(str(path.relative_to(PROJECT_ROOT)) for path in run_dir.iterdir()),
            "script_versions": {
                "run_benchmark_suite": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "walk_forward_validation": str((SCRIPT_DIR / "walk_forward_validation.py").relative_to(PROJECT_ROOT)),
                "sensitivity_analysis": str((SCRIPT_DIR / "sensitivity_analysis.py").relative_to(PROJECT_ROOT)),
                "qc_diagnostics": str((SCRIPT_DIR / "qc_diagnostics.py").relative_to(PROJECT_ROOT)),
            },
            "execution_log_run_id": execution_log_run_id,
            "duration_seconds": duration_seconds,
        }
        manifest_path = write_manifest(run_dir, manifest)
        (run_dir / "execution_log.json").write_text(
            json.dumps(
                {
                    "execution_log_run_id": execution_log_run_id,
                    "duration_seconds": duration_seconds,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        append_benchmark_index(
            index_path=args.history_dir / "index.csv",
            scorecard=scorecard,
            manifest_path=manifest_path,
            benchmark_label=args.benchmark_label,
            benchmark_contract_version=BENCHMARK_CONTRACT_VERSION,
            git_commit=git_commit,
            champion_method_id=args.champion_method,
        )

        # Restore original METHOD_DISPATCH configs
        _restore_configs(saved_configs)

    print(f"Benchmark run complete: {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
