#!/usr/bin/env python3
"""
Calibration helper: Immigration Policy multiplier (v0.8.7)
==========================================================

Purpose
-------
This script computes a transparent, share-based calibration bridge for the paper's
`Immigration Policy` scenario multiplier (default 0.65×) by triangulating:

- PEP net international migration for North Dakota (annual, estimate-year basis)
- Refugee arrivals for North Dakota (RPC; fiscal-year totals by state×nationality)
- LPR admissions for North Dakota (DHS; fiscal-year totals by state)

The intent is *documentation/calibration transparency* (not re-estimation). It
produces summary statistics used to populate the calibration callout in
`journal_article/sections/02_data_methods.tex`.

Notes
-----
- PEP is a **net** flow; RPC/DHS series are **gross** inflow counts in **FY** time.
  We therefore use (Refugees / (Refugees + LPR)) as a bounded proxy for the
  humanitarian share when translating a humanitarian-channel shock to the PEP total.
- The Travel Ban DiD percentage effect is read from existing Module 7 outputs if
  present (`results/module_7_did_estimates.json`); otherwise a fallback of 0.75 is used.

Usage
-----
    uv run python sdc_2024_replication/scripts/statistical_analysis/journal_article/revision_scripts/calibrate_immigration_policy_multiplier.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils import ConfigLoader, setup_logger
from cohort_projections.utils.reproducibility import log_execution


LOGGER = setup_logger(__name__)


def _find_repo_root(start: Path) -> Path:
    """Find the repository root by walking parents until `pyproject.toml` is found."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root (pyproject.toml not found).")


def _analysis_dir(repo_root: Path) -> Path:
    """Resolve the processed immigration analysis directory from configuration."""
    cfg = ConfigLoader().get_projection_config()
    processed_dir = (
        cfg.get("data_sources", {})
        .get("acs_moved_from_abroad", {})
        .get("processed_dir", "data/processed/immigration/analysis")
    )
    return repo_root / processed_dir


def _results_dir(repo_root: Path) -> Path:
    return repo_root / "sdc_2024_replication" / "scripts" / "statistical_analysis" / "results"


@dataclass(frozen=True)
class CalibrationWindow:
    """Inclusive year window for calibration summaries."""

    start_year: int
    end_year: int

    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))


@dataclass(frozen=True)
class CalibrationSummary:
    """Computed calibration summaries for a specific year window."""

    window: CalibrationWindow
    pep_sum: float
    refugees_sum: float
    lpr_sum: float
    refugee_share_inflows_sum: float
    refugee_share_inflows_min: float
    refugee_share_inflows_max: float


def _load_travel_ban_delta(results_dir: Path) -> float:
    """Load the Travel Ban DiD % effect (as a share) from existing Module 7 outputs."""
    path = results_dir / "module_7_did_estimates.json"
    if not path.exists():
        LOGGER.warning("Missing %s; using fallback delta=0.75", path)
        return 0.75

    payload: dict[str, Any] = json.loads(path.read_text())
    pct = payload.get("travel_ban", {}).get("percentage_effect", {}).get("estimate")
    if pct is None:
        LOGGER.warning("Travel Ban percentage effect missing in %s; using fallback delta=0.75", path)
        return 0.75

    delta = abs(float(pct)) / 100.0
    return delta


def _load_pep_net_nd(processed_dir: Path) -> pd.DataFrame:
    """Load ND PEP net international migration (estimate-year basis)."""
    path = processed_dir / "nd_migration_summary.csv"
    df = pd.read_csv(path)
    return df[["year", "nd_intl_migration"]].rename(columns={"nd_intl_migration": "pep_net"})


def _load_refugees_nd(processed_dir: Path, *, state: str) -> pd.DataFrame:
    """Load ND refugee arrivals by fiscal year (RPC-derived)."""
    path = processed_dir / "refugee_arrivals_by_state_nationality.csv"
    df = pd.read_csv(path)
    df = df[df["state"] == state].copy()
    out = (
        df.groupby("fiscal_year", as_index=False)["arrivals"]
        .sum()
        .rename(columns={"fiscal_year": "year", "arrivals": "refugees_total"})
    )
    return out


def _load_lpr_nd(processed_dir: Path, *, state: str) -> pd.DataFrame:
    """Load ND LPR admissions by fiscal year (DHS)."""
    path = processed_dir / "dhs_lpr_by_state_time_states_only.csv"
    df = pd.read_csv(path)
    df = df[df["state"] == state].copy()
    out = df[["fiscal_year", "lpr_count"]].rename(
        columns={"fiscal_year": "year", "lpr_count": "lpr_total"}
    )
    return out


def compute_calibration_summary(
    *,
    processed_dir: Path,
    state: str,
    window: CalibrationWindow,
) -> CalibrationSummary:
    """Compute calibration summaries for a specific state/window."""
    pep = _load_pep_net_nd(processed_dir)
    refugees = _load_refugees_nd(processed_dir, state=state)
    lpr = _load_lpr_nd(processed_dir, state=state)

    panel = (
        pep.merge(refugees, on="year", how="left")
        .merge(lpr, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )
    panel[["refugees_total", "lpr_total"]] = panel[["refugees_total", "lpr_total"]].fillna(0.0)

    sub = panel[panel["year"].isin(window.years())].copy()
    sub["inflows_total"] = sub["refugees_total"] + sub["lpr_total"]
    sub = sub[sub["inflows_total"] > 0].copy()
    sub["refugee_share_inflows"] = sub["refugees_total"] / sub["inflows_total"]

    refugee_share_sum = (
        sub["refugees_total"].sum() / sub["inflows_total"].sum()
        if sub["inflows_total"].sum() > 0
        else float("nan")
    )

    return CalibrationSummary(
        window=window,
        pep_sum=float(sub["pep_net"].sum()),
        refugees_sum=float(sub["refugees_total"].sum()),
        lpr_sum=float(sub["lpr_total"].sum()),
        refugee_share_inflows_sum=float(refugee_share_sum),
        refugee_share_inflows_min=float(sub["refugee_share_inflows"].min()),
        refugee_share_inflows_max=float(sub["refugee_share_inflows"].max()),
    )


def main() -> int:
    """Run the calibration summary and print the key quantities via logging."""
    parser = argparse.ArgumentParser(description="Compute calibration summaries for 0.65× policy multiplier.")
    parser.add_argument("--state", default="North Dakota", help="State name (matches processed data).")
    parser.add_argument("--start-year", type=int, default=2010, help="Start year (inclusive).")
    parser.add_argument("--end-year", type=int, default=2016, help="End year (inclusive).")
    parser.add_argument(
        "--policy-multiplier",
        type=float,
        default=0.65,
        help="Policy multiplier to back out implied humanitarian share.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    processed_dir = _analysis_dir(repo_root)
    results_dir = _results_dir(repo_root)

    delta = _load_travel_ban_delta(results_dir)
    window = CalibrationWindow(start_year=args.start_year, end_year=args.end_year)

    summary = compute_calibration_summary(processed_dir=processed_dir, state=args.state, window=window)

    implied_share = (1.0 - float(args.policy_multiplier)) / delta if delta > 0 else float("nan")
    implied_multiplier_min = 1.0 - summary.refugee_share_inflows_max * delta
    implied_multiplier_max = 1.0 - summary.refugee_share_inflows_min * delta

    LOGGER.info("Travel Ban DiD delta (abs %% effect): %.4f", delta)
    LOGGER.info("Window: %s-%s (state=%s)", window.start_year, window.end_year, args.state)
    LOGGER.info("PEP net sum: %.0f", summary.pep_sum)
    LOGGER.info("Refugees sum (RPC): %.0f", summary.refugees_sum)
    LOGGER.info("LPR sum (DHS): %.0f", summary.lpr_sum)
    LOGGER.info(
        "Refugee share among (Refugees + LPR): sum-share=%.3f, annual min=%.3f, annual max=%.3f",
        summary.refugee_share_inflows_sum,
        summary.refugee_share_inflows_min,
        summary.refugee_share_inflows_max,
    )
    LOGGER.info(
        "Implied multiplier range from share bounds (m = 1 - s*delta): [%.3f, %.3f]",
        implied_multiplier_min,
        implied_multiplier_max,
    )
    LOGGER.info(
        "Policy multiplier=%.3f implies share s=(1-m)/delta = %.3f",
        float(args.policy_multiplier),
        implied_share,
    )
    return 0


if __name__ == "__main__":
    with log_execution(__file__, parameters={"task": "calibrate_immigration_policy_multiplier"}):
        raise SystemExit(main())
