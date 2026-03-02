#!/usr/bin/env python3
"""
Build interactive HTML report for ND population projection results.

Creates a single self-contained HTML file with Plotly charts, sortable tables,
and tabbed navigation across five sections: State Overview, County Explorer,
Place Projections, QA Diagnostics, and Methodology.

Output: data/exports/nd_projections_interactive_report_{datestamp}.html

Usage:
    python scripts/exports/build_interactive_report.py
    python scripts/exports/build_interactive_report.py --scenarios baseline high_growth
    python scripts/exports/build_interactive_report.py --output-dir data/exports/
    python scripts/exports/build_interactive_report.py --no-plotly-js

Key ADRs:
    ADR-012: Output and export format strategy
    ADR-037: CBO-grounded scenario methodology
    ADR-033: City/place projection methodology
    ADR-054: State-county aggregation
    ADR-057: Rolling-origin backtests
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import _report_theme as theme  # noqa: E402
from _methodology import PROVISIONAL_LABEL, SCENARIOS  # noqa: E402
from _report_sections.county_explorer import build_county_explorer  # noqa: E402
from _report_sections.methodology import build_methodology_section  # noqa: E402
from _report_sections.place_projections import build_place_projections  # noqa: E402
from _report_sections.qa_diagnostics import build_qa_diagnostics  # noqa: E402
from _report_sections.state_overview import build_state_overview  # noqa: E402
from project_utils import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
logger = setup_logger(__name__, log_level="INFO")
TODAY = datetime.now(tz=UTC).date()
DATE_STAMP = TODAY.strftime("%Y%m%d")
TEMPLATE_DIR = Path(__file__).parent / "_report_templates"


# ===================================================================
# Data Loading
# ===================================================================


def _load_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    """Load a CSV file, returning None on failure."""
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        logger.exception("Failed to read CSV: %s", path)
        return None


def _load_parquet_safe(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    """Load a Parquet file, returning None on failure."""
    if not path.exists():
        logger.warning("Parquet not found: %s", path)
        return None
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception:
        logger.exception("Failed to read Parquet: %s", path)
        return None


def _load_json_safe(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on failure."""
    if not path.exists():
        logger.warning("JSON not found: %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to read JSON: %s", path)
        return None


def load_scenario_data(scenario: str) -> dict[str, Any]:
    """Load all data artifacts for a single scenario.

    Returns a dict with keys:
        state_summary, state_parquet, county_summary, county_yearly,
        place_summary, hu_projections, qa_tier_summary, qa_reconciliation,
        qa_share_sum_validation, qa_outlier_flags
    """
    base = PROJECT_ROOT / "data" / "projections" / scenario
    data: dict[str, Any] = {}

    # State
    state_dir = base / "state"
    data["state_summary"] = _load_csv_safe(state_dir / "states_summary.csv")

    # State parquet (for population pyramid) — only for baseline to save memory
    state_pq_pattern = f"nd_state_38_projection_2025_2055_{scenario}.parquet"
    state_pq_path = state_dir / state_pq_pattern
    data["state_parquet"] = _load_parquet_safe(state_pq_path)

    # County
    county_dir = base / "county"
    data["county_summary"] = _load_csv_safe(county_dir / "countys_summary.csv")

    # County yearly totals: aggregate from parquet files
    county_yearly = _build_county_yearly(county_dir)
    data["county_yearly"] = county_yearly

    # Place
    place_dir = base / "place"
    data["place_summary"] = _load_csv_safe(
        place_dir / "places_summary.csv",
        dtype={"place_fips": str, "county_fips": str},
    )
    data["hu_projections"] = _load_parquet_safe(
        place_dir / "housing_unit_projections.parquet"
    )

    # QA
    qa_dir = place_dir / "qa"
    data["qa_tier_summary"] = _load_csv_safe(qa_dir / "qa_tier_summary.csv")
    data["qa_reconciliation"] = _load_csv_safe(qa_dir / "qa_reconciliation_magnitude.csv")
    data["qa_share_sum_validation"] = _load_csv_safe(qa_dir / "qa_share_sum_validation.csv")
    data["qa_outlier_flags"] = _load_csv_safe(qa_dir / "qa_outlier_flags.csv")

    return data


def _build_county_yearly(county_dir: Path) -> pd.DataFrame | None:
    """Build county x year population totals from individual parquet files."""
    parquet_files = sorted(county_dir.glob("nd_county_*_projection_*.parquet"))
    if not parquet_files:
        logger.warning("No county parquet files found in %s", county_dir)
        return None

    dfs = []
    for pf in parquet_files:
        parts = pf.stem.split("_")
        fips = int(parts[2])  # nd_county_38XXX_...
        try:
            df = pd.read_parquet(pf, columns=["year", "population"])
            yearly = df.groupby("year")["population"].sum().reset_index()
            yearly["fips"] = fips
            dfs.append(yearly)
        except Exception:
            logger.exception("Failed to read county parquet: %s", pf)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_backtesting_data() -> dict[str, Any]:
    """Load backtesting artifacts."""
    bt_dir = PROJECT_ROOT / "data" / "backtesting"
    data: dict[str, Any] = {}

    # Place backtest results
    pb_dir = bt_dir / "place_backtest_results"
    data["variant_scores"] = _load_csv_safe(pb_dir / "backtest_variant_scores.csv")
    data["tier_aggregates"] = _load_csv_safe(pb_dir / "backtest_tier_aggregates.csv")
    data["backtest_winner"] = _load_json_safe(pb_dir / "backtest_winner.json") or {}

    # Rolling origin results
    ro_dir = bt_dir / "rolling_origin_results"
    data["rolling_origin_per_window"] = _load_csv_safe(
        ro_dir / "rolling_origin_per_window_scores.csv"
    )
    data["rolling_origin_aggregated"] = _load_csv_safe(
        ro_dir / "rolling_origin_aggregated_scores.csv"
    )
    data["rolling_origin_winner"] = _load_json_safe(ro_dir / "rolling_origin_winner.json") or {}

    return data


def load_qa_data(scenarios: list[str]) -> dict[str, Any]:
    """Load QA artifacts from the first available scenario with QA data."""
    for scenario in scenarios:
        qa_dir = (
            PROJECT_ROOT / "data" / "projections" / scenario / "place" / "qa"
        )
        if qa_dir.exists():
            return {
                "tier_summary": _load_csv_safe(qa_dir / "qa_tier_summary.csv"),
                "reconciliation": _load_csv_safe(
                    qa_dir / "qa_reconciliation_magnitude.csv"
                ),
                "share_sum_validation": _load_csv_safe(
                    qa_dir / "qa_share_sum_validation.csv"
                ),
                "outlier_flags": _load_csv_safe(qa_dir / "qa_outlier_flags.csv"),
            }
    return {}


# ===================================================================
# Plotly.js Bundling
# ===================================================================


def get_plotly_js_tag(inline: bool = True) -> str:
    """Return the Plotly.js <script> tag, either inline or CDN.

    Parameters
    ----------
    inline : bool
        If True, read the local plotly.min.js and embed inline.
        If False, return a CDN script tag for smaller file size.
    """
    if not inline:
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    # Find the local plotly.min.js
    try:
        import plotly
        plotly_dir = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if plotly_dir.exists():
            js_content = plotly_dir.read_text(encoding="utf-8")
            return f"<script>{js_content}</script>"
        else:
            logger.warning(
                "plotly.min.js not found at %s, falling back to CDN", plotly_dir
            )
            return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    except Exception:
        logger.exception("Failed to read plotly.min.js, falling back to CDN")
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


# ===================================================================
# Report Assembly
# ===================================================================


def build_report(
    scenarios: list[str],
    output_dir: Path,
    inline_plotly: bool = True,
) -> Path:
    """Build the complete interactive HTML report.

    Parameters
    ----------
    scenarios : list[str]
        Scenario keys to include (e.g., ['baseline', 'high_growth', 'restricted_growth']).
    output_dir : Path
        Directory to write the output HTML file.
    inline_plotly : bool
        Whether to inline Plotly.js (True) or use CDN (False).

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    logger.info("=" * 60)
    logger.info("ND Population Projections - Interactive Report Builder")
    logger.info("Date: %s", TODAY.isoformat())
    logger.info("Scenarios: %s", ", ".join(scenarios))
    logger.info("=" * 60)

    # --- Load Data ---
    logger.info("Loading scenario data...")
    scenarios_data: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            logger.warning("Unknown scenario '%s', skipping", scenario)
            continue
        logger.info("  Loading %s...", scenario)
        scenarios_data[scenario] = load_scenario_data(scenario)

    logger.info("Loading backtesting data...")
    backtesting_data = load_backtesting_data()

    logger.info("Loading QA data...")
    qa_data = load_qa_data(scenarios)

    # --- Build Sections ---
    logger.info("Building report sections...")

    logger.info("  State Overview...")
    state_overview_html = build_state_overview(scenarios_data, theme)

    logger.info("  County Explorer...")
    county_explorer_html = build_county_explorer(scenarios_data, theme)

    logger.info("  Place Projections...")
    place_projections_html = build_place_projections(scenarios_data, theme)

    logger.info("  QA & Diagnostics...")
    qa_diagnostics_html = build_qa_diagnostics(backtesting_data, qa_data, theme)

    logger.info("  Methodology...")
    methodology_html = build_methodology_section(theme)

    # --- Assemble HTML ---
    logger.info("Assembling HTML report...")

    # Get Plotly.js
    plotly_js_tag = get_plotly_js_tag(inline=inline_plotly)

    # Build subtitle
    scenario_labels = [theme.get_scenario_label(s) for s in scenarios if s in SCENARIOS]
    subtitle = (
        f"Cohort-Component Projections 2025-2055 | "
        f"Scenarios: {', '.join(scenario_labels)} | "
        f"Generated {TODAY.strftime('%B %d, %Y')}"
    )

    # Render Jinja2 template
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,  # We're building HTML fragments ourselves
    )
    template = env.get_template("base.html")

    html_output = template.render(
        plotly_js=plotly_js_tag,
        subtitle=subtitle,
        provisional_label=PROVISIONAL_LABEL,
        state_overview=state_overview_html,
        county_explorer=county_explorer_html,
        place_projections=place_projections_html,
        qa_diagnostics=qa_diagnostics_html,
        methodology=methodology_html,
        generation_date=TODAY.strftime("%B %d, %Y"),
    )

    # --- Write Output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nd_projections_interactive_report_{DATE_STAMP}.html"

    output_path.write_text(html_output, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("Report generated successfully!")
    logger.info("  Output: %s", output_path)
    logger.info("  File size: %.1f MB", file_size_mb)
    logger.info("  Plotly.js: %s", "inline" if inline_plotly else "CDN")
    logger.info("  Sections: 5 (State, County, Place, QA, Methodology)")
    logger.info("  Scenarios: %s", ", ".join(scenario_labels))
    logger.info("=" * 60)

    return output_path


# ===================================================================
# CLI
# ===================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build interactive HTML report for ND population projections. "
            "Generates a single self-contained HTML file with Plotly charts "
            "and tabbed navigation."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        help=(
            f"Scenarios to include (default: all). "
            f"Options: {', '.join(SCENARIOS.keys())}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports",
        help="Output directory (default: data/exports/)",
    )
    parser.add_argument(
        "--no-plotly-js",
        action="store_true",
        help="Use CDN Plotly.js instead of inline (smaller file, requires internet)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the interactive report builder."""
    args = parse_args()

    try:
        output_path = build_report(
            scenarios=args.scenarios,
            output_dir=args.output_dir,
            inline_plotly=not args.no_plotly_js,
        )
        logger.info("Done. Report at: %s", output_path)
        return 0
    except Exception:
        logger.exception("Failed to build interactive report")
        return 1


if __name__ == "__main__":
    sys.exit(main())
