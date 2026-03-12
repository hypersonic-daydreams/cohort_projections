"""Tests for build_experiment_dashboard.py.

Validates catalog parsing, data loading, figure generation, and HTML assembly
for the experiment dashboard.
"""

from __future__ import annotations

# Setup paths before importing dashboard module
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "exports"))

from scripts.analysis.build_experiment_dashboard import (  # noqa: E402
    EXPERIMENT_COLORS,
    SLUG_TO_EXP,
    build_html,
    parse_experiment_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
MINIMAL_CATALOG = textwrap.dedent("""\
    # Experiment Catalog

    ## Tier 1 — High Priority

    ### EXP-A: Convergence Medium Hold Extension

    | Field | Value |
    |-------|-------|
    | Slug | `convergence-medium-hold-5` |
    | Parameter | `convergence_medium_hold: 3 → 5` |
    | Hypothesis | Extending medium-rate hold will reduce overshoot. |
    | Expected improvement | `county_mape_rural` |
    | Risk areas | Fast-growth counties may under-project |
    | Rationale | Tests whether more persistence helps. |
    | Results | `needs_human_review` |

    ### EXP-B: College Blend Factor 0.7

    | Field | Value |
    |-------|-------|
    | Slug | `college-blend-70` |
    | Parameter | `college_blend_factor: 0.5 → 0.7` |
    | Hypothesis | More smoothing reduces volatility. |
    | Expected improvement | `county_mape_urban_college` |
    | Risk areas | Over-smoothing |
    | Rationale | College counties show widest variance. |
    | Results | `passed_all_gates` |

    ## Tier 2 — Medium Priority

    ### EXP-C: GQ Fraction Sensitivity (0.75)

    | Field | Value |
    |-------|-------|
    | Slug | `gq-fraction-75` |
    | Parameter | `gq_correction.fraction: 1.0 → 0.75` |
    | Hypothesis | Partial GQ subtraction fits better. |
    | Notes | **Not benchmark-testable as config-only.** |

    ## Tier 3 — Exploratory

    ### EXP-I: College Blend Factor 0.3

    | Field | Value |
    |-------|-------|
    | Slug | `college-blend-30` |
    | Parameter | `college_blend_factor: 0.5 → 0.3` |
    | Hypothesis | Less smoothing preserves signal. |

    ### Generated Experiment Ideas

    | ID | Slug | Description | Source |
    |----|------|-------------|--------|
    | EXP-J | `convergence-medium-hold-4` | Test hold=4 compromise | EXP-A result |
""")


@pytest.fixture()
def catalog_file(tmp_path: Path) -> Path:
    """Write minimal catalog to a temp file."""
    p = tmp_path / "experiment-catalog.md"
    p.write_text(MINIMAL_CATALOG, encoding="utf-8")
    return p


@pytest.fixture()
def empty_dashboard_data() -> dict:
    """Minimal data structure for HTML rendering with no experiments."""
    return {
        "catalog": [],
        "exp_outcomes": {},
        "scorecards": {},
        "comparisons": {},
        "projection_curves": {},
        "horizon_summaries": {},
        "county_metrics": {},
        "state_metrics": {},
        "champion_scorecard": None,
    }


@pytest.fixture()
def dashboard_data_with_experiments(tmp_path: Path) -> dict:
    """Data structure with synthetic experiment results."""
    champion_sc = {
        "run_id": "br-test",
        "method_id": "m2026",
        "config_id": "cfg-baseline",
        "scope": "county",
        "status_at_run": "champion",
        "state_ape_recent_short": 1.0,
        "state_ape_recent_medium": 2.5,
        "state_signed_bias_recent": -1.2,
        "county_mape_overall": 8.8,
        "county_mape_urban_college": 12.0,
        "county_mape_rural": 7.3,
        "county_mape_bakken": 19.0,
        "negative_population_violations": 0,
        "scenario_order_violations": 0,
        "aggregation_violations": 0,
        "sensitivity_instability_flag": False,
        "sentinel_cass_mape": 13.0,
        "sentinel_grand_forks_mape": 15.0,
        "sentinel_ward_mape": 15.0,
        "sentinel_burleigh_mape": 7.0,
        "sentinel_williams_mape": 23.0,
        "sentinel_mckenzie_mape": 29.0,
    }
    challenger_sc = {**champion_sc, "status_at_run": "experiment", "method_id": "m2026r1", "config_id": "cfg-blend-70", "county_mape_overall": 8.7}

    catalog = [
        {"exp_id": "EXP-A", "name": "Convergence Hold", "tier": 1, "slug": "convergence-medium-hold-5", "parameter": "hold: 3→5", "hypothesis": "", "results_text": "", "config_only": True},
        {"exp_id": "EXP-B", "name": "Blend 0.7", "tier": 1, "slug": "college-blend-70", "parameter": "blend: 0.5→0.7", "hypothesis": "", "results_text": "passed_all_gates", "config_only": True},
    ]

    # Synthetic projection curves
    curves_df = pd.DataFrame({
        "origin_year": [2005] * 4 + [2010] * 4,
        "method": ["m2026", "m2026", "m2026r1", "m2026r1"] * 2,
        "year": [2005, 2010, 2005, 2010, 2010, 2015, 2010, 2015],
        "projected_state": [650000, 670000, 648000, 668000, 670000, 700000, 668000, 698000],
    })

    # Synthetic horizon summary
    hz_df = pd.DataFrame({
        "horizon": [1, 5, 10, 1, 5, 10],
        "method": ["m2026", "m2026", "m2026", "m2026r1", "m2026r1", "m2026r1"],
        "n_origins": [4, 4, 3, 4, 4, 3],
        "mean_state_ape": [1.0, 3.0, 5.0, 0.8, 2.5, 4.5],
        "mean_county_mape": [1.1, 4.0, 8.0, 0.9, 3.5, 7.5],
        "mean_state_mpe": [-0.5, -1.5, -3.0, -0.3, -1.0, -2.5],
        "mean_county_mpe": [-0.3, -1.0, -2.0, -0.2, -0.8, -1.8],
    })

    # Synthetic state metrics (for actuals)
    state_df = pd.DataFrame({
        "origin_year": [2005, 2005],
        "method": ["m2026", "m2026"],
        "validation_year": [2010, 2015],
        "horizon": [5, 10],
        "projected_state": [670000, 700000],
        "actual_state": [672000, 710000],
        "error": [-2000, -10000],
        "pct_error": [-0.3, -1.4],
        "abs_pct_error": [0.3, 1.4],
    })

    # Synthetic county metrics
    county_df = pd.DataFrame({
        "origin_year": [2005, 2005, 2005, 2005],
        "method": ["m2026", "m2026r1", "m2026", "m2026r1"],
        "validation_year": [2010, 2010, 2010, 2010],
        "horizon": [5, 5, 5, 5],
        "county_fips": ["38017", "38017", "38105", "38105"],
        "county_name": ["Cass", "Cass", "Williams", "Williams"],
        "projected": [200000, 199000, 35000, 34500],
        "actual": [201000, 201000, 36000, 36000],
        "error": [-1000, -2000, -1000, -1500],
        "pct_error": [-0.5, -1.0, -2.8, -4.2],
        "category": ["Urban/College", "Urban/College", "Bakken", "Bakken"],
    })

    comparison = {
        "champion_method_id": "m2026",
        "champion_config_id": "cfg-baseline",
        "challengers": [{
            "method_id": "m2026r1",
            "config_id": "cfg-blend-70",
            "deltas": {"county_mape_overall": -0.09, "county_mape_rural": 0.02, "county_mape_bakken": 0.33, "county_mape_urban_college": -1.78},
            "hard_constraint_regression": False,
        }],
    }

    return {
        "catalog": catalog,
        "exp_outcomes": {"EXP-B": {"outcome": "passed_all_gates", "run_id": "br-test", "key_metrics": "", "config_delta": "blend=0.7", "interpretation": ""}},
        "scorecards": {"EXP-B": [champion_sc, challenger_sc]},
        "comparisons": {"EXP-B": comparison},
        "projection_curves": {"EXP-B": curves_df},
        "horizon_summaries": {"EXP-B": hz_df},
        "county_metrics": {"EXP-B": county_df},
        "state_metrics": {"EXP-B": state_df},
        "champion_scorecard": champion_sc,
    }


# ---------------------------------------------------------------------------
# Catalog parser tests
# ---------------------------------------------------------------------------
class TestParseCatalog:
    """Tests for parse_experiment_catalog()."""

    def test_parses_all_experiments(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        exp_ids = {e["exp_id"] for e in entries}
        assert "EXP-A" in exp_ids
        assert "EXP-B" in exp_ids
        assert "EXP-C" in exp_ids
        assert "EXP-I" in exp_ids
        assert "EXP-J" in exp_ids

    def test_extracts_correct_count(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        assert len(entries) == 5

    def test_assigns_correct_tiers(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        by_id = {e["exp_id"]: e for e in entries}
        assert by_id["EXP-A"]["tier"] == 1
        assert by_id["EXP-B"]["tier"] == 1
        assert by_id["EXP-C"]["tier"] == 2
        assert by_id["EXP-I"]["tier"] == 3

    def test_extracts_slug(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        by_id = {e["exp_id"]: e for e in entries}
        assert by_id["EXP-A"]["slug"] == "convergence-medium-hold-5"
        assert by_id["EXP-B"]["slug"] == "college-blend-70"

    def test_extracts_parameter(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        by_id = {e["exp_id"]: e for e in entries}
        assert "convergence_medium_hold" in by_id["EXP-A"]["parameter"]

    def test_detects_non_config_only(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        by_id = {e["exp_id"]: e for e in entries}
        assert by_id["EXP-A"]["config_only"] is True
        assert by_id["EXP-C"]["config_only"] is False

    def test_parses_generated_experiments(self, catalog_file: Path) -> None:
        entries = parse_experiment_catalog(catalog_file)
        by_id = {e["exp_id"]: e for e in entries}
        assert "EXP-J" in by_id
        assert by_id["EXP-J"]["slug"] == "convergence-medium-hold-4"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------
class TestConstants:
    """Validate constant integrity."""

    def test_all_slug_to_exp_values_have_colors(self) -> None:
        for exp_id in SLUG_TO_EXP.values():
            assert exp_id in EXPERIMENT_COLORS, f"Missing color for {exp_id}"

    def test_slug_to_exp_covers_all_colors(self) -> None:
        exp_ids_in_slugs = set(SLUG_TO_EXP.values())
        for exp_id in EXPERIMENT_COLORS:
            assert exp_id in exp_ids_in_slugs, f"Color defined for {exp_id} but no slug mapping"


# ---------------------------------------------------------------------------
# HTML generation tests
# ---------------------------------------------------------------------------
class TestBuildHtml:
    """Tests for HTML dashboard assembly."""

    def test_empty_data_produces_valid_html(self, empty_dashboard_data: dict) -> None:
        html = build_html(empty_dashboard_data, use_cdn_plotly=True)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_all_five_tabs(self, empty_dashboard_data: dict) -> None:
        html = build_html(empty_dashboard_data, use_cdn_plotly=True)
        for tab_id in ["tracker", "spaghetti", "scorecard", "horizon", "county"]:
            assert f'id="{tab_id}"' in html, f"Missing tab: {tab_id}"

    def test_contains_nav_links(self, empty_dashboard_data: dict) -> None:
        html = build_html(empty_dashboard_data, use_cdn_plotly=True)
        assert 'data-tab="tracker"' in html
        assert 'data-tab="spaghetti"' in html

    def test_includes_plotly_cdn(self, empty_dashboard_data: dict) -> None:
        html = build_html(empty_dashboard_data, use_cdn_plotly=True)
        assert "plotly" in html.lower()

    def test_experiment_data_renders_badges(self, dashboard_data_with_experiments: dict) -> None:
        html = build_html(dashboard_data_with_experiments, use_cdn_plotly=True)
        assert "Passed All Gates" in html

    def test_experiment_data_renders_deltas(self, dashboard_data_with_experiments: dict) -> None:
        html = build_html(dashboard_data_with_experiments, use_cdn_plotly=True)
        # EXP-B overall delta should appear
        assert "-0.09" in html or "-0.0900" in html

    def test_champion_info_in_header(self, dashboard_data_with_experiments: dict) -> None:
        html = build_html(dashboard_data_with_experiments, use_cdn_plotly=True)
        assert "m2026" in html
        assert "cfg-baseline" in html

    def test_scorecard_table_has_metrics(self, dashboard_data_with_experiments: dict) -> None:
        html = build_html(dashboard_data_with_experiments, use_cdn_plotly=True)
        assert "County MAPE Overall" in html
        assert "County MAPE Bakken" in html
        assert "State APE" in html

    def test_plotly_charts_present(self, dashboard_data_with_experiments: dict) -> None:
        html = build_html(dashboard_data_with_experiments, use_cdn_plotly=True)
        assert "plotly-graph-div" in html

    def test_kpi_cards_show_counts(self, dashboard_data_with_experiments: dict) -> None:
        html = build_html(dashboard_data_with_experiments, use_cdn_plotly=True)
        assert "Total Experiments" in html
        assert "Completed" in html
        assert "Passed All Gates" in html
