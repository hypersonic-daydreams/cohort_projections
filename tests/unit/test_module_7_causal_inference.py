"""
Unit tests for Module 7 causal inference helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "sdc_2024_replication" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "statistical_analysis"))

from statistical_analysis.module_7_causal_inference import (  # noqa: E402
    ModuleResult,
    estimate_event_study,
    estimate_event_study_extended,
    prepare_travel_ban_did_data,
)


def test_prepare_travel_ban_did_data_excludes_pseudo_nationalities():
    result = ModuleResult(module_id="7", analysis_name="test")
    df_refugee = pd.DataFrame(
        {
            "fiscal_year": [2017, 2018, 2017, 2018, 2017, 2018],
            "nationality": ["Iran", "Iran", "Canada", "Canada", "Total", "Fy Refugee Admissions"],
            "arrivals": [10, 3, 8, 9, 100, 200],
        }
    )

    prepared = prepare_travel_ban_did_data(df_refugee, result)

    assert "Total" not in set(prepared["nationality"])
    assert "Fy Refugee Admissions" not in set(prepared["nationality"])
    assert set(prepared["nationality"]) == {"Iran", "Canada"}

    # Decision record created when pseudo-nationalities are present.
    decision_ids = {d.get("decision_id") for d in result.decisions}
    assert "D001A" in decision_ids


def test_event_study_max_year_filters_years():
    # Build a small synthetic nationality-year panel through 2024.
    years = list(range(2016, 2025))
    nationalities = ["Iran", "ControlA", "ControlB", "ControlC", "ControlD"]
    rows = []
    for year in years:
        for nat in nationalities:
            base = 120 if nat == "Iran" else 80
            arrivals = 10 if (nat == "Iran" and year >= 2018) else base + (year - 2016)
            rows.append({"fiscal_year": year, "nationality": nat, "arrivals": arrivals})
    df_refugee = pd.DataFrame(rows)

    result = ModuleResult(module_id="7", analysis_name="test")
    df_did = prepare_travel_ban_did_data(df_refugee, result)

    _, es_df_primary = estimate_event_study(df_did, result, max_year=2019)
    assert es_df_primary["year"].max() == 2019

    _, es_df_extended = estimate_event_study_extended(df_did, result, max_year=2024)
    assert es_df_extended["year"].max() == 2024
