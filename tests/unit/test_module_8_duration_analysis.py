"""
Unit tests for Module 8 duration analysis helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "sdc_2024_replication" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from statistical_analysis.module_8_duration_analysis import (  # noqa: E402
    ModuleResult,
    drop_states_missing_post_2020,
)


def test_drop_states_missing_post_2020_excludes_incomplete_states():
    result = ModuleResult(module_id="8", analysis_name="duration_analysis_survival")
    df = pd.DataFrame(
        {
            "fiscal_year": [2020, 2021, 2022, 2021, 2020, 2021],
            "state": ["Alpha", "Alpha", "Alpha", "Beta", "Gamma", "Beta"],
            "nationality": ["Total", "Total", "Total", "Total", "Total", "Somalia"],
            "arrivals": [10, 12, 14, 3, 1, 2],
        }
    )

    filtered = drop_states_missing_post_2020(df, result)

    assert set(filtered["state"].unique()) == {"Alpha"}
    assert result.warnings
    assert "Beta" in result.warnings[0]
    assert "Gamma" in result.warnings[0]


def test_drop_states_missing_post_2020_no_post_years_passthrough():
    result = ModuleResult(module_id="8", analysis_name="duration_analysis_survival")
    df = pd.DataFrame(
        {
            "fiscal_year": [2019, 2020],
            "state": ["Alpha", "Beta"],
            "nationality": ["Total", "Total"],
            "arrivals": [8, 9],
        }
    )

    filtered = drop_states_missing_post_2020(df, result)

    pd.testing.assert_frame_equal(filtered, df)
    assert result.warnings == []
