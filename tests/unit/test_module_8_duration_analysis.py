"""
Unit tests for Module 8 duration analysis helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("lifelines", reason="lifelines not installed")

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "sdc_2024_replication" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from statistical_analysis.module_8_duration_analysis import (  # noqa: E402
    ModuleResult,
    drop_states_missing_post_2020,
    identify_immigration_waves,
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


def test_identify_immigration_waves_fill_missing_breaks_nonconsecutive_years():
    df = pd.DataFrame(
        {
            "fiscal_year": [2002, 2004],
            "state": ["Alpha", "Alpha"],
            "nationality": ["Somalia", "Somalia"],
            "arrivals": [10, 10],
        }
    )

    waves = identify_immigration_waves(
        df,
        threshold_pct=50.0,
        min_wave_years=2,
        fill_missing_years=True,
        gap_tolerance_years=0,
    )

    assert len(waves) == 0


def test_identify_immigration_waves_gap_tolerance_merges_pause_year():
    df = pd.DataFrame(
        {
            "fiscal_year": [2002, 2004],
            "state": ["Alpha", "Alpha"],
            "nationality": ["Somalia", "Somalia"],
            "arrivals": [10, 10],
        }
    )

    waves = identify_immigration_waves(
        df,
        threshold_pct=50.0,
        min_wave_years=2,
        fill_missing_years=True,
        gap_tolerance_years=1,
    )

    assert len(waves) == 1
    wave = waves.iloc[0].to_dict()
    assert wave["wave_start"] == 2002
    assert wave["wave_end"] == 2004
    assert wave["duration_years"] == 3
    assert wave["n_above_threshold_years"] == 2


def test_identify_immigration_waves_gap_tolerance_requires_fill_missing_years():
    df = pd.DataFrame(
        {
            "fiscal_year": [2002, 2003],
            "state": ["Alpha", "Alpha"],
            "nationality": ["Somalia", "Somalia"],
            "arrivals": [10, 10],
        }
    )

    try:
        identify_immigration_waves(
            df,
            threshold_pct=50.0,
            min_wave_years=2,
            fill_missing_years=False,
            gap_tolerance_years=1,
        )
    except ValueError as exc:
        assert "requires fill_missing_years=True" in str(exc)
    else:  # pragma: no cover
        raise AssertionError(
            "Expected ValueError when gap tolerance used without fill_missing_years"
        )
