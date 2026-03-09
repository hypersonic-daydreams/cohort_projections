from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from cohort_projections.analysis.benchmarking import (
    append_benchmark_index,
    append_promotion_history,
    build_comparison_to_champion,
    build_run_id,
    build_summary_scorecard,
    compute_prediction_intervals_generic,
    decision_file_is_approved,
    load_aliases,
    load_method_profile,
    render_benchmark_decision_record,
    update_alias_mapping,
    write_manifest,
)


def test_load_method_profile_validates_ids(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile_path = profile_dir / "m2026__cfg-test-v1.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "method_id": "m2026",
                "config_id": "cfg-test-v1",
                "status": "candidate",
                "resolved_config": {"alpha": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    profile = load_method_profile("m2026", "cfg-test-v1", profile_dir=profile_dir)
    assert profile["method_id"] == "m2026"
    assert profile["config_id"] == "cfg-test-v1"
    assert profile["profile_hash"]


def test_update_alias_mapping_writes_yaml(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.yaml"
    prior, current = update_alias_mapping(
        alias_name="county_champion",
        method_id="m2026",
        config_id="cfg-a",
        alias_path=alias_path,
    )

    assert prior is None
    assert current == {"method_id": "m2026", "config_id": "cfg-a"}
    aliases = load_aliases(alias_path)
    assert aliases["county_champion"]["method_id"] == "m2026"


def test_build_summary_scorecard_and_comparison() -> None:
    annual_state = pd.DataFrame(
        [
            {"origin_year": 2015, "method": "m2026", "validation_year": 2020, "horizon": 5, "pct_error": -1.0, "projected_state": 100.0, "actual_state": 101.0},
            {"origin_year": 2015, "method": "m2026", "validation_year": 2024, "horizon": 9, "pct_error": -2.0, "projected_state": 100.0, "actual_state": 102.0},
            {"origin_year": 2020, "method": "m2026", "validation_year": 2024, "horizon": 4, "pct_error": -0.5, "projected_state": 100.0, "actual_state": 100.5},
            {"origin_year": 2015, "method": "m2026r1", "validation_year": 2020, "horizon": 5, "pct_error": -0.8, "projected_state": 100.0, "actual_state": 100.8},
            {"origin_year": 2015, "method": "m2026r1", "validation_year": 2024, "horizon": 9, "pct_error": -1.2, "projected_state": 100.0, "actual_state": 101.2},
            {"origin_year": 2020, "method": "m2026r1", "validation_year": 2024, "horizon": 4, "pct_error": -0.3, "projected_state": 100.0, "actual_state": 100.3},
        ]
    )
    annual_county = pd.DataFrame(
        [
            {"origin_year": 2015, "method": "m2026", "validation_year": 2020, "horizon": 5, "county_fips": "38017", "county_name": "Cass", "projected": 50.0, "actual": 51.0, "pct_error": -1.9608},
            {"origin_year": 2015, "method": "m2026", "validation_year": 2020, "horizon": 5, "county_fips": "38105", "county_name": "Williams", "projected": 50.0, "actual": 50.0, "pct_error": 0.0},
            {"origin_year": 2015, "method": "m2026r1", "validation_year": 2020, "horizon": 5, "county_fips": "38017", "county_name": "Cass", "projected": 50.0, "actual": 50.5, "pct_error": -0.9901},
            {"origin_year": 2015, "method": "m2026r1", "validation_year": 2020, "horizon": 5, "county_fips": "38105", "county_name": "Williams", "projected": 50.0, "actual": 50.0, "pct_error": 0.0},
            {"origin_year": 2020, "method": "m2026", "validation_year": 2024, "horizon": 4, "county_fips": "38053", "county_name": "McKenzie", "projected": 50.0, "actual": 51.0, "pct_error": -1.9608},
            {"origin_year": 2020, "method": "m2026r1", "validation_year": 2024, "horizon": 4, "county_fips": "38053", "county_name": "McKenzie", "projected": 50.0, "actual": 50.2, "pct_error": -0.3984},
        ]
    )
    sensitivity_tornado = pd.DataFrame(
        [
            {"method": "m2026", "swing_state_error": 3.0, "mape_swing": 2.0},
            {"method": "m2026r1", "swing_state_error": 4.0, "mape_swing": 1.0},
        ]
    )
    method_profiles = {
        "m2026": {"config_id": "cfg-a", "status": "champion"},
        "m2026r1": {"config_id": "cfg-b", "status": "candidate"},
    }

    scorecard = build_summary_scorecard(
        annual_state=annual_state,
        annual_county=annual_county,
        sensitivity_tornado=sensitivity_tornado,
        method_profiles=method_profiles,
        scope="county",
        run_id="br-20260309-120000-m2026r1-abcdef0",
    )
    comparison = build_comparison_to_champion(scorecard, "m2026")

    assert set(scorecard["method_id"]) == {"m2026", "m2026r1"}
    challenger = scorecard[scorecard["method_id"] == "m2026r1"].iloc[0]
    assert challenger["state_ape_recent_short"] < scorecard[scorecard["method_id"] == "m2026"].iloc[0]["state_ape_recent_short"]
    assert comparison["challengers"][0]["method_id"] == "m2026r1"


def test_compute_prediction_intervals_generic_supports_multiple_methods() -> None:
    annual_county = pd.DataFrame(
        [
            {"method": "m2026", "horizon": 1, "pct_error": -1.0},
            {"method": "m2026", "horizon": 1, "pct_error": 1.0},
            {"method": "m2026r1", "horizon": 1, "pct_error": -0.5},
            {"method": "m2026r1", "horizon": 1, "pct_error": 0.5},
        ]
    )
    annual_state = pd.DataFrame(
        [
            {"method": "m2026", "horizon": 1, "pct_error": -2.0},
            {"method": "m2026r1", "horizon": 1, "pct_error": -1.0},
        ]
    )

    intervals = compute_prediction_intervals_generic(annual_county, annual_state)
    assert set(intervals["method"]) == {"m2026", "m2026r1"}
    assert set(intervals["level"]) == {"county", "state"}


def test_manifest_index_and_decision_rendering(tmp_path: Path) -> None:
    run_dir = tmp_path / "br-20260309-120000-m2026r1-abcdef0"
    run_dir.mkdir()
    manifest = {
        "run_id": "br-20260309-120000-m2026r1-abcdef0",
        "run_date": "2026-03-09",
        "scope": "county",
        "benchmark_label": "college_fix",
    }
    manifest_path = write_manifest(run_dir, manifest)

    scorecard = pd.DataFrame(
        [
            {
                "run_id": manifest["run_id"],
                "method_id": "m2026",
                "config_id": "cfg-a",
                "scope": "county",
                "status_at_run": "champion",
                "state_ape_recent_short": 1.0,
                "state_ape_recent_medium": 2.0,
                "state_signed_bias_recent": -1.0,
                "county_mape_overall": 3.0,
                "county_mape_urban_college": 4.0,
                "county_mape_rural": 2.0,
                "county_mape_bakken": 5.0,
                "sentinel_cass_mape": 1.0,
                "sentinel_grand_forks_mape": 1.0,
                "sentinel_ward_mape": 1.0,
                "sentinel_burleigh_mape": 1.0,
                "sentinel_williams_mape": 1.0,
                "sentinel_mckenzie_mape": 1.0,
                "negative_population_violations": 0,
                "scenario_order_violations": 0,
                "aggregation_violations": 0,
                "sensitivity_instability_flag": False,
            },
            {
                "run_id": manifest["run_id"],
                "method_id": "m2026r1",
                "config_id": "cfg-b",
                "scope": "county",
                "status_at_run": "candidate",
                "state_ape_recent_short": 0.5,
                "state_ape_recent_medium": 1.5,
                "state_signed_bias_recent": -0.5,
                "county_mape_overall": 2.5,
                "county_mape_urban_college": 3.0,
                "county_mape_rural": 2.0,
                "county_mape_bakken": 4.0,
                "sentinel_cass_mape": 0.5,
                "sentinel_grand_forks_mape": 0.5,
                "sentinel_ward_mape": 0.5,
                "sentinel_burleigh_mape": 0.5,
                "sentinel_williams_mape": 0.5,
                "sentinel_mckenzie_mape": 0.5,
                "negative_population_violations": 0,
                "scenario_order_violations": 0,
                "aggregation_violations": 0,
                "sensitivity_instability_flag": False,
            },
        ]
    )
    comparison = build_comparison_to_champion(scorecard, "m2026")
    markdown = render_benchmark_decision_record(manifest, scorecard, comparison)
    assert "m2026r1" in markdown

    index_path = tmp_path / "index.csv"
    append_benchmark_index(
        index_path=index_path,
        scorecard=scorecard,
        manifest_path=manifest_path,
        benchmark_label="college_fix",
        benchmark_contract_version="1.0",
        git_commit="abcdef012345",
        champion_method_id="m2026",
    )
    index_df = pd.read_csv(index_path)
    assert len(index_df) == 2


def test_decision_approval_and_promotion_history(tmp_path: Path) -> None:
    decision_path = tmp_path / "2026-03-09-m2026r1-vs-m2026.md"
    decision_path.write_text("| Status | Approved |\n", encoding="utf-8")
    assert decision_file_is_approved(decision_path) is True

    history_path = tmp_path / "promotion_history.csv"
    append_promotion_history(
        history_path=history_path,
        alias_name="county_champion",
        prior_mapping={"method_id": "m2026", "config_id": "cfg-a"},
        new_mapping={"method_id": "m2026r1", "config_id": "cfg-b"},
        decision_id="2026-03-09-m2026r1-vs-m2026",
    )
    history_df = pd.read_csv(history_path)
    assert history_df.iloc[0]["new_method_id"] == "m2026r1"


def test_build_run_id_includes_method_and_short_git() -> None:
    run_id = build_run_id(
        primary_method_id="m2026r1",
        git_commit="abcdef0123456789",
        now=pd.Timestamp("2026-03-09T12:00:00Z").to_pydatetime(),
    )
    assert run_id == "br-20260309-120000-m2026r1-abcdef0"
