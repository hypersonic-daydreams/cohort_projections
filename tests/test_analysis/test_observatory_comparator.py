"""Tests for ObservatoryComparator (cohort_projections.analysis.observatory.comparator).

Covers ranking, delta computation, Pareto frontier, county-group impact,
best-variant-per-group, full comparison, formatting, and edge cases
(empty data, missing metrics, single-run datasets).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cohort_projections.analysis.observatory.comparator import (
    METRIC_COLUMNS,
    ComparisonResult,
    ObservatoryComparator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scorecards(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic scorecards DataFrame."""
    return pd.DataFrame(rows)


def _make_store(scorecards: pd.DataFrame) -> MagicMock:
    """Return a mock ResultsStore that returns the given scorecards."""
    store = MagicMock()
    store.get_consolidated_scorecards.return_value = scorecards
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHAMPION_ROW = {
    "run_id": "run-champ",
    "method_id": "m2026",
    "config_id": "cfg-baseline",
    "status_at_run": "champion",
    "county_mape_overall": 8.5,
    "county_mape_rural": 7.0,
    "county_mape_bakken": 19.0,
    "county_mape_urban_college": 12.0,
    "state_ape_recent_short": 1.0,
    "state_ape_recent_medium": 2.5,
}

CHALLENGER_A = {
    "run_id": "run-a",
    "method_id": "m2026r1",
    "config_id": "cfg-a",
    "status_at_run": "experiment",
    "county_mape_overall": 8.2,
    "county_mape_rural": 7.1,
    "county_mape_bakken": 18.5,
    "county_mape_urban_college": 10.5,
    "state_ape_recent_short": 1.1,
    "state_ape_recent_medium": 2.3,
}

CHALLENGER_B = {
    "run_id": "run-b",
    "method_id": "m2026r1",
    "config_id": "cfg-b",
    "status_at_run": "experiment",
    "county_mape_overall": 8.8,
    "county_mape_rural": 6.5,
    "county_mape_bakken": 20.0,
    "county_mape_urban_college": 11.0,
    "state_ape_recent_short": 0.9,
    "state_ape_recent_medium": 2.7,
}


@pytest.fixture()
def scorecards_3run() -> pd.DataFrame:
    """Three-run scorecards DataFrame (champion + 2 challengers)."""
    return _make_scorecards([CHAMPION_ROW, CHALLENGER_A, CHALLENGER_B])


@pytest.fixture()
def comparator_3run(scorecards_3run: pd.DataFrame) -> ObservatoryComparator:
    """Comparator backed by a 3-run mock store."""
    store = _make_store(scorecards_3run)
    return ObservatoryComparator(store=store)


# ---------------------------------------------------------------------------
# TestComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    """Basic tests for the ComparisonResult dataclass."""

    def test_default_fields(self) -> None:
        cr = ComparisonResult()
        assert cr.ranking.empty
        assert cr.deltas.empty
        assert cr.county_group_impact.empty
        assert cr.pareto_runs == []
        assert cr.best_per_group == {}
        assert cr.summary == {}


# ---------------------------------------------------------------------------
# TestRanking
# ---------------------------------------------------------------------------


class TestRanking:
    """Tests for rank_all and rank_by."""

    def test_rank_all_returns_all_runs(self, comparator_3run: ObservatoryComparator) -> None:
        ranked = comparator_3run.rank_all()
        assert len(ranked) == 3
        assert "run_id" in ranked.columns

    def test_rank_all_sorted_by_primary(self, comparator_3run: ObservatoryComparator) -> None:
        ranked = comparator_3run.rank_all()
        rank_col = "rank_county_mape_overall"
        assert rank_col in ranked.columns
        # First row should have rank 1
        assert ranked.iloc[0][rank_col] == 1

    def test_rank_all_has_rank_columns(self, comparator_3run: ObservatoryComparator) -> None:
        ranked = comparator_3run.rank_all()
        metric_cols = [c for c in METRIC_COLUMNS if c in ranked.columns]
        for mc in metric_cols:
            assert f"rank_{mc}" in ranked.columns

    def test_rank_by_specific_metric(self, comparator_3run: ObservatoryComparator) -> None:
        ranked = comparator_3run.rank_by("county_mape_rural")
        assert "rank" in ranked.columns
        assert len(ranked) == 3
        # Best rural MAPE is 6.5 from challenger B
        assert ranked.iloc[0]["run_id"] == "run-b"

    def test_rank_by_invalid_metric_raises(self, comparator_3run: ObservatoryComparator) -> None:
        with pytest.raises(ValueError, match="not found"):
            comparator_3run.rank_by("nonexistent_metric")

    def test_rank_all_empty_scorecards(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        assert comp.rank_all().empty

    def test_rank_by_empty_scorecards(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        assert comp.rank_by("county_mape_overall").empty


# ---------------------------------------------------------------------------
# TestDeltas
# ---------------------------------------------------------------------------


class TestDeltas:
    """Tests for compute_deltas."""

    def test_compute_deltas_auto_champion(self, comparator_3run: ObservatoryComparator) -> None:
        deltas = comparator_3run.compute_deltas()
        assert not deltas.empty
        assert "delta_county_mape_overall" in deltas.columns

    def test_champion_delta_is_zero(self, comparator_3run: ObservatoryComparator) -> None:
        deltas = comparator_3run.compute_deltas()
        champ_row = deltas[deltas["run_id"] == "run-champ"]
        assert float(champ_row["delta_county_mape_overall"].iloc[0]) == pytest.approx(0.0)

    def test_challenger_has_negative_delta(self, comparator_3run: ObservatoryComparator) -> None:
        deltas = comparator_3run.compute_deltas()
        run_a = deltas[deltas["run_id"] == "run-a"]
        # run-a overall = 8.2, champion = 8.5 => delta = -0.3
        assert float(run_a["delta_county_mape_overall"].iloc[0]) == pytest.approx(-0.3)

    def test_explicit_champion(self, comparator_3run: ObservatoryComparator) -> None:
        deltas = comparator_3run.compute_deltas(champion_run_id="run-a")
        # Now deltas are relative to run-a (8.2)
        champ_row = deltas[deltas["run_id"] == "run-champ"]
        assert float(champ_row["delta_county_mape_overall"].iloc[0]) == pytest.approx(0.3)

    def test_invalid_champion_raises(self, comparator_3run: ObservatoryComparator) -> None:
        with pytest.raises(ValueError, match="not found"):
            comparator_3run.compute_deltas(champion_run_id="nonexistent")

    def test_deltas_empty_scorecards(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        assert comp.compute_deltas().empty


# ---------------------------------------------------------------------------
# TestPareto
# ---------------------------------------------------------------------------


class TestPareto:
    """Tests for pareto_frontier."""

    def test_pareto_frontier(self, comparator_3run: ObservatoryComparator) -> None:
        pareto = comparator_3run.pareto_frontier(
            "county_mape_overall", "county_mape_rural"
        )
        assert not pareto.empty
        # run-a (8.2, 7.1) is Pareto-optimal vs run-champ (8.5, 7.0) and run-b (8.8, 6.5)
        # run-b (8.8, 6.5) is Pareto-optimal (best rural)
        # run-a (8.2, 7.1) is Pareto-optimal (best overall)
        # run-champ (8.5, 7.0) is dominated by... nothing strictly,
        #   since run-a is better on overall but worse on rural.
        # Actually: run-champ (8.5, 7.0) — run-a is (8.2, 7.1) — a is better on x but worse on y
        # So champ is NOT dominated. All 3 are Pareto-optimal.
        pareto_ids = set(pareto["run_id"].tolist())
        assert len(pareto_ids) >= 2

    def test_pareto_invalid_metric_raises(self, comparator_3run: ObservatoryComparator) -> None:
        with pytest.raises(ValueError, match="not found"):
            comparator_3run.pareto_frontier("county_mape_overall", "fake_metric")

    def test_pareto_single_run(self) -> None:
        sc = _make_scorecards([CHAMPION_ROW])
        store = _make_store(sc)
        comp = ObservatoryComparator(store=store)
        pareto = comp.pareto_frontier("county_mape_overall", "county_mape_rural")
        assert len(pareto) == 1

    def test_pareto_with_dominated_point(self) -> None:
        """One run strictly dominates another."""
        rows = [
            {**CHAMPION_ROW, "run_id": "good", "county_mape_overall": 5.0, "county_mape_rural": 5.0},
            {**CHAMPION_ROW, "run_id": "bad", "county_mape_overall": 6.0, "county_mape_rural": 6.0},
        ]
        sc = _make_scorecards(rows)
        store = _make_store(sc)
        comp = ObservatoryComparator(store=store)
        pareto = comp.pareto_frontier("county_mape_overall", "county_mape_rural")
        assert len(pareto) == 1
        assert pareto.iloc[0]["run_id"] == "good"


# ---------------------------------------------------------------------------
# TestCountyGroupImpact
# ---------------------------------------------------------------------------


class TestCountyGroupImpact:
    """Tests for county_group_impact."""

    def test_county_group_impact(self, comparator_3run: ObservatoryComparator) -> None:
        impact = comparator_3run.county_group_impact()
        assert not impact.empty
        # Should have delta columns for county groups
        assert any(c.startswith("delta_") for c in impact.columns)

    def test_county_group_impact_empty(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        assert comp.county_group_impact().empty


# ---------------------------------------------------------------------------
# TestBestVariantPerGroup
# ---------------------------------------------------------------------------


class TestBestVariantPerGroup:
    """Tests for best_variant_per_group."""

    def test_best_variant_per_group(self, comparator_3run: ObservatoryComparator) -> None:
        best = comparator_3run.best_variant_per_group()
        assert isinstance(best, dict)
        assert "overall" in best
        # Best overall is run-a (8.2)
        assert best["overall"] == "run-a"
        # Best rural is run-b (6.5)
        assert best["Rural"] == "run-b"

    def test_best_variant_empty(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        assert comp.best_variant_per_group() == {}


# ---------------------------------------------------------------------------
# TestFullComparison
# ---------------------------------------------------------------------------


class TestFullComparison:
    """Tests for full_comparison."""

    def test_full_comparison_returns_result(self, comparator_3run: ObservatoryComparator) -> None:
        result = comparator_3run.full_comparison()
        assert isinstance(result, ComparisonResult)
        assert not result.ranking.empty
        assert not result.deltas.empty
        assert result.summary["n_runs"] == 3

    def test_full_comparison_summary_has_champion(
        self, comparator_3run: ObservatoryComparator
    ) -> None:
        result = comparator_3run.full_comparison()
        assert result.summary["champion_run_id"] is not None

    def test_full_comparison_empty_scorecards(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        result = comp.full_comparison()
        assert result.ranking.empty
        assert result.summary == {}


# ---------------------------------------------------------------------------
# TestFormatting
# ---------------------------------------------------------------------------


class TestFormatting:
    """Tests for format_ranking_table and format_comparison_summary."""

    def test_format_ranking_table(self, comparator_3run: ObservatoryComparator) -> None:
        output = comparator_3run.format_ranking_table(top_n=5)
        assert "BENCHMARK RANKING" in output
        assert "run-" in output

    def test_format_ranking_table_empty(self) -> None:
        store = _make_store(pd.DataFrame())
        comp = ObservatoryComparator(store=store)
        output = comp.format_ranking_table()
        assert "No runs available" in output

    def test_format_comparison_summary(self, comparator_3run: ObservatoryComparator) -> None:
        result = comparator_3run.full_comparison()
        output = comparator_3run.format_comparison_summary(result)
        assert "OBSERVATORY COMPARISON SUMMARY" in output
        assert "Runs compared" in output
        assert "Champion" in output
        assert "METRIC RANGES" in output


# ---------------------------------------------------------------------------
# TestChampionDetection
# ---------------------------------------------------------------------------


class TestChampionDetection:
    """Tests for champion auto-detection."""

    def test_detects_champion_by_status(self, comparator_3run: ObservatoryComparator) -> None:
        sc = comparator_3run._load_all_scorecards()
        champ = comparator_3run._detect_champion(sc)
        assert champ == "run-champ"

    def test_fallback_to_best_primary_metric(self) -> None:
        """When no status_at_run == champion, uses best primary metric."""
        rows = [
            {**CHAMPION_ROW, "run_id": "r1", "status_at_run": "experiment", "county_mape_overall": 9.0},
            {**CHAMPION_ROW, "run_id": "r2", "status_at_run": "experiment", "county_mape_overall": 7.0},
        ]
        sc = _make_scorecards(rows)
        store = _make_store(sc)
        comp = ObservatoryComparator(store=store)
        champ = comp._detect_champion(sc)
        assert champ == "r2"

    def test_config_champion_override(self) -> None:
        """Config-specified champion_run_id takes precedence."""
        sc = _make_scorecards([CHAMPION_ROW, CHALLENGER_A])
        store = _make_store(sc)
        config = {"comparison": {"champion_run_id": "run-a"}}
        comp = ObservatoryComparator(store=store, config=config)
        resolved = comp._resolve_champion(sc)
        assert resolved == "run-a"
