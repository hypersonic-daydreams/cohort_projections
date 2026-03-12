"""Tests for the Observatory CLI (scripts/analysis/observatory.py).

Covers argument parsing, subcommand dispatch, graceful degradation when
modules are unavailable, and individual command handlers using mocked
store/comparator/recommender/catalog components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import observatory as cli_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_observatory_config(tmp_path: Path, history_dir: Path) -> Path:
    """Write a minimal observatory config YAML and return its path."""
    cfg = {
        "observatory": {
            "history_dir": str(history_dir),
        }
    }
    p = tmp_path / "observatory_config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


@pytest.fixture()
def history_dir(tmp_path: Path) -> Path:
    """Create a minimal benchmark history directory."""
    hdir = tmp_path / "benchmark_history"
    hdir.mkdir()
    index = pd.DataFrame({"run_id": ["run-001"], "run_date": ["2026-03-01"]})
    index.to_csv(hdir / "index.csv", index=False)

    rd = hdir / "run-001"
    rd.mkdir()
    sc = pd.DataFrame([{
        "method_id": "m2026",
        "config_id": "cfg-base",
        "status_at_run": "champion",
        "county_mape_overall": 8.5,
        "county_mape_rural": 7.0,
        "county_mape_bakken": 19.0,
        "county_mape_urban_college": 12.0,
        "state_ape_recent_short": 1.0,
        "state_ape_recent_medium": 2.5,
    }])
    sc.to_csv(rd / "summary_scorecard.csv", index=False)
    return hdir


@pytest.fixture()
def config_path(tmp_path: Path, history_dir: Path) -> Path:
    return _make_observatory_config(tmp_path, history_dir)


# ---------------------------------------------------------------------------
# TestBuildParser
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for CLI argument parsing."""

    def test_no_args_shows_help(self) -> None:
        """No subcommand should not raise — main returns 0."""
        rc = cli_mod.main(argv=[])
        assert rc == 0

    def test_status_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_compare_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["compare", "--top", "5"])
        assert args.command == "compare"
        assert args.top == 5

    def test_rank_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["rank", "county_mape_overall", "--top", "3"])
        assert args.command == "rank"
        assert args.metric == "county_mape_overall"
        assert args.top == 3

    def test_recommend_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["recommend"])
        assert args.command == "recommend"

    def test_run_pending_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["run-pending", "--dry-run"])
        assert args.command == "run-pending"
        assert args.dry_run is True

    def test_report_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["report", "--output", "/tmp/report.html"])
        assert args.command == "report"
        assert args.output == Path("/tmp/report.html")

    def test_refresh_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["refresh"])
        assert args.command == "refresh"

    def test_verbose_flag(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["-v", "status"])
        assert args.verbose is True

    def test_custom_config(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["--config", "/tmp/custom.yaml", "status"])
        assert args.config == Path("/tmp/custom.yaml")


# ---------------------------------------------------------------------------
# TestCmdStatus
# ---------------------------------------------------------------------------


class TestCmdStatus:
    """Tests for the status command handler."""

    def test_status_no_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["status"])
        rc = cli_mod.cmd_status(store=None, config={}, _args=args)
        assert rc == 1
        assert "unavailable" in capsys.readouterr().out

    def test_status_with_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        store.get_index.return_value = pd.DataFrame({
            "run_id": ["run-001"],
            "run_date": ["2026-03-01"],
        })
        store.get_run_ids.return_value = ["run-001"]
        args = cli_mod.build_parser().parse_args(["status"])

        with patch.object(cli_mod, "_load_variant_catalog", return_value=None):
            rc = cli_mod.cmd_status(store=store, config={}, _args=args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Completed runs: 1" in out


# ---------------------------------------------------------------------------
# TestCmdCompare
# ---------------------------------------------------------------------------


class TestCmdCompare:
    """Tests for the compare command handler."""

    def test_compare_no_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["compare"])
        rc = cli_mod.cmd_compare(store=None, config={}, args=args)
        assert rc == 1

    def test_compare_no_comparator(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        args = cli_mod.build_parser().parse_args(["compare"])
        with patch.object(cli_mod, "_load_comparator", return_value=None):
            rc = cli_mod.cmd_compare(store=store, config={}, args=args)
        assert rc == 1

    def test_compare_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        comparator = MagicMock()
        result = MagicMock()
        comparator.full_comparison.return_value = result
        comparator.format_comparison_summary.return_value = "COMPARISON OUTPUT"

        args = cli_mod.build_parser().parse_args(["compare"])
        with (
            patch.object(cli_mod, "_load_comparator", return_value=comparator),
            patch.object(cli_mod, "_load_recommender", return_value=None),
            patch.object(cli_mod, "_load_report_class", return_value=None),
        ):
            rc = cli_mod.cmd_compare(store=store, config={}, args=args)

        assert rc == 0
        assert "COMPARISON OUTPUT" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# TestCmdRank
# ---------------------------------------------------------------------------


class TestCmdRank:
    """Tests for the rank command handler."""

    def test_rank_no_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["rank", "county_mape_overall"])
        rc = cli_mod.cmd_rank(store=None, config={}, args=args)
        assert rc == 1

    def test_rank_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        comparator = MagicMock()
        rank_df = pd.DataFrame({
            "run_id": ["r1", "r2"],
            "county_mape_overall": [8.0, 9.0],
            "rank": [1, 2],
        })
        comparator.rank_by.return_value = rank_df

        args = cli_mod.build_parser().parse_args(["rank", "county_mape_overall"])
        with patch.object(cli_mod, "_load_comparator", return_value=comparator):
            rc = cli_mod.cmd_rank(store=store, config={}, args=args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "r1" in out


# ---------------------------------------------------------------------------
# TestCmdRecommend
# ---------------------------------------------------------------------------


class TestCmdRecommend:
    """Tests for the recommend command handler."""

    def test_recommend_no_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["recommend"])
        rc = cli_mod.cmd_recommend(store=None, config={}, _args=args)
        assert rc == 1

    def test_recommend_no_recommender(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        args = cli_mod.build_parser().parse_args(["recommend"])
        with (
            patch.object(cli_mod, "_load_comparator", return_value=None),
            patch.object(cli_mod, "_load_recommender", return_value=None),
        ):
            rc = cli_mod.cmd_recommend(store=store, config={}, _args=args)
        assert rc == 1

    def test_recommend_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        recommender = MagicMock()
        recommender.suggest_next_experiments.return_value = []

        args = cli_mod.build_parser().parse_args(["recommend"])
        with (
            patch.object(cli_mod, "_load_comparator", return_value=MagicMock()),
            patch.object(cli_mod, "_load_recommender", return_value=recommender),
        ):
            rc = cli_mod.cmd_recommend(store=store, config={}, _args=args)

        assert rc == 0
        assert "No recommendations" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# TestCmdRunPending
# ---------------------------------------------------------------------------


class TestCmdRunPending:
    """Tests for the run-pending command handler."""

    def test_run_pending_no_catalog(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["run-pending"])
        with patch.object(cli_mod, "_load_variant_catalog", return_value=None):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)
        assert rc == 1

    def test_run_pending_nothing_untested(self, capsys: pytest.CaptureFixture[str]) -> None:
        catalog = MagicMock()
        catalog.get_untested.return_value = []

        args = cli_mod.build_parser().parse_args(["run-pending"])
        with patch.object(cli_mod, "_load_variant_catalog", return_value=catalog):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)

        assert rc == 0
        assert "No untested" in capsys.readouterr().out

    def test_run_pending_dry_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        catalog = MagicMock()
        catalog.get_untested.return_value = [
            {"variant_id": "exp-x", "parameter": "alpha", "value": 0.5, "hypothesis": "test"},
        ]

        args = cli_mod.build_parser().parse_args(["run-pending", "--dry-run"])
        with patch.object(cli_mod, "_load_variant_catalog", return_value=catalog):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Dry run" in out


# ---------------------------------------------------------------------------
# TestCmdRefresh
# ---------------------------------------------------------------------------


class TestCmdRefresh:
    """Tests for the refresh command handler."""

    def test_refresh_no_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["refresh"])
        rc = cli_mod.cmd_refresh(store=None, config={}, _args=args)
        assert rc == 1

    def test_refresh_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        store.get_run_ids.return_value = ["r1"]
        store.get_consolidated_scorecards.return_value = pd.DataFrame({"x": [1]})
        store.get_consolidated_county_metrics.return_value = pd.DataFrame()
        store.get_consolidated_state_metrics.return_value = pd.DataFrame()
        store.get_consolidated_projection_curves.return_value = pd.DataFrame()
        store.get_consolidated_sensitivity_summary.return_value = pd.DataFrame()

        args = cli_mod.build_parser().parse_args(["refresh"])
        rc = cli_mod.cmd_refresh(store=store, config={}, _args=args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Refresh complete" in out
        store.clear_cache.assert_called_once()
        store.refresh.assert_called_once()
        store.write_cache.assert_called_once()


# ---------------------------------------------------------------------------
# TestVariantToSpec
# ---------------------------------------------------------------------------


class TestVariantToSpec:
    """Tests for _variant_to_spec helper."""

    def test_converts_dict_variant(self) -> None:
        variant = {"variant_id": "exp-x", "parameter": "alpha", "value": 0.5, "hypothesis": "Test"}
        spec = cli_mod._variant_to_spec(variant, config={})
        assert spec is not None
        assert spec["experiment_id"] == "exp-x"
        assert spec["resolved_config"] == {"alpha": 0.5}

    def test_returns_none_for_no_id(self) -> None:
        variant = {"parameter": "alpha", "value": 0.5}
        spec = cli_mod._variant_to_spec(variant, config={})
        assert spec is None

    def test_uses_config_base_method(self) -> None:
        variant = {"variant_id": "exp-y", "parameter": "beta", "value": 3}
        spec = cli_mod._variant_to_spec(variant, config={"challenger_base_method": "m2026"})
        assert spec is not None
        assert spec["method"] == "m2026"


# ---------------------------------------------------------------------------
# TestMainDispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    """Tests for main() dispatch logic."""

    def test_unknown_command(self) -> None:
        """Unknown command after parsing should not crash — parser handles it."""
        # argparse will raise SystemExit for unknown subcommands
        with pytest.raises(SystemExit):
            cli_mod.main(argv=["nonexistent-command"])

    def test_main_with_config_flag(self, config_path: Path) -> None:
        """Main should accept a --config flag and load config from it."""
        rc = cli_mod.main(argv=["--config", str(config_path), "status"])
        # May be 0 or 1 depending on whether catalog loads, but should not crash
        assert isinstance(rc, int)
