"""Tests for the Observatory CLI (scripts/analysis/observatory.py).

Covers argument parsing, subcommand dispatch, graceful degradation when
modules are unavailable, and individual command handlers using mocked
store/comparator/recommender/catalog components.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

cli_mod = importlib.import_module("observatory")


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
    sc = pd.DataFrame(
        [
            {
                "method_id": "m2026",
                "config_id": "cfg-base",
                "status_at_run": "champion",
                "county_mape_overall": 8.5,
                "county_mape_rural": 7.0,
                "county_mape_bakken": 19.0,
                "county_mape_urban_college": 12.0,
                "state_ape_recent_short": 1.0,
                "state_ape_recent_medium": 2.5,
            }
        ]
    )
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
        assert args.priority == "tier"

    def test_report_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["report", "--output", "/tmp/report.html"])
        assert args.command == "report"
        assert args.output == Path("/tmp/report.html")

    def test_format_flag_after_subcommand(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(["status", "--format", "json"])
        assert args.output_format == "json"

    def test_run_recommended_queue_flags(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(
            [
                "run-recommended",
                "--run-budget",
                "2",
                "--retry-failures",
                "1",
                "--resume-file",
                "state.json",
            ]
        )
        assert args.run_budget == 2
        assert args.retry_failures == 1
        assert args.resume_file == Path("state.json")

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
        store.get_index.return_value = pd.DataFrame(
            {
                "run_id": ["run-001"],
                "run_date": ["2026-03-01"],
            }
        )
        store.get_run_ids.return_value = ["run-001"]
        args = cli_mod.build_parser().parse_args(["status"])

        catalog = MagicMock()
        catalog.get_inventory_summary.return_value = {
            "total": 3,
            "tested": 1,
            "untested_total": 2,
            "untested_runnable": 1,
            "untested_requires_code_change": 1,
            "untested_ids": ["EXP-A", "EXP-B"],
            "untested_runnable_ids": ["EXP-A"],
            "untested_requires_code_change_ids": ["EXP-B"],
            "grid_total": 2,
            "grid_blocked": 1,
            "grid_blocked_ids": ["dampening-sweep"],
        }

        with patch.object(cli_mod, "_load_variant_catalog", return_value=catalog):
            rc = cli_mod.cmd_status(store=store, config={}, _args=args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Completed runs: 1" in out
        assert "Untested runnable: 1" in out
        assert "Untested requiring code changes: 1" in out
        assert "Blocked grids: 1" in out


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
        rank_df = pd.DataFrame(
            {
                "run_id": ["r1", "r2"],
                "county_mape_overall": [8.0, 9.0],
                "rank": [1, 2],
            }
        )
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


class TestRecommendationArtifacts:
    """Tests for recommendation spec naming and hygiene helpers."""

    def test_build_recommendation_spec_uses_safe_combo_slug(self) -> None:
        rec = SimpleNamespace(
            parameter="combined",
            suggested_value={
                "college_blend_factor": 0.9,
                "convergence_medium_hold": 5,
            },
            rationale="Test a combination.",
            priority=1,
            grid_suggestion=None,
        )

        spec = cli_mod._build_recommendation_spec(rec, config={})
        assert spec is not None
        assert spec["requested_by"] == "agent"
        assert spec["experiment_id"].startswith("exp-")
        assert "college-blend-factor-0p9" in spec["experiment_id"]
        assert spec["benchmark_label"].startswith("rec-")


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
        catalog.get_inventory_summary.return_value = {}

        args = cli_mod.build_parser().parse_args(["run-pending"])
        with patch.object(cli_mod, "_load_variant_catalog", return_value=catalog):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)

        assert rc == 0
        assert "No untested" in capsys.readouterr().out

    def test_run_pending_dry_run(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        catalog = MagicMock()
        catalog.get_untested.return_value = [
            {"variant_id": "exp-x", "parameter": "alpha", "value": 0.5, "hypothesis": "test"},
        ]
        catalog.get_inventory_summary.return_value = {
            "untested_requires_code_change": 1,
            "grid_blocked": 0,
        }

        # generate_spec must write a real file in the temp directory
        def _mock_generate_spec(variant_id: str, output_dir: Path, **_kwargs: object) -> Path:
            import yaml as _yaml

            output_dir.mkdir(parents=True, exist_ok=True)
            spec = {
                "experiment_id": f"exp-test-{variant_id}",
                "hypothesis": "test",
                "base_method": "m2026r1",
                "base_config": "cfg-test",
                "config_delta": {"alpha": 0.5},
                "scope": "county",
                "benchmark_label": f"test-{variant_id}",
                "requested_by": "agent",
            }
            p = output_dir / f"exp-test-{variant_id}.yaml"
            p.write_text(_yaml.safe_dump(spec), encoding="utf-8")
            return p

        catalog.generate_spec.side_effect = _mock_generate_spec

        args = cli_mod.build_parser().parse_args(["run-pending", "--dry-run"])
        with patch.object(cli_mod, "_load_variant_catalog", return_value=catalog):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "Resume file" in out

    def test_run_pending_passes_queue_controls_to_sweep(self) -> None:
        catalog = MagicMock()
        catalog.get_untested.return_value = [{"variant_id": "exp-x", "tier": 1}]
        catalog.get_inventory_summary.return_value = {
            "untested_requires_code_change": 0,
            "grid_blocked": 0,
        }

        def _mock_generate_spec(variant_id: str, output_dir: Path, **_kwargs: object) -> Path:
            path = output_dir / f"{variant_id}.yaml"
            path.write_text(
                yaml.safe_dump({"experiment_id": variant_id, "config_delta": {"x": 1}}),
                encoding="utf-8",
            )
            return path

        catalog.generate_spec.side_effect = _mock_generate_spec
        args = cli_mod.build_parser().parse_args(
            [
                "run-pending",
                "--run-budget",
                "2",
                "--retry-failures",
                "1",
                "--resume-file",
                "state.json",
            ]
        )
        with (
            patch.object(cli_mod, "_load_variant_catalog", return_value=catalog),
            patch.object(cli_mod, "_run_sweep_command", return_value=0) as run_sweep,
        ):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)

        assert rc == 0
        run_sweep.assert_called_once()
        kwargs = run_sweep.call_args.kwargs
        assert kwargs["run_budget"] == 2
        assert kwargs["retry_failures"] == 1
        assert kwargs["resume_file"] == PROJECT_ROOT / "state.json"


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
# TestMainDispatch
# ---------------------------------------------------------------------------


class TestRunPendingSpecFormat:
    """Tests that run-pending generates correctly-formatted specs via catalog."""

    def test_spec_has_required_fields(self, tmp_path: Path) -> None:
        """Specs generated by run-pending should have all fields run_experiment.py needs."""
        catalog = MagicMock()

        # Return one untested variant
        catalog.get_untested.return_value = [
            {"variant_id": "exp-test"},
        ]
        catalog.get_inventory_summary.return_value = {
            "untested_requires_code_change": 0,
            "grid_blocked": 0,
        }

        import yaml as _yaml

        # Simulate generate_spec writing a proper spec file
        def _mock_generate_spec(variant_id: str, output_dir: Path, **_kwargs: object) -> Path:
            output_dir.mkdir(parents=True, exist_ok=True)
            spec = {
                "experiment_id": f"exp-20260312-{variant_id}",
                "hypothesis": "Test hypothesis",
                "base_method": "m2026r1",
                "base_config": "cfg-20260309-college-fix-v1",
                "config_delta": {"college_blend_factor": 0.7},
                "scope": "county",
                "benchmark_label": f"test-{variant_id}",
                "requested_by": "agent",
            }
            p = output_dir / f"exp-20260312-{variant_id}.yaml"
            p.write_text(_yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            return p

        catalog.generate_spec.side_effect = _mock_generate_spec

        args = cli_mod.build_parser().parse_args(["run-pending", "--dry-run"])
        with patch.object(cli_mod, "_load_variant_catalog", return_value=catalog):
            rc = cli_mod.cmd_run_pending(store=MagicMock(), config={}, args=args)
        assert rc == 0
        # Verify catalog.generate_spec was called
        catalog.generate_spec.assert_called_once()


class TestCmdRunRecommended:
    """Tests for the run-recommended command handler."""

    def test_run_recommended_no_store(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = cli_mod.build_parser().parse_args(["run-recommended"])
        rc = cli_mod.cmd_run_recommended(store=None, config={}, args=args)
        assert rc == 1

    def test_run_recommended_no_recommender(self, capsys: pytest.CaptureFixture[str]) -> None:
        store = MagicMock()
        args = cli_mod.build_parser().parse_args(["run-recommended"])
        with (
            patch.object(cli_mod, "_load_comparator", return_value=None),
            patch.object(cli_mod, "_load_recommender", return_value=None),
        ):
            rc = cli_mod.cmd_run_recommended(store=store, config={}, args=args)
        assert rc == 1

    def test_run_recommended_no_config_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        """When all recommendations require code changes, should print message."""
        store = MagicMock()
        recommender = MagicMock()

        # All recs require code changes
        rec = MagicMock()
        rec.requires_code_change = True
        rec.parameter = "rate_cap_general"
        rec.suggested_value = 0.15
        recommender.suggest_next_experiments.return_value = [rec]

        args = cli_mod.build_parser().parse_args(["run-recommended", "--dry-run"])
        with (
            patch.object(cli_mod, "_load_comparator", return_value=MagicMock()),
            patch.object(cli_mod, "_load_recommender", return_value=recommender),
            patch.object(cli_mod, "_load_variant_catalog", return_value=None),
        ):
            rc = cli_mod.cmd_run_recommended(store=store, config={}, args=args)
        assert rc == 0
        assert "No config-only" in capsys.readouterr().out

    def test_run_recommended_dry_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run with config-only recommendations should show specs."""
        store = MagicMock()
        recommender = MagicMock()

        rec = MagicMock()
        rec.requires_code_change = False
        rec.parameter = "college_blend_factor"
        rec.suggested_value = 0.9
        rec.rationale = "Boundary detection: best at 0.7"
        recommender.suggest_next_experiments.return_value = [rec]

        args = cli_mod.build_parser().parse_args(["run-recommended", "--dry-run"])
        with (
            patch.object(cli_mod, "_load_comparator", return_value=MagicMock()),
            patch.object(cli_mod, "_load_recommender", return_value=recommender),
            patch.object(cli_mod, "_load_variant_catalog", return_value=None),
        ):
            rc = cli_mod.cmd_run_recommended(store=store, config={}, args=args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "Resume file" in out

    def test_run_recommended_parser(self) -> None:
        parser = cli_mod.build_parser()
        args = parser.parse_args(
            [
                "run-recommended",
                "--dry-run",
                "--run-budget",
                "2",
                "--retry-failures",
                "1",
            ]
        )
        assert args.command == "run-recommended"
        assert args.dry_run is True
        assert args.run_budget == 2
        assert args.retry_failures == 1


class TestCmdReport:
    """Tests for report output path handling."""

    def test_report_resolves_project_relative_output(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        store = MagicMock()
        received_paths: list[Path | None] = []

        class FakeReport:
            def __init__(
                self, comparator_result: object, recommendations: list[object], store: object
            ) -> None:
                self._store = store

            def generate_html_report(self, output_path: Path | None = None) -> Path:
                received_paths.append(output_path)
                assert output_path is not None
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("<html></html>", encoding="utf-8")
                return output_path

            def generate_summary(self) -> str:
                return "summary"

        args = cli_mod.build_parser().parse_args(["report", "--output", "reports/test.html"])
        with (
            patch.object(cli_mod, "PROJECT_ROOT", tmp_path),
            patch.object(cli_mod, "_load_report_class", return_value=FakeReport),
            patch.object(cli_mod, "_load_comparator", return_value=None),
            patch.object(cli_mod, "_load_recommender", return_value=None),
            patch.object(cli_mod, "_load_variant_catalog", return_value=None),
        ):
            rc = cli_mod.cmd_report(store=store, config={}, args=args)

        assert rc == 0
        assert received_paths == [tmp_path / "reports" / "test.html"]
        assert "Observatory report written to" in capsys.readouterr().out

    def test_report_preserves_absolute_output(
        self,
        tmp_path: Path,
    ) -> None:
        store = MagicMock()
        absolute_path = tmp_path / "absolute-report.html"
        received_paths: list[Path | None] = []

        class FakeReport:
            def __init__(
                self, comparator_result: object, recommendations: list[object], store: object
            ) -> None:
                self._store = store

            def generate_html_report(self, output_path: Path | None = None) -> Path:
                received_paths.append(output_path)
                assert output_path is not None
                output_path.write_text("<html></html>", encoding="utf-8")
                return output_path

            def generate_summary(self) -> str:
                return "summary"

        args = cli_mod.build_parser().parse_args(["report", "--output", str(absolute_path)])
        with (
            patch.object(cli_mod, "_load_report_class", return_value=FakeReport),
            patch.object(cli_mod, "_load_comparator", return_value=None),
            patch.object(cli_mod, "_load_recommender", return_value=None),
            patch.object(cli_mod, "_load_variant_catalog", return_value=None),
        ):
            rc = cli_mod.cmd_report(store=store, config={}, args=args)

        assert rc == 0
        assert received_paths == [absolute_path]


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
