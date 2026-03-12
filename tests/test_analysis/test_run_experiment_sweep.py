"""Tests for the batch experiment sweep runner (scripts/analysis/run_experiment_sweep.py).

Covers grid loading/validation, grid combination generation, spec file
generation, dedup checking, summary table formatting, experiment ID
generation, and CLI argument parsing. All tests use synthetic data and
do not invoke real experiment runs.

Ticket: BM-001
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import run_experiment_sweep as sweep_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_GRID: dict = {
    "base_method": "m2026r1",
    "base_config": "cfg-20260309-college-fix-v1",
    "scope": "county",
    "requested_by": "agent",
    "parameters": {
        "college_blend_factor": [0.5, 0.7, 0.9],
    },
}

MULTI_PARAM_GRID: dict = {
    "base_method": "m2026r1",
    "base_config": "cfg-20260309-college-fix-v1",
    "scope": "county",
    "requested_by": "agent",
    "parameters": {
        "college_blend_factor": [0.5, 0.7],
        "convergence_medium_hold": [3, 5],
    },
    "mode": "cartesian",
}

ZIP_GRID: dict = {
    "base_method": "m2026r1",
    "base_config": "cfg-20260309-college-fix-v1",
    "scope": "county",
    "requested_by": "agent",
    "parameters": {
        "college_blend_factor": [0.5, 0.7],
        "convergence_medium_hold": [3, 5],
    },
    "mode": "zip",
}


@pytest.fixture()
def grid_path(tmp_path: Path) -> Path:
    """Write a valid grid YAML and return its path."""
    p = tmp_path / "test_grid.yaml"
    p.write_text(yaml.safe_dump(VALID_GRID, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture()
def pending_dir_with_specs(tmp_path: Path) -> Path:
    """Create a pending directory with two synthetic spec files."""
    pending = tmp_path / "pending"
    pending.mkdir()
    for i, blend in enumerate([0.5, 0.7]):
        spec = {
            "experiment_id": f"exp-20260312-test-blend-{i}",
            "hypothesis": f"Test blend factor {blend}",
            "base_method": "m2026r1",
            "base_config": "cfg-20260309-college-fix-v1",
            "config_delta": {"college_blend_factor": blend},
            "scope": "county",
            "benchmark_label": f"blend-{blend}",
            "requested_by": "agent",
        }
        path = pending / f"exp-20260312-test-blend-{i}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return pending


# ---------------------------------------------------------------------------
# TestLoadGrid
# ---------------------------------------------------------------------------


class TestLoadGrid:
    """Tests for load_grid validation."""

    def test_load_valid_grid(self, grid_path: Path) -> None:
        grid = sweep_mod.load_grid(grid_path)
        assert grid["base_method"] == "m2026r1"
        assert "college_blend_factor" in grid["parameters"]

    def test_load_grid_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sweep_mod.load_grid(tmp_path / "nonexistent.yaml")

    def test_load_grid_missing_required_field(self, tmp_path: Path) -> None:
        incomplete = {k: v for k, v in VALID_GRID.items() if k != "parameters"}
        p = tmp_path / "incomplete.yaml"
        p.write_text(yaml.safe_dump(incomplete), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            sweep_mod.load_grid(p)

    def test_load_grid_empty_parameters(self, tmp_path: Path) -> None:
        bad = {**VALID_GRID, "parameters": {}}
        p = tmp_path / "empty_params.yaml"
        p.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty mapping"):
            sweep_mod.load_grid(p)

    def test_load_grid_parameter_not_list(self, tmp_path: Path) -> None:
        bad = {**VALID_GRID, "parameters": {"college_blend_factor": 0.5}}
        p = tmp_path / "scalar_param.yaml"
        p.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty list"):
            sweep_mod.load_grid(p)


# ---------------------------------------------------------------------------
# TestGenerateGridCombinations
# ---------------------------------------------------------------------------


class TestGenerateGridCombinations:
    """Tests for cartesian and zip grid combination generation."""

    def test_cartesian_single_param(self) -> None:
        params = {"alpha": [0.1, 0.2, 0.3]}
        combos = sweep_mod.generate_grid_combinations(params, mode="cartesian")
        assert len(combos) == 3
        assert combos[0] == {"alpha": 0.1}
        assert combos[2] == {"alpha": 0.3}

    def test_cartesian_multi_param(self) -> None:
        params = {"a": [1, 2], "b": [10, 20]}
        combos = sweep_mod.generate_grid_combinations(params, mode="cartesian")
        assert len(combos) == 4
        # Should contain all four combinations
        values = {(c["a"], c["b"]) for c in combos}
        assert values == {(1, 10), (1, 20), (2, 10), (2, 20)}

    def test_zip_mode(self) -> None:
        params = {"a": [1, 2, 3], "b": [10, 20, 30]}
        combos = sweep_mod.generate_grid_combinations(params, mode="zip")
        assert len(combos) == 3
        assert combos[0] == {"a": 1, "b": 10}
        assert combos[2] == {"a": 3, "b": 30}

    def test_zip_unequal_lengths_raises(self) -> None:
        params = {"a": [1, 2], "b": [10, 20, 30]}
        with pytest.raises(ValueError, match="equal length"):
            sweep_mod.generate_grid_combinations(params, mode="zip")

    def test_unknown_mode_raises(self) -> None:
        params = {"a": [1]}
        with pytest.raises(ValueError, match="Unknown grid mode"):
            sweep_mod.generate_grid_combinations(params, mode="random")


# ---------------------------------------------------------------------------
# TestGenerateSpecsFromGrid
# ---------------------------------------------------------------------------


class TestGenerateSpecsFromGrid:
    """Tests for spec file generation from a grid definition."""

    def test_generates_correct_number_of_specs(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "specs"
        specs = sweep_mod.generate_specs_from_grid(VALID_GRID, output_dir)
        assert len(specs) == 3  # 3 values for college_blend_factor

    def test_generated_specs_are_valid_yaml(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "specs"
        specs = sweep_mod.generate_specs_from_grid(VALID_GRID, output_dir)
        for spec_path in specs:
            data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            assert "experiment_id" in data
            assert "hypothesis" in data
            assert "config_delta" in data
            assert data["base_method"] == "m2026r1"
            assert data["scope"] == "county"

    def test_cartesian_multi_param_count(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "specs"
        specs = sweep_mod.generate_specs_from_grid(MULTI_PARAM_GRID, output_dir)
        assert len(specs) == 4  # 2 x 2

    def test_zip_mode_count(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "specs"
        specs = sweep_mod.generate_specs_from_grid(ZIP_GRID, output_dir)
        assert len(specs) == 2  # zip of 2 pairs

    def test_config_delta_contains_parameter(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "specs"
        specs = sweep_mod.generate_specs_from_grid(VALID_GRID, output_dir)
        first = yaml.safe_load(specs[0].read_text(encoding="utf-8"))
        assert "college_blend_factor" in first["config_delta"]
        assert first["config_delta"]["college_blend_factor"] == 0.5


# ---------------------------------------------------------------------------
# TestExperimentIdGeneration
# ---------------------------------------------------------------------------


class TestExperimentIdGeneration:
    """Tests for experiment ID auto-generation."""

    def test_id_contains_date(self) -> None:
        import datetime as _dt

        today = _dt.datetime.now(tz=_dt.UTC).date().strftime("%Y%m%d")
        eid = sweep_mod.generate_experiment_id(
            "blend", {"college_blend_factor": 0.7}, 0
        )
        assert today in eid

    def test_id_contains_sweep_prefix(self) -> None:
        eid = sweep_mod.generate_experiment_id(
            "blend", {"college_blend_factor": 0.7}, 0
        )
        assert "sweep-blend" in eid

    def test_id_contains_value_slug(self) -> None:
        eid = sweep_mod.generate_experiment_id(
            "blend", {"college_blend_factor": 0.7}, 0
        )
        # 0.7 becomes "0p7"
        assert "0p7" in eid

    def test_slugify_value(self) -> None:
        assert sweep_mod._slugify_value(0.7) == "0p7"
        assert sweep_mod._slugify_value(3) == "3"
        assert sweep_mod._slugify_value("hello world") == "hello-world"


# ---------------------------------------------------------------------------
# TestCollectPendingSpecs
# ---------------------------------------------------------------------------


class TestCollectPendingSpecs:
    """Tests for pending spec collection."""

    def test_collect_existing_specs(
        self, pending_dir_with_specs: Path
    ) -> None:
        specs = sweep_mod.collect_pending_specs(pending_dir_with_specs)
        assert len(specs) == 2
        assert all(p.suffix == ".yaml" for p in specs)

    def test_collect_nonexistent_dir(self, tmp_path: Path) -> None:
        specs = sweep_mod.collect_pending_specs(tmp_path / "nonexistent")
        assert specs == []

    def test_collect_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        specs = sweep_mod.collect_pending_specs(empty)
        assert specs == []


# ---------------------------------------------------------------------------
# TestFormatSummaryTable
# ---------------------------------------------------------------------------


class TestFormatSummaryTable:
    """Tests for summary table formatting."""

    def test_empty_results(self) -> None:
        table = sweep_mod.format_summary_table([])
        assert "No experiments" in table

    def test_table_contains_experiment_id(self) -> None:
        results = [
            {
                "experiment_id": "exp-test-blend-50",
                "outcome": "passed_all_gates",
                "key_delta": "overall: -0.09",
            },
        ]
        table = sweep_mod.format_summary_table(results)
        assert "exp-test-blend-50" in table
        assert "passed_all_gates" in table
        assert "overall: -0.09" in table

    def test_table_contains_skipped(self) -> None:
        results = [
            {
                "experiment_id": "exp-test-skipped",
                "outcome": "(skipped)",
                "key_delta": "-",
            },
        ]
        table = sweep_mod.format_summary_table(results)
        assert "(skipped)" in table

    def test_table_has_header(self) -> None:
        results = [
            {
                "experiment_id": "exp-test",
                "outcome": "ok",
                "key_delta": "-",
            },
        ]
        table = sweep_mod.format_summary_table(results)
        assert "Experiment" in table
        assert "Outcome" in table
        assert "Key Delta" in table
        assert "Sweep Summary:" in table

    def test_table_multiple_rows(self) -> None:
        results = [
            {
                "experiment_id": f"exp-{i}",
                "outcome": "ok",
                "key_delta": f"+{i}.0",
            }
            for i in range(3)
        ]
        table = sweep_mod.format_summary_table(results)
        for i in range(3):
            assert f"exp-{i}" in table


# ---------------------------------------------------------------------------
# TestDedupChecking
# ---------------------------------------------------------------------------


class TestDedupChecking:
    """Tests for dedup checking via get_tested_hypotheses integration."""

    def test_skip_already_tested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Specs with IDs already in the experiment log should be skipped."""
        # Mock get_tested_hypotheses to return a known set
        monkeypatch.setattr(
            sweep_mod,
            "get_tested_hypotheses",
            lambda: {"exp-already-done"},
        )
        # Mock run_single_experiment to track calls
        calls: list[Path] = []

        def mock_run(spec_path: Path, dry_run: bool = False) -> dict:
            calls.append(spec_path)
            return {
                "experiment_id": "exp-new-one",
                "spec_path": str(spec_path),
                "exit_code": 0,
                "outcome": "passed_all_gates",
                "key_delta": "-",
                "stdout": "",
                "stderr": "",
            }

        monkeypatch.setattr(sweep_mod, "run_single_experiment", mock_run)

        # Create two specs: one already tested, one new
        pending = tmp_path / "pending"
        pending.mkdir()
        for eid in ["exp-already-done", "exp-new-one"]:
            spec = {
                "experiment_id": eid,
                "hypothesis": "test",
                "base_method": "m2026r1",
                "base_config": "cfg-test",
                "config_delta": {"alpha": 1},
                "scope": "county",
                "benchmark_label": "test",
                "requested_by": "agent",
            }
            p = pending / f"{eid}.yaml"
            p.write_text(yaml.safe_dump(spec), encoding="utf-8")

        spec_paths = sorted(pending.glob("*.yaml"))
        results = sweep_mod.run_sweep(spec_paths, dry_run=False)

        # The already-done one should be skipped
        skipped = [r for r in results if r["outcome"] == "(skipped)"]
        assert len(skipped) == 1
        assert skipped[0]["experiment_id"] == "exp-already-done"

        # Only the new one should have been run
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# TestParseArgs
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_specs_mode(self) -> None:
        args = sweep_mod.parse_args(["--specs", "a.yaml", "b.yaml"])
        assert len(args.specs) == 2
        assert args.grid is None
        assert args.pending is False

    def test_grid_mode(self) -> None:
        args = sweep_mod.parse_args(["--grid", "grid.yaml"])
        assert args.grid == Path("grid.yaml")
        assert args.specs is None
        assert args.pending is False

    def test_pending_mode(self) -> None:
        args = sweep_mod.parse_args(["--pending"])
        assert args.pending is True
        assert args.specs is None
        assert args.grid is None

    def test_dry_run_flag(self) -> None:
        args = sweep_mod.parse_args(["--pending", "--dry-run"])
        assert args.dry_run is True

    def test_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            sweep_mod.parse_args(["--specs", "a.yaml", "--pending"])


# ---------------------------------------------------------------------------
# TestExtractExperimentId
# ---------------------------------------------------------------------------


class TestExtractExperimentId:
    """Tests for experiment ID extraction from spec files."""

    def test_extract_from_valid_spec(self, tmp_path: Path) -> None:
        spec = {"experiment_id": "exp-20260312-test"}
        p = tmp_path / "test.yaml"
        p.write_text(yaml.safe_dump(spec), encoding="utf-8")
        assert sweep_mod._extract_experiment_id(p) == "exp-20260312-test"

    def test_fallback_to_stem(self, tmp_path: Path) -> None:
        p = tmp_path / "bad-spec.yaml"
        p.write_text("not: valid: yaml: {{{{", encoding="utf-8")
        # Should fall back to filename stem
        assert sweep_mod._extract_experiment_id(p) == "bad-spec"


# ---------------------------------------------------------------------------
# TestJsonParsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    """Tests for parsing JSON output from run_experiment.py."""

    def test_parse_outcome_from_json(self) -> None:
        stdout = (
            'Running experiment...\n'
            '{\n'
            '  "outcome": "passed_all_gates",\n'
            '  "run_id": "br-123",\n'
            '  "experiment_id": "exp-test"\n'
            '}'
        )
        assert sweep_mod._parse_outcome_from_stdout(stdout) == "passed_all_gates"

    def test_parse_outcome_no_json(self) -> None:
        assert sweep_mod._parse_outcome_from_stdout("no json here") == "unknown"

    def test_extract_json_block(self) -> None:
        text = 'some output\n{\n  "key": "value"\n}'
        result = sweep_mod._extract_json_block(text)
        parsed = __import__("json").loads(result)
        assert parsed["key"] == "value"

    def test_extract_json_block_no_json(self) -> None:
        with pytest.raises(ValueError, match="No JSON block"):
            sweep_mod._extract_json_block("no json here")
