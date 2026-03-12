"""Tests for cohort_projections.analysis.experiment_log."""

from __future__ import annotations

from pathlib import Path

import pytest

from cohort_projections.analysis.experiment_log import (
    LOG_COLUMNS,
    _match_config_delta,
    append_experiment_entry,
    config_delta_summary,
    get_tested_hypotheses,
    is_config_delta_tested,
    read_experiment_log,
)


def _make_entry(**overrides: object) -> dict[str, object]:
    """Return a valid experiment log entry with sensible defaults."""
    defaults: dict[str, object] = {
        "experiment_id": "EXP-001",
        "run_date": "2026-03-09",
        "hypothesis": "Smoothing improves accuracy",
        "base_method": "m2026",
        "config_delta_summary": "smoothing_window: 3",
        "run_id": "run-abc123",
        "outcome": "passed_all_gates",
        "key_metrics_summary": "MAPE 2.1%",
        "interpretation": "Gate passed at all horizons",
        "next_action": "proceed_to_next",
        "agent_or_human": "agent",
        "spec_path": "specs/exp-001.yaml",
    }
    defaults.update(overrides)
    return defaults


class TestAppendExperimentEntry:
    """Tests for append_experiment_entry."""

    def test_append_creates_file_with_header(self, tmp_path: Path) -> None:
        """Append to non-existent path creates file with header + 1 data row."""
        log_path = tmp_path / "log.csv"
        entry = _make_entry()
        append_experiment_entry(entry, log_path=log_path)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2  # header + 1 data row
        assert lines[0] == ",".join(LOG_COLUMNS)

    def test_append_appends_to_existing(self, tmp_path: Path) -> None:
        """Appending two entries produces header + 2 data rows."""
        log_path = tmp_path / "log.csv"
        append_experiment_entry(_make_entry(experiment_id="EXP-001"), log_path=log_path)
        append_experiment_entry(_make_entry(experiment_id="EXP-002"), log_path=log_path)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3  # header + 2 data rows

    def test_append_missing_column_raises(self, tmp_path: Path) -> None:
        """Entry missing a required column raises ValueError."""
        log_path = tmp_path / "log.csv"
        entry = _make_entry()
        del entry["hypothesis"]

        with pytest.raises(ValueError, match="Missing required columns"):
            append_experiment_entry(entry, log_path=log_path)

    def test_append_invalid_outcome_raises(self, tmp_path: Path) -> None:
        """Invalid outcome value raises ValueError."""
        log_path = tmp_path / "log.csv"
        entry = _make_entry(outcome="bogus")

        with pytest.raises(ValueError, match="Invalid outcome"):
            append_experiment_entry(entry, log_path=log_path)

    def test_append_invalid_next_action_raises(self, tmp_path: Path) -> None:
        """Invalid next_action value raises ValueError."""
        log_path = tmp_path / "log.csv"
        entry = _make_entry(next_action="bogus")

        with pytest.raises(ValueError, match="Invalid next_action"):
            append_experiment_entry(entry, log_path=log_path)

    def test_append_only_property(self, tmp_path: Path) -> None:
        """Two appends produce exactly header + 2 rows (no overwrites)."""
        log_path = tmp_path / "log.csv"
        append_experiment_entry(_make_entry(experiment_id="EXP-A"), log_path=log_path)
        append_experiment_entry(_make_entry(experiment_id="EXP-B"), log_path=log_path)

        raw = log_path.read_text()
        lines = raw.strip().splitlines()
        assert len(lines) == 3
        # First data row should contain EXP-A, second EXP-B
        assert "EXP-A" in lines[1]
        assert "EXP-B" in lines[2]


class TestReadExperimentLog:
    """Tests for read_experiment_log."""

    def test_read_existing_log(self, tmp_path: Path) -> None:
        """Write 2 entries, read back, verify shape and values."""
        log_path = tmp_path / "log.csv"
        append_experiment_entry(
            _make_entry(experiment_id="EXP-001", hypothesis="H1"),
            log_path=log_path,
        )
        append_experiment_entry(
            _make_entry(experiment_id="EXP-002", hypothesis="H2"),
            log_path=log_path,
        )

        df = read_experiment_log(log_path)
        assert df.shape[0] == 2
        assert list(df.columns) == LOG_COLUMNS
        assert df.iloc[0]["experiment_id"] == "EXP-001"
        assert df.iloc[1]["hypothesis"] == "H2"

    def test_read_nonexistent_log(self, tmp_path: Path) -> None:
        """Non-existent file returns empty DataFrame with LOG_COLUMNS."""
        log_path = tmp_path / "does_not_exist.csv"
        df = read_experiment_log(log_path)
        assert df.empty
        assert list(df.columns) == LOG_COLUMNS

    def test_read_header_only_log(self, tmp_path: Path) -> None:
        """File with header but no data rows returns empty DataFrame."""
        log_path = tmp_path / "log.csv"
        log_path.write_text(",".join(LOG_COLUMNS) + "\n")

        df = read_experiment_log(log_path)
        assert df.empty


class TestGetTestedHypotheses:
    """Tests for get_tested_hypotheses."""

    def test_get_tested_hypotheses(self, tmp_path: Path) -> None:
        """Three entries with distinct ids returns correct set."""
        log_path = tmp_path / "log.csv"
        for exp_id in ("EXP-001", "EXP-002", "EXP-003"):
            append_experiment_entry(
                _make_entry(experiment_id=exp_id),
                log_path=log_path,
            )

        result = get_tested_hypotheses(log_path)
        assert result == {"EXP-001", "EXP-002", "EXP-003"}

    def test_get_tested_hypotheses_empty_log(self, tmp_path: Path) -> None:
        """Non-existent file returns empty set."""
        log_path = tmp_path / "no_such_file.csv"
        result = get_tested_hypotheses(log_path)
        assert result == set()


class TestConfigDeltaSummary:
    """Tests for config_delta_summary."""

    def test_scalar_value(self) -> None:
        result = config_delta_summary({"college_blend_factor": 0.7})
        assert result == "college_blend_factor=0.7"

    def test_dict_value(self) -> None:
        result = config_delta_summary({"boom_period_dampening": {"2005-2010": 0.5}})
        assert "boom_period_dampening" in result
        assert "2005-2010=0.5" in result

    def test_list_value(self) -> None:
        result = config_delta_summary({"bakken_fips": [38017, 38105]})
        assert "bakken_fips=" in result

    def test_multiple_params(self) -> None:
        result = config_delta_summary({"a": 1, "b": 2})
        assert "a=1" in result
        assert "b=2" in result
        assert "; " in result

    def test_empty_dict(self) -> None:
        result = config_delta_summary({})
        assert result == ""

    def test_consistent_output(self) -> None:
        """Same input always produces the same output."""
        delta = {"college_blend_factor": 0.7}
        assert config_delta_summary(delta) == config_delta_summary(delta)


class TestMatchConfigDelta:
    """Tests for _match_config_delta."""

    def test_exact_match(self) -> None:
        assert _match_config_delta("college_blend_factor=0.7", {"college_blend_factor": 0.7})

    def test_no_match(self) -> None:
        assert not _match_config_delta("college_blend_factor=0.5", {"college_blend_factor": 0.7})

    def test_non_string_log_summary(self) -> None:
        assert not _match_config_delta(None, {"a": 1})  # type: ignore[arg-type]

    def test_subset_match(self) -> None:
        """Candidate delta is a subset of the log entry."""
        log = "college_blend_factor=0.7; convergence_medium_hold=5"
        assert _match_config_delta(log, {"college_blend_factor": 0.7})

    def test_dict_value_match(self) -> None:
        log = "boom_period_dampening: {2005-2010=0.5}"
        assert _match_config_delta(log, {"boom_period_dampening": {"2005-2010": 0.5}})


class TestIsConfigDeltaTested:
    """Tests for is_config_delta_tested."""

    def test_match_in_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.csv"
        append_experiment_entry(
            _make_entry(config_delta_summary="college_blend_factor=0.7"),
            log_path=log_path,
        )
        assert is_config_delta_tested(
            {"college_blend_factor": 0.7}, log_path=log_path
        )

    def test_no_match_in_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.csv"
        append_experiment_entry(
            _make_entry(config_delta_summary="college_blend_factor=0.5"),
            log_path=log_path,
        )
        assert not is_config_delta_tested(
            {"college_blend_factor": 0.7}, log_path=log_path
        )

    def test_nonexistent_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nonexistent.csv"
        assert not is_config_delta_tested(
            {"college_blend_factor": 0.7}, log_path=log_path
        )
