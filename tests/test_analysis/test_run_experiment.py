"""Tests for the experiment orchestrator (scripts/analysis/run_experiment.py).

Covers spec loading/validation, derivation helpers, deep merge, config delta
summary formatting, METHOD_DISPATCH checking, and method profile creation.

Ticket: BM-001-07
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import run_experiment as re_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_SPEC: dict = {
    "experiment_id": "exp-20260309-test-smoothing",
    "hypothesis": "Smoothing improves accuracy",
    "base_method": "m2026",
    "base_config": "cfg-20260101-baseline",
    "config_delta": {"alpha": 0.5, "nested": {"beta": 1.0}},
    "scope": "county",
    "benchmark_label": "m2026-smoothing-test",
    "requested_by": "agent",
}


@pytest.fixture()
def spec_path(tmp_path: Path) -> Path:
    """Write a valid experiment spec YAML and return its path."""
    p = tmp_path / "test_spec.yaml"
    p.write_text(yaml.safe_dump(VALID_SPEC, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture()
def base_profile_dir(tmp_path: Path) -> Path:
    """Create a directory with a minimal base method profile."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    base_profile = {
        "method_id": "m2026",
        "config_id": "cfg-20260101-baseline",
        "scope": "county",
        "status": "candidate",
        "created_date": "2026-01-01",
        "created_from": "test",
        "description": "test profile",
        "code_refs": [],
        "adr_refs": [],
        "resolved_config": {
            "param_a": 1,
            "nested": {
                "param_b": 2,
            },
        },
    }
    profile_path = profile_dir / "m2026__cfg-20260101-baseline.yaml"
    profile_path.write_text(
        yaml.safe_dump(base_profile, sort_keys=False), encoding="utf-8"
    )
    return profile_dir


# ---------------------------------------------------------------------------
# _load_spec
# ---------------------------------------------------------------------------


class TestLoadSpec:
    def test_load_spec_valid(self, spec_path: Path) -> None:
        spec = re_mod._load_spec(spec_path)
        for field in re_mod.REQUIRED_SPEC_FIELDS:
            assert field in spec

    def test_load_spec_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            re_mod._load_spec(tmp_path / "nonexistent.yaml")

    def test_load_spec_missing_field(self, tmp_path: Path) -> None:
        incomplete = {k: v for k, v in VALID_SPEC.items() if k != "hypothesis"}
        p = tmp_path / "incomplete.yaml"
        p.write_text(yaml.safe_dump(incomplete), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            re_mod._load_spec(p)

    def test_load_spec_invalid_scope(self, tmp_path: Path) -> None:
        bad = {**VALID_SPEC, "scope": "state"}
        p = tmp_path / "bad_scope.yaml"
        p.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="scope='county'"):
            re_mod._load_spec(p)


# ---------------------------------------------------------------------------
# _derive_method_id
# ---------------------------------------------------------------------------


class TestDeriveMethodId:
    def test_derive_method_id_no_override(self) -> None:
        assert re_mod._derive_method_id(VALID_SPEC) == "m2026"

    def test_derive_method_id_with_override(self) -> None:
        spec = {**VALID_SPEC, "method_id_override": "m2026r1"}
        assert re_mod._derive_method_id(spec) == "m2026r1"


# ---------------------------------------------------------------------------
# _derive_config_id
# ---------------------------------------------------------------------------


class TestDeriveConfigId:
    def test_derive_config_id_no_override(self) -> None:
        result = re_mod._derive_config_id(VALID_SPEC)
        # experiment_id = "exp-20260309-test-smoothing"
        # parts split on "-" with maxsplit=3 → ["exp", "20260309", "test", "smoothing"]
        # slug = parts[3] = "smoothing"
        import datetime as dt

        today = dt.date.today().strftime("%Y%m%d")
        assert result == f"cfg-{today}-smoothing"

    def test_derive_config_id_with_override(self) -> None:
        spec = {**VALID_SPEC, "config_id_override": "cfg-custom-id"}
        assert re_mod._derive_config_id(spec) == "cfg-custom-id"


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_deep_merge_nested(self) -> None:
        base = {"a": 1, "nested": {"b": 2, "c": 3}}
        override = {"nested": {"b": 99, "d": 4}}
        re_mod._deep_merge(base, override)
        assert base == {"a": 1, "nested": {"b": 99, "c": 3, "d": 4}}

    def test_deep_merge_override_scalar(self) -> None:
        base = {"x": 10, "y": 20}
        override = {"x": 42}
        re_mod._deep_merge(base, override)
        assert base["x"] == 42
        assert base["y"] == 20


# ---------------------------------------------------------------------------
# _config_delta_summary
# ---------------------------------------------------------------------------


class TestConfigDeltaSummary:
    def test_config_delta_summary_simple(self) -> None:
        result = re_mod._config_delta_summary({"alpha": 0.5, "beta": 1.0})
        assert "alpha=0.5" in result
        assert "beta=1.0" in result

    def test_config_delta_summary_nested(self) -> None:
        result = re_mod._config_delta_summary({"outer": {"inner_a": 1, "inner_b": 2}})
        assert "outer:" in result
        assert "inner_a=1" in result
        assert "inner_b=2" in result

    def test_config_delta_summary_empty(self) -> None:
        assert re_mod._config_delta_summary({}) == "no changes"


# ---------------------------------------------------------------------------
# _check_method_dispatch
# ---------------------------------------------------------------------------


class TestCheckMethodDispatch:
    def test_check_method_dispatch_valid(self) -> None:
        assert re_mod._check_method_dispatch("m2026") is True

    def test_check_method_dispatch_invalid(self) -> None:
        assert re_mod._check_method_dispatch("nonexistent_method") is False


# ---------------------------------------------------------------------------
# _create_method_profile
# ---------------------------------------------------------------------------


class TestCreateMethodProfile:
    def test_create_method_profile(
        self, tmp_path: Path, base_profile_dir: Path
    ) -> None:
        spec = {**VALID_SPEC}
        method_id = "m2026"
        config_id = "cfg-test-experiment"

        profile_path = re_mod._create_method_profile(
            spec, method_id, config_id, base_profile_dir
        )

        assert profile_path.exists()
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        assert profile["method_id"] == method_id
        assert profile["config_id"] == config_id
        assert profile["status"] == "experiment"
        # config_delta should have been merged into resolved_config
        assert profile["resolved_config"]["alpha"] == 0.5
        assert profile["resolved_config"]["nested"]["beta"] == 1.0
        # Original base value should be preserved where not overridden
        assert profile["resolved_config"]["param_a"] == 1
        assert profile["resolved_config"]["nested"]["param_b"] == 2

    def test_create_method_profile_idempotent(
        self, tmp_path: Path, base_profile_dir: Path
    ) -> None:
        spec = {**VALID_SPEC}
        method_id = "m2026"
        config_id = "cfg-test-idempotent"

        # Create profile first time
        path1 = re_mod._create_method_profile(
            spec, method_id, config_id, base_profile_dir
        )
        content_after_first = path1.read_text(encoding="utf-8")

        # Create profile second time — should skip, no error
        path2 = re_mod._create_method_profile(
            spec, method_id, config_id, base_profile_dir
        )
        assert path1 == path2
        # Content should be unchanged
        assert path2.read_text(encoding="utf-8") == content_after_first


# ---------------------------------------------------------------------------
# _CLASSIFICATION_TO_ACTION mapping
# ---------------------------------------------------------------------------


class TestClassificationToAction:
    def test_classification_to_action_mapping(self) -> None:
        expected_keys = {
            "passed_all_gates",
            "failed_hard_gate",
            "needs_human_review",
            "inconclusive",
        }
        assert set(re_mod._CLASSIFICATION_TO_ACTION.keys()) == expected_keys
        # All values should be non-empty strings
        for key, value in re_mod._CLASSIFICATION_TO_ACTION.items():
            assert isinstance(value, str)
            assert len(value) > 0
