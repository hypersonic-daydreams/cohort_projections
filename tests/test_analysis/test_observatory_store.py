"""Tests for the ResultsStore class (cohort_projections.analysis.observatory.results_store).

Covers index loading, run enumeration, consolidated metric loading, caching,
manifest reading, config resolution, experiment log loading, and edge cases
(missing files, empty directories, stale caches).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cohort_projections.analysis.observatory.results_store import (
    ResultsStore,
    load_observatory_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCORECARD_COLUMNS = [
    "method_id",
    "config_id",
    "status_at_run",
    "county_mape_overall",
    "county_mape_rural",
    "county_mape_bakken",
    "county_mape_urban_college",
    "state_ape_recent_short",
    "state_ape_recent_medium",
]


def _write_index(history_dir: Path, run_ids: list[str]) -> None:
    """Write a minimal index.csv for the given run IDs."""
    df = pd.DataFrame({"run_id": run_ids, "run_date": ["2026-03-01"] * len(run_ids)})
    df.to_csv(history_dir / "index.csv", index=False)


def _write_scorecard(run_dir: Path, method_id: str = "m2026r1", **overrides: float) -> None:
    """Write a single-row summary_scorecard.csv inside a run directory."""
    row = {
        "method_id": method_id,
        "config_id": "cfg-test",
        "status_at_run": "champion",
        "county_mape_overall": 8.5,
        "county_mape_rural": 7.0,
        "county_mape_bakken": 19.0,
        "county_mape_urban_college": 12.0,
        "state_ape_recent_short": 1.0,
        "state_ape_recent_medium": 2.5,
    }
    row.update(overrides)
    pd.DataFrame([row]).to_csv(run_dir / "summary_scorecard.csv", index=False)


def _write_manifest(run_dir: Path, run_id: str, methods: list[dict] | None = None) -> None:
    """Write a minimal manifest.json."""
    manifest = {
        "run_id": run_id,
        "methods": methods or [{"method_id": "m2026r1"}],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _write_county_metrics(run_dir: Path) -> None:
    """Write a small county_metrics.csv."""
    df = pd.DataFrame({
        "county_fips": ["38017", "38105"],
        "county_name": ["Cass", "Williams"],
        "mape": [3.5, 12.2],
    })
    df.to_csv(run_dir / "county_metrics.csv", index=False)


@pytest.fixture()
def history_dir(tmp_path: Path) -> Path:
    """Create a benchmark history directory with two runs."""
    hdir = tmp_path / "benchmark_history"
    hdir.mkdir()

    run_ids = ["run-001", "run-002"]
    _write_index(hdir, run_ids)

    for rid in run_ids:
        rd = hdir / rid
        rd.mkdir()
        _write_scorecard(rd, county_mape_overall=8.5 if rid == "run-001" else 8.2)
        _write_manifest(rd, rid)
        _write_county_metrics(rd)

    return hdir


@pytest.fixture()
def store(history_dir: Path) -> ResultsStore:
    """Return a ResultsStore pointing at the fixture history directory."""
    return ResultsStore(history_dir=history_dir)


@pytest.fixture()
def store_with_cache(history_dir: Path, tmp_path: Path) -> ResultsStore:
    """Return a ResultsStore with a cache directory configured."""
    cache_dir = tmp_path / "cache"
    return ResultsStore(history_dir=history_dir, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# TestLoadObservatoryConfig
# ---------------------------------------------------------------------------


class TestLoadObservatoryConfig:
    """Tests for the standalone config loader."""

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        cfg = {"observatory": {"history_dir": "data/analysis/benchmark_history"}}
        p = tmp_path / "obs.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        result = load_observatory_config(p)
        assert result["history_dir"] == "data/analysis/benchmark_history"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_observatory_config(tmp_path / "nope.yaml")

    def test_missing_observatory_key_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"other": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="observatory"):
            load_observatory_config(p)

    def test_non_dict_content_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "scalar.yaml"
        p.write_text("just a string\n", encoding="utf-8")
        with pytest.raises(ValueError, match="observatory"):
            load_observatory_config(p)


# ---------------------------------------------------------------------------
# TestResultsStoreInit
# ---------------------------------------------------------------------------


class TestResultsStoreInit:
    """Tests for ResultsStore construction and factory methods."""

    def test_constructor(self, history_dir: Path) -> None:
        store = ResultsStore(history_dir=history_dir)
        assert repr(store).startswith("ResultsStore(")

    def test_from_config(self, tmp_path: Path, history_dir: Path) -> None:
        cfg = {
            "observatory": {
                "history_dir": str(history_dir),
            }
        }
        cfg_path = tmp_path / "obs_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        store = ResultsStore.from_config(cfg_path)
        # The from_config resolves relative to PROJECT_ROOT, but we passed
        # an absolute path — just verify it constructed without error.
        assert store is not None


# ---------------------------------------------------------------------------
# TestIndex
# ---------------------------------------------------------------------------


class TestIndex:
    """Tests for index loading and run enumeration."""

    def test_get_run_ids(self, store: ResultsStore) -> None:
        ids = store.get_run_ids()
        assert set(ids) == {"run-001", "run-002"}

    def test_get_index_returns_dataframe(self, store: ResultsStore) -> None:
        index = store.get_index()
        assert isinstance(index, pd.DataFrame)
        assert "run_id" in index.columns
        assert len(index) == 2

    def test_get_index_returns_copy(self, store: ResultsStore) -> None:
        idx1 = store.get_index()
        idx2 = store.get_index()
        assert idx1 is not idx2

    def test_missing_index_returns_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_history"
        empty_dir.mkdir()
        store = ResultsStore(history_dir=empty_dir)
        assert store.get_run_ids() == []
        assert store.get_index().empty


# ---------------------------------------------------------------------------
# TestConsolidatedMetrics
# ---------------------------------------------------------------------------


class TestConsolidatedMetrics:
    """Tests for consolidated scorecard and metric loading."""

    def test_get_consolidated_scorecards(self, store: ResultsStore) -> None:
        sc = store.get_consolidated_scorecards()
        assert not sc.empty
        assert "run_id" in sc.columns
        assert set(sc["run_id"].unique()) == {"run-001", "run-002"}

    def test_scorecards_have_expected_columns(self, store: ResultsStore) -> None:
        sc = store.get_consolidated_scorecards()
        for col in ("county_mape_overall", "method_id"):
            assert col in sc.columns

    def test_get_consolidated_county_metrics(self, store: ResultsStore) -> None:
        cm = store.get_consolidated_county_metrics()
        assert not cm.empty
        assert "run_id" in cm.columns

    def test_consolidated_returns_copy(self, store: ResultsStore) -> None:
        sc1 = store.get_consolidated_scorecards()
        sc2 = store.get_consolidated_scorecards()
        assert sc1 is not sc2

    def test_missing_csv_yields_empty(self, tmp_path: Path) -> None:
        """Runs that lack a particular CSV file are skipped gracefully."""
        hdir = tmp_path / "partial"
        hdir.mkdir()
        _write_index(hdir, ["run-x"])
        rd = hdir / "run-x"
        rd.mkdir()
        # Only write manifest, no scorecard
        _write_manifest(rd, "run-x")

        store = ResultsStore(history_dir=hdir)
        sc = store.get_consolidated_scorecards()
        assert sc.empty

    def test_empty_state_metrics(self, store: ResultsStore) -> None:
        """State metrics file doesn't exist — returns empty frame."""
        sm = store.get_consolidated_state_metrics()
        assert sm.empty


# ---------------------------------------------------------------------------
# TestManifest
# ---------------------------------------------------------------------------


class TestManifest:
    """Tests for manifest loading."""

    def test_get_run_manifest(self, store: ResultsStore) -> None:
        manifest = store.get_run_manifest("run-001")
        assert manifest["run_id"] == "run-001"

    def test_manifest_returns_copy(self, store: ResultsStore) -> None:
        m1 = store.get_run_manifest("run-001")
        m2 = store.get_run_manifest("run-001")
        assert m1 is not m2

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        hdir = tmp_path / "no_manifest"
        hdir.mkdir()
        _write_index(hdir, ["run-bad"])
        (hdir / "run-bad").mkdir()

        store = ResultsStore(history_dir=hdir)
        with pytest.raises(FileNotFoundError, match="run-bad"):
            store.get_run_manifest("run-bad")


# ---------------------------------------------------------------------------
# TestRunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    """Tests for get_run_config (resolved config extraction)."""

    def test_fallback_to_manifest_methods(self, store: ResultsStore) -> None:
        """When no resolved_configs/ dir exists, falls back to manifest."""
        configs = store.get_run_config("run-001")
        assert isinstance(configs, dict)
        # Should have at least one method entry from the manifest
        assert len(configs) >= 1

    def test_resolved_configs_directory(self, history_dir: Path) -> None:
        """When resolved_configs/ exists, its YAML files are used."""
        run_dir = history_dir / "run-001"
        rc_dir = run_dir / "resolved_configs"
        rc_dir.mkdir()
        cfg = {"method_id": "m2026r1", "resolved_config": {"college_blend_factor": 0.7}}
        (rc_dir / "m2026r1.yaml").write_text(
            yaml.safe_dump(cfg), encoding="utf-8"
        )

        store = ResultsStore(history_dir=history_dir)
        configs = store.get_run_config("run-001")
        assert "m2026r1" in configs
        assert configs["m2026r1"]["college_blend_factor"] == 0.7


# ---------------------------------------------------------------------------
# TestCache
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for the Parquet caching mechanism."""

    def test_write_and_read_cache(self, store_with_cache: ResultsStore) -> None:
        # Force-load scorecards, then write cache
        store_with_cache.get_consolidated_scorecards()
        store_with_cache.write_cache()

        # Verify cache file exists
        cache_dir = store_with_cache._cache_dir
        assert cache_dir is not None
        assert (cache_dir / "consolidated_scorecards.parquet").exists()

    def test_cache_hit(self, store_with_cache: ResultsStore) -> None:
        """Second load should use cache."""
        store_with_cache.get_consolidated_scorecards()
        store_with_cache.write_cache()

        # Create a new store pointing at same dirs
        store2 = ResultsStore(
            history_dir=store_with_cache._history_dir,
            cache_dir=store_with_cache._cache_dir,
        )
        sc = store2.get_consolidated_scorecards()
        assert not sc.empty

    def test_clear_cache(self, store_with_cache: ResultsStore) -> None:
        store_with_cache.get_consolidated_scorecards()
        store_with_cache.write_cache()
        store_with_cache.clear_cache()

        cache_dir = store_with_cache._cache_dir
        assert cache_dir is not None
        assert not (cache_dir / "consolidated_scorecards.parquet").exists()

    def test_write_cache_no_cache_dir(self, store: ResultsStore) -> None:
        """write_cache() when no cache_dir is set should not raise."""
        store.get_consolidated_scorecards()
        store.write_cache()  # Should log warning but not error

    def test_clear_cache_no_cache_dir(self, store: ResultsStore) -> None:
        """clear_cache() when no cache_dir is set should not raise."""
        store.clear_cache()


# ---------------------------------------------------------------------------
# TestRefresh
# ---------------------------------------------------------------------------


class TestRefresh:
    """Tests for the refresh() method."""

    def test_refresh_reloads_index(self, history_dir: Path) -> None:
        store = ResultsStore(history_dir=history_dir)
        assert len(store.get_run_ids()) == 2

        # Add a third run to the index
        _write_index(history_dir, ["run-001", "run-002", "run-003"])
        rd = history_dir / "run-003"
        rd.mkdir()
        _write_scorecard(rd, county_mape_overall=7.9)
        _write_manifest(rd, "run-003")

        store.refresh()
        assert len(store.get_run_ids()) == 3

    def test_refresh_invalidates_cached_frames(self, store: ResultsStore) -> None:
        # Load scorecards to populate internal cache
        store.get_consolidated_scorecards()
        assert store._scorecards is not None

        store.refresh()
        assert store._scorecards is None


# ---------------------------------------------------------------------------
# TestExperimentLog
# ---------------------------------------------------------------------------


class TestExperimentLog:
    """Tests for experiment log loading."""

    def test_missing_log_returns_empty(self, history_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Point _resolve_path at tmp so the default log path doesn't exist
        monkeypatch.setattr(
            "cohort_projections.analysis.observatory.results_store._resolve_path",
            lambda rel: history_dir / rel,
        )
        store = ResultsStore(history_dir=history_dir)
        log = store.get_experiment_log()
        assert log.empty

    def test_log_loads_from_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When experiment_log CSV exists, it loads correctly."""
        hdir = tmp_path / "history"
        hdir.mkdir()
        _write_index(hdir, [])

        # Write a small experiment log
        log_path = tmp_path / "experiment_log.csv"
        pd.DataFrame({"experiment_id": ["exp-1"], "outcome": ["ok"]}).to_csv(
            log_path, index=False
        )

        # Make _resolve_path resolve relative to tmp_path
        monkeypatch.setattr(
            "cohort_projections.analysis.observatory.results_store._resolve_path",
            lambda rel: tmp_path / rel,
        )

        store = ResultsStore(
            history_dir=hdir,
            config={"experiment_log": "experiment_log.csv"},
        )
        log = store.get_experiment_log()
        assert not log.empty
        assert len(log) == 1


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """Tests for __repr__."""

    def test_repr_contains_run_count(self, store: ResultsStore) -> None:
        r = repr(store)
        assert "runs=2" in r

    def test_repr_contains_history_dir(self, store: ResultsStore) -> None:
        r = repr(store)
        assert "benchmark_history" in r
