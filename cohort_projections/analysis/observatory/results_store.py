"""Consolidated results loading and storage for the Projection Observatory.

Provides a single entry point for loading, indexing, and querying all benchmark
results across runs stored in the benchmark history directory.

The store is lazy: consolidated DataFrames are not built until a specific
getter is called.  Results can optionally be cached to Parquet files in an
observatory cache directory to speed up repeated access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "observatory_config.yaml"

# CSV filenames expected inside each run directory.
_COUNTY_METRICS_FILE = "county_metrics.csv"
_STATE_METRICS_FILE = "state_metrics.csv"
_SUMMARY_SCORECARD_FILE = "summary_scorecard.csv"
_PROJECTION_CURVES_FILE = "projection_curves.csv"
_SENSITIVITY_SUMMARY_FILE = "sensitivity_summary.csv"
_MANIFEST_FILE = "manifest.json"

# Parquet cache filenames written to ``cache_dir``.
_CACHE_COUNTY = "consolidated_county_metrics.parquet"
_CACHE_STATE = "consolidated_state_metrics.parquet"
_CACHE_SCORECARDS = "consolidated_scorecards.parquet"
_CACHE_PROJECTION_CURVES = "consolidated_projection_curves.parquet"
_CACHE_SENSITIVITY = "consolidated_sensitivity_summary.parquet"


def load_observatory_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the observatory configuration from YAML.

    Parameters
    ----------
    config_path:
        Path to the observatory config file.  Defaults to
        ``config/observatory_config.yaml`` relative to the project root.

    Returns
    -------
    dict
        The parsed ``observatory`` section of the config.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the file does not contain a valid ``observatory`` key.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Observatory config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "observatory" not in raw:
        raise ValueError(
            f"Observatory config must contain an 'observatory' key: {config_path}"
        )
    return raw["observatory"]


def _resolve_path(relative: str) -> Path:
    """Resolve a project-relative path string to an absolute ``Path``."""
    return PROJECT_ROOT / relative


class ResultsStore:
    """Consolidated results store for benchmark history.

    Parameters
    ----------
    history_dir:
        Directory containing benchmark run sub-directories and ``index.csv``.
    cache_dir:
        Optional directory for Parquet caches.  When provided the store will
        write consolidated DataFrames here and read them back on subsequent
        instantiations if they are still up-to-date.
    config:
        Pre-loaded observatory config dict.  If *None*, the store operates
        without recommendation/comparison settings.
    """

    def __init__(
        self,
        history_dir: Path,
        cache_dir: Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._history_dir = Path(history_dir)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._config = config or {}

        # Index state
        self._index: pd.DataFrame | None = None
        self._run_ids: list[str] | None = None

        # Lazy-loaded consolidated frames
        self._county_metrics: pd.DataFrame | None = None
        self._state_metrics: pd.DataFrame | None = None
        self._scorecards: pd.DataFrame | None = None
        self._projection_curves: pd.DataFrame | None = None
        self._sensitivity_summary: pd.DataFrame | None = None

        # Manifests keyed by run_id
        self._manifests: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> ResultsStore:
        """Create a ``ResultsStore`` from the observatory config file.

        This is the recommended entry point.  It reads path settings from the
        YAML config so that nothing is hard-coded in calling code.
        """
        cfg = load_observatory_config(config_path)
        history_dir = _resolve_path(cfg["history_dir"])
        cache_dir = _resolve_path(cfg["cache_dir"]) if cfg.get("cache_dir") else None
        return cls(history_dir=history_dir, cache_dir=cache_dir, config=cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the history directory and invalidate all cached data."""
        self._index = None
        self._run_ids = None
        self._county_metrics = None
        self._state_metrics = None
        self._scorecards = None
        self._projection_curves = None
        self._sensitivity_summary = None
        self._manifests.clear()
        self._load_index()
        logger.info("ResultsStore refreshed with %d runs.", len(self.get_run_ids()))

    def get_run_ids(self) -> list[str]:
        """Return all known run IDs from the index."""
        index = self._ensure_index()
        if self._run_ids is None:
            self._run_ids = sorted(index["run_id"].unique().tolist())
        return list(self._run_ids)

    def get_index(self) -> pd.DataFrame:
        """Return the full benchmark index as a DataFrame.

        The index is loaded from ``index.csv`` in the history directory on
        first access.
        """
        return self._ensure_index().copy()

    def get_run_manifest(self, run_id: str) -> dict[str, Any]:
        """Load and return the manifest for a single run.

        Parameters
        ----------
        run_id:
            The benchmark run identifier (directory name).

        Returns
        -------
        dict
            Parsed ``manifest.json`` contents.

        Raises
        ------
        FileNotFoundError
            If the manifest file does not exist.
        """
        if run_id not in self._manifests:
            manifest_path = self._history_dir / run_id / _MANIFEST_FILE
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest not found for run {run_id}: {manifest_path}"
                )
            self._manifests[run_id] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        return dict(self._manifests[run_id])

    def get_consolidated_county_metrics(self) -> pd.DataFrame:
        """All runs' county-level metrics with a ``run_id`` column.

        Returns an outer-joined DataFrame so that schema drift across runs is
        handled gracefully (missing columns filled with ``NaN``).
        """
        if self._county_metrics is None:
            self._county_metrics = self._load_or_cache(
                filename=_COUNTY_METRICS_FILE,
                cache_name=_CACHE_COUNTY,
            )
        return self._county_metrics.copy()

    def get_consolidated_state_metrics(self) -> pd.DataFrame:
        """All runs' state-level metrics with a ``run_id`` column."""
        if self._state_metrics is None:
            self._state_metrics = self._load_or_cache(
                filename=_STATE_METRICS_FILE,
                cache_name=_CACHE_STATE,
            )
        return self._state_metrics.copy()

    def get_consolidated_scorecards(self) -> pd.DataFrame:
        """One row per run x method, with all scorecard columns."""
        if self._scorecards is None:
            self._scorecards = self._load_or_cache(
                filename=_SUMMARY_SCORECARD_FILE,
                cache_name=_CACHE_SCORECARDS,
            )
        return self._scorecards.copy()

    def get_consolidated_projection_curves(self) -> pd.DataFrame:
        """All runs' projection curves with a ``run_id`` column."""
        if self._projection_curves is None:
            self._projection_curves = self._load_or_cache(
                filename=_PROJECTION_CURVES_FILE,
                cache_name=_CACHE_PROJECTION_CURVES,
            )
        return self._projection_curves.copy()

    def get_consolidated_sensitivity_summary(self) -> pd.DataFrame:
        """All runs' sensitivity summaries with a ``run_id`` column."""
        if self._sensitivity_summary is None:
            self._sensitivity_summary = self._load_or_cache(
                filename=_SENSITIVITY_SUMMARY_FILE,
                cache_name=_CACHE_SENSITIVITY,
            )
        return self._sensitivity_summary.copy()

    def get_run_config(self, run_id: str) -> dict[str, Any]:
        """Return the resolved MethodConfig dict used for a specific run.

        This reads from the ``resolved_configs/`` sub-directory inside the run
        directory.  If no resolved configs are found, falls back to the
        ``methods`` list inside ``manifest.json``.

        Parameters
        ----------
        run_id:
            The benchmark run identifier.

        Returns
        -------
        dict
            Mapping of ``method_id`` to its resolved config dict.
        """
        run_dir = self._history_dir / run_id
        configs: dict[str, Any] = {}

        resolved_dir = run_dir / "resolved_configs"
        if resolved_dir.is_dir():
            for config_file in sorted(resolved_dir.glob("*.yaml")):
                parsed = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    method_id = parsed.get("method_id", config_file.stem)
                    # Extract the resolved_config sub-dict if present;
                    # this is the actual MethodConfig parameters.
                    rc = parsed.get("resolved_config", parsed)
                    configs[method_id] = rc
                else:
                    configs[config_file.stem] = parsed

        if not configs:
            # Fall back to manifest methods list
            manifest = self.get_run_manifest(run_id)
            for method_info in manifest.get("methods", []):
                method_id = method_info.get("method_id", "unknown")
                rc = method_info.get("resolved_config", method_info)
                configs[method_id] = rc

        return configs

    def get_experiment_log(self) -> pd.DataFrame:
        """Load the experiment log CSV if it exists.

        Returns
        -------
        pd.DataFrame
            The experiment log, or an empty DataFrame if the file is missing.
        """
        log_rel = self._config.get(
            "experiment_log", "data/analysis/experiments/experiment_log.csv"
        )
        log_path = _resolve_path(log_rel)
        if not log_path.exists():
            logger.warning("Experiment log not found: %s", log_path)
            return pd.DataFrame()
        return pd.read_csv(log_path)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def write_cache(self) -> None:
        """Force-write all loaded consolidated frames to Parquet cache.

        Calling this after the first load avoids re-reading all individual run
        CSVs on subsequent instantiations.
        """
        if self._cache_dir is None:
            logger.warning("No cache_dir configured; skipping cache write.")
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        pairs: list[tuple[str, pd.DataFrame | None]] = [
            (_CACHE_COUNTY, self._county_metrics),
            (_CACHE_STATE, self._state_metrics),
            (_CACHE_SCORECARDS, self._scorecards),
            (_CACHE_PROJECTION_CURVES, self._projection_curves),
            (_CACHE_SENSITIVITY, self._sensitivity_summary),
        ]
        for cache_name, frame in pairs:
            if frame is not None and not frame.empty:
                out_path = self._cache_dir / cache_name
                frame.to_parquet(out_path, index=False)
                logger.info("Wrote cache: %s (%d rows)", out_path, len(frame))

    def clear_cache(self) -> None:
        """Delete all Parquet cache files."""
        if self._cache_dir is None:
            return
        for name in (
            _CACHE_COUNTY,
            _CACHE_STATE,
            _CACHE_SCORECARDS,
            _CACHE_PROJECTION_CURVES,
            _CACHE_SENSITIVITY,
        ):
            cache_path = self._cache_dir / name
            if cache_path.exists():
                cache_path.unlink()
                logger.info("Removed cache: %s", cache_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_index(self) -> pd.DataFrame:
        """Return the index, loading it on first access."""
        if self._index is None:
            self._load_index()
        assert self._index is not None  # noqa: S101
        return self._index

    def _load_index(self) -> None:
        """Read ``index.csv`` from the history directory."""
        index_path = self._history_dir / "index.csv"
        if not index_path.exists():
            logger.warning("Benchmark index not found: %s", index_path)
            self._index = pd.DataFrame()
            self._run_ids = []
            return
        self._index = pd.read_csv(index_path)
        self._run_ids = None  # Force re-derivation

    def _load_or_cache(
        self,
        filename: str,
        cache_name: str,
    ) -> pd.DataFrame:
        """Load a consolidated DataFrame from cache or by scanning all runs.

        If a Parquet cache exists and contains the same set of run IDs as the
        current index, the cache is used directly.  Otherwise every run
        directory is scanned and the individual CSVs are concatenated.

        Parameters
        ----------
        filename:
            The CSV filename to look for inside each run directory.
        cache_name:
            The Parquet filename used inside ``cache_dir``.

        Returns
        -------
        pd.DataFrame
            Consolidated frame with a ``run_id`` column.
        """
        run_ids = self.get_run_ids()

        # Try Parquet cache
        if self._cache_dir is not None:
            cache_path = self._cache_dir / cache_name
            if cache_path.exists():
                cached = pd.read_parquet(cache_path)
                cached_ids = set(cached["run_id"].unique()) if "run_id" in cached.columns else set()
                if cached_ids == set(run_ids):
                    logger.debug("Cache hit for %s (%d runs).", cache_name, len(run_ids))
                    return cached
                logger.info(
                    "Cache stale for %s (cached %d vs index %d runs); rebuilding.",
                    cache_name,
                    len(cached_ids),
                    len(run_ids),
                )

        # Load from individual run CSVs
        frames: list[pd.DataFrame] = []
        for run_id in run_ids:
            csv_path = self._history_dir / run_id / filename
            if not csv_path.exists():
                logger.debug("Missing %s for run %s; skipping.", filename, run_id)
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                logger.warning(
                    "Failed to read %s for run %s; skipping.", filename, run_id
                )
                continue
            df["run_id"] = run_id
            frames.append(df)

        if not frames:
            logger.warning("No %s files found across %d runs.", filename, len(run_ids))
            return pd.DataFrame()

        # Outer-join concat handles schema drift across runs
        consolidated = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(
            "Loaded %s: %d rows across %d runs.",
            filename,
            len(consolidated),
            len(frames),
        )
        return consolidated

    def __repr__(self) -> str:
        n_runs = len(self.get_run_ids())
        return (
            f"ResultsStore(history_dir={self._history_dir!r}, "
            f"runs={n_runs}, "
            f"cache_dir={self._cache_dir!r})"
        )
