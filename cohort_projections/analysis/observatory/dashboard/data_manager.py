"""Centralised data manager for the Observatory Panel dashboard.

Wraps :class:`ResultsStore`, :class:`ObservatoryComparator`,
:class:`ObservatoryRecommender`, and :class:`VariantCatalog` behind a single
facade with lazy-loaded, cached properties for every consolidated DataFrame
the dashboard needs.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.analysis.observatory.comparator import ObservatoryComparator
from cohort_projections.analysis.observatory.recommender import ObservatoryRecommender
from cohort_projections.analysis.observatory.results_store import (
    ResultsStore,
    load_observatory_config,
)
from cohort_projections.analysis.observatory.variant_catalog import VariantCatalog

logger = logging.getLogger(__name__)

# Per-run CSV filenames that are NOT part of the consolidated ResultsStore
# but are useful for the dashboard.  Each is loaded from every run directory,
# tagged with ``run_id``, and concatenated.
_EXTRA_CSVS: dict[str, str] = {
    "annual_horizon_summary": "annual_horizon_summary.csv",
    "bias_analysis": "bias_analysis.csv",
    "county_report_cards": "county_report_cards.csv",
    "outlier_flags": "outlier_flags.csv",
    "residual_diagnostics": "residual_diagnostics.csv",
    "uncertainty_summary": "uncertainty_summary.csv",
}


class DashboardDataManager:
    """Single data facade for the Observatory dashboard.

    Parameters
    ----------
    store:
        A pre-built :class:`ResultsStore`.  If ``None``, one is created
        via :meth:`ResultsStore.from_config`.
    config:
        Pre-loaded observatory config dict.  If ``None`` and *store* is
        also ``None``, the default config file is read.
    """

    def __init__(
        self,
        store: ResultsStore | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if store is not None:
            self._store = store
        else:
            self._store = ResultsStore.from_config()

        self._config = config or self._store._config  # noqa: SLF001

        # Eagerly build sub-components (cheap — they are lazy internally)
        self._comparator = ObservatoryComparator(
            self._store, config=self._config
        )
        self._recommender = ObservatoryRecommender(
            self._store,
            comparator=self._comparator,
            config=self._config.get("recommender"),
            bounds_catalog=self._try_build_catalog(),
        )
        self._catalog = self._try_build_catalog()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_build_catalog(self) -> VariantCatalog | None:
        """Attempt to build a :class:`VariantCatalog` from config.

        Returns ``None`` if the catalog YAML does not exist rather than
        raising.
        """
        try:
            return VariantCatalog()
        except FileNotFoundError:
            logger.warning("VariantCatalog YAML not found; catalog unavailable.")
            return None

    def _load_extra_csv(self, filename: str) -> pd.DataFrame:
        """Load a per-run CSV from every run directory and concatenate.

        Follows the same pattern as
        :meth:`ResultsStore._load_or_cache` — iterates all run IDs,
        reads each CSV, adds a ``run_id`` column, and concatenates.
        Missing files are silently skipped.

        Parameters
        ----------
        filename:
            The CSV basename to look for inside each run directory
            (e.g. ``"bias_analysis.csv"``).

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame with a ``run_id`` column, or an
            empty DataFrame if no files were found.
        """
        history_dir: Path = self._store._history_dir  # noqa: SLF001
        run_ids = self._store.get_run_ids()
        frames: list[pd.DataFrame] = []

        for run_id in run_ids:
            csv_path = history_dir / run_id / filename
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                logger.warning(
                    "Failed to read %s for run %s; skipping.",
                    filename,
                    run_id,
                )
                continue
            df["run_id"] = run_id
            frames.append(df)

        if not frames:
            logger.debug("No %s files found across %d runs.", filename, len(run_ids))
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True, sort=False)

    # ------------------------------------------------------------------
    # Sub-component accessors
    # ------------------------------------------------------------------

    @property
    def store(self) -> ResultsStore:
        """The underlying :class:`ResultsStore`."""
        return self._store

    @property
    def comparator(self) -> ObservatoryComparator:
        """The :class:`ObservatoryComparator` instance."""
        return self._comparator

    @property
    def recommender(self) -> ObservatoryRecommender:
        """The :class:`ObservatoryRecommender` instance."""
        return self._recommender

    @property
    def catalog(self) -> VariantCatalog | None:
        """The :class:`VariantCatalog`, or ``None`` if unavailable."""
        return self._catalog

    # ------------------------------------------------------------------
    # Convenience index properties
    # ------------------------------------------------------------------

    @property
    def run_ids(self) -> list[str]:
        """All known run IDs."""
        return self._store.get_run_ids()

    @functools.cached_property
    def champion_id(self) -> str | None:
        """Auto-detected champion run ID (from comparator logic)."""
        scorecards = self._store.get_consolidated_scorecards()
        if scorecards.empty:
            return None
        return self._comparator._resolve_champion(scorecards)  # noqa: SLF001

    @functools.cached_property
    def index(self) -> pd.DataFrame:
        """The benchmark index DataFrame."""
        return self._store.get_index()

    @functools.cached_property
    def experiment_log(self) -> pd.DataFrame:
        """The experiment log DataFrame."""
        return self._store.get_experiment_log()

    # ------------------------------------------------------------------
    # Consolidated DataFrames from ResultsStore (lazy / cached)
    # ------------------------------------------------------------------

    @functools.cached_property
    def scorecards(self) -> pd.DataFrame:
        """Consolidated scorecards across all runs."""
        return self._store.get_consolidated_scorecards()

    @functools.cached_property
    def county_metrics(self) -> pd.DataFrame:
        """Consolidated county-level metrics across all runs."""
        return self._store.get_consolidated_county_metrics()

    @functools.cached_property
    def state_metrics(self) -> pd.DataFrame:
        """Consolidated state-level metrics across all runs."""
        return self._store.get_consolidated_state_metrics()

    @functools.cached_property
    def projection_curves(self) -> pd.DataFrame:
        """Consolidated projection curves across all runs."""
        return self._store.get_consolidated_projection_curves()

    @functools.cached_property
    def sensitivity_summary(self) -> pd.DataFrame:
        """Consolidated sensitivity summaries across all runs."""
        return self._store.get_consolidated_sensitivity_summary()

    # ------------------------------------------------------------------
    # Extra per-run CSVs (not in ResultsStore)
    # ------------------------------------------------------------------

    @functools.cached_property
    def annual_horizon_summary(self) -> pd.DataFrame:
        """Concatenated ``annual_horizon_summary.csv`` across runs."""
        return self._load_extra_csv(_EXTRA_CSVS["annual_horizon_summary"])

    @functools.cached_property
    def bias_analysis(self) -> pd.DataFrame:
        """Concatenated ``bias_analysis.csv`` across runs."""
        return self._load_extra_csv(_EXTRA_CSVS["bias_analysis"])

    @functools.cached_property
    def county_report_cards(self) -> pd.DataFrame:
        """Concatenated ``county_report_cards.csv`` across runs."""
        return self._load_extra_csv(_EXTRA_CSVS["county_report_cards"])

    @functools.cached_property
    def outlier_flags(self) -> pd.DataFrame:
        """Concatenated ``outlier_flags.csv`` across runs."""
        return self._load_extra_csv(_EXTRA_CSVS["outlier_flags"])

    @functools.cached_property
    def residual_diagnostics(self) -> pd.DataFrame:
        """Concatenated ``residual_diagnostics.csv`` across runs."""
        return self._load_extra_csv(_EXTRA_CSVS["residual_diagnostics"])

    @functools.cached_property
    def uncertainty_summary(self) -> pd.DataFrame:
        """Concatenated ``uncertainty_summary.csv`` across runs."""
        return self._load_extra_csv(_EXTRA_CSVS["uncertainty_summary"])

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Clear all cached data and re-scan the results store.

        After calling this method, the next access to any property will
        trigger a fresh load from disk.
        """
        # Invalidate the underlying store
        self._store.refresh()

        # Clear all cached_property values by deleting them from the
        # instance __dict__ — they will be recomputed on next access.
        cached_attrs = [
            "champion_id",
            "index",
            "experiment_log",
            "scorecards",
            "county_metrics",
            "state_metrics",
            "projection_curves",
            "sensitivity_summary",
            "annual_horizon_summary",
            "bias_analysis",
            "county_report_cards",
            "outlier_flags",
            "residual_diagnostics",
            "uncertainty_summary",
        ]
        for attr in cached_attrs:
            self.__dict__.pop(attr, None)

        # Rebuild sub-components
        self._comparator = ObservatoryComparator(
            self._store, config=self._config
        )
        self._catalog = self._try_build_catalog()
        self._recommender = ObservatoryRecommender(
            self._store,
            comparator=self._comparator,
            config=self._config.get("recommender"),
            bounds_catalog=self._catalog,
        )

        logger.info("DashboardDataManager refreshed.")

    def __repr__(self) -> str:
        n_runs = len(self.run_ids)
        return f"DashboardDataManager(runs={n_runs}, champion={self.champion_id!r})"
