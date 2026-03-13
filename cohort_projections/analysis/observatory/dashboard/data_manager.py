"""Centralised data manager for the Observatory Panel dashboard.

Wraps :class:`ResultsStore`, :class:`ObservatoryComparator`,
:class:`ObservatoryRecommender`, and :class:`VariantCatalog` behind a single
facade with lazy-loaded, cached properties for every consolidated DataFrame
the dashboard needs.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.analysis.observatory.comparator import ObservatoryComparator
from cohort_projections.analysis.observatory.recommender import ObservatoryRecommender
from cohort_projections.analysis.observatory.results_store import (
    ResultsStore,
)
from cohort_projections.analysis.observatory.status import (
    STATUS_PRIORITY,
    normalize_status,
    resolve_observatory_status,
    status_label,
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

_PRESET_TOP_CHALLENGERS = "Champion vs top challengers"
_PRESET_NEEDS_REVIEW = "Needs review"
_PRESET_PASSED_ONLY = "Passed only"
_PRESET_LATEST = "Latest 3"
_PRESET_ALL = "All runs"
RUN_SELECTION_PRESETS: tuple[str, ...] = (
    _PRESET_TOP_CHALLENGERS,
    _PRESET_NEEDS_REVIEW,
    _PRESET_PASSED_ONLY,
    _PRESET_LATEST,
    _PRESET_ALL,
)


def _status_label(value: object) -> str:
    """Return a short, human-readable label for a status code."""
    return status_label(value)


def _format_run_date(value: object) -> str:
    """Format YYYYMMDD-ish run dates as ISO dates for display."""
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return raw


def _short_config_label(value: object) -> str:
    """Convert verbose config IDs to a short, readable label."""
    if value is None or pd.isna(value):
        return ""
    config_id = str(value).strip()
    config_id = re.sub(r"^cfg-\d{8}-", "", config_id)
    config_id = re.sub(r"-v\d+$", "", config_id)
    config_id = config_id.replace("_", " ").replace("-", " ")
    config_id = re.sub(r"\s+", " ", config_id).strip()
    return config_id.title()


def _humanize_identifier(value: object) -> str:
    """Convert slugs and experiment IDs to title-cased display text."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"^exp-\d{8}-", "", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


def _contains_text(base: object, candidate: object) -> bool:
    """Return True if *candidate* already appears in *base* (case-insensitive)."""
    if base is None or candidate is None or pd.isna(base) or pd.isna(candidate):
        return False
    return str(candidate).strip().lower() in str(base).strip().lower()


def _has_text(value: object) -> bool:
    """Return True if a value is a non-empty string-like scalar."""
    if value is None or pd.isna(value):
        return False
    return str(value).strip() != ""


def _sort_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Sort metadata by metric first, then newest date."""
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "selected_county_mape_overall" in df.columns:
        sort_cols.append("selected_county_mape_overall")
        ascending.append(True)
    if "run_date_sort" in df.columns:
        sort_cols.append("run_date_sort")
        ascending.append(False)
    if "run_id" in df.columns:
        sort_cols.append("run_id")
        ascending.append(True)
    if not sort_cols:
        return df
    return df.sort_values(sort_cols, ascending=ascending, na_position="last")


def build_comparison_rows(
    scorecards: pd.DataFrame,
    champion_id: str | None,
) -> pd.DataFrame:
    """Select one comparison row per benchmark bundle.

    For most bundles this prefers the non-champion experiment/candidate row.
    For the current champion bundle it prefers the champion baseline row so the
    dashboard has a stable reference model.
    """
    if scorecards.empty or "run_id" not in scorecards.columns:
        return pd.DataFrame()

    rows: list[pd.Series] = []
    for run_id, group in scorecards.groupby("run_id", sort=False):
        work = group.copy()
        status = (
            work["status_at_run"].fillna("").astype(str).str.lower()
            if "status_at_run" in work.columns
            else pd.Series("", index=work.index)
        )
        if champion_id is not None and str(run_id) == champion_id:
            work["_selection_priority"] = (status != "champion").astype(int)
        else:
            work["_selection_priority"] = (status == "champion").astype(int)
        sort_cols = ["_selection_priority"]
        if "county_mape_overall" in work.columns:
            sort_cols.append("county_mape_overall")
        selected = work.sort_values(sort_cols, na_position="last").iloc[0].drop(
            labels="_selection_priority"
        )
        rows.append(selected)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_reference_rows(scorecards: pd.DataFrame) -> pd.DataFrame:
    """Select the champion/reference row for each benchmark bundle."""
    if scorecards.empty or "run_id" not in scorecards.columns:
        return pd.DataFrame()

    rows: list[pd.Series] = []
    for _, group in scorecards.groupby("run_id", sort=False):
        work = group.copy()
        status = (
            work["status_at_run"].fillna("").astype(str).str.lower()
            if "status_at_run" in work.columns
            else pd.Series("", index=work.index)
        )
        work["_selection_priority"] = (status != "champion").astype(int)
        sort_cols = ["_selection_priority"]
        if "county_mape_overall" in work.columns:
            sort_cols.append("county_mape_overall")
        selected = work.sort_values(sort_cols, na_position="last").iloc[0].drop(
            labels="_selection_priority"
        )
        rows.append(selected)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def build_run_metadata_frame(
    index: pd.DataFrame,
    scorecards: pd.DataFrame,
    experiment_log: pd.DataFrame,
    champion_id: str | None,
) -> pd.DataFrame:
    """Build one metadata row per benchmark bundle for UI controls."""
    run_ids = sorted(
        {
            *index.get("run_id", pd.Series(dtype=str)).dropna().astype(str).tolist(),
            *scorecards.get("run_id", pd.Series(dtype=str)).dropna().astype(str).tolist(),
            *experiment_log.get("run_id", pd.Series(dtype=str)).dropna().astype(str).tolist(),
        }
    )
    if not run_ids:
        return pd.DataFrame(columns=["run_id", "selector_label", "legend_label"])

    metadata = pd.DataFrame({"run_id": run_ids})

    if not index.empty and "run_id" in index.columns:
        idx_cols = [
            col
            for col in ["run_id", "run_date", "benchmark_label", "decision_status"]
            if col in index.columns
        ]
        idx_unique = index[idx_cols].drop_duplicates(subset=["run_id"])
        metadata = metadata.merge(idx_unique, on="run_id", how="left")

    comparison_rows = build_comparison_rows(scorecards, champion_id)
    if not comparison_rows.empty:
        selected_cols = [
            col
            for col in [
                "run_id",
                "method_id",
                "config_id",
                "status_at_run",
                "county_mape_overall",
                "state_ape_recent_short",
                "state_ape_recent_medium",
            ]
            if col in comparison_rows.columns
        ]
        selected = comparison_rows[selected_cols].rename(
            columns={
                "method_id": "selected_method_id",
                "config_id": "selected_config_id",
                "status_at_run": "selected_status_at_run",
                "county_mape_overall": "selected_county_mape_overall",
                "state_ape_recent_short": "selected_state_ape_recent_short",
                "state_ape_recent_medium": "selected_state_ape_recent_medium",
            }
        )
        metadata = metadata.merge(selected, on="run_id", how="left")

    reference_rows = _build_reference_rows(scorecards)
    if not reference_rows.empty:
        ref_cols = [
            col
            for col in [
                "run_id",
                "method_id",
                "config_id",
                "county_mape_overall",
                "state_ape_recent_short",
            ]
            if col in reference_rows.columns
        ]
        reference = reference_rows[ref_cols].rename(
            columns={
                "method_id": "reference_method_id",
                "config_id": "reference_config_id",
                "county_mape_overall": "reference_county_mape_overall",
                "state_ape_recent_short": "reference_state_ape_recent_short",
            }
        )
        metadata = metadata.merge(reference, on="run_id", how="left")

    if not experiment_log.empty and "run_id" in experiment_log.columns:
        log_cols = [
            col
            for col in [
                "run_id",
                "experiment_id",
                "outcome",
                "next_action",
                "hypothesis",
            ]
            if col in experiment_log.columns
        ]
        log_unique = experiment_log[log_cols].drop_duplicates(subset=["run_id"])
        metadata = metadata.merge(log_unique, on="run_id", how="left")

    metadata["run_date_label"] = metadata.get("run_date", pd.Series(dtype=object)).map(
        _format_run_date
    )
    metadata["display_name"] = ""
    if "experiment_id" in metadata.columns:
        metadata["display_name"] = metadata["experiment_id"].map(_humanize_identifier)
    if "benchmark_label" in metadata.columns:
        metadata["display_name"] = metadata["display_name"].mask(
            metadata["display_name"].eq(""),
            metadata["benchmark_label"].map(_humanize_identifier),
        )
    if "selected_config_id" in metadata.columns:
        metadata["display_name"] = metadata["display_name"].mask(
            metadata["display_name"].eq(""),
            metadata["selected_config_id"].map(_short_config_label),
        )
    metadata["display_name"] = metadata["display_name"].mask(
        metadata["display_name"].eq(""),
        metadata["run_id"].map(_humanize_identifier),
    )
    metadata["short_config"] = metadata.get(
        "selected_config_id", pd.Series(dtype=object)
    ).map(_short_config_label)

    metadata["status_code"] = metadata.apply(
        lambda row: resolve_observatory_status(
            experiment_outcome=row.get("outcome"),
            catalog_status=row.get("decision_status"),
            scorecard_status=row.get("selected_status_at_run"),
            is_champion=(champion_id is not None and row.get("run_id") == champion_id),
        ),
        axis=1,
    )
    metadata["status_label"] = metadata["status_code"].map(_status_label)
    metadata["status_priority"] = metadata["status_code"].map(
        lambda status: STATUS_PRIORITY.get(normalize_status(status), 99)
    )

    metadata["legend_label"] = metadata["display_name"]
    use_short = metadata.apply(
        lambda row: _has_text(row.get("short_config"))
        and not _contains_text(row.get("display_name"), row.get("short_config")),
        axis=1,
    )
    metadata.loc[use_short, "legend_label"] = metadata.loc[use_short, "short_config"]
    if champion_id is not None:
        metadata.loc[metadata["run_id"] == champion_id, "legend_label"] = "Champion"

    metadata["selector_label"] = metadata["display_name"]
    with_config = metadata.apply(
        lambda row: _has_text(row.get("short_config"))
        and not _contains_text(row.get("selector_label"), row.get("short_config")),
        axis=1,
    )
    metadata.loc[with_config, "selector_label"] = (
        metadata.loc[with_config, "selector_label"]
        + " | "
        + metadata.loc[with_config, "short_config"]
    )
    with_date = metadata["run_date_label"].ne("")
    metadata.loc[with_date, "selector_label"] = (
        metadata.loc[with_date, "selector_label"]
        + " | "
        + metadata.loc[with_date, "run_date_label"]
    )
    metadata["selector_label"] = (
        "[" + metadata["status_label"] + "] " + metadata["selector_label"]
    )

    metadata["run_date_sort"] = pd.to_datetime(
        metadata.get("run_date", pd.Series(dtype=object)).astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    sort_cols = ["status_priority"]
    ascending = [True]
    if "run_date_sort" in metadata.columns:
        sort_cols.append("run_date_sort")
        ascending.append(False)
    if "selected_county_mape_overall" in metadata.columns:
        sort_cols.append("selected_county_mape_overall")
        ascending.append(True)
    sort_cols.append("run_id")
    ascending.append(True)
    metadata = metadata.sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)
    return metadata


def select_run_preset(
    run_metadata: pd.DataFrame,
    preset: str,
    champion_id: str | None,
    limit: int = 3,
) -> list[str]:
    """Return run IDs for a named selector preset."""
    if run_metadata.empty:
        return []

    ordered = run_metadata["run_id"].astype(str).tolist()
    if preset == _PRESET_ALL:
        return ordered

    if preset == _PRESET_LATEST:
        latest = run_metadata.sort_values("run_date_sort", ascending=False, na_position="last")
        return latest["run_id"].head(limit).astype(str).tolist()

    if preset == _PRESET_PASSED_ONLY:
        passed = run_metadata[run_metadata["status_code"] == "passed_all_gates"]
        return passed["run_id"].astype(str).tolist()

    if preset == _PRESET_NEEDS_REVIEW:
        review = run_metadata[run_metadata["status_code"] == "needs_human_review"]
        return review["run_id"].astype(str).tolist()

    challengers = run_metadata.copy()
    if champion_id is not None:
        challengers = challengers[challengers["run_id"] != champion_id]
    challengers = _sort_by_metric(challengers)
    top = challengers["run_id"].head(limit).astype(str).tolist()
    return top or ordered[:limit]


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
    def comparison_rows(self) -> pd.DataFrame:
        """One selected comparison row per benchmark bundle."""
        return build_comparison_rows(self.scorecards, self.champion_id)

    @functools.cached_property
    def run_metadata(self) -> pd.DataFrame:
        """Human-readable benchmark metadata used by selectors and tables."""
        return build_run_metadata_frame(
            index=self.index,
            scorecards=self.scorecards,
            experiment_log=self.experiment_log,
            champion_id=self.champion_id,
        )

    @functools.cached_property
    def ordered_run_ids(self) -> list[str]:
        """Run IDs ordered for UI display and presets."""
        if self.run_metadata.empty:
            return self.run_ids
        return self.run_metadata["run_id"].astype(str).tolist()

    @functools.cached_property
    def champion_method_id(self) -> str | None:
        """Method ID used for the global champion reference line."""
        champion_id = self.champion_id
        if champion_id is None or self.run_metadata.empty:
            return None
        row = self.run_metadata[self.run_metadata["run_id"] == champion_id]
        if row.empty:
            return None
        reference_method = row.iloc[0].get("reference_method_id")
        if pd.notna(reference_method):
            return str(reference_method)
        selected_method = row.iloc[0].get("selected_method_id")
        if pd.notna(selected_method):
            return str(selected_method)
        return None

    def run_label(self, run_id: str, *, short: bool = False) -> str:
        """Return a human-readable label for a run ID."""
        if self.run_metadata.empty:
            return run_id
        row = self.run_metadata[self.run_metadata["run_id"] == run_id]
        if row.empty:
            return run_id
        column = "legend_label" if short else "selector_label"
        value = row.iloc[0].get(column)
        return str(value) if pd.notna(value) else run_id

    def run_option_map(self) -> dict[str, str]:
        """Return ordered selector options mapping label -> run ID."""
        if self.run_metadata.empty:
            return {run_id: run_id for run_id in self.run_ids}
        return {
            str(row["selector_label"]): str(row["run_id"])
            for _, row in self.run_metadata.iterrows()
        }

    def preset_run_ids(self, preset: str, limit: int = 3) -> list[str]:
        """Resolve a named preset to benchmark run IDs."""
        return select_run_preset(
            run_metadata=self.run_metadata,
            preset=preset,
            champion_id=self.champion_id,
            limit=limit,
        )

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
            "comparison_rows",
            "run_metadata",
            "ordered_run_ids",
            "champion_method_id",
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
