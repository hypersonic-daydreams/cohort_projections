"""Multi-run comparison, ranking, and Pareto analysis for the Projection Observatory.

Provides N-way comparison across completed benchmark runs, computing
per-metric rankings, champion deltas, county-group breakdowns, and
Pareto frontier identification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cohort_projections.analysis.observatory.results_store import ResultsStore

logger = logging.getLogger(__name__)

# Scorecard columns that are comparable numeric metrics (lower is better).
METRIC_COLUMNS: list[str] = [
    "county_mape_overall",
    "county_mape_rural",
    "county_mape_bakken",
    "county_mape_urban_college",
    "state_ape_recent_short",
    "state_ape_recent_medium",
]

# Sentinel county columns from the scorecard.
SENTINEL_COLUMNS: list[str] = [
    "sentinel_cass_mape",
    "sentinel_grand_forks_mape",
    "sentinel_ward_mape",
    "sentinel_burleigh_mape",
    "sentinel_williams_mape",
    "sentinel_mckenzie_mape",
]

# County groups as used by the benchmarking module.
COUNTY_GROUPS: list[str] = [
    "overall",
    "Rural",
    "Bakken",
    "Urban/College",
]

# Map from county group name to the scorecard column that stores its MAPE.
_GROUP_TO_SCORECARD_COL: dict[str, str] = {
    "overall": "county_mape_overall",
    "Rural": "county_mape_rural",
    "Bakken": "county_mape_bakken",
    "Urban/College": "county_mape_urban_college",
}

# Default config for comparison behaviour.
_DEFAULT_CONFIG: dict[str, Any] = {
    "comparison": {
        "primary_metric": "county_mape_overall",
        "secondary_metrics": [
            "county_mape_rural",
            "county_mape_bakken",
            "county_mape_urban_college",
            "state_ape_recent_short",
            "state_ape_recent_medium",
        ],
    },
}


@dataclass
class ComparisonResult:
    """Container for a full N-way comparison analysis."""

    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    deltas: pd.DataFrame = field(default_factory=pd.DataFrame)
    county_group_impact: pd.DataFrame = field(default_factory=pd.DataFrame)
    pareto_runs: list[str] = field(default_factory=list)
    best_per_group: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


class ObservatoryComparator:
    """Compare and rank benchmark runs stored in a :class:`ResultsStore`.

    Parameters
    ----------
    store
        A ``ResultsStore`` instance providing access to scorecard data.
    config
        Optional configuration dict.  Expected shape::

            {"comparison": {
                "primary_metric": "county_mape_overall",
                "secondary_metrics": ["county_mape_rural", ...],
                "champion_run_id": "<run_id>"   # optional
            }}

        If ``None``, sensible defaults are used.
    """

    def __init__(self, store: ResultsStore, config: dict[str, Any] | None = None) -> None:
        self.store = store
        self.config: dict[str, Any] = config or _DEFAULT_CONFIG
        comparison_cfg = self.config.get("comparison", {})
        self.primary_metric: str = comparison_cfg.get(
            "primary_metric", "county_mape_overall"
        )
        self.secondary_metrics: list[str] = comparison_cfg.get(
            "secondary_metrics",
            _DEFAULT_CONFIG["comparison"]["secondary_metrics"],
        )
        self._champion_run_id: str | None = comparison_cfg.get("champion_run_id")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_scorecards(self) -> pd.DataFrame:
        """Load and concatenate scorecards from all runs in the store.

        Returns a DataFrame with one row per (run_id, method_id) pair
        containing all scorecard columns.  Delegates to
        :meth:`ResultsStore.get_consolidated_scorecards`.
        """
        return self.store.get_consolidated_scorecards()

    def _detect_champion(self, scorecards: pd.DataFrame) -> str | None:
        """Auto-detect the champion run_id from the scorecard data.

        Heuristic: the champion is the run whose ``status_at_run`` is
        ``'champion'``.  If multiple exist, pick the one with the lowest
        ``county_mape_overall``.  Falls back to the run with the best
        primary metric overall.
        """
        if scorecards.empty:
            return None

        champions = scorecards[
            scorecards["status_at_run"].str.lower() == "champion"
        ]
        if not champions.empty:
            best_idx = champions[self.primary_metric].idxmin()
            return str(champions.loc[best_idx, "run_id"])

        # Fallback: best primary metric across all rows.
        best_idx = scorecards[self.primary_metric].idxmin()
        return str(scorecards.loc[best_idx, "run_id"])

    def _resolve_champion(
        self, scorecards: pd.DataFrame, champion_run_id: str | None = None
    ) -> str | None:
        """Return the champion run_id, resolving from config/auto-detect."""
        if champion_run_id is not None:
            return champion_run_id
        if self._champion_run_id is not None:
            return self._champion_run_id
        return self._detect_champion(scorecards)

    def _available_metrics(self, scorecards: pd.DataFrame) -> list[str]:
        """Return the subset of METRIC_COLUMNS actually present in the data."""
        return [c for c in METRIC_COLUMNS + SENTINEL_COLUMNS if c in scorecards.columns]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_all(self) -> pd.DataFrame:
        """Rank all runs by every available metric.

        Returns a DataFrame with ``run_id``, ``method_id``, ``config_id``,
        each metric column, and a ``rank_<metric>`` column for each metric.
        """
        scorecards = self._load_all_scorecards()
        if scorecards.empty:
            return pd.DataFrame()

        metrics = self._available_metrics(scorecards)
        id_cols = ["run_id", "method_id", "config_id"]
        present_id_cols = [c for c in id_cols if c in scorecards.columns]
        result = scorecards[present_id_cols + metrics].copy()

        for metric in metrics:
            result[f"rank_{metric}"] = result[metric].rank(method="min").astype(int)

        # Sort by primary metric rank.
        primary_rank_col = f"rank_{self.primary_metric}"
        if primary_rank_col in result.columns:
            result = result.sort_values(primary_rank_col).reset_index(drop=True)

        return result

    def rank_by(self, metric: str, ascending: bool = True) -> pd.DataFrame:
        """Rank runs by a specific metric.

        Parameters
        ----------
        metric
            Column name in the scorecard to rank by.
        ascending
            If ``True``, lower values rank first (default for error metrics).

        Returns a DataFrame sorted by the metric with a ``rank`` column.
        """
        scorecards = self._load_all_scorecards()
        if scorecards.empty:
            return pd.DataFrame()

        if metric not in scorecards.columns:
            raise ValueError(
                f"Metric '{metric}' not found in scorecards. "
                f"Available: {list(scorecards.columns)}"
            )

        id_cols = ["run_id", "method_id", "config_id"]
        present_id_cols = [c for c in id_cols if c in scorecards.columns]
        result = scorecards[present_id_cols + [metric]].copy()
        result["rank"] = result[metric].rank(ascending=ascending, method="min").astype(int)
        return result.sort_values("rank").reset_index(drop=True)

    def compute_deltas(self, champion_run_id: str | None = None) -> pd.DataFrame:
        """Compute improvement deltas vs champion for every run.

        A negative delta means the run improved (lower error) relative
        to the champion.

        Parameters
        ----------
        champion_run_id
            Explicit champion run ID.  If ``None``, auto-detected.

        Returns a DataFrame with ``run_id``, ``method_id``, ``config_id``,
        and ``delta_<metric>`` columns for each metric.
        """
        scorecards = self._load_all_scorecards()
        if scorecards.empty:
            return pd.DataFrame()

        champion_id = self._resolve_champion(scorecards, champion_run_id)
        if champion_id is None:
            logger.warning("No champion could be resolved; returning empty deltas.")
            return pd.DataFrame()

        champion_rows = scorecards[scorecards["run_id"] == champion_id]
        if champion_rows.empty:
            raise ValueError(f"Champion run_id not found in scorecards: {champion_id}")

        # Use the first champion row (there should typically be one per method
        # family, but the champion baseline is the reference).
        champion = champion_rows.iloc[0]
        metrics = self._available_metrics(scorecards)

        id_cols = ["run_id", "method_id", "config_id"]
        present_id_cols = [c for c in id_cols if c in scorecards.columns]
        result = scorecards[present_id_cols].copy()

        for metric in metrics:
            champion_val = float(champion[metric])
            result[f"delta_{metric}"] = scorecards[metric] - champion_val

        return result

    def pareto_frontier(self, x_metric: str, y_metric: str) -> pd.DataFrame:
        """Find Pareto-optimal runs on two metrics (both minimise).

        A run is Pareto-optimal if no other run is strictly better on
        both metrics simultaneously.

        Parameters
        ----------
        x_metric
            First metric column name.
        y_metric
            Second metric column name.

        Returns a DataFrame containing only the Pareto-optimal rows,
        sorted by ``x_metric``.
        """
        scorecards = self._load_all_scorecards()
        if scorecards.empty:
            return pd.DataFrame()

        for m in (x_metric, y_metric):
            if m not in scorecards.columns:
                raise ValueError(f"Metric '{m}' not found in scorecards.")

        xs = scorecards[x_metric].values
        ys = scorecards[y_metric].values
        n = len(xs)
        is_pareto = [True] * n

        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                # j dominates i if j is <= on both and < on at least one.
                if (
                    xs[j] <= xs[i]
                    and ys[j] <= ys[i]
                    and (xs[j] < xs[i] or ys[j] < ys[i])
                ):
                    is_pareto[i] = False
                    break

        pareto_df = scorecards[is_pareto].copy()
        return pareto_df.sort_values(x_metric).reset_index(drop=True)

    def county_group_impact(self) -> pd.DataFrame:
        """Show per-county-group MAPE delta vs champion for each run.

        Returns a DataFrame with ``run_id``, ``method_id``, ``config_id``,
        and one column per county group containing the MAPE delta
        (negative = improvement).
        """
        scorecards = self._load_all_scorecards()
        if scorecards.empty:
            return pd.DataFrame()

        champion_id = self._resolve_champion(scorecards)
        if champion_id is None:
            return pd.DataFrame()

        champion_rows = scorecards[scorecards["run_id"] == champion_id]
        if champion_rows.empty:
            return pd.DataFrame()
        champion = champion_rows.iloc[0]

        id_cols = ["run_id", "method_id", "config_id"]
        present_id_cols = [c for c in id_cols if c in scorecards.columns]
        result = scorecards[present_id_cols].copy()

        for group, col in _GROUP_TO_SCORECARD_COL.items():
            if col in scorecards.columns:
                champion_val = float(champion[col])
                result[f"delta_{group}"] = scorecards[col] - champion_val

        return result

    def best_variant_per_group(self) -> dict[str, dict[str, str]]:
        """Identify which run is best for each county group.

        Returns a dict mapping county group name to a dict with
        ``run_id`` and ``config_id`` for the variant with the lowest
        MAPE for that group.
        """
        scorecards = self._load_all_scorecards()
        if scorecards.empty:
            return {}

        best: dict[str, dict[str, str]] = {}
        for group, col in _GROUP_TO_SCORECARD_COL.items():
            if col in scorecards.columns:
                best_idx = scorecards[col].idxmin()
                run_id = str(scorecards.loc[best_idx, "run_id"])
                config_id = str(
                    scorecards.loc[best_idx, "config_id"]
                ) if "config_id" in scorecards.columns else run_id
                best[group] = {"run_id": run_id, "config_id": config_id}
        return best

    def full_comparison(
        self, champion_run_id: str | None = None
    ) -> ComparisonResult:
        """Run the complete comparison analysis.

        Parameters
        ----------
        champion_run_id
            Explicit champion run ID.  If ``None``, auto-detected.

        Returns a :class:`ComparisonResult` aggregating all analyses.
        """
        ranking = self.rank_all()
        if ranking.empty:
            return ComparisonResult()

        scorecards = self._load_all_scorecards()
        champion_id = self._resolve_champion(scorecards, champion_run_id)
        deltas = self.compute_deltas(champion_id)
        group_impact = self.county_group_impact()
        best_groups = self.best_variant_per_group()

        # Pareto frontier on primary vs the first secondary metric.
        pareto_x = self.primary_metric
        pareto_y = (
            self.secondary_metrics[0]
            if self.secondary_metrics
            else "state_ape_recent_short"
        )
        try:
            pareto_df = self.pareto_frontier(pareto_x, pareto_y)
            pareto_run_ids = pareto_df["run_id"].tolist() if "run_id" in pareto_df.columns else []
        except ValueError:
            pareto_run_ids = []

        # Summary statistics.
        n_runs = len(scorecards)
        metrics = self._available_metrics(scorecards)
        summary: dict[str, Any] = {
            "n_runs": n_runs,
            "champion_run_id": champion_id,
            "pareto_metric_x": pareto_x,
            "pareto_metric_y": pareto_y,
            "n_pareto_optimal": len(pareto_run_ids),
        }
        for metric in metrics:
            vals = scorecards[metric].dropna()
            if not vals.empty:
                summary[f"{metric}_min"] = round(float(vals.min()), 6)
                summary[f"{metric}_max"] = round(float(vals.max()), 6)
                summary[f"{metric}_mean"] = round(float(vals.mean()), 6)

        return ComparisonResult(
            ranking=ranking,
            deltas=deltas,
            county_group_impact=group_impact,
            pareto_runs=pareto_run_ids,
            best_per_group=best_groups,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_ranking_table(self, top_n: int = 10) -> str:
        """Format a human-readable ranking table for console output.

        Parameters
        ----------
        top_n
            Maximum number of runs to display.

        Returns a formatted string with aligned columns.
        """
        ranking = self.rank_all()
        if ranking.empty:
            return "No runs available for ranking."

        ranking = ranking.head(top_n)

        # Build column specs: id columns + primary metric + rank.
        display_cols: list[str] = []
        if "run_id" in ranking.columns:
            display_cols.append("run_id")
        if "method_id" in ranking.columns:
            display_cols.append("method_id")
        if "config_id" in ranking.columns:
            display_cols.append("config_id")

        metric_cols = [self.primary_metric] + [
            m for m in self.secondary_metrics if m in ranking.columns
        ]
        rank_cols = [f"rank_{m}" for m in metric_cols if f"rank_{m}" in ranking.columns]
        display_cols.extend(metric_cols)
        display_cols.extend(rank_cols)

        # Compute column widths.
        widths: dict[str, int] = {}
        for col in display_cols:
            header_len = len(col)
            max_val_len = ranking[col].astype(str).str.len().max() if col in ranking.columns else 0
            widths[col] = max(header_len, int(max_val_len)) + 2

        # Header line.
        header = "".join(col.ljust(widths[col]) for col in display_cols)
        separator = "-" * len(header)
        lines = [
            "BENCHMARK RANKING",
            f"(sorted by {self.primary_metric}, top {top_n})",
            "",
            separator,
            header,
            separator,
        ]

        for _, row in ranking.iterrows():
            cells: list[str] = []
            for col in display_cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    cells.append(f"{val:.4f}".ljust(widths[col]))
                else:
                    cells.append(str(val).ljust(widths[col]))
            lines.append("".join(cells))

        lines.append(separator)
        return "\n".join(lines)

    def format_comparison_summary(self, result: ComparisonResult) -> str:
        """Format the full comparison as a human-readable summary.

        Parameters
        ----------
        result
            A :class:`ComparisonResult` from :meth:`full_comparison`.

        Returns a formatted multi-line string.
        """
        lines: list[str] = []
        sep = "=" * 72

        lines.append(sep)
        lines.append("OBSERVATORY COMPARISON SUMMARY")
        lines.append(sep)
        lines.append("")

        # Summary stats.
        n_runs = result.summary.get("n_runs", 0)
        champion = result.summary.get("champion_run_id", "unknown")
        lines.append(f"  Runs compared:    {n_runs}")
        lines.append(f"  Champion:         {champion}")
        lines.append(f"  Pareto-optimal:   {result.summary.get('n_pareto_optimal', 0)}")
        lines.append("")

        # Metric ranges.
        lines.append("METRIC RANGES")
        lines.append("-" * 72)
        metrics = [self.primary_metric] + self.secondary_metrics
        for metric in metrics:
            lo = result.summary.get(f"{metric}_min")
            hi = result.summary.get(f"{metric}_max")
            avg = result.summary.get(f"{metric}_mean")
            if lo is not None:
                lines.append(
                    f"  {metric:<35s}  min={lo:.4f}  max={hi:.4f}  mean={avg:.4f}"
                )
        lines.append("")

        # Best per county group.
        if result.best_per_group:
            lines.append("BEST RUN PER COUNTY GROUP")
            lines.append("-" * 72)
            for group, info in result.best_per_group.items():
                if isinstance(info, dict):
                    label = info.get("config_id", info.get("run_id", "?"))
                else:
                    label = str(info)
                lines.append(f"  {group:<20s}  {label}")
            lines.append("")

        # Pareto frontier.
        if result.pareto_runs:
            lines.append("PARETO FRONTIER")
            lines.append(
                f"({result.summary.get('pareto_metric_x', '?')} vs "
                f"{result.summary.get('pareto_metric_y', '?')})"
            )
            lines.append("-" * 72)
            lines.extend(f"  {run_id}" for run_id in result.pareto_runs)
            lines.append("")

        # Top 5 from ranking by primary metric.
        if not result.ranking.empty:
            lines.append("TOP 5 BY PRIMARY METRIC")
            lines.append(f"({self.primary_metric})")
            lines.append("-" * 72)
            top5 = result.ranking.head(5)
            for _, row in top5.iterrows():
                config = row.get("config_id", "?")
                val = row.get(self.primary_metric, float("nan"))
                run_id = row.get("run_id", "?")
                lines.append(f"  {str(config):<30s}  {val:.4f}  (run: {run_id})")
            lines.append("")

        # Deltas summary (non-champion rows with notable improvements).
        if not result.deltas.empty:
            delta_col = f"delta_{self.primary_metric}"
            if delta_col in result.deltas.columns:
                improved = result.deltas[result.deltas[delta_col] < 0].copy()
                if not improved.empty:
                    improved = improved.sort_values(delta_col)
                    lines.append("RUNS IMPROVING ON CHAMPION (primary metric)")
                    lines.append("-" * 72)
                    for _, row in improved.head(10).iterrows():
                        config = row.get("config_id", row.get("run_id", "?"))
                        delta = row[delta_col]
                        lines.append(f"  {str(config):<30s}  {delta:+.4f} pp")
                    lines.append("")

        lines.append(sep)
        return "\n".join(lines)
