"""Automated experiment recommendation based on completed benchmark results.

Analyzes the history of benchmark runs and suggests parameter values to test
next.  Heuristics include boundary detection, untested-catalog surfacing,
interaction detection, persistent-weakness identification, and diminishing-
returns flagging.

Dependencies (forward-declared; modules may not exist yet):
    - ``ResultsStore``  — loads scorecards and manifests from benchmark history
    - ``ObservatoryComparator`` — computes deltas between runs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cohort_projections.analysis.observatory.comparator import ObservatoryComparator
from cohort_projections.analysis.observatory.results_store import ResultsStore
from cohort_projections.analysis.observatory.variant_catalog import VariantCatalog

# ---------------------------------------------------------------------------
# MethodConfig parameters that can be injected at runtime (config-only).
# Parameters NOT in this set require upstream code changes.
# ---------------------------------------------------------------------------
CONFIG_ONLY_PARAMS: frozenset[str] = frozenset(
    {
        "bakken_fips",
        "boom_period_dampening",
        "boom_male_dampening",
        "college_blend_factor",
        "sdc_bakken_dampening",
        "convergence_recent_hold",
        "convergence_medium_hold",
        "convergence_transition_hold",
        "gq_correction_fraction",
        "rate_cap_general",
    }
)

# Scorecard metrics used for sensitivity analysis (lower is better).
TRADEOFF_METRICS: list[str] = [
    "county_mape_overall",
    "county_mape_urban_college",
    "county_mape_rural",
    "county_mape_bakken",
    "state_ape_recent_short",
    "state_ape_recent_medium",
]

# County-group columns in the scorecard.
COUNTY_GROUP_METRICS: list[str] = [
    "county_mape_overall",
    "county_mape_urban_college",
    "county_mape_rural",
    "county_mape_bakken",
]


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    """A single experiment recommendation."""

    parameter: str
    suggested_value: Any
    direction: str  # "increase", "decrease", "explore"
    rationale: str
    expected_impact: str  # e.g. "~0.1pp county_mape_urban_college improvement"
    priority: int  # 1 = highest
    requires_code_change: bool  # True if not injectable via MethodConfig
    grid_suggestion: dict | None = None  # Optional grid spec for a sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric(value: Any) -> bool:
    """Return True if *value* is a scalar numeric (int/float, not bool)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and math.isfinite(value)


def _numeric_values_sorted(values: list[Any], *, unique: bool = False) -> list[float]:
    """Filter to finite numerics and return sorted.

    Parameters
    ----------
    unique : bool
        If True, deduplicate values before sorting so each distinct numeric
        appears only once.
    """
    nums = sorted(float(v) for v in values if _is_numeric(v))
    if unique:
        nums = sorted(set(nums))
    return nums


def _linear_trend(xs: list[float], ys: list[float]) -> float | None:
    """Return the slope of a simple OLS fit, or None if too few points."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0:
        return None
    return ss_xy / ss_xx


def _next_step(values: list[float], direction: str) -> float:
    """Suggest the next value to test given a sorted list and direction."""
    if not values:
        return 0.0
    step = values[-1] - values[0]
    if len(values) >= 2:
        step = values[-1] - values[-2]
    if direction == "increase":
        return round(values[-1] + step, 6)
    return round(values[0] - step, 6)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------
class ObservatoryRecommender:
    """Analyze completed experiment results and suggest what to test next.

    Parameters
    ----------
    store : ResultsStore
        Provides access to benchmark run scorecards and manifests.
    comparator : ObservatoryComparator
        Computes deltas between any two benchmark results.
    variant_catalog : Any, optional
        A list of dicts (or DataFrame) representing the experiment catalog.
        Each entry should have at minimum ``slug``, ``parameter``, ``status``
        (one of ``passed_all_gates``, ``needs_human_review``, ``untested``,
        etc.), and ``tier`` (1/2/3).
    config : dict, optional
        Overrides for internal thresholds:
        - ``plateau_threshold`` (float): slope magnitude below which a
          parameter is considered converged.  Default ``0.02``.
        - ``max_regression_hard`` (dict[str, float]): per-metric ceilings
          taken from the evaluation policy.
    """

    def __init__(
        self,
        store: ResultsStore,
        comparator: ObservatoryComparator,
        variant_catalog: Any | None = None,
        config: dict | None = None,
        bounds_catalog: VariantCatalog | None = None,
    ) -> None:
        self.store = store
        self.comparator = comparator
        self.variant_catalog = variant_catalog or []
        self._config = config or {}
        self._plateau_threshold: float = self._config.get("plateau_threshold", 0.02)
        self._bounds_catalog: VariantCatalog | None = bounds_catalog

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_next_experiments(self, n: int = 5) -> list[Recommendation]:
        """Return up to *n* prioritised experiment recommendations.

        The recommendations are produced by four independent heuristics and
        then merged, deduplicated by parameter, and sorted by priority.

        Heuristics
        ----------
        1. Boundary detection — extend the tested range of a parameter whose
           best value sits at a boundary.
        2. Untested catalog entries — surface experiments from the catalog
           that have not been run yet.
        3. Interaction detection — suggest combining two independently
           beneficial parameters.
        4. Diminishing returns — deprioritise parameters whose metric
           sensitivity has plateaued.
        """
        recs: list[Recommendation] = []

        sensitivity = self.parameter_sensitivity_summary()
        if not sensitivity.empty:
            recs.extend(self._boundary_recommendations(sensitivity))
            recs.extend(self._diminishing_returns_flags(sensitivity))

        recs.extend(self._untested_catalog_recommendations())
        recs.extend(self._interaction_recommendations(sensitivity))

        # Deduplicate by (parameter, suggested_value), keeping highest priority
        seen: dict[tuple[str, str], Recommendation] = {}
        for rec in recs:
            key = (rec.parameter, str(rec.suggested_value))
            if key not in seen or rec.priority < seen[key].priority:
                seen[key] = rec
        unique = sorted(seen.values(), key=lambda r: r.priority)
        return unique[:n]

    def identify_persistent_weaknesses(self) -> pd.DataFrame:
        """County groups / horizons where no tested variant improves over champion.

        Returns a DataFrame with columns ``[metric, champion_value,
        best_challenger_delta, best_challenger_run]``.  Rows where
        ``best_challenger_delta >= 0`` represent persistent weaknesses.
        """
        empty = pd.DataFrame(
            columns=[
                "metric",
                "champion_value",
                "best_challenger_delta",
                "best_challenger_run",
            ]
        )

        group_impact = self.comparator.county_group_impact()
        if group_impact.empty:
            return empty

        scorecards = self.store.get_consolidated_scorecards()
        if scorecards.empty:
            return empty

        # Identify the champion row (first row from the first run, status=champion)
        champion_rows = scorecards[
            scorecards["status_at_run"].str.lower() == "champion"
        ]
        if champion_rows.empty:
            champion_rows = scorecards.head(1)

        records: list[dict[str, Any]] = []
        for metric in COUNTY_GROUP_METRICS:
            champion_value: float | None = None
            if metric in champion_rows.columns:
                champion_value = float(champion_rows.iloc[0][metric])

            # Look for the delta column in group_impact (e.g. delta_rural)
            # The comparator uses _GROUP_TO_SCORECARD_COL mapping which
            # produces columns like delta_overall, delta_urban_college, etc.
            group_suffix = metric.replace("county_mape_", "")
            delta_col = f"delta_{group_suffix}"

            best_delta: float | None = None
            best_run: str = ""

            if delta_col in group_impact.columns:
                valid = group_impact.dropna(subset=[delta_col])
                if not valid.empty:
                    best_idx = valid[delta_col].idxmin()
                    best_delta = float(valid.loc[best_idx, delta_col])
                    if "run_id" in valid.columns:
                        best_run = str(valid.loc[best_idx, "run_id"])

            records.append(
                {
                    "metric": metric,
                    "champion_value": champion_value,
                    "best_challenger_delta": best_delta,
                    "best_challenger_run": best_run,
                }
            )

        return pd.DataFrame(records)

    def parameter_sensitivity_summary(self) -> pd.DataFrame:
        """For each tested parameter, show metric sensitivity.

        Returns a DataFrame with columns ``[parameter, value, <metric>_delta,
        …, classification, run_id]`` — one row per (parameter, value) tested.
        """
        param_rows = self._collect_param_value_rows()
        if not param_rows:
            return pd.DataFrame()
        return pd.DataFrame(param_rows)

    def format_recommendations(self, recommendations: list[Recommendation]) -> str:
        """Render a human-readable recommendation report."""
        if not recommendations:
            return "No recommendations — insufficient experiment data.\n"

        lines: list[str] = [
            "=" * 72,
            "  EXPERIMENT RECOMMENDATIONS",
            "=" * 72,
            "",
        ]
        for i, rec in enumerate(recommendations, 1):
            code_flag = " [CODE CHANGE REQUIRED]" if rec.requires_code_change else ""
            lines.append(f"  #{i}  {rec.parameter} -> {rec.suggested_value}"
                         f"  ({rec.direction}){code_flag}")
            lines.append(f"      Priority: {rec.priority}")
            lines.append(f"      Rationale: {rec.rationale}")
            lines.append(f"      Expected impact: {rec.expected_impact}")
            if rec.grid_suggestion:
                lines.append(f"      Grid suggestion: {rec.grid_suggestion}")
            lines.append("")

        # Persistent weaknesses summary
        weaknesses = self.identify_persistent_weaknesses()
        persistent = weaknesses[
            weaknesses["best_challenger_delta"].notna()
            & (weaknesses["best_challenger_delta"] >= 0)
        ]
        if not persistent.empty:
            lines.append("-" * 72)
            lines.append("  PERSISTENT WEAKNESSES (no variant improves over champion)")
            lines.append("-" * 72)
            for _, row in persistent.iterrows():
                lines.append(
                    f"  {row['metric']:36s}  champion={row['champion_value']:.4f}"
                    f"  best_delta={row['best_challenger_delta']:+.4f}"
                )
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal heuristics
    # ------------------------------------------------------------------

    def _collect_param_value_rows(self) -> list[dict[str, Any]]:
        """Walk all runs and extract per-parameter deltas.

        For each run, diff the resolved configs of champion vs challenger
        to identify which parameters changed, then pair those changes with
        the scorecard metric deltas from the comparator.

        Returns a list of dicts suitable for building the sensitivity
        DataFrame.
        """
        run_ids = self.store.get_run_ids()
        rows: list[dict[str, Any]] = []

        # Pre-compute deltas across all runs
        deltas_df = self.comparator.compute_deltas()
        if deltas_df.empty:
            return rows

        for run_id in run_ids:
            try:
                manifest = self.store.get_run_manifest(run_id)
            except FileNotFoundError:
                continue

            run_configs = self.store.get_run_config(run_id)
            if not run_configs:
                continue

            champion_method_id = manifest.get("champion_method_id", "")
            champion_config = run_configs.get(champion_method_id)

            # Deltas for this run
            run_deltas = deltas_df[deltas_df["run_id"] == run_id]

            for method_id, method_config in run_configs.items():
                if method_id == champion_method_id:
                    continue
                if champion_config is None:
                    continue

                changed = self._diff_configs(champion_config, method_config)
                if not changed:
                    continue

                # Get the delta row for this method from the comparator
                method_delta_rows = run_deltas[
                    run_deltas["method_id"] == method_id
                ]
                delta_dict: dict[str, float] = {}
                if not method_delta_rows.empty:
                    delta_row = method_delta_rows.iloc[0]
                    for metric in TRADEOFF_METRICS:
                        delta_col = f"delta_{metric}"
                        if delta_col in delta_row.index:
                            val = delta_row[delta_col]
                            if pd.notna(val):
                                delta_dict[metric] = float(val)

                # Build one sensitivity row per changed parameter
                for param, value in changed.items():
                    row: dict[str, Any] = {
                        "parameter": param,
                        "value": value,
                        "champion_value": champion_config.get(param),
                        "classification": "unknown",
                        "run_id": run_id,
                    }
                    for metric in TRADEOFF_METRICS:
                        row[f"{metric}_delta"] = delta_dict.get(metric)
                    rows.append(row)

        return rows

    @staticmethod
    def _resolved_config_for(
        manifest: dict[str, Any], method_id: str
    ) -> dict[str, Any] | None:
        """Extract resolved_config for *method_id* from a manifest."""
        for method in manifest.get("methods", []):
            if method.get("method_id") == method_id:
                return method.get("resolved_config")
        return None

    @staticmethod
    def _diff_configs(
        base: dict[str, Any], variant: dict[str, Any]
    ) -> dict[str, Any]:
        """Return keys from *variant* that differ from *base*.

        Only top-level scalar differences are detected; nested structures
        (dicts, lists) use equality comparison.
        """
        changed: dict[str, Any] = {}
        all_keys = set(base) | set(variant)
        for key in all_keys:
            base_val = base.get(key)
            variant_val = variant.get(key)
            if base_val != variant_val:
                changed[key] = variant_val
        return changed

    # ------------------------------------------------------------------
    # Heuristic 1 — boundary detection
    # ------------------------------------------------------------------

    def _boundary_recommendations(
        self, sensitivity: pd.DataFrame
    ) -> list[Recommendation]:
        """For each numeric parameter, check if the best value is at a boundary."""
        recs: list[Recommendation] = []
        if sensitivity.empty:
            return recs

        for param, group in sensitivity.groupby("parameter"):
            param = str(param)
            all_values = list(group["value"])
            numeric_vals = _numeric_values_sorted(all_values, unique=True)
            if len(numeric_vals) < 2:
                continue

            # Find the value with the best (most negative) overall MAPE delta
            metric_col = "county_mape_overall_delta"
            if metric_col not in group.columns:
                continue

            valid = group.dropna(subset=[metric_col])
            if valid.empty:
                continue

            best_idx = valid[metric_col].idxmin()
            best_row = valid.loc[best_idx]
            best_value = best_row["value"]

            if not _is_numeric(best_value):
                continue

            best_f = float(best_value)
            is_at_max = best_f == numeric_vals[-1]
            is_at_min = best_f == numeric_vals[0]

            if is_at_max:
                direction = "increase"
                suggested = _next_step(numeric_vals, direction)
                step = numeric_vals[-1] - numeric_vals[-2]
                grid = {
                    param: [
                        round(numeric_vals[-1] + step * i, 6)
                        for i in range(1, 3)
                    ]
                }
            elif is_at_min:
                direction = "decrease"
                suggested = _next_step(numeric_vals, direction)
                step = numeric_vals[1] - numeric_vals[0]
                grid = {
                    param: [
                        round(numeric_vals[0] - step * i, 6)
                        for i in range(1, 3)
                    ]
                }
            else:
                # Best is interior — no boundary recommendation
                continue

            # --- Apply parameter bounds clamping ---
            clamped = False
            if self._bounds_catalog is not None:
                bounds = self._bounds_catalog.get_bounds(param)
                if bounds is not None:
                    original_suggested = suggested
                    suggested = self._bounds_catalog.clamp_value(
                        param, suggested
                    )
                    if suggested != original_suggested:
                        clamped = True

                    # Clamp grid values too
                    clamped_grid_vals: list[float] = []
                    for gv in grid[param]:
                        clamped_gv = self._bounds_catalog.clamp_value(param, gv)
                        if clamped_gv not in clamped_grid_vals:
                            clamped_grid_vals.append(clamped_gv)
                    grid[param] = clamped_grid_vals

            # Skip if clamped value was already tested
            if clamped and suggested in numeric_vals:
                continue

            best_delta = float(best_row[metric_col])
            clamp_note = ""
            if clamped:
                clamp_note = (
                    f" (clamped to "
                    f"{'max' if is_at_max else 'min'} "
                    f"{suggested})"
                )
            recs.append(
                Recommendation(
                    parameter=param,
                    suggested_value=suggested,
                    direction=direction,
                    rationale=(
                        f"Best tested value ({best_f}) is at the "
                        f"{'upper' if is_at_max else 'lower'} boundary of "
                        f"tested range {numeric_vals}. "
                        f"Best overall MAPE delta was {best_delta:+.4f}pp."
                        f"{clamp_note}"
                    ),
                    expected_impact=(
                        f"~{abs(best_delta):.2f}pp county_mape_overall "
                        f"improvement if trend continues"
                    ),
                    priority=2,
                    requires_code_change=param not in CONFIG_ONLY_PARAMS,
                    grid_suggestion=grid,
                )
            )
        return recs

    # ------------------------------------------------------------------
    # Heuristic 2 — untested catalog entries
    # ------------------------------------------------------------------

    def _untested_catalog_recommendations(self) -> list[Recommendation]:
        """Surface experiments from the catalog that have not been run."""
        recs: list[Recommendation] = []
        catalog = self._normalize_catalog()
        if not catalog:
            return recs

        for entry in catalog:
            status = str(entry.get("status", "untested")).lower()
            if status not in ("untested", ""):
                continue
            tier = int(entry.get("tier", 3))
            param = str(entry.get("parameter", "unknown"))
            slug = str(entry.get("slug", ""))
            hypothesis = str(entry.get("hypothesis", ""))
            value = entry.get("suggested_value", entry.get("value"))
            requires_code = entry.get("requires_code_change", False)
            if not isinstance(requires_code, bool):
                requires_code = param not in CONFIG_ONLY_PARAMS

            priority = tier  # Tier 1 → priority 1, etc.
            recs.append(
                Recommendation(
                    parameter=param,
                    suggested_value=value,
                    direction="explore",
                    rationale=f"Untested catalog entry '{slug}'. {hypothesis}",
                    expected_impact=str(
                        entry.get("expected_improvement", "unknown")
                    ),
                    priority=priority,
                    requires_code_change=requires_code,
                    grid_suggestion=None,
                )
            )
        return recs

    def _normalize_catalog(self) -> list[dict[str, Any]]:
        """Coerce variant_catalog into a list of dicts."""
        if isinstance(self.variant_catalog, pd.DataFrame):
            return self.variant_catalog.to_dict("records")  # type: ignore[union-attr]
        if isinstance(self.variant_catalog, list):
            return self.variant_catalog
        return []

    # ------------------------------------------------------------------
    # Heuristic 3 — interaction detection
    # ------------------------------------------------------------------

    def _interaction_recommendations(
        self, sensitivity: pd.DataFrame
    ) -> list[Recommendation]:
        """If two parameters independently improve metrics, suggest testing together."""
        recs: list[Recommendation] = []
        if sensitivity.empty:
            return recs

        metric_col = "county_mape_overall_delta"
        if metric_col not in sensitivity.columns:
            return recs

        # Find parameters with at least one improving value
        improving_params: dict[str, dict[str, Any]] = {}
        for param, group in sensitivity.groupby("parameter"):
            valid = group.dropna(subset=[metric_col])
            best_delta = valid[metric_col].min() if not valid.empty else None
            if best_delta is not None and best_delta < 0:
                best_idx = valid[metric_col].idxmin()
                best_row = valid.loc[best_idx]
                improving_params[str(param)] = {
                    "best_value": best_row["value"],
                    "best_delta": float(best_delta),
                }

        # Filter to config-only parameters — interactions involving params
        # that require code changes are not actionable via sweep.
        improving_params = {
            p: v for p, v in improving_params.items() if p in CONFIG_ONLY_PARAMS
        }

        # Generate pairwise interaction suggestions
        param_list = sorted(improving_params.keys())
        for i, p1 in enumerate(param_list):
            for p2 in param_list[i + 1 :]:
                info1 = improving_params[p1]
                info2 = improving_params[p2]
                combined_delta = info1["best_delta"] + info2["best_delta"]
                # Both p1 and p2 are guaranteed config-only after filtering above
                requires_code = False
                recs.append(
                    Recommendation(
                        parameter=f"{p1} + {p2}",
                        suggested_value={
                            p1: info1["best_value"],
                            p2: info2["best_value"],
                        },
                        direction="explore",
                        rationale=(
                            f"Both '{p1}' ({info1['best_delta']:+.4f}pp) and "
                            f"'{p2}' ({info2['best_delta']:+.4f}pp) independently "
                            f"improve county_mape_overall. Testing together may "
                            f"reveal synergies or interference."
                        ),
                        expected_impact=(
                            f"Additive estimate: ~{abs(combined_delta):.2f}pp "
                            f"county_mape_overall improvement"
                        ),
                        priority=3,
                        requires_code_change=requires_code,
                        grid_suggestion={
                            p1: [info1["best_value"]],
                            p2: [info2["best_value"]],
                        },
                    )
                )
        return recs

    # ------------------------------------------------------------------
    # Heuristic 5 — diminishing returns
    # ------------------------------------------------------------------

    def _diminishing_returns_flags(
        self, sensitivity: pd.DataFrame
    ) -> list[Recommendation]:
        """Detect parameters whose metric sensitivity has plateaued.

        When the slope of (parameter value -> metric delta) is below
        ``plateau_threshold``, the parameter is deprioritised and a note
        is returned as a low-priority recommendation.
        """
        recs: list[Recommendation] = []
        if sensitivity.empty:
            return recs

        metric_col = "county_mape_overall_delta"
        if metric_col not in sensitivity.columns:
            return recs

        for param, group in sensitivity.groupby("parameter"):
            param = str(param)
            valid = group.dropna(subset=[metric_col])
            numeric_rows = valid[valid["value"].apply(_is_numeric)]
            if numeric_rows.empty:
                continue

            # Deduplicate: average metric deltas across runs for each unique value
            deduped = (
                numeric_rows.assign(value_f=numeric_rows["value"].apply(float))
                .groupby("value_f")[metric_col]
                .mean()
                .reset_index()
            )
            if len(deduped) < 3:
                continue

            xs_sorted = sorted(deduped["value_f"].tolist())
            ys_sorted = [
                float(deduped.loc[deduped["value_f"] == x, metric_col].iloc[0])
                for x in xs_sorted
            ]

            slope = _linear_trend(xs_sorted, ys_sorted)
            if slope is None:
                continue

            if abs(slope) < self._plateau_threshold:
                recs.append(
                    Recommendation(
                        parameter=param,
                        suggested_value=None,
                        direction="plateau",
                        rationale=(
                            f"Sensitivity of county_mape_overall to '{param}' "
                            f"has plateaued (slope={slope:+.4f}). Further "
                            f"sweeps of this parameter are unlikely to yield "
                            f"meaningful improvement."
                        ),
                        expected_impact="Diminishing returns — deprioritise",
                        priority=9,
                        requires_code_change=param not in CONFIG_ONLY_PARAMS,
                        grid_suggestion=None,
                    )
                )
        return recs
