"""
Rolling-origin cross-validation for place share-trending backtests (PP-005 WS-A).

Replaces the static two-window backtest with expanding-window rolling-origin
cross-validation to provide more rigorous variant selection evidence.

Instead of evaluating variants on a single fixed train/test split, this module
generates a sequence of expanding training windows each followed by a fixed-length
test horizon.  Per-window results are aggregated into an overall score that is
more robust to the choice of evaluation period.

Design
------
All per-window computation is delegated to the existing ``place_backtest`` module:
``run_single_variant``, ``compute_per_place_metrics``, ``compute_tier_aggregates``,
and ``compute_variant_score``.  This module adds only the rolling-window orchestration
and cross-window aggregation logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cohort_projections.data.process.place_backtest import (
    compute_per_place_metrics,
    compute_tier_aggregates,
    compute_variant_score,
    run_single_variant,
    select_winner,
)
from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


def generate_rolling_windows(
    history_start: int,
    history_end: int,
    min_train_years: int = 5,
    test_horizon: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Generate expanding train/test windows for rolling-origin cross-validation.

    Windows are constructed so that the training set always begins at
    ``history_start`` and expands forward in ``test_horizon``-year increments.
    Each window is followed by a non-overlapping test period of
    ``test_horizon`` years.

    Args:
        history_start: First available history year (inclusive).
        history_end: Last available history year (inclusive).
        min_train_years: Minimum number of training years required for the
            first window (inclusive count, e.g. 5 means years 0-4).
        test_horizon: Number of test years per window (inclusive count).

    Returns:
        List of ``(train_start, train_end, test_start, test_end)`` tuples.
        Empty list if no valid window can be constructed.

    Raises:
        ValueError: If ``min_train_years`` or ``test_horizon`` is < 1, or
            if ``history_start > history_end``.
    """
    if min_train_years < 1:
        raise ValueError(f"min_train_years must be >= 1, got {min_train_years}")
    if test_horizon < 1:
        raise ValueError(f"test_horizon must be >= 1, got {test_horizon}")
    if history_start > history_end:
        raise ValueError(
            f"history_start ({history_start}) must be <= history_end ({history_end})"
        )

    total_span = history_end - history_start + 1
    if total_span < min_train_years + test_horizon:
        return []

    windows: list[tuple[int, int, int, int]] = []

    # First train_end is (history_start + min_train_years - 1).
    # Then we step forward by test_horizon each iteration.
    train_end = history_start + min_train_years - 1
    while train_end + test_horizon <= history_end:
        test_start = train_end + 1
        test_end = test_start + test_horizon - 1
        windows.append((history_start, train_end, test_start, test_end))
        train_end += test_horizon

    return windows


# ---------------------------------------------------------------------------
# Per-window execution
# ---------------------------------------------------------------------------


def _run_variant_on_window(
    variant_id: str,
    fitting_method: str,
    constraint_method: str,
    window: tuple[int, int, int, int],
    share_history: pd.DataFrame,
    county_pop: pd.DataFrame,
    tier_assignments: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one variant on one window and return score + metrics.

    Returns a dictionary with keys:
        ``window``, ``variant_id``, ``fitting_method``, ``constraint_method``,
        ``score``, ``tier_aggregates``, ``place_metrics``.
    """
    train_start, train_end, test_start, test_end = window
    window_label = f"{train_start}-{train_end}/{test_start}-{test_end}"

    result = run_single_variant(
        variant_id=variant_id,
        fitting_method=fitting_method,
        constraint_method=constraint_method,
        train_years=[train_start, train_end],
        test_years=[test_start, test_end],
        share_history=share_history,
        county_pop=county_pop,
        config=config,
    )

    place_metrics = compute_per_place_metrics(
        projected=result["projected"],
        actual=result["actual"],
    )
    tier_aggregates = compute_tier_aggregates(
        place_metrics=place_metrics,
        tier_assignments=tier_assignments,
    )
    score = compute_variant_score(tier_aggregates)

    logger.info(
        "Rolling window %s, variant %s: score=%.4f",
        window_label,
        variant_id,
        score,
    )

    return {
        "window": window_label,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "variant_id": variant_id,
        "fitting_method": fitting_method,
        "constraint_method": constraint_method,
        "score": score,
        "tier_aggregates": tier_aggregates,
        "place_metrics": place_metrics,
    }


# ---------------------------------------------------------------------------
# Full rolling-origin backtest
# ---------------------------------------------------------------------------


def run_rolling_origin_backtest(
    share_history: pd.DataFrame,
    county_pop: pd.DataFrame,
    tier_assignments: pd.DataFrame,
    variants: Mapping[str, dict[str, str]],
    config: Mapping[str, Any],
    windows: Sequence[tuple[int, int, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Run all variants across all rolling-origin windows.

    Args:
        share_history: Historical place-share rows.
        county_pop: County population in long format with ``county_fips``,
            ``year``, and a population column.
        tier_assignments: Place-tier mapping with ``place_fips``,
            ``confidence_tier``, and optional ``population_2024``.
        variants: Mapping from variant ID to dict with ``fitting_method``
            and ``constraint_method`` keys.
        config: Project configuration dictionary.
        windows: Pre-generated windows (if ``None``, generated from config).

    Returns:
        List of per-window-per-variant result dicts (from
        ``_run_variant_on_window``).
    """
    if windows is None:
        ro_cfg = config.get("place_projections", {}).get("rolling_origin_backtest", {})
        model_cfg = config.get("place_projections", {}).get("model", {})
        history_start = int(model_cfg.get("history_start", 2000))
        history_end = int(model_cfg.get("history_end", 2024))
        min_train = int(ro_cfg.get("min_train_years", 5))
        horizon = int(ro_cfg.get("test_horizon", 5))
        windows = generate_rolling_windows(
            history_start=history_start,
            history_end=history_end,
            min_train_years=min_train,
            test_horizon=horizon,
        )

    if not windows:
        raise ValueError(
            "No valid rolling-origin windows could be generated from the "
            "configured history range and min_train_years / test_horizon."
        )

    logger.info(
        "Rolling-origin backtest: %d windows, %d variants.",
        len(windows),
        len(variants),
    )

    results: list[dict[str, Any]] = []
    for window in windows:
        for variant_id, spec in variants.items():
            result = _run_variant_on_window(
                variant_id=variant_id,
                fitting_method=spec["fitting_method"],
                constraint_method=spec["constraint_method"],
                window=window,
                share_history=share_history,
                county_pop=county_pop,
                tier_assignments=tier_assignments,
                config=config,
            )
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Cross-window aggregation
# ---------------------------------------------------------------------------


def aggregate_rolling_metrics(
    per_window_results: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate per-window scores into one row per variant.

    For each variant the following summary statistics are computed:
    - ``mean_score``: arithmetic mean of per-window scores.
    - ``median_score``: median of per-window scores.
    - ``std_score``: standard deviation (ddof=1, NaN if < 2 windows).
    - ``min_score``: minimum per-window score.
    - ``max_score``: maximum per-window score.
    - ``n_windows``: number of windows evaluated.

    Args:
        per_window_results: Output of ``run_rolling_origin_backtest``.

    Returns:
        DataFrame with one row per variant.
    """
    if not per_window_results:
        raise ValueError("per_window_results is empty; nothing to aggregate.")

    rows = [
        {
            "variant_id": r["variant_id"],
            "fitting_method": r["fitting_method"],
            "constraint_method": r["constraint_method"],
            "window": r["window"],
            "score": r["score"],
        }
        for r in per_window_results
    ]
    scores_df = pd.DataFrame(rows)

    aggregated = (
        scores_df.groupby(
            ["variant_id", "fitting_method", "constraint_method"], as_index=False
        )
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
            min_score=("score", "min"),
            max_score=("score", "max"),
            n_windows=("score", "count"),
        )
        .sort_values("mean_score")
        .reset_index(drop=True)
    )
    return aggregated


def build_per_window_summary(
    per_window_results: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Build a flat summary table with one row per window-variant pair.

    Args:
        per_window_results: Output of ``run_rolling_origin_backtest``.

    Returns:
        DataFrame with columns ``window``, ``train_start``, ``train_end``,
        ``test_start``, ``test_end``, ``variant_id``, ``fitting_method``,
        ``constraint_method``, ``score``.
    """
    if not per_window_results:
        raise ValueError("per_window_results is empty; nothing to summarise.")

    rows = [
        {
            "window": r["window"],
            "train_start": r["train_start"],
            "train_end": r["train_end"],
            "test_start": r["test_start"],
            "test_end": r["test_end"],
            "variant_id": r["variant_id"],
            "fitting_method": r["fitting_method"],
            "constraint_method": r["constraint_method"],
            "score": r["score"],
        }
        for r in per_window_results
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def select_rolling_winner(
    aggregated_scores: pd.DataFrame,
    acceptance_criteria: str = "mean_score",
) -> dict[str, Any]:
    """Select the winning variant from rolling-origin aggregated scores.

    Args:
        aggregated_scores: Output of ``aggregate_rolling_metrics`` with at
            least ``variant_id`` and ``mean_score`` columns.
        acceptance_criteria: One of ``"mean_score"`` or
            ``"median_score"``.  Determines which aggregation column is
            used for ranking.

    Returns:
        Dictionary with ``winner_variant_id``, ``acceptance_criteria``,
        ``winner_score``, and the full ``scores`` table as a list of dicts.

    Raises:
        ValueError: If ``acceptance_criteria`` is not recognised or if the
            scores table is empty.
    """
    valid_criteria = {"mean_score", "median_score"}
    if acceptance_criteria not in valid_criteria:
        raise ValueError(
            f"acceptance_criteria must be one of {sorted(valid_criteria)}, "
            f"got '{acceptance_criteria}'."
        )

    if aggregated_scores.empty:
        raise ValueError("aggregated_scores is empty; cannot select winner.")

    score_col = acceptance_criteria
    if score_col not in aggregated_scores.columns:
        raise ValueError(
            f"aggregated_scores missing '{score_col}' column."
        )

    # Delegate to the existing select_winner with {variant_id: score} mapping
    score_map = dict(
        zip(
            aggregated_scores["variant_id"],
            aggregated_scores[score_col],
            strict=False,
        )
    )
    winner_id = select_winner(score_map)
    winner_row = aggregated_scores[aggregated_scores["variant_id"] == winner_id].iloc[0]

    return {
        "winner_variant_id": winner_id,
        "acceptance_criteria": acceptance_criteria,
        "winner_score": float(winner_row[score_col]),
        "winner_mean_score": float(winner_row["mean_score"]),
        "winner_median_score": float(winner_row["median_score"]),
        "winner_n_windows": int(winner_row["n_windows"]),
        "scores": aggregated_scores.to_dict(orient="records"),
    }


__all__ = [
    "aggregate_rolling_metrics",
    "build_per_window_summary",
    "generate_rolling_windows",
    "run_rolling_origin_backtest",
    "select_rolling_winner",
]
