"""Analysis package: benchmarking helpers and evaluation framework."""

from .benchmarking import (
    BENCHMARK_CONTRACT_VERSION,
    append_benchmark_index,
    append_promotion_history,
    build_comparison_to_champion,
    build_run_id,
    build_summary_scorecard,
    compute_prediction_intervals_generic,
    decision_file_is_approved,
    load_aliases,
    load_method_profile,
    render_benchmark_decision_record,
    update_alias_mapping,
    with_county_categories,
    write_manifest,
)

__all__ = [
    "BENCHMARK_CONTRACT_VERSION",
    "append_benchmark_index",
    "append_promotion_history",
    "build_comparison_to_champion",
    "build_run_id",
    "build_summary_scorecard",
    "compute_prediction_intervals_generic",
    "decision_file_is_approved",
    "load_aliases",
    "load_method_profile",
    "render_benchmark_decision_record",
    "update_alias_mapping",
    "with_county_categories",
    "write_manifest",
]
