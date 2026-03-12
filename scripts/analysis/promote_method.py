#!/usr/bin/env python3
"""Promote a benchmarked method by updating an alias pointer."""

from __future__ import annotations

import argparse
from pathlib import Path

from cohort_projections.analysis.benchmarking import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_PROFILE_DIR,
    DEFAULT_PROMOTION_HISTORY,
    PROJECT_ROOT,
    append_promotion_history,
    decision_file_is_approved,
    load_method_profile,
    update_alias_mapping,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a method alias after approved review.")
    parser.add_argument("--scope", default="county", help="Promotion scope; stored for operator context.")
    parser.add_argument("--alias", required=True, help="Alias to update.")
    parser.add_argument("--method-id", required=True, help="Target immutable method ID.")
    parser.add_argument("--config-id", required=True, help="Target immutable config ID.")
    parser.add_argument("--decision-id", required=True, help="Approved decision record ID.")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="Directory containing immutable method profiles.",
    )
    parser.add_argument(
        "--alias-path",
        type=Path,
        default=DEFAULT_ALIAS_PATH,
        help="Alias mapping YAML path.",
    )
    parser.add_argument(
        "--decision-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "reviews" / "benchmark_decisions",
        help="Directory containing approved decision records.",
    )
    parser.add_argument(
        "--promotion-history",
        type=Path,
        default=DEFAULT_PROMOTION_HISTORY,
        help="CSV file for alias promotion audit history.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_method_profile(args.method_id, args.config_id, profile_dir=args.profile_dir)

    decision_path = args.decision_dir / f"{args.decision_id}.md"
    if not decision_file_is_approved(decision_path):
        raise ValueError(
            f"Decision record must exist and contain '| Status | Approved |': {decision_path}"
        )

    prior_mapping, new_mapping = update_alias_mapping(
        alias_name=args.alias,
        method_id=args.method_id,
        config_id=args.config_id,
        alias_path=args.alias_path,
    )
    append_promotion_history(
        history_path=args.promotion_history,
        alias_name=args.alias,
        prior_mapping=prior_mapping,
        new_mapping=new_mapping,
        decision_id=args.decision_id,
    )
    print(f"Updated {args.alias} -> {args.method_id} / {args.config_id}")


if __name__ == "__main__":
    main()
