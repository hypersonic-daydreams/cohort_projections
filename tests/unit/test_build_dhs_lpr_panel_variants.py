"""
Unit tests for DHS LPR panel variant builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def test_build_lpr_panel_variants_filters_and_balances(tmp_path: Path):
    scripts_dir = (
        Path(__file__).parent.parent.parent
        / "sdc_2024_replication"
        / "data_immigration_policy"
        / "scripts"
    )
    sys.path.insert(0, str(scripts_dir))
    try:
        from build_dhs_lpr_panel_variants import (  # noqa: E402
            build_lpr_panel_variants,
            build_state_fips_mapping,
        )

        components = pd.DataFrame(
            {
                "state": ["United States", "Alabama", "North Dakota", "Puerto Rico"],
                "state_fips": [0, 1, 38, 72],
            }
        )
        components_path = tmp_path / "combined_components_of_change.csv"
        components.to_csv(components_path, index=False)

        state_fips_map = build_state_fips_mapping(components_path)
        assert set(state_fips_map["state"]) == {"Alabama", "North Dakota"}

        lpr = pd.DataFrame(
            {
                "state_or_territory": [
                    "Alabama",
                    "North Dakota",
                    "Puerto Rico",
                    "   Total",
                    "Alabama",
                ],
                "fiscal_year": [2000, 2000, 2000, 2000, 2001],
                "lpr_count": [10, 5, 99, 999, 11],
            }
        )

        variants = build_lpr_panel_variants(lpr, state_fips_map)

        states_only = variants["dhs_lpr_by_state_time_states_only"]
        assert set(states_only["state"]) == {"Alabama", "North Dakota"}
        assert set(states_only["state_fips"]) == {1, 38}

        balanced = variants["dhs_lpr_by_state_time_states_only_balanced"]
        assert set(balanced["fiscal_year"].tolist()) == {2000}

        us_total = variants["dhs_lpr_us_total_time"]
        totals = dict(zip(us_total["fiscal_year"], us_total["us_total_lpr_count"], strict=True))
        assert totals[2000] == 15
        assert totals[2001] == 11

        nd_share = variants["dhs_lpr_nd_share_time"]
        share_2000 = nd_share.loc[nd_share["fiscal_year"] == 2000, "nd_share_pct"].iloc[0]
        assert round(float(share_2000), 4) == round(5 / 15 * 100, 4)
    finally:
        sys.path.remove(str(scripts_dir))
