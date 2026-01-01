"""
Multi-State Placebo Analysis Module
====================================

This module tests whether North Dakota's vintage transition patterns
are unusual compared to other states, supporting the "real driver"
vs. "methodology artifact" interpretation per ADR-020.

Components:
- data_loader: Load 50-state panel data with vintage labels
- regime_shift_calculator: Compute shift statistics per state
- oil_state_hypothesis: Test energy state grouping

Key Question:
"If everyone jumps similarly, that screams 'methodology.'
 If ND (and oil-adjacent states) are outliers, that supports
 a real driver story."

Usage:
    from module_B2_multistate_placebo import (
        load_state_panel,
        calculate_all_state_shifts,
        test_oil_state_hypothesis,
        rank_states_by_shift,
    )
"""

from .data_loader import load_state_panel, add_vintage_labels
from .regime_shift_calculator import (
    calculate_state_shift,
    calculate_all_state_shifts,
    rank_states_by_shift,
    get_nd_percentile,
)
from .oil_state_hypothesis import (
    # Legacy oil state classification
    OIL_STATES,
    SECONDARY_OIL_STATES,
    ALL_OIL_STATES,
    test_oil_state_hypothesis,
    compare_oil_vs_non_oil,
    get_nd_rank_among_oil_states,
    # Boom-timing classification (new)
    BAKKEN_BOOM_STATES,
    PERMIAN_BOOM_STATES,
    OTHER_SHALE_STATES,
    MATURE_OIL_STATES,
    ALL_BOOM_STATES,
    get_boom_category,
    test_boom_state_hypothesis,
    test_bakken_specific_hypothesis,
    get_nd_rank_among_boom_states,
    compare_boom_categories,
)

__all__ = [
    # Data loading
    "load_state_panel",
    "add_vintage_labels",
    # Regime shift calculation
    "calculate_state_shift",
    "calculate_all_state_shifts",
    "rank_states_by_shift",
    "get_nd_percentile",
    # Legacy oil state hypothesis
    "OIL_STATES",
    "SECONDARY_OIL_STATES",
    "ALL_OIL_STATES",
    "test_oil_state_hypothesis",
    "compare_oil_vs_non_oil",
    "get_nd_rank_among_oil_states",
    # Boom-timing hypothesis (new)
    "BAKKEN_BOOM_STATES",
    "PERMIAN_BOOM_STATES",
    "OTHER_SHALE_STATES",
    "MATURE_OIL_STATES",
    "ALL_BOOM_STATES",
    "get_boom_category",
    "test_boom_state_hypothesis",
    "test_bakken_specific_hypothesis",
    "get_nd_rank_among_boom_states",
    "compare_boom_categories",
]
