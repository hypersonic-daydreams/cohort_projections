"""
Shared constants for ND population projection export scripts.

Centralises methodology text, scenario definitions, and labels so that
both ``build_detail_workbooks.py`` and ``build_provisional_workbook.py``
stay in sync.  Runtime values (TODAY, BASE_YEAR, FINAL_YEAR) are left as
format placeholders where needed — importing scripts call ``.format()``
or use the constants directly.

Sources and traceability:
    ADR-004: Core projection engine architecture
    ADR-007: Race/ethnicity categorization (6-category system)
    ADR-035: Census PEP migration data source
    ADR-036: Migration averaging methodology (BEBR multi-period, convergence)
    ADR-037: CBO-grounded scenario methodology (amended by ADR-039, ADR-040)
    ADR-039: International-only migration factor
    ADR-040: Bakken boom dampening extension (2015-2020)
    ADR-041: Census+PUMS hybrid base population distribution
"""

# ---------------------------------------------------------------------------
# Provisional label
# ---------------------------------------------------------------------------
PROVISIONAL_LABEL = "PROVISIONAL \u2014 Pending Review \u2014 Subject to Change"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Full names (for sheet titles, TOC descriptions, methodology text)
# Scenario names from ADR-037 (CBO-grounded methodology)
SCENARIOS = {
    "baseline": "Baseline (Trend Continuation)",
    "restricted_growth": "Restricted Growth (CBO Policy-Adjusted)",
    "high_growth": "High Growth (Pre-Policy Elevated Immigration)",
}

# Short names (for tab / sheet-name labels where length matters)
SCENARIO_SHORT_NAMES = {
    "baseline": "Baseline",
    "restricted_growth": "Restricted Growth",
    "high_growth": "High Growth",
}

# ---------------------------------------------------------------------------
# Canonical methodology lines
# ---------------------------------------------------------------------------
# Lines that contain {base_year} or {final_year} must be .format()-ed by the
# caller with the appropriate values.
METHODOLOGY_LINES = [
    # ADR-004: Cohort-component engine
    "Model: Cohort-component population projection (single-year age, sex, race/ethnicity).",
    # ADR-041: Census+PUMS hybrid; V2025 vintage
    "Base year: {base_year} (Census Population Estimates Program, 2025 vintage).",
    # Config: projection_horizon = 30
    "Horizon: 30 years ({base_year}\u2013{final_year}), annual steps.",
    # ADR-001: Fertility rate processing; CDC/NCHS source
    (
        "Fertility: CDC/NCHS age-specific fertility rates "
        "(2024 for major groups, 2022 national rates for AIAN/Asian), held constant."
    ),
    # ADR-002: Survival rate processing; mortality improvement from NP2023
    (
        "Mortality: CDC/NCHS life tables (2023) with time-varying improvement "
        "from Census Bureau NP2023 survival projections, adjusted for North Dakota."
    ),
    # ADR-035, ADR-036, ADR-040: PEP components, BEBR averaging, boom dampening
    (
        "Migration: Census PEP components of change (2000\u20132025), regime-weighted "
        "multi-period averaging (BEBR method), Rogers-Castro age allocation, "
        "convergence interpolation toward long-term rates."
    ),
    # ADR-037, ADR-039: CBO-grounded scenarios; intl-only factor
    (
        "Baseline: Recent trend continuation. "
        "Restricted Growth: CBO time-varying factor on international migration only "
        "(domestic migration unchanged), \u22125% fertility. "
        "High Growth: +15% migration, +5% fertility. "
        "CBO Demographic Outlook (Pub. 60875, Jan 2025; Pub. 61879, Jan 2026)."
    ),
    "Geography: All 53 North Dakota counties; state totals are county sums.",
]

# ---------------------------------------------------------------------------
# Additional footer lines
# ---------------------------------------------------------------------------
ORGANIZATION_ATTRIBUTION = "Produced by the North Dakota State Data Center."

CONDITIONAL_CAVEAT = (
    "These projections are conditional on stated assumptions "
    "and should not be interpreted as forecasts."
)

DATA_AVAILABILITY_NOTE = (
    "Detailed age-sex-race data available in parquet format for advanced analysis."
)
