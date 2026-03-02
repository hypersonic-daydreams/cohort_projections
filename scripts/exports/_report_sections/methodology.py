"""
Methodology section for the interactive HTML report.

Produces:
    1. Scenario descriptions table
    2. Methodology overview (from shared _methodology.py constants)
    3. Data sources table
    4. Caveats and limitations

References:
    ADR-037: CBO-grounded scenario methodology
    ADR-033: Place projection methodology
    _methodology.py: Shared constants
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure we can import sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from _methodology import (
    CONDITIONAL_CAVEAT,
    DATA_AVAILABILITY_NOTE,
    METHODOLOGY_LINES,
    ORGANIZATION_ATTRIBUTION,
    PLACE_METHODOLOGY_LINE,
    SCENARIOS,
)

logger = logging.getLogger(__name__)

BASE_YEAR = 2025
FINAL_YEAR = 2055

# Data sources with vintages
DATA_SOURCES = [
    {
        "source": "Census Population Estimates Program (PEP)",
        "vintage": "2025 Vintage",
        "usage": "Base population, components of change (2000-2025)",
    },
    {
        "source": "CDC/NCHS Vital Statistics",
        "vintage": "2023-2024",
        "usage": "Age-specific fertility rates, life tables",
    },
    {
        "source": "Census Bureau NP2023",
        "vintage": "2023",
        "usage": "Mortality improvement factors, survival projections",
    },
    {
        "source": "CBO Demographic Outlook",
        "vintage": "Pub. 60875 (Jan 2025), Pub. 61879 (Jan 2026)",
        "usage": "Time-varying international migration factors",
    },
    {
        "source": "Census Bureau TIGER/Line",
        "vintage": "2024",
        "usage": "County and place boundaries for geospatial exports",
    },
    {
        "source": "Census ACS / Building Permits",
        "vintage": "2022-2024",
        "usage": "Housing unit data for HU cross-validation",
    },
]


def build_methodology_section(theme: Any) -> str:
    """Build the complete Methodology section as an HTML string.

    Parameters
    ----------
    theme : module
        The _report_theme module.

    Returns
    -------
    str
        HTML fragment for the methodology section.
    """
    parts = ['<h2>Methodology</h2>']

    # --- Scenario Descriptions Table ---
    parts.append('<h3>Scenario Descriptions</h3>')
    scenario_details = [
        {
            "key": "baseline",
            "name": SCENARIOS["baseline"],
            "description": (
                "Continuation of recent demographic trends. Migration rates "
                "based on regime-weighted multi-period averaging (BEBR method) "
                "with convergence interpolation toward long-term rates."
            ),
        },
        {
            "key": "high_growth",
            "name": SCENARIOS["high_growth"],
            "description": (
                "Elevated immigration scenario using BEBR-optimistic migration rates "
                "(most favorable historical period per county, approximately +1,300 "
                "additional net migrants per year vs baseline). Fertility +5%."
            ),
        },
        {
            "key": "restricted_growth",
            "name": SCENARIOS["restricted_growth"],
            "description": (
                "CBO policy-adjusted scenario applying time-varying reduction factors "
                "to international migration only (domestic migration unchanged). "
                "Fertility -5%. Based on CBO Demographic Outlook projections."
            ),
        },
    ]

    rows_html = []
    for sc in scenario_details:
        color = theme.get_scenario_color(sc["key"])
        rows_html.append(f"""
            <tr>
                <td><span class="scenario-dot" style="background: {color};"></span> {sc["name"]}</td>
                <td>{sc["description"]}</td>
            </tr>
        """)

    parts.append(f"""
        <div class="table-wrapper">
            <table class="data-table" id="scenarios-table">
                <thead>
                    <tr>
                        <th style="width: 30%;">Scenario</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    """)

    # --- Methodology Overview ---
    parts.append('<h3>Methodology Overview</h3>')
    parts.append('<div class="methodology-text">')
    parts.append("<ul>")
    for line in METHODOLOGY_LINES:
        formatted = line.format(base_year=BASE_YEAR, final_year=FINAL_YEAR)
        parts.append(f"<li>{formatted}</li>")
    parts.append("</ul>")

    # Place methodology
    parts.append(f"<p><strong>Place Projections:</strong> {PLACE_METHODOLOGY_LINE}</p>")
    parts.append("</div>")

    # --- Data Sources Table ---
    parts.append('<h3>Data Sources</h3>')
    source_rows = []
    for ds in DATA_SOURCES:
        source_rows.append(f"""
            <tr>
                <td>{ds["source"]}</td>
                <td>{ds["vintage"]}</td>
                <td>{ds["usage"]}</td>
            </tr>
        """)

    parts.append(f"""
        <div class="table-wrapper">
            <table class="data-table" id="data-sources-table">
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Vintage</th>
                        <th>Usage</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(source_rows)}
                </tbody>
            </table>
        </div>
    """)

    # --- Caveats ---
    parts.append('<h3>Caveats and Limitations</h3>')
    parts.append('<div class="methodology-text caveats">')
    parts.append(f"<p><strong>{CONDITIONAL_CAVEAT}</strong></p>")
    parts.append("<ul>")
    caveats = [
        (
            "Population projections are inherently uncertain and become less reliable "
            "over longer time horizons. The 30-year projection period (2025-2055) "
            "should be interpreted with appropriate caution."
        ),
        (
            "Scenario boundaries (baseline, high growth, restricted growth) are not "
            "statistical confidence intervals. They represent plausible alternative "
            "futures under different assumption sets."
        ),
        (
            "Small-area projections (places with populations under 2,500) carry higher "
            "uncertainty than larger geographies. Confidence tiers reflect this: "
            "HIGH tier places have the most reliable projections."
        ),
        (
            "Migration assumptions have the largest impact on projection outcomes. "
            "Unexpected economic developments (e.g., energy sector changes) could "
            "significantly alter actual population trajectories."
        ),
        (
            "Race/ethnicity projections are constrained by data availability and "
            "methodological limitations in small populations."
        ),
        f"{DATA_AVAILABILITY_NOTE}",
    ]
    for caveat in caveats:
        parts.append(f"<li>{caveat}</li>")
    parts.append("</ul>")
    parts.append(f"<p class='attribution'>{ORGANIZATION_ATTRIBUTION}</p>")
    parts.append("</div>")

    return "\n".join(parts)
