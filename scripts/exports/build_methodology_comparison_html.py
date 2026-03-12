#!/usr/bin/env python3
"""
Build a standalone HTML rendering of the SDC methodology comparison document.

Reads ``docs/methodology_comparison_sdc_2024.md`` (a detailed comparison of
the ND State Data Center 2024 projections with this project's methodology)
and produces a self-contained HTML file with embedded CSS, a sticky
table-of-contents sidebar, alternating-row tables, and print-friendly styles.

Output
------
    data/exports/nd_methodology_comparison_sdc_{YYYYMMDD}.html

Usage
-----
    python scripts/exports/build_methodology_comparison_html.py

Key dependencies
----------------
    - markdown (pip package, v3.x) — Markdown-to-HTML conversion
    - MathJax 3 CDN — client-side LaTeX math rendering (loaded in output HTML)

Data lineage
------------
    Input:  docs/methodology_comparison_sdc_2024.md
    Output: data/exports/nd_methodology_comparison_sdc_{datestamp}.html

SOP-002 compliance
------------------
    Purpose:       Convert SDC comparison Markdown to distributable HTML
    Author:        Projection team
    Date created:  2026-03-02
    Last modified: 2026-03-02
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _markdown_to_html import build_html_document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
COMPARISON_MD = PROJECT_ROOT / "docs" / "methodology_comparison_sdc_2024.md"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

TODAY = datetime.now(tz=UTC).date()
DATE_STAMP = TODAY.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Read methodology_comparison_sdc_2024.md and write the HTML export."""
    if not COMPARISON_MD.exists():
        print(f"ERROR: Source file not found: {COMPARISON_MD}")
        sys.exit(1)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    md_text = COMPARISON_MD.read_text(encoding="utf-8")

    html = build_html_document(
        title="Methodology Comparison: ND State Data Center 2024 vs. Our Project",
        version="",
        date_str="February 2026",
        md_source=md_text,
        subtitle="Cohort-Component Projection Model — Side-by-Side Analysis",
    )

    out_path = EXPORT_DIR / f"nd_methodology_comparison_sdc_{DATE_STAMP}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
