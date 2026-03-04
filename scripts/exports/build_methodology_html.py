#!/usr/bin/env python3
"""
Build a standalone HTML rendering of the full methodology document.

Reads ``docs/methodology.md`` (the comprehensive methodology for the North
Dakota Cohort-Component Population Projection Model) and produces a
self-contained HTML file with embedded CSS, MathJax-rendered LaTeX formulas,
a sticky table-of-contents sidebar, alternating-row tables, syntax-highlighted
code blocks, and print-friendly styles.

Output
------
    data/exports/nd_methodology_{YYYYMMDD}.html

Usage
-----
    python scripts/exports/build_methodology_html.py

Key dependencies
----------------
    - markdown (pip package, v3.x) — Markdown-to-HTML conversion
    - MathJax 3 CDN — client-side LaTeX math rendering (loaded in output HTML)

Data lineage
------------
    Input:  docs/methodology.md
    Output: data/exports/nd_methodology_{datestamp}.html

SOP-002 compliance
------------------
    Purpose:       Convert methodology Markdown to distributable HTML
    Author:        Projection team
    Date created:  2026-03-02
    Last modified: 2026-03-02
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _markdown_to_html import build_html_document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
METHODOLOGY_MD = PROJECT_ROOT / "docs" / "methodology.md"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

TODAY = datetime.now(tz=UTC).date()
DATE_STAMP = TODAY.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Extract metadata from the markdown front matter
# ---------------------------------------------------------------------------
def _extract_metadata(md_text: str) -> tuple[str, str]:
    """Pull Version and Date from the methodology Markdown header lines.

    Returns (version, date_str).  Falls back to defaults if not found.
    """
    version = "1.0"
    date_str = "February 2026"

    version_match = re.search(r"^\*\*Version:\*\*\s*(.+)$", md_text, re.MULTILINE)
    if version_match:
        version = version_match.group(1).strip()

    date_match = re.search(r"^\*\*Date:\*\*\s*(.+)$", md_text, re.MULTILINE)
    if date_match:
        date_str = date_match.group(1).strip()

    return version, date_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Read methodology.md and write the HTML export."""
    if not METHODOLOGY_MD.exists():
        print(f"ERROR: Source file not found: {METHODOLOGY_MD}")
        sys.exit(1)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    md_text = METHODOLOGY_MD.read_text(encoding="utf-8")
    version, date_str = _extract_metadata(md_text)

    html = build_html_document(
        title="North Dakota Cohort-Component Population Projection Model: Methodology",
        version=version,
        date_str=date_str,
        md_source=md_text,
        subtitle="Base Year 2025 (Census PEP Vintage 2025) — Projection Horizon 2025–2055",
    )

    out_path = EXPORT_DIR / f"nd_methodology_{DATE_STAMP}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
