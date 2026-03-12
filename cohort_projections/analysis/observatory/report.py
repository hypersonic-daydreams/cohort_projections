"""Observatory report generation in console and HTML formats.

Created: 2026-03-12
Author: Claude Code / N. Haarstad

Purpose:
    Generates Projection Observatory reports for terminal output and self-contained
    HTML files.  Reports summarize N-way comparisons across benchmark runs,
    surface recommendations for next experiments, and highlight persistent
    weaknesses in the projection methodology.

Method:
    1. Accept a comparator result, recommendation list, and results store.
    2. Build structured report sections: inventory, ranking, county group impact,
       Pareto frontier, persistent weaknesses, and recommendations.
    3. For console output, format as aligned text tables with box-drawing chars.
    4. For HTML output, produce a single-file document with embedded CSS/JS
       following the same visual language as the evaluation HTML reports.

Inputs:
    - ``ComparisonResult`` from ``ObservatoryComparator``
    - ``list[Recommendation]`` from ``ObservatoryRecommender``
    - ``ResultsStore`` for run metadata and inventory counts

Outputs:
    - Console text (returned as string)
    - HTML file written to ``data/analysis/observatory/observatory_report_{timestamp}.html``
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "observatory"


# ===================================================================
# Public API
# ===================================================================


class ObservatoryReport:
    """Generate Observatory reports in console and HTML formats.

    Parameters
    ----------
    comparator_result:
        Output from ``ObservatoryComparator.compare()``.  Expected to be a
        dict-like or dataclass with fields such as ``ranking``,
        ``county_group_impact``, ``pareto_frontier``, and
        ``persistent_weaknesses``.  When *None* or empty, affected sections
        degrade gracefully to placeholder text.
    recommendations:
        List of recommendation objects from ``ObservatoryRecommender``.
        Each item is expected to expose ``experiment_id``, ``rationale``,
        and ``priority`` attributes (or dict keys).
    store:
        A ``ResultsStore`` used for run inventory metadata (counts, date
        range, methods).
    timestamp:
        ISO-format timestamp string.  Defaults to the current UTC time
        if not provided.
    """

    def __init__(
        self,
        comparator_result: Any,
        recommendations: list[Any],
        store: Any,
        timestamp: str | None = None,
    ) -> None:
        self._result = comparator_result
        self._recommendations = recommendations or []
        self._store = store
        self._timestamp = timestamp or datetime.now(tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def generate_console_report(self) -> str:
        """Full text report for terminal output.

        Returns
        -------
        str
            Multi-line report suitable for ``print()`` output.
        """
        sections: list[str] = [
            self._console_header(),
            self._console_run_inventory(),
            self._console_ranking_table(),
            self._console_county_group_impact(),
            self._console_pareto_frontier(),
            self._console_persistent_weaknesses(),
            self._console_recommendations(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html_report(self, output_path: Path | None = None) -> Path:
        """Self-contained HTML report with embedded CSS.

        Parameters
        ----------
        output_path:
            Where to write the file.  Defaults to
            ``data/analysis/observatory/observatory_report_{timestamp}.html``.

        Returns
        -------
        Path
            The path the HTML was written to.
        """
        if output_path is None:
            safe_ts = self._timestamp.replace(":", "").replace("-", "")
            output_path = DEFAULT_OUTPUT_DIR / f"observatory_report_{safe_ts}.html"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        body_sections: list[str] = [
            self._html_header(),
            self._html_run_inventory(),
            self._html_ranking_table(),
            self._html_county_group_impact(),
            self._html_pareto_frontier(),
            self._html_persistent_weaknesses(),
            self._html_recommendations(),
        ]

        body = "\n".join(s for s in body_sections if s)
        toc = self._html_toc(body_sections)
        html = _wrap_html(
            title=f"Projection Observatory Report -- {self._timestamp[:10]}",
            toc=toc,
            body=body,
        )

        output_path.write_text(html, encoding="utf-8")
        logger.info("Observatory HTML report written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(self) -> str:
        """Brief 5-10 line executive summary.

        Returns
        -------
        str
            Compact summary suitable for log output or chat.
        """
        lines: list[str] = []

        # Inventory headline
        n_runs, date_range, methods = self._inventory_stats()
        lines.append(f"Observatory: {n_runs} run(s), {len(methods)} method(s)")
        if date_range:
            lines.append(f"Date range: {date_range[0]} to {date_range[1]}")

        # Top variant
        ranking = self._get_ranking()
        if ranking:
            top = ranking[0]
            label = _dict_get(top, "method", "run_id", default="?")
            metric_val = _dict_get(top, "primary_metric", "value", default="?")
            lines.append(f"Top variant: {label} (primary metric: {metric_val})")

        # Persistent weaknesses count
        weaknesses = self._get_persistent_weaknesses()
        if weaknesses:
            lines.append(f"Persistent weaknesses: {len(weaknesses)}")
        else:
            lines.append("No persistent weaknesses identified.")

        # Recommendation count
        n_rec = len(self._recommendations)
        lines.append(f"Recommendations: {n_rec} next experiment(s)")

        # Top recommendation
        if self._recommendations:
            top_rec = self._recommendations[0]
            rec_id = _dict_get(top_rec, "experiment_id", default="?")
            rec_reason = _dict_get(top_rec, "rationale", default="")
            if rec_reason:
                lines.append(f"  #1: {rec_id} -- {rec_reason[:80]}")
            else:
                lines.append(f"  #1: {rec_id}")

        return "\n".join(lines)

    # ==================================================================
    # Console section builders
    # ==================================================================

    def _console_header(self) -> str:
        date_str = self._timestamp[:10] if len(self._timestamp) >= 10 else self._timestamp
        width = 60
        return "\n".join([
            "=" * width,
            f"  Projection Observatory Report -- {date_str}",
            "=" * width,
        ])

    def _console_run_inventory(self) -> str:
        n_runs, date_range, methods = self._inventory_stats()
        lines = ["Run Inventory", "-" * 40]
        lines.append(f"  Completed runs: {n_runs}")
        if date_range:
            lines.append(f"  Date range:     {date_range[0]} to {date_range[1]}")
        else:
            lines.append("  Date range:     N/A")
        lines.append(f"  Methods tested: {', '.join(methods) if methods else 'none'}")
        return "\n".join(lines)

    def _console_ranking_table(self) -> str:
        ranking = self._get_ranking()
        if not ranking:
            return "Ranking Table\n" + "-" * 40 + "\n  No ranking data available."

        lines = ["Ranking Table (Top 10)", "-" * 40]

        # Build rows
        headers = ["Rank", "Variant", "Primary Metric"]
        # Detect extra metric keys from first entry
        extra_keys = _extra_metric_keys(ranking[0]) if ranking else []
        headers.extend(extra_keys)

        rows: list[list[str]] = []
        for i, entry in enumerate(ranking[:10], 1):
            label = _dict_get(entry, "method", "run_id", default="?")
            primary = _fmt_metric(_dict_get(entry, "primary_metric", "value", default=None))
            row = [str(i), str(label), primary]
            for k in extra_keys:
                row.append(_fmt_metric(_dict_get(entry, k, default=None)))
            rows.append(row)

        lines.append(_text_table(headers, rows))
        return "\n".join(lines)

    def _console_county_group_impact(self) -> str:
        best_per_group = _safe_attr(self._result, "best_per_group", {})
        if not best_per_group:
            return "County Group Impact\n" + "-" * 40 + "\n  No county group data available."

        lines = ["County Group Impact", "-" * 40]
        headers = ["Group", "Best Variant"]
        rows: list[list[str]] = []
        for group, run_id in best_per_group.items():
            rows.append([str(group), str(run_id)])
        lines.append(_text_table(headers, rows))
        return "\n".join(lines)

    def _console_pareto_frontier(self) -> str:
        pareto_runs = _safe_attr(self._result, "pareto_runs", [])
        if not pareto_runs:
            return "Pareto Frontier\n" + "-" * 40 + "\n  No Pareto-optimal runs identified."

        summary = _safe_attr(self._result, "summary", {})
        x_m = summary.get("pareto_metric_x", "?") if isinstance(summary, dict) else "?"
        y_m = summary.get("pareto_metric_y", "?") if isinstance(summary, dict) else "?"

        lines = [f"Pareto Frontier ({x_m} vs {y_m})", "-" * 40]
        for run_id in pareto_runs:
            lines.append(f"  {run_id}")
        return "\n".join(lines)

    def _console_persistent_weaknesses(self) -> str:
        weaknesses = self._get_persistent_weaknesses()
        if not weaknesses:
            return "Persistent Weaknesses\n" + "-" * 40 + "\n  No persistent weaknesses identified."

        lines = ["Persistent Weaknesses", "-" * 40]
        for i, w in enumerate(weaknesses, 1):
            desc = _dict_get(w, "description", "county", default="?")
            detail = _dict_get(w, "detail", "severity", default="")
            lines.append(f"  {i}. {desc}")
            if detail:
                lines.append(f"     {detail}")
        return "\n".join(lines)

    def _console_recommendations(self) -> str:
        if not self._recommendations:
            return "Recommendations\n" + "-" * 40 + "\n  No recommendations available."

        lines = ["Recommendations (Top 5)", "-" * 40]
        for i, rec in enumerate(self._recommendations[:5], 1):
            param = _safe_attr(rec, "parameter", "?")
            value = _safe_attr(rec, "suggested_value", "?")
            rationale = _safe_attr(rec, "rationale", "")
            priority = _safe_attr(rec, "priority", "")
            code_flag = " [CODE]" if _safe_attr(rec, "requires_code_change", False) else ""
            line = f"  {i}. [{priority}] {param} -> {value}{code_flag}"
            lines.append(line)
            if rationale:
                lines.append(f"     {rationale}")
        return "\n".join(lines)

    # ==================================================================
    # HTML section builders
    # ==================================================================

    def _html_header(self) -> str:
        date_str = self._timestamp[:10] if len(self._timestamp) >= 10 else self._timestamp
        ts_display = self._timestamp
        return f"""
<section id="report-header" class="header-section">
  <h1>Projection Observatory Report</h1>
  <div class="header-meta">
    <span class="meta-item"><strong>Date:</strong> {_esc(date_str)}</span>
    <span class="meta-item"><strong>Generated:</strong> {_esc(ts_display)}</span>
  </div>
</section>
"""

    def _html_run_inventory(self) -> str:
        n_runs, date_range, methods = self._inventory_stats()
        dr_text = f"{date_range[0]} to {date_range[1]}" if date_range else "N/A"
        methods_text = ", ".join(methods) if methods else "none"

        return f"""
<section id="section-inventory" data-toc="Run Inventory">
  <h2>Run Inventory</h2>
  <div class="inventory-grid">
    <div class="inventory-card">
      <div class="inv-label">Completed Runs</div>
      <div class="inv-value">{n_runs}</div>
    </div>
    <div class="inventory-card">
      <div class="inv-label">Date Range</div>
      <div class="inv-value inv-value-sm">{_esc(dr_text)}</div>
    </div>
    <div class="inventory-card">
      <div class="inv-label">Methods Tested</div>
      <div class="inv-value inv-value-sm">{_esc(methods_text)}</div>
    </div>
  </div>
</section>
"""

    def _html_ranking_table(self) -> str:
        ranking = self._get_ranking()
        if not ranking:
            return """
<section id="section-ranking" data-toc="Ranking Table">
  <h2>Ranking Table</h2>
  <p>No ranking data available.</p>
</section>
"""

        headers = ["Rank", "Variant", "Primary Metric"]
        extra_keys = _extra_metric_keys(ranking[0]) if ranking else []
        headers.extend(extra_keys)

        rows_html: list[str] = []
        for i, entry in enumerate(ranking[:10], 1):
            label = _esc(str(_dict_get(entry, "method", "run_id", default="?")))
            primary_val = _dict_get(entry, "primary_metric", "value", default=None)
            primary_str = _fmt_metric(primary_val)
            primary_cls = _metric_cell_class(primary_val)

            cells = [
                f"<td>{i}</td>",
                f"<td>{label}</td>",
                f'<td class="{primary_cls}">{primary_str}</td>',
            ]
            for k in extra_keys:
                val = _dict_get(entry, k, default=None)
                cells.append(f'<td class="{_metric_cell_class(val)}">{_fmt_metric(val)}</td>')
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        header_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

        return f"""
<section id="section-ranking" data-toc="Ranking Table">
  <h2>Ranking Table (Top 10)</h2>
  <div class="table-wrapper">
    <table class="data-table sortable">
      <thead><tr>{header_html}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</section>
"""

    def _html_county_group_impact(self) -> str:
        groups = self._get_county_group_impact()
        if not groups:
            return """
<section id="section-county-groups" data-toc="County Group Impact">
  <h2>County Group Impact</h2>
  <p>No county group data available.</p>
</section>
"""

        rows_html: list[str] = []
        for group in groups:
            name = _esc(str(_dict_get(group, "group", "name", default="?")))
            best = _esc(str(_dict_get(group, "best_variant", "method", default="?")))
            val = _dict_get(group, "best_value", "value", default=None)
            val_str = _fmt_metric(val)
            rows_html.append(
                f"<tr><td>{name}</td><td>{best}</td>"
                f'<td class="{_metric_cell_class(val)}">{val_str}</td></tr>'
            )

        return f"""
<section id="section-county-groups" data-toc="County Group Impact">
  <h2>County Group Impact</h2>
  <div class="table-wrapper">
    <table class="data-table sortable">
      <thead><tr><th>Group</th><th>Best Variant</th><th>Metric Value</th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</section>
"""

    def _html_pareto_frontier(self) -> str:
        frontier = self._get_pareto_frontier()
        if not frontier:
            return """
<section id="section-pareto" data-toc="Pareto Frontier">
  <h2>Pareto Frontier</h2>
  <p>No Pareto-optimal runs identified.</p>
</section>
"""

        rows_html: list[str] = []
        for entry in frontier:
            label = _esc(str(_dict_get(entry, "method", "run_id", default="?")))
            near = _dict_get(entry, "near_term", default=None)
            far = _dict_get(entry, "long_term", default=None)
            rows_html.append(
                f"<tr><td>{label}</td>"
                f'<td class="{_metric_cell_class(near)}">{_fmt_metric(near)}</td>'
                f'<td class="{_metric_cell_class(far)}">{_fmt_metric(far)}</td></tr>'
            )

        return f"""
<section id="section-pareto" data-toc="Pareto Frontier">
  <h2>Pareto Frontier (Near-term vs Long-term)</h2>
  <div class="table-wrapper">
    <table class="data-table sortable">
      <thead><tr><th>Variant</th><th>Near-term</th><th>Long-term</th></tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</section>
"""

    def _html_persistent_weaknesses(self) -> str:
        weaknesses = self._get_persistent_weaknesses()
        if not weaknesses:
            return """
<section id="section-weaknesses" data-toc="Persistent Weaknesses">
  <h2>Persistent Weaknesses</h2>
  <p>No persistent weaknesses identified across tested variants.</p>
</section>
"""

        items: list[str] = []
        for w in weaknesses:
            desc = _esc(str(_dict_get(w, "description", "county", default="?")))
            detail = _esc(str(_dict_get(w, "detail", "severity", default="")))
            item = f"<li><strong>{desc}</strong>"
            if detail:
                item += f"<br/><span class='weakness-detail'>{detail}</span>"
            item += "</li>"
            items.append(item)

        return f"""
<section id="section-weaknesses" data-toc="Persistent Weaknesses">
  <h2>Persistent Weaknesses</h2>
  <ul class="weakness-list">
    {"".join(items)}
  </ul>
</section>
"""

    def _html_recommendations(self) -> str:
        if not self._recommendations:
            return """
<section id="section-recommendations" data-toc="Recommendations">
  <h2>Recommendations</h2>
  <p>No recommendations available.</p>
</section>
"""

        cards: list[str] = []
        for i, rec in enumerate(self._recommendations[:5], 1):
            exp_id = _esc(str(_dict_get(rec, "experiment_id", default="?")))
            rationale = _esc(str(_dict_get(rec, "rationale", default="")))
            priority = _esc(str(_dict_get(rec, "priority", default="")))
            priority_cls = _priority_class(priority)

            card = f"""
    <div class="rec-card">
      <div class="rec-rank">{i}</div>
      <div class="rec-body">
        <div class="rec-header">
          <span class="rec-id">{exp_id}</span>
          {f'<span class="rec-priority {priority_cls}">{priority}</span>' if priority else ''}
        </div>
        {f'<div class="rec-rationale">{rationale}</div>' if rationale else ''}
      </div>
    </div>"""
            cards.append(card)

        return f"""
<section id="section-recommendations" data-toc="Recommendations">
  <h2>Recommendations (Top 5)</h2>
  <div class="rec-list">
    {"".join(cards)}
  </div>
</section>
"""

    # ==================================================================
    # Data extraction helpers
    # ==================================================================

    def _inventory_stats(self) -> tuple[int, tuple[str, str] | None, list[str]]:
        """Extract run count, date range, and method list from the store.

        Returns
        -------
        tuple
            ``(n_runs, (earliest_date, latest_date) | None, methods_list)``
        """
        try:
            index = self._store.get_index()
        except Exception:
            return 0, None, []

        if index.empty:
            return 0, None, []

        n_runs = len(index["run_id"].unique()) if "run_id" in index.columns else 0

        # Date range
        date_range: tuple[str, str] | None = None
        for date_col in ("run_date", "created_at_utc", "created_at"):
            if date_col in index.columns:
                dates = index[date_col].dropna()
                if not dates.empty:
                    date_range = (str(dates.min())[:10], str(dates.max())[:10])
                    break

        # Methods
        methods: list[str] = []
        for method_col in ("method", "method_id", "challenger_method_id"):
            if method_col in index.columns:
                methods = sorted(index[method_col].dropna().unique().tolist())
                break
        if not methods and "champion_method_id" in index.columns:
            champ = index["champion_method_id"].dropna().unique().tolist()
            methods = sorted(set(champ))

        return n_runs, date_range, methods

    def _get_ranking(self) -> list[Any]:
        """Extract ranking list from comparator result."""
        val = _safe_attr(self._result, "ranking", [])
        return _df_to_records(val)

    def _get_county_group_impact(self) -> list[Any]:
        """Extract county group impact from comparator result."""
        val = _safe_attr(self._result, "county_group_impact", [])
        return _df_to_records(val)

    def _get_pareto_frontier(self) -> list[Any]:
        """Extract Pareto frontier from comparator result."""
        val = _safe_attr(self._result, "pareto_runs", [])
        if isinstance(val, list):
            return val
        return _df_to_records(val)

    def _get_persistent_weaknesses(self) -> list[Any]:
        """Extract persistent weaknesses from comparator result."""
        return _safe_attr(self._result, "persistent_weaknesses", [])

    # ==================================================================
    # HTML TOC
    # ==================================================================

    def _html_toc(self, sections: list[str]) -> str:
        """Build floating TOC from section HTML."""
        import re

        items: list[str] = []
        for section_html in sections:
            if not section_html:
                continue
            toc_match = re.search(r'data-toc="([^"]+)"', section_html)
            id_match = re.search(r'id="([^"]+)"', section_html)
            if toc_match and id_match:
                label = toc_match.group(1)
                section_id = id_match.group(1)
                items.append(
                    f'<li><a href="#{_esc(section_id)}">{_esc(label)}</a></li>'
                )

        if not items:
            return ""

        return f"""
<nav class="toc-sidebar" id="toc-nav">
  <div class="toc-title">Contents</div>
  <ul>
    {"".join(items)}
  </ul>
</nav>
"""


# ===================================================================
# Helper utilities
# ===================================================================


def _df_to_records(val: Any) -> list[Any]:
    """Convert a pandas DataFrame to a list of dicts, or return as-is if already a list."""
    if val is None:
        return []
    try:
        import pandas as pd
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return []
            return val.to_dict("records")
    except ImportError:
        pass
    if isinstance(val, list):
        return val
    return []


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute or dict key from *obj*, returning *default* on failure."""
    if obj is None:
        return default
    # Try attribute access (dataclass / named tuple)
    try:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
    except Exception:
        pass
    # Try dict access
    try:
        val = obj[attr]  # type: ignore[index]
        if val is not None:
            return val
    except (KeyError, TypeError, IndexError):
        pass
    return default


def _dict_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Try multiple attribute/dict keys in order, return first non-None."""
    for key in keys:
        val = _safe_attr(obj, key)
        if val is not None:
            return val
    return default


def _extra_metric_keys(entry: Any) -> list[str]:
    """Detect extra metric column names from a ranking entry."""
    known = {"method", "run_id", "primary_metric", "value", "rank"}
    extras: list[str] = []
    try:
        keys = entry.keys() if hasattr(entry, "keys") else []
    except Exception:
        keys = []
    for k in keys:
        if k not in known and not k.startswith("_"):
            extras.append(str(k))
    return extras


def _fmt_metric(value: Any) -> str:
    """Format a metric value for display."""
    if value is None:
        return "-"
    try:
        v = float(value)
        if abs(v) < 0.001:
            return f"{v:.4f}"
        return f"{v:.3f}"
    except (ValueError, TypeError):
        return str(value)


def _metric_cell_class(value: Any) -> str:
    """Return a CSS class based on metric value magnitude."""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return ""
    # Lower is better for MAPE-like metrics
    if v < 3.0:
        return "cell-good"
    if v < 7.0:
        return "cell-warn"
    return "cell-bad"


def _priority_class(priority: str) -> str:
    """Map recommendation priority to a CSS class."""
    p = priority.lower()
    if p in ("high", "critical"):
        return "priority-high"
    if p in ("medium", "moderate"):
        return "priority-medium"
    return "priority-low"


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple text-aligned table.

    Parameters
    ----------
    headers:
        Column header strings.
    rows:
        List of row lists (each the same length as *headers*).

    Returns
    -------
    str
        Aligned text table with ``+--`` borders.
    """
    if not rows:
        return "  (no data)"

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def _sep() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i] if i < len(widths) else len(cell)
            parts.append(f" {cell:<{w}} ")
        return "|" + "|".join(parts) + "|"

    lines = [_sep(), _row(headers), _sep()]
    for row in rows:
        lines.append(_row(row))
    lines.append(_sep())
    return "\n".join(lines)


# ===================================================================
# Full HTML wrapper
# ===================================================================


def _wrap_html(title: str, toc: str, body: str) -> str:
    """Wrap body content in a complete HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  {toc}
  <main class="report-main">
    {body}
  </main>
  <script>
{_JS}
  </script>
</body>
</html>"""


# ===================================================================
# Embedded CSS
# ===================================================================

_CSS = """
/* ---- Reset & Typography ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: #1a1a2e;
  background: #f8f9fa;
}

/* ---- Layout ---- */
.report-main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 2.5rem 4rem 2.5rem;
  margin-left: 240px;
}

/* ---- TOC Sidebar ---- */
.toc-sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: 220px;
  height: 100vh;
  background: #16213e;
  color: #e8e8e8;
  padding: 1.5rem 1rem;
  overflow-y: auto;
  z-index: 100;
}

.toc-title {
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #2a3a5c;
}

.toc-sidebar ul { list-style: none; }
.toc-sidebar li { margin-bottom: 0.25rem; }

.toc-sidebar a {
  color: #cbd5e1;
  text-decoration: none;
  font-size: 0.82rem;
  display: block;
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
  transition: background 0.15s, color 0.15s;
}

.toc-sidebar a:hover {
  background: #1e3a5f;
  color: #ffffff;
}

/* ---- Header ---- */
.header-section {
  background: linear-gradient(135deg, #16213e 0%, #1a3a5c 100%);
  color: #ffffff;
  padding: 2.5rem 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.header-section h1 {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
}

.header-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  font-size: 0.85rem;
  color: #94a3b8;
}

.meta-item strong { color: #cbd5e1; }

/* ---- Section Headings ---- */
section { margin-bottom: 2.5rem; }

h2 {
  font-size: 1.35rem;
  font-weight: 700;
  color: #16213e;
  border-bottom: 2px solid #2563eb;
  padding-bottom: 0.5rem;
  margin-bottom: 1.25rem;
}

h3 {
  font-size: 1.05rem;
  font-weight: 600;
  color: #334155;
  margin: 1.25rem 0 0.75rem;
}

/* ---- Inventory Grid ---- */
.inventory-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.inventory-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.inv-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
  margin-bottom: 0.25rem;
}

.inv-value {
  font-size: 2rem;
  font-weight: 800;
  color: #16213e;
  font-variant-numeric: tabular-nums;
}

.inv-value-sm {
  font-size: 1rem;
  font-weight: 600;
}

/* ---- Tables ---- */
.table-wrapper {
  overflow-x: auto;
  margin-bottom: 1rem;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  overflow: hidden;
}

.data-table thead {
  position: sticky;
  top: 0;
  z-index: 10;
}

.data-table th {
  background: #16213e;
  color: #e8e8e8;
  font-weight: 600;
  text-align: left;
  padding: 0.6rem 0.75rem;
  white-space: nowrap;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.data-table.sortable th {
  cursor: pointer;
  user-select: none;
}

.data-table.sortable th:hover { background: #1e3a5f; }

.data-table.sortable th::after {
  content: " \\2195";
  font-size: 0.7rem;
  opacity: 0.5;
}

.data-table td {
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #f1f5f9;
  font-variant-numeric: tabular-nums;
  font-family: "SF Mono", "Fira Code", "Fira Mono", "Roboto Mono", monospace;
  font-size: 0.8rem;
}

.data-table tbody tr:nth-child(even) { background: #f8fafc; }
.data-table tbody tr:hover { background: #eef2ff; }

/* Cell color coding */
.cell-good { background: #ecfdf5 !important; color: #065f46; font-weight: 600; }
.cell-warn { background: #fffbeb !important; color: #92400e; font-weight: 600; }
.cell-bad  { background: #fef2f2 !important; color: #991b1b; font-weight: 600; }

/* ---- Weakness List ---- */
.weakness-list {
  margin-left: 1.25rem;
}

.weakness-list li {
  margin-bottom: 0.75rem;
  color: #334155;
  font-size: 0.9rem;
}

.weakness-detail {
  font-size: 0.82rem;
  color: #64748b;
}

/* ---- Recommendation Cards ---- */
.rec-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.rec-card {
  display: flex;
  align-items: flex-start;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.rec-rank {
  font-size: 1.5rem;
  font-weight: 800;
  color: #2563eb;
  min-width: 2rem;
  margin-right: 1rem;
  line-height: 1;
  padding-top: 0.2rem;
}

.rec-body { flex: 1; }

.rec-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.25rem;
}

.rec-id {
  font-weight: 700;
  font-size: 0.95rem;
  color: #16213e;
}

.rec-priority {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 700;
  padding: 0.15rem 0.5rem;
  border-radius: 3px;
}

.priority-high { background: #fef2f2; color: #991b1b; }
.priority-medium { background: #fffbeb; color: #92400e; }
.priority-low { background: #ecfdf5; color: #065f46; }

.rec-rationale {
  font-size: 0.85rem;
  color: #475569;
  line-height: 1.5;
}

/* ---- Print Styles ---- */
@media print {
  body { background: #ffffff; font-size: 11pt; }
  .toc-sidebar { display: none; }
  .report-main { margin-left: 0; padding: 0; max-width: 100%; }
  .header-section {
    background: #16213e !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  section { page-break-inside: avoid; }
  .data-table th {
    background: #16213e !important;
    color: #ffffff !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .cell-good, .cell-warn, .cell-bad {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}

/* ---- Responsive ---- */
@media (max-width: 900px) {
  .toc-sidebar { display: none; }
  .report-main { margin-left: 0; padding: 1rem; }
  .inventory-grid { grid-template-columns: 1fr; }
}
"""


# ===================================================================
# Embedded JavaScript
# ===================================================================

_JS = """
// Table sorting
document.querySelectorAll("table.sortable th").forEach(function(th) {
  th.addEventListener("click", function() {
    var table = th.closest("table");
    var tbody = table.querySelector("tbody");
    var rows = Array.from(tbody.querySelectorAll("tr"));
    var colIdx = Array.from(th.parentNode.children).indexOf(th);
    var asc = th.dataset.sortDir !== "asc";
    th.dataset.sortDir = asc ? "asc" : "desc";

    th.parentNode.querySelectorAll("th").forEach(function(h) {
      if (h !== th) delete h.dataset.sortDir;
    });

    rows.sort(function(a, b) {
      var aText = a.children[colIdx].textContent.trim();
      var bText = b.children[colIdx].textContent.trim();
      var aNum = parseFloat(aText.replace(/[+%,]/g, ""));
      var bNum = parseFloat(bText.replace(/[+%,]/g, ""));
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return asc ? aNum - bNum : bNum - aNum;
      }
      return asc ? aText.localeCompare(bText) : bText.localeCompare(aText);
    });

    rows.forEach(function(row) { tbody.appendChild(row); });
  });
});

// Smooth scroll for TOC links
document.querySelectorAll(".toc-sidebar a").forEach(function(link) {
  link.addEventListener("click", function(e) {
    e.preventDefault();
    var target = document.querySelector(this.getAttribute("href"));
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
});
"""
