"""Generate a self-contained HTML evaluation report from evaluation results.

Produces a single-file HTML report with embedded CSS, JavaScript, and
base64-encoded chart images.  No external dependencies are required to
view the output -- it can be opened in any modern browser or printed to
PDF for distribution.

Design rationale:
    - Government / demographic office audience: professional, subdued color
      palette (blues, grays) with clear data hierarchy.
    - Single-file delivery: all assets inlined so the report can be emailed
      or archived without auxiliary files.
    - Minimal JavaScript: table sorting and smooth-scroll TOC only; no
      frameworks, no fetch calls, no build step.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_structures import ScorecardEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib (optional, for figure encoding)
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ===================================================================
# Public API
# ===================================================================


def generate_html_report(
    evaluation_results: dict[str, Any],
    config: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "Population Projection Evaluation Report",
    include_appendix: bool = True,
) -> Path:
    """Generate a self-contained HTML evaluation report.

    Args:
        evaluation_results: Output dict from
            ``EvaluationRunner.run_full_evaluation()``.  Expected keys:
            ``accuracy_diagnostics``, ``realism_diagnostics``,
            ``component_diagnostics``, ``comparison``, ``sensitivity``,
            ``scorecard``, ``figures``.
        config: Evaluation config dict (from ``evaluation_config.yaml``).
        output_path: Path to write the HTML file.
        title: Report title shown in the header.
        include_appendix: Whether to include the full diagnostics appendix.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scorecard: ScorecardEntry | None = evaluation_results.get("scorecard")
    accuracy_df: pd.DataFrame = evaluation_results.get(
        "accuracy_diagnostics", pd.DataFrame()
    )
    realism_df: pd.DataFrame = evaluation_results.get(
        "realism_diagnostics", pd.DataFrame()
    )
    comparison_df: pd.DataFrame | None = evaluation_results.get("comparison")
    sensitivity: dict[str, Any] | None = evaluation_results.get("sensitivity")
    figures: dict[str, Any] = evaluation_results.get("figures", {})

    # Build encoded figure map
    encoded_figures = _encode_figures(figures)

    # Determine model / run metadata
    model_name = scorecard.model_name if scorecard else ""
    run_id = scorecard.run_id if scorecard else ""

    # Compute derived tables
    near_term_max = config.get("near_term_max_horizon", 5)
    long_term_min = config.get("long_term_min_horizon", 10)
    county_groups = config.get("county_groups", {})
    sentinel_counties = config.get("sentinel_counties", {})

    # Assemble sections
    sections: list[str] = []
    sections.append(_build_header(title, model_name, run_id))
    sections.append(
        _build_executive_summary(scorecard, accuracy_df, near_term_max, long_term_min)
    )
    sections.append(
        _build_accuracy_section(
            accuracy_df,
            encoded_figures,
            near_term_max,
            long_term_min,
            county_groups,
            sentinel_counties,
        )
    )
    sections.append(_build_bias_section(accuracy_df, encoded_figures))

    if not realism_df.empty:
        sections.append(_build_realism_section(realism_df, encoded_figures))

    if comparison_df is not None and not comparison_df.empty:
        sections.append(_build_comparison_section(comparison_df, encoded_figures))

    if sensitivity is not None:
        sections.append(
            _build_sensitivity_section(sensitivity, encoded_figures)
        )

    if include_appendix:
        sections.append(
            _build_appendix(accuracy_df, realism_df, config)
        )

    body = "\n".join(sections)
    toc = _build_toc(sections)
    html = _wrap_html(title, toc, body)

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML evaluation report written to %s", output_path)
    return output_path


# ===================================================================
# Figure encoding
# ===================================================================


def _encode_figures(figures: dict[str, Any]) -> dict[str, str]:
    """Convert matplotlib Figures to base64 PNG data URIs."""
    encoded: dict[str, str] = {}
    if not _MPL_AVAILABLE or not figures:
        return encoded
    for name, fig in figures.items():
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            encoded[name] = f"data:image/png;base64,{b64}"
        except Exception:
            logger.warning("Failed to encode figure '%s'", name, exc_info=True)
    return encoded


# ===================================================================
# Section builders
# ===================================================================

_SECTION_COUNTER = 0


def _section_id() -> str:
    global _SECTION_COUNTER  # noqa: PLW0603
    _SECTION_COUNTER += 1
    return f"section-{_SECTION_COUNTER}"


def _reset_section_counter() -> None:
    global _SECTION_COUNTER  # noqa: PLW0603
    _SECTION_COUNTER = 0


def _build_header(title: str, model_name: str, run_id: str) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""
<section id="report-header" class="header-section">
  <h1>{_esc(title)}</h1>
  <div class="header-meta">
    <span class="meta-item"><strong>Generated:</strong> {timestamp}</span>
    <span class="meta-item"><strong>Model:</strong> {_esc(model_name) or "N/A"}</span>
    <span class="meta-item"><strong>Run ID:</strong> {_esc(run_id) or "N/A"}</span>
  </div>
</section>
"""


def _build_executive_summary(
    scorecard: ScorecardEntry | None,
    accuracy_df: pd.DataFrame,
    near_term_max: int,
    long_term_min: int,
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Executive Summary">',
        "<h2>Executive Summary</h2>",
    ]

    if scorecard is not None:
        parts.append(_render_scorecard_visual(scorecard))
        parts.append(_render_key_findings(scorecard, accuracy_df, near_term_max, long_term_min))
    else:
        parts.append("<p>No scorecard data available.</p>")

    parts.append("</section>")
    return "\n".join(parts)


def _render_scorecard_visual(sc: ScorecardEntry) -> str:
    """Render CSS-only scorecard gauge bars."""
    axes = [
        ("Near-term Accuracy", sc.near_term_accuracy, True),
        ("Long-term Accuracy", sc.long_term_accuracy, True),
        ("Bias Calibration", abs(sc.bias_calibration), True),
        ("Age-structure Realism", sc.age_structure_realism, False),
        ("Robustness / Stability", sc.robustness_stability, False),
        ("Interpretability", sc.interpretability, False),
    ]

    rows: list[str] = []
    for label, value, is_error in axes:
        if is_error:
            # Error-like: lower is better, scale for display (cap at 20%)
            pct = max(0.0, min(100.0, (1.0 - value / 0.20) * 100.0))
            display_val = f"{value:.4f}"
            color_class = _error_color_class(value)
        else:
            # Quality-like: higher is better (0-1 scale)
            pct = max(0.0, min(100.0, value * 100.0))
            display_val = f"{value:.4f}"
            color_class = _quality_color_class(value)

        rows.append(f"""
        <div class="scorecard-row">
          <div class="scorecard-label">{label}</div>
          <div class="scorecard-bar-container">
            <div class="scorecard-bar {color_class}" style="width: {pct:.1f}%"></div>
          </div>
          <div class="scorecard-value">{display_val}</div>
        </div>""")

    bias_direction = "overprojection" if sc.bias_calibration > 0 else "underprojection"
    composite_class = _error_color_class(sc.composite_score)

    return f"""
    <div class="scorecard-container">
      <div class="composite-score {composite_class}">
        <div class="composite-label">Composite Score</div>
        <div class="composite-value">{sc.composite_score:.4f}</div>
        <div class="composite-note">Lower is better</div>
      </div>
      <div class="scorecard-axes">
        {"".join(rows)}
      </div>
      <div class="bias-indicator">
        Bias direction: <strong>{bias_direction}</strong>
        ({sc.bias_calibration:+.4f})
      </div>
    </div>
    """


def _render_key_findings(
    sc: ScorecardEntry,
    accuracy_df: pd.DataFrame,
    near_term_max: int,
    long_term_min: int,
) -> str:
    """Auto-generate key findings bullets from data."""
    findings: list[str] = []

    if accuracy_df.empty:
        return "<ul><li>No accuracy data available for analysis.</li></ul>"

    mape_df = accuracy_df.loc[accuracy_df["metric_name"] == "mape"]

    if not mape_df.empty and "geography" in mape_df.columns:
        # Best and worst county
        county_mape = mape_df.groupby("geography")["value"].mean()
        if len(county_mape) > 1:
            best = county_mape.idxmin()
            worst = county_mape.idxmax()
            findings.append(
                f"Best-performing geography: <strong>{_esc(str(best))}</strong> "
                f"(mean MAPE {county_mape[best]:.2f}%)"
            )
            findings.append(
                f"Worst-performing geography: <strong>{_esc(str(worst))}</strong> "
                f"(mean MAPE {county_mape[worst]:.2f}%)"
            )

    # Horizon where error exceeds 5%
    if not mape_df.empty:
        horizon_mape = mape_df.groupby("horizon")["value"].mean().sort_index()
        threshold_horizons = horizon_mape[horizon_mape > 5.0]
        if not threshold_horizons.empty:
            first_breach = threshold_horizons.index[0]
            findings.append(
                f"Mean MAPE exceeds 5% at horizon "
                f"<strong>{first_breach} years</strong> "
                f"({horizon_mape[first_breach]:.2f}%)"
            )
        else:
            findings.append(
                "Mean MAPE remains below 5% across all evaluated horizons."
            )

    # Near-term vs long-term
    near = mape_df.loc[mape_df["horizon"] <= near_term_max, "value"]
    far = mape_df.loc[mape_df["horizon"] >= long_term_min, "value"]
    if not near.empty and not far.empty:
        findings.append(
            f"Near-term MAPE (horizon 1-{near_term_max}): "
            f"<strong>{near.mean():.2f}%</strong> | "
            f"Long-term MAPE (horizon {long_term_min}+): "
            f"<strong>{far.mean():.2f}%</strong>"
        )

    # Bias
    if sc.bias_calibration != 0:
        direction = "overprojection" if sc.bias_calibration > 0 else "underprojection"
        findings.append(
            f"Systematic bias: <strong>{abs(sc.bias_calibration):.2f}%</strong> "
            f"{direction} on average"
        )

    items = "\n".join(f"<li>{f}</li>" for f in findings)
    return f"""
    <div class="key-findings">
      <h3>Key Findings</h3>
      <ul>{items}</ul>
    </div>
    """


def _build_accuracy_section(
    accuracy_df: pd.DataFrame,
    encoded_figures: dict[str, str],
    near_term_max: int,
    long_term_min: int,
    county_groups: dict[str, list[str]],
    sentinel_counties: dict[str, str],
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Forecast Accuracy">',
        "<h2>Forecast Accuracy</h2>",
    ]

    if accuracy_df.empty:
        parts.append("<p>No accuracy diagnostics available.</p>")
        parts.append("</section>")
        return "\n".join(parts)

    # Summary table by horizon band
    parts.append("<h3>Accuracy by Horizon Band</h3>")
    parts.append(
        _build_horizon_band_table(accuracy_df, near_term_max, long_term_min)
    )

    # County group table
    if "geography_group" in accuracy_df.columns:
        parts.append("<h3>Accuracy by County Group</h3>")
        parts.append(_build_county_group_table(accuracy_df))

    # Sentinel county detail
    if sentinel_counties and "geography" in accuracy_df.columns:
        parts.append("<h3>Sentinel County Detail</h3>")
        parts.append(_build_sentinel_table(accuracy_df, sentinel_counties))

    # Embedded charts
    if "horizon_profile" in encoded_figures:
        parts.append("<h3>Horizon Profile</h3>")
        parts.append(_embed_image(encoded_figures["horizon_profile"], "MAPE by Horizon"))

    if "county_horizon_heatmap" in encoded_figures:
        parts.append("<h3>County-Horizon Heatmap</h3>")
        parts.append(
            _embed_image(encoded_figures["county_horizon_heatmap"], "County-Horizon Heatmap")
        )

    parts.append("</section>")
    return "\n".join(parts)


def _build_horizon_band_table(
    df: pd.DataFrame, near_term_max: int, long_term_min: int
) -> str:
    mape = df.loc[df["metric_name"] == "mape"]
    mae = df.loc[df["metric_name"] == "mae"]
    rmse = df.loc[df["metric_name"] == "rmse"]

    def _band_stats(metric_df: pd.DataFrame, label: str) -> dict[str, str]:
        near = metric_df.loc[metric_df["horizon"] <= near_term_max, "value"]
        far = metric_df.loc[metric_df["horizon"] >= long_term_min, "value"]
        overall = metric_df["value"]
        return {
            "Metric": label,
            f"Near-term (1-{near_term_max}yr)": f"{near.mean():.3f}" if not near.empty else "N/A",
            f"Long-term ({long_term_min}+yr)": f"{far.mean():.3f}" if not far.empty else "N/A",
            "Overall": f"{overall.mean():.3f}" if not overall.empty else "N/A",
        }

    rows = []
    if not mape.empty:
        rows.append(_band_stats(mape, "MAPE (%)"))
    if not mae.empty:
        rows.append(_band_stats(mae, "MAE"))
    if not rmse.empty:
        rows.append(_band_stats(rmse, "RMSE"))

    if not rows:
        return "<p>No accuracy metrics computed.</p>"

    return _render_table(rows, sortable=False)


def _build_county_group_table(df: pd.DataFrame) -> str:
    mape = df.loc[df["metric_name"] == "mape"]
    if mape.empty or "geography_group" not in mape.columns:
        return "<p>No county group data available.</p>"

    rows = []
    for group, grp in mape.groupby("geography_group"):
        rows.append({
            "County Group": str(group),
            "Mean MAPE (%)": f"{grp['value'].mean():.3f}",
            "Median MAPE (%)": f"{grp['value'].median():.3f}",
            "Max MAPE (%)": f"{grp['value'].max():.3f}",
            "N": str(len(grp)),
        })

    return _render_table(rows, sortable=True)


def _build_sentinel_table(
    df: pd.DataFrame, sentinel_counties: dict[str, str]
) -> str:
    mape = df.loc[df["metric_name"] == "mape"]
    mspe = df.loc[df["metric_name"] == "mean_signed_percentage_error"]

    rows = []
    for fips, name in sentinel_counties.items():
        county_mape = mape.loc[mape["geography"] == fips, "value"]
        county_mspe = mspe.loc[mspe["geography"] == fips, "value"]
        rows.append({
            "FIPS": fips,
            "County": name,
            "Mean MAPE (%)": f"{county_mape.mean():.3f}" if not county_mape.empty else "N/A",
            "Bias (MSPE %)": f"{county_mspe.mean():+.3f}" if not county_mspe.empty else "N/A",
            "Max MAPE (%)": f"{county_mape.max():.3f}" if not county_mape.empty else "N/A",
        })

    return _render_table(rows, sortable=True)


def _build_bias_section(
    accuracy_df: pd.DataFrame,
    encoded_figures: dict[str, str],
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Bias Analysis">',
        "<h2>Bias Analysis</h2>",
    ]

    mspe = accuracy_df.loc[
        accuracy_df["metric_name"] == "mean_signed_percentage_error"
    ] if not accuracy_df.empty else pd.DataFrame()

    if mspe.empty:
        parts.append("<p>No bias data available.</p>")
        parts.append("</section>")
        return "\n".join(parts)

    overall_bias = mspe["value"].mean()
    direction = "overprojection" if overall_bias > 0 else "underprojection"
    parts.append(
        f"<p>Overall bias direction: <strong>{direction}</strong> "
        f"(mean signed percentage error: {overall_bias:+.3f}%)</p>"
    )

    # County bias table
    if "geography" in mspe.columns:
        parts.append("<h3>Bias by County</h3>")
        county_bias = mspe.groupby("geography")["value"].mean().reset_index()
        county_bias["abs_value"] = county_bias["value"].abs()
        county_bias = county_bias.sort_values("abs_value", ascending=False)

        rows = []
        for _, row in county_bias.iterrows():
            bias_val = row["value"]
            rows.append({
                "Geography": str(row["geography"]),
                "Mean MSPE (%)": f"{bias_val:+.3f}",
                "Abs MSPE (%)": f"{row['abs_value']:.3f}",
                "Direction": "Over" if bias_val > 0 else "Under",
            })

        parts.append(_render_table(rows, sortable=True))

    if "bias_map" in encoded_figures:
        parts.append("<h3>Bias Map</h3>")
        parts.append(_embed_image(encoded_figures["bias_map"], "Bias by County"))

    parts.append("</section>")
    return "\n".join(parts)


def _build_realism_section(
    realism_df: pd.DataFrame,
    encoded_figures: dict[str, str],
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Structural Realism">',
        "<h2>Structural Realism</h2>",
    ]

    jsd_df = realism_df.loc[realism_df["metric_name"] == "jsd"]
    if jsd_df.empty:
        parts.append("<p>No realism diagnostics available.</p>")
        parts.append("</section>")
        return "\n".join(parts)

    overall_jsd = jsd_df["value"].mean()
    parts.append(
        f"<p>Mean Jensen-Shannon Divergence across all horizons and geographies: "
        f"<strong>{overall_jsd:.5f}</strong></p>"
    )

    # JSD by horizon
    if "horizon" in jsd_df.columns:
        parts.append("<h3>JSD by Horizon</h3>")
        horizon_jsd = jsd_df.groupby("horizon")["value"].agg(["mean", "min", "max"])
        rows = []
        for h, row in horizon_jsd.iterrows():
            rows.append({
                "Horizon": str(h),
                "Mean JSD": f"{row['mean']:.5f}",
                "Min JSD": f"{row['min']:.5f}",
                "Max JSD": f"{row['max']:.5f}",
            })
        parts.append(_render_table(rows, sortable=True))

    if "age_divergence" in encoded_figures:
        parts.append("<h3>Age Divergence Chart</h3>")
        parts.append(
            _embed_image(encoded_figures["age_divergence"], "Age Distribution Divergence")
        )

    parts.append("</section>")
    return "\n".join(parts)


def _build_comparison_section(
    comparison_df: pd.DataFrame,
    encoded_figures: dict[str, str],
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Model Comparison">',
        "<h2>Model Comparison</h2>",
    ]

    mape = comparison_df.loc[comparison_df["metric_name"] == "mape"]
    if mape.empty:
        parts.append("<p>No comparison data available.</p>")
        parts.append("</section>")
        return "\n".join(parts)

    # Side-by-side accuracy table
    has_delta = "delta_vs_baseline" in mape.columns
    model_summary = mape.groupby("model_name").agg(
        mean_mape=("value", "mean"),
        **({"mean_delta": ("delta_vs_baseline", "mean")} if has_delta else {}),
    ).reset_index()

    rows = []
    for _, row in model_summary.iterrows():
        r: dict[str, str] = {
            "Model": str(row["model_name"]),
            "Mean MAPE (%)": f"{row['mean_mape']:.3f}",
        }
        if has_delta:
            delta = row["mean_delta"]
            color = "cell-good" if delta < 0 else "cell-bad" if delta > 0 else ""
            r["Delta vs Baseline (pp)"] = f'<span class="{color}">{delta:+.3f}</span>'
        rows.append(r)

    parts.append(_render_table(rows, sortable=True, raw_html=True))

    # Comparison charts
    for fig_name in ("comparison_horizon", "stability_scatter"):
        if fig_name in encoded_figures:
            parts.append(_embed_image(encoded_figures[fig_name], fig_name))

    parts.append("</section>")
    return "\n".join(parts)


def _build_sensitivity_section(
    sensitivity: dict[str, Any],
    encoded_figures: dict[str, str],
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Sensitivity Analysis">',
        "<h2>Sensitivity Analysis</h2>",
    ]

    # Perturbation results
    perturbation = sensitivity.get("perturbation")
    if isinstance(perturbation, pd.DataFrame) and not perturbation.empty:
        parts.append("<h3>Perturbation Results</h3>")
        rows = []
        for _, row in perturbation.head(50).iterrows():
            r = {col: str(row[col]) for col in perturbation.columns[:6]}
            rows.append(r)
        parts.append(_render_table(rows, sortable=True))

    # Stability index
    stability = sensitivity.get("stability_index")
    if isinstance(stability, pd.DataFrame) and not stability.empty:
        parts.append("<h3>Stability Index</h3>")
        rows = []
        for _, row in stability.iterrows():
            r = {col: str(row[col]) for col in stability.columns}
            rows.append(r)
        parts.append(_render_table(rows, sortable=True))

    # Parameter sweeps
    sweeps = sensitivity.get("parameter_sweeps")
    if isinstance(sweeps, dict):
        for param_name, sweep_df in sweeps.items():
            fig_key = f"param_response_{param_name}"
            if fig_key in encoded_figures:
                parts.append(f"<h3>Parameter Response: {_esc(param_name)}</h3>")
                parts.append(_embed_image(encoded_figures[fig_key], param_name))

    parts.append("</section>")
    return "\n".join(parts)


def _build_appendix(
    accuracy_df: pd.DataFrame,
    realism_df: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    sid = _section_id()
    parts: list[str] = [
        f'<section id="{sid}" data-toc="Appendix">',
        "<h2>Appendix</h2>",
    ]

    # Full diagnostics table (collapsed)
    if not accuracy_df.empty:
        parts.append("""
        <details>
          <summary>Full Accuracy Diagnostics Table</summary>
          <div class="appendix-table-wrapper">
        """)
        display_cols = [
            c for c in accuracy_df.columns
            if c in (
                "run_id", "metric_name", "metric_group", "geography",
                "geography_group", "target", "horizon", "value", "model_name",
            )
        ]
        rows = []
        for _, row in accuracy_df[display_cols].iterrows():
            r = {}
            for col in display_cols:
                val = row[col]
                if isinstance(val, float):
                    r[col] = f"{val:.4f}"
                else:
                    r[col] = str(val)
            rows.append(r)
        parts.append(_render_table(rows, sortable=True))
        parts.append("</div></details>")

    # Configuration summary
    parts.append("""
    <details>
      <summary>Configuration Summary</summary>
      <div class="config-summary">
    """)

    horizons = config.get("horizons", [])
    parts.append(f"<p><strong>Horizons:</strong> {horizons}</p>")

    county_groups = config.get("county_groups", {})
    if county_groups:
        parts.append("<p><strong>County Groups:</strong></p><ul>")
        for group, fips_list in county_groups.items():
            parts.append(f"<li>{_esc(group)}: {fips_list}</li>")
        parts.append("</ul>")

    weights = config.get("scorecard_weights", {})
    if weights:
        parts.append("<p><strong>Scorecard Weights:</strong></p><ul>")
        for axis, w in weights.items():
            parts.append(f"<li>{_esc(axis)}: {w}</li>")
        parts.append("</ul>")

    parts.append("</div></details>")
    parts.append("</section>")
    return "\n".join(parts)


# ===================================================================
# TOC builder
# ===================================================================


def _build_toc(sections: list[str]) -> str:
    """Extract data-toc attributes from sections and build floating TOC."""
    items: list[str] = []
    for section_html in sections:
        # Find data-toc and id
        import re

        toc_match = re.search(r'data-toc="([^"]+)"', section_html)
        id_match = re.search(r'id="([^"]+)"', section_html)
        if toc_match and id_match:
            label = toc_match.group(1)
            section_id = id_match.group(1)
            items.append(
                f'<li><a href="#{section_id}">{_esc(label)}</a></li>'
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
# HTML primitives
# ===================================================================


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _embed_image(data_uri: str, alt: str) -> str:
    return f'<div class="chart-container"><img src="{data_uri}" alt="{_esc(alt)}" /></div>'


def _render_table(
    rows: list[dict[str, str]],
    *,
    sortable: bool = False,
    raw_html: bool = False,
) -> str:
    """Render a list of row dicts as an HTML table.

    Args:
        rows: List of dicts mapping column names to display values.
        sortable: Whether to add the ``sortable`` class for JS sorting.
        raw_html: If True, cell values are rendered as raw HTML (no escaping).
    """
    if not rows:
        return "<p>No data.</p>"

    cols = list(rows[0].keys())
    cls = "data-table sortable" if sortable else "data-table"
    header = "".join(f"<th>{_esc(c)}</th>" for c in cols)

    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for col in cols:
            val = row.get(col, "")
            cell_class = _cell_color_class(col, val)
            cell_content = val if raw_html else _esc(val)
            cls_attr = f' class="{cell_class}"' if cell_class else ""
            cells.append(f"<td{cls_attr}>{cell_content}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="table-wrapper">
      <table class="{cls}">
        <thead><tr>{header}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
    </div>
    """


def _cell_color_class(col_name: str, value: str) -> str:
    """Determine color class for a table cell based on column and value."""
    if "MAPE" in col_name or col_name in ("Mean JSD",):
        try:
            v = float(value.replace("%", "").strip())
        except (ValueError, AttributeError):
            return ""
        if "JSD" in col_name:
            if v < 0.01:
                return "cell-good"
            if v < 0.05:
                return "cell-warn"
            return "cell-bad"
        # MAPE thresholds
        if v < 3.0:
            return "cell-good"
        if v < 7.0:
            return "cell-warn"
        return "cell-bad"

    if col_name == "Direction":
        return "cell-bad" if value == "Over" else "cell-warn"

    return ""


def _error_color_class(value: float) -> str:
    """CSS class for error-like metrics (lower is better)."""
    if value < 0.03:
        return "score-good"
    if value < 0.07:
        return "score-ok"
    return "score-poor"


def _quality_color_class(value: float) -> str:
    """CSS class for quality-like metrics (higher is better, 0-1 scale)."""
    if value >= 0.7:
        return "score-good"
    if value >= 0.4:
        return "score-ok"
    return "score-poor"


# ===================================================================
# Full HTML wrapper
# ===================================================================


def _wrap_html(title: str, toc: str, body: str) -> str:
    """Wrap body content in a full HTML document with CSS and JS."""
    _reset_section_counter()
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

.toc-sidebar ul {
  list-style: none;
}

.toc-sidebar li {
  margin-bottom: 0.25rem;
}

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
section {
  margin-bottom: 2.5rem;
}

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

/* ---- Scorecard ---- */
.scorecard-container {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  align-items: flex-start;
  margin-bottom: 1.5rem;
}

.composite-score {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.5rem 2rem;
  text-align: center;
  min-width: 160px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.composite-label {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
  margin-bottom: 0.25rem;
}

.composite-value {
  font-size: 2.2rem;
  font-weight: 800;
}

.composite-note {
  font-size: 0.7rem;
  color: #94a3b8;
  margin-top: 0.25rem;
}

.scorecard-axes {
  flex: 1;
  min-width: 400px;
}

.scorecard-row {
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
}

.scorecard-label {
  width: 200px;
  font-size: 0.82rem;
  font-weight: 500;
  color: #475569;
}

.scorecard-bar-container {
  flex: 1;
  height: 18px;
  background: #e2e8f0;
  border-radius: 9px;
  overflow: hidden;
  margin: 0 0.75rem;
}

.scorecard-bar {
  height: 100%;
  border-radius: 9px;
  transition: width 0.3s ease;
}

.scorecard-value {
  width: 70px;
  text-align: right;
  font-size: 0.82rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: #334155;
}

.bias-indicator {
  background: #f1f5f9;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.85rem;
  color: #475569;
  width: 100%;
}

/* Score colors */
.score-good .composite-value,
.score-good { color: #059669; }
.score-good .scorecard-bar,
.scorecard-bar.score-good { background: #059669; }

.score-ok .composite-value,
.score-ok { color: #d97706; }
.score-ok .scorecard-bar,
.scorecard-bar.score-ok { background: #d97706; }

.score-poor .composite-value,
.score-poor { color: #dc2626; }
.score-poor .scorecard-bar,
.scorecard-bar.score-poor { background: #dc2626; }

/* ---- Key Findings ---- */
.key-findings {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.key-findings h3 {
  margin-top: 0;
  margin-bottom: 0.75rem;
}

.key-findings ul {
  margin-left: 1.25rem;
}

.key-findings li {
  margin-bottom: 0.4rem;
  color: #334155;
  font-size: 0.9rem;
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

.data-table.sortable th:hover {
  background: #1e3a5f;
}

.data-table.sortable th::after {
  content: " \\2195";
  font-size: 0.7rem;
  opacity: 0.5;
}

.data-table td {
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid #f1f5f9;
  font-variant-numeric: tabular-nums;
}

.data-table tbody tr:nth-child(even) {
  background: #f8fafc;
}

.data-table tbody tr:hover {
  background: #eef2ff;
}

/* Cell color coding */
.cell-good { background: #ecfdf5 !important; color: #065f46; font-weight: 600; }
.cell-warn { background: #fffbeb !important; color: #92400e; font-weight: 600; }
.cell-bad  { background: #fef2f2 !important; color: #991b1b; font-weight: 600; }

/* ---- Charts ---- */
.chart-container {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.chart-container img {
  max-width: 100%;
  height: auto;
}

/* ---- Appendix ---- */
details {
  margin-bottom: 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #ffffff;
}

summary {
  padding: 0.75rem 1rem;
  font-weight: 600;
  cursor: pointer;
  color: #16213e;
  background: #f1f5f9;
  border-radius: 6px;
}

summary:hover {
  background: #e2e8f0;
}

details[open] summary {
  border-bottom: 1px solid #e2e8f0;
  border-radius: 6px 6px 0 0;
}

.appendix-table-wrapper {
  max-height: 600px;
  overflow-y: auto;
  padding: 0.5rem;
}

.config-summary {
  padding: 1rem;
  font-size: 0.85rem;
  color: #475569;
}

.config-summary p { margin-bottom: 0.5rem; }
.config-summary ul { margin-left: 1.25rem; margin-bottom: 0.75rem; }
.config-summary li { margin-bottom: 0.2rem; }

/* ---- Print Styles ---- */
@media print {
  body { background: #ffffff; font-size: 11pt; }

  .toc-sidebar { display: none; }

  .report-main {
    margin-left: 0;
    padding: 0;
    max-width: 100%;
  }

  .header-section {
    background: #16213e !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  section { page-break-inside: avoid; }

  .chart-container { page-break-inside: avoid; }

  .data-table th {
    background: #16213e !important;
    color: #ffffff !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  details { break-inside: avoid; }

  .cell-good, .cell-warn, .cell-bad {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}

/* ---- Responsive ---- */
@media (max-width: 900px) {
  .toc-sidebar { display: none; }
  .report-main { margin-left: 0; padding: 1rem; }
  .scorecard-axes { min-width: 100%; }
  .scorecard-label { width: 140px; }
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

    // Reset other headers
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
