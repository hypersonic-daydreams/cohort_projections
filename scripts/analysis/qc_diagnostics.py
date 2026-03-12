#!/usr/bin/env python3
"""Quality control diagnostics for walk-forward validation of population projections.

Created: 2026-03-03
Author: Claude Code (automated)
SOP-002 compliant: Yes

Purpose
-------
Compute comprehensive QC diagnostics from the walk-forward validation results
produced by ``walk_forward_validation.py``.  Analyses include:

1. **Systematic bias** by county type, horizon, and origin year.
2. **Residual analysis**: autocorrelation, heteroscedasticity, outlier detection,
   error-normality tests.
3. **Structural break detection**: pre-boom vs post-boom accuracy, paired
   method-comparison tests, regime identification.
4. **County-level report cards**: letter grades (A-D) per county per method.
5. **Interactive HTML report** with Plotly visualisations and sortable tables.

Inputs
------
- data/analysis/walk_forward/annual_county_detail.csv — county-level errors at
  every origin x method x validation_year (4 876 rows, 10 columns).
- data/analysis/walk_forward/annual_state_results.csv — state-level errors.
- data/analysis/walk_forward/annual_method_comparison.csv — side-by-side
  comparison by horizon.

Outputs
-------
- data/analysis/walk_forward/bias_analysis.csv
- data/analysis/walk_forward/residual_diagnostics.csv
- data/analysis/walk_forward/outlier_flags.csv
- data/analysis/walk_forward/county_report_cards.csv
- data/analysis/walk_forward/qc_diagnostics_report.html

Usage
-----
    python scripts/analysis/qc_diagnostics.py
"""

from __future__ import annotations

import html as html_lib
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "walk_forward"
OUTPUT_DIR = INPUT_DIR  # outputs go to the same directory

# ---------------------------------------------------------------------------
# County category definitions
# (Mirrors walk_forward_validation.py constants plus reservation/urban/rural)
# ---------------------------------------------------------------------------
BAKKEN_FIPS = {"38105", "38053", "38061", "38025", "38089"}
COLLEGE_FIPS = {"38017", "38035", "38101", "38015"}
RESERVATION_FIPS = {"38005", "38085", "38079"}
URBAN_FIPS = {"38017", "38015", "38035", "38101"}  # same as college

# All ND county FIPS (53 counties, 38001-38105 odd numbers)
ALL_ND_FIPS = {f"38{i:03d}" for i in range(1, 106, 2)}


def assign_county_category(fips: str) -> str:
    """Return the primary category for a county FIPS code.

    Priority order (a county gets its *first* matching label):
    Bakken > Reservation > Urban/College > Rural.
    """
    if fips in BAKKEN_FIPS:
        return "Bakken"
    if fips in RESERVATION_FIPS:
        return "Reservation"
    if fips in URBAN_FIPS:
        return "Urban/College"
    return "Rural"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_county_detail() -> pd.DataFrame:
    """Load annual county-level walk-forward results."""
    df = pd.read_csv(INPUT_DIR / "annual_county_detail.csv")
    # Ensure FIPS is zero-padded 5-char string
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["category"] = df["county_fips"].map(assign_county_category)
    return df


def load_state_results() -> pd.DataFrame:
    """Load annual state-level walk-forward results."""
    return pd.read_csv(INPUT_DIR / "annual_state_results.csv")


def load_method_comparison() -> pd.DataFrame:
    """Load annual method comparison summary."""
    return pd.read_csv(INPUT_DIR / "annual_method_comparison.csv")


# ===================================================================
# 1. SYSTEMATIC BIAS ANALYSIS
# ===================================================================


def compute_bias_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean signed error (pct_error) by category x horizon x method.

    Also computes by origin_year for temporal-bias analysis.
    """
    rows: list[dict] = []

    # --- By category x horizon x method ---
    for (cat, horizon, method), grp in df.groupby(
        ["category", "horizon", "method"]
    ):
        rows.append(
            {
                "category": cat,
                "horizon": horizon,
                "method": method,
                "origin_year": "all",
                "mean_signed_pct_error": grp["pct_error"].mean(),
                "median_signed_pct_error": grp["pct_error"].median(),
                "std_pct_error": grp["pct_error"].std(),
                "n_counties": grp["county_fips"].nunique(),
                "n_obs": len(grp),
            }
        )

    # --- By category x horizon x method x origin (temporal bias) ---
    for (cat, horizon, method, origin), grp in df.groupby(
        ["category", "horizon", "method", "origin_year"]
    ):
        rows.append(
            {
                "category": cat,
                "horizon": horizon,
                "method": method,
                "origin_year": int(origin),
                "mean_signed_pct_error": grp["pct_error"].mean(),
                "median_signed_pct_error": grp["pct_error"].median(),
                "std_pct_error": grp["pct_error"].std(),
                "n_counties": grp["county_fips"].nunique(),
                "n_obs": len(grp),
            }
        )

    bias_df = pd.DataFrame(rows)
    return bias_df


# ===================================================================
# 2. RESIDUAL ANALYSIS
# ===================================================================


def compute_residual_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute autocorrelation, heteroscedasticity tests, and normality tests.

    Returns one row per method x horizon bucket with test statistics.
    """
    # Horizon buckets for aggregation
    def _horizon_bucket(h: int) -> str:
        if h <= 5:
            return "1-5yr"
        elif h <= 10:
            return "6-10yr"
        elif h <= 15:
            return "11-15yr"
        else:
            return "16-19yr"

    df = df.copy()
    df["horizon_bucket"] = df["horizon"].apply(_horizon_bucket)

    rows: list[dict] = []

    for (method, bucket), grp in df.groupby(["method", "horizon_bucket"]):
        errors = grp["pct_error"].dropna().values

        # --- Autocorrelation ---
        # For each county x origin, sort by horizon and compute lag-1 corr
        autocorrs: list[float] = []
        for (_county, _origin), sub in grp.groupby(["county_fips", "origin_year"]):
            sub_sorted = sub.sort_values("horizon")
            errs = sub_sorted["pct_error"].values
            if len(errs) >= 3:
                ac = np.corrcoef(errs[:-1], errs[1:])[0, 1]
                if np.isfinite(ac):
                    autocorrs.append(ac)
        mean_autocorr = float(np.mean(autocorrs)) if autocorrs else np.nan

        # --- Heteroscedasticity (Breusch-Pagan-like) ---
        # Regress squared errors on county population (actual) to detect
        # whether error variance changes with county size
        sq_errors = grp["pct_error"].values ** 2
        county_sizes = grp["actual"].values
        if len(sq_errors) > 10 and np.std(county_sizes) > 0:
            slope, intercept, r_value, p_value_bp, std_err = stats.linregress(
                county_sizes, sq_errors
            )
            het_r2 = r_value**2
            het_p = p_value_bp
        else:
            het_r2 = np.nan
            het_p = np.nan

        # --- Normality (Shapiro-Wilk) ---
        # Shapiro-Wilk works best for n < 5000; sample if larger
        if len(errors) >= 8:
            sample = errors if len(errors) <= 5000 else np.random.default_rng(42).choice(
                errors, 5000, replace=False
            )
            sw_stat, sw_p = stats.shapiro(sample)
        else:
            sw_stat, sw_p = np.nan, np.nan

        rows.append(
            {
                "method": method,
                "horizon_bucket": bucket,
                "n_obs": len(errors),
                "mean_autocorr_lag1": round(mean_autocorr, 4),
                "n_autocorr_series": len(autocorrs),
                "het_r2_vs_pop_size": round(het_r2, 6) if np.isfinite(het_r2) else np.nan,
                "het_p_value": round(het_p, 6) if np.isfinite(het_p) else np.nan,
                "het_significant": het_p < 0.05 if np.isfinite(het_p) else None,
                "shapiro_w": round(sw_stat, 4) if np.isfinite(sw_stat) else np.nan,
                "shapiro_p": round(sw_p, 6) if np.isfinite(sw_p) else np.nan,
                "normal_at_05": sw_p >= 0.05 if np.isfinite(sw_p) else None,
                "error_mean": round(float(np.mean(errors)), 4),
                "error_std": round(float(np.std(errors)), 4),
                "error_skew": round(float(stats.skew(errors)), 4) if len(errors) > 2 else np.nan,
                "error_kurtosis": round(float(stats.kurtosis(errors)), 4)
                if len(errors) > 3
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag county x origin x year combos where |error| > 2 SD from horizon mean.

    Returns a DataFrame of flagged rows with z-scores.
    """
    flags: list[pd.DataFrame] = []

    for (method, horizon), grp in df.groupby(["method", "horizon"]):
        mean_err = grp["pct_error"].mean()
        std_err = grp["pct_error"].std()
        if std_err == 0 or np.isnan(std_err):
            continue
        grp = grp.copy()
        grp["z_score"] = (grp["pct_error"] - mean_err) / std_err
        grp["horizon_mean_error"] = mean_err
        grp["horizon_std_error"] = std_err
        outliers = grp[grp["z_score"].abs() > 2.0]
        if len(outliers) > 0:
            flags.append(outliers)

    if flags:
        result = pd.concat(flags, ignore_index=True)
        result = result.sort_values(["method", "horizon", "z_score"], ascending=[True, True, False])
        return result
    return pd.DataFrame()


# ===================================================================
# 3. STRUCTURAL BREAK DETECTION
# ===================================================================


def compute_structural_breaks(df: pd.DataFrame) -> dict:
    """Analyse structural breaks: pre/post boom, method improvement, regimes.

    Returns a dict of DataFrames keyed by analysis name.
    """
    results: dict[str, pd.DataFrame] = {}

    # --- Pre-boom vs post-boom ---
    # Origins 2005 = pre-boom; 2015, 2020 = post-boom; 2010 = transition
    df = df.copy()
    df["era"] = df["origin_year"].map(
        {2005: "pre_boom", 2010: "transition", 2015: "post_boom", 2020: "post_boom"}
    )

    era_rows: list[dict] = []
    for (method, era, cat), grp in df.groupby(["method", "era", "category"]):
        era_rows.append(
            {
                "method": method,
                "era": era,
                "category": cat,
                "mean_abs_pct_error": grp["pct_error"].abs().mean(),
                "mean_signed_pct_error": grp["pct_error"].mean(),
                "n_obs": len(grp),
            }
        )
    results["pre_vs_post_boom"] = pd.DataFrame(era_rows)

    # --- Method improvement: paired tests ---
    # For each county x origin x horizon, pair SDC and M2026 errors
    paired_rows: list[dict] = []
    sdc = df[df["method"] == "sdc_2024"].set_index(
        ["county_fips", "origin_year", "horizon"]
    )["pct_error"]
    m26 = df[df["method"] == "m2026"].set_index(
        ["county_fips", "origin_year", "horizon"]
    )["pct_error"]
    common_idx = sdc.index.intersection(m26.index)

    if len(common_idx) > 10:
        sdc_vals = sdc.loc[common_idx].values
        m26_vals = m26.loc[common_idx].values
        abs_sdc = np.abs(sdc_vals)
        abs_m26 = np.abs(m26_vals)

        # Paired t-test on absolute errors
        t_stat, t_p = stats.ttest_rel(abs_sdc, abs_m26)
        # Wilcoxon signed-rank on absolute errors
        try:
            w_stat, w_p = stats.wilcoxon(abs_sdc, abs_m26, alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        paired_rows.append(
            {
                "comparison": "overall",
                "n_pairs": len(common_idx),
                "sdc_mean_abs_error": round(float(abs_sdc.mean()), 4),
                "m2026_mean_abs_error": round(float(abs_m26.mean()), 4),
                "diff_mean": round(float((abs_sdc - abs_m26).mean()), 4),
                "paired_t_stat": round(float(t_stat), 4),
                "paired_t_p": round(float(t_p), 6),
                "wilcoxon_stat": round(float(w_stat), 2) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_p": round(float(w_p), 6) if np.isfinite(w_p) else np.nan,
                "sdc_better": "Yes" if abs_sdc.mean() < abs_m26.mean() else "No",
            }
        )

        # Also by horizon bucket
        for bucket_label, h_range in [
            ("1-5yr", (1, 5)),
            ("6-10yr", (6, 10)),
            ("11-15yr", (11, 15)),
            ("16-19yr", (16, 19)),
        ]:
            mask = np.array(
                [h_range[0] <= idx[2] <= h_range[1] for idx in common_idx]
            )
            if mask.sum() > 5:
                s = abs_sdc[mask]
                m = abs_m26[mask]
                t_s, t_pp = stats.ttest_rel(s, m)
                try:
                    w_s, w_pp = stats.wilcoxon(s, m, alternative="two-sided")
                except ValueError:
                    w_s, w_pp = np.nan, np.nan
                paired_rows.append(
                    {
                        "comparison": bucket_label,
                        "n_pairs": int(mask.sum()),
                        "sdc_mean_abs_error": round(float(s.mean()), 4),
                        "m2026_mean_abs_error": round(float(m.mean()), 4),
                        "diff_mean": round(float((s - m).mean()), 4),
                        "paired_t_stat": round(float(t_s), 4),
                        "paired_t_p": round(float(t_pp), 6),
                        "wilcoxon_stat": round(float(w_s), 2) if np.isfinite(w_s) else np.nan,
                        "wilcoxon_p": round(float(w_pp), 6) if np.isfinite(w_pp) else np.nan,
                        "sdc_better": "Yes" if s.mean() < m.mean() else "No",
                    }
                )
    results["paired_method_tests"] = pd.DataFrame(paired_rows)

    # --- Regime analysis: which method dominates in each calendar year ---
    regime_rows: list[dict] = []
    for val_year in sorted(df["validation_year"].unique()):
        yr_data = df[df["validation_year"] == val_year]
        for method in ["sdc_2024", "m2026"]:
            m_data = yr_data[yr_data["method"] == method]
            regime_rows.append(
                {
                    "validation_year": int(val_year),
                    "method": method,
                    "mean_abs_pct_error": round(m_data["pct_error"].abs().mean(), 4),
                    "mean_signed_pct_error": round(m_data["pct_error"].mean(), 4),
                    "n_obs": len(m_data),
                }
            )
    regime_df = pd.DataFrame(regime_rows)
    if not regime_df.empty:
        # Pivot to find winner per year
        pivot = regime_df.pivot_table(
            index="validation_year",
            columns="method",
            values="mean_abs_pct_error",
        )
        if "sdc_2024" in pivot.columns and "m2026" in pivot.columns:
            pivot["winner"] = pivot.apply(
                lambda r: "m2026"
                if r.get("m2026", np.inf) < r.get("sdc_2024", np.inf)
                else "sdc_2024",
                axis=1,
            )
            regime_df = regime_df.merge(
                pivot[["winner"]].reset_index(), on="validation_year", how="left"
            )
    results["regime_analysis"] = regime_df

    return results


# ===================================================================
# 4. COUNTY REPORT CARDS
# ===================================================================


def compute_county_report_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Compute QC grades (A-D) per county per method.

    Grading scale (based on mean absolute percent error):
      A: < 3% MAPE  (reliable)
      B: 3-7% MAPE  (moderate)
      C: 7-15% MAPE (concerning)
      D: > 15% MAPE (poor)
    """

    def _grade(mape: float) -> str:
        if mape < 3:
            return "A"
        elif mape < 7:
            return "B"
        elif mape < 15:
            return "C"
        return "D"

    rows: list[dict] = []
    for (county, method), grp in df.groupby(["county_fips", "method"]):
        abs_errors = grp["pct_error"].abs()
        signed_errors = grp["pct_error"]
        mape = abs_errors.mean()
        county_name = grp["county_name"].iloc[0]

        rows.append(
            {
                "county_fips": county,
                "county_name": county_name,
                "method": method,
                "category": grp["category"].iloc[0],
                "mape": round(mape, 2),
                "mean_signed_error": round(signed_errors.mean(), 2),
                "median_abs_error": round(abs_errors.median(), 2),
                "std_error": round(signed_errors.std(), 2),
                "worst_case_error": round(
                    float(signed_errors.loc[abs_errors.idxmax()]), 2
                ),
                "worst_case_abs_error": round(float(abs_errors.max()), 2),
                "n_validations": len(grp),
                "bias_direction": "over" if signed_errors.mean() > 0 else "under",
                "grade": _grade(mape),
            }
        )

    rc = pd.DataFrame(rows).sort_values(["method", "mape"])
    return rc


# ===================================================================
# 5. AUTOCORRELATION COMPUTATION (for plotting)
# ===================================================================


def compute_autocorrelation_by_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average autocorrelation at each lag (1..max) by method.

    For each county x origin series, compute the correlation at lag k
    between error[t] and error[t+k].  Then average across all series.
    """
    max_lag = 10
    rows: list[dict] = []

    for method in df["method"].unique():
        mdf = df[df["method"] == method]
        for lag in range(1, max_lag + 1):
            corrs: list[float] = []
            for (_county, _origin), sub in mdf.groupby(["county_fips", "origin_year"]):
                sub_sorted = sub.sort_values("horizon")
                errs = sub_sorted["pct_error"].values
                if len(errs) > lag + 1:
                    c = np.corrcoef(errs[:-lag], errs[lag:])[0, 1]
                    if np.isfinite(c):
                        corrs.append(c)
            if corrs:
                rows.append(
                    {
                        "method": method,
                        "lag": lag,
                        "mean_autocorrelation": round(float(np.mean(corrs)), 4),
                        "std_autocorrelation": round(float(np.std(corrs)), 4),
                        "n_series": len(corrs),
                    }
                )

    return pd.DataFrame(rows)


# ===================================================================
# 6. INTERACTIVE HTML REPORT
# ===================================================================

# Plotly CDN URL for self-contained HTML
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _build_html_report(
    bias_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    report_cards: pd.DataFrame,
    county_df: pd.DataFrame,
    structural: dict,
    acf_df: pd.DataFrame,
) -> str:
    """Build a self-contained HTML report with Plotly charts and sortable tables."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ---- Helper: make a figure and return its HTML div ----
    def _fig_to_div(fig: go.Figure, div_id: str) -> str:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=div_id,
        )

    # ================================================================
    # Chart 1: Bias Heatmap (category x horizon, mean signed error)
    # ================================================================
    heatmap_divs: list[str] = []
    for method in ["sdc_2024", "m2026"]:
        sub = bias_df[
            (bias_df["method"] == method) & (bias_df["origin_year"] == "all")
        ]
        pivot = sub.pivot_table(
            index="category", columns="horizon", values="mean_signed_pct_error"
        )
        # Ensure consistent ordering
        cat_order = ["Bakken", "Urban/College", "Reservation", "Rural"]
        pivot = pivot.reindex([c for c in cat_order if c in pivot.index])

        method_label = "SDC 2024" if method == "sdc_2024" else "M2026"
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=list(pivot.index),
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(pivot.values, 1),
                texttemplate="%{text}%",
                colorbar=dict(title="Mean Signed<br>Error (%)"),
            )
        )
        fig.update_layout(
            title=f"Systematic Bias: {method_label}",
            xaxis_title="Projection Horizon (years)",
            yaxis_title="County Category",
            height=350,
            margin=dict(l=120, r=60, t=50, b=50),
        )
        heatmap_divs.append(_fig_to_div(fig, f"heatmap_{method}"))

    # ================================================================
    # Chart 2: County Report Card Table (interactive, sortable)
    # ================================================================
    # Handled with JavaScript table below

    # ================================================================
    # Chart 3: Outlier Scatter Plot
    # ================================================================
    outlier_scatter_div = ""
    if not outlier_df.empty:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["SDC 2024", "M2026"])
        for i, method in enumerate(["sdc_2024", "m2026"], 1):
            # Background: all points
            all_pts = county_df[county_df["method"] == method]
            fig.add_trace(
                go.Scatter(
                    x=all_pts["horizon"],
                    y=all_pts["pct_error"],
                    mode="markers",
                    marker=dict(color="lightgray", size=3, opacity=0.3),
                    name="All observations",
                    showlegend=(i == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=i,
            )
            # Outliers
            out = outlier_df[outlier_df["method"] == method]
            if not out.empty:
                fig.add_trace(
                    go.Scatter(
                        x=out["horizon"],
                        y=out["pct_error"],
                        mode="markers",
                        marker=dict(
                            color=out["z_score"],
                            colorscale="RdYlGn_r",
                            size=8,
                            line=dict(width=1, color="black"),
                            colorbar=dict(title="Z-score") if i == 2 else None,
                        ),
                        text=out.apply(
                            lambda r: (
                                f"{r['county_name']}<br>"
                                f"Origin: {r['origin_year']}<br>"
                                f"Error: {r['pct_error']:.1f}%<br>"
                                f"Z: {r['z_score']:.2f}"
                            ),
                            axis=1,
                        ),
                        hoverinfo="text",
                        name="Outliers (|z|>2)",
                        showlegend=(i == 1),
                    ),
                    row=1,
                    col=i,
                )
            fig.update_xaxes(title_text="Horizon (years)", row=1, col=i)
            fig.update_yaxes(title_text="% Error", row=1, col=i)

        fig.update_layout(
            title="Outlier Detection: County x Origin Combinations with |z| > 2 SD",
            height=450,
            margin=dict(l=60, r=60, t=70, b=50),
        )
        outlier_scatter_div = _fig_to_div(fig, "outlier_scatter")

    # ================================================================
    # Chart 4: Box Plots — Error Distribution by County Category
    # ================================================================
    fig = go.Figure()
    cat_order = ["Bakken", "Urban/College", "Reservation", "Rural"]
    colors = {"Bakken": "#e74c3c", "Urban/College": "#3498db",
              "Reservation": "#e67e22", "Rural": "#2ecc71"}
    for method in ["sdc_2024", "m2026"]:
        mdf = county_df[county_df["method"] == method]
        for cat in cat_order:
            cdf = mdf[mdf["category"] == cat]
            if cdf.empty:
                continue
            method_label = "SDC" if method == "sdc_2024" else "M2026"
            fig.add_trace(
                go.Box(
                    y=cdf["pct_error"],
                    name=f"{cat} ({method_label})",
                    marker_color=colors.get(cat, "#95a5a6"),
                    opacity=0.8 if method == "sdc_2024" else 0.5,
                    legendgroup=cat,
                    showlegend=True,
                )
            )
    fig.update_layout(
        title="Error Distribution by County Category and Method",
        yaxis_title="Signed % Error",
        height=500,
        boxmode="group",
        margin=dict(l=60, r=60, t=50, b=50),
    )
    boxplot_div = _fig_to_div(fig, "category_boxplot")

    # ================================================================
    # Chart 5: Paired Comparison Scatter (SDC error vs M2026 error)
    # ================================================================
    sdc_data = county_df[county_df["method"] == "sdc_2024"].set_index(
        ["county_fips", "origin_year", "horizon"]
    )
    m26_data = county_df[county_df["method"] == "m2026"].set_index(
        ["county_fips", "origin_year", "horizon"]
    )
    common = sdc_data.index.intersection(m26_data.index)

    paired_fig = go.Figure()
    if len(common) > 0:
        sdc_errs = sdc_data.loc[common, "pct_error"].abs().values
        m26_errs = m26_data.loc[common, "pct_error"].abs().values
        horizons = np.array([idx[2] for idx in common])

        paired_fig.add_trace(
            go.Scatter(
                x=sdc_errs,
                y=m26_errs,
                mode="markers",
                marker=dict(
                    color=horizons,
                    colorscale="Viridis",
                    size=5,
                    opacity=0.6,
                    colorbar=dict(title="Horizon"),
                ),
                text=[
                    f"FIPS: {idx[0]}<br>Origin: {idx[1]}<br>Horizon: {idx[2]}"
                    for idx in common
                ],
                hoverinfo="text+x+y",
                name="County x Origin x Horizon",
            )
        )
        # 45-degree reference line
        max_val = max(sdc_errs.max(), m26_errs.max()) * 1.05
        paired_fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Equal error line",
                showlegend=True,
            )
        )
    paired_fig.update_layout(
        title="Paired Method Comparison: |SDC Error| vs |M2026 Error|",
        xaxis_title="|SDC 2024 Error| (%)",
        yaxis_title="|M2026 Error| (%)",
        height=500,
        margin=dict(l=60, r=60, t=50, b=60),
    )
    # Add annotation for interpretation
    n_m2026_better = int((sdc_errs > m26_errs).sum()) if len(common) > 0 else 0
    n_sdc_better = int((m26_errs > sdc_errs).sum()) if len(common) > 0 else 0
    paired_fig.add_annotation(
        text=(
            f"Points below line: M2026 better ({n_m2026_better})<br>"
            f"Points above line: SDC better ({n_sdc_better})"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)",
    )
    paired_div = _fig_to_div(paired_fig, "paired_scatter")

    # ================================================================
    # Chart 6: Temporal Stability (error by origin year per category)
    # ================================================================
    temporal_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Bakken", "Urban/College", "Reservation", "Rural"],
    )
    cat_positions = {
        "Bakken": (1, 1),
        "Urban/College": (1, 2),
        "Reservation": (2, 1),
        "Rural": (2, 2),
    }
    method_colors = {"sdc_2024": "#2c3e50", "m2026": "#e74c3c"}
    method_dashes = {"sdc_2024": "solid", "m2026": "dash"}

    # Get temporal bias data
    temporal_bias = bias_df[bias_df["origin_year"] != "all"].copy()
    temporal_bias["origin_year"] = temporal_bias["origin_year"].astype(int)

    for cat, (row, col) in cat_positions.items():
        for method in ["sdc_2024", "m2026"]:
            sub = temporal_bias[
                (temporal_bias["category"] == cat) & (temporal_bias["method"] == method)
            ]
            if sub.empty:
                continue
            # Average across horizons for each origin year
            origin_avg = sub.groupby("origin_year")["mean_signed_pct_error"].mean()
            method_label = "SDC" if method == "sdc_2024" else "M2026"
            temporal_fig.add_trace(
                go.Scatter(
                    x=origin_avg.index.tolist(),
                    y=origin_avg.values.tolist(),
                    mode="lines+markers",
                    name=method_label,
                    line=dict(
                        color=method_colors[method],
                        dash=method_dashes[method],
                    ),
                    showlegend=(cat == "Bakken"),
                    legendgroup=method,
                ),
                row=row,
                col=col,
            )
        temporal_fig.update_xaxes(title_text="Origin Year", row=row, col=col)
        temporal_fig.update_yaxes(title_text="Mean Signed Error (%)", row=row, col=col)

    temporal_fig.update_layout(
        title="Temporal Stability: How Bias Changes by Origin Year",
        height=600,
        margin=dict(l=60, r=60, t=70, b=50),
    )
    temporal_div = _fig_to_div(temporal_fig, "temporal_stability")

    # ================================================================
    # Chart 7: Autocorrelation Plot
    # ================================================================
    acf_fig = go.Figure()
    for method in acf_df["method"].unique():
        mdf = acf_df[acf_df["method"] == method]
        method_label = "SDC 2024" if method == "sdc_2024" else "M2026"
        acf_fig.add_trace(
            go.Bar(
                x=mdf["lag"],
                y=mdf["mean_autocorrelation"],
                name=method_label,
                error_y=dict(
                    type="data",
                    array=(mdf["std_autocorrelation"] / np.sqrt(mdf["n_series"])).tolist(),
                    visible=True,
                ),
                opacity=0.7,
            )
        )
    # Add significance bounds (approximate: +/- 2/sqrt(n))
    if not acf_df.empty:
        avg_n = acf_df["n_series"].mean()
        bound = 2.0 / np.sqrt(avg_n)
        acf_fig.add_hline(y=bound, line_dash="dash", line_color="gray", opacity=0.5)
        acf_fig.add_hline(y=-bound, line_dash="dash", line_color="gray", opacity=0.5)
        acf_fig.add_hline(y=0, line_color="black", line_width=1)

    acf_fig.update_layout(
        title="Error Autocorrelation by Lag (averaged across county x origin series)",
        xaxis_title="Lag (years)",
        yaxis_title="Mean Autocorrelation",
        height=400,
        barmode="group",
        margin=dict(l=60, r=60, t=50, b=50),
    )
    acf_div = _fig_to_div(acf_fig, "acf_plot")

    # ================================================================
    # Build the sortable report-card table HTML
    # ================================================================
    report_card_html = _build_report_card_table(report_cards)

    # ================================================================
    # Summary statistics for header
    # ================================================================
    n_counties = county_df["county_fips"].nunique()
    n_origins = county_df["origin_year"].nunique()
    n_obs = len(county_df)
    n_outliers = len(outlier_df) if not outlier_df.empty else 0

    # Grade distribution — extract values for use in HTML template
    grade_dist: dict[str, dict] = {}
    for method in ["sdc_2024", "m2026"]:
        mrc = report_cards[report_cards["method"] == method]
        grade_dist[method] = mrc["grade"].value_counts().to_dict()

    m2026_grade_a = grade_dist.get("m2026", {}).get("A", 0)
    m2026_grade_d = grade_dist.get("m2026", {}).get("D", 0)

    # Paired test result
    paired_tests = structural.get("paired_method_tests", pd.DataFrame())
    overall_test = ""
    if not paired_tests.empty:
        row = paired_tests[paired_tests["comparison"] == "overall"]
        if not row.empty:
            r = row.iloc[0]
            better = "M2026" if r["sdc_better"] == "No" else "SDC 2024"
            sig = "significant" if r["paired_t_p"] < 0.05 else "not significant"
            overall_test = (
                f"{better} has lower mean absolute error "
                f"(paired t-test p={r['paired_t_p']:.4f}, {sig}). "
                f"SDC mean |error|: {r['sdc_mean_abs_error']:.2f}%, "
                f"M2026 mean |error|: {r['m2026_mean_abs_error']:.2f}%."
            )

    # ================================================================
    # Assemble full HTML
    # ================================================================
    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>QC Diagnostics: Walk-Forward Validation</title>
        <script src="{PLOTLY_CDN}"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #f5f6fa;
                color: #2c3e50;
                line-height: 1.6;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            h1 {{
                text-align: center;
                padding: 30px 0 10px;
                color: #2c3e50;
                font-size: 2em;
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
                font-size: 1.1em;
            }}
            .summary-cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                text-align: center;
            }}
            .card .number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
            .card .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; }}
            .section {{
                background: white;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            .section h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .section h3 {{ color: #34495e; margin: 15px 0 10px; }}
            .insight {{
                background: #eef7ff;
                border-left: 4px solid #3498db;
                padding: 12px 16px;
                margin: 15px 0;
                border-radius: 0 4px 4px 0;
            }}
            .warning {{
                background: #fff8e1;
                border-left: 4px solid #f39c12;
                padding: 12px 16px;
                margin: 15px 0;
                border-radius: 0 4px 4px 0;
            }}
            .grade-A {{ color: #27ae60; font-weight: bold; }}
            .grade-B {{ color: #2980b9; font-weight: bold; }}
            .grade-C {{ color: #f39c12; font-weight: bold; }}
            .grade-D {{ color: #e74c3c; font-weight: bold; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 0.9em;
            }}
            th, td {{
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
            }}
            th {{
                background: #34495e;
                color: white;
                cursor: pointer;
                user-select: none;
                position: sticky;
                top: 0;
            }}
            th:hover {{ background: #2c3e50; }}
            tr:hover {{ background: #f8f9fa; }}
            .sort-arrow {{ margin-left: 4px; font-size: 0.8em; }}
            .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
            .nav {{
                position: sticky;
                top: 0;
                background: #2c3e50;
                padding: 10px 20px;
                z-index: 1000;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
            }}
            .nav a {{
                color: white;
                text-decoration: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-size: 0.9em;
            }}
            .nav a:hover {{ background: #3498db; }}
            .method-toggle {{
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }}
            .method-toggle button {{
                padding: 6px 16px;
                border: 2px solid #3498db;
                background: white;
                color: #3498db;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9em;
            }}
            .method-toggle button.active {{
                background: #3498db;
                color: white;
            }}
            .table-wrapper {{
                max-height: 600px;
                overflow-y: auto;
                border: 1px solid #ecf0f1;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
    <nav class="nav">
        <a href="#summary">Summary</a>
        <a href="#bias">Bias Analysis</a>
        <a href="#residuals">Residual Diagnostics</a>
        <a href="#structural">Structural Breaks</a>
        <a href="#report-cards">County Report Cards</a>
        <a href="#outliers">Outliers</a>
    </nav>
    <div class="container">
    <h1>QC Diagnostics: Walk-Forward Validation</h1>
    <p class="subtitle">
        North Dakota County Population Projections &mdash; SDC 2024 vs M2026
    </p>

    <!-- Summary Cards -->
    <div id="summary" class="summary-cards">
        <div class="card">
            <div class="number">{n_counties}</div>
            <div class="label">Counties Evaluated</div>
        </div>
        <div class="card">
            <div class="number">{n_origins}</div>
            <div class="label">Origin Years</div>
        </div>
        <div class="card">
            <div class="number">{n_obs:,}</div>
            <div class="label">Total Observations</div>
        </div>
        <div class="card">
            <div class="number">{n_outliers}</div>
            <div class="label">Outlier Flags</div>
        </div>
        <div class="card">
            <div class="number">{m2026_grade_a}</div>
            <div class="label">Counties Grade A (M2026)</div>
        </div>
        <div class="card">
            <div class="number">{m2026_grade_d}</div>
            <div class="label">Counties Grade D (M2026)</div>
        </div>
    </div>

    <!-- Key Finding -->
    <div class="section">
        <h2>Key Finding</h2>
        <div class="insight">{html_lib.escape(overall_test)}</div>
    </div>

    <!-- 1. Bias Analysis -->
    <div id="bias" class="section">
        <h2>1. Systematic Bias Analysis</h2>
        <p>Mean signed percent error by county category and projection horizon.
           Positive values = over-projection; negative = under-projection.</p>
        <div class="chart-row">
            {"".join(heatmap_divs)}
        </div>
        <h3>Temporal Stability</h3>
        <p>How does bias vary depending on which origin year the projection starts from?</p>
        {temporal_div}
    </div>

    <!-- 2. Residual Diagnostics -->
    <div id="residuals" class="section">
        <h2>2. Residual Diagnostics</h2>
        {acf_div}
        <h3>Diagnostics by Horizon Bucket</h3>
        {_residual_table_html(residual_df)}
    </div>

    <!-- 3. Structural Breaks -->
    <div id="structural" class="section">
        <h2>3. Structural Break Detection</h2>
        {paired_div}
        <h3>Pre-Boom vs Post-Boom Accuracy</h3>
        {_era_table_html(structural.get("pre_vs_post_boom", pd.DataFrame()))}
        <h3>Paired Method Comparison Tests</h3>
        {_paired_tests_html(structural.get("paired_method_tests", pd.DataFrame()))}
    </div>

    <!-- 4. Error Distribution -->
    <div class="section">
        <h2>4. Error Distribution by County Category</h2>
        {boxplot_div}
    </div>

    <!-- 5. County Report Cards -->
    <div id="report-cards" class="section">
        <h2>5. County Report Cards</h2>
        <p>Click column headers to sort. Grades: A (&lt;3% MAPE), B (3-7%), C (7-15%), D (&gt;15%).</p>
        {report_card_html}
    </div>

    <!-- 6. Outliers -->
    <div id="outliers" class="section">
        <h2>6. Outlier Detection</h2>
        <p>County x origin x year combinations where |error| exceeds 2 standard deviations
           from the horizon-specific mean.</p>
        {outlier_scatter_div}
        <h3>Top 20 Most Extreme Outliers</h3>
        {_outlier_table_html(outlier_df)}
    </div>

    </div><!-- /container -->

    <!-- Sortable table JavaScript -->
    <script>
    document.querySelectorAll('table.sortable').forEach(function(table) {{
        const headers = table.querySelectorAll('th');
        headers.forEach(function(header, index) {{
            header.addEventListener('click', function() {{
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const isAsc = header.classList.contains('asc');

                // Clear other sort indicators
                headers.forEach(h => {{
                    h.classList.remove('asc', 'desc');
                    const arrow = h.querySelector('.sort-arrow');
                    if (arrow) arrow.textContent = '';
                }});

                rows.sort(function(a, b) {{
                    let aVal = a.cells[index].getAttribute('data-sort') || a.cells[index].textContent.trim();
                    let bVal = b.cells[index].getAttribute('data-sort') || b.cells[index].textContent.trim();

                    // Try numeric comparison
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return isAsc ? bNum - aNum : aNum - bNum;
                    }}
                    return isAsc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
                }});

                rows.forEach(row => tbody.appendChild(row));
                header.classList.add(isAsc ? 'desc' : 'asc');
                const arrow = header.querySelector('.sort-arrow');
                if (arrow) arrow.textContent = isAsc ? ' \\u25BC' : ' \\u25B2';
            }});
        }});
    }});

    // Method toggle for report card table
    document.querySelectorAll('.method-toggle button').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            const method = this.getAttribute('data-method');
            const table = this.closest('.section').querySelector('table.sortable');
            if (!table) return;

            // Toggle active
            this.parentNode.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Filter rows
            const tbody = table.querySelector('tbody');
            tbody.querySelectorAll('tr').forEach(function(row) {{
                const rowMethod = row.getAttribute('data-method');
                if (method === 'all' || rowMethod === method) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }});
    }});
    </script>
    </body>
    </html>
    """)
    return html


def _build_report_card_table(rc: pd.DataFrame) -> str:
    """Build an HTML sortable table for county report cards."""
    lines = [
        '<div class="method-toggle">',
        '  <button class="active" data-method="all">All Methods</button>',
        '  <button data-method="sdc_2024">SDC 2024</button>',
        '  <button data-method="m2026">M2026</button>',
        "</div>",
        '<div class="table-wrapper">',
        '<table class="sortable">',
        "<thead><tr>",
        '<th>County<span class="sort-arrow"></span></th>',
        '<th>FIPS<span class="sort-arrow"></span></th>',
        '<th>Category<span class="sort-arrow"></span></th>',
        '<th>Method<span class="sort-arrow"></span></th>',
        '<th>Grade<span class="sort-arrow"></span></th>',
        '<th>MAPE (%)<span class="sort-arrow"></span></th>',
        '<th>Mean Signed Error (%)<span class="sort-arrow"></span></th>',
        '<th>Std Error<span class="sort-arrow"></span></th>',
        '<th>Worst Case (%)<span class="sort-arrow"></span></th>',
        '<th>Bias<span class="sort-arrow"></span></th>',
        '<th>N<span class="sort-arrow"></span></th>',
        "</tr></thead>",
        "<tbody>",
    ]

    grade_class = {"A": "grade-A", "B": "grade-B", "C": "grade-C", "D": "grade-D"}

    for _, row in rc.iterrows():
        g = row["grade"]
        gc = grade_class.get(g, "")
        method_label = "SDC 2024" if row["method"] == "sdc_2024" else "M2026"
        lines.append(
            f'<tr data-method="{row["method"]}">'
            f'<td>{html_lib.escape(str(row["county_name"]))}</td>'
            f'<td>{row["county_fips"]}</td>'
            f'<td>{row["category"]}</td>'
            f'<td>{method_label}</td>'
            f'<td class="{gc}" data-sort="{g}">{g}</td>'
            f'<td data-sort="{row["mape"]}">{row["mape"]:.2f}</td>'
            f'<td data-sort="{row["mean_signed_error"]}">{row["mean_signed_error"]:+.2f}</td>'
            f'<td data-sort="{row["std_error"]}">{row["std_error"]:.2f}</td>'
            f'<td data-sort="{abs(row["worst_case_error"])}">{row["worst_case_error"]:+.2f}</td>'
            f'<td>{row["bias_direction"]}</td>'
            f'<td data-sort="{row["n_validations"]}">{row["n_validations"]}</td>'
            f"</tr>"
        )
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _residual_table_html(residual_df: pd.DataFrame) -> str:
    """Build HTML table for residual diagnostics."""
    if residual_df.empty:
        return "<p>No residual diagnostic data available.</p>"

    lines = [
        '<table class="sortable">',
        "<thead><tr>",
        '<th>Method<span class="sort-arrow"></span></th>',
        '<th>Horizon Bucket<span class="sort-arrow"></span></th>',
        '<th>N<span class="sort-arrow"></span></th>',
        '<th>Mean ACF(1)<span class="sort-arrow"></span></th>',
        '<th>Het. R^2<span class="sort-arrow"></span></th>',
        '<th>Het. p-val<span class="sort-arrow"></span></th>',
        '<th>Het. Sig?<span class="sort-arrow"></span></th>',
        '<th>Shapiro W<span class="sort-arrow"></span></th>',
        '<th>Shapiro p<span class="sort-arrow"></span></th>',
        '<th>Normal?<span class="sort-arrow"></span></th>',
        '<th>Skewness<span class="sort-arrow"></span></th>',
        '<th>Kurtosis<span class="sort-arrow"></span></th>',
        "</tr></thead>",
        "<tbody>",
    ]
    for _, r in residual_df.iterrows():
        method_label = "SDC 2024" if r["method"] == "sdc_2024" else "M2026"
        het_sig = "Yes" if r.get("het_significant") else "No"
        normal = "Yes" if r.get("normal_at_05") else "No"
        het_style = ' style="color:#e74c3c;font-weight:bold"' if het_sig == "Yes" else ""
        norm_style = ' style="color:#27ae60;font-weight:bold"' if normal == "Yes" else ' style="color:#e74c3c"'
        lines.append(
            f"<tr>"
            f"<td>{method_label}</td>"
            f'<td>{r["horizon_bucket"]}</td>'
            f'<td>{r["n_obs"]}</td>'
            f'<td>{r["mean_autocorr_lag1"]:.4f}</td>'
            f'<td>{r["het_r2_vs_pop_size"]:.6f}</td>'
            f'<td>{r["het_p_value"]:.6f}</td>'
            f"<td{het_style}>{het_sig}</td>"
            f'<td>{r["shapiro_w"]:.4f}</td>'
            f'<td>{r["shapiro_p"]:.6f}</td>'
            f"<td{norm_style}>{normal}</td>"
            f'<td>{r["error_skew"]:.4f}</td>'
            f'<td>{r["error_kurtosis"]:.4f}</td>'
            f"</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _era_table_html(era_df: pd.DataFrame) -> str:
    """Build HTML table for pre/post boom comparison."""
    if era_df.empty:
        return "<p>No era comparison data available.</p>"

    lines = [
        '<table class="sortable">',
        "<thead><tr>",
        '<th>Method<span class="sort-arrow"></span></th>',
        '<th>Era<span class="sort-arrow"></span></th>',
        '<th>Category<span class="sort-arrow"></span></th>',
        '<th>Mean |Error| (%)<span class="sort-arrow"></span></th>',
        '<th>Mean Signed Error (%)<span class="sort-arrow"></span></th>',
        '<th>N<span class="sort-arrow"></span></th>',
        "</tr></thead>",
        "<tbody>",
    ]
    for _, r in era_df.iterrows():
        method_label = "SDC 2024" if r["method"] == "sdc_2024" else "M2026"
        lines.append(
            f"<tr>"
            f"<td>{method_label}</td>"
            f'<td>{r["era"]}</td>'
            f'<td>{r["category"]}</td>'
            f'<td>{r["mean_abs_pct_error"]:.2f}</td>'
            f'<td>{r["mean_signed_pct_error"]:+.2f}</td>'
            f'<td>{r["n_obs"]}</td>'
            f"</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _paired_tests_html(paired_df: pd.DataFrame) -> str:
    """Build HTML table for paired method comparison tests."""
    if paired_df.empty:
        return "<p>No paired test data available.</p>"

    lines = [
        "<table>",
        "<thead><tr>",
        "<th>Comparison</th>",
        "<th>N Pairs</th>",
        "<th>SDC Mean |Error|</th>",
        "<th>M2026 Mean |Error|</th>",
        "<th>Difference</th>",
        "<th>Paired t-stat</th>",
        "<th>Paired t p-val</th>",
        "<th>Wilcoxon stat</th>",
        "<th>Wilcoxon p-val</th>",
        "<th>SDC Better?</th>",
        "</tr></thead>",
        "<tbody>",
    ]
    for _, r in paired_df.iterrows():
        better_style = ' style="color:#27ae60;font-weight:bold"' if r["sdc_better"] == "No" else ' style="color:#e74c3c;font-weight:bold"'
        sig_style = ' style="font-weight:bold"' if r["paired_t_p"] < 0.05 else ""
        lines.append(
            f"<tr>"
            f'<td>{r["comparison"]}</td>'
            f'<td>{r["n_pairs"]}</td>'
            f'<td>{r["sdc_mean_abs_error"]:.2f}%</td>'
            f'<td>{r["m2026_mean_abs_error"]:.2f}%</td>'
            f'<td>{r["diff_mean"]:+.2f}pp</td>'
            f'<td{sig_style}>{r["paired_t_stat"]:.4f}</td>'
            f'<td{sig_style}>{r["paired_t_p"]:.6f}</td>'
            f'<td>{r["wilcoxon_stat"]}</td>'
            f'<td>{r["wilcoxon_p"]:.6f}</td>'
            f'<td{better_style}>{r["sdc_better"]}</td>'
            f"</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _outlier_table_html(outlier_df: pd.DataFrame) -> str:
    """Build HTML table for top outliers."""
    if outlier_df.empty:
        return "<p>No outliers detected.</p>"

    top = outlier_df.sort_values("z_score", key=abs, ascending=False).head(20)

    lines = [
        '<table class="sortable">',
        "<thead><tr>",
        '<th>County<span class="sort-arrow"></span></th>',
        '<th>FIPS<span class="sort-arrow"></span></th>',
        '<th>Method<span class="sort-arrow"></span></th>',
        '<th>Origin<span class="sort-arrow"></span></th>',
        '<th>Validation Year<span class="sort-arrow"></span></th>',
        '<th>Horizon<span class="sort-arrow"></span></th>',
        '<th>% Error<span class="sort-arrow"></span></th>',
        '<th>Z-Score<span class="sort-arrow"></span></th>',
        '<th>Projected<span class="sort-arrow"></span></th>',
        '<th>Actual<span class="sort-arrow"></span></th>',
        "</tr></thead>",
        "<tbody>",
    ]
    for _, r in top.iterrows():
        method_label = "SDC 2024" if r["method"] == "sdc_2024" else "M2026"
        z_style = ' style="color:#e74c3c;font-weight:bold"' if abs(r["z_score"]) > 3 else ""
        lines.append(
            f"<tr>"
            f'<td>{html_lib.escape(str(r["county_name"]))}</td>'
            f'<td>{r["county_fips"]}</td>'
            f"<td>{method_label}</td>"
            f'<td>{int(r["origin_year"])}</td>'
            f'<td>{int(r["validation_year"])}</td>'
            f'<td>{int(r["horizon"])}</td>'
            f'<td data-sort="{r["pct_error"]}">{r["pct_error"]:+.2f}%</td>'
            f'<td{z_style} data-sort="{abs(r["z_score"])}">{r["z_score"]:+.2f}</td>'
            f'<td>{r["projected"]:,.0f}</td>'
            f'<td>{r["actual"]:,.0f}</td>'
            f"</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


# ===================================================================
# MAIN
# ===================================================================


def main() -> None:
    """Run all QC diagnostics and write outputs."""
    print("=" * 70)
    print("QC DIAGNOSTICS: Walk-Forward Validation")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading walk-forward results ...")
    county_df = load_county_detail()
    _state_df = load_state_results()
    _comparison_df = load_method_comparison()
    print(f"  County detail: {len(county_df):,} rows, {county_df['county_fips'].nunique()} counties")
    print(f"  Methods: {', '.join(county_df['method'].unique())}")
    print(f"  Origins: {sorted(county_df['origin_year'].unique())}")
    print(f"  Horizons: {sorted(county_df['horizon'].unique())}")

    # --- 1. Bias analysis ---
    print("\n1. Computing systematic bias analysis ...")
    bias_df = compute_bias_analysis(county_df)
    bias_path = OUTPUT_DIR / "bias_analysis.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"   Wrote {bias_path.relative_to(PROJECT_ROOT)}")

    # Print bias summary
    bias_all = bias_df[bias_df["origin_year"] == "all"]
    for method in ["sdc_2024", "m2026"]:
        print(f"\n   {method.upper()} — Mean signed error by category (all horizons averaged):")
        for cat in ["Bakken", "Urban/College", "Reservation", "Rural"]:
            sub = bias_all[(bias_all["method"] == method) & (bias_all["category"] == cat)]
            if not sub.empty:
                avg = sub["mean_signed_pct_error"].mean()
                direction = "over-projection" if avg > 0 else "under-projection"
                print(f"     {cat:18s}: {avg:+.2f}% ({direction})")

    # --- 2. Residual diagnostics ---
    print("\n2. Computing residual diagnostics ...")
    residual_df = compute_residual_diagnostics(county_df)
    residual_path = OUTPUT_DIR / "residual_diagnostics.csv"
    residual_df.to_csv(residual_path, index=False)
    print(f"   Wrote {residual_path.relative_to(PROJECT_ROOT)}")

    for _, r in residual_df.iterrows():
        method_label = "SDC" if r["method"] == "sdc_2024" else "M2026"
        print(
            f"   {method_label} {r['horizon_bucket']:8s}: "
            f"ACF(1)={r['mean_autocorr_lag1']:.3f}, "
            f"Het p={r['het_p_value']:.4f}, "
            f"Shapiro p={r['shapiro_p']:.4f}, "
            f"Skew={r['error_skew']:.2f}"
        )

    # --- Outlier detection ---
    print("\n   Detecting outliers (|z| > 2 SD) ...")
    outlier_df = detect_outliers(county_df)
    outlier_path = OUTPUT_DIR / "outlier_flags.csv"
    if not outlier_df.empty:
        outlier_df.to_csv(outlier_path, index=False)
        print(f"   Wrote {outlier_path.relative_to(PROJECT_ROOT)} ({len(outlier_df)} flagged)")
        # Top 5 most extreme
        top5 = outlier_df.sort_values("z_score", key=abs, ascending=False).head(5)
        for _, r in top5.iterrows():
            print(
                f"     {r['county_name']:20s} origin={int(r['origin_year'])} "
                f"h={int(r['horizon'])} {r['method']:8s} "
                f"error={r['pct_error']:+.1f}% z={r['z_score']:+.2f}"
            )
    else:
        pd.DataFrame().to_csv(outlier_path, index=False)
        print(f"   No outliers detected.")

    # --- 3. Structural breaks ---
    print("\n3. Computing structural break detection ...")
    structural = compute_structural_breaks(county_df)
    paired = structural.get("paired_method_tests", pd.DataFrame())
    if not paired.empty:
        overall = paired[paired["comparison"] == "overall"]
        if not overall.empty:
            r = overall.iloc[0]
            better = "M2026" if r["sdc_better"] == "No" else "SDC 2024"
            sig = "SIGNIFICANT" if r["paired_t_p"] < 0.05 else "not significant"
            print(
                f"   Overall: {better} has lower |error| "
                f"(diff={r['diff_mean']:+.2f}pp, t={r['paired_t_stat']:.3f}, "
                f"p={r['paired_t_p']:.4f} [{sig}])"
            )

    # --- 4. County report cards ---
    print("\n4. Computing county report cards ...")
    report_cards = compute_county_report_cards(county_df)
    rc_path = OUTPUT_DIR / "county_report_cards.csv"
    report_cards.to_csv(rc_path, index=False)
    print(f"   Wrote {rc_path.relative_to(PROJECT_ROOT)}")

    for method in ["sdc_2024", "m2026"]:
        mrc = report_cards[report_cards["method"] == method]
        dist = mrc["grade"].value_counts().sort_index()
        method_label = "SDC 2024" if method == "sdc_2024" else "M2026"
        print(f"\n   {method_label} Grade Distribution:")
        for grade in ["A", "B", "C", "D"]:
            count = dist.get(grade, 0)
            print(f"     {grade}: {count} counties")

        # Counties needing attention (grade D)
        d_counties = mrc[mrc["grade"] == "D"]
        if not d_counties.empty:
            print(f"   Counties needing attention ({method_label}, grade D):")
            for _, r in d_counties.iterrows():
                print(
                    f"     {r['county_name']:20s} (FIPS {r['county_fips']}): "
                    f"MAPE={r['mape']:.1f}%, bias={r['bias_direction']}, "
                    f"worst={r['worst_case_error']:+.1f}%"
                )

    # --- 5. ACF computation ---
    print("\n5. Computing autocorrelation by lag ...")
    acf_df = compute_autocorrelation_by_lag(county_df)

    # --- 6. Interactive HTML report ---
    print("\n6. Building interactive HTML report ...")
    html_content = _build_html_report(
        bias_df=bias_df,
        residual_df=residual_df,
        outlier_df=outlier_df,
        report_cards=report_cards,
        county_df=county_df,
        structural=structural,
        acf_df=acf_df,
    )
    html_path = OUTPUT_DIR / "qc_diagnostics_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"   Wrote {html_path.relative_to(PROJECT_ROOT)}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("QC DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    for p in [bias_path, residual_path, outlier_path, rc_path, html_path]:
        print(f"  {p.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
