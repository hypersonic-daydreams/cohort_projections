"""Build a standalone interactive North Dakota state population-pyramid explorer.

Produces a single self-contained HTML file with a year slider that scrubs the
statewide population pyramid across:

- the previous ~10 observed years (Census PEP, 5-year age groups), and
- the locked baseline projection years 2025-2055
  (``m2026r1`` / ``cfg-20260611-production-lock``; ADR-065 CBO-adjusted baseline).

Both eras are binned to the same 18 five-year age groups (0-4 ... 85+) so the
scrub is seamless across the observed->projected boundary. Open the HTML in any
browser; no server is required.

Data sources (single source of truth):
- Projection: ``data/projections/baseline/state/nd_state_38_projection_2025_2055_baseline.parquet``
  (single-year age x sex x race; aggregated over race and binned to 5-year groups).
- Historical: Census PEP county age-sex via
  ``scripts.analysis.walk_forward_validation.load_population_snapshot`` (summed to state).

Usage:
    python scripts/exports/build_pyramid_explorer.py
    python scripts/exports/build_pyramid_explorer.py --history-years 10 --output <path.html>
"""

from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Make `scripts.analysis.walk_forward_validation` importable when run as a script
# (its own dir, not the project root, is on sys.path[0] under `python scripts/...`).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECTION_PATH = (
    PROJECT_ROOT / "data" / "projections" / "baseline" / "state" /
    "nd_state_38_projection_2025_2055_baseline.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "exports" / "pyramid_explorer" / "nd_state_pyramid_explorer.html"

PROJ_FIRST_YEAR = 2025
PROJ_LAST_YEAR = 2055
RUN_TAG = "m2026r1 / cfg-20260611-production-lock (2026-06-13)"

# Accessible, non-red/green pairing (also distinguished by side + label).
MALE_COLOR = "#2c5f8a"   # SDC blue
FEMALE_COLOR = "#d99536"  # warm amber

_wfv = importlib.import_module("scripts.analysis.walk_forward_validation")
AGE_GROUPS: list[str] = list(_wfv.AGE_GROUP_LABELS)  # ["0-4", ..., "85+"], bottom -> top


def _age_to_group(age: float) -> str:
    """Map a single-year age (0-90) to its 5-year group label, capping at 85+."""
    idx = min(int(age) // 5, len(AGE_GROUPS) - 1)
    return AGE_GROUPS[idx]


def load_state_history(years: list[int]) -> pd.DataFrame:
    """Observed statewide population by 5-year age group and sex for the given years."""
    frames = []
    for year in years:
        snap = _wfv.load_population_snapshot(year)  # county x age_group x sex
        agg = (
            snap.groupby(["age_group", "sex"], observed=True)["population"].sum().reset_index()
        )
        agg["year"] = year
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True)
    out["era"] = "observed"
    return out


def load_state_projection() -> pd.DataFrame:
    """Locked baseline statewide population by 5-year age group and sex, 2025-2055."""
    df = pd.read_parquet(PROJECTION_PATH)
    df["age_group"] = df["age"].map(_age_to_group)
    agg = (
        df.groupby(["year", "age_group", "sex"], observed=True)["population"].sum().reset_index()
    )
    agg["era"] = "projected"
    return agg


def assemble(history_years: int) -> pd.DataFrame:
    """Combine observed + projected into one tidy frame ordered by year."""
    hist_years = list(range(PROJ_FIRST_YEAR - history_years, PROJ_FIRST_YEAR))
    hist = load_state_history(hist_years)
    proj = load_state_projection()
    combined = pd.concat([hist, proj], ignore_index=True)
    # Stable categorical ordering for the y-axis (0-4 bottom ... 85+ top).
    combined["age_group"] = pd.Categorical(combined["age_group"], categories=AGE_GROUPS, ordered=True)
    return combined.sort_values(["year", "age_group", "sex"]).reset_index(drop=True)


def _year_pivot(df: pd.DataFrame, year: int) -> tuple[list[float], list[float], float]:
    """Return (male_pops, female_pops, total) aligned to AGE_GROUPS for one year."""
    sub = df[df["year"] == year]
    piv = sub.pivot_table(
        index="age_group", columns="sex", values="population", aggfunc="sum", observed=False
    ).reindex(AGE_GROUPS).fillna(0.0)
    male = piv.get("Male", pd.Series(0.0, index=AGE_GROUPS)).tolist()
    female = piv.get("Female", pd.Series(0.0, index=AGE_GROUPS)).tolist()
    return male, female, float(piv.to_numpy().sum())


def _era_label(year: int, era: str) -> str:
    if era == "observed":
        return f"{year} — Observed (Census PEP)"
    return f"{year} — Projected (baseline, CBO-adjusted)"


def _xaxis_ticks(max_val: float) -> tuple[list[float], list[str], list[float]]:
    """Symmetric tick values/labels (absolute) and the axis range."""
    # Round the half-range up to a clean step.
    step = 5_000 if max_val <= 30_000 else 10_000
    top = (int(max_val // step) + 1) * step
    vals = list(range(-top, top + 1, step))
    text = [f"{abs(v):,}" for v in vals]
    return vals, text, [-top * 1.02, top * 1.02]


def build_figure(df: pd.DataFrame) -> go.Figure:
    years = sorted(df["year"].unique())
    era_by_year = df.drop_duplicates("year").set_index("year")["era"].to_dict()

    # Fixed axis range across all years so bars are comparable while scrubbing.
    max_val = (
        df.groupby(["year", "age_group", "sex"], observed=False)["population"].sum().max()
    )
    tickvals, ticktext, xrange = _xaxis_ticks(float(max_val))

    def traces_for(year: int) -> list[go.Bar]:
        male, female, _ = _year_pivot(df, year)
        return [
            go.Bar(
                y=AGE_GROUPS, x=[-m for m in male], name="Male", orientation="h",
                marker_color=MALE_COLOR,
                customdata=[[m] for m in male],
                hovertemplate="Age %{y}<br>Male: %{customdata[0]:,.0f}<extra></extra>",
            ),
            go.Bar(
                y=AGE_GROUPS, x=female, name="Female", orientation="h",
                marker_color=FEMALE_COLOR,
                hovertemplate="Age %{y}<br>Female: %{x:,.0f}<extra></extra>",
            ),
        ]

    def annotation_for(year: int) -> dict:
        _, _, total = _year_pivot(df, year)
        return {
            "x": 0.5, "y": 1.05, "xref": "paper", "yref": "paper", "showarrow": False,
            "text": f"<b>{_era_label(year, era_by_year[year])}</b>  ·  Total: {total:,.0f}",
            "font": {"size": 15},
        }

    first = years[0]
    fig = go.Figure(data=traces_for(first))
    fig.frames = [
        go.Frame(
            name=str(y), data=traces_for(y),
            layout=go.Layout(annotations=[annotation_for(y)]),
        )
        for y in years
    ]

    steps = [
        {
            "method": "animate", "label": str(y),
            "args": [[str(y)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                 "transition": {"duration": 0}}],
        }
        for y in years
    ]
    # Mark where projection begins on the slider.
    proj_index = years.index(PROJ_FIRST_YEAR) if PROJ_FIRST_YEAR in years else 0

    fig.update_layout(
        title={
            "text": "North Dakota Statewide Population Pyramid<br>"
            "<sub>Observed (Census PEP) through projected (baseline, "
            f"CBO-adjusted) · {RUN_TAG}</sub>",
            "x": 0.5,
        },
        barmode="overlay", bargap=0.08,
        xaxis={"title": "Population", "tickvals": tickvals, "ticktext": ticktext, "range": xrange,
                   "zeroline": True, "zerolinecolor": "#888", "zerolinewidth": 1},
        yaxis={"title": "Age group", "categoryorder": "array", "categoryarray": AGE_GROUPS},
        annotations=[annotation_for(first)],
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.14},
        template="plotly_white", height=720, width=900,
        margin={"l": 70, "r": 40, "t": 110, "b": 120},
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "Year: ", "font": {"size": 16}},
            "pad": {"t": 40, "b": 10}, "x": 0.08, "len": 0.88, "steps": steps,
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.08, "y": -0.02, "xanchor": "left", "yanchor": "top",
            "pad": {"t": 10}, "showactive": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True,
                                      "transition": {"duration": 0}}]},
                {"label": "❚❚ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
    )
    # Caption explaining the observed->projected handoff and the 2028 dip cue.
    fig.add_annotation(
        x=0.5, y=-0.26, xref="paper", yref="paper", showarrow=False,
        text=(
            f"Years {first}–{PROJ_FIRST_YEAR - 1} are observed Census PEP estimates (5-year age "
            f"groups). Years {PROJ_FIRST_YEAR}–{PROJ_LAST_YEAR} are the locked public baseline "
            "(state dips to a 2028 trough under the CBO migration ramp, then recovers). "
            "Group quarters are held constant in the projection."
        ),
        font={"size": 11, "color": "#555"}, align="center",
    )
    _ = proj_index  # reserved for an optional projection-start slider marker
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-years", type=int, default=10,
                        help="Observed years before 2025 to include (default 10 -> from 2015).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = assemble(args.history_years)
    fig = build_figure(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        args.output, include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    years = sorted(df["year"].unique())
    print(f"Wrote pyramid explorer: {args.output}")
    print(f"  Years: {years[0]}-{years[-1]} ({len(years)} frames; "
          f"observed {years[0]}-{PROJ_FIRST_YEAR - 1}, projected {PROJ_FIRST_YEAR}-{years[-1]})")
    print(f"  Open in a browser: file://{args.output.resolve()}")


if __name__ == "__main__":
    main()
