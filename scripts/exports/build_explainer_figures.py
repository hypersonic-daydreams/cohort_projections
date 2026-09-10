"""Mock up the five candidate figures for the public companion explainer.

Drafts only — for brainstorming/selection, not yet wired into
``how-these-projections-work.md``. See
``docs/plans/2026-public-projection-release-handoff/figure-drafts.md``.

Figures
-------
1. Same size today, different futures (schematic; banner-exempt)
2. One year of the method (schematic cycle diagram; banner-exempt)
3. One path, with its built-in dip (locked state trajectory; data-bearing)
4. The same population, two shapes (Williams vs. Ward pyramids; data-bearing)
5. Where the growth actually goes (county choropleth; data-bearing)

Data sources (corrected ADR-068 full-horizon run, 2026-06-16)
    - State trajectory: data/projections/baseline/state/...baseline_summary.csv
    - County totals:    data/projections/baseline/county/countys_summary.csv
    - County detail:    data/projections/baseline/county/nd_county_<fips>_..._baseline.parquet
    - County geometry:  data/exports/baseline/county/geojson/nd_county_baseline_2055.geojson

Usage
    python scripts/exports/build_explainer_figures.py
Outputs PNGs to docs/plans/2026-public-projection-release-handoff/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "data" / "projections" / "baseline"
GEOJSON = (
    ROOT
    / "data"
    / "exports"
    / "baseline"
    / "county"
    / "geojson"
    / "nd_county_baseline_2055.geojson"
)
OUTDIR = ROOT / "docs" / "plans" / "2026-public-projection-release-handoff" / "figures"

# Reserved, low-chrome palette
INK = "#222222"
MUTED = "#6b6b6b"
GROW = "#2b6f6a"  # teal — growth
DECLINE = "#b4654a"  # terracotta — decline
HILITE = "#1f4e79"  # deep blue — the three concentrators
LIGHT = "#d9d9d9"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    }
)


def _save(fig, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Figure 1 — Same size today, different futures (schematic)
# --------------------------------------------------------------------------- #
def fig1_diverge() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    x = np.linspace(0, 30, 200)

    # Left: one shared trend line
    trend = 100 + 1.4 * x
    ax_l.plot(x, trend, color=INK, lw=2.4)
    ax_l.text(30.4, trend[-1], "both", va="center", color=INK)
    ax_l.set_title("Same size, same growth today", color=INK, fontsize=11.5, pad=8)
    ax_l.text(
        0, 56, "the trend-line view", color=MUTED, fontsize=9.5, style="italic"
    )

    # Right: diverging paths driven by age structure
    young = 100 + 1.2 * x + 0.036 * x**2
    old = 100 + 1.2 * x - 0.050 * x**2
    ax_r.plot(x, young, color=GROW, lw=2.4)
    ax_r.plot(x, old, color=DECLINE, lw=2.4)
    ax_r.text(30.6, young[-1], "County A\n(young families)", va="center", color=GROW, fontsize=9)
    ax_r.text(30.6, old[-1], "County B\n(past middle age)", va="center", color=DECLINE, fontsize=9)
    ax_r.set_title("But not headed the same place", color=INK, fontsize=11.5, pad=8)
    ax_r.text(
        0, 56, "what age structure reveals", color=MUTED, fontsize=9.5, style="italic"
    )

    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 40)
        ax.set_ylim(50, 185)
        ax.set_xticks([0, 30])
        ax.set_xticklabels(["today", "+30 yrs"])
        ax.set_yticks([])
        ax.set_ylabel("population", color=MUTED, fontsize=9.5)

    fig.suptitle(
        "Two counties, identical today — different futures",
        fontsize=13,
        y=1.06,
        color=INK,
    )
    _save(fig, "fig1_same_size_different_futures.png")


# --------------------------------------------------------------------------- #
# Figure 2 — One year of the method (schematic cycle)
# --------------------------------------------------------------------------- #
def fig2_cycle() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f3f1ec", ec=MUTED, tcol=INK):
        b = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.3,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(b)
        ax.text(x, y + (0.16 if body else 0), title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=tcol)
        if body:
            ax.text(x, y - 0.42, body, ha="center", va="center", fontsize=8.2,
                    color=MUTED)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                            lw=1.3, color=color, shrinkA=2, shrinkB=2)
        )

    # Top: population on July 1
    box(5, 9.0, 6.6, 1.2, "POPULATION on July 1",
        "every cohort:  age × sex × race", fc="#eef2f4", ec=HILITE, tcol=HILITE)

    # Three components
    comps = [
        (2.2, "1 · SURVIVAL", "everyone ages +1;\nsubtract deaths"),
        (5.0, "2 · FERTILITY", "add the new\nage-0 cohort"),
        (7.8, "3 · MIGRATION", "net of arrivals\n& departures"),
    ]
    for cx, t, b in comps:
        box(cx, 6.0, 2.4, 1.4, t, b)
        arrow(5, 8.4, cx, 6.7)  # from population to component
        arrow(cx, 5.3, 5, 4.3)  # from component to next-year box

    # Next year
    box(5, 3.6, 6.6, 1.05, "POPULATION on the next July 1", "",
        fc="#eef2f4", ec=HILITE, tcol=HILITE)

    # Loop back arrow (down the side and up)
    ax.add_patch(
        FancyArrowPatch(
            (8.45, 3.6), (8.45, 9.0),
            connectionstyle="arc3,rad=-0.55",
            arrowstyle="-|>", mutation_scale=15, lw=1.4, color=INK,
        )
    )
    ax.text(9.9, 6.3, "repeat,\nonce a year\n2025 → 2055", ha="center", va="center",
            fontsize=8.6, color=INK, style="italic")

    ax.set_title("One year of the method, run thirty times over",
                 fontsize=13, color=INK, pad=4)
    _save(fig, "fig2_one_year_of_the_method.png")


# --------------------------------------------------------------------------- #
# Figure 3 — One path, with its built-in dip (locked trajectory)
# --------------------------------------------------------------------------- #
def fig3_trajectory() -> None:
    df = pd.read_csv(
        PROJ / "state" / "nd_state_38_projection_2025_2055_baseline_summary.csv"
    )
    yr = df["year"].to_numpy()
    pop = df["total_population"].to_numpy() / 1000.0

    start = pop[0]
    trough_i = int(np.argmin(pop))
    end = pop[-1]
    pct = (pop[-1] - pop[0]) / pop[0] * 100

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(yr, pop, color=HILITE, lw=2.6)
    ax.fill_between(yr, pop, start, where=(pop < start), color=DECLINE, alpha=0.10)

    # markers
    ax.scatter([yr[0]], [start], color=INK, zorder=5, s=26)
    ax.scatter([yr[trough_i]], [pop[trough_i]], color=DECLINE, zorder=5, s=30)
    ax.scatter([yr[-1]], [end], color=GROW, zorder=5, s=30)

    ax.annotate(
        f"{start*1000:,.0f}\n(2025)",
        (yr[0], start), textcoords="offset points", xytext=(6, 14),
        fontsize=9, color=INK,
    )
    # trough callout placed in the empty upper-left, arrow down to the dip
    ax.annotate(
        f"shallow low ≈ {pop[trough_i]*1000:,.0f}  ({yr[trough_i]})\n"
        "a feature of the migration assumption —\n"
        "not a downturn the model found on its own",
        (yr[trough_i], pop[trough_i]),
        xytext=(2031, 868), textcoords="data",
        fontsize=8.6, color=DECLINE, va="center",
        arrowprops={"arrowstyle": "-|>", "color": DECLINE, "lw": 1.1,
                    "connectionstyle": "arc3,rad=0.25"},
    )
    ax.annotate(
        f"≈ {end*1000:,.0f}  (2055)\n+{pct:.0f}% vs. 2025",
        (yr[-1], end), textcoords="offset points", xytext=(-12, -20),
        ha="right", fontsize=9, color=GROW,
    )

    ax.set_ylabel("population (thousands)", color=MUTED)
    ax.set_xlim(2023.5, 2057)
    ax.set_ylim(790, 912)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050, 2055])
    ax.set_title("One projected path — and its built-in early dip",
                 fontsize=13, color=INK, pad=12)
    _save(fig, "fig3_one_path_with_dip.png")


# --------------------------------------------------------------------------- #
# Figure 4 — The same population, two shapes (pyramids)
# --------------------------------------------------------------------------- #
_AGE_BINS = list(range(0, 90, 5))  # 0,5,...,85 ; top bin = 85+/90+ collapsed
_AGE_LABELS = [f"{a}-{a+4}" for a in range(0, 85, 5)] + ["85+"]


def _pyramid_data(fips: str):
    df = pd.read_parquet(
        PROJ / "county" / f"nd_county_{fips}_projection_2025_2055_baseline.parquet"
    )
    d = df[df["year"] == 2025].copy()
    d["grp"] = np.minimum((d["age"] // 5) * 5, 85)
    g = d.groupby(["grp", "sex"])["population"].sum().unstack(fill_value=0.0)
    total = g.to_numpy().sum()
    male = (g.get("Male", 0) / total * 100).reindex(range(0, 90, 5), fill_value=0)
    female = (g.get("Female", 0) / total * 100).reindex(range(0, 90, 5), fill_value=0)
    return male.to_numpy(), female.to_numpy(), total


def fig4_pyramids() -> None:
    # Williams (38105) young / oil; Ward (38101) older
    young_m, young_f, young_t = _pyramid_data("38105")
    old_m, old_f, old_t = _pyramid_data("38101")
    y = np.arange(len(_AGE_LABELS))
    xmax = max(young_m.max(), young_f.max(), old_m.max(), old_f.max()) * 1.15

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.4, 4.8), sharey=True)

    def draw(ax, male, female, title, sub):
        ax.barh(y, -male, color=HILITE, alpha=0.85, height=0.82)
        ax.barh(y, female, color=GROW, alpha=0.85, height=0.82)
        ax.set_xlim(-xmax, xmax)
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.text(0, len(y) - 0.2, sub, ha="center", fontsize=8.6,
                color=MUTED, style="italic")
        ax.axvline(0, color="white", lw=0.8)
        ax.set_xticks([])
        ax.text(-xmax * 0.55, -1.4, "← men", color=HILITE, fontsize=8.5, ha="center")
        ax.text(xmax * 0.55, -1.4, "women →", color=GROW, fontsize=8.5, ha="center")

    draw(ax_l, young_m, young_f, "A young county  (Williams)",
         "wide base — children & prime-age adults")
    draw(ax_r, old_m, old_f, "An older county  (Ward)",
         "narrower base, heavier middle & top")

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(_AGE_LABELS, fontsize=8)
    ax_l.set_ylim(-2.2, len(y) + 0.4)

    fig.suptitle("A population is a shape, not a number  (share of county pop., 2025)",
                 fontsize=12.5, y=1.02, color=INK)
    _save(fig, "fig4_two_shapes_pyramids.png")


# --------------------------------------------------------------------------- #
# Figure 5 — Where the growth actually goes (choropleth)
# --------------------------------------------------------------------------- #
def fig5_map() -> None:
    summ = pd.read_csv(PROJ / "county" / "countys_summary.csv")
    summ["fips"] = summ["fips"].astype(str)
    growth = dict(zip(summ["fips"], summ["growth_rate"], strict=True))
    absgain = dict(zip(summ["fips"], summ["absolute_growth"], strict=True))

    geo = json.loads(GEOJSON.read_text())

    n_decline = int((summ["growth_rate"] < 0).sum())
    n_grow = int((summ["growth_rate"] >= 0).sum())
    total_gain = summ.loc[summ["absolute_growth"] > 0, "absolute_growth"].sum()
    top3 = {"38017", "38105", "38015"}  # Cass, Williams, Burleigh
    top3_share = sum(absgain[f] for f in top3) / total_gain * 100

    # diverging color by growth rate, clipped to +/-30%
    def color_for(rate):
        r = max(-0.30, min(0.30, rate)) / 0.30
        if r < 0:
            t = -r
            return tuple(np.array(matplotlib.colors.to_rgb("#f0ddd5")) * (1 - t)
                         + np.array(matplotlib.colors.to_rgb(DECLINE)) * t)
        t = r
        return tuple(np.array(matplotlib.colors.to_rgb("#dcebe9")) * (1 - t)
                     + np.array(matplotlib.colors.to_rgb(GROW)) * t)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    patches, colors = [], []
    label_pts = {}
    for feat in geo["features"]:
        fips = feat["properties"]["county_fips"]
        geom = feat["geometry"]
        rings = (
            [geom["coordinates"][0]]
            if geom["type"] == "Polygon"
            else [poly[0] for poly in geom["coordinates"]]
        )
        for ring in rings:
            arr = np.array(ring)
            patches.append(Polygon(arr, closed=True))
            colors.append(color_for(growth.get(fips, 0.0)))
            if fips in top3:
                label_pts[fips] = arr.mean(axis=0)

    pc = PatchCollection(patches, facecolors=colors, edgecolors="white", linewidths=0.6)
    ax.add_collection(pc)

    # outline + label the three concentrators
    names = {"38017": "Cass\n(Fargo)", "38105": "Williams\n(Williston)",
             "38015": "Burleigh\n(Bismarck)"}
    for feat in geo["features"]:
        fips = feat["properties"]["county_fips"]
        if fips in top3:
            ring = np.array(feat["geometry"]["coordinates"][0])
            ax.add_patch(Polygon(ring, closed=True, fill=False,
                                 edgecolor=HILITE, linewidth=2.0))
    # nudge labels off the polygon edge: Cass hugs the eastern border
    nudge = {"38017": (-0.45, 0.0), "38105": (0.0, 0.05), "38015": (0.0, 0.0)}
    for fips, pt in label_pts.items():
        dx, dy = nudge.get(fips, (0.0, 0.0))
        ax.annotate(names[fips], (pt[0] + dx, pt[1] + dy), fontsize=8.2,
                    ha="center", color=HILITE, fontweight="bold")

    ax.autoscale_view()
    ax.set_aspect(1.45)  # ~1/cos(48°N): keep ND from squashing
    # headroom on the right so the easternmost (Cass) label isn't clipped
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - 0.1, x1 + 0.5)
    ax.axis("off")
    ax.set_title("Where the growth actually goes — projected change, 2025–2055",
                 fontsize=13, color=INK, pad=6)
    ax.text(
        0.5, -0.04,
        f"{n_grow} counties grow, {n_decline} decline.  "
        f"Three (outlined) hold ≈{top3_share:.0f}% of all projected gains.",
        transform=ax.transAxes, ha="center", fontsize=9.3, color=MUTED,
    )
    # legend
    leg = [
        mpatches.Patch(color=GROW, label="stronger growth"),
        mpatches.Patch(color="#e7efed", label="≈ flat"),
        mpatches.Patch(color=DECLINE, label="decline"),
        mpatches.Patch(facecolor="white", edgecolor=HILITE, linewidth=2,
                       label=f"3 counties ≈ {top3_share:.0f}% of gains"),
    ]
    ax.legend(handles=leg, loc="lower left", frameon=False, fontsize=8.4)
    _save(fig, "fig5_where_growth_goes.png")

    print(f"   [check] grow={n_grow} decline={n_decline} top3_share={top3_share:.1f}%")


def main() -> None:
    fig1_diverge()
    fig2_cycle()
    fig3_trajectory()
    fig4_pyramids()
    fig5_map()
    print(f"\nAll figures written to {OUTDIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
