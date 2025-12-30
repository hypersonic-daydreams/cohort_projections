#!/usr/bin/env python3
"""
Create the analytical pipeline diagram for the journal article.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LOGGER = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, center, text, color, width=0.18, height=0.12) -> None:
    """Add a labeled rounded box to the axes."""
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#111111",
        wrap=True,
    )


def add_arrow(ax, start, end, dashed=False) -> None:
    """Add an arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color="#333333",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(arrow)


def create_pipeline_diagram() -> Path:
    """Generate the pipeline diagram figure."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "data": "#E6E6E6",
        "forecast": "#B3D3EA",
        "scenario": "#B8E3C6",
        "causal": "#F2C29E",
        "output": "#E6E6E6",
    }

    positions = {
        "data": (0.10, 0.5),
        "diagnostic": (0.28, 0.5),
        "predictive": (0.46, 0.5),
        "scenario": (0.70, 0.5),
        "output": (0.88, 0.5),
        "duration": (0.70, 0.78),
        "causal": (0.70, 0.22),
    }

    add_box(
        ax,
        positions["data"],
        "Data Inputs\n(PEP, DHS, ACS, RPC)",
        colors["data"],
    )
    add_box(
        ax,
        positions["diagnostic"],
        "Modules 1–2\nDescriptive +\nTime Series",
        colors["forecast"],
    )
    add_box(
        ax,
        positions["predictive"],
        "Modules 3–6\nPanel, Gravity,\nML, Robust",
        colors["forecast"],
    )
    add_box(
        ax,
        positions["duration"],
        "Module 8\nWave Duration",
        colors["forecast"],
        width=0.16,
    )
    add_box(
        ax,
        positions["causal"],
        "Module 7\nCausal Evidence",
        colors["causal"],
        width=0.16,
    )
    add_box(
        ax,
        positions["scenario"],
        "Module 9\nScenario Modeling\n+ Monte Carlo",
        colors["scenario"],
        width=0.20,
    )
    add_box(
        ax,
        positions["output"],
        "Outputs\nScenarios, PIs,\nForecasts",
        colors["output"],
        width=0.16,
    )

    add_arrow(ax, (0.18, 0.5), (0.21, 0.5))
    add_arrow(ax, (0.36, 0.5), (0.39, 0.5))
    add_arrow(ax, (0.56, 0.5), (0.60, 0.5))
    add_arrow(ax, (0.80, 0.5), (0.84, 0.5))
    add_arrow(ax, (0.70, 0.70), (0.70, 0.58))
    add_arrow(ax, (0.70, 0.30), (0.70, 0.42), dashed=True)

    output_base = FIGURES_DIR / "analysis_pipeline"
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved pipeline diagram to %s.[png|pdf]", output_base)
    return output_base.with_suffix(".pdf")


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    create_pipeline_diagram()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
