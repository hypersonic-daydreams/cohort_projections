#!/usr/bin/env python3
"""
Create the analytical pipeline diagram for the journal article.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as path_effects

LOGGER = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Use standard text rendering to avoid system dependency issues
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 10,
})

@dataclass(frozen=True)
class Node:
    """A labeled rounded-rectangle node in the pipeline diagram."""

    key: str
    center: tuple[float, float]
    text: str
    color: str
    width: float
    height: float
    fontsize: int = 10

    @property
    def left(self) -> float:
        return self.center[0] - self.width / 2

    @property
    def right(self) -> float:
        return self.center[0] + self.width / 2

    @property
    def bottom(self) -> float:
        return self.center[1] - self.height / 2

    @property
    def top(self) -> float:
        return self.center[1] + self.height / 2


def add_box(ax, node: Node) -> None:
    """Add a labeled rounded box to the axes with a drop shadow."""
    x, y = node.center

    # Soft drop shadow
    shadow_offset = 0.008
    shadow = FancyBboxPatch(
        (node.left + shadow_offset, node.bottom - shadow_offset),
        node.width,
        node.height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="black",
        alpha=0.15,
        zorder=1,
        mutation_scale=0,
    )
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch(
        (node.left, node.bottom),
        node.width,
        node.height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=node.color,
        edgecolor="#555555",
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x,
        y,
        node.text,
        ha="center",
        va="center",
        fontsize=node.fontsize,
        color="#222222",
        wrap=True,
        zorder=3,
        linespacing=1.4,
    )





def add_arrow(ax, start, end, *, dashed: bool = False, label: str = None, label_pos: str = "right") -> None:
    """Add an arrow between two points."""
    # Balanced dash pattern: (on, off)
    linestyle = (0, (4, 3)) if dashed else "-"

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_width=5,head_length=9",
        mutation_scale=1,
        linewidth=1.4,
        color="#333333",
        linestyle=linestyle,
        shrinkA=0, # No shrink, let zorder handle overlap if any (or rely on precise spacing)
        shrinkB=0,
        zorder=2.5, # Bring arrows to front to avoid being hidden by boxes
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Adjust position
        if label_pos == "right":
            txt_x, txt_y = mid_x + 0.02, mid_y
            ha = "left"
        elif label_pos == "left":
            txt_x, txt_y = mid_x - 0.02, mid_y
            ha = "right"
        elif label_pos == "above":
            txt_x, txt_y = mid_x, mid_y + 0.02
            ha = "center"
        else: # below
            txt_x, txt_y = mid_x, mid_y - 0.02
            ha = "center"

        ax.text(txt_x, txt_y, label, fontsize=9, color="#222222", ha=ha, va="center", style="italic")


def create_pipeline_diagram() -> Path:
    """Generate the pipeline diagram figure."""
    fig, ax = plt.subplots(figsize=(13, 6)) # Increased height slightly for breathing room
    # Fix clipping by expanding limits slightly beyond 0-1
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    colors = {
        "gray": "#F0F0F0",       # Data / Outputs
        "blue_light": "#E3F2FD", # Diagnostic (Module 1-2)
        "blue_med": "#BBDEFB",   # Predictive (Module 3-6)
        "green_soft": "#C8E6C9", # Status/Scenario (Module 9)
        "purple_soft": "#E1BEE7",# Causal (Module 7)
        "teal_soft": "#B2DFDB",  # Duration (Module 8)
    }

    # Top-to-bottom main flow
    x_center = 0.50

    # Perfectly uniform vertical spacing (interval = 0.20)
    y_data = 0.90
    y_diag = 0.70
    y_pred = 0.50
    y_scenario = 0.30
    y_output = 0.10

    y_side = y_scenario

    # Define Nodes
    nodes: dict[str, Node] = {
        "data": Node(
            key="data",
            center=(x_center, y_data),
            text="DATA INPUTS\nTarget: PEP (net)\nAux: DHS, ACS, RPC",
            color=colors["gray"],
            width=0.30,
            height=0.12,
        ),
        "diagnostic": Node(
            key="diagnostic",
            center=(x_center, y_diag),
            text="MODULES 1–2\nDescriptives +\nTime-Series Baselines",
            color=colors["blue_light"],
            width=0.28,
            height=0.12,
        ),
        "predictive": Node(
            key="predictive",
            center=(x_center, y_pred),
            text="MODULES 3–6\nPredictive Models\n(Panel, Gravity, ML)",
            color=colors["blue_med"],
            width=0.30,
            height=0.12,
        ),
        "scenario": Node(
            key="scenario",
            center=(x_center, y_scenario),
            text="MODULE 9\nScenario Engine\n+ Monte Carlo",
            color=colors["green_soft"],
            width=0.28,
            height=0.14,
        ),
        "output": Node(
            key="output",
            center=(x_center, y_output),
            text="OUTPUTS\nScenario Paths,\nPrediction Intervals",
            color=colors["gray"],
            width=0.26,
            height=0.12,
        ),
        "duration": Node(
            key="duration",
            center=(0.90, y_side),
            text="MODULE 8\nWave Duration",
            color=colors["teal_soft"],
            width=0.18,
            height=0.11,
        ),
        "causal": Node(
            key="causal",
            center=(0.10, y_side),
            text="MODULE 7\nCausal Evidence",
            color=colors["purple_soft"],
            width=0.18,
            height=0.11,
        ),
    }

    # Draw arrows
    # Main spine - Calculate precise contact points
    # Box bottom is center_y - height/2
    # Box top is center_y + height/2
    draw_spine_arrow = lambda n1, n2: add_arrow(
        ax,
        (x_center, nodes[n1].center[1] - nodes[n1].height/2),
        (x_center, nodes[n2].center[1] + nodes[n2].height/2)
    )

    draw_spine_arrow("data", "diagnostic")
    draw_spine_arrow("diagnostic", "predictive")
    draw_spine_arrow("predictive", "scenario")
    draw_spine_arrow("scenario", "output")

    # Side inputs
    # Module 8 -> Module 9
    add_arrow(
        ax,
        (nodes["duration"].center[0] - nodes["duration"].width/2, nodes["duration"].center[1]), # Left edge of 8
        (nodes["scenario"].center[0] + nodes["scenario"].width/2, nodes["scenario"].center[1]), # Right edge of 9
    )

    # Module 7 -> Module 9 (Dashed)
    add_arrow(
        ax,
        (nodes["causal"].center[0] + nodes["causal"].width/2, nodes["causal"].center[1]), # Right edge of 7
        (nodes["scenario"].center[0] - nodes["scenario"].width/2, nodes["scenario"].center[1]), # Left edge of 9
        dashed=True,
    )

    # Draw boxes
    for node in nodes.values():
        add_box(ax, node)

    # Legend
    legend_y = 0.98
    legend_x = 0.05

    # 1. Forecasting flow
    add_arrow(ax, (legend_x, legend_y), (legend_x + 0.06, legend_y))
    ax.text(legend_x + 0.07, legend_y, "Forecasting flow", va="center", fontsize=9, color="#333333")

    # 2. Policy-evidence input
    add_arrow(ax, (legend_x, legend_y - 0.05), (legend_x + 0.06, legend_y - 0.05), dashed=True)
    ax.text(
        legend_x + 0.07,
        legend_y - 0.05,
        "Policy-evidence input",
        va="center",
        fontsize=9,
        color="#333333",
    )

    output_base = FIGURES_DIR / "analysis_pipeline"
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    # Save PDF
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.05)
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
