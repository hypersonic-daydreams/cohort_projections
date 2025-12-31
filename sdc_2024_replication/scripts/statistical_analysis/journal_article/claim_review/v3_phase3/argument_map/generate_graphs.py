#!/usr/bin/env python3
"""Generate Graphviz DOT files from argument_map.jsonl."""

import json
from pathlib import Path
from collections import defaultdict


def load_argument_map(path: Path) -> list:
    """Load argument nodes from JSONL."""
    nodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))
    return nodes


def role_to_color(role: str) -> str:
    """Map Toulmin roles to colors."""
    colors = {
        "claim": "#4CAF50",  # Green
        "grounds": "#2196F3",  # Blue
        "warrant": "#FF9800",  # Orange
        "backing": "#9C27B0",  # Purple
        "qualifier": "#607D8B",  # Grey
        "rebuttal": "#F44336",  # Red
    }
    return colors.get(role, "#FFFFFF")


def role_to_shape(role: str) -> str:
    """Map Toulmin roles to shapes."""
    shapes = {
        "claim": "box",
        "grounds": "ellipse",
        "warrant": "diamond",
        "backing": "parallelogram",
        "qualifier": "octagon",
        "rebuttal": "hexagon",
    }
    return shapes.get(role, "box")


def escape_label(text: str, max_len: int = 60) -> str:
    """Escape and truncate text for DOT labels."""
    text = text.replace('"', '\\"').replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def generate_group_dot(nodes: list, group_id: str) -> str:
    """Generate DOT for a single argument group."""
    group_nodes = [n for n in nodes if n.get("argument_group_id") == group_id]
    if not group_nodes:
        return ""

    lines = [f"digraph {group_id} {{"]
    lines.append("  rankdir=TB;")
    lines.append('  node [fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')
    lines.append("")

    # Add nodes
    for node in group_nodes:
        aid = node["argument_id"]
        role = node.get("role", "unknown")
        text = escape_label(node.get("text", ""))
        color = role_to_color(role)
        shape = role_to_shape(role)
        lines.append(
            f'  {aid} [label="{aid}\\n{role}\\n{text}", shape={shape}, style=filled, fillcolor="{color}"];'
        )

    lines.append("")

    # Add edges
    for node in group_nodes:
        aid = node["argument_id"]
        for target in node.get("supports_argument_ids", []):
            lines.append(f'  {aid} -> {target} [label="supports"];')
        for target in node.get("rebuts_argument_ids", []):
            lines.append(
                f'  {aid} -> {target} [label="rebuts", style=dashed, color=red];'
            )

    lines.append("}")
    return "\n".join(lines)


def generate_full_paper_dot(nodes: list) -> str:
    """Generate DOT for the full paper argument structure."""
    lines = ["digraph FullPaper {"]
    lines.append("  rankdir=LR;")
    lines.append('  node [fontname="Helvetica", fontsize=8];')
    lines.append('  edge [fontname="Helvetica", fontsize=6];')
    lines.append("  compound=true;")
    lines.append("")

    # Group nodes by argument_group_id
    groups = defaultdict(list)
    for node in nodes:
        gid = node.get("argument_group_id", "unknown")
        groups[gid].append(node)

    # Create subgraphs for each group
    for gid, group_nodes in sorted(groups.items()):
        lines.append(f"  subgraph cluster_{gid} {{")
        lines.append(f'    label="{gid}";')
        lines.append("    style=rounded;")
        lines.append('    bgcolor="#f0f0f0";')

        for node in group_nodes:
            aid = node["argument_id"]
            role = node.get("role", "unknown")
            color = role_to_color(role)
            lines.append(
                f'    {aid} [label="{aid}", shape=circle, style=filled, fillcolor="{color}"];'
            )

        lines.append("  }")
        lines.append("")

    # Add edges (only cross-group edges visible at this level)
    for node in nodes:
        aid = node["argument_id"]
        for target in node.get("supports_argument_ids", []):
            lines.append(f"  {aid} -> {target};")

    lines.append("}")
    return "\n".join(lines)


def main():
    map_path = Path("argument_map.jsonl")
    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)

    nodes = load_argument_map(map_path)
    print(f"Loaded {len(nodes)} argument nodes")

    # Get unique group IDs
    groups = sorted(set(n.get("argument_group_id", "unknown") for n in nodes))
    print(f"Found {len(groups)} argument groups: {groups}")

    # Generate per-group DOT files
    for gid in groups:
        dot = generate_group_dot(nodes, gid)
        if dot:
            (output_dir / f"{gid}.dot").write_text(dot)
            print(f"  Generated {gid}.dot")

    # Generate full paper DOT
    full_dot = generate_full_paper_dot(nodes)
    (output_dir / "full_paper.dot").write_text(full_dot)
    print("  Generated full_paper.dot")

    print(f"\nDOT files written to {output_dir}/")
    print("To render: dot -Tpng graphs/G001.dot -o graphs/G001.png")


if __name__ == "__main__":
    main()
