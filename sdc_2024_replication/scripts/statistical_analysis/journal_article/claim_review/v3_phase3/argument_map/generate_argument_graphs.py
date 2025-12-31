#!/usr/bin/env python3
"""Generate Graphviz DOT graphs from argument_map.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROLE_STYLES: Dict[str, Dict[str, str]] = {
    "claim": {"shape": "box", "fillcolor": "#e8f0fe"},
    "grounds": {"shape": "ellipse", "fillcolor": "#e6f4ea"},
    "warrant": {"shape": "diamond", "fillcolor": "#fff4e5"},
    "backing": {"shape": "hexagon", "fillcolor": "#fce8e6"},
    "qualifier": {"shape": "note", "fillcolor": "#f3e8fd"},
    "rebuttal": {"shape": "octagon", "fillcolor": "#fde7f3"},
    "missing": {"shape": "box", "fillcolor": "#f0f0f0"},
}


def configure_logging(verbose: bool) -> None:
    """Configure logging for the script."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of records."""

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logging.warning(
                    "Skipping invalid JSON on line %d: %s", line_number, exc
                )
    return records


def normalize_text(text: str, max_length: int) -> str:
    """Normalize whitespace and truncate text for labels."""

    normalized = " ".join(text.split())
    if max_length > 0 and len(normalized) > max_length:
        return normalized[: max_length - 3].rstrip() + "..."
    return normalized


def escape_label(text: str) -> str:
    """Escape labels for Graphviz DOT output."""

    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_nodes(
    records: Iterable[Dict[str, Any]], max_label_length: int
) -> Dict[str, Dict[str, Any]]:
    """Build a node map keyed by argument_id."""

    nodes: Dict[str, Dict[str, Any]] = {}
    for record in records:
        argument_id = record.get("argument_id")
        if not argument_id:
            logging.warning("Skipping record without argument_id: %s", record)
            continue
        text = normalize_text(str(record.get("text", "")), max_label_length)
        record = dict(record)
        record["_label_text"] = text
        nodes[argument_id] = record
    return nodes


def add_missing_nodes(nodes: Dict[str, Dict[str, Any]], targets: Iterable[str]) -> None:
    """Add placeholder nodes for missing references."""

    for target in targets:
        if target not in nodes:
            nodes[target] = {
                "argument_id": target,
                "role": "missing",
                "_label_text": f"MISSING {target}",
            }


def collect_edges(nodes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Collect support and rebuttal edges from nodes."""

    edges: List[Tuple[str, str, str]] = []
    for node in nodes.values():
        source_id = node.get("argument_id")
        if not source_id:
            continue
        for target in node.get("supports_argument_ids", []) or []:
            edges.append((source_id, target, "supports"))
        for target in node.get("rebuts_argument_ids", []) or []:
            edges.append((source_id, target, "rebuts"))
    return edges


def render_dot(
    nodes: Dict[str, Dict[str, Any]],
    edges: Iterable[Tuple[str, str, str]],
    cluster_by_group: bool,
) -> str:
    """Render Graphviz DOT content for the given nodes and edges."""

    lines: List[str] = [
        "digraph ArgumentGraph {",
        '  graph [rankdir=LR, fontname="Helvetica", fontsize=12];',
        '  node [fontname="Helvetica", fontsize=10, style=filled, fillcolor="#f7f7f7", color="#444444"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#555555"];',
        "",
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if cluster_by_group:
        for node in nodes.values():
            group_id = node.get("argument_group_id") or "UNGROUPED"
            grouped.setdefault(group_id, []).append(node)
    else:
        grouped["ALL"] = list(nodes.values())

    for group_id, group_nodes in grouped.items():
        if cluster_by_group:
            cluster_name = f"cluster_{group_id}"
            lines.append(f"  subgraph {cluster_name} {{")
            lines.append(f'    label="{escape_label(group_id)}";')
            lines.append("    style=dashed;")
            lines.append('    color="#bbbbbb";')

        for node in sorted(group_nodes, key=lambda n: n.get("argument_id", "")):
            argument_id = node.get("argument_id")
            role = node.get("role", "missing")
            label_text = node.get("_label_text", "")
            label = escape_label(f"{argument_id}\n{role}\n{label_text}")
            style = ROLE_STYLES.get(role, ROLE_STYLES["missing"])
            shape = style["shape"]
            fillcolor = style["fillcolor"]
            lines.append(
                f'    {argument_id} [label="{label}", shape={shape}, fillcolor="{fillcolor}"];'
            )

        if cluster_by_group:
            lines.append("  }")
        lines.append("")

    for source, target, label in edges:
        lines.append(f'  {source} -> {target} [label="{escape_label(label)}"];')

    lines.append("}")
    return "\n".join(lines)


def write_dot(path: Path, content: str) -> None:
    """Write DOT content to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    """Run the argument graph generator."""

    parser = argparse.ArgumentParser(
        description="Generate Graphviz DOT graphs from argument_map.jsonl"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "argument_map.jsonl",
        help="Path to argument_map.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "graphs",
        help="Directory for DOT outputs",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=180,
        help="Maximum label length before truncation (0 disables)",
    )
    parser.add_argument(
        "--per-group",
        action="store_true",
        help="Generate per-group DOT files",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="Comma-separated list of argument_group_id values to include",
    )
    parser.add_argument(
        "--no-clusters",
        action="store_true",
        help="Disable group clustering in the full graph",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(args.verbose)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    records = read_jsonl(args.input)
    nodes = build_nodes(records, args.max_label_length)

    edges = collect_edges(nodes)
    targets = [target for _, target, _ in edges]
    add_missing_nodes(nodes, targets)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_graph = render_dot(nodes, edges, cluster_by_group=not args.no_clusters)
    write_dot(output_dir / "argument_graph_all.dot", full_graph)
    logging.info("Wrote full graph to %s", output_dir / "argument_graph_all.dot")

    if args.per_group:
        selected_groups = {g.strip() for g in args.groups.split(",") if g.strip()}
        group_ids = sorted(
            {
                node.get("argument_group_id")
                for node in nodes.values()
                if node.get("argument_group_id")
            }
        )

        for group_id in group_ids:
            if selected_groups and group_id not in selected_groups:
                continue
            group_nodes = {
                k: v for k, v in nodes.items() if v.get("argument_group_id") == group_id
            }
            group_edges = [
                (src, tgt, label)
                for src, tgt, label in edges
                if src in group_nodes and tgt in group_nodes
            ]
            group_graph = render_dot(group_nodes, group_edges, cluster_by_group=False)
            filename = f"argument_graph_{group_id}.dot"
            write_dot(output_dir / filename, group_graph)
            logging.info("Wrote group graph to %s", output_dir / filename)


if __name__ == "__main__":
    main()
