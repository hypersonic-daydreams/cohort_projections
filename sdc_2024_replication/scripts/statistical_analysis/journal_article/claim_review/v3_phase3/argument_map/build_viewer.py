#!/usr/bin/env python3
"""Generate interactive HTML viewer from argument_map.jsonl.

This script reads the argument map and generates a self-contained HTML file
with an interactive Cytoscape.js visualization.

Usage:
    python argument_map/build_viewer.py
    python argument_map/build_viewer.py --output custom_viewer.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Section mapping based on argument group ranges
GROUP_SECTIONS = {
    "G001": "Abstract + Introduction",
    "G002": "Abstract + Introduction",
    "G003": "Abstract + Introduction",
    "G004": "Abstract + Introduction",
    "G005": "Abstract + Introduction",
    "G006": "Abstract + Introduction",
    "G007": "Abstract + Introduction",
    "G008": "Abstract + Introduction",
    "G009": "Abstract + Introduction",
    "G010": "Data and Methods",
    "G011": "Data and Methods",
    "G012": "Data and Methods",
    "G013": "Data and Methods",
    "G014": "Data and Methods",
    "G015": "Data and Methods",
    "G016": "Data and Methods",
    "G017": "Data and Methods",
    "G018": "Data and Methods",
    "G019": "Data and Methods",
    "G020": "Results",
    "G021": "Results",
    "G022": "Results",
    "G023": "Results",
    "G024": "Results",
    "G025": "Results",
    "G026": "Results",
    "G027": "Results",
    "G030": "Discussion",
    "G031": "Discussion",
    "G032": "Discussion",
    "G033": "Discussion",
    "G034": "Discussion",
    "G040": "Conclusion",
    "G041": "Conclusion",
    "G042": "Conclusion",
    "G043": "Conclusion",
    "G044": "Conclusion",
    "G045": "Conclusion",
    "G050": "Appendix",
    "G051": "Appendix",
    "G052": "Appendix",
    "G053": "Appendix",
    "G054": "Appendix",
    "G055": "Appendix",
}


def load_argument_map(path: Path) -> list[dict]:
    """Load argument map from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                # Infer section from group ID if not present
                if "section" not in record:
                    group_id = record.get("argument_group_id", "")
                    record["section"] = GROUP_SECTIONS.get(
                        group_id, "Abstract + Introduction"
                    )
                elif isinstance(record.get("source"), dict):
                    record["section"] = record["source"].get(
                        "section", "Abstract + Introduction"
                    )
                records.append(record)
    return records


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argument Map Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; color: #eee; height: 100vh;
            display: flex; flex-direction: column;
        }}
        header {{
            background: #16213e; padding: 1rem 2rem;
            border-bottom: 1px solid #0f3460;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 1rem;
        }}
        h1 {{ font-size: 1.5rem; color: #e94560; }}
        .controls {{ display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }}
        select, button {{
            padding: 0.5rem 1rem; border: 1px solid #0f3460; border-radius: 4px;
            background: #16213e; color: #eee; font-size: 0.9rem; cursor: pointer;
        }}
        select:hover, button:hover {{ background: #0f3460; }}
        button.active {{ background: #e94560; border-color: #e94560; }}
        .main-container {{ display: flex; flex: 1; overflow: hidden; }}
        #cy {{ flex: 1; background: #1a1a2e; }}
        .sidebar {{
            width: 400px; background: #16213e; border-left: 1px solid #0f3460;
            overflow-y: auto; padding: 1rem;
        }}
        .sidebar h2 {{ color: #e94560; margin-bottom: 1rem; font-size: 1.2rem; }}
        .node-info {{
            background: #1a1a2e; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;
        }}
        .node-info h3 {{ color: #00d9ff; margin-bottom: 0.5rem; font-size: 1rem; }}
        .node-info p {{ line-height: 1.6; color: #ccc; font-size: 0.9rem; }}
        .badge {{
            display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
            font-size: 0.75rem; font-weight: bold; margin-right: 0.5rem; margin-bottom: 0.5rem;
        }}
        .badge-claim {{ background: #4CAF50; color: white; }}
        .badge-grounds {{ background: #2196F3; color: white; }}
        .badge-warrant {{ background: #FF9800; color: white; }}
        .badge-backing {{ background: #9C27B0; color: white; }}
        .badge-qualifier {{ background: #607D8B; color: white; }}
        .badge-rebuttal {{ background: #F44336; color: white; }}
        .badge-orphan {{ background: #795548; color: white; }}
        .badge-external {{ background: #E91E63; color: white; }}
        .badge-study_generated {{ background: #00BCD4; color: white; }}
        .claim-ids {{ margin-top: 0.5rem; font-size: 0.8rem; color: #888; }}
        .legend {{
            display: flex; flex-wrap: wrap; gap: 0.5rem; padding: 1rem;
            background: #16213e; border-top: 1px solid #0f3460;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 50%; }}
        .legend-color.dashed {{ border: 2px dashed #795548; background: transparent; }}
        .stats {{
            font-size: 0.85rem; color: #888; margin-top: 1rem;
            padding-top: 1rem; border-top: 1px solid #0f3460;
        }}
        .stats span {{ color: #00d9ff; }}
        .search-box {{
            width: 200px; padding: 0.5rem; border: 1px solid #0f3460;
            border-radius: 4px; background: #1a1a2e; color: #eee;
        }}
        .help-text {{ font-size: 0.8rem; color: #888; margin-top: 1rem; line-height: 1.5; }}
        .filter-group {{ display: flex; gap: 0.5rem; align-items: center; }}
        .filter-group label {{ font-size: 0.8rem; color: #888; }}
        @media (max-width: 900px) {{
            .main-container {{ flex-direction: column; }}
            .sidebar {{ width: 100%; max-height: 300px; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Argument Map: North Dakota Migration Study</h1>
        <div class="controls">
            <select id="groupFilter"><option value="all">All Groups</option></select>
            <select id="sectionFilter">
                <option value="all">All Sections</option>
                <option value="Abstract + Introduction">Abstract + Introduction</option>
                <option value="Data and Methods">Data and Methods</option>
                <option value="Results">Results</option>
                <option value="Discussion">Discussion</option>
                <option value="Conclusion">Conclusion</option>
                <option value="Appendix">Appendix</option>
            </select>
            <select id="categoryFilter">
                <option value="all">All Categories</option>
                <option value="structured">Structured Only</option>
                <option value="orphan">Orphans Only</option>
                <option value="external">External Claims</option>
                <option value="study_generated">Study-Generated</option>
            </select>
            <input type="text" class="search-box" id="searchBox" placeholder="Search nodes...">
            <button id="resetBtn">Reset View</button>
            <button id="fitBtn">Fit to Screen</button>
        </div>
    </header>
    <div class="main-container">
        <div id="cy"></div>
        <div class="sidebar">
            <h2>Node Details</h2>
            <div id="nodeDetails">
                <p class="help-text">Click on a node to see its details. Drag to pan, scroll to zoom. Double-click a node to focus on its connections.</p>
                <p class="help-text" style="margin-top: 0.5rem;"><strong>Orphan claims</strong> (dashed border) are claims not yet linked to argument structure. <strong>External claims</strong> require citation verification.</p>
            </div>
            <div class="stats" id="stats">Loading statistics...</div>
        </div>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #4CAF50;"></div> Claim</div>
        <div class="legend-item"><div class="legend-color" style="background: #2196F3;"></div> Grounds</div>
        <div class="legend-item"><div class="legend-color" style="background: #FF9800;"></div> Warrant</div>
        <div class="legend-item"><div class="legend-color" style="background: #9C27B0;"></div> Backing</div>
        <div class="legend-item"><div class="legend-color" style="background: #607D8B;"></div> Qualifier</div>
        <div class="legend-item"><div class="legend-color" style="background: #F44336;"></div> Rebuttal</div>
        <div class="legend-item"><div class="legend-color dashed"></div> Orphan (unlinked)</div>
        <div class="legend-item"><div class="legend-color" style="background: #E91E63;"></div> External</div>
        <div class="legend-item"><div class="legend-color" style="background: #00BCD4;"></div> Study-Generated</div>
    </div>
    <script>
    const argumentData = {argument_data_json};
    const roleColors = {{
        'claim': '#4CAF50', 'grounds': '#2196F3', 'warrant': '#FF9800',
        'backing': '#9C27B0', 'qualifier': '#607D8B', 'rebuttal': '#F44336',
        'orphan': '#795548'
    }};
    const categoryColors = {{ 'external': '#E91E63', 'study_generated': '#00BCD4' }};
    function buildElements(data) {{
        const nodes = [], edges = [];
        data.forEach(arg => {{
            const isOrphan = arg.role === 'orphan' || arg.argument_group_id === 'ORPHAN';
            nodes.push({{
                data: {{
                    id: arg.argument_id, label: arg.argument_id, role: arg.role,
                    text: arg.text, group: arg.argument_group_id,
                    section: arg.section || 'Unknown',
                    claim_ids: arg.claim_ids || [],
                    source_category: arg.source_category || 'unknown',
                    isOrphan: isOrphan,
                    color: roleColors[arg.role] || '#888'
                }}
            }});
            if (arg.supports_argument_ids) {{
                arg.supports_argument_ids.forEach(targetId => {{
                    edges.push({{ data: {{ id: `${{arg.argument_id}}->${{targetId}}`, source: arg.argument_id, target: targetId }} }});
                }});
            }}
        }});
        return {{ nodes, edges }};
    }}
    const elements = buildElements(argumentData);
    const cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: [...elements.nodes, ...elements.edges],
        style: [
            {{ selector: 'node', style: {{ 'background-color': 'data(color)', 'label': 'data(label)', 'color': '#fff', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px', 'width': 40, 'height': 40, 'border-width': 2, 'border-color': '#fff' }} }},
            {{ selector: 'node[?isOrphan]', style: {{ 'border-style': 'dashed', 'border-width': 3, 'border-color': '#795548', 'width': 35, 'height': 35 }} }},
            {{ selector: 'node[source_category="external"]', style: {{ 'shape': 'diamond' }} }},
            {{ selector: 'node:selected', style: {{ 'border-width': 4, 'border-color': '#00d9ff', 'width': 50, 'height': 50 }} }},
            {{ selector: 'node.highlighted', style: {{ 'border-width': 3, 'border-color': '#e94560' }} }},
            {{ selector: 'node.faded', style: {{ 'opacity': 0.2 }} }},
            {{ selector: 'edge', style: {{ 'width': 2, 'line-color': '#555', 'target-arrow-color': '#555', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'arrow-scale': 1.2 }} }},
            {{ selector: 'edge.highlighted', style: {{ 'line-color': '#00d9ff', 'target-arrow-color': '#00d9ff', 'width': 3 }} }},
            {{ selector: 'edge.faded', style: {{ 'opacity': 0.1 }} }}
        ],
        layout: {{ name: 'cose', idealEdgeLength: 100, nodeOverlap: 20, refresh: 20, fit: true, padding: 30, randomize: false, componentSpacing: 100, nodeRepulsion: 400000, edgeElasticity: 100, nestingFactor: 5, gravity: 80, numIter: 1000, initialTemp: 200, coolingFactor: 0.95, minTemp: 1.0 }}
    }});
    const groups = [...new Set(argumentData.map(a => a.argument_group_id))].sort();
    const groupFilter = document.getElementById('groupFilter');
    groups.forEach(g => {{ const opt = document.createElement('option'); opt.value = g; opt.textContent = g; groupFilter.appendChild(opt); }});
    function updateStats() {{
        const visibleNodes = cy.nodes(':visible'), visibleEdges = cy.edges(':visible');
        const roleCounts = {{}}, orphanCount = visibleNodes.filter(n => n.data('isOrphan')).length;
        const externalCount = visibleNodes.filter(n => n.data('source_category') === 'external').length;
        visibleNodes.forEach(n => {{ const role = n.data('role'); roleCounts[role] = (roleCounts[role] || 0) + 1; }});
        let statsHtml = `<strong>Visible:</strong> <span>${{visibleNodes.length}}</span> nodes, <span>${{visibleEdges.length}}</span> edges`;
        statsHtml += `<br><strong>Orphans:</strong> <span>${{orphanCount}}</span> | <strong>External:</strong> <span>${{externalCount}}</span>`;
        statsHtml += `<br>${{Object.entries(roleCounts).map(([role, count]) => `<span class="badge badge-${{role}}">${{role}}: ${{count}}</span>`).join(' ')}}`;
        document.getElementById('stats').innerHTML = statsHtml;
    }}
    function showNodeDetails(node) {{
        const data = node.data();
        const categoryBadge = data.source_category ? `<span class="badge badge-${{data.source_category}}">${{data.source_category}}</span>` : '';
        const orphanBadge = data.isOrphan ? '<span class="badge badge-orphan">ORPHAN</span>' : '';
        document.getElementById('nodeDetails').innerHTML = `<div class="node-info"><h3>${{data.id}} <span class="badge badge-${{data.role}}">${{data.role}}</span> ${{categoryBadge}} ${{orphanBadge}}</h3><p><strong>Group:</strong> ${{data.group}}</p><p><strong>Section:</strong> ${{data.section}}</p><p style="margin-top: 0.5rem;">${{data.text}}</p><p class="claim-ids"><strong>Claims:</strong> ${{data.claim_ids.join(', ') || 'None'}}</p></div>`;
    }}
    function applyFilters() {{
        const groupVal = document.getElementById('groupFilter').value;
        const sectionVal = document.getElementById('sectionFilter').value;
        const categoryVal = document.getElementById('categoryFilter').value;
        cy.nodes().forEach(n => {{
            let show = true;
            if (groupVal !== 'all' && n.data('group') !== groupVal) show = false;
            if (sectionVal !== 'all' && n.data('section') !== sectionVal) show = false;
            if (categoryVal === 'structured' && n.data('isOrphan')) show = false;
            if (categoryVal === 'orphan' && !n.data('isOrphan')) show = false;
            if (categoryVal === 'external' && n.data('source_category') !== 'external') show = false;
            if (categoryVal === 'study_generated' && n.data('source_category') !== 'study_generated') show = false;
            if (show) n.show(); else n.hide();
        }});
        cy.edges().forEach(e => {{
            const s = cy.getElementById(e.data('source')), t = cy.getElementById(e.data('target'));
            if (s.visible() && t.visible()) e.show(); else e.hide();
        }});
        cy.fit(cy.nodes(':visible'), 30);
        updateStats();
    }}
    document.getElementById('groupFilter').addEventListener('change', applyFilters);
    document.getElementById('sectionFilter').addEventListener('change', applyFilters);
    document.getElementById('categoryFilter').addEventListener('change', applyFilters);
    document.getElementById('searchBox').addEventListener('input', (e) => {{
        const query = e.target.value.toLowerCase();
        cy.nodes().removeClass('highlighted faded'); cy.edges().removeClass('highlighted faded');
        if (query.length < 2) return;
        const matches = cy.nodes().filter(n => n.data('id').toLowerCase().includes(query) || n.data('text').toLowerCase().includes(query) || n.data('group').toLowerCase().includes(query) || (n.data('claim_ids') || []).join(' ').toLowerCase().includes(query));
        if (matches.length > 0) {{ cy.nodes().addClass('faded'); cy.edges().addClass('faded'); matches.removeClass('faded').addClass('highlighted'); matches.connectedEdges().removeClass('faded').addClass('highlighted'); matches.neighborhood('node').removeClass('faded'); }}
    }});
    cy.on('tap', 'node', (evt) => {{ showNodeDetails(evt.target); }});
    cy.on('dbltap', 'node', (evt) => {{
        const node = evt.target;
        cy.nodes().removeClass('highlighted faded'); cy.edges().removeClass('highlighted faded');
        cy.nodes().addClass('faded'); cy.edges().addClass('faded');
        node.removeClass('faded').addClass('highlighted');
        node.neighborhood().removeClass('faded'); node.connectedEdges().removeClass('faded').addClass('highlighted');
        cy.fit(node.neighborhood().add(node), 50);
    }});
    document.getElementById('resetBtn').addEventListener('click', () => {{
        cy.nodes().removeClass('highlighted faded').show(); cy.edges().removeClass('highlighted faded').show();
        document.getElementById('searchBox').value = ''; document.getElementById('groupFilter').value = 'all';
        document.getElementById('sectionFilter').value = 'all'; document.getElementById('categoryFilter').value = 'all';
        cy.fit(cy.nodes(), 30); updateStats();
    }});
    document.getElementById('fitBtn').addEventListener('click', () => {{ cy.fit(cy.nodes(':visible'), 30); }});
    updateStats();
    </script>
</body>
</html>
"""


def build_html(records: list[dict]) -> str:
    """Build HTML with embedded argument data."""
    # Prepare data for JSON embedding
    data_for_json = []
    for rec in records:
        # Determine section from source if available
        section = rec.get("section")
        if not section and isinstance(rec.get("source"), dict):
            section = rec["source"].get("section", "Unknown")
        if not section:
            group_id = rec.get("argument_group_id", "")
            section = GROUP_SECTIONS.get(group_id, "Unknown")

        data_for_json.append(
            {
                "argument_id": rec.get("argument_id", ""),
                "argument_group_id": rec.get("argument_group_id", ""),
                "role": rec.get("role", "claim"),
                "text": rec.get("text", ""),
                "claim_ids": rec.get("claim_ids", []),
                "supports_argument_ids": rec.get("supports_argument_ids", []),
                "section": section,
                "source_category": rec.get("source_category", "unknown"),
            }
        )

    return HTML_TEMPLATE.format(argument_data_json=json.dumps(data_for_json, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build interactive argument map viewer"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("argument_map/argument_map.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("argument_map/argument_map_viewer.html"),
        help="Output HTML file",
    )
    args = parser.parse_args()

    records = load_argument_map(args.input)
    html = build_html(records)

    with open(args.output, "w") as f:
        f.write(html)

    print(f"Generated {args.output} with {len(records)} nodes")


if __name__ == "__main__":
    main()
