#!/usr/bin/env python3
"""Guide LLM agents in building complete Toulmin argument maps for a section.

This script prepares structured prompts for LLM-based argument mapping,
ensuring each argument group has the full Toulmin structure:
- Claim (central assertion)
- Grounds (evidence supporting the claim)
- Warrant (logical bridge: why grounds support claim)
- Backing (support for warrants, optional)
- Qualifier (scope limitations, optional)
- Rebuttal (counter-arguments/limitations, optional)

Usage:
    # List existing incomplete groups
    python argument_map/map_section_arguments.py --audit

    # Get claims for a section to map
    python argument_map/map_section_arguments.py --section Results

    # Get claims for a specific group to complete
    python argument_map/map_section_arguments.py --group G020
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Section to group ID mapping
SECTION_GROUPS = {
    "Data and Methods": [
        "G010",
        "G011",
        "G012",
        "G013",
        "G014",
        "G015",
        "G016",
        "G017",
        "G018",
        "G019",
    ],
    "Results": ["G020", "G021", "G022", "G023", "G024", "G025", "G026", "G027"],
    "Discussion": ["G030", "G031", "G032", "G033", "G034"],
    "Conclusion": ["G040", "G041", "G042", "G043", "G044", "G045"],
    "Appendix": ["G050", "G051", "G052", "G053", "G054", "G055"],
}

# Section to claim ID range mapping (approximate)
SECTION_CLAIMS = {
    "Data and Methods": (144, 272),
    "Results": (273, 341),
    "Discussion": (342, 377),
    "Conclusion": (91, 143),
    "Appendix": (378, 452),
}

TOULMIN_GUIDANCE = """
## Toulmin Argument Structure

Each argument group should have AT MINIMUM:
1. **Claim** - The central assertion being argued (supports_argument_ids: [])
2. **Grounds** - Evidence that supports the claim (supports_argument_ids: [claim_id])
3. **Warrant** - The logical bridge explaining WHY the grounds support the claim
   (supports_argument_ids: [claim_id])

Optional but recommended:
4. **Backing** - Additional support for the warrant (supports_argument_ids: [warrant_id])
5. **Qualifier** - Scope limitations ("typically", "in most cases")
   (supports_argument_ids: [claim_id])
6. **Rebuttal** - Counter-arguments or exceptions (supports_argument_ids: [claim_id])

## Key Principles

1. GROUNDS are EVIDENCE (data, statistics, observations from the paper)
2. WARRANTS are REASONING (why the evidence supports the conclusion)
3. Every claim needs BOTH grounds AND warrant - data alone is not an argument

## Example Structure

Claim: "North Dakota's migration is highly volatile"
  ↑
Grounds: "CV of 82.5%, range from 30 to 3,000 migrants"
  ↑
Warrant: "A CV this high means annual flows can deviate from the mean by nearly
          the mean itself, making prediction extremely difficult"
  ↑
Backing: "National CV is only 73.5%, showing state-level amplification"
"""


def load_argument_map(path: Path) -> list[dict]:
    """Load argument nodes from JSONL."""
    nodes = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    nodes.append(json.loads(line))
    return nodes


def load_claims_manifest(path: Path) -> list[dict]:
    """Load claims from JSONL."""
    claims = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    claims.append(json.loads(line))
    return claims


def audit_completeness(nodes: list[dict]) -> dict:
    """Audit argument groups for Toulmin completeness."""
    groups = defaultdict(list)
    for node in nodes:
        groups[node.get("argument_group_id", "unknown")].append(node)

    results = {"complete": [], "incomplete": []}

    for gid, members in sorted(groups.items()):
        roles = {m["role"] for m in members}
        has_claim = "claim" in roles
        has_grounds = "grounds" in roles
        has_warrant = "warrant" in roles

        info = {
            "group_id": gid,
            "node_count": len(members),
            "roles": sorted(roles),
            "has_claim": has_claim,
            "has_grounds": has_grounds,
            "has_warrant": has_warrant,
            "claim_ids": [],
        }

        # Collect claim IDs referenced by this group
        for m in members:
            info["claim_ids"].extend(m.get("claim_ids", []))

        if has_claim and has_grounds and has_warrant:
            results["complete"].append(info)
        else:
            missing = []
            if not has_claim:
                missing.append("claim")
            if not has_grounds:
                missing.append("grounds")
            if not has_warrant:
                missing.append("warrant")
            info["missing"] = missing
            results["incomplete"].append(info)

    return results


def get_section_claims(claims: list[dict], section: str) -> list[dict]:
    """Get claims for a section."""
    return [c for c in claims if c.get("source", {}).get("section") == section]


def get_claims_by_ids(claims: list[dict], claim_ids: list[str]) -> list[dict]:
    """Get claims by their IDs."""
    id_set = set(claim_ids)
    return [c for c in claims if c.get("claim_id") in id_set]


def format_claim_for_prompt(claim: dict) -> str:
    """Format a claim for inclusion in a prompt."""
    cid = claim.get("claim_id", "?")
    text = claim.get("claim_text", "")
    ctype = claim.get("claim_type", "unknown")
    support = claim.get("support_primary", "")
    return f"  {cid} ({ctype}): {text}\n    [Support type: {support}]"


def generate_section_prompt(
    section: str,
    claims: list[dict],
    existing_nodes: list[dict],
    groups: list[str],
) -> str:
    """Generate a prompt for mapping a section's arguments."""
    section_claims = get_section_claims(claims, section)

    # Find which groups are incomplete
    audit = audit_completeness(existing_nodes)
    incomplete_groups = [g for g in audit["incomplete"] if g["group_id"] in groups]

    lines = [
        f"# Argument Mapping: {section}",
        "",
        TOULMIN_GUIDANCE,
        "",
        f"## Section Claims ({len(section_claims)} total)",
        "",
    ]

    for claim in section_claims:
        lines.append(format_claim_for_prompt(claim))
        lines.append("")

    lines.append("")
    lines.append("## Existing Argument Groups to Complete")
    lines.append("")

    for g in incomplete_groups:
        lines.append(f"### {g['group_id']} - Missing: {g['missing']}")
        lines.append(f"Current roles: {g['roles']}")
        lines.append(f"Referenced claims: {g['claim_ids']}")

        # Show the existing claim text for context
        existing_claim_ids = g["claim_ids"]
        if existing_claim_ids:
            lines.append("Existing claim content:")
            for cid in existing_claim_ids[:3]:  # Show first 3
                matching = [c for c in claims if c.get("claim_id") == cid]
                if matching:
                    lines.append(
                        f"  {cid}: {matching[0].get('claim_text', '')[:100]}..."
                    )
        lines.append("")

    lines.append("")
    lines.append("## Your Task")
    lines.append("")
    lines.append("For each incomplete group, identify from the claims list:")
    lines.append("1. Which claims serve as GROUNDS (evidence)?")
    lines.append("2. What is the WARRANT (logical bridge)?")
    lines.append("3. Any BACKING, QUALIFIER, or REBUTTAL?")
    lines.append("")
    lines.append("Output each new argument node as JSONL:")
    lines.append(
        '{"argument_id": "A####", "argument_group_id": "G###", "role": "grounds|warrant|backing|qualifier|rebuttal", "text": "...", "claim_ids": ["C####"], "supports_argument_ids": ["A####"], "status": "draft"}'
    )

    return "\n".join(lines)


def generate_group_prompt(
    group_id: str,
    claims: list[dict],
    existing_nodes: list[dict],
) -> str:
    """Generate a prompt for completing a single argument group."""
    group_nodes = [n for n in existing_nodes if n.get("argument_group_id") == group_id]

    if not group_nodes:
        return f"No existing nodes found for group {group_id}"

    # Get claim IDs referenced by this group
    referenced_claim_ids = []
    for node in group_nodes:
        referenced_claim_ids.extend(node.get("claim_ids", []))

    # Determine section from first claim
    section = None
    for cid in referenced_claim_ids:
        matching = [c for c in claims if c.get("claim_id") == cid]
        if matching:
            section = matching[0].get("source", {}).get("section")
            break

    # Get all claims from this section for context
    section_claims = get_section_claims(claims, section) if section else []

    lines = [
        f"# Complete Argument Group: {group_id}",
        "",
        TOULMIN_GUIDANCE,
        "",
        "## Existing Nodes in This Group",
        "",
    ]

    for node in group_nodes:
        lines.append(f"**{node['argument_id']}** ({node['role']})")
        lines.append(f"  Text: {node.get('text', '')}")
        lines.append(f"  Claims: {node.get('claim_ids', [])}")
        lines.append(f"  Supports: {node.get('supports_argument_ids', [])}")
        lines.append("")

    # Identify what's missing
    roles = {n["role"] for n in group_nodes}
    missing = []
    if "claim" not in roles:
        missing.append("claim")
    if "grounds" not in roles:
        missing.append("grounds")
    if "warrant" not in roles:
        missing.append("warrant")

    lines.append(f"## Missing Roles: {missing}")
    lines.append("")

    lines.append(f"## Available Claims from {section} Section")
    lines.append("")

    for claim in section_claims:
        lines.append(format_claim_for_prompt(claim))
        lines.append("")

    lines.append("")
    lines.append("## Your Task")
    lines.append("")
    lines.append(f"Complete group {group_id} by adding the missing {missing}.")
    lines.append("")
    lines.append("For GROUNDS: Find claims that provide EVIDENCE for the main claim.")
    lines.append("For WARRANT: Explain WHY the evidence supports the claim.")
    lines.append("")
    lines.append("Output new argument nodes as JSONL.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Guide argument mapping with Toulmin structure"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit existing argument map for completeness",
    )
    parser.add_argument(
        "--section",
        type=str,
        choices=list(SECTION_GROUPS.keys()),
        help="Generate prompt for a section",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="Generate prompt for a specific group (e.g., G020)",
    )
    parser.add_argument(
        "--argument-map",
        type=Path,
        default=Path("argument_map/argument_map.jsonl"),
        help="Path to argument_map.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("claims/claims_manifest.jsonl"),
        help="Path to claims_manifest.jsonl",
    )
    args = parser.parse_args()

    nodes = load_argument_map(args.argument_map)
    claims = load_claims_manifest(args.manifest)

    if args.audit:
        audit = audit_completeness(nodes)
        print("=== Argument Map Audit ===")
        print(f"Total groups: {len(audit['complete']) + len(audit['incomplete'])}")
        print(f"Complete (claim+grounds+warrant): {len(audit['complete'])}")
        print(f"Incomplete: {len(audit['incomplete'])}")
        print()

        if audit["incomplete"]:
            print("=== Incomplete Groups ===")
            for g in audit["incomplete"]:
                print(
                    f"{g['group_id']}: {g['node_count']} nodes, missing {g['missing']}"
                )

    elif args.section:
        groups = SECTION_GROUPS.get(args.section, [])
        prompt = generate_section_prompt(args.section, claims, nodes, groups)
        print(prompt)

    elif args.group:
        prompt = generate_group_prompt(args.group, claims, nodes)
        print(prompt)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
