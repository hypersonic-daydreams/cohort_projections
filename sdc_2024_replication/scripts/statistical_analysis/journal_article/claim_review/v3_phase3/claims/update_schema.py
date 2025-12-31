#!/usr/bin/env python3
"""
Update claims manifest with new schema fields and add orphan claims to argument map.

New fields added to claims:
- source_category: "external" | "study_generated"
- citation_keys: [] (to be populated during citation linking pass)
- citation_accuracy: "unverified" | "verified" | "partial" | "unsupported" | "uncited"
- verification_status: "unverified" | "verified" | "disputed" | "outdated"
- verification_notes: ""

Orphan claims (not in any argument node) are added to argument_map.jsonl as standalone nodes.
"""

import json
from pathlib import Path

CLAIMS_PATH = Path(__file__).parent / "claims_manifest.jsonl"
ARGUMENT_MAP_PATH = Path(__file__).parent.parent / "argument_map" / "argument_map.jsonl"

# Heuristics for source_category classification
EXTERNAL_INDICATORS = [
    "population of",
    "ranks",
    "has experienced",
    "has stagnated",
    "persists",
    "literature",
    "research",
    "studies",
    "according to",
    "reported by",
    "Beine et al",
    "Wilson",
    "twentieth century",
    "twenty-first century",
    "oil boom",
    "Bakken",
]

STUDY_GENERATED_INDICATORS = [
    "coefficient of variation",
    "CV =",
    "p <",
    "p =",
    "F =",
    "R-squared",
    "chi-squared",
    "ATT",
    "elasticity",
    "Location Quotient",
    "LQ",
    "ARIMA",
    "ADF",
    "Monte Carlo",
    "scenario",
    "projection",
    "forecast",
    "Module",
    "difference-in-differences",
    "DiD",
    "regression",
    "Kaplan-Meier",
    "Cox",
    "hazard",
    "structural break",
]


def classify_source_category(claim_text: str, support_primary: str) -> str:
    """Heuristically classify whether claim is external or study-generated."""
    text_lower = claim_text.lower()

    # Check for study-generated indicators first (more specific)
    for indicator in STUDY_GENERATED_INDICATORS:
        if indicator.lower() in text_lower:
            return "study_generated"

    # Check for external indicators
    for indicator in EXTERNAL_INDICATORS:
        if indicator.lower() in text_lower:
            return "external"

    # Use support_primary as fallback
    if support_primary in ["external_data", "citation"]:
        return "external"
    elif support_primary in [
        "model_output",
        "quantitative_data",
        "methodology_reference",
    ]:
        return "study_generated"

    return "study_generated"  # Default to study_generated


def update_claims_schema():
    """Add new fields to all claims in manifest."""
    claims = []
    with open(CLAIMS_PATH) as f:
        for line in f:
            claim = json.loads(line)

            # Add new fields if not present
            if "source_category" not in claim:
                claim["source_category"] = classify_source_category(
                    claim.get("claim_text", ""), claim.get("support_primary", "")
                )

            if "citation_keys" not in claim:
                claim["citation_keys"] = []

            if "citation_accuracy" not in claim:
                # If no citations yet, mark as uncited for external claims
                if claim.get("source_category") == "external" and not claim.get(
                    "citation_keys"
                ):
                    claim["citation_accuracy"] = "uncited"
                else:
                    claim["citation_accuracy"] = "unverified"

            if "verification_status" not in claim:
                claim["verification_status"] = "unverified"

            if "verification_notes" not in claim:
                claim["verification_notes"] = ""

            claims.append(claim)

    # Write back
    with open(CLAIMS_PATH, "w") as f:
        for claim in claims:
            f.write(json.dumps(claim) + "\n")

    return claims


def find_orphan_claims(claims: list) -> list:
    """Find claims not referenced in any argument node."""
    # Load argument map and collect all referenced claim IDs
    referenced_claims = set()
    with open(ARGUMENT_MAP_PATH) as f:
        for line in f:
            node = json.loads(line)
            for cid in node.get("claim_ids", []):
                referenced_claims.add(cid)

    # Find orphans
    orphans = []
    for claim in claims:
        if claim["claim_id"] not in referenced_claims:
            orphans.append(claim)

    return orphans


def add_orphans_to_argument_map(orphans: list):
    """Add orphan claims as standalone nodes in argument map."""
    # Load existing argument map
    existing_nodes = []
    with open(ARGUMENT_MAP_PATH) as f:
        for line in f:
            existing_nodes.append(json.loads(line))

    # Find max argument ID
    max_id = 0
    for node in existing_nodes:
        aid = node.get("argument_id", "A0000")
        num = int(aid[1:])
        if num > max_id:
            max_id = num

    # Create new nodes for orphans
    new_nodes = []
    for i, claim in enumerate(orphans):
        node = {
            "argument_id": f"A{max_id + i + 1:04d}",
            "argument_group_id": "ORPHAN",
            "role": "orphan",
            "text": claim["claim_text"],
            "claim_ids": [claim["claim_id"]],
            "supports_argument_ids": [],
            "source": claim.get("source", {}),
            "source_category": claim.get("source_category", "unknown"),
            "status": "unlinked",
        }
        new_nodes.append(node)

    # Write back with new nodes
    with open(ARGUMENT_MAP_PATH, "w") as f:
        for node in existing_nodes:
            f.write(json.dumps(node) + "\n")
        for node in new_nodes:
            f.write(json.dumps(node) + "\n")

    return new_nodes


def main():
    print("=== Updating Claims Schema ===")
    claims = update_claims_schema()
    print(f"Updated {len(claims)} claims with new schema fields")

    # Count source categories
    external_count = sum(1 for c in claims if c.get("source_category") == "external")
    study_count = sum(
        1 for c in claims if c.get("source_category") == "study_generated"
    )
    print(f"  - External claims: {external_count}")
    print(f"  - Study-generated claims: {study_count}")

    print("\n=== Finding Orphan Claims ===")
    orphans = find_orphan_claims(claims)
    print(f"Found {len(orphans)} orphan claims not in argument map")

    # Count by section
    by_section = {}
    for o in orphans:
        section = o.get("source", {}).get("section", "Unknown")
        by_section[section] = by_section.get(section, 0) + 1
    for section, count in sorted(by_section.items(), key=lambda x: -x[1]):
        print(f"  - {section}: {count}")

    print("\n=== Adding Orphans to Argument Map ===")
    new_nodes = add_orphans_to_argument_map(orphans)
    print(
        f"Added {len(new_nodes)} orphan nodes (A{189+1:04d}-A{189+len(new_nodes):04d})"
    )

    print("\n=== Summary ===")
    print(f"Claims manifest: {len(claims)} claims with updated schema")
    print(f"Argument map: {189 + len(new_nodes)} total nodes")
    print("  - Structured arguments: 189 nodes")
    print(f"  - Orphan claims: {len(new_nodes)} nodes")


if __name__ == "__main__":
    main()
