#!/usr/bin/env python3
"""
Merge citation updates from individual section files into the main claims manifest.

This script:
1. Reads all citation_updates_*.jsonl files
2. Merges citation_keys and citation_accuracy into claims_manifest.jsonl
3. Creates a backup before modifying
4. Reports statistics on the merge
"""

import json
from pathlib import Path
from datetime import datetime
import shutil

CLAIMS_DIR = Path(__file__).parent
CLAIMS_PATH = CLAIMS_DIR / "claims_manifest.jsonl"


def load_citation_updates() -> dict:
    """Load all citation update files and combine into a single dict."""
    updates = {}
    update_files = list(CLAIMS_DIR.glob("citation_updates_*.jsonl"))

    print(f"Found {len(update_files)} citation update files:")
    for f in sorted(update_files):
        print(f"  - {f.name}")
        with open(f) as fp:
            for line in fp:
                if line.strip():
                    entry = json.loads(line)
                    claim_id = entry["claim_id"]
                    updates[claim_id] = {
                        "citation_keys": entry.get("citation_keys", []),
                        "citation_accuracy": entry.get(
                            "citation_accuracy", "unverified"
                        ),
                    }

    return updates


def merge_updates(updates: dict) -> tuple[list, dict]:
    """Merge citation updates into claims manifest."""
    # Create backup
    backup_path = CLAIMS_PATH.with_suffix(
        f".jsonl.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy(CLAIMS_PATH, backup_path)
    print(f"\nCreated backup: {backup_path.name}")

    # Read and update claims
    claims = []
    stats = {
        "total_claims": 0,
        "updated_claims": 0,
        "claims_with_citations": 0,
        "unique_citation_keys": set(),
        "by_section": {},
    }

    with open(CLAIMS_PATH) as f:
        for line in f:
            claim = json.loads(line)
            claim_id = claim["claim_id"]
            section = claim.get("source", {}).get("section", "Unknown")
            stats["total_claims"] += 1

            if section not in stats["by_section"]:
                stats["by_section"][section] = {"total": 0, "with_citations": 0}
            stats["by_section"][section]["total"] += 1

            if claim_id in updates:
                update = updates[claim_id]
                claim["citation_keys"] = update["citation_keys"]
                claim["citation_accuracy"] = update["citation_accuracy"]
                stats["updated_claims"] += 1

                if update["citation_keys"]:
                    stats["claims_with_citations"] += 1
                    stats["by_section"][section]["with_citations"] += 1
                    for key in update["citation_keys"]:
                        stats["unique_citation_keys"].add(key)

            claims.append(claim)

    # Write updated claims
    with open(CLAIMS_PATH, "w") as f:
        for claim in claims:
            f.write(json.dumps(claim) + "\n")

    return claims, stats


def main():
    print("=== Merging Citation Updates ===\n")

    updates = load_citation_updates()
    print(f"\nTotal citation updates loaded: {len(updates)}")

    claims, stats = merge_updates(updates)

    print("\n=== Merge Statistics ===")
    print(f"Total claims: {stats['total_claims']}")
    print(f"Updated claims: {stats['updated_claims']}")
    print(f"Claims with citations: {stats['claims_with_citations']}")
    print(f"Unique citation keys used: {len(stats['unique_citation_keys'])}")

    print("\n=== By Section ===")
    for section, data in sorted(
        stats["by_section"].items(), key=lambda x: -x[1]["total"]
    ):
        pct = (data["with_citations"] / data["total"] * 100) if data["total"] > 0 else 0
        print(
            f"  {section}: {data['with_citations']}/{data['total']} ({pct:.1f}%) have citations"
        )

    print("\n=== Citation Keys Used ===")
    for key in sorted(stats["unique_citation_keys"]):
        print(f"  - {key}")


if __name__ == "__main__":
    main()
