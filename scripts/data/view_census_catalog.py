#!/usr/bin/env python3
"""
View the Census PEP data catalog.

Shows what datasets have been downloaded and their metadata.

Usage:
    python scripts/data/view_census_catalog.py
"""

import os
import sys
from pathlib import Path

import yaml


def main():
    # Get catalog path from environment
    census_popest_dir = os.getenv("CENSUS_POPEST_DIR")
    if not census_popest_dir:
        print("ERROR: CENSUS_POPEST_DIR environment variable not set.", file=sys.stderr)
        print(file=sys.stderr)
        print("Set it to your Census PEP data directory:", file=sys.stderr)
        print("  export CENSUS_POPEST_DIR=~/workspace/shared-data/census/popest", file=sys.stderr)
        sys.exit(1)

    catalog_path = Path(census_popest_dir).expanduser() / "catalog.yaml"

    if not catalog_path.exists():
        print(f"ERROR: Catalog not found at {catalog_path}", file=sys.stderr)
        print(file=sys.stderr)
        print("Run the download script first:", file=sys.stderr)
        print("  python scripts/data/download_census_pep.py --category all", file=sys.stderr)
        sys.exit(1)

    # Load catalog
    with open(catalog_path) as f:
        catalog = yaml.safe_load(f)

    # Display header
    print("=" * 80)
    print("CENSUS PEP DATA CATALOG")
    print("=" * 80)
    print(f"Location: {catalog_path.parent}")
    print(f"Last updated: {catalog.get('last_updated', 'Unknown')}")
    print(f"Version: {catalog.get('version', 'Unknown')}")
    print()

    # Get datasets
    datasets = catalog.get("datasets", [])
    if not datasets:
        print("No datasets downloaded yet.")
        print()
        print("Run the download script:")
        print("  python scripts/data/download_census_pep.py --category all")
        return

    # Group by level and vintage
    by_level: dict[str, list] = {}
    for ds in datasets:
        level = ds.get("level", "unknown")
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(ds)

    # Display by level
    total_size = 0
    for level in ["place", "county", "state", "docs"]:
        if level not in by_level:
            continue

        datasets_for_level = by_level[level]
        level_size = sum(ds.get("file_size_bytes", 0) or 0 for ds in datasets_for_level)
        total_size += level_size

        print(f"\n{level.upper()}-LEVEL DATA")
        print("-" * 80)

        for ds in sorted(datasets_for_level, key=lambda x: x.get("vintage", "")):
            size_mb = (ds.get("file_size_bytes", 0) or 0) / 1024 / 1024
            status = ds.get("status", "unknown")
            vintage = ds.get("vintage", "unknown")

            status_icon = "✓" if status == "downloaded" else "?"
            print(f"  {status_icon} {ds['id']:30s} {vintage:15s} {size_mb:6.1f} MB")

            # Show description if available
            if ds.get("description"):
                print(f"     └─ {ds['description']}")

    # Summary
    print("\n" + "=" * 80)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
