#!/usr/bin/env python3
"""
Download Census Population Estimates Program (PEP) data.

This script downloads historical population estimates from the Census Bureau's
FTP server and organizes them in the shared data directory according to ADR-034.

Data is stored in a shared workspace location to prevent duplication across
multiple repositories. The location is configured via the CENSUS_POPEST_DIR
environment variable.

Usage:
    # Set environment variable (or add to .env)
    export CENSUS_POPEST_DIR=~/workspace/shared-data/census/popest

    # Run download
    python scripts/data/download_census_pep.py [--dry-run] [--category CATEGORY]

Categories:
    all         Download everything (default)
    places      Place/city-level estimates (2000-2024)
    counties    County-level estimates (all decades)
    states      State-level estimates (all decades)
    docs        Technical documentation only

See: docs/governance/adrs/034-census-pep-data-archive.md
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Base URLs
BASE_URL = "https://www2.census.gov/programs-surveys/popest"
DATASETS_URL = f"{BASE_URL}/datasets"
DOCS_URL = f"{BASE_URL}/technical-documentation"

# Data directory from environment
# This points to the shared workspace location, not the repository
CENSUS_POPEST_DIR = os.getenv("CENSUS_POPEST_DIR")
if not CENSUS_POPEST_DIR:
    print("ERROR: CENSUS_POPEST_DIR environment variable not set.", file=sys.stderr)
    print(file=sys.stderr)
    print("Census PEP data is stored in a shared workspace directory to prevent", file=sys.stderr)
    print("duplication across repositories. Please set the environment variable:", file=sys.stderr)
    print(file=sys.stderr)
    print("  export CENSUS_POPEST_DIR=~/workspace/shared-data/census/popest", file=sys.stderr)
    print(file=sys.stderr)
    print("Or add it to your .env file:", file=sys.stderr)
    print(
        "  echo 'CENSUS_POPEST_DIR=~/workspace/shared-data/census/popest' >> .env", file=sys.stderr
    )
    print(file=sys.stderr)
    sys.exit(1)

DATA_DIR = Path(CENSUS_POPEST_DIR).expanduser()
RAW_DIR = DATA_DIR / "raw"
CATALOG_PATH = DATA_DIR / "catalog.yaml"


# Dataset definitions organized by category and decade
DATASETS: dict[str, list[dict[str, Any]]] = {
    # Place-level data (2000-2024 only)
    "places": [
        # 2020s postcensal
        {
            "id": "sub-est2024",
            "vintage": "2020-2024",
            "level": "place",
            "url": f"{DATASETS_URL}/2020-2024/cities/totals/sub-est2024.csv",
            "local_path": "2020-2024/place/sub-est2024.csv",
            "description": "City/place estimates 2020-2024 (postcensal)",
        },
        # 2010-2020 intercensal (REVISED - supersedes 2010-2019 postcensal)
        {
            "id": "sub-est2020int",
            "vintage": "2010-2020",
            "level": "place",
            "url": f"{DATASETS_URL}/2010-2020/intercensal/cities/sub-est2020int.csv",
            "local_path": "2010-2020/place/sub-est2020int.csv",
            "description": "City/place estimates 2010-2020 intercensal (8.9 MB)",
        },
        # 2010-2019 postcensal (superseded by intercensal above)
        {
            "id": "sub-est2019-all",
            "vintage": "2010-2019",
            "level": "place",
            "url": f"{DATASETS_URL}/2010-2019/cities/totals/sub-est2019_all.csv",
            "local_path": "2010-2019/place/sub-est2019_all.csv",
            "description": "City/place estimates 2010-2019 (postcensal, superseded)",
            "optional": True,  # Superseded by 2010-2020 intercensal
        },
        # 2000-2010 intercensal
        {
            "id": "sub-est00int",
            "vintage": "2000-2010",
            "level": "place",
            "url": f"{DATASETS_URL}/2000-2010/intercensal/cities/sub-est00int.csv",
            "local_path": "2000-2010/place/sub-est00int.csv",
            "description": "City/place estimates 2000-2010 (intercensal)",
        },
        {
            "id": "sub-est2010-alt",
            "vintage": "2000-2010",
            "level": "place",
            "url": f"{DATASETS_URL}/2000-2010/intercensal/cities/sub-est2010-alt.csv",
            "local_path": "2000-2010/place/sub-est2010-alt.csv",
            "description": "City/place estimates 2010 alternative",
        },
    ],
    # County-level data (all decades)
    "counties": [
        # 2020s postcensal
        {
            "id": "co-est2024-alldata",
            "vintage": "2020-2024",
            "level": "county",
            "url": f"{DATASETS_URL}/2020-2024/counties/totals/co-est2024-alldata.csv",
            "local_path": "2020-2024/county/co-est2024-alldata.csv",
            "description": "County estimates 2020-2024 (postcensal)",
        },
        # 2010-2020 intercensal (REVISED - supersedes 2010-2019 postcensal)
        {
            "id": "cc-est2020int-alldata",
            "vintage": "2010-2020",
            "level": "county",
            "url": f"{DATASETS_URL}/2010-2020/intercensal/county/asrh/cc-est2020int-alldata.csv",
            "local_path": "2010-2020/county/cc-est2020int-alldata.csv",
            "description": "County estimates 2010-2020 intercensal (LARGE: 169 MB)",
        },
        # 2010-2019 postcensal (superseded by intercensal above)
        {
            "id": "co-est2019-alldata",
            "vintage": "2010-2019",
            "level": "county",
            "url": f"{DATASETS_URL}/2010-2019/counties/totals/co-est2019-alldata.csv",
            "local_path": "2010-2019/county/co-est2019-alldata.csv",
            "description": "County estimates 2010-2019 (postcensal, superseded)",
            "optional": True,  # Superseded by 2010-2020 intercensal
        },
        # 2000-2010 intercensal
        {
            "id": "co-est00int-tot",
            "vintage": "2000-2010",
            "level": "county",
            "url": f"{DATASETS_URL}/2000-2010/intercensal/county/co-est00int-tot.csv",
            "local_path": "2000-2010/county/co-est00int-tot.csv",
            "description": "County totals 2000-2010 (intercensal, limited)",
        },
        # 2000-2009 comprehensive
        {
            "id": "co-est2009-alldata",
            "vintage": "2000-2009",
            "level": "county",
            "url": f"{DATASETS_URL}/2000-2009/counties/totals/co-est2009-alldata.csv",
            "local_path": "2000-2009/county/co-est2009-alldata.csv",
            "description": "County estimates 2000-2009 (comprehensive)",
        },
        # 1990s
        {
            "id": "co-99-10",
            "vintage": "1990-2000",
            "level": "county",
            "url": f"{DATASETS_URL}/1990-2000/counties/asrh/co-99-10.zip",
            "local_path": "1990-2000/county/co-99-10.zip",
            "description": "County estimates 1990-2000 (age/sex/race/Hispanic)",
        },
        # 1980s
        {
            "id": "comp8090",
            "vintage": "1980-1990",
            "level": "county",
            "url": f"{DATASETS_URL}/1980-1990/counties/totals/comp8090.zip",
            "local_path": "1980-1990/county/comp8090.zip",
            "description": "County estimates 1980-1990 (compressed)",
        },
        # 1970s
        {
            "id": "e7079co",
            "vintage": "1970-1980",
            "level": "county",
            "url": f"{DATASETS_URL}/1970-1980/national/asrh/e7079co.zip",
            "local_path": "1970-1980/county/e7079co.zip",
            "description": "County estimates 1970-1979 (compressed)",
        },
    ],
    # State-level data
    "states": [
        # 2020s postcensal
        {
            "id": "NST-EST2024-ALLDATA",
            "vintage": "2020-2024",
            "level": "state",
            "url": f"{DATASETS_URL}/2020-2024/state/totals/NST-EST2024-ALLDATA.csv",
            "local_path": "2020-2024/state/NST-EST2024-ALLDATA.csv",
            "description": "State estimates 2020-2024 (postcensal)",
        },
        # 2010-2020 intercensal (REVISED - supersedes 2010-2019 postcensal)
        {
            "id": "sc-est2020int-alldata5",
            "vintage": "2010-2020",
            "level": "state",
            "url": f"{DATASETS_URL}/2010-2020/intercensal/state/asrh/sc-est2020int-alldata5.csv",
            "local_path": "2010-2020/state/sc-est2020int-alldata5.csv",
            "description": "State estimates 2010-2020 intercensal (15 MB)",
        },
        {
            "id": "sc-est2020int-alldata6",
            "vintage": "2010-2020",
            "level": "state",
            "url": f"{DATASETS_URL}/2010-2020/intercensal/state/asrh/sc-est2020int-alldata6.csv",
            "local_path": "2010-2020/state/sc-est2020int-alldata6.csv",
            "description": "State estimates 2010-2020 intercensal detailed (17 MB)",
        },
        # 2010-2019 postcensal (superseded by intercensal above)
        {
            "id": "sc-est2019-alldata6",
            "vintage": "2010-2019",
            "level": "state",
            "url": f"{DATASETS_URL}/2010-2019/state/detail/sc-est2019-alldata6.csv",
            "local_path": "2010-2019/state/sc-est2019-alldata6.csv",
            "description": "State estimates 2010-2019 (postcensal, superseded)",
            "optional": True,  # Superseded by 2010-2020 intercensal
        },
        # 2000-2010 intercensal
        {
            "id": "st-est00int-alldata",
            "vintage": "2000-2010",
            "level": "state",
            "url": f"{DATASETS_URL}/2000-2010/intercensal/state/st-est00int-alldata.csv",
            "local_path": "2000-2010/state/st-est00int-alldata.csv",
            "description": "State estimates 2000-2010 (intercensal)",
        },
        # 1980s
        {
            "id": "st_int_asrh",
            "vintage": "1980-1990",
            "level": "state",
            "url": f"{DATASETS_URL}/1980-1990/state/asrh/st_int_asrh.zip",
            "local_path": "1980-1990/state/st_int_asrh.zip",
            "description": "State estimates 1980-1990 (age/sex/race/Hispanic)",
        },
        # 1970s
        {
            "id": "e7080sta",
            "vintage": "1970-1980",
            "level": "state",
            "url": f"{DATASETS_URL}/1970-1980/national/asrh/e7080sta.txt",
            "local_path": "1970-1980/state/e7080sta.txt",
            "description": "State estimates 1970-1980",
        },
    ],
    # Technical documentation
    "docs": [
        # File layouts
        {
            "id": "layout-2020-2024",
            "vintage": "2020-2024",
            "level": "docs",
            "url": f"{DOCS_URL}/file-layouts/2020-2024/SUB-EST2024.pdf",
            "local_path": "docs/file-layouts/SUB-EST2024-layout.pdf",
            "description": "File layout for 2020-2024 place estimates",
            "optional": True,
        },
        {
            "id": "layout-co-est2024",
            "vintage": "2020-2024",
            "level": "docs",
            "url": f"{DOCS_URL}/file-layouts/2020-2024/CO-EST2024-ALLDATA.pdf",
            "local_path": "docs/file-layouts/CO-EST2024-ALLDATA-layout.pdf",
            "description": "File layout for 2020-2024 county estimates",
            "optional": True,
        },
        # Methodology
        {
            "id": "methodology-2020s",
            "vintage": "2020-2024",
            "level": "docs",
            "url": f"{DOCS_URL}/methodology/2020-2024/2024-subco-meth.pdf",
            "local_path": "docs/methodology/2024-subcounty-methodology.pdf",
            "description": "Methodology for 2020s subcounty estimates",
            "optional": True,
        },
    ],
}


def download_file(url: str, dest_path: Path, dry_run: bool = False) -> dict[str, str | int | None]:
    """
    Download a file from URL to destination path.

    Returns dict with download metadata.
    """
    result: dict[str, str | int | None] = {
        "url": url,
        "dest": str(dest_path),
        "status": "pending",
        "size_bytes": None,
        "md5": None,
        "error": None,
    }

    if dry_run:
        print(f"  [DRY RUN] Would download: {url}")
        print(f"            To: {dest_path}")
        result["status"] = "dry_run"
        return result

    # Create parent directory
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Downloading: {url}")

        # Create request with user agent
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Census PEP Data Archive Script)"},
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()

            # Write to file
            with open(dest_path, "wb") as f:
                f.write(content)

            result["status"] = "success"
            result["size_bytes"] = len(content)
            result["md5"] = hashlib.md5(content).hexdigest()

            print(f"  Saved: {dest_path} ({len(content):,} bytes)")

    except urllib.error.HTTPError as e:
        result["status"] = "error"
        result["error"] = f"HTTP {e.code}: {e.reason}"
        print(f"  ERROR: {result['error']}")

    except urllib.error.URLError as e:
        result["status"] = "error"
        result["error"] = str(e.reason)
        print(f"  ERROR: {result['error']}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ERROR: {result['error']}")

    return result


def update_catalog(datasets_downloaded: list[dict]) -> None:
    """Update catalog.yaml with downloaded dataset metadata."""
    # Load existing catalog
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH) as f:
            catalog = yaml.safe_load(f) or {}
    else:
        catalog = {"version": "1.0", "datasets": [], "download_queue": []}

    # Ensure datasets list exists
    if "datasets" not in catalog:
        catalog["datasets"] = []

    # Update with new downloads
    existing_ids = {d.get("id") for d in catalog["datasets"]}

    for ds in datasets_downloaded:
        if ds.get("status") != "success":
            continue

        entry = {
            "id": ds["id"],
            "vintage": ds["vintage"],
            "level": ds["level"],
            "source_url": ds["url"],
            "downloaded": datetime.now(UTC).strftime("%Y-%m-%d"),
            "raw_file": ds["local_path"],
            "file_size_bytes": ds.get("size_bytes"),
            "md5": ds.get("md5"),
            "description": ds.get("description", ""),
            "status": "downloaded",
        }

        if ds["id"] in existing_ids:
            # Update existing entry
            for i, existing in enumerate(catalog["datasets"]):
                if existing.get("id") == ds["id"]:
                    catalog["datasets"][i] = entry
                    break
        else:
            catalog["datasets"].append(entry)

    # Update timestamp
    catalog["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Write back
    with open(CATALOG_PATH, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)

    print(f"\nCatalog updated: {CATALOG_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Census PEP population estimates data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--category",
        choices=["all", "places", "counties", "states", "docs"],
        default="all",
        help="Category of data to download (default: all)",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip datasets marked as optional",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between downloads (default: 0.5, set to 0 to disable)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Census PEP Data Download")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Category: {args.category}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Determine which datasets to download
    if args.category == "all":
        categories = ["places", "counties", "states", "docs"]
    else:
        categories = [args.category]

    # Collect all datasets
    to_download = []
    for cat in categories:
        for ds in DATASETS.get(cat, []):
            if args.skip_optional and ds.get("optional"):
                continue
            to_download.append({**ds, "category": cat})

    print(f"Datasets to download: {len(to_download)}")
    print()

    # Download each dataset
    results = []
    for i, ds in enumerate(to_download):
        print(f"\n[{ds['id']}] {ds['description']}")

        dest_path = RAW_DIR / ds["local_path"]

        # Check if already exists
        if dest_path.exists() and not args.dry_run:
            print(f"  Already exists: {dest_path}")
            size = dest_path.stat().st_size
            results.append(
                {
                    **ds,
                    "status": "exists",
                    "size_bytes": size,
                    "dest": str(dest_path),
                }
            )
            continue

        result = download_file(ds["url"], dest_path, dry_run=args.dry_run)
        results.append({**ds, **result})

        # Polite delay between downloads (but not after the last one)
        if args.delay > 0 and i < len(to_download) - 1 and result["status"] == "success":
            time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success = [r for r in results if r["status"] == "success"]
    exists = [r for r in results if r["status"] == "exists"]
    errors = [r for r in results if r["status"] == "error"]
    dry_run = [r for r in results if r["status"] == "dry_run"]

    print(f"  Downloaded: {len(success)}")
    print(f"  Already existed: {len(exists)}")
    print(f"  Errors: {len(errors)}")
    if dry_run:
        print(f"  Would download (dry run): {len(dry_run)}")

    # List successfully downloaded files
    if success:
        print("\n  Successfully downloaded:")
        for r in success:
            size_mb = r.get("size_bytes", 0) / 1024 / 1024
            print(f"    ✓ {r['id']} ({size_mb:.1f} MB)")

    # List files that already existed
    if exists:
        print("\n  Already existed (skipped):")
        for r in exists:
            size_mb = r.get("size_bytes", 0) / 1024 / 1024
            print(f"    - {r['id']} ({size_mb:.1f} MB)")

    # List errors
    if errors:
        print("\n  Failed downloads:")
        for r in errors:
            print(f"    ✗ {r['id']}: {r.get('error', 'Unknown error')}")

    # Update catalog (unless dry run)
    if not args.dry_run:
        update_catalog(results)

    # Calculate total size
    total_bytes = sum(int(r.get("size_bytes") or 0) for r in results)
    print(f"\n  Total size: {total_bytes / 1024 / 1024:.1f} MB")

    # Show catalog location
    if not args.dry_run and (success or exists):
        print("\n" + "=" * 70)
        print(f"Catalog updated: {CATALOG_PATH}")
        print("\nView downloaded datasets:")
        print(f"  cat {CATALOG_PATH}")
        print("\nView directory structure:")
        print(f"  tree -L 3 {DATA_DIR}")
        print("  # or: ls -lhR {DATA_DIR}")
        print("=" * 70)


if __name__ == "__main__":
    main()
