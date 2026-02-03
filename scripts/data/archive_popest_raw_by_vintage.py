#!/usr/bin/env python3
"""
Archive Census PEP (POPEST) raw staging files by vintage.

This is Phase 3 of ADR-034. For each vintage in the shared `catalog.yaml`, it
creates a single ZIP archive in `raw-archives/` containing:
  - raw source files exactly as downloaded (CSV/ZIP/TXT/PDF)
  - a machine-readable `manifest.json` with checksums + source URLs

After integrity verification, it can delete the uncompressed `raw/` staging
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import zipfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cohort_projections.data.popest_shared import (
    configure_logging,
    load_catalog,
    resolve_popest_paths,
    write_catalog,
)
from cohort_projections.utils.reproducibility import log_execution

LOGGER = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5_stream(stream, chunk_size: int = 4 * 1024 * 1024) -> str:
    md5 = hashlib.md5()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        md5.update(chunk)
    return md5.hexdigest()


def _build_manifest(
    *,
    vintage: str,
    datasets: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for ds in datasets:
        raw_file = ds.get("raw_file")
        if not raw_file:
            continue
        files.append(
            {
                "dataset_id": ds.get("id"),
                "level": ds.get("level"),
                "raw_file": raw_file,
                "file_size_bytes": ds.get("file_size_bytes"),
                "md5": ds.get("md5"),
                "source_url": ds.get("source_url"),
                "downloaded": ds.get("downloaded"),
                "description": ds.get("description"),
            }
        )

    return {
        "schema_version": "1.0",
        "vintage": vintage,
        "created_at": created_at,
        "files": files,
    }


def _verify_zip_against_manifest(zip_path: Path, manifest: dict[str, Any]) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        missing = []
        for entry in manifest.get("files", []):
            rel = entry["raw_file"]
            if rel not in zf.namelist():
                missing.append(rel)
        if missing:
            raise RuntimeError(f"Archive missing {len(missing)} files: {missing[:5]}")

        for entry in manifest.get("files", []):
            rel = entry["raw_file"]
            expected_md5 = entry.get("md5")
            expected_size = entry.get("file_size_bytes")

            info = zf.getinfo(rel)
            if expected_size is not None and info.file_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {rel}: expected {expected_size}, got {info.file_size}"
                )

            if expected_md5:
                with zf.open(rel) as f:
                    actual_md5 = _md5_stream(f)
                if actual_md5 != expected_md5:
                    raise RuntimeError(
                        f"MD5 mismatch for {rel}: expected {expected_md5}, got {actual_md5}"
                    )

        if "manifest.json" not in zf.namelist():
            raise RuntimeError("Archive missing manifest.json")


def _read_manifest_from_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read("manifest.json")
    return json.loads(raw)


def _delete_archived_files(raw_dir: Path, file_rels: list[str]) -> None:
    for rel in file_rels:
        path = raw_dir / rel
        if path.exists():
            path.unlink()

    # Clean up empty directories bottom-up
    for subdir in sorted(raw_dir.rglob("*"), reverse=True):
        if subdir.is_dir():
            with suppress(OSError):
                subdir.rmdir()

    if raw_dir.exists():
        with suppress(OSError):
            raw_dir.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive POPEST raw staging files by vintage (ADR-034 Phase 3)"
    )
    parser.add_argument(
        "--popest-dir", help="Override CENSUS_POPEST_DIR for the shared data directory."
    )
    parser.add_argument("--vintage", help="Archive only a single vintage (e.g., 2020-2024).")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing vintage archives."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write archives or update catalog."
    )
    parser.add_argument("--verify-only", action="store_true", help="Verify existing archives only.")
    parser.add_argument(
        "--delete-staging", action="store_true", help="Delete raw staging files after verification."
    )
    parser.add_argument(
        "--yes-delete",
        action="store_true",
        help="Required with --delete-staging to confirm deletion of raw staging files.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    configure_logging(args.verbose)
    paths = resolve_popest_paths(args.popest_dir)
    catalog = load_catalog(paths.catalog_path)

    datasets: list[dict[str, Any]] = list(catalog.get("datasets", []))
    by_vintage: dict[str, list[dict[str, Any]]] = {}
    for ds in datasets:
        v = ds.get("vintage")
        if not v:
            continue
        by_vintage.setdefault(v, []).append(ds)

    vintages = sorted(by_vintage.keys())
    if args.vintage:
        if args.vintage not in by_vintage:
            raise ValueError(f"Vintage not found in catalog: {args.vintage}")
        vintages = [args.vintage]

    if args.delete_staging and not args.yes_delete:
        raise ValueError("--delete-staging requires --yes-delete")

    paths.raw_archives_dir.mkdir(parents=True, exist_ok=True)

    with log_execution(
        __file__,
        parameters={
            "vintage": args.vintage,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "verify_only": args.verify_only,
            "delete_staging": args.delete_staging,
        },
        inputs=[paths.catalog_path],
        outputs=[paths.catalog_path],
    ):
        archives_index: list[dict[str, Any]] = list(catalog.get("raw_archives", []))
        archives_by_vintage: dict[str, dict[str, Any]] = {
            str(a["vintage"]): a for a in archives_index if "vintage" in a and a["vintage"]
        }

        for vintage in vintages:
            vintage_datasets = by_vintage[vintage]
            archive_name = f"{vintage}-raw.zip"
            archive_path = paths.raw_archives_dir / archive_name
            created_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            manifest = _build_manifest(
                vintage=vintage, datasets=vintage_datasets, created_at=created_at
            )
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
            manifest_sha256 = f"sha256:{_sha256_bytes(manifest_bytes)}"

            if args.verify_only:
                if not archive_path.exists():
                    raise FileNotFoundError(f"Archive not found: {archive_path}")
                _verify_zip_against_manifest(archive_path, manifest)
                LOGGER.info("Verified: %s", archive_path)
                continue

            if archive_path.exists() and not args.overwrite:
                LOGGER.info("Archive exists: %s", archive_path)
                existing_manifest = _read_manifest_from_zip(archive_path)
                _verify_zip_against_manifest(archive_path, existing_manifest)
                LOGGER.info("Verified archive integrity: %s", archive_path)

                if args.delete_staging:
                    LOGGER.warning("Deleting archived raw staging files for vintage=%s", vintage)
                    _delete_archived_files(
                        paths.raw_dir, [e["raw_file"] for e in existing_manifest.get("files", [])]
                    )

                if vintage not in archives_by_vintage:
                    archives_by_vintage[vintage] = {
                        "vintage": vintage,
                        "archive_file": f"raw-archives/{archive_name}",
                        "created_at": None,
                        "manifest_sha256": None,
                        "member_count": None,
                        "archive_size_bytes": archive_path.stat().st_size,
                    }

                continue

            if args.dry_run:
                LOGGER.info("Dry run: would create %s", archive_path)
                continue

            LOGGER.info("Creating archive: %s", archive_path)
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as zf:
                zf.writestr("manifest.json", manifest_bytes)

                for entry in manifest["files"]:
                    rel = entry["raw_file"]
                    src = paths.raw_dir / rel
                    if not src.exists():
                        raise FileNotFoundError(f"Missing raw file for archive: {src}")
                    zf.write(src, arcname=rel)

            _verify_zip_against_manifest(archive_path, manifest)
            LOGGER.info("Verified archive integrity: %s", archive_path)

            for ds in vintage_datasets:
                ds["raw_archive"] = f"raw-archives/{archive_name}"
                ds["raw_archived"] = datetime.now(UTC).strftime("%Y-%m-%d")

            archives_by_vintage[vintage] = {
                "vintage": vintage,
                "archive_file": f"raw-archives/{archive_name}",
                "created_at": created_at,
                "manifest_sha256": manifest_sha256,
                "member_count": len(manifest["files"]) + 1,
                "archive_size_bytes": archive_path.stat().st_size,
            }

            if args.delete_staging:
                LOGGER.warning("Deleting archived raw staging files for vintage=%s", vintage)
                _delete_archived_files(paths.raw_dir, [e["raw_file"] for e in manifest["files"]])

        if not args.dry_run:
            catalog["raw_archives"] = [
                archives_by_vintage[v] for v in sorted(archives_by_vintage.keys())
            ]
            write_catalog(paths.catalog_path, catalog)
            LOGGER.info("Updated catalog: %s", paths.catalog_path)

    # If the caller asked to delete staging and there are still files, remove the whole dir.
    if args.delete_staging and paths.raw_dir.exists() and not any(paths.raw_dir.rglob("*")):
        shutil.rmtree(paths.raw_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
