"""
Shared helpers for Census PEP (POPEST) archive scripts (ADR-034).

The POPEST archive is stored in a shared workspace directory configured via
`CENSUS_POPEST_DIR`. These helpers are used by CLI scripts under `scripts/data/`
to resolve paths, read/write the shared `catalog.yaml`, and compute stable file
and schema fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PopestPaths:
    """Resolved filesystem locations under the shared POPEST directory."""

    base_dir: Path
    catalog_path: Path
    raw_dir: Path
    parquet_dir: Path
    raw_archives_dir: Path
    derived_dir: Path
    metadata_dir: Path


def configure_logging(verbose: bool) -> None:
    """Configure a basic logging setup for CLI scripts."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_popest_paths(popest_dir: str | None) -> PopestPaths:
    """Resolve shared POPEST directory paths from `--popest-dir` or `CENSUS_POPEST_DIR`."""

    configured = popest_dir or os.getenv("CENSUS_POPEST_DIR")
    if not configured:
        raise ValueError("CENSUS_POPEST_DIR is not set and --popest-dir was not provided.")

    base = Path(configured).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"POPEST directory not found: {base}")

    return PopestPaths(
        base_dir=base,
        catalog_path=base / "catalog.yaml",
        raw_dir=base / "raw",
        parquet_dir=base / "parquet",
        raw_archives_dir=base / "raw-archives",
        derived_dir=base / "derived",
        metadata_dir=base / "metadata",
    )


def load_catalog(path: Path) -> dict[str, Any]:
    """Load the shared YAML catalog."""

    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def write_catalog(path: Path, catalog: dict[str, Any]) -> None:
    """Write the shared YAML catalog, updating the `last_updated` timestamp."""

    catalog["last_updated"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(yaml.safe_dump(catalog, sort_keys=False))


def md5_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute the MD5 checksum of a file."""

    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def schema_fingerprint(schema: pa.Schema) -> str:
    """Compute a stable schema fingerprint for an Arrow schema."""

    payload = {
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"sha256:{digest}"
