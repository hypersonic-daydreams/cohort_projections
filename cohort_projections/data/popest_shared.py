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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_POPEST_ENV_VAR = "CENSUS_POPEST_DIR"


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


def _default_popest_candidates(home: Path | None = None) -> list[Path]:
    """Return compatibility search paths for the shared POPEST archive."""

    resolved_home = (home or Path.home()).expanduser()
    return [
        resolved_home / "workspace" / "shared-data" / "census" / "popest",
        resolved_home / "workspace" / "workspace" / "shared-data" / "census" / "popest",
    ]


def _format_missing_popest_error(searched_paths: Iterable[Path]) -> str:
    """Format a clear error message for an unresolved POPEST root."""

    searched = ", ".join(str(path) for path in searched_paths)
    return (
        "Unable to locate the shared Census POPEST archive. "
        f"Set {DEFAULT_POPEST_ENV_VAR} to the correct root "
        "(for example `/path/to/shared-data/census/popest`). "
        f"Searched: {searched}"
    )


def resolve_popest_root(popest_dir: str | None = None) -> Path:
    """Resolve the shared POPEST root from explicit config, env, or defaults.

    Resolution order:
    1. ``popest_dir`` argument, if provided (strict)
    2. ``CENSUS_POPEST_DIR`` environment variable, if set (strict)
    3. Compatibility defaults under ``~/workspace/...``
    """

    if popest_dir:
        base = Path(popest_dir).expanduser()
        if not base.exists():
            raise FileNotFoundError(f"POPEST directory not found: {base}")
        return base

    configured = os.getenv(DEFAULT_POPEST_ENV_VAR)
    if configured:
        base = Path(configured).expanduser()
        if not base.exists():
            raise FileNotFoundError(
                f"{DEFAULT_POPEST_ENV_VAR} points to a missing directory: {base}"
            )
        return base

    candidates = _default_popest_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(_format_missing_popest_error(candidates))


def configure_logging(verbose: bool) -> None:
    """Configure a basic logging setup for CLI scripts."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_popest_paths(popest_dir: str | None) -> PopestPaths:
    """Resolve shared POPEST directory paths from config, env, or defaults."""

    base = resolve_popest_root(popest_dir)

    return PopestPaths(
        base_dir=base,
        catalog_path=base / "catalog.yaml",
        raw_dir=base / "raw",
        parquet_dir=base / "parquet",
        raw_archives_dir=base / "raw-archives",
        derived_dir=base / "derived",
        metadata_dir=base / "metadata",
    )


def resolve_popest_file(
    relative_path: str | Path,
    popest_dir: str | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one file under the shared POPEST root.

    Parameters
    ----------
    relative_path:
        Path relative to the POPEST root, for example
        ``parquet/2020-2024/county/cc-est2024-agesex-all.parquet``.
    popest_dir:
        Optional explicit POPEST root override.
    must_exist:
        When ``True``, raise ``FileNotFoundError`` if the resolved file is
        missing.
    """

    base = resolve_popest_root(popest_dir)
    path = base / Path(relative_path)
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Required POPEST file not found: {path}. "
            f"Resolved from {DEFAULT_POPEST_ENV_VAR if os.getenv(DEFAULT_POPEST_ENV_VAR) else 'default shared-data search path'}."
        )
    return path


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
