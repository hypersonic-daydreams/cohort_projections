"""Path helpers for locating the SDC 2024 replication repository."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute project root for this repository."""
    return Path(__file__).resolve().parents[2]


def get_sdc_replication_candidates(project_root: Path | None = None) -> list[Path]:
    """Return ordered candidate paths for the SDC replication repository.

    Candidate order:
    1. `SDC_2024_REPLICATION_ROOT` environment variable (if set)
    2. sibling repo under the shared demography workspace
    3. in-repo fallback path (pre-extraction compatibility)

    Args:
        project_root: Optional project root override.

    Returns:
        Ordered, de-duplicated candidate paths.
    """
    root = project_root or get_project_root()

    candidates: list[Path] = []
    env_root = os.getenv("SDC_2024_REPLICATION_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(root.parent / "sdc_2024_replication")
    candidates.append(root / "sdc_2024_replication")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)

    return deduped


def resolve_sdc_replication_root(
    project_root: Path | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve the SDC replication repository root path.

    Args:
        project_root: Optional project root override.
        must_exist: If true, require an existing directory.

    Returns:
        Resolved path to the SDC replication repository.

    Raises:
        FileNotFoundError: If `must_exist=True` and no candidate exists.
    """
    candidates = get_sdc_replication_candidates(project_root=project_root)

    if not must_exist:
        return candidates[0]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate sdc_2024_replication repository. "
        f"Checked: {rendered}"
    )


def resolve_sdc_rate_file(
    filename: str,
    project_root: Path | None = None,
) -> Path:
    """Resolve a required CSV under `sdc_2024_replication/data/`.

    Args:
        filename: CSV filename (for example, `base_population_by_county.csv`).
        project_root: Optional project root override.

    Returns:
        Path to the requested CSV.

    Raises:
        FileNotFoundError: If the repository root or file is missing.
    """
    sdc_root = resolve_sdc_replication_root(project_root=project_root, must_exist=True)
    path = sdc_root / "data" / filename
    if not path.exists():
        raise FileNotFoundError(f"Required SDC rate file not found: {path}")
    return path
