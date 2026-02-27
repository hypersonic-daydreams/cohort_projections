"""Test helpers for locating the SDC 2024 replication repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from cohort_projections.utils.sdc_paths import resolve_sdc_replication_root


def get_sdc_repo_root(*, skip_if_missing: bool = True) -> Path:
    """Resolve SDC repo root for tests, optionally skipping when unavailable.

    Args:
        skip_if_missing: If true, skip the calling test/module when missing.

    Returns:
        Resolved path to `sdc_2024_replication`.

    Raises:
        FileNotFoundError: If missing and `skip_if_missing=False`.
    """
    try:
        return resolve_sdc_replication_root(must_exist=True)
    except FileNotFoundError:
        pass

    message = (
        "sdc_2024_replication repository not found. "
        "Set SDC_2024_REPLICATION_ROOT or place it as a sibling "
        "directory next to cohort_projections."
    )
    if skip_if_missing:
        pytest.skip(message, allow_module_level=True)
    raise FileNotFoundError(message)
