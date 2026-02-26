"""Version information for cohort projections."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to determine project version from pyproject.toml")
    return match.group(1)


try:
    __version__ = version("cohort-projections")
except PackageNotFoundError:
    __version__ = _read_pyproject_version()

__author__ = "North Dakota Population Projections Team"
__description__ = "Cohort component population projections for North Dakota"
