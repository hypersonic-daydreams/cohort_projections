"""Config loader re-export with project convenience helpers."""

from pathlib import Path
from typing import Any

import yaml
from project_utils.config import ConfigLoader as ConfigLoader

__all__ = ["ConfigLoader", "load_projection_config"]


def load_projection_config(config_path: Path | None = None) -> Any:
    """
    Convenience function to load projection configuration.

    Args:
        config_path: Path to config file (default: config/projection_config.yaml).

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "projection_config.yaml"

    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
