"""
Utility modules for cohort projections.

Re-exports from project_utils for backward compatibility, plus project-specific utilities.
"""

import logging
from pathlib import Path
from typing import Any

from project_utils import ConfigLoader as _ConfigLoader
from project_utils import get_logger_from_config as _get_logger_from_config

# Re-export from project_utils for backward compatibility
from project_utils import setup_logger

# Project-specific utilities
from .bigquery_client import BigQueryClient, get_bigquery_client
from .config_loader import load_projection_config

# Project root for default paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class ConfigLoader(_ConfigLoader):
    """
    Configuration loader with project-specific defaults.

    This is a wrapper around project_utils.ConfigLoader that provides
    the cohort_projections default config directory.
    """

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize config loader.

        Args:
            config_dir: Path to configuration directory. If None, uses
                       the project default (config/).
        """
        if config_dir is None:
            config_dir = _PROJECT_ROOT / "config"
        super().__init__(config_dir)

    def get_projection_config(self) -> dict[str, Any]:
        """Load main projection configuration."""
        return self.load_config("projection_config")

    def get_fertility_schedules(self) -> dict[str, Any]:
        """Load fertility schedules (when available)."""
        try:
            return self.load_config("fertility_schedules")
        except FileNotFoundError:
            return {}

    def get_mortality_schedules(self) -> dict[str, Any]:
        """Load mortality schedules (when available)."""
        try:
            return self.load_config("mortality_schedules")
        except FileNotFoundError:
            return {}

    def get_migration_assumptions(self) -> dict[str, Any]:
        """Load migration assumptions (when available)."""
        try:
            return self.load_config("migration_assumptions")
        except FileNotFoundError:
            return {}


def get_logger_from_config(name: str, config_path: Path | None = None) -> logging.Logger:
    """
    Get logger configured from YAML config file.

    This is a project-specific wrapper that provides a default config path
    for the cohort_projections project.

    Args:
        name: Logger name (typically __name__ from calling module)
        config_path: Path to config YAML file. If None, uses the project
                    default (config/projection_config.yaml)

    Returns:
        Configured logger
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "projection_config.yaml"

    return _get_logger_from_config(name, config_path)


__all__ = [
    # From project_utils (with project-specific wrappers)
    "ConfigLoader",
    "get_logger_from_config",
    "setup_logger",
    # Project-specific
    "load_projection_config",
    "BigQueryClient",
    "get_bigquery_client",
]
