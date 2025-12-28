"""
Configuration file loader for cohort projections.

Loads and validates YAML configuration files.
"""

from pathlib import Path
from typing import Any

import yaml

from .logger import setup_logger

logger = setup_logger(__name__)


class ConfigLoader:
    """Load and manage project configuration."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize config loader.

        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self._configs = {}

    def load_config(self, config_name: str) -> dict[str, Any]:
        """
        Load a configuration file.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config

    def get_projection_config(self) -> dict[str, Any]:
        """Load main projection configuration."""
        return self.load_config("projection_config")

    def get_fertility_schedules(self) -> dict[str, Any]:
        """Load fertility schedules (when available)."""
        try:
            return self.load_config("fertility_schedules")
        except FileNotFoundError:
            logger.warning("Fertility schedules not found, will be calculated")
            return {}

    def get_mortality_schedules(self) -> dict[str, Any]:
        """Load mortality schedules (when available)."""
        try:
            return self.load_config("mortality_schedules")
        except FileNotFoundError:
            logger.warning("Mortality schedules not found, will be calculated")
            return {}

    def get_migration_assumptions(self) -> dict[str, Any]:
        """Load migration assumptions (when available)."""
        try:
            return self.load_config("migration_assumptions")
        except FileNotFoundError:
            logger.warning("Migration assumptions not found, will be calculated")
            return {}

    def get_parameter(self, *keys: str, default: Any = None) -> Any:
        """
        Get a specific parameter from configuration.

        Args:
            *keys: Nested keys to traverse (e.g., 'demographics', 'age_groups', 'type')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.get_projection_config()

        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value


def load_projection_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Convenience function to load projection configuration.

    Args:
        config_path: Path to config file (default: config/projection_config.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "projection_config.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)
