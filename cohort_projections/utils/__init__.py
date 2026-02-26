"""
Utility modules for cohort projections.

Re-exports from project_utils for backward compatibility, plus project-specific utilities.
"""

import logging
from pathlib import Path
from typing import Any

from .config_loader import ConfigLoader, load_projection_config

try:
    from project_utils import get_logger_from_config as _get_logger_from_config
    from project_utils import setup_logger as setup_logger

    _HAS_PROJECT_UTILS = True
except ModuleNotFoundError:
    _HAS_PROJECT_UTILS = False

    def setup_logger(name: str, level: int = logging.INFO, **_kwargs: Any) -> logging.Logger:
        """
        Fallback logger setup when `project_utils` is unavailable.

        Args:
            name: Logger name (typically `__name__`).
            level: Logging level.
            **_kwargs: Ignored, for API compatibility with `project_utils.setup_logger`.

        Returns:
            Configured logger.
        """
        logger = logging.getLogger(name)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=level)
        logger.setLevel(level)
        return logger


try:
    from .bigquery_client import BigQueryClient, get_bigquery_client

    _HAS_BIGQUERY = True
except ModuleNotFoundError:
    _HAS_BIGQUERY = False

    class BigQueryClient:  # type: ignore[no-redef]
        """Placeholder when optional BigQuery dependencies are not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any):
            raise ModuleNotFoundError(
                "Optional BigQuery dependencies are not installed; "
                "install the appropriate extras to use BigQueryClient."
            )

    def get_bigquery_client(  # type: ignore[misc]
        config: dict[str, Any] | None = None,
    ) -> "BigQueryClient":
        raise ModuleNotFoundError(
            "Optional BigQuery dependencies are not installed; "
            "install the appropriate extras to use get_bigquery_client()."
        )


# Project root for default paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent


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

    if _HAS_PROJECT_UTILS:
        return _get_logger_from_config(name, config_path)

    return setup_logger(name)


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
