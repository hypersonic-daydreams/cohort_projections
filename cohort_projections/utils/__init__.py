"""
Utility modules for cohort projections.

Re-exports from project_utils for backward compatibility, plus project-specific utilities.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config_loader import ConfigLoader, load_projection_config
from .sdc_paths import (
    get_sdc_replication_candidates,
    resolve_sdc_rate_file,
    resolve_sdc_replication_root,
)

SetupLoggerFn = Callable[..., logging.Logger]
GetLoggerFromConfigFn = Callable[[str, Path | str], logging.Logger]

_project_setup_logger: SetupLoggerFn | None = None
_get_logger_from_config: GetLoggerFromConfigFn | None = None

try:
    from project_utils import get_logger_from_config as _project_get_logger_from_config
    from project_utils import setup_logger as _project_setup_logger_impl

    _get_logger_from_config = _project_get_logger_from_config
    _project_setup_logger = _project_setup_logger_impl
    _HAS_PROJECT_UTILS = True
except ModuleNotFoundError:
    _HAS_PROJECT_UTILS = False


def setup_logger(
    name: str,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console: bool = True,
    format_string: str | None = None,
    log_level: str | int | None = None,
    log_to_file: bool = False,
    log_to_console: bool | None = None,
    **_kwargs: Any,
) -> logging.Logger:
    """
    Set up a logger, delegating to project_utils when available.

    Args:
        name: Logger name (typically ``__name__``).
        log_file: Optional file destination.
        level: Numeric logging level fallback.
        console: Whether to log to console when using project_utils.
        format_string: Optional logging format override.
        log_level: Optional string/int level override.
        log_to_file: Whether to emit logs to file when using project_utils.
        log_to_console: Optional explicit console toggle for project_utils.
        **_kwargs: Forward-compatible passthrough for project_utils.

    Returns:
        Configured logger.
    """
    if _HAS_PROJECT_UTILS and _project_setup_logger is not None:
        return _project_setup_logger(
            name=name,
            log_file=log_file,
            level=level,
            console=console,
            format_string=format_string,
            log_level=log_level,
            log_to_file=log_to_file,
            log_to_console=log_to_console,
            **_kwargs,
        )

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
        assert _get_logger_from_config is not None
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
    "get_sdc_replication_candidates",
    "resolve_sdc_rate_file",
    "resolve_sdc_replication_root",
]
