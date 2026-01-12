"""Logging setup utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logger(
    name: str,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console: bool = True,
    format_string: str | None = None,
    log_level: str | int | None = None,
    log_to_file: bool = True,
    log_to_console: bool | None = None,
    **_kwargs: Any,
) -> logging.Logger:
    """
    Set up a logger with optional file and console handlers.

    Args:
        name: Logger name (typically `__name__`).
        log_file: Optional path to a log file.
        level: Logging level.
        console: Whether to add a console handler.
        format_string: Optional log format string.

    Returns:
        Configured logger.
    """
    if log_level is not None:
        if isinstance(log_level, str):
            level = getattr(logging, log_level.upper(), level)
        elif isinstance(log_level, int):
            level = log_level

    if log_to_console is not None:
        console = bool(log_to_console)

    if not log_to_file:
        log_file = None

    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(format_string or _DEFAULT_FORMAT)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        _add_handler_if_missing(logger, file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(fmt)
        _add_handler_if_missing(logger, stream_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return an existing logger by name."""
    return logging.getLogger(name)


def get_logger_from_config(name: str, config_path: Path | str) -> logging.Logger:
    """
    Configure a logger from a YAML config file if logging settings exist.

    The function is intentionally defensive: if expected keys are missing it
    falls back to sane defaults rather than failing.

    Supported keys:
    - `logging.level`: string like "INFO"
    - `logging.file`: path to log file
    - `logging.console`: boolean
    - `logging.format`: format string

    Args:
        name: Logger name.
        config_path: Path to YAML config file.

    Returns:
        Configured logger.
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return setup_logger(name)

    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle) or {}

    logging_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}

    level_raw = logging_cfg.get("level", "INFO")
    if isinstance(level_raw, str):
        level = getattr(logging, level_raw.upper(), logging.INFO)
    else:
        level = logging.INFO

    log_file = logging_cfg.get("file")
    if isinstance(log_file, str):
        log_file = os.path.expandvars(log_file)

    console = bool(logging_cfg.get("console", True))
    fmt = logging_cfg.get("format")
    fmt = fmt if isinstance(fmt, str) else None

    return setup_logger(
        name=name, log_file=log_file, level=level, console=console, format_string=fmt
    )


def _add_handler_if_missing(logger: logging.Logger, handler: logging.Handler) -> None:
    for existing in logger.handlers:
        if type(existing) is type(handler) and getattr(existing, "baseFilename", None) == getattr(
            handler, "baseFilename", None
        ):
            return
    logger.addHandler(handler)
