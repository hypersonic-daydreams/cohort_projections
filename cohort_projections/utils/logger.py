"""
Logging configuration for cohort projections.

Provides consistent logging across all modules with file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import yaml


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: project_root/logs)
        log_to_file: Whether to write logs to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        if log_dir is None:
            # Default to project root/logs
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger_from_config(
    name: str,
    config_path: Optional[Path] = None
) -> logging.Logger:
    """
    Get logger configured from YAML config file.

    Args:
        name: Logger name
        config_path: Path to config YAML (default: config/projection_config.yaml)

    Returns:
        Configured logger
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "projection_config.yaml"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        logging_config = config.get("logging", {})
        return setup_logger(
            name=name,
            log_level=logging_config.get("level", "INFO"),
            log_dir=Path(logging_config.get("log_directory", "logs")),
            log_to_file=logging_config.get("log_to_file", True)
        )
    except Exception as e:
        # Fallback to default logger if config fails
        print(f"Warning: Could not load logging config: {e}")
        return setup_logger(name)
