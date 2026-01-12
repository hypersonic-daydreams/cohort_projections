"""YAML configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Load and manage project YAML configuration files from a directory."""

    def __init__(self, config_dir: Path | str | None = None):
        """
        Initialize the loader.

        Args:
            config_dir: Directory containing `*.yaml` config files. If None, defaults to
                `./config` relative to the current working directory.
        """
        self.config_dir = Path(config_dir) if config_dir is not None else Path.cwd() / "config"
        self._configs: dict[str, dict[str, Any]] = {}

    def load_config(self, config_name: str) -> dict[str, Any]:
        """
        Load a YAML config file by stem name.

        Args:
            config_name: Filename stem, without `.yaml` extension.

        Returns:
            Parsed configuration dictionary.
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as handle:
            config = yaml.safe_load(handle) or {}

        config = _interpolate_env_vars(config)
        self._configs[config_name] = config
        return config

    def get_projection_config(self) -> dict[str, Any]:
        """Load the `projection_config.yaml` configuration."""
        return self.load_config("projection_config")

    @property
    def config(self) -> dict[str, Any]:
        """Return the projection config if available; otherwise an empty dict."""
        try:
            return self.get_projection_config()
        except FileNotFoundError:
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot-notation from the projection config.

        Args:
            key: Dot-notation key (e.g., `logging.level`).
            default: Value to return if the key is not present.

        Returns:
            The value if found, otherwise `default`.
        """
        value: Any = self.config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    def get_path(self, key: str, default: Path | None = None) -> Path:
        """
        Retrieve a path-like value and resolve it relative to `config_dir`.

        Args:
            key: Dot-notation key.
            default: Default path if not found.

        Returns:
            Resolved path.
        """
        raw = self.get(key)
        if raw is None:
            if default is None:
                raise KeyError(f"Missing config key: {key}")
            return default
        path = Path(str(raw))
        return path if path.is_absolute() else (self.config_dir / path).resolve()


def _interpolate_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
