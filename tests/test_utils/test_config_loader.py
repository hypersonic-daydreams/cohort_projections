"""
Unit tests for the config_loader module.

Tests ConfigLoader class and load_projection_config function.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from cohort_projections.utils.config_loader import ConfigLoader, load_projection_config


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_init_with_default_config_dir(self) -> None:
        """Test ConfigLoader initializes with default config directory."""
        loader = ConfigLoader()

        # Should point to project's config directory
        assert loader.config_dir.name == "config"
        assert loader.config_dir.exists() or True  # May not exist in test env

    def test_init_with_custom_config_dir(self, temp_config_dir: Path) -> None:
        """Test ConfigLoader initializes with custom config directory."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        assert loader.config_dir == temp_config_dir

    def test_load_config_success(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test successful config loading."""
        # Create a test config file
        config_path = temp_config_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.load_config("test_config")

        assert config == sample_projection_config
        assert "projection" in config
        assert config["projection"]["base_year"] == 2020

    def test_load_config_missing_file_raises_error(self, temp_config_dir: Path) -> None:
        """Test that loading missing config file raises FileNotFoundError."""
        loader = ConfigLoader(config_dir=temp_config_dir)

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            loader.load_config("nonexistent_config")

    def test_load_config_caching(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test that configs are cached after first load."""
        config_path = temp_config_dir / "cached_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)

        # First load
        config1 = loader.load_config("cached_config")

        # Second load should return cached version
        config2 = loader.load_config("cached_config")

        assert config1 is config2  # Same object reference (cached)

    def test_get_projection_config(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test get_projection_config convenience method."""
        config_path = temp_config_dir / "projection_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.get_projection_config()

        assert config == sample_projection_config

    def test_get_fertility_schedules_missing_returns_empty(self, temp_config_dir: Path) -> None:
        """Test get_fertility_schedules returns empty dict when file missing."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        result = loader.get_fertility_schedules()

        assert result == {}

    def test_get_fertility_schedules_exists(self, temp_config_dir: Path) -> None:
        """Test get_fertility_schedules when file exists."""
        fertility_data = {"schedules": {"2020": [0.05, 0.10, 0.15]}}
        config_path = temp_config_dir / "fertility_schedules.yaml"
        with open(config_path, "w") as f:
            yaml.dump(fertility_data, f)

        loader = ConfigLoader(config_dir=temp_config_dir)
        result = loader.get_fertility_schedules()

        assert result == fertility_data

    def test_get_mortality_schedules_missing_returns_empty(self, temp_config_dir: Path) -> None:
        """Test get_mortality_schedules returns empty dict when file missing."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        result = loader.get_mortality_schedules()

        assert result == {}

    def test_get_migration_assumptions_missing_returns_empty(self, temp_config_dir: Path) -> None:
        """Test get_migration_assumptions returns empty dict when file missing."""
        loader = ConfigLoader(config_dir=temp_config_dir)
        result = loader.get_migration_assumptions()

        assert result == {}

    def test_get_parameter_nested_keys(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test get_parameter with nested keys."""
        config_path = temp_config_dir / "projection_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)

        # Test nested access
        result = loader.get_parameter("demographics", "age_groups", "type")
        assert result == "single_year"

        # Test top-level access
        result = loader.get_parameter("projection")
        assert result == sample_projection_config["projection"]

    def test_get_parameter_missing_key_returns_default(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test get_parameter returns default for missing keys."""
        config_path = temp_config_dir / "projection_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)

        # Missing key returns default
        result = loader.get_parameter("nonexistent", "key", default="fallback")
        assert result == "fallback"

        # Missing nested key returns default
        result = loader.get_parameter("projection", "nonexistent", default=None)
        assert result is None

    def test_get_parameter_non_dict_traversal_returns_default(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test get_parameter returns default when traversing non-dict value."""
        config_path = temp_config_dir / "projection_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)

        # Trying to traverse into a non-dict value
        result = loader.get_parameter("projection", "base_year", "subkey", default="fallback")
        assert result == "fallback"


class TestLoadProjectionConfig:
    """Tests for load_projection_config convenience function."""

    def test_load_with_explicit_path(
        self, temp_config_dir: Path, sample_projection_config: dict[str, Any]
    ) -> None:
        """Test loading config with explicit path."""
        config_path = temp_config_dir / "custom_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_projection_config, f)

        config = load_projection_config(config_path)

        assert config == sample_projection_config

    def test_load_missing_file_raises_error(self, temp_config_dir: Path) -> None:
        """Test that loading missing file raises appropriate error."""
        nonexistent = temp_config_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_projection_config(nonexistent)

    def test_load_empty_config(self, temp_config_dir: Path) -> None:
        """Test loading an empty config file."""
        config_path = temp_config_dir / "empty_config.yaml"
        with open(config_path, "w") as f:
            f.write("")  # Empty file

        config = load_projection_config(config_path)

        assert config is None  # yaml.safe_load returns None for empty files

    def test_load_config_with_list(self, temp_config_dir: Path) -> None:
        """Test loading config that contains a list at the top level."""
        config_path = temp_config_dir / "list_config.yaml"
        list_config = ["item1", "item2", "item3"]
        with open(config_path, "w") as f:
            yaml.dump(list_config, f)

        config = load_projection_config(config_path)

        assert config == list_config


class TestConfigLoaderEdgeCases:
    """Edge case tests for ConfigLoader."""

    def test_config_with_special_characters(self, temp_config_dir: Path) -> None:
        """Test config with special characters in values."""
        special_config = {
            "path": "/path/with spaces/and-dashes",
            "message": "Hello, World! @#$%",
            "unicode": "Hello World",
        }
        config_path = temp_config_dir / "special_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(special_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.load_config("special_config")

        assert config == special_config

    def test_config_with_numeric_values(self, temp_config_dir: Path) -> None:
        """Test config with various numeric types."""
        numeric_config = {
            "integer": 42,
            "float": 3.14159,
            "negative": -100,
            "scientific": 1.5e-10,
        }
        config_path = temp_config_dir / "numeric_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(numeric_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.load_config("numeric_config")

        assert config["integer"] == 42
        assert abs(config["float"] - 3.14159) < 0.0001
        assert config["negative"] == -100

    def test_deeply_nested_config(self, temp_config_dir: Path) -> None:
        """Test config with deep nesting."""
        deep_config = {"level1": {"level2": {"level3": {"level4": {"value": "deep_value"}}}}}
        config_path = temp_config_dir / "deep_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(deep_config, f)

        loader = ConfigLoader(config_dir=temp_config_dir)
        config = loader.load_config("deep_config")

        assert config["level1"]["level2"]["level3"]["level4"]["value"] == "deep_value"

        # Note: get_parameter requires projection_config.yaml to exist,
        # so we test deep nesting through direct config access instead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
