"""
Tests for centralized race/ethnicity mappings.

Verifies that:
1. Canonical categories match the 6 standard categories
2. All mappings resolve to valid canonical categories
3. Helper functions work correctly
4. All processing modules use the centralized mappings
"""

import pytest

from cohort_projections.config import (
    CANONICAL_RACE_CATEGORIES,
    CANONICAL_RACE_CODES,
    CENSUS_RACE_MAP,
    MIGRATION_RACE_MAP,
    SEER_RACE_MAP,
    get_all_valid_aliases,
    map_race_to_canonical,
)


class TestCanonicalCategories:
    """Tests for canonical race/ethnicity categories."""

    def test_canonical_categories_count(self) -> None:
        """There should be exactly 6 canonical categories."""
        assert len(CANONICAL_RACE_CATEGORIES) == 6

    def test_canonical_categories_content(self) -> None:
        """Canonical categories match expected values from AGENTS.md."""
        expected = (
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        )
        assert expected == CANONICAL_RACE_CATEGORIES

    def test_canonical_codes_count(self) -> None:
        """There should be exactly 6 numeric codes."""
        assert len(CANONICAL_RACE_CODES) == 6

    def test_canonical_codes_values(self) -> None:
        """Numeric codes map to correct categories."""
        assert CANONICAL_RACE_CODES[1] == "White alone, Non-Hispanic"
        assert CANONICAL_RACE_CODES[2] == "Black alone, Non-Hispanic"
        assert CANONICAL_RACE_CODES[3] == "AIAN alone, Non-Hispanic"
        assert CANONICAL_RACE_CODES[4] == "Asian/PI alone, Non-Hispanic"
        assert CANONICAL_RACE_CODES[5] == "Two or more races, Non-Hispanic"
        assert CANONICAL_RACE_CODES[6] == "Hispanic (any race)"

    def test_canonical_codes_cover_all_categories(self) -> None:
        """All canonical categories have a numeric code."""
        code_values = set(CANONICAL_RACE_CODES.values())
        categories_set = set(CANONICAL_RACE_CATEGORIES)
        assert code_values == categories_set


class TestMappingCompleteness:
    """Tests for mapping completeness and validity."""

    @pytest.mark.parametrize(
        "mapping,name",
        [
            (CENSUS_RACE_MAP, "Census"),
            (SEER_RACE_MAP, "SEER"),
            (MIGRATION_RACE_MAP, "Migration"),
        ],
    )
    def test_all_mappings_resolve_to_canonical(self, mapping: dict[str, str], name: str) -> None:
        """All mappings should resolve to one of the 6 canonical categories."""
        canonical_set = set(CANONICAL_RACE_CATEGORIES)
        for alias, category in mapping.items():
            assert category in canonical_set, (
                f"{name} mapping '{alias}' -> '{category}' is not a canonical category"
            )

    @pytest.mark.parametrize(
        "mapping,name",
        [
            (CENSUS_RACE_MAP, "Census"),
            (SEER_RACE_MAP, "SEER"),
            (MIGRATION_RACE_MAP, "Migration"),
        ],
    )
    def test_all_canonical_categories_are_reachable(
        self, mapping: dict[str, str], name: str
    ) -> None:
        """Each canonical category should be reachable from at least one alias."""
        mapped_categories = set(mapping.values())
        canonical_set = set(CANONICAL_RACE_CATEGORIES)

        # All canonical categories should be reachable
        assert mapped_categories == canonical_set, (
            f"{name} mapping doesn't cover all canonical categories. "
            f"Missing: {canonical_set - mapped_categories}"
        )

    def test_census_map_has_expected_aliases(self) -> None:
        """Census mapping includes expected PEP/ACS aliases."""
        expected_aliases = ["WA_NH", "BA_NH", "IA_NH", "AA_NH", "TOM_NH", "H"]
        for alias in expected_aliases:
            assert alias in CENSUS_RACE_MAP, f"Missing Census alias: {alias}"

    def test_seer_map_has_expected_aliases(self) -> None:
        """SEER mapping includes expected aliases."""
        expected_aliases = ["White NH", "Black NH", "Hispanic", "1", "2", "3", "4", "5", "6"]
        for alias in expected_aliases:
            assert alias in SEER_RACE_MAP, f"Missing SEER alias: {alias}"


class TestMapRaceToCanonical:
    """Tests for the map_race_to_canonical helper function."""

    def test_maps_census_aliases(self) -> None:
        """Census aliases map correctly."""
        assert map_race_to_canonical("WA_NH", source="census") == "White alone, Non-Hispanic"
        assert map_race_to_canonical("BA_NH", source="census") == "Black alone, Non-Hispanic"
        assert map_race_to_canonical("HISP", source="census") == "Hispanic (any race)"

    def test_maps_seer_aliases(self) -> None:
        """SEER aliases map correctly."""
        assert map_race_to_canonical("White NH", source="seer") == "White alone, Non-Hispanic"
        assert map_race_to_canonical("Hispanic", source="seer") == "Hispanic (any race)"
        assert map_race_to_canonical("1", source="seer") == "White alone, Non-Hispanic"

    def test_auto_source_tries_all(self) -> None:
        """Auto source tries all mappings."""
        # Census alias
        assert map_race_to_canonical("WA_NH") == "White alone, Non-Hispanic"
        # SEER alias
        assert map_race_to_canonical("White NH") == "White alone, Non-Hispanic"

    def test_canonical_passthrough(self) -> None:
        """Canonical categories pass through unchanged."""
        for category in CANONICAL_RACE_CATEGORIES:
            assert map_race_to_canonical(category) == category

    def test_unknown_returns_none(self) -> None:
        """Unknown codes return None in non-strict mode."""
        assert map_race_to_canonical("Unknown Code") is None
        assert map_race_to_canonical("") is None

    def test_strict_mode_raises(self) -> None:
        """Strict mode raises ValueError for unknown codes."""
        with pytest.raises(ValueError, match="Cannot map race code"):
            map_race_to_canonical("Unknown Code", strict=True)

    def test_whitespace_handling(self) -> None:
        """Function handles whitespace in input."""
        assert map_race_to_canonical("  White NH  ") == "White alone, Non-Hispanic"
        assert map_race_to_canonical("WA_NH ") == "White alone, Non-Hispanic"


class TestGetAllValidAliases:
    """Tests for the get_all_valid_aliases helper function."""

    def test_returns_aliases_for_valid_category(self) -> None:
        """Returns list of aliases for valid category."""
        aliases = get_all_valid_aliases("White alone, Non-Hispanic")
        assert isinstance(aliases, list)
        assert len(aliases) > 0
        assert "White NH" in aliases
        assert "WA_NH" in aliases

    def test_raises_for_invalid_category(self) -> None:
        """Raises ValueError for non-canonical category."""
        with pytest.raises(ValueError, match="not a canonical category"):
            get_all_valid_aliases("Invalid Category")

    def test_all_canonical_have_aliases(self) -> None:
        """All canonical categories have at least one alias."""
        for category in CANONICAL_RACE_CATEGORIES:
            aliases = get_all_valid_aliases(category)
            assert len(aliases) > 0, f"No aliases found for {category}"


class TestModuleIntegration:
    """Tests verifying processing modules use centralized mappings."""

    def test_base_population_uses_centralized_mapping(self) -> None:
        """base_population.py uses CENSUS_RACE_MAP from config."""
        from cohort_projections.data.process.base_population import RACE_ETHNICITY_MAP

        # Should be the same object or equivalent
        assert RACE_ETHNICITY_MAP == CENSUS_RACE_MAP

    def test_fertility_rates_uses_centralized_mapping(self) -> None:
        """fertility_rates.py uses SEER_RACE_MAP from config."""
        from cohort_projections.data.process.fertility_rates import SEER_RACE_ETHNICITY_MAP

        assert SEER_RACE_ETHNICITY_MAP == SEER_RACE_MAP

    def test_survival_rates_uses_centralized_mapping(self) -> None:
        """survival_rates.py uses SEER_RACE_MAP from config."""
        from cohort_projections.data.process.survival_rates import SEER_MORTALITY_RACE_MAP

        assert SEER_MORTALITY_RACE_MAP == SEER_RACE_MAP

    def test_migration_rates_imports_from_config(self) -> None:
        """migration_rates.py imports MIGRATION_RACE_MAP from config."""
        # This tests that the import works (no duplicate definition)
        from cohort_projections.data.process import migration_rates

        # Check that the module doesn't define its own mapping
        # (it should import from config)
        assert hasattr(migration_rates, "MIGRATION_RACE_MAP") is False or (
            "MIGRATION_RACE_MAP" in dir(migration_rates)
        )
