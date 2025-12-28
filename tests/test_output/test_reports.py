"""
Unit tests for reports module.

Tests the generate_summary_statistics, compare_scenarios, generate_html_report,
and generate_text_report functions for proper report generation.
"""

import pandas as pd
import pytest

# Try to import the reports module
try:
    from cohort_projections.output.reports import (
        _calculate_median_age,
        compare_scenarios,
        generate_html_report,
        generate_summary_statistics,
        generate_text_report,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestGenerateSummaryStatistics:
    """Test generate_summary_statistics function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data with multiple years."""
        data = []
        for year in [2025, 2030, 2035, 2040, 2045]:
            for age in range(91):
                for sex in ["Male", "Female"]:
                    for race in ["White", "Black", "Hispanic"]:
                        # Create realistic population distribution
                        base_pop = 100.0
                        # More young people initially, shifts to elderly over time
                        age_factor = max(0.1, 1.0 - age / 100)
                        time_factor = 1.0 + (year - 2025) * 0.005
                        pop = base_pop * age_factor * time_factor

                        data.append(
                            {
                                "year": year,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop,
                            }
                        )
        return pd.DataFrame(data)

    @pytest.fixture
    def simple_projection(self):
        """Create a simpler projection for basic tests."""
        data = []
        for year in [2025, 2026]:
            for age in [0, 10, 30, 65, 80]:
                for sex in ["Male", "Female"]:
                    data.append(
                        {
                            "year": year,
                            "age": age,
                            "sex": sex,
                            "race": "White",
                            "population": 100.0 + (year - 2025) * 10,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_summary_returns_dict(self, simple_projection):
        """Test that summary statistics returns a dictionary."""
        stats = generate_summary_statistics(simple_projection)

        assert isinstance(stats, dict)
        assert len(stats) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_summary_has_required_keys(self, simple_projection):
        """Test that summary has all required keys."""
        stats = generate_summary_statistics(simple_projection)

        assert "by_year" in stats
        assert "age_structure" in stats
        assert "demographic_indicators" in stats
        assert "growth_analysis" in stats
        assert "generated_at" in stats

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_by_year_statistics(self, simple_projection):
        """Test by_year statistics calculation."""
        stats = generate_summary_statistics(simple_projection)

        by_year = stats["by_year"]
        assert len(by_year) == 2  # 2025 and 2026

        for year_data in by_year:
            assert "year" in year_data
            assert "total_population" in year_data
            assert "male" in year_data
            assert "female" in year_data
            assert "sex_ratio" in year_data
            assert "dependency_ratio" in year_data
            assert "median_age" in year_data

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_demographic_indicators(self, sample_projection):
        """Test demographic indicators calculation."""
        stats = generate_summary_statistics(sample_projection)

        indicators = stats["demographic_indicators"]
        assert "base_year" in indicators
        assert "final_year" in indicators
        assert "base_population" in indicators
        assert "final_population" in indicators
        assert "absolute_growth" in indicators
        assert "percent_growth" in indicators
        assert "annual_growth_rate" in indicators
        assert "dependency_ratio_base" in indicators
        assert "dependency_ratio_final" in indicators

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_age_structure(self, sample_projection):
        """Test age structure calculation."""
        stats = generate_summary_statistics(sample_projection)

        age_structure = stats["age_structure"]
        # Should have data for first, middle, and last years
        assert len(age_structure) >= 2

        # Check age groups exist
        first_year_key = list(age_structure.keys())[0]
        first_year = age_structure[first_year_key]
        assert "0-4" in first_year
        assert "65-74" in first_year
        assert "85+" in first_year

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_diversity_metrics_included(self, sample_projection):
        """Test diversity metrics are included by default."""
        stats = generate_summary_statistics(sample_projection)

        assert "diversity" in stats
        diversity = stats["diversity"]
        assert len(diversity) >= 2  # Base and final years

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_diversity_metrics_excluded(self, sample_projection):
        """Test diversity metrics can be excluded."""
        stats = generate_summary_statistics(sample_projection, include_diversity_metrics=False)

        assert stats["diversity"] == {}

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_growth_analysis(self, sample_projection):
        """Test growth analysis calculation."""
        stats = generate_summary_statistics(sample_projection)

        growth = stats["growth_analysis"]
        assert "annual_growth_rates" in growth
        assert "period_growth_rates" in growth
        # Should have growth rates for each year after base
        assert len(growth["annual_growth_rates"]) >= 1

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_empty_dataframe_returns_empty_dict(self):
        """Test empty DataFrame returns empty dictionary."""
        empty_df = pd.DataFrame()

        stats = generate_summary_statistics(empty_df)

        assert stats == {}

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_missing_columns_raises_error(self):
        """Test missing required columns raises ValueError."""
        incomplete_df = pd.DataFrame(
            {
                "year": [2025],
                "age": [0],
                # Missing sex, race, population
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            generate_summary_statistics(incomplete_df)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_custom_base_year(self, sample_projection):
        """Test custom base year parameter."""
        stats = generate_summary_statistics(sample_projection, base_year=2030)

        indicators = stats["demographic_indicators"]
        assert indicators["base_year"] == 2030


class TestCompareScenarios:
    """Test compare_scenarios function."""

    @pytest.fixture
    def baseline_projection(self):
        """Create baseline projection."""
        data = []
        for year in [2025, 2030, 2035]:
            for age in [0, 10, 30, 65]:
                for sex in ["Male", "Female"]:
                    data.append(
                        {
                            "year": year,
                            "age": age,
                            "sex": sex,
                            "race": "White",
                            "population": 1000.0,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def high_growth_projection(self):
        """Create high growth projection."""
        data = []
        for year in [2025, 2030, 2035]:
            for age in [0, 10, 30, 65]:
                for sex in ["Male", "Female"]:
                    # 10% higher than baseline each period
                    growth_factor = 1.0 + (year - 2025) * 0.02
                    data.append(
                        {
                            "year": year,
                            "age": age,
                            "sex": sex,
                            "race": "White",
                            "population": 1000.0 * growth_factor,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_compare_returns_dataframe(self, baseline_projection, high_growth_projection):
        """Test comparison returns DataFrame."""
        comparison = compare_scenarios(baseline_projection, high_growth_projection)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_comparison_has_required_columns(self, baseline_projection, high_growth_projection):
        """Test comparison has required columns."""
        comparison = compare_scenarios(baseline_projection, high_growth_projection)

        assert "year" in comparison.columns
        assert "difference" in comparison.columns
        assert "percent_difference" in comparison.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_custom_scenario_names(self, baseline_projection, high_growth_projection):
        """Test custom scenario names in output."""
        comparison = compare_scenarios(
            baseline_projection,
            high_growth_projection,
            baseline_name="Low Growth",
            scenario_name="High Growth",
        )

        assert "Low Growth_total" in comparison.columns
        assert "High Growth_total" in comparison.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_filter_specific_years(self, baseline_projection, high_growth_projection):
        """Test filtering to specific years."""
        comparison = compare_scenarios(
            baseline_projection,
            high_growth_projection,
            years_to_compare=[2025, 2035],
        )

        assert len(comparison) == 2
        assert set(comparison["year"]) == {2025, 2035}

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_comparison_includes_age_groups(self, baseline_projection, high_growth_projection):
        """Test comparison includes age group breakdowns."""
        comparison = compare_scenarios(baseline_projection, high_growth_projection)

        # Should have youth, working, and elderly columns
        cols = comparison.columns
        youth_cols = [c for c in cols if "youth" in c.lower()]
        working_cols = [c for c in cols if "working" in c.lower()]
        elderly_cols = [c for c in cols if "elderly" in c.lower()]

        assert len(youth_cols) >= 2  # Both scenarios
        assert len(working_cols) >= 2
        assert len(elderly_cols) >= 2


class TestGenerateHTMLReport:
    """Test generate_html_report function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data."""
        data = []
        for year in [2025, 2030, 2035]:
            for age in range(0, 91, 5):
                for sex in ["Male", "Female"]:
                    data.append(
                        {
                            "year": year,
                            "age": age,
                            "sex": sex,
                            "race": "White",
                            "population": 500.0 + (year - 2025) * 5,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_created(self, sample_projection, tmp_path):
        """Test HTML report is created."""
        output_file = tmp_path / "test_report.html"

        result_path = generate_html_report(sample_projection, output_file)

        assert output_file.exists()
        assert result_path == output_file

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_has_content(self, sample_projection, tmp_path):
        """Test HTML report has expected content."""
        output_file = tmp_path / "test_report.html"

        generate_html_report(sample_projection, output_file)

        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_custom_title(self, sample_projection, tmp_path):
        """Test HTML report with custom title."""
        output_file = tmp_path / "test_report.html"
        custom_title = "Custom Population Report"

        generate_html_report(sample_projection, output_file, title=custom_title)

        content = output_file.read_text()
        assert custom_title in content

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_with_pre_computed_stats(self, sample_projection, tmp_path):
        """Test HTML report with pre-computed statistics."""
        output_file = tmp_path / "test_report.html"
        stats = generate_summary_statistics(sample_projection)

        generate_html_report(sample_projection, output_file, summary_stats=stats)

        assert output_file.exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_with_metadata(self, sample_projection, tmp_path):
        """Test HTML report with metadata."""
        output_file = tmp_path / "test_report.html"
        metadata = {
            "projection_type": "baseline",
            "author": "Test Author",
        }

        generate_html_report(sample_projection, output_file, metadata=metadata)

        assert output_file.exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_without_methodology(self, sample_projection, tmp_path):
        """Test HTML report without methodology section."""
        output_file = tmp_path / "test_report.html"

        generate_html_report(sample_projection, output_file, include_methodology=False)

        content = output_file.read_text()
        # Should still have basic content
        assert "<!DOCTYPE html>" in content

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_html_report_creates_parent_directory(self, sample_projection, tmp_path):
        """Test HTML report creates parent directories."""
        output_file = tmp_path / "nested" / "dir" / "report.html"

        generate_html_report(sample_projection, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestGenerateTextReport:
    """Test generate_text_report function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection data."""
        data = []
        for year in [2025, 2030, 2035]:
            for age in range(0, 91, 5):
                for sex in ["Male", "Female"]:
                    data.append(
                        {
                            "year": year,
                            "age": age,
                            "sex": sex,
                            "race": "White",
                            "population": 500.0 + (year - 2025) * 5,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_text_report_created(self, sample_projection, tmp_path):
        """Test text report is created."""
        output_file = tmp_path / "test_report.txt"

        result_path = generate_text_report(sample_projection, output_file)

        assert output_file.exists()
        assert result_path == output_file

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_text_report_format(self, sample_projection, tmp_path):
        """Test text format output."""
        output_file = tmp_path / "test_report.txt"

        generate_text_report(sample_projection, output_file, format_type="text")

        content = output_file.read_text()
        # Text format should have section separators
        assert "=" in content or "-" in content

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_markdown_report_format(self, sample_projection, tmp_path):
        """Test markdown format output."""
        output_file = tmp_path / "test_report.md"

        generate_text_report(sample_projection, output_file, format_type="markdown")

        content = output_file.read_text()
        # Markdown format should have headers
        assert content.startswith("#")
        assert "##" in content

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_text_report_custom_title(self, sample_projection, tmp_path):
        """Test text report with custom title."""
        output_file = tmp_path / "test_report.txt"
        custom_title = "Custom Report Title"

        generate_text_report(sample_projection, output_file, title=custom_title)

        content = output_file.read_text()
        assert custom_title in content

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_text_report_with_tables(self, sample_projection, tmp_path):
        """Test text report includes tables."""
        output_file = tmp_path / "test_report.txt"

        generate_text_report(sample_projection, output_file, include_tables=True)

        content = output_file.read_text()
        # Should contain population numbers
        assert "Population" in content or "population" in content.lower()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_text_report_without_tables(self, sample_projection, tmp_path):
        """Test text report without tables."""
        output_file = tmp_path / "test_report.txt"

        generate_text_report(sample_projection, output_file, include_tables=False)

        content = output_file.read_text()
        # Should still have content
        assert len(content) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_text_report_with_pre_computed_stats(self, sample_projection, tmp_path):
        """Test text report with pre-computed statistics."""
        output_file = tmp_path / "test_report.txt"
        stats = generate_summary_statistics(sample_projection)

        generate_text_report(sample_projection, output_file, summary_stats=stats)

        assert output_file.exists()


class TestCalculateMedianAge:
    """Test _calculate_median_age helper function."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_median_age_basic(self):
        """Test basic median age calculation."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2, 3, 4],
                "population": [100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )

        median = _calculate_median_age(df)

        assert median == 2.0  # Median of 0-4 with equal weights

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_median_age_weighted(self):
        """Test weighted median age calculation."""
        df = pd.DataFrame(
            {
                "age": [20, 40, 60],
                "population": [0.0, 1000.0, 0.0],  # All at age 40
            }
        )

        median = _calculate_median_age(df)

        assert median == 40.0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_median_age_empty_dataframe(self):
        """Test median age with empty DataFrame."""
        df = pd.DataFrame()

        median = _calculate_median_age(df)

        assert median == 0.0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_median_age_zero_population(self):
        """Test median age with zero population."""
        df = pd.DataFrame(
            {
                "age": [0, 10, 20],
                "population": [0.0, 0.0, 0.0],
            }
        )

        median = _calculate_median_age(df)

        assert median == 0.0


class TestReportEdgeCases:
    """Test edge cases for report generation."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_single_year_projection(self, tmp_path):
        """Test report with single year projection."""
        df = pd.DataFrame(
            {
                "year": [2025] * 4,
                "age": [0, 10, 20, 65],
                "sex": ["Male", "Male", "Female", "Female"],
                "race": ["White", "White", "White", "White"],
                "population": [100.0, 100.0, 100.0, 100.0],
            }
        )

        output_file = tmp_path / "single_year.html"
        generate_html_report(df, output_file)

        assert output_file.exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_single_race_projection(self, tmp_path):
        """Test report with single race."""
        df = pd.DataFrame(
            {
                "year": [2025, 2030],
                "age": [0, 0],
                "sex": ["Male", "Female"],
                "race": ["White", "White"],
                "population": [100.0, 100.0],
            }
        )

        stats = generate_summary_statistics(df)
        assert "diversity" in stats

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_large_population_values(self, tmp_path):
        """Test report with large population values."""
        df = pd.DataFrame(
            {
                "year": [2025, 2030],
                "age": [0, 0],
                "sex": ["Male", "Female"],
                "race": ["White", "White"],
                "population": [10_000_000.0, 10_500_000.0],
            }
        )

        output_file = tmp_path / "large_pop.txt"
        generate_text_report(df, output_file)

        content = output_file.read_text()
        # Should format large numbers properly
        assert "10" in content  # At minimum should have the number


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
