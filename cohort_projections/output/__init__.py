"""
Enhanced output module for projection results.

This module provides comprehensive output capabilities for population projection data,
including formatted exports, reports, and visualizations.

Main Components:
    - writers: Enhanced export to Excel, CSV, Parquet, and other formats
    - reports: Summary statistics and formatted reports (HTML, text, markdown)
    - visualizations: Charts and graphs (pyramids, trends, growth rates)

Quick Start:
    >>> from cohort_projections.output import (
    ...     write_projection_excel,
    ...     generate_summary_statistics,
    ...     plot_population_pyramid,
    ...     save_all_visualizations
    ... )
    >>>
    >>> # Export to Excel with formatting
    >>> write_projection_excel(projection_df, 'output/results.xlsx')
    >>>
    >>> # Generate comprehensive statistics
    >>> stats = generate_summary_statistics(projection_df)
    >>>
    >>> # Create visualizations
    >>> save_all_visualizations(projection_df, 'output/charts', 'projection')

Modules:
    writers: Export functions for various file formats
    reports: Report generation and summary statistics
    visualizations: Chart and graph creation
"""

from .reports import (
    compare_scenarios,
    generate_html_report,
    generate_summary_statistics,
    generate_text_report,
)
from .visualizations import (
    plot_component_analysis,
    plot_growth_rates,
    plot_population_pyramid,
    plot_population_trends,
    plot_scenario_comparison,
    save_all_visualizations,
)
from .writers import (
    write_projection_csv,
    write_projection_excel,
    write_projection_formats,
    write_projection_shapefile,
)

__all__ = [
    # Writers
    "write_projection_excel",
    "write_projection_csv",
    "write_projection_formats",
    "write_projection_shapefile",
    # Reports
    "generate_summary_statistics",
    "compare_scenarios",
    "generate_html_report",
    "generate_text_report",
    # Visualizations
    "plot_population_pyramid",
    "plot_population_trends",
    "plot_growth_rates",
    "plot_component_analysis",
    "plot_scenario_comparison",
    "save_all_visualizations",
]

__version__ = "1.0.0"
