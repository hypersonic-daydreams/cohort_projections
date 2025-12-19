"""
Pipeline orchestration scripts for North Dakota Population Projection System.

This package contains three main pipeline scripts that automate the end-to-end
workflow from raw data to dissemination-ready outputs:

1. 01_process_demographic_data.py - Data processing pipeline
   - Processes fertility, survival, and migration rates
   - Validates outputs and generates processing reports

2. 02_run_projections.py - Projection execution pipeline
   - Runs cohort-component projections for all geographies
   - Supports multiple scenarios and parallel processing

3. 03_export_results.py - Export and dissemination pipeline
   - Converts outputs to various formats
   - Packages results for distribution

Each script can be run independently or as part of a complete pipeline.
"""

__version__ = "0.1.0"
