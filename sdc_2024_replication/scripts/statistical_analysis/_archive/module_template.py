#!/usr/bin/env python3
"""
Statistical Analysis Module Template
====================================

This template provides the standard structure for all statistical analysis modules.
Copy this file and modify for specific module implementations.

Usage:
    micromamba run -n cohort_proj python module_X_Y_name.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.warnings: list[str] = []
        self.next_steps: list[str] = []

    def to_dict(self) -> dict:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
        return output_path


def load_data(filename: str) -> pd.DataFrame:
    """Load data file from analysis directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    elif filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def load_previous_results(module_id: str) -> dict:
    """Load results from a previous module (for dependencies)."""
    # Find all result files for the module
    pattern = f"module_{module_id.replace('.', '_')}*.json"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results found for module {module_id}")

    results = {}
    for f in files:
        with open(f) as fp:
            results[f.stem] = json.load(fp)
    return results


def validate_data(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Validate dataframe has required columns. Return list of warnings."""
    warnings = []
    missing = set(required_columns) - set(df.columns)
    if missing:
        warnings.append(f"Missing columns: {missing}")

    # Check for missing values
    null_counts = df[list(set(required_columns) & set(df.columns))].isnull().sum()
    if null_counts.any():
        warnings.append(f"Missing values: {null_counts[null_counts > 0].to_dict()}")

    return warnings


# =============================================================================
# MODULE IMPLEMENTATION BELOW
# =============================================================================


def run_analysis() -> ModuleResult:
    """
    Main analysis function - implement module-specific logic here.

    Returns:
        ModuleResult object with all findings
    """
    # Initialize result container
    result = ModuleResult(
        module_id="X.Y",  # e.g., "1.1", "2.1.2"
        analysis_name="template_analysis",
    )

    # Load required data
    # df = load_data("nd_migration_summary.csv")
    # result.input_files.append("nd_migration_summary.csv")

    # Record parameters
    # result.parameters = {"param1": value1, ...}

    # Perform analysis
    # ...

    # Store results
    # result.results = {...}

    # Store diagnostics
    # result.diagnostics = {...}

    # Add any warnings
    # result.warnings.append("...")

    # Suggest next steps
    # result.next_steps.append("...")

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Statistical Analysis Module Template")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    try:
        result = run_analysis()
        output_file = result.save("module_X_Y_template.json")
        print("\nAnalysis completed successfully!")
        print(f"Output: {output_file}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
