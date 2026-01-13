#!/usr/bin/env python3
"""
Concordance Parser - Programmatic Access to Statistical Methods Documentation
=============================================================================

This module provides utilities for AI agents to programmatically access
the statistical concordance data, including:

- Lookup equations by number or name
- Get Python implementations for any equation
- Find the Python module implementing a specific test
- Cross-reference tests and equations

Usage:
    from concordance_parser import Concordance

    concordance = Concordance()

    # Look up an equation
    eq = concordance.get_equation('eq_did')
    print(eq['latex'])
    print(eq['python_implementation'])

    # Look up a test
    test = concordance.get_test('adf_test')
    print(test['python_library'])

    # Find all tests in a category
    unit_root_tests = concordance.get_tests_by_category('unit_root')
"""

from pathlib import Path

import yaml


class Concordance:
    """Parser and accessor for the statistical concordance."""

    def __init__(self, concordance_path: str | Path | None = None):
        """
        Initialize the concordance parser.

        Args:
            concordance_path: Path to statistical_concordance.yaml.
                             If None, uses default location relative to this file.
        """
        if concordance_path is None:
            concordance_path = Path(__file__).parent / "statistical_concordance.yaml"
        else:
            concordance_path = Path(concordance_path)

        if not concordance_path.exists():
            raise FileNotFoundError(f"Concordance file not found: {concordance_path}")

        with open(concordance_path) as f:
            self._data = yaml.safe_load(f)

        self._equations = self._data.get("equations", {})
        self._tests = self._data.get("statistical_tests", {})
        self._scripts = self._data.get("python_scripts", {})
        self._metadata = self._data.get("metadata", {})

    @property
    def metadata(self) -> dict:
        """Return article/concordance metadata."""
        return self._metadata

    @property
    def total_equations(self) -> int:
        """Return total number of equations."""
        return len(self._equations)

    @property
    def total_tests(self) -> int:
        """Return total number of statistical tests."""
        return len(self._tests)

    def get_equation(self, key: str) -> dict | None:
        """
        Get equation by key (e.g., 'eq_did', 'eq_hp_filter').

        Args:
            key: Equation key from the concordance

        Returns:
            Dictionary with equation details or None if not found
        """
        return self._equations.get(key)

    def get_equation_by_number(self, number: int) -> dict | None:
        """
        Get equation by its number in the paper.

        Args:
            number: Equation number (1-17)

        Returns:
            Dictionary with equation details or None if not found
        """
        for key, eq in self._equations.items():
            if eq.get("number") == number:
                return {**eq, "key": key}
        return None

    def get_test(self, key: str) -> dict | None:
        """
        Get statistical test by key (e.g., 'adf_test', 'log_rank_test').

        Args:
            key: Test key from the concordance

        Returns:
            Dictionary with test details or None if not found
        """
        return self._tests.get(key)

    def get_tests_by_category(self, category: str) -> list[dict]:
        """
        Get all tests in a specific category.

        Args:
            category: Category string (e.g., 'unit_root', 'survival_comparison')

        Returns:
            List of test dictionaries matching the category
        """
        results = []
        for key, test in self._tests.items():
            if test.get("category") == category:
                results.append({**test, "key": key})
        return results

    def get_equations_by_category(self, category: str) -> list[dict]:
        """
        Get all equations in a specific category.

        Args:
            category: Category string (e.g., 'causal_inference', 'survival_analysis')

        Returns:
            List of equation dictionaries matching the category
        """
        results = []
        for key, eq in self._equations.items():
            if eq.get("category") == category:
                results.append({**eq, "key": key})
        return results

    def get_python_module(self, module_key: str) -> dict | None:
        """
        Get information about a Python module.

        Args:
            module_key: Module key (e.g., 'module_7_causal')

        Returns:
            Dictionary with module details
        """
        return self._scripts.get(module_key)

    def find_module_for_equation(self, equation_key: str) -> str | None:
        """
        Find which Python module implements a given equation.

        Args:
            equation_key: Equation key (e.g., 'eq_did')

        Returns:
            Module filename or None if not found
        """
        for module_key, module_info in self._scripts.items():
            if equation_key in module_info.get("implements", []):
                return module_info.get("filename")
        return None

    def find_module_for_test(self, test_key: str) -> str | None:
        """
        Find which Python module implements a given test.

        Args:
            test_key: Test key (e.g., 'adf_test')

        Returns:
            Module filename or None if not found
        """
        for module_key, module_info in self._scripts.items():
            if test_key in module_info.get("implements", []):
                return module_info.get("filename")
        return None

    def list_all_equations(self) -> list[dict]:
        """
        List all equations with their numbers and names.

        Returns:
            List of equation summaries sorted by number
        """
        summaries = []
        for key, eq in self._equations.items():
            summaries.append(
                {
                    "key": key,
                    "number": eq.get("number"),
                    "name": eq.get("name"),
                    "category": eq.get("category"),
                }
            )
        return sorted(summaries, key=lambda x: x["number"] or 999)

    def list_all_tests(self) -> list[dict]:
        """
        List all statistical tests with their names and categories.

        Returns:
            List of test summaries
        """
        summaries = []
        for key, test in self._tests.items():
            summaries.append(
                {
                    "key": key,
                    "full_name": test.get("full_name"),
                    "category": test.get("category"),
                    "python_library": test.get("python_library"),
                }
            )
        return summaries

    def get_latex(self, equation_key: str) -> str | None:
        """
        Get the LaTeX representation of an equation.

        Args:
            equation_key: Equation key

        Returns:
            LaTeX string or None
        """
        eq = self.get_equation(equation_key)
        return eq.get("latex") if eq else None

    def get_python_implementation(self, equation_key: str) -> str | None:
        """
        Get the Python implementation snippet for an equation.

        Args:
            equation_key: Equation key

        Returns:
            Python code string or None
        """
        eq = self.get_equation(equation_key)
        return eq.get("python_implementation") if eq else None

    def get_symbols(self, equation_key: str) -> dict | None:
        """
        Get symbol definitions for an equation.

        Args:
            equation_key: Equation key

        Returns:
            Dictionary mapping symbols to descriptions
        """
        eq = self.get_equation(equation_key)
        return eq.get("symbols") if eq else None

    def search_equations(self, query: str) -> list[dict]:
        """
        Search equations by name or category.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching equations
        """
        query_lower = query.lower()
        results = []
        for key, eq in self._equations.items():
            if (
                query_lower in key.lower()
                or query_lower in eq.get("name", "").lower()
                or query_lower in eq.get("category", "").lower()
            ):
                results.append({**eq, "key": key})
        return results

    def search_tests(self, query: str) -> list[dict]:
        """
        Search tests by name, category, or library.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching tests
        """
        query_lower = query.lower()
        results = []
        for key, test in self._tests.items():
            if (
                query_lower in key.lower()
                or query_lower in test.get("full_name", "").lower()
                or query_lower in test.get("category", "").lower()
                or query_lower in test.get("python_library", "").lower()
            ):
                results.append({**test, "key": key})
        return results


def main():
    """Demo usage of the concordance parser."""
    concordance = Concordance()

    print("=" * 70)
    print("STATISTICAL CONCORDANCE PARSER")
    print("=" * 70)

    print(f"\nTotal Equations: {concordance.total_equations}")
    print(f"Total Statistical Tests: {concordance.total_tests}")

    print("\n" + "-" * 70)
    print("ALL EQUATIONS:")
    print("-" * 70)
    for eq in concordance.list_all_equations():
        print(f"  Eq {eq['number']:2d}: {eq['name']} ({eq['category']})")

    print("\n" + "-" * 70)
    print("ALL STATISTICAL TESTS:")
    print("-" * 70)
    for test in concordance.list_all_tests():
        print(f"  {test['full_name']}")
        print(f"       Category: {test['category']}")
        print(f"       Library: {test['python_library']}")
        print()

    print("\n" + "-" * 70)
    print("EXAMPLE: DiD Equation Details")
    print("-" * 70)
    did = concordance.get_equation("eq_did")
    if did:
        print(f"Name: {did['name']}")
        print(f"Paper Section: {did['paper_section']}")
        print(f"\nLaTeX:\n{did['latex']}")
        print(f"\nSymbols:")
        for symbol, desc in did.get("symbols", {}).items():
            print(f"  {symbol}: {desc}")

    print("\n" + "-" * 70)
    print("EXAMPLE: Find Module for eq_km")
    print("-" * 70)
    module = concordance.find_module_for_equation("eq_km")
    print(f"Kaplan-Meier is implemented in: {module}")


if __name__ == "__main__":
    main()
