#!/usr/bin/env python3
"""
Check for orphaned tests and untested production code.

This script identifies:
1. Test files that don't have corresponding production modules
2. Production modules that don't have corresponding test files
3. Test functions that import non-existent functions

Run: python scripts/check_test_coverage.py
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROD_DIR = PROJECT_ROOT / "cohort_projections"
TEST_DIR = PROJECT_ROOT / "tests"


def get_production_modules() -> dict[str, Path]:
    """Get all production Python modules."""
    modules = {}
    for py_file in PROD_DIR.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        # Create module path like "core.fertility"
        rel_path = py_file.relative_to(PROD_DIR)
        module_name = str(rel_path.with_suffix("")).replace("/", ".")
        modules[module_name] = py_file
    return modules


def get_test_modules() -> dict[str, set[Path]]:
    """Get all test Python modules and what they actually import."""
    modules: dict[str, set[Path]] = {}
    for py_file in TEST_DIR.rglob("test_*.py"):
        # Instead of guessing from path, look at actual imports
        imports = extract_imports_from_test(py_file)
        for imp in imports:
            # Convert cohort_projections.core.fertility -> core.fertility
            if imp.startswith("cohort_projections."):
                module = imp[len("cohort_projections.") :]
                if module not in modules:
                    modules[module] = set()
                modules[module].add(py_file)
    return modules


def extract_imports_from_test(test_file: Path) -> list[str]:
    """Extract cohort_projections imports from a test file."""
    imports = []
    try:
        tree = ast.parse(test_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("cohort_projections"):
                    imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("cohort_projections"):
                        imports.append(alias.name)
    except SyntaxError:
        pass
    return imports


def check_import_exists(import_path: str) -> bool:
    """Check if an import path corresponds to an existing module."""
    # Convert cohort_projections.core.fertility to a file path
    parts = import_path.split(".")
    if parts[0] == "cohort_projections":
        parts = parts[1:]  # Remove "cohort_projections" prefix

    # Check if it's a module file
    module_path = (
        PROD_DIR / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        if len(parts) > 1
        else PROD_DIR / f"{parts[0]}.py"
    )
    if module_path.exists():
        return True

    # Check if it's a package (directory with __init__.py)
    package_path = PROD_DIR / "/".join(parts)
    if package_path.is_dir() and (package_path / "__init__.py").exists():
        return True

    # Check parent package
    parent_path = PROD_DIR / "/".join(parts[:-1]) / f"{parts[-2]}.py" if len(parts) > 1 else None
    return bool(parent_path and parent_path.exists())


def main():
    print("=" * 60)
    print("Test Coverage Check")
    print("=" * 60)

    prod_modules = get_production_modules()
    test_modules = get_test_modules()

    # Find production modules without tests
    print("\n## Production modules WITHOUT test files:")
    untested = []
    for module in sorted(prod_modules.keys()):
        # Check if any test imports this module
        has_test = module in test_modules
        if not has_test:
            untested.append(module)
            print(f"  - {module}")
    if not untested:
        print("  (none - all modules have tests)")

    # Find test imports that don't correspond to production modules
    print("\n## Test imports that don't match production modules:")
    orphaned_imports = []
    for module, test_paths in sorted(test_modules.items()):
        if module not in prod_modules:
            # Check if it's a submodule import (e.g., importing a function from a module)
            # by checking if the parent module exists
            parent = ".".join(module.split(".")[:-1])
            if parent and parent in prod_modules:
                continue
            for test_path in test_paths:
                orphaned_imports.append((module, test_path))
                print(f"  - {test_path.relative_to(PROJECT_ROOT)} imports: {module}")
    if not orphaned_imports:
        print("  (none - all test imports resolve to production modules)")

    # Check for broken imports in test files
    print("\n## Test files with potentially broken imports:")
    broken_imports = []
    for test_file in TEST_DIR.rglob("test_*.py"):
        imports = extract_imports_from_test(test_file)
        for imp in imports:
            if not check_import_exists(imp):
                broken_imports.append((test_file, imp))
                print(f"  - {test_file.relative_to(PROJECT_ROOT)}: {imp}")
    if not broken_imports:
        print("  (none - all imports resolve)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Production modules: {len(prod_modules)}")
    print(f"  Tested modules: {len(test_modules)}")
    print(f"  Untested modules: {len(untested)}")
    print(f"  Orphaned imports: {len(orphaned_imports)}")
    print(f"  Broken imports: {len(broken_imports)}")

    if orphaned_imports or broken_imports:
        print("\n⚠️  Issues found - consider cleaning up orphaned tests")
        return 1
    else:
        print("\n✓ All tests are properly mapped to production code")
        return 0


if __name__ == "__main__":
    sys.exit(main())
