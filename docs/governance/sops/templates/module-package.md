# Module Package Structure Template

Use this template when creating new Python module packages for Phase B implementation.

---

## Directory Structure

```
sdc_2024_replication/scripts/statistical_analysis/module_BN_name/
├── __init__.py              # Public API exports
├── component1.py            # Core functionality
├── component2.py            # Supporting functionality
├── component3.py            # Additional functionality
└── conftest.py              # Optional: pytest collection exclusion
```

---

## __init__.py Template

```python
"""
[Module Name]
=============

[Brief description of what this module does]

Components:
- component1: [Description]
- component2: [Description]
- component3: [Description]

Key Question:
"[The research/analysis question this module addresses]"

Usage:
    from module_BN_name import (
        function_1,
        function_2,
        CONSTANT_1,
    )
"""

from .component1 import (
    function_1,
    function_2,
    HelperClass,
)
from .component2 import (
    function_3,
    function_4,
)
from .component3 import (
    CONSTANT_1,
    CONSTANT_2,
    function_5,
)

__all__ = [
    # Component 1
    "function_1",
    "function_2",
    "HelperClass",
    # Component 2
    "function_3",
    "function_4",
    # Component 3
    "CONSTANT_1",
    "CONSTANT_2",
    "function_5",
]
```

---

## Component File Template

```python
"""
[Component Name]
----------------

[Description of what this component does]

This module is part of the module_BN_name package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Type-only imports to avoid circular dependencies
    pass


# === Constants ===

CONSTANT_1 = [
    "Value 1",
    "Value 2",
]


# === Public Functions ===

def function_1(
    data: pd.DataFrame,
    param1: str,
    param2: Optional[float] = None,
) -> dict:
    """
    Brief description of function.

    Parameters
    ----------
    data : pd.DataFrame
        Description of data parameter
    param1 : str
        Description of param1
    param2 : float, optional
        Description of param2, by default None

    Returns
    -------
    dict
        Description of return value

    Examples
    --------
    >>> result = function_1(df, "value")
    >>> result["key"]
    expected_value
    """
    # Implementation
    result = {}
    return result


def function_2(
    input_value: float,
    *,
    strict: bool = False,
) -> float:
    """
    Brief description.

    Parameters
    ----------
    input_value : float
        Description
    strict : bool, optional
        Description, by default False

    Returns
    -------
    float
        Description
    """
    # Implementation
    return input_value


# === Private Helper Functions ===

def _helper_function(x: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    return x
```

---

## conftest.py Template (if needed)

Use this if your module has functions with names that might be collected by pytest:

```python
# This conftest.py prevents pytest from collecting this directory as tests.
# The run_* functions are for API use, not pytest tests.

collect_ignore_glob = ["*.py"]
```

---

## Naming Conventions

### Function Names
- **DO**: Use descriptive names like `calculate_shift`, `run_hypothesis_test`, `get_percentile`
- **DON'T**: Use `test_*` prefix (conflicts with pytest collection)
- **Migration**: If renaming from `test_*`, use `run_*_test` pattern

### Module Names
- Use `module_BN_descriptive_name` format
- Examples: `module_B1_regime_aware_models`, `module_B2_multistate_placebo`

### Package Directory Names
- Match the runner script name
- Examples: `module_B1_regime_aware/`, `module_B2_multistate_placebo/`

---

## Runner Script Template

Create alongside the package at `module_BN_name.py`:

```python
#!/usr/bin/env python3
"""
Module BN: [Title]
==================

[Description of what this module does]

This script runs the [analysis type] and outputs results to the results/ directory.

Usage:
    python module_BN_name.py [--option1] [--option2]

Outputs:
    results/module_BN_*.json - [Description]
    figures/module_BN_*.png  - [Description] (if applicable)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from module_BN_name import (
    function_1,
    function_2,
    CONSTANT_1,
)


# === Configuration ===

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"


def main(args: argparse.Namespace) -> dict:
    """Main analysis function."""
    print(f"Running Module BN analysis at {datetime.now()}")

    # Create output directories
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Run analysis
    results = {}

    # Example: Call module functions
    result_1 = function_1(data, "param")
    results["analysis_1"] = result_1

    # Save results
    output_file = RESULTS_DIR / "module_BN_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Module BN: [Title]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--option1",
        type=str,
        default="default",
        help="Description of option1",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
```

---

## Test File Template

Create in `tests/test_statistical/test_module_bn.py`:

```python
"""
Tests for module_BN_name package.
"""

import numpy as np
import pandas as pd
import pytest

# Import from the module being tested
from sdc_2024_replication.scripts.statistical_analysis.module_BN_name import (
    function_1,
    function_2,
    CONSTANT_1,
)


class TestFunction1:
    """Tests for function_1."""

    def test_basic_functionality(self, sample_data_fixture):
        """Test basic function behavior."""
        result = function_1(sample_data_fixture, "param")
        assert "expected_key" in result
        assert result["expected_key"] > 0

    def test_edge_case(self):
        """Test edge case handling."""
        # Test with edge case input
        pass

    @pytest.mark.parametrize("param,expected", [
        ("value1", 1.0),
        ("value2", 2.0),
    ])
    def test_parameterized(self, param, expected):
        """Test with multiple parameter values."""
        result = function_2(param)
        assert result == expected


class TestConstants:
    """Tests for module constants."""

    def test_constant_values(self):
        """Test that constants have expected values."""
        assert len(CONSTANT_1) > 0
        assert "Expected Value" in CONSTANT_1
```

---

## Integration Checklist

- [ ] Package directory created with all component files
- [ ] `__init__.py` exports all public API
- [ ] Runner script created and tested
- [ ] No `test_*` function names in source code
- [ ] Type hints on all public functions
- [ ] Docstrings with Parameters/Returns sections
- [ ] conftest.py added if needed for pytest exclusion
- [ ] Test file created in `tests/test_statistical/`
- [ ] Pre-commit hooks pass

---

*Template Version: 1.0*
