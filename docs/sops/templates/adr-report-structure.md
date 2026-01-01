# ADR Report Directory Structure Template

Use this template when creating the `docs/adr/0XX-reports/` directory for a new ADR investigation.

---

## Directory Structure

```
docs/adr/0XX-reports/
├── README.md                      # Overview and coordination
├── PLANNING_SYNTHESIS.md          # Phase B planning (created later)
├── phase_b_plans/                 # Individual agent plans
│   ├── AGENT_B0_PLAN.md
│   ├── AGENT_B1_PLAN.md
│   └── ...
├── agent1_[description].py        # Phase A exploratory script
├── agent2_[description].py        # Phase A exploratory script
├── agent3_[description].py        # Phase A exploratory script
├── figures/                       # Generated visualizations
│   └── *.png, *.pdf
├── data/                          # Intermediate data outputs
│   └── *.csv, *.json
└── results/                       # Final analysis results
    └── *.json
```

---

## README.md Template

```markdown
# ADR-0XX Reports: [Investigation Title]

## Overview

Brief description of the investigation and its purpose.

## Investigation Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase A | Complete/In Progress | Exploratory analysis |
| Phase B | Complete/In Progress | Implementation |

## Agent Coordination

| Agent | Responsibility | Status |
|-------|----------------|--------|
| Agent 1 | [Task description] | Complete |
| Agent 2 | [Task description] | Complete |
| Agent 3 | [Task description] | Complete |

## Key Findings

1. Finding 1
2. Finding 2
3. Finding 3

## Outputs

- `agent1_*.py` - [Description]
- `agent2_*.py` - [Description]
- `figures/` - Visualizations for [purpose]
- `results/` - JSON outputs for [purpose]

## Related Documents

- [ADR-0XX](../0XX-title.md) - Parent ADR
- [PLANNING_SYNTHESIS.md](./PLANNING_SYNTHESIS.md) - Implementation plan
```

---

## Exploratory Script Template

```python
#!/usr/bin/env python3
"""
Agent N: [Analysis Title]
ADR-0XX Investigation

[Brief description of what this script analyzes]

.. deprecated:: YYYY-MM-DD
    This is a **legacy Phase A research script** from the ADR-0XX investigation.
    It was used for one-time exploratory analysis and is retained for reproducibility
    and audit purposes only. This script is NOT production code and should NOT be
    modified or extended.

    The analysis outputs from this script have been incorporated into the final
    ADR-0XX decision. For current methodology, see:
    - [path to production module 1]
    - [path to production module 2]

Status: DEPRECATED / ARCHIVED
Linting: Exempted from strict linting (see pyproject.toml per-file-ignores)
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np


# === Configuration ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"


def main():
    """Main analysis function."""
    # Create output directories
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Your analysis here
    results = {}

    # Save results
    with open(RESULTS_DIR / "agent_N_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Analysis complete. Results saved to:", RESULTS_DIR)
    return results


if __name__ == "__main__":
    main()
```

---

## pyproject.toml Exemptions Template

Add to `[tool.ruff.lint.per-file-ignores]`:

```toml
# Legacy Phase A research scripts from ADR-0XX investigation (deprecated YYYY-MM-DD)
# These are one-time exploratory analysis scripts, not production code.
"docs/adr/0XX-reports/agent*_analysis.py" = [
    "N803",    # Argument name should be lowercase
    "N806",    # Variable in function should be lowercase
    "T201",    # Allow print statements
    "D",       # Ignore docstring requirements
    "F841",    # Unused variable (common in exploratory code)
]
```

---

*Template Version: 1.0*
