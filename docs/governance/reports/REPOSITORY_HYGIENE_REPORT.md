# Repository Hygiene and Quality Assurance Report

**Strategies for Rigor, Reproducibility, and Maintainability**

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-01-01 |
| **Project** | North Dakota Population Projection System |
| **Status** | Final |
| **Audience** | Technical Leadership, Project Stakeholders |

---

## Executive Summary

As the North Dakota Population Projection System has evolved from a targeted production tool into a comprehensive demographic research platform, maintaining technical rigor has become paramount. We have implemented a multi-layered "Defense in Depth" approach to repository hygiene and quality assurance. This strategy goes beyond standard code linting to include a **PostgreSQL-backed "Repository Intelligence" system**, formally automated governance structures, and strict methodological reproducibility protocols.

This report details these systems, demonstrating how they mitigate risk, ensure valid demographic outputs, and provide an audit trail for every result generated. The methods outlined here serve as a proven framework that can be replicated in future high-stakes data projects.

---

## 1. Repository Intelligence System

At the core of our hygiene strategy is a "Repository Intelligence" system that treats the codebase itself as data. This allows us to track the state of our software with the same rigor we track demographic data.

### 1.1 PostgreSQL-Backed State Tracking (`cohort_projections_meta`)
We maintain a dedicated PostgreSQL database (`cohort_projections_meta`) to track:
*   **Active Code Inventory**: Every script, module, and configuration file is registered in a `code_inventory` table.
*   **Execution History**: Key analytical runs are logged to `run_history`, capturing input data versions, parameters, and Git commit hashes.
*   **Documentation Links**: A `documentation_links` table maps documentation to the code it describes.

### 1.2 Automated Inventory Updates
To prevent the database from becoming stale, we implemented a custom pre-commit hook (`scripts/hooks/update_code_inventory.py`).
*   **Mechanism**: When a developer commits changes, the hook automatically detects added, modified, or deleted files.
*   **Action**: It "upserts" (updates or inserts) these records in the database immediately.
*   **Result**: The database always reflects the exact state of the repository at every commit, requiring zero manual effort from developers.

### 1.3 Methodological Reproducibility & Type II SCD
We employ a **Slowly Changing Dimension (Type II)** approach to track the evolution of our codebase and methodology.
*   **Audit Trail**: Rather than deleting records when files are removed, we mark them as `deprecated` (soft delete) in the `code_inventory`.
*   **Time-Travel**: This allows us to reconstruct the exact state of the "active" codebase at any point in history.
*   **Reproducibility**: By linking `run_history` to specific versions of code and data, we can guarantee that any historical projection can be exactly reproduced, satisfying strict academic and state agency reporting requirements.

---

## 2. Governance & Knowledge Management

We have formalized our decision-making and operational processes to ensure that "shortcuts" do not compromise long-term quality.

### 2.1 Architecture Decision Records (ADRs)
We use ADRs to capture the "why" behind every significant technical decision.
*   **Format**: Standardized template including Context, Options Considered, Decision, and Consequences.
*   **Hygiene**: ADRs are treated as immutable records of agreement. When a decision changes, a new ADR supersedes the old one, preserving the history of our architectural thinking.
*   **Examples**:
    *   `ADR-005`: Configuration Management Strategy.
    *   `ADR-011`: Testing Strategy.
    *   `ADR-022`: Unified Documentation Strategy.

### 2.2 Standard Operating Procedures (SOPs)
Complex, high-risk workflows are codified into SOPs to ensure consistency.
*   **Example**: `SOP-001 External AI Analysis Integration` defines a strict protocol for incorporating AI-generated code or insights. It mandates a phased approach: Intake -> Exploratory Analysis (Phase A) -> Implementation Planning -> Production Implementation (Phase B) -> Testing.
*   **Benefit**: This prevents "spaghetti code" from ad-hoc AI suggestions and ensures all external code meets our internal quality standards.

### 2.3 Templates
We force consistency through the use of templates for all major artifacts:
*   **ADR Template**: Ensures all decisions consider risks and alternatives.
*   **Report Template**: Standardizes findings from sub-agents/analysts.
*   **SOP Template**: Ensures all procedures define prerequisites and validation steps.

---

## 3. Automated Quality Assurance

We rely on automation to enforce standards, freeing developers to focus on logic rather than formatting.

### 3.1 Pre-commit Hooks Pipeline
Our `.pre-commit-config.yaml` defines a rigorous gauntlet that every commit must pass:
*   **Syntactic Hygiene**: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`.
*   **Linting & Formatting**: `ruff` enforces PEP 8 standards and catches common bugs (e.g., undefined variables).
*   **Type Safety**: `mypy` performs static type checking to prevent type-related runtime errors.
*   **Custom Checks**:
    *   `pytest-check`: Runs fast unit tests to prevent committing broken code.
    *   `data-manifest-check`: Enforces rules about data file locations (e.g., no raw data in git).
    *   `update-code-inventory`: Syncs the Postgres database (as described above).

### 3.2 Testing Strategy (The "Pragmatic Pyramid")
Defined in `ADR-011`, our testing strategy balances rigor with the reality of a research codebase:
*   **Built-in Validation**: Functions include runtime checks for demographic plausibility (e.g., "Population cannot be negative", "Fertility rates must be within biological limits").
*   **Integration Examples**: Scripts in `examples/` serve as both documentation and end-to-end integration tests.
*   **Selective Unit Tests**: Critical mathematical functions (mortality handling, cohort aging) are unit tested.

---

## 4. Configuration & Data Hygiene

### 4.1 Centralized Configuration (`ADR-005`)
Hard-coded values are strictly forbidden. All parameters (demographic assumptions, file paths, scenario toggles) are centralized in `config/projection_config.yaml`.
*   **Type Safety**: A `ConfigLoader` utility ensures typed access to these values.
*   **Clarity**: Non-technical stakeholders can review assumptions by reading a single YAML file.

### 4.2 Data Immutability
*   **Raw Data**: `data/raw/` is read-only. We never modify source files in place.
*   **Processing**: All transformations output to `data/processed/`, ensuring we can always re-run the pipeline from the original source.
*   **Versioning**: Data syncs are managed via `rclone bisync`, not Git, keeping the repository light while ensuring data availability.

### 4.3 Modern Package Management (`uv`)
We utilize `uv` for lightning-fast, reproducible dependency management. This ensures that the exact environment used to generate a result can be recreated on any machine in seconds.

---

## Conclusion

The North Dakota Population Projection System demonstrates that **repository hygiene is not an afterthought, but a feature**. By layering automated checks (pre-commit, linters) with structural governance (ADRs, SOPs) and a novel "Repository Intelligence" system, we have created a codebase that is essentially self-documenting and self-monitoring. This approach provides the high level of trust required for state government reporting while maintaining the agility needed for academic research.
