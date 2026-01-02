# ADR-023c: Codebase Catalog Package

## Purpose

This document specifies the extraction of repository scanning, code inventory, and pre-commit hook utilities into a standalone `codebase-catalog` Python package.

**Parent ADR**: [ADR-023: Package Extraction Strategy](023-package-extraction-strategy.md)

## Package Overview

| Attribute | Value |
|-----------|-------|
| **Package Name** | `codebase-catalog` |
| **Repository** | `~/workspace/libs/codebase_catalog` |
| **Python Module** | `codebase_catalog` |
| **Primary Purpose** | Codebase analysis and hygiene: code inventory, governance metadata, pre-commit validation, AI agent awareness |

## Scope

### Feature Modules

#### 1. Codebase Scanner (`codebase_catalog.scanner`)

**Source**: `scripts/intelligence/scan_repository.py` (287 lines)

**Capabilities**:
- Scan repository for Python files, scripts, and modules
- Extract function and class definitions with signatures
- Parse docstrings and comments
- Identify dependencies and imports
- Generate structured inventory of code artifacts
- Output to PostgreSQL database or JSON/JSONL files

#### 2. Code Inventory Generator (`codebase_catalog.inventory`)

**Source**: `scripts/intelligence/generate_inventory_docs.py` (91 lines)

**Capabilities**:
- Generate Markdown documentation from code inventory
- Create navigable index of all scripts and modules
- Document function signatures and purposes
- Track code changes over time
- Support multiple output formats (Markdown, HTML, JSON)

#### 3. Archive Manager (`codebase_catalog.archive`)

**Source**: `scripts/intelligence/archive_manager.py` (98 lines)

**Capabilities**:
- Manage archived/deprecated code tracking
- Identify stale or unused code
- Generate archive reports
- Support archival workflows

#### 4. Pre-commit Hooks (`codebase_catalog.hooks`)

**Source**: `scripts/hooks/update_code_inventory.py` (230 lines), `scripts/hooks/check_data_manifest.py` (280 lines)

**Capabilities**:
- Pre-commit hook for automatic code inventory updates
- Data manifest validation hook
- Governance metadata enforcement
- Git integration for change detection
- Exit codes for CI/CD integration

#### 5. Governance Metadata Parser (`codebase_catalog.governance`)

**Source**: Embedded in scanner and hooks

**Capabilities**:
- Parse governance metadata from markdown files (ADRs, SOPs)
- Extract structured information from documentation
- Track documentation coverage
- Link code to governance documents

## Package Structure

```
codebase_catalog/
├── pyproject.toml
├── uv.lock
├── .envrc
├── .gitignore
├── README.md
├── AGENTS.md                          # AI agent guidance
├── codebase_catalog/
│   ├── __init__.py
│   ├── scanner/
│   │   ├── __init__.py
│   │   ├── python_scanner.py          # Python file analysis
│   │   ├── ast_utils.py               # AST parsing utilities
│   │   └── dependency_analyzer.py     # Import/dependency tracking
│   ├── inventory/
│   │   ├── __init__.py
│   │   ├── generator.py               # Inventory generation
│   │   ├── markdown_writer.py         # Markdown output
│   │   └── db_writer.py               # PostgreSQL output
│   ├── archive/
│   │   ├── __init__.py
│   │   └── manager.py                 # Archive tracking
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── inventory_hook.py          # Code inventory hook
│   │   ├── manifest_hook.py           # Data manifest hook
│   │   └── base.py                    # Hook utilities
│   ├── governance/
│   │   ├── __init__.py
│   │   ├── parser.py                  # Governance metadata parser
│   │   └── linker.py                  # Code-to-docs linking
│   └── cli/
│       ├── __init__.py
│       ├── scan.py                    # CLI: codebase-scan
│       ├── inventory.py               # CLI: codebase-inventory
│       └── hooks.py                   # CLI: codebase-hooks
├── docs/
│   ├── scanning.md
│   ├── inventory.md
│   ├── hooks.md
│   └── governance.md
└── tests/
    ├── unit/
    │   ├── test_scanner.py
    │   ├── test_inventory.py
    │   └── test_hooks.py
    ├── integration/
    │   └── test_full_scan.py
    └── fixtures/
        ├── sample_repo/
        └── expected_outputs/
```

## CLI Entry Points

```toml
[project.scripts]
codebase-scan = "codebase_catalog.cli.scan:main"
codebase-inventory = "codebase_catalog.cli.inventory:main"
codebase-hooks = "codebase_catalog.cli.hooks:main"
```

## API Design

### Scanner API

```python
from codebase_catalog import CodebaseScanner

# Initialize scanner
scanner = CodebaseScanner(
    root_path="/path/to/repo",
    exclude_patterns=["**/test_*", "**/__pycache__/*"],
    include_patterns=["**/*.py"]
)

# Scan repository
inventory = scanner.scan()

# Access results
for module in inventory.modules:
    print(f"Module: {module.name}")
    for func in module.functions:
        print(f"  Function: {func.name}({func.signature})")
        print(f"  Docstring: {func.docstring[:100]}...")

# Export
inventory.to_json("code_inventory.json")
inventory.to_jsonl("code_inventory.jsonl")
inventory.to_postgres(connection_string="postgresql://...")
```

### Inventory Generator API

```python
from codebase_catalog import InventoryGenerator

# Generate documentation
generator = InventoryGenerator(
    inventory_source="code_inventory.json",  # or PostgreSQL connection
    output_dir="docs/code/"
)

# Generate all documentation
generator.generate_all()

# Or generate specific outputs
generator.generate_index("docs/code/INDEX.md")
generator.generate_module_docs("docs/code/modules/")
```

### Pre-commit Hook API

```python
from codebase_catalog.hooks import InventoryHook, ManifestHook

# Run inventory update hook
hook = InventoryHook(
    repo_root="/path/to/repo",
    inventory_db="postgresql://...",
    staged_only=True
)
result = hook.run()
if not result.success:
    print(f"Hook failed: {result.message}")
    sys.exit(1)

# Run manifest validation hook
manifest_hook = ManifestHook(
    manifest_path="data/MANIFEST.md",
    data_dir="data/"
)
result = manifest_hook.run()
```

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: update-code-inventory
        name: Update Code Inventory
        entry: codebase-hooks inventory
        language: system
        types: [python]
        pass_filenames: false

      - id: check-data-manifest
        name: Check Data Manifest
        entry: codebase-hooks manifest --manifest data/MANIFEST.md
        language: system
        types: [file]
        pass_filenames: false
```

## Database Schema

The package can output to PostgreSQL using this schema (compatible with existing `cohort_projections_meta`):

```sql
-- Code inventory table
CREATE TABLE code_inventory (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    module_name TEXT,
    item_type TEXT NOT NULL,  -- 'function', 'class', 'method'
    item_name TEXT NOT NULL,
    signature TEXT,
    docstring TEXT,
    line_start INTEGER,
    line_end INTEGER,
    dependencies JSONB,
    git_commit TEXT,
    last_modified TIMESTAMP,
    UNIQUE(file_path, item_type, item_name)
);

-- Governance links table
CREATE TABLE governance_links (
    id SERIAL PRIMARY KEY,
    code_item_id INTEGER REFERENCES code_inventory(id),
    doc_type TEXT,  -- 'adr', 'sop', 'readme'
    doc_path TEXT,
    link_type TEXT,  -- 'implements', 'documents', 'references'
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Migration Plan

### Phase 1: Package Scaffolding (30 min)

1. Create repository at `~/workspace/libs/codebase_catalog`
2. Initialize with `uv init`
3. Create `.envrc` per REPOSITORY_STANDARDS.md
4. Add dependencies (`psycopg2`, `pathspec`)

### Phase 2: Code Migration (1.5 hours)

1. **Scanner**: Refactor `scan_repository.py` into modular structure
2. **Inventory**: Extract `generate_inventory_docs.py`
3. **Archive**: Extract `archive_manager.py`
4. **Hooks**: Refactor `update_code_inventory.py` and `check_data_manifest.py`
5. **Governance**: Extract governance parsing logic

### Phase 3: API Refinement (30 min)

1. Design clean public API
2. Add `__init__.py` exports
3. Create CLI entry points
4. Write comprehensive docstrings

### Phase 4: Testing (1 hour)

1. Create test fixtures (sample repository structure)
2. Write unit tests for scanner
3. Write unit tests for hooks
4. Create integration tests

### Phase 5: Documentation (30 min)

1. Create package README.md
2. Create AGENTS.md for AI agent guidance
3. Document database schema
4. Add pre-commit configuration examples

### Phase 6: Integration (30 min)

1. Add package to cohort_projections as editable dependency
2. Update pre-commit hooks to use package
3. Verify all hooks work correctly
4. Update REPOSITORY_INVENTORY.md

## Dependencies

### Required

- `pathspec>=0.11.0` - Gitignore-style pattern matching
- `psycopg2>=2.9.0` - PostgreSQL connection (optional, for DB output)

### Development Only

- `pytest` - Testing
- `pytest-cov` - Coverage
- `ruff` - Linting
- `mypy` - Type checking

## Use Cases

### 1. AI Agent Onboarding

When an AI agent starts working on a repository, it can use the inventory to quickly understand:
- What scripts exist and their purposes
- Function signatures and documentation
- Dependencies between modules
- Links to governance documentation

### 2. Documentation Generation

Automatically generate and maintain:
- Code index with navigation
- Module documentation
- Function reference
- Change logs

### 3. Pre-commit Quality Gates

Enforce repository hygiene:
- Code inventory stays current
- Data manifest matches actual files
- Governance links are maintained
- No orphaned code

### 4. Repository Auditing

Answer questions like:
- What code has no documentation?
- What governance documents have no linked code?
- What code hasn't been modified in 6 months?
- What are the most complex modules?

## Success Criteria

The extraction is complete when:

1. [ ] Package is installable via `uv add --editable ../libs/codebase_catalog`
2. [ ] CLI commands work: `codebase-scan`, `codebase-inventory`, `codebase-hooks`
3. [ ] Pre-commit hooks work with new package
4. [ ] All tests pass
5. [ ] cohort_projections hooks updated and working
6. [ ] Package is listed in REPOSITORY_INVENTORY.md
7. [ ] Database schema is documented

## Future Enhancements

After initial extraction, consider:

1. **Language expansion**: Support for JavaScript, TypeScript, SQL
2. **Visualization**: Dependency graphs, module relationships
3. **Metrics**: Code complexity, documentation coverage scores
4. **GitHub integration**: Issue/PR linking, commit history
5. **VS Code extension**: Real-time inventory updates
6. **Multi-repo support**: Scan across workspace repositories

## Related Documents

- [ADR-023](023-package-extraction-strategy.md): Parent extraction strategy
- [scan_repository.py](../../scripts/intelligence/scan_repository.py): Source code
- [generate_inventory_docs.py](../../scripts/intelligence/generate_inventory_docs.py): Source code
- [update_code_inventory.py](../../scripts/hooks/update_code_inventory.py): Source code
- [check_data_manifest.py](../../scripts/hooks/check_data_manifest.py): Source code
