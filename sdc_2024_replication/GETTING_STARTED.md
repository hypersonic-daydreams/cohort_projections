# Getting Started with SDC 2024 Replication

This guide helps you quickly get started with running SDC 2024 methodology variants.

## Purpose

The SDC 2024 Replication package reproduces the North Dakota State Data Center's 2024 population projection methodology. It enables comparison of different data inputs and assumptions to understand their effects on demographic projections.

## Quick Start

### Run All Variants (Recommended)

```bash
cd sdc_2024_replication/scripts
python run_all_variants.py
```

This runs all three variants and produces a comparison table.

### Run Specific Variants

```bash
# Run only the original (2020 data) variant
python run_all_variants.py --variant original

# Run only the updated (2024 data) variant
python run_all_variants.py --variant updated

# Run only the immigration policy variant
python run_all_variants.py --variant policy

# Run multiple specific variants
python run_all_variants.py --variant original --variant updated
```

### List Available Variants

```bash
python run_all_variants.py --list
```

## Available Variants

| Variant | Flag | Description |
|---------|------|-------------|
| **Original** | `--variant original` | SDC methodology with original 2020 data |
| **Updated** | `--variant updated` | SDC methodology with 2024 Census + 2023 CDC data |
| **Policy** | `--variant policy` | Updated data with CBO-based immigration adjustment |

## Output Locations

All outputs are saved to `sdc_2024_replication/output/`:

| File | Description |
|------|-------------|
| `three_variant_comparison.csv` | Main comparison table with all variants |
| `original_variant_state_totals.csv` | Original variant detailed results |
| `updated_variant_state_totals.csv` | Updated variant detailed results |
| `policy_variant_state_totals.csv` | Policy variant detailed results |

## Data Directories

| Variant | Data Directory |
|---------|----------------|
| Original | `sdc_2024_replication/data/` |
| Updated | `sdc_2024_replication/data_updated/` |
| Policy | `data/processed/immigration/rates/` (project root) |

## Example Output

When you run all variants, you'll see output like:

```
============================================================
SDC 2024 Methodology: Variant Comparison
============================================================
Generated: 2025-12-28 15:30:00 UTC
Running variants: original, policy, updated

...

COMPARISON TABLE
================================================================================
 year  SDC Official  Original (2020 data)  Updated (2024 data)  Immigration Policy
 2020       779,094               779,094              814,044             814,044
 2025       796,989               803,312              831,456             823,987
 ...
 2050       957,194               971,055            1,013,400             944,587
```

## Common Tasks

### Regenerate Immigration Policy Data

If the immigration policy data needs to be regenerated:

```bash
cd sdc_2024_replication/scripts

# Analyze migration components from Census data
python analyze_migration_components.py

# Prepare the adjusted migration rates
python prepare_immigration_policy_data.py

# Then run the variants
python run_all_variants.py
```

### View Documentation

- **Methodology**: See [METHODOLOGY_SPEC.md](METHODOLOGY_SPEC.md) for the complete SDC methodology
- **Data Sources**: See manifests in each data directory
- **Architecture Decisions**: See [docs/governance/adrs/](../docs/governance/adrs/) for ADRs

## Deprecated Scripts

The following scripts are deprecated. Please use `run_all_variants.py` instead:

| Deprecated Script | Replacement Command |
|-------------------|---------------------|
| `run_replication.py` | `python run_all_variants.py --variant original` |
| `run_both_variants.py` | `python run_all_variants.py --variant original --variant updated` |
| `run_three_variants.py` | `python run_all_variants.py --all` |

## Troubleshooting

### "Data not found" errors

Ensure you're running from the correct directory:
```bash
cd sdc_2024_replication/scripts
python run_all_variants.py
```

### Missing dependencies

Install required packages:
```bash
uv sync  # or pip install pandas numpy
```

### Policy variant missing data

The policy variant data is located in the project-level data directory. Ensure it exists:
```bash
ls ../data/processed/immigration/rates/
```

If missing, regenerate with `prepare_immigration_policy_data.py`.

## Further Reading

- [README.md](README.md) - Full project documentation
- [METHODOLOGY_SPEC.md](METHODOLOGY_SPEC.md) - Detailed methodology specification
- [ADR-017](../docs/governance/adrs/017-sdc-2024-methodology-comparison.md) - SDC vs Baseline comparison
- [ADR-018](../docs/governance/adrs/018-immigration-policy-scenario-methodology.md) - Immigration policy methodology

---

**Last Updated:** 2026-02-02
