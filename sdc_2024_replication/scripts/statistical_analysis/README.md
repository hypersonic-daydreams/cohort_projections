# Statistical Analysis Scripts

**Directory:** `sdc_2024_replication/scripts/statistical_analysis/`

This directory contains the Python modules for the SDC 2024 replication study and subsequent journal revisions.

## 🏗️ Architecture: The "Data Loader" Pattern

All active analysis modules must load data via `data_loader.py`. This ensures consistent access to the PostgreSQL database (`demography_db`) and avoids scattered `pd.read_csv` calls.

**Do not add new CSV loading logic.** Add a function to `data_loader.py` if new data is needed.

### Core Modules (Active)
| Module | Description | Status |
|--------|-------------|--------|
| `data_loader.py` | **Central Data Access**. Used by all active modules. | ✅ Core |
| `module_7_causal_inference.py` | DiD and Synthetic Control models (Travel Ban/COVID). | ✅ Reworked |
| `module_2_1_1_unit_root_tests.py` | Stationarity tests for time series. | ✅ Active |
| `module_1_1_descriptive_statistics.py` | Summary stats strings and tables. | ✅ Active |
| `module_10_two_component_estimand.py`| Immigration status durability estimand. | ⚠️ Pending Refactor |

## 🗄️ Archive Policy

Legacy scripts that are no longer in active use (e.g., old ARIMA models, deprecated cointegration tests) are moved to `_archive/`.

**To Restore:**
Move the script back to the root of this directory and update it to import `data_loader`.

## 📚 Deep Research

"Deep Research" is a **manual workflow**.
- Reports are generated via Gemini Interface.
- Outputs are saved to `docs/research/`.
- There is no python script for this.
