# Configuration Reference

Complete reference for all configuration files in the cohort projections system.

**Related**: [environment-setup.md](./environment-setup.md) | [AGENTS.md](../../AGENTS.md)

---

## Configuration Files Overview

| File | Purpose | Required |
|------|---------|----------|
| `config/projection_config.yaml` | Main projection settings | Yes |
| `config/data_sources.yaml` | Data acquisition manifest | No |
| `config/nd_brand.yaml` | Visualization colors/styling | No |
| `config/evaluation_config.yaml` | Evaluation framework settings | No |
| `config/observatory_config.yaml` | Projection Observatory settings | No |
| `config/observatory_variants.yaml` | Observatory variant catalog | No |
| `.env` | Environment variables (secrets) | No |

---

## projection_config.yaml

The primary configuration file for the projection system. All paths, parameters, and settings should be defined here rather than hard-coded.

### Loading Configuration

```python
from cohort_projections.utils import load_projection_config

config = load_projection_config()
project_id = config["bigquery"]["project_id"]
base_year = config["project"]["base_year"]
```

### Section Reference

#### project

Basic project metadata and timeline settings.

```yaml
project:
  name: "ND Population Projections 2025-2045"
  base_year: 2025              # Starting year for projections
  projection_horizon: 20       # Number of years to project
  projection_interval: 1       # Years between projection points
```

| Key | Type | Description |
|-----|------|-------------|
| `name` | string | Human-readable project name |
| `base_year` | int | First year of projections |
| `projection_horizon` | int | Number of years to project forward |
| `projection_interval` | int | Step size (1 = annual projections) |

---

#### geography

Geographic scope and hierarchy settings.

```yaml
geography:
  state: "38"  # North Dakota FIPS code

  counties:
    mode: "all"  # "all", "list", or "threshold"
    # fips_codes: ["38101", "38015"]  # If mode="list"
    # min_population: 1000            # If mode="threshold"

  places:
    mode: "threshold"
    min_population: 500  # Only places with 500+ population

  hierarchy:
    validate_aggregation: true
    aggregation_tolerance: 0.01  # 1% tolerance for rounding
    include_balance: true        # Calculate unincorporated areas
```

| Key | Type | Description |
|-----|------|-------------|
| `state` | string | State FIPS code |
| `counties.mode` | enum | "all", "list", or "threshold" |
| `counties.fips_codes` | list | County FIPS codes (if mode="list") |
| `counties.min_population` | int | Minimum population (if mode="threshold") |
| `places.mode` | enum | Same options as counties |
| `hierarchy.validate_aggregation` | bool | Check that county sums equal state |
| `hierarchy.aggregation_tolerance` | float | Acceptable rounding error |
| `hierarchy.include_balance` | bool | Calculate unincorporated remainder |

---

#### geographic

Multi-geography processing settings.

```yaml
geographic:
  parallel_processing:
    enabled: false     # Disabled due to pickle issues
    max_workers: null  # null = auto-detect CPU count
    chunk_size: 10     # Geographies per batch

  error_handling:
    mode: "continue"   # "continue" or "fail_fast"
    log_failed_geographies: true
    retry_failed: false
```

| Key | Type | Description |
|-----|------|-------------|
| `parallel_processing.enabled` | bool | Use multiprocessing (experimental) |
| `parallel_processing.max_workers` | int/null | Worker count (null = auto) |
| `error_handling.mode` | enum | "continue" or "fail_fast" |

---

#### bigquery

Google BigQuery integration settings.

```yaml
bigquery:
  enabled: true
  project_id: "antigravity-sandbox"
  credentials_path: "~/.config/gcloud/cohort-projections-key.json"
  dataset_id: "demographic_data"
  location: "US"
  use_public_data: true
  cache_queries: true
```

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable BigQuery features |
| `project_id` | string | GCP project ID |
| `credentials_path` | string | Path to service account JSON key |
| `dataset_id` | string | BigQuery dataset name |
| `location` | string | Dataset location (e.g., "US") |
| `use_public_data` | bool | Query bigquery-public-data |
| `cache_queries` | bool | Cache query results locally |

**Note**: Environment variables `GCP_PROJECT_ID` and `GCP_CREDENTIALS_PATH` override these values if set.

**Setup**: See [BIGQUERY_SETUP.md](../BIGQUERY_SETUP.md) for detailed setup instructions.

---

#### demographics

Demographic dimension definitions.

```yaml
demographics:
  age_groups:
    type: "single_year"  # or "5_year"
    min_age: 0
    max_age: 90          # 90+ is final open-ended group

  sex:
    - "Male"
    - "Female"

  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"
```

**Important**: The 6 race/ethnicity categories are mandatory. Never create new categories - map source data to these 6.

| Key | Type | Description |
|-----|------|-------------|
| `age_groups.type` | enum | "single_year" or "5_year" |
| `age_groups.min_age` | int | Youngest age (typically 0) |
| `age_groups.max_age` | int | Oldest age before open-ended group |
| `sex` | list | Sex categories |
| `race_ethnicity.categories` | list | Exactly 6 categories |

---

#### rates

Demographic rate sources and assumptions.

```yaml
rates:
  fertility:
    source: "SEER"           # or "NVSS", "custom"
    averaging_period: 5      # Years to average
    assumption: "constant"   # "constant", "trending", "scenario"
    apply_to_ages: [15, 49]  # Reproductive age range

  mortality:
    source: "SEER"
    life_table_year: 2020
    improvement_factor: 0.005  # Annual improvement (0.5%)
    cap_survival_at: 1.0       # No survival > 100%

  migration:
    domestic:
      method: "IRS_county_flows"
      averaging_period: 5
      smooth_extreme_outliers: true
    international:
      method: "ACS_foreign_born"
      allocation: "proportional"
      state_total_source: "Census_PEP"
```

| Key | Type | Description |
|-----|------|-------------|
| `fertility.source` | string | Data source for fertility rates |
| `fertility.averaging_period` | int | Years to average for stable rates |
| `mortality.improvement_factor` | float | Annual mortality improvement rate |
| `migration.domestic.method` | string | Domestic migration estimation method |

---

#### scenarios

Named projection scenarios with different assumptions.

```yaml
scenarios:
  baseline:
    name: "Baseline Projection"
    description: "Recent trends continuation"
    fertility: "constant"
    mortality: "improving"
    migration: "recent_average"
    active: true

  high_growth:
    name: "High Growth Scenario"
    fertility: "+10_percent"
    mortality: "constant"
    migration: "+25_percent"
    active: false

  sdc_2024:
    name: "SDC 2024 Methodology"
    description: "Replicates ND State Data Center 2024 projections"
    fertility: "sdc_2024_blended"
    mortality: "constant"
    migration: "sdc_2024_dampened"
    active: false
```

| Key | Type | Description |
|-----|------|-------------|
| `active` | bool | Include in projection runs |
| `fertility` | string | Fertility assumption code |
| `mortality` | string | Mortality assumption code |
| `migration` | string | Migration assumption code |

---

#### output

Output format and file settings.

```yaml
output:
  formats:
    - "parquet"
    - "csv"
  compression: "gzip"
  include_zero_cells: true
  aggregation_levels:
    - "state"
    - "county"
    - "place"
  decimal_places: 2

  excel:
    include_charts: true
    include_metadata: true
    format_numbers: true

  visualizations:
    enabled: true
    format: "png"
    dpi: 300
    style: "seaborn-v0_8-darkgrid"
    figure_size: [10, 6]
```

| Key | Type | Description |
|-----|------|-------------|
| `formats` | list | Output file formats |
| `compression` | string | Compression method for parquet |
| `aggregation_levels` | list | Geographic levels to output |
| `visualizations.dpi` | int | Image resolution |
| `visualizations.figure_size` | list | [width, height] in inches |

---

#### validation

Validation rules and thresholds.

```yaml
validation:
  compare_to_census: true
  census_benchmark_years: [2020, 2021, 2022, 2023, 2024]
  plausibility_checks:
    - "negative_population"
    - "sex_ratio"
    - "age_distribution"
    - "extreme_growth_rates"
  error_tolerance: 0.05  # 5%
```

---

#### pipeline

Pipeline orchestration settings.

```yaml
pipeline:
  data_processing:
    input_dir: "data/raw"
    output_dir: "data/processed"
    validate_outputs: true
    fail_fast: false

  projection:
    output_dir: "data/projections"
    scenarios: ["baseline"]
    resume_on_restart: true
    max_retries: 2

  export:
    output_dir: "data/exports"
    formats: ["csv", "excel"]
    create_packages: true
```

---

## data_sources.yaml

Documents all data sources required for the projection system. Used by `scripts/fetch_data.py` and as a reference for manual data acquisition.

### Structure

```yaml
data_sources:
  geographic:
    nd_counties:
      description: |
        North Dakota county reference data (53 counties).
      source_paths:
        - "${HOME}/projects/ndx-econareas/data/..."
      destination: "data/raw/geographic/nd_counties.csv"
      external_source: "Census Bureau Population Estimates Program"
      external_url: "https://..."
      required_columns:
        - "county_fips"
        - "county_name"
      notes: |
        Filter to STATE=38 for North Dakota.

  population:
    # Population data sources...

  fertility:
    # Fertility rate sources...

  mortality:
    # Life table sources...

  migration:
    # Migration data sources...
```

### Key Fields

| Field | Description |
|-------|-------------|
| `description` | What this data provides |
| `source_paths` | Local paths to copy from (supports `${HOME}`) |
| `destination` | Where to place in this repo |
| `external_source` | Official source name |
| `external_url` | URL to download manually |
| `required_columns` | Columns that must be present |
| `notes` | Processing instructions |

---

## nd_brand.yaml

North Dakota Department of Commerce brand colors for consistent visualization styling.

### Color Definitions

```yaml
brand:
  colors:
    primary:
      horizon_blue: "#0e406a"
      harvest_orange: "#faa21b"
      freshwater_blue: "#049fda"
      summer_green: "#709749"
      earthy_teal: "#087482"

    secondary:
      rustic_brown: "#a8353a"
      springtime_green: "#b3bd35"
      warm_gray: "#796e66"
      warm_gray_light: "#b6b0a2"
```

### Transparency Rules

Colors may be used with transparency in 10% increments (100%, 90%, 80%, etc.).

```yaml
transparency:
  allowed_values: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  hex_suffixes:
    100: "FF"
    90: "E6"
    80: "CC"
    # etc.
```

### Growth Color Rules

Automatic color selection based on growth rates:

```yaml
growth_color_rules:
  strong_positive:
    operator: ">"
    threshold: 2.0
    color_name: summer_green
  positive:
    operator: ">"
    threshold: 0.0
    color_name: springtime_green
  negative:
    operator: "else"
    color_name: rustic_brown
```

### Theme Settings

Pre-configured settings for Plotly and Matplotlib:

```yaml
plotly:
  colorway_names:
    - horizon_blue
    - harvest_orange
    - freshwater_blue
  font:
    family: "Segoe UI, Arial, sans-serif"

matplotlib:
  colorway_names:
    - horizon_blue
    - harvest_orange
  font_family: "sans-serif"
  grid_alpha: 0.3
```

---

## evaluation_config.yaml

Configuration for the evaluation framework (`cohort_projections/analysis/evaluation/`). Defines metrics, thresholds, county groupings, and scorecard weights used by walk-forward validation and benchmark comparison runs.

### Loading Evaluation Configuration

```python
from cohort_projections.utils import load_yaml_config

eval_config = load_yaml_config("config/evaluation_config.yaml")
horizons = eval_config["horizons"]
```

### Evaluation Section Reference

#### horizons

Standard forecast horizons to evaluate.

```yaml
horizons: [1, 2, 3, 5, 10, 15, 20]
```

---

#### county_groups

Named county-type groups for stratified reporting. Rural is computed dynamically as all counties not in any named group.

```yaml
county_groups:
  bakken: ["38105", "38053", "38061", "38025", "38089"]
  reservation: ["38005", "38085", "38079"]
  urban_college: ["38017", "38015", "38035", "38101"]
  # rural: all remaining counties (computed dynamically)
```

---

#### sentinel_counties

FIPS-to-name mapping for per-county monitoring in scorecards and reports.

```yaml
sentinel_counties:
  "38017": "Cass"
  "38035": "Grand Forks"
  "38101": "Ward"
  "38015": "Burleigh"
  "38105": "Williams"
  "38053": "McKenzie"
```

---

#### targets

Target variables to evaluate.

```yaml
targets:
  - population
  - births
  - deaths
  - net_migration
```

---

#### age_groups

18 standard 5-year age groups used for age-structure evaluation.

```yaml
age_groups:
  - "0-4"
  - "5-9"
  # ... through "80-84"
  - "85+"
```

---

#### accuracy_metrics

Metrics computed for every evaluation run. Each maps to a function in the `METRIC_REGISTRY` (see `schemas.py`).

```yaml
accuracy_metrics:
  - mae
  - rmse
  - mape
  - median_ape
  - wape
  - mean_signed_error
  - mean_signed_percentage_error
```

| Metric | Description |
|--------|-------------|
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `mape` | Mean absolute percentage error |
| `median_ape` | Median absolute percentage error |
| `wape` | Weighted absolute percentage error |
| `mean_signed_error` | Mean signed error (bias) |
| `mean_signed_percentage_error` | Mean signed percentage error (bias) |

---

#### realism

Structural realism thresholds. Projections exceeding these trigger warnings.

```yaml
realism:
  max_jsd_age_distribution: 0.05
  max_cohort_survival_residual: 0.10
  plausible_fertility_range: [0.0, 0.30]
  plausible_migration_rate_range: [-0.50, 0.50]
```

| Key | Type | Description |
|-----|------|-------------|
| `max_jsd_age_distribution` | float | Jensen-Shannon divergence warning threshold |
| `max_cohort_survival_residual` | float | Maximum cohort survival residual (fraction) |
| `plausible_fertility_range` | list | Age-specific TFR bounds [min, max] |
| `plausible_migration_rate_range` | list | Net migration rate bounds [min, max] |

---

#### sensitivity

Sensitivity testing defaults for perturbation analysis and Monte Carlo simulation.

```yaml
sensitivity:
  perturbation_pct:
    base_population: [0.5, 1.0]
    births: [1.0]
    deaths: [1.0]
    migration: [5.0]
  parameter_sweep_levels: 3
  monte_carlo_iterations: 500
```

| Key | Type | Description |
|-----|------|-------------|
| `perturbation_pct.*` | list | Perturbation percentages per component |
| `parameter_sweep_levels` | int | Number of sweep levels (low/medium/high) |
| `monte_carlo_iterations` | int | Number of Monte Carlo iterations |

---

#### scorecard_weights

Six-axis weights for the composite scorecard. Must sum to 1.0.

```yaml
scorecard_weights:
  near_term_accuracy: 0.25
  long_term_accuracy: 0.25
  bias_calibration: 0.15
  age_structure_realism: 0.15
  robustness_stability: 0.10
  interpretability: 0.10
```

---

#### near_term_max_horizon / long_term_min_horizon

Horizon classification thresholds used by `HorizonBands` in `schemas.py`.

```yaml
near_term_max_horizon: 5
long_term_min_horizon: 10
```

| Key | Type | Description |
|-----|------|-------------|
| `near_term_max_horizon` | int | Maximum horizon classified as near-term (inclusive) |
| `long_term_min_horizon` | int | Minimum horizon classified as long-term (inclusive) |

---

#### regimes

North Dakota historical regimes for regime-specific backtesting.

```yaml
regimes:
  stable: { start: 2000, end: 2007 }
  energy_boom: { start: 2008, end: 2014 }
  energy_bust: { start: 2015, end: 2019 }
  pandemic: { start: 2020, end: 2021 }
  post_pandemic: { start: 2022, end: 2024 }
```

---

## observatory_config.yaml

Configuration for the Projection Observatory (`cohort_projections/analysis/observatory/`). Defines paths, comparison metrics, and recommendation behavior.

### Section Reference

#### observatory

Top-level paths and method identity settings.

```yaml
observatory:
  history_dir: data/analysis/benchmark_history
  cache_dir: data/analysis/observatory
  experiment_log: data/analysis/experiments/experiment_log.csv
  variant_catalog: config/observatory_variants.yaml
  champion_method: m2026
  challenger_base_method: m2026r1
```

| Key | Type | Description |
|-----|------|-------------|
| `history_dir` | string | Directory containing benchmark run bundles |
| `cache_dir` | string | Directory for observatory cache files |
| `experiment_log` | string | Path to the append-only experiment log CSV |
| `variant_catalog` | string | Path to the variant catalog YAML |
| `champion_method` | string | Method ID of the current production champion |
| `challenger_base_method` | string | Method ID used as the base for challenger variants |

#### observatory.comparison

Metrics used for N-way comparison and ranking.

```yaml
  comparison:
    primary_metric: county_mape_overall
    secondary_metrics:
      - county_mape_rural
      - county_mape_bakken
      - county_mape_urban_college
```

| Key | Type | Description |
|-----|------|-------------|
| `primary_metric` | string | Default metric for ranking and comparison |
| `secondary_metrics` | list | Additional metrics shown in comparison tables |

#### observatory.recommendation

Settings controlling the recommender's suggestion behavior.

```yaml
  recommendation:
    max_suggestions: 5
    min_improvement_threshold: 0.02
```

| Key | Type | Description |
|-----|------|-------------|
| `max_suggestions` | int | Maximum number of next-experiment suggestions |
| `min_improvement_threshold` | float | Minimum expected metric improvement to recommend a variant |

---

## observatory_variants.yaml

Machine-readable experiment definitions for automated sweep execution. This file is the single source of truth for what parameter variants can be tested. The `VariantCatalog` module reads this file and can generate experiment spec YAML files for any variant or grid.

### Top-Level Fields

```yaml
base_method: m2026r1
base_config: cfg-20260309-college-fix-v1
```

| Key | Type | Description |
|-----|------|-------------|
| `base_method` | string | Method ID that all variants build on |
| `base_config` | string | Config ID that all variants build on |

### parameter_bounds

Min/max constraints for numeric parameters. Used by the recommender to clamp suggestions and avoid nonsensical values.

```yaml
parameter_bounds:
  college_blend_factor:
    min: 0.0
    max: 1.0
    description: "Weight between historical and converged rates"
```

Each entry has `min`, `max` (numeric bounds), and `description` (human-readable purpose).

### variants

Individual experiment definitions. Each variant overrides one or more parameters from the base config.

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Human-readable variant name |
| `slug` | yes | Short identifier used in experiment spec filenames |
| `parameter` | yes | MethodConfig key being varied |
| `value` | yes | Override value (scalar, dict, or list) |
| `hypothesis` | yes | What the experiment is testing |
| `expected_improvement` | yes | List of metrics expected to improve |
| `risk_areas` | yes | List of metrics or areas at risk of regression |
| `tier` | yes | Priority tier (1 = highest) |
| `config_only` | yes | Whether the variant can be tested via config injection alone |
| `results` | no | Populated after execution with `status`, `run_id`, `deltas`, and optional `notes` |
| `notes` | no | Additional context (e.g., why a variant requires code changes) |

### grids

Cartesian product sweep definitions for multi-parameter exploration.

```yaml
grids:
  blend-factor-sweep:
    parameters:
      college_blend_factor: [0.3, 0.5, 0.7, 0.8, 0.9]
    hypothesis: Find optimal college blend factor across the full range.
```

| Field | Required | Description |
|-------|----------|-------------|
| `parameters` | yes | Dict of parameter names to lists of values |
| `mode` | no | `cartesian` (default) or `zip` for paired parameters |
| `hypothesis` | yes | What the grid sweep is testing |

---

## Environment Variables (.env)

Environment variables for secrets and machine-specific settings. See `.env.example` for a template.

### Loading

The `.envrc` file automatically loads `.env` via direnv:

```bash
dotenv_if_exists .env
```

### Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CENSUS_API_KEY` | No | Census Bureau API key (higher rate limits) |
| `GCP_PROJECT_ID` | No | Google Cloud project ID |
| `GCP_CREDENTIALS_PATH` | No | Path to BigQuery service account key |
| `DEMOGRAPHY_DB_*` | No | PostgreSQL connection settings |
| `SDC_ANALYSIS_DATA_SOURCE` | No | Data source mode: auto/file/database |

### Precedence

Environment variables override config file values:
1. Environment variable (highest priority)
2. Config file value
3. Default value (lowest priority)

---

## Overriding Configuration

### Per-Session Override

Set environment variables before running:

```bash
export GCP_PROJECT_ID="my-other-project"
python scripts/run_projections.py
```

### Programmatic Override

Load config and modify:

```python
from cohort_projections.utils import load_projection_config

config = load_projection_config()
config["project"]["base_year"] = 2024  # Override
```

### Alternative Config Files

For experiments, copy and modify:

```bash
cp config/projection_config.yaml config/projection_config_experiment.yaml
# Edit the copy
export PROJECTION_CONFIG=projection_config_experiment
python scripts/run_projections.py
```

---

## Best Practices

1. **Never hard-code values** - Put everything in config
2. **Use `.env` for secrets** - Never commit API keys
3. **Document changes** - Update this guide when adding config options
4. **Validate early** - Check config at startup, not deep in processing
5. **Provide defaults** - Make optional settings work without configuration

---

*Last Updated: 2026-03-11*
