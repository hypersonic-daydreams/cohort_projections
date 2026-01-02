# ADR-010: Geographic Scope and Granularity

## Status
Accepted

## Date
2025-12-18

## Context

Population projections can be produced at multiple geographic levels (nation, state, county, city, tract, block). The choice of geographic scope and granularity involves trade-offs between detail, data availability, computational requirements, and statistical reliability.

### Requirements

1. **Primary Scope**: North Dakota (state FIPS code 38)
2. **Sub-State Detail**: County and place-level projections
3. **Data Availability**: Sufficient base population and rates for each geography
4. **Statistical Reliability**: Large enough populations for stable estimates
5. **Policy Relevance**: Match how demographic data is used in planning
6. **Computational Feasibility**: System can handle chosen granularity
7. **Consistency**: Geographic definitions stable over projection period

### North Dakota Geographic Context

- **State**: North Dakota (FIPS 38)
- **Counties**: 53 counties
- **Incorporated Places**: 406 cities/towns
- **Population Range**: Counties 600-191,000; Places 10-130,000 (Fargo)

## Decision

### Decision 1: Three Geographic Levels (State, County, Place)

**Decision**: Implement projections at three geographic levels: state, county, and incorporated place.

**The Three Levels**:

1. **State-Level**: North Dakota as single unit
   - Total: 1 geography
   - Use: Statewide planning, federal reporting

2. **County-Level**: All 53 North Dakota counties
   - Total: 53 geographies
   - Use: County planning, regional analysis

3. **Place-Level**: Incorporated cities and towns
   - Total: 406 geographies
   - Use: Municipal planning, local decision-making

**Rationale**:

**Why These Levels**:
- **Census Standard**: Census provides data at these levels
- **Administrative Relevance**: Match government jurisdictions
- **Data Availability**: Sufficient data for base population and rates
- **Planning Use**: Stakeholders plan at these levels

**Why Not More Detail** (tract, block group):
- Sample sizes too small for reliable demographic rates
- Fertility/mortality rates not available at this detail
- Migration data unavailable
- Too many geographies (computational burden)

**Why Not Less Detail** (state only):
- Counties and places need their own projections
- Hides within-state variation
- Less useful for local planning

### Decision 2: FIPS Codes as Primary Geographic Identifier

**Decision**: Use FIPS (Federal Information Processing Standards) codes as the primary, stable geographic identifier.

**FIPS Code Structure**:
- **State**: 2 digits (ND = 38)
- **County**: State (2) + County (3) = 5 digits (e.g., 38101 = Cass County)
- **Place**: State (2) + Place (5) = 7 digits (e.g., 3825700 = Fargo city)

**Example**:
```python
# State
fips_state = "38"  # North Dakota

# Counties (5 digits)
fips_cass = "38101"  # Cass County
fips_burleigh = "38015"  # Burleigh County

# Places (7 digits)
fips_fargo = "3825700"  # Fargo city
fips_bismarck = "3807200"  # Bismarck city
```

**Rationale**:
- **Stable**: FIPS codes don't change (unlike names)
- **Unique**: No ambiguity (unlike names - multiple "Washington" cities)
- **Standard**: All Census data uses FIPS codes
- **Sortable**: Natural ordering by code
- **Hierarchical**: Can infer state from county/place code

**Why Not Names**:
- Names change (cities rename themselves)
- Ambiguous (duplicate names across states)
- Harder to match across datasets
- Informal variations ("St." vs "Saint")

**Supplementary**: Include names for human-readability, but FIPS as key.

### Decision 3: All 53 Counties, Subset of Places (Configuration-Driven)

**Decision**: Project all 53 counties by default, but allow configuration-driven selection of places.

**County Coverage**:
```yaml
geography:
  counties: "all"  # All 53 counties
```

**Place Selection**:
```yaml
geography:
  places: "all"  # All 406 places

  # OR specify threshold
  places:
    mode: "threshold"
    min_population: 500  # Only places with 500+ population

  # OR specify list
  places:
    mode: "list"
    fips_codes:
      - "3825700"  # Fargo
      - "3807200"  # Bismarck
      - "3833900"  # Grand Forks
      - "3841500"  # Minot
```

**Rationale**:

**All Counties**:
- Only 53 counties (manageable)
- All counties need projections for planning
- Complete coverage expected

**Configurable Places**:
- 406 places is many (some very small)
- Small places (<100 population) projections unstable
- Users may only need major cities
- Allows focusing computational resources

**Typical Configuration**:
- Major places (500+ population): ~150 places
- All incorporated places: 406 places

### Decision 4: Single-Year Ages (0-90+), Not Age Groups

**Decision**: Use single-year ages (0, 1, 2, ..., 90+) rather than 5-year age groups (0-4, 5-9, etc.).

**Age Structure**:
- Ages: 0, 1, 2, ..., 89, 90+
- Total: 91 age categories
- 90+ is open-ended group (survives in place)

**Rationale**:

**Why Single-Year**:
- **Cohort-Component Method**: Designed for single-year ages
- **Precision**: Captures age-specific patterns better
- **Flexibility**: Can aggregate to 5-year groups if needed
- **Data Availability**: Census provides single-year data
- **Standard Practice**: Census Bureau uses single-year

**Why Not 5-Year Groups**:
- Loses detail (especially important ages like 18, 65)
- Can't track birth cohorts precisely
- Harder to apply fertility rates (reproductive ages 15-49)
- Requires interpolation (adds complexity and error)

**Computational Impact**:
- State: 91 ages × 2 sexes × 6 races = 1,092 cohorts
- County: 53 × 1,092 = 57,876 cohorts
- Places: 406 × 1,092 = 443,352 cohorts

**Performance**: Still tractable (< 5 minutes for all places).

### Decision 5: Age 90+ as Open-Ended Group

**Decision**: Treat age 90 and above as a single open-ended category, not separate ages 90, 91, 92, etc.

**Implication**:
- Population at age 90+ survives in place (doesn't age out)
- Survival rate for 90+ is within-group survival (staying alive), not aging forward

**Rationale**:

**Data Limitations**:
- Census often top-codes at 90+
- Life tables rarely go beyond 100
- Sample sizes very small for 95+, 100+

**Demographic Validity**:
- Standard practice in cohort-component projections
- Captures oldest-old without excessive detail
- Survival rates at very old ages are estimates anyway

**Computational Simplicity**:
- Avoids dealing with theoretical maximum age
- Standard 91-category structure

**Alternative Considered**: Age 85+ (rejected as loses useful detail; 85-89 still substantial)

### Decision 6: Geographic Hierarchy (Places Within Counties Within State)

**Decision**: Maintain explicit geographic hierarchy with aggregation capabilities.

**Hierarchy**:
```
North Dakota (State 38)
├── Cass County (38101)
│   ├── Fargo city (3825700)
│   ├── West Fargo city (3885100)
│   └── Casselton city (3812540)
├── Burleigh County (38015)
│   ├── Bismarck city (3807200)
│   ├── Lincoln city (3847020)
│   └── Mandan city (3850420)
└── ...
```

**Aggregation Rules**:
- Sum of all counties = State total
- Sum of all places within county ≤ County total (places don't cover entire county)
- Remainder: "Unincorporated areas" or "Balance of county"

**Use Cases**:
- Validate: county totals should sum to state
- Derive: calculate unincorporated population
- Analyze: compare urban (places) vs. rural (balance)

**Implementation**:
```python
# State total
state_pop = county_pops.sum()

# County total
county_pop = place_pops[place_pops['county_fips'] == '38101'].sum()

# Unincorporated
unincorporated = county_pop - place_pop
```

**Rationale**:
- **Validation**: Ensures consistency across levels
- **Flexibility**: Can aggregate to any level
- **Analysis**: Urban vs. rural trends
- **Standard Practice**: Census uses this hierarchy

### Decision 7: No Sub-County Geographies Beyond Places

**Decision**: Do not implement projections for census tracts, block groups, or ZIP codes.

**Not Included**:
- Census tracts (~20-30 in larger ND counties)
- Block groups
- ZIP Code Tabulation Areas (ZCTAs)
- School districts
- Legislative districts

**Rationale**:

**Data Limitations**:
- Fertility/mortality rates not available at tract level
- Migration data unavailable or unreliable
- Sample sizes too small for stable rates

**Statistical Reliability**:
- Tracts/blocks have small populations
- Projection errors compound quickly
- Results would be unreliable

**Computational Burden**:
- Hundreds of tracts × 1,092 cohorts each
- Processing time increases substantially
- Storage requirements increase

**Demand**:
- Limited demand for tract-level projections
- Most planning done at county/place level

**When Needed**:
- Can use proportional allocation from county projections
- E.g., "Tract X = 15% of county, apply 15% to projection"

## Consequences

### Positive

1. **Complete Coverage**: All counties, configurable places
2. **Standard Geographies**: Match Census definitions
3. **Policy Relevant**: Geographic levels match planning jurisdictions
4. **Stable Identifiers**: FIPS codes don't change
5. **Hierarchical**: Can aggregate/validate across levels
6. **Single-Year Ages**: Maximum precision and flexibility
7. **Computationally Feasible**: All places processable in minutes
8. **Data Availability**: Sufficient data at these levels

### Negative

1. **No Sub-County Detail**: Can't project tracts/blocks
2. **Small Place Instability**: Very small places have volatile projections
3. **Unincorporated Areas**: No direct projection (must calculate as remainder)
4. **Computational Load**: 443K cohorts for all places (manageable but substantial)
5. **Age 90+ Aggregation**: Loses detail for very old ages

### Risks and Mitigations

**Risk**: Small places have unreliable projections
- **Mitigation**: Warn users about small place volatility
- **Mitigation**: Offer population threshold (e.g., 500+)
- **Mitigation**: Document uncertainty/confidence intervals

**Risk**: Unincorporated areas unaccounted for
- **Mitigation**: Calculate as county minus sum of places
- **Mitigation**: Document that places ≠ entire county
- **Mitigation**: Provide "balance of county" calculations

**Risk**: FIPS codes change (rare but possible)
- **Mitigation**: Use Census vintage-specific FIPS
- **Mitigation**: Document FIPS vintage used
- **Mitigation**: Include crosswalks if codes change

**Risk**: Computational limits for all 406 places
- **Mitigation**: Batch processing if needed
- **Mitigation**: Allow user to select subset
- **Mitigation**: Optimize code for performance

## Alternatives Considered

### Alternative 1: State-Level Only

**Description**: Only produce state-level projections.

**Pros**:
- Simplest
- Most reliable (largest sample)
- Fastest

**Cons**:
- No local detail
- Limited usefulness for planning
- Doesn't meet stakeholder needs

**Why Rejected**:
- Counties and cities need projections
- State-level alone insufficient

### Alternative 2: Include Census Tracts

**Description**: Project at tract level (~200 tracts in ND).

**Pros**:
- Very detailed
- Useful for hyper-local planning

**Cons**:
- No fertility/mortality data at tract level
- Migration data unavailable
- Statistically unreliable
- Computationally expensive
- Sample sizes too small

**Why Rejected**:
- Data not available
- Results would be unreliable
- Not standard practice

### Alternative 3: 5-Year Age Groups

**Description**: Use 0-4, 5-9, 10-14, ... age groups.

**Pros**:
- Simpler (fewer cohorts)
- Matches some published tables

**Cons**:
- Loses precision
- Harder to apply fertility rates
- Can't track birth cohorts
- Requires interpolation

**Why Rejected**:
- Single-year is standard for cohort-component
- Aggregation to 5-year easy; opposite is hard

### Alternative 4: Age 85+ (not 90+) as Open-Ended

**Description**: Top-code at 85 instead of 90.

**Pros**:
- Slightly simpler
- Some projections use this

**Cons**:
- Loses detail for 85-89 age group
- 85-89 still substantial population
- 90+ is more standard

**Why Rejected**:
- 90+ more common in Census data
- Age 85-89 useful detail

### Alternative 5: Include ZIP Codes

**Description**: Project by ZIP Code Tabulation Area (ZCTA).

**Pros**:
- ZIPs familiar to users
- Useful for businesses

**Cons**:
- ZIPs cross county boundaries (messy hierarchy)
- ZIPs change over time (unstable)
- No demographic rates by ZIP
- Not designed for demographic analysis

**Why Rejected**:
- ZIPs not demographic units
- Unstable boundaries
- Places are better for planning

## Implementation Notes

### Geographic Data Storage

**Reference Data**:
```
data/
  geographic/
    nd_counties.csv          # FIPS, name, area
    nd_places.csv            # FIPS, name, county, population
    nd_fips_crosswalk.csv    # Handle any FIPS changes
```

**Projection Outputs**:
```
data/
  output/
    projections/
      state/
        nd_state_projection_2025_2045.parquet
      counties/
        nd_county_38101_projection_2025_2045.parquet  # Cass
        nd_county_38015_projection_2025_2045.parquet  # Burleigh
        ...
      places/
        nd_place_3825700_projection_2025_2045.parquet  # Fargo
        nd_place_3807200_projection_2025_2045.parquet  # Bismarck
        ...
```

### Configuration

**In `projection_config.yaml`**:
```yaml
geography:
  state: "38"                    # North Dakota FIPS
  counties: "all"               # All 53 counties

  places:
    mode: "threshold"           # "all", "threshold", or "list"
    min_population: 500         # Minimum population for inclusion

  hierarchy:
    validate_aggregation: true  # Check county sums = state
    include_balance: true       # Calculate unincorporated areas

demographics:
  age_groups:
    type: "single_year"
    min_age: 0
    max_age: 90                 # 90+ is open-ended
```

### Processing Pattern

**Multi-Geography Projections**:
```python
# State-level
state_projection = run_projection(geog="state", fips="38")

# All counties
for county_fips in nd_counties['fips']:
    county_projection = run_projection(geog="county", fips=county_fips)

# Selected places
selected_places = get_places(min_population=500)
for place_fips in selected_places['fips']:
    place_projection = run_projection(geog="place", fips=place_fips)

# Validate hierarchy
validate_aggregation(county_projections, state_projection)
```

## References

1. **Census Geographic Concepts**: https://www.census.gov/programs-surveys/geography/about/glossary.html
2. **FIPS Codes**: https://www.census.gov/library/reference/code-lists/ansi.html
3. **County Projections Methodology**: Smith, Tayman & Swanson (2001), Chapter 7
4. **Census Geographic Hierarchy**: https://www2.census.gov/geo/pdfs/reference/geodiagram.pdf

## Revision History

- **2025-12-18**: Initial version (ADR-010) - Geographic scope and granularity

## Related ADRs

- ADR-004: Core projection engine (processes by geography)
- ADR-006: Data pipeline (geographic reference data)
- ADR-007: Race and ethnicity (geography × race intersections)
- ADR-012: Output formats (geographic file organization)
