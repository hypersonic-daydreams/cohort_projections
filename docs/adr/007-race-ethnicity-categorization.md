# ADR-007: Race and Ethnicity Categorization

## Status
Accepted

## Date
2025-12-18

## Context

Population projections require categorizing individuals by race and ethnicity to project demographic diversity accurately. The U.S. uses complex race/ethnicity classification systems that have evolved over time, with different data sources using different coding schemes, granularity levels, and combinations.

### The Challenge

**Multiple Classification Systems**:
1. **Census Bureau**: Uses detailed race categories (White, Black, AIAN, Asian, NHPI, Some Other Race) crossed with Hispanic ethnicity
2. **SEER**: Uses bridged-race categories compatible with life tables
3. **OMB Standards**: Official federal standards (Directive 15)
4. **Historical Data**: Pre-2000 categories different from post-2000

**Trade-offs**:
- **Granularity vs. Sample Size**: More categories = smaller sample sizes, less reliable rates
- **Historical Consistency**: Need to compare across time periods with different categorizations
- **Data Availability**: Not all data sources provide all categories
- **Statistical Power**: Small populations have unstable rates
- **Policy Relevance**: Categories should match how demographic data is used

### Requirements

1. **Consistency**: Same categories across all data sources and projection components
2. **Completeness**: All individuals must fit into a category
3. **Mutual Exclusivity**: Each individual in exactly one category
4. **Data Availability**: Categories must have available fertility/mortality/migration data
5. **Statistical Validity**: Categories large enough for stable rate estimation
6. **Policy Relevance**: Match categories used in planning and reporting
7. **SEER Compatibility**: Must map to SEER life tables and vital statistics

### Key Decisions Required

1. How many categories?
2. Which specific categories?
3. How to handle Hispanic ethnicity?
4. How to handle multiracial individuals?
5. How to bridge between different source coding systems?

## Decision

### Decision 1: 6-Category System

**Decision**: Use a 6-category race/ethnicity system that balances granularity with data availability and statistical power.

**The 6 Categories**:
1. **White alone, Non-Hispanic**
2. **Black alone, Non-Hispanic**
3. **AIAN alone, Non-Hispanic** (American Indian/Alaska Native)
4. **Asian/PI alone, Non-Hispanic** (Asian/Pacific Islander combined)
5. **Two or more races, Non-Hispanic**
6. **Hispanic (any race)**

**Rationale**:

**Why 6 Categories**:
- **Data Availability**: SEER provides fertility/mortality data for these categories
- **Statistical Power**: Each category large enough in North Dakota for stable estimates
- **Census Compatibility**: Can aggregate from detailed Census categories
- **Historical Bridge**: Compatible with bridged-race time series
- **Policy Standard**: Used by many state demographic offices

**Why Not More Granular (7+)**:
- Would split Asian/Pacific Islander (small in ND: ~2% combined)
- Would split American Indian/Alaska Native by tribe (data limitations)
- Would have separate "Some Other Race" (coding ambiguity)
- Sample sizes too small for reliable fertility/mortality rates

**Why Not Less Granular (5 or fewer)**:
- Would combine minority groups, losing demographic detail
- Wouldn't match SEER categories
- Would lose policy-relevant distinctions

### Decision 2: Hispanic Ethnicity as Single Category (Not Crossed with Race)

**Decision**: Treat "Hispanic (any race)" as a single mutually exclusive category, not crossed with racial categories.

**Implication**: Someone who is Hispanic White is categorized as "Hispanic (any race)", not "White alone, Non-Hispanic".

**Comparison with Full Census Approach**:

**Full Census** (18 categories):
```
Non-Hispanic:
  - White alone
  - Black alone
  - AIAN alone
  - Asian alone
  - NHPI alone
  - Some Other Race alone
  - Two or more races

Hispanic:
  - White alone
  - Black alone
  - AIAN alone
  - Asian alone
  - NHPI alone
  - Some Other Race alone
  - Two or more races
```

**Our Approach** (6 categories):
```
1. White alone, Non-Hispanic
2. Black alone, Non-Hispanic
3. AIAN alone, Non-Hispanic
4. Asian/PI alone, Non-Hispanic
5. Two or more races, Non-Hispanic
6. Hispanic (any race)  ← Includes all racial backgrounds
```

**Rationale**:

**Demographic Justification**:
- Hispanic identity often stronger than racial identity
- Fertility/mortality patterns more similar within Hispanic group across races
- Matches how SEER publishes life tables
- Matches Census Bureau Population Estimates Program (PEP)

**Data Availability**:
- SEER provides aggregate Hispanic fertility/mortality rates
- Crossing Hispanic with race would create 12+ categories
- Many race×Hispanic cells would have insufficient sample sizes

**Practical Considerations**:
- North Dakota Hispanic population is small (~4%)
- Further subdivision not statistically reliable
- Standard practice for state-level projections

**Alternative Considered**: Full 18-category Census approach rejected due to data limitations and small sample sizes.

### Decision 3: Combine Asian and Pacific Islander

**Decision**: Combine Asian and Native Hawaiian/Pacific Islander (NHPI) into single "Asian/PI" category.

**Rationale**:

**Sample Size**:
- Combined Asian/PI: ~2% of North Dakota population
- Separate would be: Asian ~1.8%, NHPI ~0.2%
- NHPI sample too small for reliable demographic rates

**Data Availability**:
- SEER often provides combined Asian/PI life tables
- Many fertility data sources combine these groups
- Historical data pre-2000 always combined

**Demographic Similarity**:
- Both groups have lower mortality than national average
- Both have below-replacement fertility in U.S.
- Migration patterns similar (international migration driven)

**Trade-off Acknowledged**:
- Loses ability to distinguish Asian vs. Pacific Islander populations
- These groups have cultural and demographic differences
- Could revisit if North Dakota Asian/PI population grows significantly

### Decision 4: "Two or More Races" as Distinct Category

**Decision**: Include "Two or more races, Non-Hispanic" as a separate category rather than distributing to single-race groups.

**Rationale**:

**Growing Population**:
- Fastest-growing Census category nationally
- Important for future projections
- Allows tracking multiracial demographic trends

**Data Availability**:
- Census 2000+ collects multiracial data
- SEER provides bridged-race estimates
- Can estimate fertility/mortality rates

**Demographic Validity**:
- Multiracial individuals may have distinct demographic patterns
- Distributing artificially inflates single-race groups
- Preserves data as collected

**Challenges**:
- Historical data (pre-2000) lacks this category (bridge estimates needed)
- Smaller sample sizes for rate estimation
- Interpretation complexity (which races?)

**Alternative Considered**: Distribute proportionally to single-race categories (rejected as loses information).

### Decision 5: Explicit Mapping from Source Categories

**Decision**: Maintain explicit mapping dictionaries for each data source (Census, SEER, IRS) to standard 6 categories.

**Census Mapping** (from detailed categories):
```python
CENSUS_RACE_ETHNICITY_MAP = {
    # White alone, Non-Hispanic
    'NHWA': 'White alone, Non-Hispanic',
    'NH_WHITE': 'White alone, Non-Hispanic',
    'WA_NH': 'White alone, Non-Hispanic',

    # Black alone, Non-Hispanic
    'NHBA': 'Black alone, Non-Hispanic',
    'NH_BLACK': 'Black alone, Non-Hispanic',
    'BA_NH': 'Black alone, Non-Hispanic',

    # AIAN alone, Non-Hispanic
    'NHIA': 'AIAN alone, Non-Hispanic',
    'NH_AIAN': 'AIAN alone, Non-Hispanic',
    'IA_NH': 'AIAN alone, Non-Hispanic',

    # Asian/PI alone, Non-Hispanic (COMBINED)
    'NHAA': 'Asian/PI alone, Non-Hispanic',
    'NH_ASIAN': 'Asian/PI alone, Non-Hispanic',
    'NH_NHPI': 'Asian/PI alone, Non-Hispanic',  # Pacific Islander → combined
    'AA_NH': 'Asian/PI alone, Non-Hispanic',
    'NA_NH': 'Asian/PI alone, Non-Hispanic',

    # Two or more races, Non-Hispanic
    'NHTOM': 'Two or more races, Non-Hispanic',
    'NH_TOM': 'Two or more races, Non-Hispanic',
    'TOM_NH': 'Two or more races, Non-Hispanic',
    'NH_TWO_OR_MORE': 'Two or more races, Non-Hispanic',

    # Hispanic (any race)
    'H': 'Hispanic (any race)',
    'HISP': 'Hispanic (any race)',
    'HISPANIC': 'Hispanic (any race)',
    'H_WHITE': 'Hispanic (any race)',  # All Hispanic races → Hispanic
    'H_BLACK': 'Hispanic (any race)',
    'H_AIAN': 'Hispanic (any race)',
    'H_ASIAN': 'Hispanic (any race)',
    'H_NHPI': 'Hispanic (any race)',
    'H_TOM': 'Hispanic (any race)',
}
```

**SEER Mapping** (from SEER race codes):
```python
SEER_RACE_ETHNICITY_MAP = {
    # SEER uses numeric codes in some files
    1: 'White alone, Non-Hispanic',
    2: 'Black alone, Non-Hispanic',
    3: 'AIAN alone, Non-Hispanic',
    4: 'Asian/PI alone, Non-Hispanic',
    5: 'Two or more races, Non-Hispanic',  # SEER bridged-race estimate
    8: 'Hispanic (any race)',

    # SEER also uses text labels (varies by file)
    'NH White': 'White alone, Non-Hispanic',
    'NH Black': 'Black alone, Non-Hispanic',
    'NH AIAN': 'AIAN alone, Non-Hispanic',
    'NH API': 'Asian/PI alone, Non-Hispanic',
    'Hispanic': 'Hispanic (any race)',
}
```

**Rationale**:
- **Explicit is Better**: Clear mapping prevents errors
- **Auditable**: Can verify categorization logic
- **Maintainable**: Easy to update if source codes change
- **Documented**: Self-documenting code
- **Testable**: Can unit test mapping

**Error Handling**: If unmapped code encountered, log warning and drop record (don't silently misclassify).

### Decision 6: Consistent Ordering Across All Outputs

**Decision**: Always present categories in same order: White NH → Black NH → AIAN NH → Asian/PI NH → Two+ NH → Hispanic.

**Order Rationale**:
1. **White alone, Non-Hispanic**: Majority group (largest)
2. **Black alone, Non-Hispanic**: Next largest minority
3. **AIAN alone, Non-Hispanic**: Historically significant in ND
4. **Asian/PI alone, Non-Hispanic**: Growing group
5. **Two or more races, Non-Hispanic**: Smallest currently, growing
6. **Hispanic (any race)**: Cross-racial category, listed last

**Consistency Benefits**:
- Tables, charts, and outputs always in same order
- Easier to compare across reports
- Muscle memory for users
- Standard practice (matches Census tables)

**Implementation**:
```python
# Categorical type with fixed order
RACE_ETHNICITY_CATEGORIES = [
    'White alone, Non-Hispanic',
    'Black alone, Non-Hispanic',
    'AIAN alone, Non-Hispanic',
    'Asian/PI alone, Non-Hispanic',
    'Two or more races, Non-Hispanic',
    'Hispanic (any race)'
]

# Use in DataFrames
df['race'] = pd.Categorical(
    df['race'],
    categories=RACE_ETHNICITY_CATEGORIES,
    ordered=True
)
```

### Decision 7: No Residual/Unknown Category

**Decision**: Do not create an "Unknown/Other" category. All individuals must be assigned to one of the 6 categories.

**Implication**: Records with unmapped race codes are dropped (with warning) rather than placed in residual category.

**Rationale**:
- **Demographic Validity**: "Unknown" group has no meaningful fertility/mortality rates
- **Data Quality**: Forces investigation of unmapped codes
- **Completeness**: Census data should have complete race classification
- **Projection Integrity**: Can't project "Unknown" group forward

**Handling Unmapped Codes**:
```python
# Map race codes
df['race_ethnicity'] = df['race_code'].map(RACE_ETHNICITY_MAP)

# Check for unmapped
unmapped = df[df['race_ethnicity'].isna()]['race_code'].unique()
if len(unmapped) > 0:
    logger.warning(f"Unmapped race codes found: {unmapped}")
    logger.warning("These records will be dropped. Please update RACE_ETHNICITY_MAP if valid codes.")
    df = df.dropna(subset=['race_ethnicity'])
```

**When to Update Mapping**: If Census introduces new codes or SEER changes classification.

## Consequences

### Positive

1. **Consistency**: Same 6 categories across entire projection system
2. **Data Availability**: All categories have SEER fertility/mortality data
3. **Statistical Validity**: Sample sizes adequate for North Dakota
4. **Simplicity**: Manageable number of categories (6 vs. 18+)
5. **SEER Compatibility**: Matches SEER life table structure
6. **Policy Relevance**: Aligns with state demographic reporting
7. **Historical Continuity**: Can bridge to historical data
8. **Explicit Mapping**: Clear transformation from source data
9. **Projection Tractable**: 91 ages × 2 sexes × 6 races = 1,092 cohorts (manageable)
10. **Auditability**: Explicit mapping dictionaries document decisions

### Negative

1. **Lost Granularity**: Can't distinguish Asian vs. Pacific Islander
2. **Hispanic Heterogeneity**: Combines Hispanic White, Hispanic Black, etc.
3. **Small Group Challenges**: AIAN and Two+ races have small samples in ND
4. **Multiracial Complexity**: "Two or more races" is heterogeneous
5. **Future Changes**: May need to update if population composition shifts
6. **Dropped Records**: Unmapped codes result in data loss (but rare)

### Risks and Mitigations

**Risk**: Sample sizes too small for reliable rates in minority categories
- **Mitigation**: Use multi-year averaging (5 years) to increase sample
- **Mitigation**: Borrow strength from national rates if necessary
- **Mitigation**: Document uncertainty in projections

**Risk**: Data sources use incompatible race coding
- **Mitigation**: Explicit mapping dictionaries handle variations
- **Mitigation**: Comprehensive testing of mapping logic
- **Mitigation**: Validation checks for unmapped codes

**Risk**: Census or SEER changes race categories
- **Mitigation**: Version-specific mapping dictionaries
- **Mitigation**: Clear documentation of which vintage used
- **Mitigation**: Update mapping when new vintages released

**Risk**: Losing information by combining categories
- **Mitigation**: Document trade-offs in methodology notes
- **Mitigation**: Can provide more detailed categories as auxiliary outputs
- **Mitigation**: Re-evaluate if population composition changes significantly

## Alternatives Considered

### Alternative 1: Full 18-Category Census System

**Description**: Cross Hispanic ethnicity with all 6 racial categories (plus Some Other Race).

**Structure**:
- Non-Hispanic: White, Black, AIAN, Asian, NHPI, Some Other, Two+
- Hispanic: White, Black, AIAN, Asian, NHPI, Some Other, Two+

**Pros**:
- Maximum detail
- Matches Census published tables
- Captures Hispanic racial diversity

**Cons**:
- Many categories (18+)
- Sample sizes too small in ND
- SEER doesn't provide rates at this detail
- Projection complexity increases (1,092 → 3,276 cohorts)

**Why Rejected**:
- Data availability constraints
- Statistical instability
- Projection complexity not justified

### Alternative 2: 5-Category System (Combine AIAN with Other)

**Description**: Combine AIAN and Asian/PI into single "All Other" category.

**Categories**:
1. White alone, Non-Hispanic
2. Black alone, Non-Hispanic
3. All Other, Non-Hispanic
4. Two or more races, Non-Hispanic
5. Hispanic (any race)

**Pros**:
- Larger sample sizes for combined group
- Simpler projection

**Cons**:
- AIAN is significant population in ND (5%+)
- Loses important demographic distinctions
- AIAN has distinct fertility/mortality patterns

**Why Rejected**:
- AIAN population too important to combine
- Policy relevance (tribal nations in ND)
- Data available at more detail

### Alternative 3: 7-Category System (Separate Asian and Pacific Islander)

**Description**: Split Asian/PI into separate categories.

**Additional Categories**:
1. Asian alone, Non-Hispanic
2. Native Hawaiian/PI alone, Non-Hispanic

**Pros**:
- Matches Census standard
- Recognizes distinct populations

**Cons**:
- NHPI in ND is <0.2% (very small)
- Insufficient sample for reliable rates
- SEER often combines in life tables

**Why Rejected**:
- Sample size constraints
- Data availability limitations
- Can revisit if ND demographics change

### Alternative 4: Separate "Some Other Race" Category

**Description**: Include Census "Some Other Race" as distinct category.

**Pros**:
- Matches how Census collects data
- Captures growing category

**Cons**:
- "Some Other Race" is ambiguous
- Census recodes most to specific races
- SEER doesn't provide separate rates
- Not stable category over time

**Why Rejected**:
- Category has definitional issues
- Census often recodes algorithmically
- Not useful for projections

### Alternative 5: Distribute "Two or More Races" Proportionally

**Description**: Allocate multiracial individuals to single-race categories based on proportions.

**Method**:
- 50% to largest race component
- Or proportional to population shares

**Pros**:
- Avoids small multiracial category
- Simpler projection

**Cons**:
- Loses information
- Arbitrary allocation
- Ignores distinct multiracial demographic patterns
- Inflates single-race groups artificially

**Why Rejected**:
- Multiracial population growing rapidly
- Information loss not justified
- Census collects this data explicitly

## Implementation Notes

### Mapping Dictionaries Location

**File**: `/home/nigel/cohort_projections/cohort_projections/data/process/base_population.py`

**Also Used In**:
- `fertility_rates.py`
- `survival_rates.py`
- `migration_rates.py`

**Could Be Centralized**: Consider moving to `cohort_projections/utils/demographic_utils.py` for reuse.

### Validation Function

```python
def validate_race_categories(df, config=None):
    """Validate race/ethnicity categories in DataFrame."""
    if config is None:
        config = load_projection_config()

    expected_categories = config['demographics']['race_ethnicity']['categories']

    # Check all expected categories present
    actual_categories = df['race'].unique()
    missing = set(expected_categories) - set(actual_categories)
    if missing:
        logger.warning(f"Missing race categories: {missing}")

    # Check no unexpected categories
    unexpected = set(actual_categories) - set(expected_categories)
    if unexpected:
        logger.error(f"Unexpected race categories: {unexpected}")
        return False

    return True
```

### Usage in Data Processing

```python
from cohort_projections.data.process.base_population import harmonize_race_categories

# Load raw Census data
raw_pop = pd.read_csv('census_population.csv')

# Harmonize to 6 categories
pop = harmonize_race_categories(raw_pop)

# Result: 'race_ethnicity' column with 6 standard categories
```

### Configuration

**In `projection_config.yaml`**:
```yaml
demographics:
  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"
```

## References

1. **OMB Directive 15**: "Revisions to the Standards for the Classification of Federal Data on Race and Ethnicity" (1997)
   - https://www.govinfo.gov/content/pkg/FR-1997-10-30/pdf/97-28653.pdf

2. **Census Bureau Race Guidelines**: "Race and Ethnicity in the United States: 2020 Census"
   - https://www.census.gov/library/fact-sheets/2021/dec/2020-census-race-ethnicity.html

3. **SEER Bridged-Race Categories**: "Bridged-Race Population Estimates"
   - https://seer.cancer.gov/popdata/methods.html

4. **NCHS Bridged-Race**: "Bridged-Race Population Estimates Methodology"
   - https://www.cdc.gov/nchs/nvss/bridged_race.htm

5. **State Demographer Practices**: Survey of state projection methodologies
   - Washington State OFM
   - California DOF
   - Texas State Data Center

6. **Demographic Literature**:
   - Williams, D.R. (1994). "The Concept of Race in Health Services Research"
   - Humes, K.R., et al. (2011). "Overview of Race and Hispanic Origin: 2010 Census"

## Revision History

- **2025-12-18**: Initial version (ADR-007) - Race and ethnicity categorization

## Related ADRs

- ADR-001: Fertility rate processing (uses race categories)
- ADR-002: Survival rate processing (uses race categories)
- ADR-003: Migration rate processing (planned, will use race categories)
- ADR-004: Core projection engine (projects by race)
- ADR-006: Data pipeline architecture (harmonization step)
- ADR-010: Geographic scope (geographic × race intersections)

## Appendix: North Dakota Demographic Context

**2020 Census Race/Ethnicity Distribution** (approximate):

| Category | Population | Percentage |
|----------|-----------|------------|
| White alone, Non-Hispanic | 660,000 | 86% |
| Black alone, Non-Hispanic | 25,000 | 3% |
| AIAN alone, Non-Hispanic | 38,000 | 5% |
| Asian/PI alone, Non-Hispanic | 15,000 | 2% |
| Two or more races, Non-Hispanic | 12,000 | 1.5% |
| Hispanic (any race) | 30,000 | 4% |
| **Total** | **780,000** | **100%** |

**Key Observations**:
- White NH is overwhelming majority (86%)
- AIAN significant due to tribal nations (Standing Rock, Turtle Mountain, etc.)
- Hispanic population growing but still small (4%)
- Asian/PI very small (2%)
- Two or more races growing rapidly from small base

**Implication for Projections**:
- White NH rates dominate state totals
- AIAN rates critical for tribal areas
- Minority categories require multi-year averaging for stability
- Geographic variation important (AIAN concentrated in specific counties)

This demographic context supports the 6-category decision: enough detail to capture diversity, not so much that categories become statistically unstable.
