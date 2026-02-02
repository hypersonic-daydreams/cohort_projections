# Geographic Hierarchy Reference

FIPS code reference and geographic validation rules for North Dakota population projections.

**Related**: [configuration-reference.md](../guides/configuration-reference.md) | [geography_loader.py](../../cohort_projections/geographic/geography_loader.py)

---

## Overview

The projection system uses the Census Bureau's FIPS (Federal Information Processing Standards) codes to identify geographic areas. North Dakota has a three-level geographic hierarchy:

```
State (38)
  |
  +-- Counties (53 total)
        |
        +-- Incorporated Places (~406 total)
```

---

## FIPS Code Structure

### State FIPS

| Code | Name |
|------|------|
| `38` | North Dakota |

### County FIPS Format

County FIPS codes are **5 digits**: `SSCCC`
- `SS` = State FIPS (38 for North Dakota)
- `CCC` = County code (001-105, odd numbers only for ND)

**Example**: `38101` = Ward County, North Dakota

### Place FIPS Format

Place FIPS codes are **7 digits**: `SSPPPPP`
- `SS` = State FIPS (38)
- `PPPPP` = Place code

**Example**: `3825700` = Fargo city, North Dakota

---

## North Dakota Counties (53 Total)

All 53 North Dakota counties with FIPS codes:

| FIPS | County Name | Major City |
|------|-------------|------------|
| 38001 | Adams County | Hettinger |
| 38003 | Barnes County | Valley City |
| 38005 | Benson County | Minnewaukan |
| 38007 | Billings County | Medora |
| 38009 | Bottineau County | Bottineau |
| 38011 | Bowman County | Bowman |
| 38013 | Burke County | Bowbells |
| 38015 | Burleigh County | Bismarck |
| 38017 | Cass County | Fargo |
| 38019 | Cavalier County | Langdon |
| 38021 | Dickey County | Ellendale |
| 38023 | Divide County | Crosby |
| 38025 | Dunn County | Killdeer |
| 38027 | Eddy County | New Rockford |
| 38029 | Emmons County | Linton |
| 38031 | Foster County | Carrington |
| 38033 | Golden Valley County | Beach |
| 38035 | Grand Forks County | Grand Forks |
| 38037 | Grant County | Carson |
| 38039 | Griggs County | Cooperstown |
| 38041 | Hettinger County | Mott |
| 38043 | Kidder County | Steele |
| 38045 | LaMoure County | LaMoure |
| 38047 | Logan County | Napoleon |
| 38049 | McHenry County | Towner |
| 38051 | McIntosh County | Ashley |
| 38053 | McKenzie County | Watford City |
| 38055 | McLean County | Washburn |
| 38057 | Mercer County | Beulah |
| 38059 | Morton County | Mandan |
| 38061 | Mountrail County | Stanley |
| 38063 | Nelson County | Lakota |
| 38065 | Oliver County | Center |
| 38067 | Pembina County | Cavalier |
| 38069 | Pierce County | Rugby |
| 38071 | Ramsey County | Devils Lake |
| 38073 | Ransom County | Lisbon |
| 38075 | Renville County | Mohall |
| 38077 | Richland County | Wahpeton |
| 38079 | Rolette County | Rolla |
| 38081 | Sargent County | Forman |
| 38083 | Sheridan County | McClusky |
| 38085 | Sioux County | Fort Yates |
| 38087 | Slope County | Amidon |
| 38089 | Stark County | Dickinson |
| 38091 | Steele County | Finley |
| 38093 | Stutsman County | Jamestown |
| 38095 | Towner County | Cando |
| 38097 | Traill County | Hillsboro |
| 38099 | Walsh County | Grafton |
| 38101 | Ward County | Minot |
| 38103 | Wells County | Fessenden |
| 38105 | Williams County | Williston |

---

## Major North Dakota Places

Top 20 incorporated places by population:

| FIPS | Place Name | County | 2024 Pop (approx) |
|------|------------|--------|-------------------|
| 3825700 | Fargo city | Cass | 125,990 |
| 3807200 | Bismarck city | Burleigh | 73,622 |
| 3833900 | Grand Forks city | Grand Forks | 59,166 |
| 3841500 | Minot city | Ward | 48,415 |
| 3885100 | West Fargo city | Cass | 38,626 |
| 3877100 | Williston city | Williams | 29,160 |
| 3811380 | Dickinson city | Stark | 25,679 |
| 3850420 | Mandan city | Morton | 24,206 |
| 3840740 | Jamestown city | Stutsman | 15,849 |
| 3884940 | Watford City city | McKenzie | 11,085 |
| 3822020 | Devils Lake city | Ramsey | 7,396 |
| 3881300 | Valley City city | Barnes | 6,483 |
| 3826660 | Grafton city | Walsh | 4,190 |
| 3878140 | Wahpeton city | Richland | 7,762 |
| 3805860 | Beulah city | Mercer | 3,264 |
| 3856260 | New Town city | Mountrail | 2,724 |
| 3877420 | Williston (west) | Williams | - |
| 3844140 | Lincoln city | Burleigh | 4,167 |
| 3816900 | Casselton city | Cass | 2,738 |
| 3835420 | Hazen city | Mercer | 2,350 |

---

## Metropolitan Statistical Areas

North Dakota has 2 Metropolitan Statistical Areas (MSAs) and several Micropolitan areas:

### Metropolitan Areas

| CBSA Code | Name | Principal Counties |
|-----------|------|-------------------|
| 22020 | Fargo-Moorhead, ND-MN | Cass (ND), Clay (MN) |
| 13900 | Bismarck, ND | Burleigh, Morton |

### Micropolitan Areas

| CBSA Code | Name | Principal County |
|-----------|------|-----------------|
| 24220 | Grand Forks, ND-MN | Grand Forks |
| 34060 | Minot, ND | Ward |
| 22100 | Dickinson, ND | Stark |
| 48700 | Williston, ND | Williams |
| 27420 | Jamestown, ND | Stutsman |
| 48380 | Wahpeton, ND-MN | Richland |

---

## Geographic Validation Rules

### FIPS Code Validation

```python
def validate_fips(fips: str, level: str) -> bool:
    """
    Validate FIPS code format.

    Rules:
    - Must be string (not integer)
    - State: exactly 2 digits, "38" for ND
    - County: exactly 5 digits, starts with "38"
    - Place: exactly 7 digits, starts with "38"
    """
    if not isinstance(fips, str):
        return False

    if level == "state":
        return fips == "38"
    elif level == "county":
        return len(fips) == 5 and fips.startswith("38")
    elif level == "place":
        return len(fips) == 7 and fips.startswith("38")
    return False
```

### Aggregation Validation

The system validates that geographic aggregations are consistent:

1. **County totals sum to state total** (within tolerance)
   ```
   sum(all_county_populations) == state_population +/- 1%
   ```

2. **Place totals <= County total**
   ```
   sum(places_in_county) <= county_population
   ```
   (Difference is unincorporated area)

### Configuration Settings

From `config/projection_config.yaml`:

```yaml
geography:
  state: "38"  # North Dakota FIPS

  hierarchy:
    validate_aggregation: true      # Enable validation
    aggregation_tolerance: 0.01     # 1% tolerance for rounding
    include_balance: true           # Calculate unincorporated areas
```

---

## Geographic Loading Functions

### Loading Counties

```python
from cohort_projections.geographic import load_nd_counties

# Load all 53 counties
counties = load_nd_counties()
# Returns DataFrame with: state_fips, county_fips, county_name, population

# Load from specific source
counties = load_nd_counties(source="local")  # From CSV file
counties = load_nd_counties(source="tiger")  # From Census TIGER
```

### Loading Places

```python
from cohort_projections.geographic import load_nd_places

# Load all places
places = load_nd_places()

# Load places with minimum population
places = load_nd_places(min_population=500)  # ~150 places
places = load_nd_places(min_population=1000)  # ~80 places
```

### Loading Geography Lists

```python
from cohort_projections.geographic import load_geography_list

# Load FIPS codes for projection
state_list = load_geography_list("state")     # ['38']
county_list = load_geography_list("county")   # All 53 counties
place_list = load_geography_list("place")     # All ~406 places

# With custom filtering
from cohort_projections.utils import load_projection_config
config = load_projection_config()
config['geography']['places']['mode'] = 'threshold'
config['geography']['places']['min_population'] = 500
places = load_geography_list("place", config=config)
```

### Name Lookup

```python
from cohort_projections.geographic import get_geography_name

get_geography_name("38")        # "North Dakota"
get_geography_name("38017")     # "Cass County"
get_geography_name("3825700")   # "Fargo city"
```

---

## Place-to-County Mapping

Each place belongs to exactly one county. The mapping is stored in the places reference file.

```python
from cohort_projections.geographic import get_place_to_county_mapping

mapping = get_place_to_county_mapping()
# Returns DataFrame: place_fips, county_fips, place_name, county_name
```

**Example mappings:**

| Place | Place FIPS | County | County FIPS |
|-------|------------|--------|-------------|
| Fargo city | 3825700 | Cass County | 38017 |
| West Fargo city | 3885100 | Cass County | 38017 |
| Bismarck city | 3807200 | Burleigh County | 38015 |
| Mandan city | 3850420 | Morton County | 38059 |
| Minot city | 3841500 | Ward County | 38101 |

---

## Data Sources

### Reference Data Files

| File | Location | Content |
|------|----------|---------|
| County reference | `data/raw/geographic/nd_counties.csv` | All 53 counties |
| Place reference | `data/raw/geographic/nd_places.csv` | All incorporated places |
| Metro crosswalk | `data/raw/geographic/metro_crosswalk.csv` | County-to-CBSA mapping |

### External Sources

| Data Type | Source | URL |
|-----------|--------|-----|
| County FIPS | Census Bureau | census.gov/library/reference/code-lists |
| Place FIPS | Census Bureau | census.gov/geographies |
| CBSA Definitions | OMB | census.gov/programs-surveys/metro-micro |
| TIGER/Line Files | Census Bureau | census.gov/geographies/mapping-files |

---

## Common Issues

### Integer vs String FIPS

**Problem**: FIPS codes loaded as integers lose leading zeros.

```python
# Wrong
38017  # Loses context that it's a FIPS code
17     # County portion without state

# Correct
"38017"  # Full 5-digit county FIPS
```

**Solution**: Always specify dtype when loading:
```python
df = pd.read_csv("file.csv", dtype={"county_fips": str})
```

### Missing Geographies

**Problem**: Some counties missing from aggregation.

**Solution**: Use `load_nd_counties()` which validates all 53 are present.

### Place Spanning Multiple Counties

**Problem**: Some places cross county boundaries.

**Note**: In this system, each place is assigned to its primary county. Multi-county places are relatively rare in North Dakota.

---

*Last Updated: 2026-02-02*
