# ND State Data Center 2024 Population Projections

## Source Information

- **Title**: 2024 North Dakota State Data Center Population Projections of the State, Regions, and Counties
- **Publisher**: North Dakota Department of Commerce - State Data Center
- **Author**: Kevin Iverson, State Data Center Manager
- **Prepared**: February 6, 2024
- **Original File**: `ND Population Projections.pdf`

## Methodology Summary

The projections use a **modified cohort survival component method**:

1. **Base Population**: Census 2020 population by 5-year age group and sex for each county
2. **Fertility Rates**:
   - ND DHHS Vital Statistics 2016-2022
   - Blended with state and national rates (CDC NVSS 2021)
   - Rates smoothed to reduce anomalies
3. **Survival Rates**: CDC life tables for ND, published 2022 (for year 2020)
4. **Migration**:
   - Estimated from residual method comparing expected vs actual population 2000-2020
   - Four 5-year periods averaged: 2000-2005, 2005-2010, 2010-2015, 2015-2020
   - Migration rates reduced to ~60% of historical due to Bakken boom being atypical
   - Additional adjustments for college-age populations and male migration

## Key Projections

| Year | State Population | % Change from 2020 |
|------|------------------|-------------------|
| 2020 | 779,094 | - |
| 2025 | 796,989 | +2.3% |
| 2030 | 831,543 | +6.7% |
| 2035 | 865,397 | +11.1% |
| 2040 | 890,424 | +14.3% |
| 2045 | 925,101 | +18.7% |
| 2050 | 957,194 | +22.9% |

## Files in this Directory

| File | Description |
|------|-------------|
| `state_projections.csv` | Total state population by year |
| `regional_projections.csv` | Population by 8 economic regions |
| `regional_east_west_summary.csv` | East/West regional totals |
| `county_projections.csv` | Population by 53 counties |
| `state_age_sex_male.csv` | Male population by 5-year age groups |
| `state_age_sex_female.csv` | Female population by 5-year age groups |
| `state_age_sex_total.csv` | Total population by 5-year age groups |
| `components_of_change.csv` | Natural change and net migration by period |
| `demographic_age_groups.csv` | Summary age categories (children, workforce, seniors) |
| `sex_totals.csv` | Male/female totals by year |

## Economic Planning Regions

| Region # | Name | Major City | Counties |
|----------|------|------------|----------|
| 1 | Williston | Williston | Williams, McKenzie, Divide, Burke, Mountrail |
| 2 | Minot | Minot | Ward, McHenry, Pierce, Bottineau, Renville, Burke |
| 3 | Devils Lake | Devils Lake | Ramsey, Benson, Cavalier, Towner, Rolette, Nelson, Eddy, Wells |
| 4 | Grand Forks | Grand Forks | Grand Forks, Walsh, Pembina, Traill |
| 5 | Fargo | Fargo | Cass, Richland, Ransom, Sargent, Steele |
| 6 | Jamestown | Jamestown | Stutsman, Barnes, Foster, Griggs, LaMoure, Dickey, Logan, McIntosh |
| 7 | Bismarck | Bismarck | Burleigh, Morton, McLean, Mercer, Oliver, Sheridan, Kidder, Emmons, Grant, Sioux |
| 8 | Dickinson | Dickinson | Stark, Dunn, Hettinger, Adams, Bowman, Golden Valley, Billings, Slope |

## Key Findings from Report

1. **Growth driven by migration**: ~75% of 2010-2020 population change was due to migration
2. **Urbanization continues**: By 2050, ~30% of state population in Cass County alone
3. **Fargo region dominates**: Will reach nearly 1/3 of state population by 2050
4. **Rural decline**: Most rural counties expected to continue losing population
5. **Aging population**: 65+ population grows significantly through 2035
6. **Sex ratio**: 104 males per 100 females in 2020; expected to remain similar

## Caveats from Report

- Long-term projections are inherently uncertain
- Western ND oil regions are particularly unpredictable
- Migration patterns may shift unexpectedly
- 2018 projections overestimated 2020 population by ~5,500

## Contact

North Dakota Department of Commerce - State Data Center
1600 E. Century Ave., Suite 6 | PO Box 2057
Bismarck, ND 58503
701.328.5300 | commerce@nd.gov
