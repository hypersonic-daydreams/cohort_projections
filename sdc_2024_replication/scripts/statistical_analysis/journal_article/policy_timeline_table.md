# Policy Timeline Table: Federal Immigration Events and Model Variables

## Overview

This table maps federal immigration policy events to their expected mechanisms and empirical model variables. It provides journal-standard documentation of how policy operates in the forecasting framework, following Recommendation #8 from ADR-021.

**Conceptual Framework**: Immigration policy affects North Dakota's foreign-born population through three channels:
1. **Supply/Faucet**: Federal policy sets admission ceilings and program availability
2. **Allocation/Pipe**: State-level reception capacity determines ND's share
3. **Retention/Stickiness**: Legal status durability affects long-term presence

---

## Policy Events and Model Variables

### Refugee Ceiling Determinations (Annual)

| Event | Date Range | Expected Mechanism | Model Variable | Primary Source |
|-------|------------|-------------------|----------------|----------------|
| FY2010 Ceiling: 80,000 | Oct 2009 - Sep 2010 | Supply constraint | Baseline period | Presidential Determination 2009-32 (Oct 2009) |
| FY2011 Ceiling: 80,000 | Oct 2010 - Sep 2011 | Supply constraint | Baseline period | Presidential Determination 2010-14 (Sep 2010) |
| FY2012 Ceiling: 76,000 | Oct 2011 - Sep 2012 | Supply constraint | Baseline period | Presidential Determination 2011-17 (Sep 2011) |
| FY2013 Ceiling: 70,000 | Oct 2012 - Sep 2013 | Supply constraint | Baseline period | Presidential Determination 2012-19 (Sep 2012) |
| FY2014 Ceiling: 70,000 | Oct 2013 - Sep 2014 | Supply constraint | Baseline period | Presidential Determination 2013-14 (Sep 2013) |
| FY2015 Ceiling: 70,000 | Oct 2014 - Sep 2015 | Supply constraint | Baseline period | Presidential Determination 2014-16 (Sep 2014) |
| FY2016 Ceiling: 85,000 | Oct 2015 - Sep 2016 | Expanded supply | Expansion regime end | Presidential Determination 2015-14 (Sep 2015) |
| FY2017 Ceiling: 110,000 | Oct 2016 - Sep 2017 | Ceiling high but admissions restricted | Regime break marker | Presidential Determination 2016-13 (Sep 2016) |
| FY2018 Ceiling: 45,000 | Oct 2017 - Sep 2018 | Sharp supply reduction | Restriction regime | Presidential Determination 2017-13 (Oct 2017) |
| FY2019 Ceiling: 30,000 | Oct 2018 - Sep 2019 | Historic low ceiling | Restriction regime | Presidential Determination 2018-10 (Oct 2018) |
| FY2020 Ceiling: 18,000 | Oct 2019 - Sep 2020 | Historic minimum | Restriction regime | Presidential Determination 2019-11 (Nov 2019) |
| FY2021 Ceiling: 15,000 → 62,500 | Oct 2020 - Sep 2021 | Initial restriction, late increase | COVID + transition | PD 2020-12 (Oct 2020); PD 2021-05 (May 2021) |
| FY2022 Ceiling: 125,000 | Oct 2021 - Sep 2022 | Large increase, parole surge | Volatility regime | Presidential Determination 2021-10 (Oct 2021) |
| FY2023 Ceiling: 125,000 | Oct 2022 - Sep 2023 | Maintained high ceiling | Volatility regime | Presidential Determination 2022-07 (Sep 2022) |
| FY2024 Ceiling: 125,000 | Oct 2023 - Sep 2024 | Maintained high ceiling | Volatility regime | Presidential Determination 2023-09 (Sep 2023) |

---

### Executive Actions and Federal Rules

| Event | Date Range | Expected Mechanism | Model Variable | Primary Source |
|-------|------------|-------------------|----------------|----------------|
| **Travel Ban (EO 13769)** | Jan 27 - Feb 3, 2017 | Processing halt (7 nations); refugee suspension | Intervention dummy (D_travel_ban) | Executive Order 13769 (Jan 27, 2017); *Washington v. Trump* TRO (Feb 3, 2017) |
| **Travel Ban (EO 13780)** | Mar 6, 2017 - Jun 26, 2018 | Modified ban; 120-day refugee suspension | Intervention dummy | Executive Order 13780 (Mar 6, 2017); *Trump v. Hawaii*, 585 U.S. ___ (2018) |
| **Travel Ban v3.0 (Proclamation 9645)** | Sep 24, 2017 - Jan 20, 2021 | Permanent nationality restrictions | Regime dummy (R_restriction) | Proclamation 9645 (Sep 24, 2017); upheld *Trump v. Hawaii* (Jun 26, 2018) |
| **Public Charge Rule (Proposed)** | Oct 10, 2018 | Chilling effect on LPR applications | Anticipated friction | 83 Fed. Reg. 51114 (Oct 10, 2018) |
| **Public Charge Rule (Final)** | Feb 24, 2020 - Mar 9, 2021 | Wealth test for LPR admission | Processing friction | 84 Fed. Reg. 41292 (Aug 14, 2019); enjoined Mar 9, 2021 |
| **Title 42 (COVID Border)** | Mar 20, 2020 - May 11, 2023 | Border expulsions; asylum restrictions | Border supply constraint | CDC Order (Mar 20, 2020); 85 Fed. Reg. 17060; ended May 11, 2023 |
| **COVID-19 Travel Restrictions** | Mar 2020 - Nov 2021 | Consular closures; flight suspensions | Processing friction | Proclamations 9984, 9992, 9993 (Mar 2020); lifted Nov 8, 2021 |
| **Travel Ban Revocation** | Jan 20, 2021 | Removed nationality restrictions | Regime transition marker | Proclamation 10141 (Jan 20, 2021) |

---

### Parole and Humanitarian Programs

| Event | Date Range | Expected Mechanism | Model Variable | Primary Source |
|-------|------------|-------------------|----------------|----------------|
| **Operation Allies Welcome (OAW)** | Aug 2021 - present | Afghan parole; temporary status | Temporary-status component (Y_t^temp) | DHS OAW Fact Sheet (Aug 2021); Afghan Supplemental Appropriations Act (2021) |
| **Afghan Adjustment Act** (proposed) | 2022 - (not enacted) | Would convert parole to LPR | Regularization probability parameter | H.R. 8685 (117th Congress); S. 4787 |
| **Uniting for Ukraine (U4U)** | Apr 21, 2022 - present | Ukrainian parole; 2-year status | Temporary-status component (Y_t^temp) | DHS Announcement (Apr 21, 2022); 8 U.S.C. 1182(d)(5) |
| **CHNV Parole Programs** | Jan 2023 - present | Cuba, Haiti, Nicaragua, Venezuela parole | Temporary-status component (Y_t^temp) | DHS Announcement (Jan 5, 2023); 30,000/month cap |
| **Welcome Corps (Private Sponsorship)** | Jan 2023 - present | Private refugee sponsorship pilot | Capacity expansion channel | DOS Announcement (Jan 19, 2023); 5,000 FY2023 target |
| **Welcome Corps Extended** | 2024 - present | Expanded private sponsorship | Capacity expansion channel | DOS Updates (2024) |

---

### North Dakota-Specific Events

| Event | Date Range | Expected Mechanism | Model Variable | Primary Source |
|-------|------------|-------------------|----------------|----------------|
| **LSSND Closure** | Jan 2021 | Reception capacity shock; resettlement halt | DiD/Synthetic Control treatment | Lutheran Social Services ND Announcement (2020); LIRS transition |
| **Global Refuge (LIRS) Assumes Operations** | 2021 - present | Capacity rebuilding; new intake pipeline | Capacity recovery parameter | Global Refuge (formerly LIRS) partnership with ND |
| **Bakken Oil Boom Peak** | 2012 - 2015 | Labor demand pull; secondary migration | Labor demand covariate | NDIC Production Data; ACS migration flows |
| **Bakken Oil Decline** | 2015 - 2016 | Reduced labor demand pull | Labor demand covariate | NDIC Production Data |

---

## Model Variable Summary

### Regime Variables (R_t)

| Regime | Period | Definition | Key Policy Features |
|--------|--------|------------|---------------------|
| **Expansion** | 2010-2016 | R_t = 1 | High ceilings (70K-110K); stable USRAP; strong pull factors |
| **Restriction** | 2017-2020 | R_t = 2 | Low ceilings (18K-45K); travel bans; processing friction |
| **Volatility** | 2021-2024 | R_t = 3 | High ceilings (62.5K-125K); parole surge; capacity rebuilding |

### Intervention Dummies

| Variable | Type | Timing | Interpretation |
|----------|------|--------|----------------|
| D_travel_ban | Pulse/Step | 2017Q1 - 2021Q1 | Immediate processing disruption |
| D_covid | Pulse | 2020Q1 - 2021Q4 | Temporary supply + processing shock |
| D_lssnd | Step (ND-specific) | 2021Q1 onwards | Permanent capacity shock (treatment for synthetic control) |
| D_parole | Step | 2021Q3 onwards | Shift in status composition |

### Status-Specific Components

| Variable | Definition | Status Type | Retention Hazard |
|----------|------------|-------------|------------------|
| Y_t^dur | Durable-status arrivals | Refugee, LPR, SIV | Low attrition; path to citizenship |
| Y_t^temp | Temporary-status arrivals | Parole (OAW, U4U, CHNV) | High attrition at years 2-4 without regularization |

---

## Scenario Lever Mapping

| Scenario | Ceiling Lever | Parole Lever | Regularization Lever | Capacity Lever |
|----------|--------------|--------------|---------------------|----------------|
| **Durable-Growth** | High (125K+) | Programs continue | High probability (legislation passes) | Full recovery |
| **Parole-Cliff** | Moderate | Programs continue | Low probability (no legislation) | Moderate |
| **Restriction-Drag** | Low (< 50K) | Programs end | N/A | Constrained |
| **Private Sponsorship** | Moderate | Decline | Moderate | Welcome Corps expansion |

---

## Data Sources and Notes

### Primary Source Categories

1. **Presidential Determinations**: Annual refugee ceiling documents, available via Federal Register and White House archives
2. **Executive Orders**: Published in Federal Register; legal status via court decisions
3. **Federal Register Rules**: Proposed and final rules with regulatory impact
4. **DHS/DOS Announcements**: Official program announcements; archived on agency websites
5. **Court Decisions**: Supreme Court and Circuit Court opinions on immigration policy
6. **State-Level Sources**: LSSND closure announcements; ND administrative data

### Temporal Alignment Notes

- **Fiscal Year**: Federal immigration data uses Oct 1 - Sep 30 fiscal year
- **Calendar Year**: Census PEP uses Jan 1 - Dec 31 calendar year
- **Alignment**: FY data lagged 3 months relative to CY; refugee share calculations may show >100% due to timing mismatch

---

## References

1. Presidential Determinations on Refugee Admissions. Office of the Federal Register. https://www.federalregister.gov/presidential-documents/presidential-determinations
2. Executive Order 13769, "Protecting the Nation from Foreign Terrorist Entry into the United States," 82 Fed. Reg. 8977 (Jan 27, 2017)
3. Trump v. Hawaii, 585 U.S. ___ (2018)
4. CDC Order: "Order Suspending Introduction of Certain Persons From Countries Where a Communicable Disease Exists," 85 Fed. Reg. 17060 (Mar 26, 2020)
5. DHS, "Operation Allies Welcome" Fact Sheets. https://www.dhs.gov/allieswelcome
6. DHS, "Uniting for Ukraine" Program Announcement (Apr 21, 2022). https://www.uscis.gov/ukraine
7. DHS, "Processes for Cubans, Haitians, Nicaraguans, and Venezuelans" (Jan 5, 2023). https://www.dhs.gov/news/2023/01/05
8. State Department, "Welcome Corps" Program Announcement (Jan 19, 2023). https://welcomecorps.org/
9. Refugee Processing Center, Historical Arrivals Data. https://www.rpc.state.gov/archives/

---

*Document Version: 1.0*
*Created: 2026-01-02*
*ADR Reference: ADR-021 Recommendation #8*
