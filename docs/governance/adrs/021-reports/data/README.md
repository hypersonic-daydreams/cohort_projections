# ADR-021 Data Acquisition

This directory contains data acquisition reports and any acquired data files for ADR-021.

## Acquisition Status

| Data Category | Status | Report | Key Finding |
|---------------|--------|--------|-------------|
| Refugee Arrivals FY2021-2024 | **Complete** | [refugee_data_acquisition.md](./refugee_data_acquisition.md) | Data available from RPC; ND: 35→261→~350→397 |
| Parole Data (OAW, U4U) | **Complete** | [parole_data_acquisition.md](./parole_data_acquisition.md) | No state-level data published; ND estimate: 650-900 |
| ACS Secondary Migration | **Complete** | [acs_migration_data_acquisition.md](./acs_migration_data_acquisition.md) | PUMS needed; B05006 is stock not flow |

## Executive Summary

### Refugee Data (Critical Path - UNBLOCKED)
- **Source**: Refugee Processing Center (rpc.state.gov, formerly WRAPSNET)
- **Availability**: FY2021-2024 PDF/Excel reports available for download
- **ND Arrivals**:
  - FY2021: 35 (lowest since 1997; LSSND closure + COVID)
  - FY2022: 261 (71 refugees + 78 Afghan SIV + 112 parolees)
  - FY2023: ~300-350 (estimated)
  - FY2024: 397 (confirmed)
- **Key Context**: LSSND closed January 2021; geographic distribution shifted from 80% Fargo to 53% Fargo

### Parole Data (Limited Availability)
- **State-level parole data is NOT systematically published by DHS/USCIS**
- **ND Estimates** (from news sources, low-medium confidence):
  - Afghan (OAW): 50-100
  - Ukrainian (U4U): 600-800
  - CHNV: Unknown (likely minimal)
  - **Total**: 650-900 parolees
- **Proxy options**: ACS PUMS by country of birth, contact Global Refuge directly, FOIA request

### ACS Secondary Migration (Requires PUMS)
- **Current project data**: B05006 (population stock by origin country, 2009-2023)
- **Gap**: No migration flow data by nativity status
- **Solution**: ACS PUMS microdata with MIGSP + NATIVITY variables
- **Access**: IPUMS USA (free registration) or Census FTP
- **Challenge**: Small ND foreign-born population (~23,000) means high variance in estimates

## Next Steps

### Immediate Actions
1. Download RPC PDF reports for FY2021-2024 from https://www.rpc.state.gov/archives/
2. Extract ND rows and integrate into project data pipeline
3. Fetch B07007/B07407 via Census API for aggregate secondary migration validation

### Medium-Term Actions
1. Register for IPUMS USA and create PUMS extract for secondary migration analysis
2. Contact Global Refuge ND for parolee-specific data
3. Re-run Phase A Agent 1 with updated refugee data to validate estimand composition

## Data Files

| File | Format | Status |
|------|--------|--------|
| refugee_data_acquisition.md | Report | Complete |
| parole_data_acquisition.md | Report | Complete |
| acs_migration_data_acquisition.md | Report | Complete |
| nd_refugee_fy2021_2024.csv | Data | Pending (requires PDF extraction) |

---

*Last Updated: 2026-01-01*
*Status: Data acquisition research complete; data download pending*
