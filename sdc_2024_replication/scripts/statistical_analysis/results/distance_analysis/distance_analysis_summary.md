# Distance Analysis for North Dakota Immigration Gravity Model

**Generated:** 2025-12-31 17:24 UTC

## Overview

This exploratory analysis adds geographic distance as a covariate to the gravity model
of immigration flows to US states. Traditional gravity models predict that migration
flows decrease with distance (negative coefficient on log distance).

## Data

### North Dakota Centroid
- Latitude: 47.5515
- Longitude: -100.0659

### Distance Coverage
- Countries with centroid data: 110
- Distance range: 1046 km to 14874 km

### Nearest Countries to North Dakota
| Country | Distance (km) |
|---------|--------------|
| Canada | 1046 |
| Other Northern America | 1046 |
| Mexico | 2669 |
| Central America | 3765 |
| Caribbean | 3873 |
| Ireland | 6115 |
| United Kingdom | 6198 |
| United Kingdom (inc. Crown Dependencies) | 6198 |
| Norway | 6401 |
| Denmark | 6759 |

### Farthest Countries from North Dakota
| Country | Distance (km) |
|---------|--------------|
| Kenya | 13356 |
| Other Eastern Africa | 13762 |
| Sri Lanka | 13852 |
| Malaysia | 13870 |
| Indonesia | 13870 |
| Singapore | 14104 |
| Other Southern Africa | 14401 |
| Australia | 14730 |
| Other Australian and New Zealand Subregion | 14730 |
| South Africa | 14874 |

## Model Results

### Model Comparison

| Metric | Without Distance | With Distance |
|--------|-----------------|---------------|
| N observations | 4845 | 4845 |
| AIC | 1469265.35 | 1386919.63 |
| BIC | 1412816.00 | 1330476.77 |
| Pseudo R-squared | 0.3993 | 0.4330 |

### Distance Coefficient

- **Coefficient:** 1.2357
- **Standard Error:** 0.0051
- **p-value:** 0.000000
- **Significant at 5%:** YES

### Network Elasticity Comparison

| Model | Diaspora Elasticity |
|-------|-------------------|
| Without Distance | 0.1395 |
| With Distance | 0.1411 |

## Interpretation

**UNEXPECTED:** Distance has a significant **positive** effect on immigration flows.

A 1% increase in distance is associated with a 1.24% **increase** in flows.
This counterintuitive result may reflect:
1. Refugee resettlement patterns (distant conflict zones)
2. Selection effects (only highly motivated migrants from distant countries)
3. Omitted variables correlated with distance (e.g., colonial ties, language)
4. Policy factors (visa quotas, bilateral agreements)

## Technical Notes

### Distance Calculation
- Used Haversine (great-circle) formula
- Earth radius: 6,371 km
- Country centroids from Natural Earth / CEPII GeoDist database

### Model Specification
- Estimator: Poisson Pseudo-Maximum Likelihood (PPML)
- Standard errors: Model-based (not clustered in this exploratory analysis)
- Dependent variable: LPR admissions count by state-country pair

### Limitations
1. Centroids may not represent population-weighted origins
2. Distance may be endogenous to migration networks
3. Cross-sectional data cannot capture dynamic effects
4. Regional aggregates use proxy country centroids

---

*This analysis is exploratory and intended to inform whether distance should be included
in the main gravity model specification.*
