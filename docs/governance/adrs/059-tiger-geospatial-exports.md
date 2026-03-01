# ADR-059: TIGER Geospatial Exports

## Status
Proposed

## Date
2026-03-01

## Context

The codebase has placeholder stubs for TIGER integration (`geography_loader.py`: `_load_counties_from_tiger()`, `_load_places_from_tiger()`) and geospatial export (`writers.py`: `write_projection_shapefile()`). Census TIGER 2020 shapefiles are already available locally at `data/interim/geographic/tiger2020/`. The `geopandas>=0.13.0` and `shapely>=2.0.0` dependencies are in the project's optional `geo` group. Implementing these stubs would enable GeoJSON and Shapefile exports for projection outputs.

### Requirements
- Load TIGER boundary files for ND counties and places
- Join projection data to geographic boundaries by FIPS code
- Export GeoJSON (primary) and Shapefile (optional) formats
- Wire into the export pipeline (`03_export_results.py`) as an additional format option
- Respect confidence tiers for publication granularity

### Challenges
- Geospatial dependencies (geopandas, shapely) are optional — must degrade gracefully
- TIGER place boundaries may not perfectly match the project's place universe (dissolved/new places)
- File sizes for multi-year exports could be large

## Decision

*To be completed during implementation.*

## Consequences

### Positive
1. Enables GIS-ready projection outputs for mapping and spatial analysis
2. Implements long-standing placeholder stubs
3. Supports stakeholder workflows that require geographic visualization

### Negative
1. Adds optional dependency on geopandas/shapely for geo exports
2. Increases export output file count and storage requirements

## Implementation Notes

### Key Functions/Classes
- `_load_counties_from_tiger(vintage)`: Load county boundaries from TIGER shapefiles
- `_load_places_from_tiger(vintage)`: Load place boundaries from TIGER shapefiles
- `write_projection_shapefile()`: Join projections to boundaries and export

### Configuration Integration
Add TIGER shapefile paths to `geographic.reference_data` in `projection_config.yaml`.

### Testing Strategy
Unit tests for TIGER loading, boundary filtering, projection join. Integration tests for pipeline export with `--formats geojson`.

### Documentation
- [ ] Update `docs/methodology.md` with geospatial output documentation

## References

1. Census TIGER/Line Shapefiles: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
2. TIGER 2020 (local): `data/interim/geographic/tiger2020/`
3. ADR-012: Output Export Format Strategy

## Revision History

- **2026-03-01**: Initial proposal (ADR-059) - TIGER geospatial export implementation

## Related ADRs

- ADR-012: Output Export Format Strategy (extends with geospatial formats)
- ADR-010: Geographic Scope (FIPS-based geography hierarchy)
- ADR-033: City-Level Projection Methodology (place-level exports)
