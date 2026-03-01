# ADR-059: TIGER/Line Geospatial Exports

## Status
Accepted

## Date
2026-03-01

## Context

Stakeholders and GIS analysts need projection results delivered as
geospatial files (GeoJSON, Shapefile) that can be loaded directly into
mapping tools (QGIS, ArcGIS, web dashboards) without manual boundary
joining. The projection system already stores Census TIGER/Line
shapefiles locally for the place-county crosswalk build
(`scripts/data/build_place_county_crosswalk.py`), but the geography
loader and output writer contained only stub implementations for TIGER
integration and geospatial export.

### Requirements
- Export county and place projection totals joined to TIGER 2020
  boundary geometries.
- Support GeoJSON (primary, single-file, web-friendly) and ESRI
  Shapefile (secondary, GIS-tool compatible) output formats.
- Integrate with the existing export pipeline (`03_export_results.py`)
  via the `--formats` flag.
- Handle the optional geopandas dependency gracefully -- the rest of the
  pipeline must continue to work when geopandas is not installed.
- Use config-driven TIGER file paths (no hard-coded paths).

### Challenges
- TIGER place shapefiles do not include county assignments; the
  place-county crosswalk is a separate artifact.
- geopandas is an optional dependency; all code paths must guard against
  its absence.
- The export pipeline must remain backward-compatible with existing
  tabular-only workflows.

## Decision

### Decision 1: TIGER boundary loading via config-driven paths

**Decision**: Implement `_load_counties_from_tiger()` and
`_load_places_from_tiger()` in `geography_loader.py`, reading shapefile
paths from `geography.reference_data.tiger_boundaries` in
`projection_config.yaml`.

**Rationale**:
- Follows the project rule of never hard-coding file paths.
- Reuses the same TIGER vintage (2020) already used for crosswalk
  construction.
- Config-driven paths allow easy updates when future TIGER vintages are
  adopted.

**Implementation**:
```yaml
# config/projection_config.yaml
geography:
  reference_data:
    tiger_boundaries:
      county_shapefile: "data/interim/geographic/tiger2020/tl_2020_us_county.shp"
      place_shapefile: "data/interim/geographic/tiger2020/tl_2020_38_place.shp"
      vintage: 2020
```

### Decision 2: GeoJSON as primary geospatial format

**Decision**: GeoJSON is the default and primary geospatial output
format. Shapefile is supported as a secondary format.

**Rationale**:
- GeoJSON is a single file (simpler to distribute), human-readable,
  widely supported by web mapping libraries.
- Shapefile requires companion files (.shx, .dbf, .prj) and has column
  name length limits (10 characters).
- Both formats are supported via the `format_type` parameter.

### Decision 3: Pipeline integration via --formats flag

**Decision**: Add `"geojson"` and `"shapefile"` as valid choices for the
existing `--formats` argument in `03_export_results.py`.

**Rationale**:
- Consistent with existing format-selection patterns (csv, excel,
  parquet).
- No new CLI flags needed.
- Geospatial export is triggered only when explicitly requested.

## Consequences

### Positive
1. Stakeholders receive map-ready projection files without manual
   boundary joining.
2. Pipeline remains backward-compatible; geospatial formats are
   opt-in.
3. Graceful degradation when geopandas is not installed.
4. Reuses existing TIGER files already stored for crosswalk builds.

### Negative
1. geopandas adds a transitive dependency on GDAL/Fiona, which can
   complicate installation on some platforms.
2. Geospatial files are larger than tabular equivalents for the same
   data.

### Risks and Mitigations

**Risk**: TIGER shapefiles may not be present on all machines.
- **Mitigation**: `FileNotFoundError` is caught and logged as a warning;
  the pipeline continues without producing geospatial output.

**Risk**: Large number of key years could produce many output files.
- **Mitigation**: Export respects the `key_years` config list, defaulting
  to the same years used for place projection output.

## Alternatives Considered

### Alternative 1: Web API boundary download at export time

**Description**: Download TIGER boundaries from the Census web API on
each export run instead of using local shapefiles.

**Pros**:
- No local shapefile storage needed.

**Cons**:
- Requires network access at export time.
- Slower and less reproducible.
- The project already stores TIGER files locally.

**Why Rejected**: Local files are faster, reproducible, and already
available.

### Alternative 2: Pre-joined spatial database

**Description**: Store projection results in a spatial database
(GeoPackage or PostGIS) with boundaries pre-joined.

**Pros**:
- Single artifact for both tabular and spatial queries.

**Cons**:
- Adds database dependency.
- Departs from the project's file-based pipeline pattern.

**Why Rejected**: Overkill for the current use case; file-based export
is simpler and sufficient.

## Implementation Notes

### Key Functions/Classes
- `_load_counties_from_tiger(vintage)`: Loads US county shapefile,
  filters to ND, maps column names.
- `_load_places_from_tiger(vintage)`: Loads ND place shapefile, maps
  column names.
- `load_tiger_boundaries(level, vintage)`: Public entry point
  dispatching to county or place loader.
- `write_projection_shapefile(...)`: Joins projection data to
  boundaries and exports GeoJSON or Shapefile.
- `export_geospatial_outputs(...)`: Pipeline-level orchestrator that
  iterates over levels and key years.

### Configuration Integration
- `geography.reference_data.tiger_boundaries` in
  `config/projection_config.yaml` provides shapefile paths and vintage.
- `place_projections.output.key_years` determines which years are
  exported.

### Testing Strategy
- Unit tests: `tests/test_output/test_geospatial_export.py` --
  synthetic DataFrames, mocked file loading, 12+ tests covering
  loading, joining, export, guards, filtering, and validation.
- Integration tests: `tests/test_integration/test_geospatial_pipeline.py`
  -- end-to-end pipeline execution with mock boundaries, dry-run
  verification, graceful error handling.

### Documentation
- [ ] Update `docs/methodology.md` if this ADR changes formulas, rates, data sources, or projection logic

## References

1. **Census TIGER/Line Shapefiles**: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
2. **GeoJSON RFC 7946**: https://tools.ietf.org/html/rfc7946
3. **geopandas documentation**: https://geopandas.org/

## Revision History

- **2026-03-01**: Initial version (ADR-059) -- TIGER geospatial export implementation

## Related ADRs

- ADR-012: Output and export format strategy (extended with geospatial formats)
- ADR-033: City-level projections (place geography)
- ADR-055: Group quarters separation (GQ-aware population totals)
