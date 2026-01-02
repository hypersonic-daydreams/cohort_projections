# ADR 015: Output Format and Visualization Design

**Status**: Accepted
**Date**: 2025-01-18
**Deciders**: Development Team
**Context**: Enhanced output module for projection results

## Context and Problem Statement

The North Dakota Population Projection System generates detailed projection results that need to be exported, analyzed, and presented to various stakeholders. We need a comprehensive output system that supports:

1. Multiple export formats for different use cases
2. Professional visualizations for presentations and reports
3. Summary statistics and automated report generation
4. Integration with existing tools and workflows

**Key Questions**:
- Which file formats should we support?
- How should we format Excel outputs for maximum usability?
- Which visualization library should we use?
- How should we generate reports (HTML vs PDF)?
- Should we support geospatial outputs?

## Decision Drivers

### Stakeholder Requirements
- **State Demographers**: Need Excel files for exploration and ad-hoc analysis
- **Planning Agencies**: Need formatted reports (HTML/PDF) for publications
- **Data Analysts**: Need efficient formats (Parquet) for data science workflows
- **GIS Specialists**: Want geospatial outputs for mapping (future consideration)

### Technical Requirements
- Must support large datasets (millions of rows)
- Must be performant (export should complete in seconds)
- Must follow existing naming conventions
- Must integrate with current data pipeline
- Should be extensible for future formats

### Usability Requirements
- Excel outputs must be immediately usable (formatted, not raw data)
- Visualizations must be publication-ready
- Reports must be professional and comprehensive
- API must be simple and intuitive

## Considered Options

### File Format Options

#### Option 1: Excel Only
**Pros**:
- Universal tool, all stakeholders familiar
- Interactive exploration
- Can embed charts and formatting

**Cons**:
- Row limit (1,048,576 rows)
- Large files
- Not suitable for programmatic analysis

#### Option 2: CSV Only
**Pros**:
- Universal compatibility
- Simple to generate and parse
- No row limits

**Cons**:
- No formatting
- No metadata storage
- Large uncompressed files
- No embedded visualizations

#### Option 3: Parquet Only
**Pros**:
- Extremely efficient storage
- Fast read/write
- Column-oriented (good for analytics)
- Built-in compression

**Cons**:
- Not human-readable
- Requires special tools
- Less familiar to some stakeholders

#### Option 4: Multi-Format Support (CHOSEN)
**Pros**:
- Meets all stakeholder needs
- Flexibility for different use cases
- Can choose best format for each task

**Cons**:
- More complex implementation
- Multiple files to manage

**Decision**: Support Excel, CSV, Parquet, and JSON with a unified API.

### Excel Formatting Strategy

#### Option 1: Raw Data Export
Simple `.to_excel()` with no formatting.

**Rejected**: Not user-friendly, requires manual formatting.

#### Option 2: Basic Formatting
Headers and number formatting only.

**Rejected**: Doesn't leverage Excel's full capabilities.

#### Option 3: Rich Formatting with Multiple Sheets (CHOSEN)
- Multiple worksheets (Summary, By Age, By Sex, By Race, Detail, Metadata)
- Professional formatting (colors, borders, fonts)
- Number formatting (thousands separators)
- Auto-width columns and freeze panes
- Embedded charts (population trends)

**Pros**:
- Immediately usable by stakeholders
- Professional appearance
- Multiple views of same data
- Self-documenting with metadata sheet

**Cons**:
- Slightly slower export
- Larger file size
- Requires openpyxl library

### Visualization Library

#### Option 1: Matplotlib (CHOSEN)
**Pros**:
- Industry standard, widely used
- Excellent documentation
- Highly customizable
- Publication-quality output
- Multiple output formats (PNG, SVG, PDF)
- Works well without display (Agg backend)

**Cons**:
- API can be verbose
- Default styling dated (addressed with seaborn)

#### Option 2: Plotly
**Pros**:
- Interactive charts
- Modern, attractive defaults
- Easy to embed in HTML

**Cons**:
- Larger dependency
- Interactive features not needed for static reports
- Heavier files
- More complex to export static images

#### Option 3: Seaborn
**Pros**:
- Beautiful defaults
- High-level API

**Cons**:
- Built on matplotlib (not a replacement)
- Limited chart types

**Decision**: Use matplotlib as primary library, with optional seaborn for styling.

### Report Format

#### Option 1: PDF Reports (via ReportLab or LaTeX)
**Pros**:
- Professional print-ready format
- Fixed layout
- Universal viewing

**Cons**:
- Complex to generate programmatically
- Large dependencies
- Hard to update/modify
- Not web-friendly

#### Option 2: HTML Reports (CHOSEN)
**Pros**:
- Easy to generate programmatically
- Viewable in any browser
- Can embed visualizations
- Easy to style with CSS
- Can convert to PDF if needed (browser print)
- Lightweight

**Cons**:
- Layout may vary by browser
- Requires browser to view

**Decision**: Generate HTML reports as primary format. Users can print to PDF if needed.

#### Option 3: Markdown Reports (SUPPLEMENTARY)
Also support plain text/Markdown for:
- Console output
- Email summaries
- Documentation
- README files

### Geospatial Output

#### Option 1: Full Integration with TIGER Boundaries
Automatically join projection data with Census TIGER shapefiles.

**Rejected**: Too complex, requires downloading/managing large boundary files.

#### Option 2: Placeholder with Manual Integration (CHOSEN)
Provide API function but leave implementation to user.

**Pros**:
- Keeps core system lightweight
- Users who need GIS can implement
- Avoids large data dependencies

**Cons**:
- Not immediately usable for GIS

**Decision**: Include `write_projection_shapefile()` as placeholder. Users requiring geospatial output can implement using geopandas and their own TIGER files.

## Decision Outcome

### Chosen Options

**1. Multi-Format Export Strategy**
- Excel (`.xlsx`) with rich formatting for stakeholder exploration
- CSV (`.csv`, `.csv.gz`) for data sharing and legacy systems
- Parquet (`.parquet`) for efficient storage and data science
- JSON (`.json`) for web applications and metadata

**2. Excel Design**
- Multiple worksheets for different views
- Professional formatting (colors, fonts, borders)
- Embedded charts in Summary sheet
- Metadata worksheet for documentation
- openpyxl library for formatting support

**3. Visualization Approach**
- Matplotlib as primary library
- Optional seaborn for enhanced styling
- Static charts (PNG, SVG, PDF)
- Publication-ready quality (300 DPI default)
- Consistent styling across all charts

**4. Report Generation**
- HTML as primary format (with embedded CSS)
- Plain text/Markdown as supplementary format
- Programmatically generated (no templates initially)
- Professional styling with responsive design

**5. Geospatial Support**
- Placeholder function for future development
- Requires geopandas (optional dependency)
- Users provide own TIGER boundary data

### Chart Types Implemented

1. **Population Pyramid**: Side-by-side male/female bars, optional race stacking
2. **Population Trends**: Line charts by total, sex, age group, or race
3. **Growth Rates**: Bar chart showing annual or period growth
4. **Scenario Comparison**: Multi-line chart comparing scenarios
5. **Component Analysis**: Placeholder for births/deaths/migration (requires component tracking)

### API Design Principles

1. **Simplicity**: Common tasks should be one function call
2. **Flexibility**: Support customization without overwhelming defaults
3. **Consistency**: Similar parameters across functions
4. **Documentation**: Comprehensive docstrings with examples
5. **Error Handling**: Clear error messages, graceful degradation

## Consequences

### Positive

- **Comprehensive**: Meets needs of all stakeholder groups
- **Flexible**: Users can choose appropriate formats for their use case
- **Professional**: Excel and reports are immediately presentable
- **Efficient**: Parquet for large datasets, gzip compression for CSV
- **Extensible**: Easy to add new formats or chart types
- **Well-Documented**: README and docstrings provide clear guidance

### Negative

- **Dependencies**: Requires openpyxl and matplotlib (but both are optional)
- **Complexity**: More code to maintain across three modules
- **File Proliferation**: Multiple formats means more files
- **Excel Limitations**: Still limited to ~1M rows for detail sheet

### Mitigations

- Make heavy dependencies (openpyxl, matplotlib) optional with graceful fallback
- Provide `write_projection_formats()` to generate all formats consistently
- Clear documentation on which format to use when
- Warn users when Excel detail sheet is truncated

## Implementation Notes

### Module Structure
```
cohort_projections/output/
├── __init__.py          # Public API
├── writers.py           # Export functions (~400 lines)
├── reports.py           # Report generation (~450 lines)
├── visualizations.py    # Charts and graphs (~450 lines)
├── templates/           # Future Jinja2 templates
└── README.md           # Comprehensive documentation
```

### Configuration Integration
Added to `config/projection_config.yaml`:
```yaml
output:
  excel:
    include_charts: true
    include_metadata: true
    format_numbers: true

  reports:
    generate_html: true
    generate_text: true
    include_methodology: true

  visualizations:
    format: "png"
    dpi: 300
    style: "seaborn-v0_8-darkgrid"
    color_palette: "Set2"
    figure_size: [10, 6]
```

### File Naming Convention
- Projections: `{geography}_{level}_{fips}_projection_{start}_{end}_{scenario}.{ext}`
- Charts: `{base_filename}_{chart_type}_{year}.{ext}`
- Reports: `{geography}_{level}_report_{date}.{ext}`

### Quality Standards
- Type hints throughout
- Comprehensive docstrings (Google style)
- Error handling and validation
- Logging integration
- Examples in docstrings
- Unit tests (future)

## Alternatives Considered But Rejected

### Template-Based HTML Reports
Using Jinja2 templates for HTML generation.

**Rejected**: Adds dependency, increases complexity. Programmatic generation is sufficient for current needs. Can add templates later if needed.

### Interactive Visualizations with Plotly
Generate interactive HTML charts.

**Rejected**: Stakeholders primarily need static charts for publications. Interactive features would go unused. Matplotlib is simpler and more lightweight.

### Automatic PDF Generation
Generate PDF reports instead of HTML.

**Rejected**: Requires heavy dependencies (ReportLab/wkhtmltopdf). HTML can be printed to PDF by user if needed.

### Database Export
Export directly to PostgreSQL/SQLite.

**Rejected**: Out of scope for initial implementation. Users can import CSV/Parquet to database if needed.

## Related ADRs

- ADR 001: Project Structure and Module Organization
- ADR 002: Configuration Management Strategy
- ADR 003: Data Pipeline Architecture

## References

- [openpyxl Documentation](https://openpyxl.readthedocs.io/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Pandas to_excel Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html)
- [Parquet Format Specification](https://parquet.apache.org/)
- [Census TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

## Notes

This ADR documents the design decisions for the enhanced output module. The implementation follows these decisions to provide comprehensive output capabilities while maintaining flexibility and ease of use.

Future enhancements could include:
- Jinja2 templates for customizable HTML reports
- Interactive dashboards with Plotly/Dash
- Automated PDF generation
- Database export functionality
- Component tracking for component analysis charts
- Full geospatial integration with TIGER boundaries
