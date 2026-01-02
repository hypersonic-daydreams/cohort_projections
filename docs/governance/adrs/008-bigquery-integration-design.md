# ADR-008: BigQuery Integration Design

## Status
Accepted

## Date
2025-12-18

## Context

While primary demographic data sources (SEER, Census files, IRS data) are typically downloaded as files, Google BigQuery provides access to Census public datasets and offers a platform for storing and querying demographic data. Integration with BigQuery can supplement file-based data sources and provide additional analytical capabilities.

### Potential Use Cases

1. **Census Public Datasets**: Access to `bigquery-public-data.census_bureau_usa.*`
2. **Supplementary Data**: Geographic reference data, crosswalks, historical series
3. **Data Exploration**: Quick queries for data discovery and validation
4. **Intermediate Storage**: Store processed data for collaborative analysis
5. **Validation**: Compare projections to published Census tables

### Challenges

1. **Authentication**: Requires Google Cloud service account credentials
2. **Cost**: BigQuery queries can incur costs (though public datasets are free to query)
3. **Dependency**: Adds Google Cloud dependency to otherwise standalone system
4. **Complexity**: Additional infrastructure to manage
5. **Primary vs. Supplementary**: Need clear guidance on when to use BigQuery vs. direct downloads

### Requirements

1. **Optional Integration**: Should work without BigQuery if not configured
2. **Simple Authentication**: Easy credential management
3. **Query Caching**: Avoid redundant queries
4. **Public Dataset Access**: Leverage Census public datasets
5. **Configuration**: Controlled via configuration file
6. **Graceful Degradation**: Fail gracefully if BigQuery unavailable

## Decision

### Decision 1: BigQuery as Supplementary (Not Primary) Data Source

**Decision**: Use BigQuery for supplementary data and exploration, not as the primary source for projection inputs.

**Primary Data Sources** (file-based):
- **Fertility**: SEER downloads, NVSS files
- **Mortality**: SEER life tables, CDC files
- **Migration**: IRS county flows (files), Census migration files
- **Base Population**: Census API → local files

**BigQuery Use Cases**:
- **Geographic Reference**: FIPS codes, county names, place boundaries
- **Historical Validation**: Compare projections to historical Census tables
- **Data Exploration**: Quick queries during development
- **Supplementary Variables**: Economic data, demographic characteristics
- **Cross-Validation**: Verify processed data against published totals

**Rationale**:

**Why Not Primary**:
- **Availability**: SEER life tables not fully available in BigQuery
- **Versioning**: File-based data easier to version control
- **Offline Development**: Can work without internet/BigQuery access
- **Reproducibility**: Files provide exact data snapshot
- **Simplicity**: Fewer dependencies for core functionality

**Why Include**:
- **Convenience**: Public datasets are pre-loaded and cleaned
- **Exploration**: Fast queries for data discovery
- **Validation**: Easy comparison to published figures
- **Geographic Data**: FIPS codes, geographic hierarchies readily available

### Decision 2: Service Account Authentication

**Decision**: Use Google Cloud service account JSON key files for authentication, not OAuth or user credentials.

**Authentication Flow**:
```python
from google.cloud import bigquery
from google.oauth2 import service_account

# Load credentials from JSON key file
credentials = service_account.Credentials.from_service_account_file(
    '/path/to/service-account-key.json'
)

# Create BigQuery client
client = bigquery.Client(
    credentials=credentials,
    project='your-project-id'
)
```

**Configuration** (in `projection_config.yaml`):
```yaml
bigquery:
  enabled: true
  project_id: "antigravity-sandbox"
  credentials_path: "~/.config/gcloud/cohort-projections-key.json"
  dataset_id: "demographic_data"
  location: "US"
```

**Rationale**:

**Why Service Accounts**:
- **Automation-Friendly**: No interactive login required
- **Portable**: JSON key file can be shared securely
- **Scriptable**: Works in automated pipelines
- **Consistent**: Same credentials across environments

**Why Not OAuth/User Credentials**:
- Requires browser login
- Not suitable for automated scripts
- Credentials expire

**Why Not Application Default Credentials**:
- Assumes gcloud CLI configured
- Less explicit, harder to troubleshoot

**Security Considerations**:
- Store credentials outside repository (not in git)
- Use environment variable or config path
- Document credential creation process
- Limit service account permissions (read-only for public datasets)

### Decision 3: Query Result Caching

**Decision**: Enable BigQuery query cache by default to avoid redundant queries and reduce costs.

**Implementation**:
```python
class BigQueryClient:
    def query(self, sql, use_cache=None):
        if use_cache is None:
            use_cache = self.cache_queries  # From config (default: True)

        job_config = bigquery.QueryJobConfig(use_query_cache=use_cache)
        query_job = self.client.query(sql, job_config=job_config)
        return query_job.to_dataframe()
```

**Configuration**:
```yaml
bigquery:
  cache_queries: true  # Use BigQuery's query cache
```

**Rationale**:
- **Performance**: Cached results return instantly
- **Cost Savings**: Avoid re-processing same queries
- **Development**: Iterative development without repeated costs
- **BigQuery Native**: Uses BigQuery's built-in cache (24 hours)

**When to Disable Cache**:
- Testing with updated data
- Ensuring fresh results for production runs
- Can override per-query

### Decision 4: Public Dataset Usage Over Custom Storage

**Decision**: Primarily use `bigquery-public-data` datasets rather than uploading data to custom datasets.

**Public Datasets Used**:
- `bigquery-public-data.census_bureau_usa.*`: Census tables
- `bigquery-public-data.geo_us_boundaries.*`: Geographic boundaries
- Other public demographic datasets as available

**Custom Dataset Usage** (limited):
- Store processed projection outputs for sharing
- Store intermediate analysis tables
- Not for primary input data

**Example**:
```python
# ✅ PREFERRED: Query public data
sql = """
SELECT *
FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010`
WHERE state_code = '38'  -- North Dakota
"""
df = bq.query(sql)

# ❌ AVOID: Upload then query
# bq.upload_dataframe(census_df, 'my_census_table')  # Unnecessary
```

**Rationale**:
- **Free**: Public dataset queries are free (no storage costs)
- **Up-to-Date**: Google maintains public datasets
- **No Maintenance**: Don't need to update tables ourselves
- **Shared Access**: Anyone with BigQuery can access

**When to Use Custom Dataset**:
- Sharing processed projections with collaborators
- Storing derived tables for analysis
- Testing data processing with small examples

### Decision 5: Wrapper Client Class for Convenience

**Decision**: Provide `BigQueryClient` wrapper class with convenience methods rather than using `google.cloud.bigquery.Client` directly.

**Implementation** (`cohort_projections/utils/bigquery_client.py`):
```python
class BigQueryClient:
    """Wrapper for BigQuery client with demographic data convenience methods."""

    def __init__(self, credentials_path=None, project_id=None, config=None):
        # Load config, authenticate, create client
        ...

    def query(self, sql, to_dataframe=True, use_cache=None):
        """Execute query and return DataFrame."""
        ...

    def list_public_datasets(self, filter_census=True):
        """List available Census public datasets."""
        ...

    def list_tables(self, dataset='bigquery-public-data.census_bureau_usa'):
        """List tables in dataset."""
        ...

    def get_table_schema(self, table_ref):
        """Get schema for table."""
        ...

    def preview_table(self, table_ref, limit=10):
        """Preview table rows."""
        ...

    def upload_dataframe(self, df, table_name, write_disposition='WRITE_TRUNCATE'):
        """Upload DataFrame to BigQuery (for custom datasets)."""
        ...
```

**Rationale**:
- **Convenience**: Common operations simplified
- **Consistency**: Standard API across project
- **Configuration**: Automatic config loading
- **Defaults**: Sensible defaults (caching, location)
- **Discoverability**: Helper methods for data exploration

**Why Not Direct google.cloud.bigquery.Client**:
- More verbose for common operations
- Requires manual configuration
- No project-specific defaults

### Decision 6: Graceful Degradation Without BigQuery

**Decision**: All functionality should work without BigQuery if not configured or unavailable.

**Pattern**:
```python
def fetch_geographic_data(use_bigquery=None):
    """
    Fetch geographic data.

    Falls back to file-based sources if BigQuery unavailable.
    """
    config = load_projection_config()

    if use_bigquery is None:
        use_bigquery = config.get('bigquery', {}).get('enabled', False)

    if use_bigquery:
        try:
            bq = BigQueryClient(config=config)
            return fetch_from_bigquery(bq)
        except Exception as e:
            logger.warning(f"BigQuery unavailable: {e}")
            logger.info("Falling back to file-based source")
            return fetch_from_file()
    else:
        return fetch_from_file()
```

**Rationale**:
- **Optional Dependency**: System works without BigQuery
- **Offline Development**: Can work without internet
- **Robustness**: Doesn't fail if BigQuery down
- **User Choice**: Users can disable BigQuery entirely

**When BigQuery Required**:
- Only when explicitly requested in code
- Never for core projection functionality

### Decision 7: No BigQuery for Storing Primary Projection Outputs

**Decision**: Store projection outputs as Parquet/CSV files, not in BigQuery tables.

**Output Storage**:
```
data/
  output/
    projections/
      nd_state_projection_2025_2045.parquet      # ✅ File-based
      nd_state_projection_2025_2045.csv
      nd_county_projections_2025_2045.parquet
      ...
```

**Not**:
```
BigQuery:
  antigravity-sandbox.projections.nd_state_2025_2045  # ❌ Avoid
```

**Rationale**:

**Why File-Based Outputs**:
- **Portability**: Files can be shared without BigQuery access
- **Version Control**: Can commit summary outputs to git
- **Simplicity**: No BigQuery setup required for users
- **Cost**: No storage costs for files
- **Offline Access**: Results available without internet

**When BigQuery Output Acceptable**:
- Collaborative analysis with team (share via BigQuery)
- Integration with BI tools (Looker, Data Studio)
- Optional upload for users who want it

**Provision**: Can add optional `upload_to_bigquery=True` parameter for users who want to store in BigQuery.

## Consequences

### Positive

1. **Optional**: System works fully without BigQuery
2. **Convenience**: Easy access to Census public datasets
3. **Exploration**: Fast queries for data discovery
4. **Validation**: Compare projections to published Census tables
5. **Simple Auth**: Service account JSON key straightforward
6. **Cached Queries**: No redundant processing/costs
7. **Graceful Degradation**: Falls back to files if unavailable
8. **No Lock-In**: Not dependent on BigQuery

### Negative

1. **Additional Complexity**: Another system to understand
2. **Credential Management**: Need to create and secure service account key
3. **Potential Costs**: Large queries could incur costs (though public data free)
4. **Dependency**: Requires `google-cloud-bigquery` package
5. **Internet Required**: BigQuery requires internet connection
6. **Learning Curve**: Users need to understand BigQuery SQL

### Risks and Mitigations

**Risk**: Users accidentally run expensive queries
- **Mitigation**: Use query cache by default
- **Mitigation**: Document cost considerations
- **Mitigation**: Recommend query limits for exploration

**Risk**: Service account credentials compromised
- **Mitigation**: Store credentials outside repository
- **Mitigation**: Use read-only permissions
- **Mitigation**: Document credential rotation process

**Risk**: BigQuery public datasets change or removed
- **Mitigation**: Don't depend on BigQuery for primary data
- **Mitigation**: Document which public datasets used
- **Mitigation**: Fall back to file-based sources

**Risk**: BigQuery unavailable when needed
- **Mitigation**: Graceful degradation to file-based sources
- **Mitigation**: Cache critical data locally
- **Mitigation**: Don't use for time-critical operations

## Alternatives Considered

### Alternative 1: BigQuery as Primary Data Source

**Description**: Store all input data in BigQuery, query for projections.

**Pros**:
- Centralized data storage
- SQL queries for data prep
- Collaborative access

**Cons**:
- Requires BigQuery for all operations
- Internet dependency
- Cost for storage
- Less portable
- Harder to version control data

**Why Rejected**:
- File-based sources more portable
- Offline development important
- SEER data not available in BigQuery

### Alternative 2: No BigQuery Integration

**Description**: Use only file-based data sources, no BigQuery at all.

**Pros**:
- Simpler architecture
- No cloud dependencies
- No credential management
- Fully offline

**Cons**:
- Miss out on Census public datasets
- No easy exploration of BigQuery data
- Manual download of all reference data

**Why Rejected**:
- Public datasets are valuable resource
- Optional integration provides benefit without requirement
- Minimal added complexity

### Alternative 3: Upload All Data to BigQuery

**Description**: Upload Census, SEER, IRS data to custom BigQuery dataset.

**Pros**:
- All data in one place
- SQL-based processing
- Collaborative access

**Cons**:
- Significant upload time
- Storage costs
- Maintenance burden (keeping updated)
- Internet required
- Duplicates public datasets

**Why Rejected**:
- Unnecessary duplication
- Public datasets already available
- File-based processing adequate

### Alternative 4: OAuth User Authentication

**Description**: Use OAuth flow for user authentication.

**Pros**:
- No JSON key files to manage
- Individual user accounts
- Audit trail per user

**Cons**:
- Interactive login required
- Not scriptable
- Harder automation
- Token expiration

**Why Rejected**:
- Service accounts better for automation
- No need for per-user tracking
- Simpler credential management

### Alternative 5: Mandatory BigQuery for All Operations

**Description**: Require BigQuery setup for system to work.

**Pros**:
- Consistent data access
- Leverages cloud infrastructure

**Cons**:
- Barrier to entry for users
- Internet required
- Cloud account required
- Cost concerns
- Less portable

**Why Rejected**:
- Want low barrier to entry
- Offline use important
- Not everyone has/wants GCP account

## Implementation Notes

### File Location

**BigQuery Client**: `/home/nigel/cohort_projections/cohort_projections/utils/bigquery_client.py`

### Configuration

**In `projection_config.yaml`**:
```yaml
bigquery:
  enabled: true                                              # Enable/disable BigQuery
  project_id: "antigravity-sandbox"                         # GCP project ID
  credentials_path: "~/.config/gcloud/cohort-projections-key.json"  # Service account key
  dataset_id: "demographic_data"                            # Default dataset for custom tables
  location: "US"                                            # BigQuery location
  use_public_data: true                                     # Use public datasets
  cache_queries: true                                       # Cache query results
```

### Usage Examples

**Basic Query**:
```python
from cohort_projections.utils.bigquery_client import get_bigquery_client

bq = get_bigquery_client()

# Query Census public data
sql = """
SELECT geo_id, population
FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010`
WHERE state_code = '38'
LIMIT 10
"""
df = bq.query(sql)
```

**Data Exploration**:
```python
# List available Census datasets
datasets = bq.list_public_datasets(filter_census=True)

# List tables in dataset
tables = bq.list_tables('bigquery-public-data.census_bureau_usa')

# Preview table
preview = bq.preview_table(
    'bigquery-public-data.census_bureau_usa.population_by_zip_2010',
    limit=5
)
```

**Optional Upload**:
```python
# Upload projection results (optional)
projection_df = pd.DataFrame(...)
bq.upload_dataframe(
    projection_df,
    table_name='nd_projection_2025_2045',
    write_disposition='WRITE_TRUNCATE'
)
```

### Dependencies

**Required Python Packages**:
```
google-cloud-bigquery>=3.0.0
google-auth>=2.0.0
pyarrow>=12.0.0  # For DataFrame conversion
```

**In `requirements.txt`**:
```txt
google-cloud-bigquery>=3.0.0
google-auth>=2.0.0
```

### Service Account Setup

**Steps to Create Service Account**:
1. Go to Google Cloud Console → IAM & Admin → Service Accounts
2. Create service account: `cohort-projections`
3. Grant role: `BigQuery Data Viewer` (read-only)
4. Create JSON key → Download
5. Save to `~/.config/gcloud/cohort-projections-key.json`
6. Update `projection_config.yaml` with path

**Security Best Practices**:
- Use read-only permissions
- Don't commit credentials to git
- Rotate keys periodically
- Restrict to specific IP ranges if possible

## References

1. **BigQuery Documentation**: https://cloud.google.com/bigquery/docs
2. **Census Public Datasets**: https://console.cloud.google.com/marketplace/browse?filter=solution-type:dataset&filter=category:demographics
3. **Service Account Authentication**: https://cloud.google.com/docs/authentication/production
4. **BigQuery Python Client**: https://googleapis.dev/python/bigquery/latest/index.html
5. **BigQuery Pricing**: https://cloud.google.com/bigquery/pricing (public datasets free to query)

## Revision History

- **2025-12-18**: Initial version (ADR-008) - BigQuery integration design

## Related ADRs

- ADR-005: Configuration management (BigQuery configuration section)
- ADR-006: Data pipeline architecture (supplementary data source)
- ADR-010: Geographic scope (geographic reference data from BigQuery)
