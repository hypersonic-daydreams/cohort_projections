# BigQuery Setup Guide

This guide walks you through setting up Google BigQuery credentials for accessing Census and demographic data.

## Prerequisites

- Google Cloud Platform (GCP) account
- Existing GCP project (e.g., `antigravity-sandbox`)
- Billing enabled on the project (required for BigQuery, but querying public datasets is free)

## Step 1: Create Service Account

1. **Go to GCP Console**
   - Navigate to: https://console.cloud.google.com/
   - Select your project: `antigravity-sandbox`

2. **Create Service Account**
   - Go to: IAM & Admin → Service Accounts
   - Click **"+ CREATE SERVICE ACCOUNT"**
   - Fill in details:
     - **Service account name**: `cohort-projections`
     - **Service account ID**: `cohort-projections` (auto-filled)
     - **Description**: `Service account for North Dakota cohort projection analysis`
   - Click **"CREATE AND CONTINUE"**

3. **Grant Permissions**
   - Select role: **BigQuery User**
   - (Optional) Also add: **BigQuery Data Viewer** for read access to your datasets
   - Click **"CONTINUE"**

4. **Skip Grant Users Access** (optional)
   - Click **"DONE"**

## Step 2: Create Service Account Key

1. **Find Your Service Account**
   - In Service Accounts list, find `cohort-projections@antigravity-sandbox.iam.gserviceaccount.com`

2. **Create Key**
   - Click on the service account
   - Go to **"KEYS"** tab
   - Click **"ADD KEY"** → **"Create new key"**
   - Select **JSON** format
   - Click **"CREATE"**
   - Key file will download automatically (e.g., `antigravity-sandbox-xxxxx.json`)

## Step 3: Save Key File

1. **Create Directory**
   ```bash
   mkdir -p ~/.config/gcloud
   ```

2. **Move and Rename Key**
   ```bash
   # Move downloaded key to config directory
   mv ~/Downloads/antigravity-sandbox-*.json ~/.config/gcloud/cohort-projections-key.json
   ```

3. **Set Permissions** (security best practice)
   ```bash
   chmod 600 ~/.config/gcloud/cohort-projections-key.json
   ```

## Step 4: Verify Configuration

The project is already configured to use:
- **Credentials path**: `~/.config/gcloud/cohort-projections-key.json`
- **Project ID**: `antigravity-sandbox`
- **Dataset location**: `US`

This is set in [config/projection_config.yaml](../config/projection_config.yaml):

```yaml
bigquery:
  enabled: true
  project_id: "antigravity-sandbox"
  credentials_path: "~/.config/gcloud/cohort-projections-key.json"
  dataset_id: "demographic_data"
  location: "US"
  use_public_data: true
```

## Step 5: Install Dependencies

```bash
# Activate your environment
micromamba activate cohort_proj

# Install BigQuery dependencies
pip install -r requirements.txt
```

This will install:
- `google-cloud-bigquery>=3.11.0`
- `google-auth>=2.22.0`
- `db-dtypes>=1.1.1`

## Step 6: Test Connection

Run the test script to verify everything works:

```bash
python scripts/setup/02_test_bigquery_connection.py
```

**Expected output:**
```
Testing BigQuery Connection
================================================================================
✓ BigQuery client initialized successfully
  Project ID: antigravity-sandbox
  Dataset ID: demographic_data
  Location: US

Exploring Census & Demographic Datasets
================================================================================
Found 12 Census-related datasets:
  - census_bureau_acs
  - census_bureau_construction
  - census_bureau_international
  - census_bureau_tiger
  - census_bureau_usa
  ...
```

## What BigQuery Public Datasets Provide

### Available Census Data

**`bigquery-public-data.census_bureau_usa`**
- Population estimates by geography (state, county, ZIP)
- Demographic breakdowns by age, sex, race
- Historical census data (2010, 2020)

**`bigquery-public-data.census_bureau_acs`**
- American Community Survey 5-year estimates
- Detailed demographic data by ZIP code, place, county
- Migration and mobility data
- Income, education, housing characteristics

**`bigquery-public-data.census_bureau_tiger`**
- Geographic boundary files (TIGER/Line)
- FIPS codes and geographic relationships

### What We Still Need

BigQuery public datasets provide **base population** data, but we still need:

1. **Fertility Rates**: From SEER or NVSS (downloadable tables)
2. **Life Tables/Survival Rates**: From SEER or CDC (downloadable)
3. **Migration Flows**: From IRS county-to-county data (downloadable)

These will need to be downloaded separately and processed.

## Troubleshooting

### Issue: "Credentials file not found"

**Solution:**
```bash
# Check if file exists
ls -la ~/.config/gcloud/cohort-projections-key.json

# If not, verify path in config file matches where you saved it
cat config/projection_config.yaml | grep credentials_path
```

### Issue: "Permission denied"

**Solution:**
```bash
# Fix file permissions
chmod 600 ~/.config/gcloud/cohort-projections-key.json
```

### Issue: "Access Denied" when querying

**Solution:**
- Verify service account has **BigQuery User** role
- For public datasets, no additional permissions needed
- For your own datasets, add **BigQuery Data Viewer** role

### Issue: "Billing not enabled"

**Solution:**
- Go to: https://console.cloud.google.com/billing
- Link a billing account to `antigravity-sandbox`
- Note: Querying public datasets is free, but billing must still be enabled

## Security Best Practices

1. **Never commit credentials to git**
   ```bash
   # Verify .gitignore includes:
   *.json
   .config/
   ```

2. **Restrict file permissions**
   ```bash
   chmod 600 ~/.config/gcloud/*.json
   ```

3. **Rotate keys periodically**
   - Create new key every 90 days
   - Delete old keys in GCP Console

4. **Use minimal permissions**
   - Only grant **BigQuery User** role
   - Don't use Owner or Editor roles

## Next Steps

Once BigQuery is set up and tested:

1. **Explore available data**
   ```bash
   python scripts/setup/02_test_bigquery_connection.py
   ```

2. **Review output** to identify useful tables for:
   - Base population by demographics
   - Geographic reference data
   - Any available demographic rates

3. **Begin data pipeline implementation**
   - Start with fertility rates processor
   - Then survival rates
   - Then migration rates

## Additional Resources

- [BigQuery Public Datasets](https://cloud.google.com/bigquery/public-data)
- [Census Bureau in BigQuery](https://console.cloud.google.com/marketplace/product/united-states-census-bureau/us-census-data)
- [BigQuery Python Client Documentation](https://cloud.google.com/python/docs/reference/bigquery/latest)
- [Service Account Key Best Practices](https://cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys)
