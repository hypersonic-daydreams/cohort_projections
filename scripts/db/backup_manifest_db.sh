#!/bin/bash
# backup_manifest_db.sh - Backup cohort_projections_meta database to SQL file
#
# This script creates a pg_dump of the manifest database and saves it to
# data/metadata/manifest_backup.sql. This file is then synced via bisync
# to ensure the manifest is portable across machines.
#
# Usage:
#   ./scripts/db/backup_manifest_db.sh           # Create backup
#   ./scripts/db/backup_manifest_db.sh --restore # Restore from backup
#
# The backup includes:
#   - Schema (tables, types, views)
#   - All data
#   - No ownership/privilege statements (portable)

set -e

# Configuration
DB_NAME="cohort_projections_meta"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="${PROJECT_DIR}/data/metadata"
BACKUP_FILE="${BACKUP_DIR}/manifest_backup.sql"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Ensure backup directory exists
mkdir -p "${BACKUP_DIR}"

# Check for restore flag
if [[ "$1" == "--restore" ]]; then
    echo -e "${YELLOW}Restoring database from backup...${NC}"

    if [[ ! -f "${BACKUP_FILE}" ]]; then
        echo -e "${RED}ERROR: Backup file not found at ${BACKUP_FILE}${NC}"
        exit 1
    fi

    # Check if database exists
    if psql -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME}"; then
        echo "Database ${DB_NAME} exists. Dropping and recreating..."
        dropdb "${DB_NAME}"
    fi

    echo "Creating database ${DB_NAME}..."
    createdb "${DB_NAME}"

    echo "Restoring from ${BACKUP_FILE}..."
    psql -d "${DB_NAME}" -f "${BACKUP_FILE}" -q

    echo -e "${GREEN}Restore complete!${NC}"
    exit 0
fi

# Create backup
echo -e "${GREEN}Backing up ${DB_NAME} to ${BACKUP_FILE}${NC}"

# Check if database exists
if ! psql -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME}"; then
    echo -e "${YELLOW}WARNING: Database ${DB_NAME} does not exist.${NC}"
    echo "Run the migration script first:"
    echo "  python scripts/db/migrate_from_markdown.py"
    exit 1
fi

# Create the backup
# --no-owner: Don't include ownership (portable between users)
# --no-privileges: Don't include GRANT/REVOKE
# --clean: Include DROP statements for clean restore
# --if-exists: Don't error if objects don't exist on restore
pg_dump \
    --dbname="${DB_NAME}" \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    --file="${BACKUP_FILE}"

# Add header comment
HEADER="-- Backup of ${DB_NAME}
-- Generated: $(date -Iseconds)
-- To restore: ./scripts/db/backup_manifest_db.sh --restore
--
"
echo -e "${HEADER}\n$(cat ${BACKUP_FILE})" > "${BACKUP_FILE}"

# Show file size
SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
echo -e "${GREEN}Backup complete: ${BACKUP_FILE} (${SIZE})${NC}"

# Count records
RECORD_COUNT=$(psql -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM data_sources" 2>/dev/null || echo "?")
echo "  Data sources: ${RECORD_COUNT}"
