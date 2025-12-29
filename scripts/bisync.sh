#!/bin/bash
# bisync.sh - Wrapper script for rclone bisync with cohort_projections
#
# USAGE:
#   ./scripts/bisync.sh              # Normal sync
#   ./scripts/bisync.sh --resync     # Force resync (after conflicts or first run)
#   ./scripts/bisync.sh --dry-run    # Preview without changes
#
# This script syncs data files between local and Google Drive while
# excluding code (which is synced via git). See ADR-016 for details.

set -e

# Configuration
PROJECT_NAME="cohort_projections"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="wsdrive:${PROJECT_NAME}"
FILTER_FILE="${HOME}/.config/rclone/cohort_projections-bisync-filter.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if filter file exists
if [[ ! -f "$FILTER_FILE" ]]; then
    echo -e "${RED}ERROR: Filter file not found at ${FILTER_FILE}${NC}"
    echo ""
    echo "Run the setup script first:"
    echo "  ./scripts/setup_rclone_bisync.sh"
    exit 1
fi

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}ERROR: rclone is not installed${NC}"
    echo "Install with: sudo apt install rclone"
    exit 1
fi

# Check if remote is configured
if ! rclone listremotes | grep -q "wsdrive:"; then
    echo -e "${RED}ERROR: rclone remote 'wsdrive:' not configured${NC}"
    echo "Run: rclone config"
    exit 1
fi

# Parse arguments
RESYNC=""
DRY_RUN=""
for arg in "$@"; do
    case $arg in
        --resync)
            RESYNC="--resync"
            echo -e "${YELLOW}WARNING: Running with --resync flag${NC}"
            echo "This will force synchronization and may overwrite files."
            read -p "Continue? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Aborted."
                exit 0
            fi
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resync    Force resync (use after conflicts or first run)"
            echo "  --dry-run   Preview changes without syncing"
            echo "  --help      Show this help message"
            echo ""
            echo "This script uses rclone bisync to sync data files between:"
            echo "  Local:  ${PROJECT_DIR}"
            echo "  Remote: ${REMOTE}"
            echo ""
            echo "Filter file: ${FILTER_FILE}"
            exit 0
            ;;
    esac
done

echo -e "${GREEN}Starting bisync for ${PROJECT_NAME}${NC}"
echo "Local:  ${PROJECT_DIR}"
echo "Remote: ${REMOTE}"
echo "Filter: ${FILTER_FILE}"
echo ""

# Run bisync
# --bind 0.0.0.0 forces IPv4 (required for WSL2/networks without IPv6 connectivity)
rclone bisync \
    "${PROJECT_DIR}" \
    "${REMOTE}" \
    --filter-from "${FILTER_FILE}" \
    --verbose \
    --check-access \
    --bind 0.0.0.0 \
    ${RESYNC} \
    ${DRY_RUN}

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo -e "${GREEN}Bisync completed successfully!${NC}"
else
    echo ""
    echo -e "${YELLOW}Dry run completed. No changes were made.${NC}"
fi
