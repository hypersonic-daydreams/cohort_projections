#!/bin/bash
# setup_rclone_bisync.sh - Initial setup for rclone bisync with cohort_projections
#
# This script:
# 1. Verifies rclone is installed and configured
# 2. Creates the remote directory structure in Google Drive
# 3. Creates a test file to verify sync works
# 4. Runs initial --resync to establish baseline
#
# Run this once on each computer before using bisync.sh

set -e

# Configuration
PROJECT_NAME="cohort_projections"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="wsdrive:${PROJECT_NAME}"
FILTER_FILE="${HOME}/.config/rclone/cohort_projections-bisync-filter.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  cohort_projections rclone bisync setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check rclone installation
echo -e "${YELLOW}Step 1: Checking rclone installation...${NC}"
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}ERROR: rclone is not installed${NC}"
    echo ""
    echo "Install rclone:"
    echo "  Ubuntu/Debian: sudo apt install rclone"
    echo "  macOS:         brew install rclone"
    echo "  Manual:        https://rclone.org/install/"
    exit 1
fi
echo -e "${GREEN}✓ rclone is installed: $(rclone version | head -1)${NC}"
echo ""

# Step 2: Check remote configuration
echo -e "${YELLOW}Step 2: Checking remote 'wsdrive:' configuration...${NC}"
if ! rclone listremotes | grep -q "wsdrive:"; then
    echo -e "${RED}ERROR: Remote 'wsdrive:' not configured${NC}"
    echo ""
    echo "Configure Google Drive remote:"
    echo "  rclone config"
    echo ""
    echo "Select 'n' for new remote, name it 'wsdrive', select 'drive' for Google Drive,"
    echo "and follow the OAuth flow."
    exit 1
fi
echo -e "${GREEN}✓ Remote 'wsdrive:' is configured${NC}"
echo ""

# Step 3: Check/create filter file
echo -e "${YELLOW}Step 3: Checking filter file...${NC}"
if [[ ! -f "$FILTER_FILE" ]]; then
    echo "Creating filter file at ${FILTER_FILE}..."
    mkdir -p "$(dirname "$FILTER_FILE")"
    cat > "$FILTER_FILE" << 'EOF'
# rclone bisync filter for cohort_projections
# STRATEGY: Git for Code/Docs, Rclone for Data Files
# See ADR-016: Raw Data Management Strategy

# 1. GLOBAL EXCLUDES (Highest Priority)
- .git/**
- .venv/**
- .mypy_cache/**
- .ruff_cache/**
- __pycache__/**
- .ipynb_checkpoints/**
- **/.pytest_cache/**
- **/node_modules/**
- **/cache/**
- htmlcov/**
- *.pyc
- *.pyo
- *.py
- *.md
- *.txt
- *.yaml
- *.yml
- *.toml
- *.log
- *.tmp
- *.swp
- .DS_Store
- Thumbs.db
- .gitkeep
- .gitignore

# 2. DATA DIRECTORIES (Whitelist - the primary sync targets)
# These are the directories for raw and processed data
+ data/raw/**
+ data/processed/**
+ data/interim/**

# 3. OUTPUT DIRECTORIES (for projection results, at project root only)
# Use anchored paths to avoid matching cohort_projections/output/
+ /output/**
+ /exports/**

# 5. SENTINEL (for testing sync)
+ RCLONE_TEST

# 6. CATCH-ALL EXCLUDE (Critical - prevents code sync)
# Code is synced via git, not rclone
- *
EOF
    echo -e "${GREEN}✓ Filter file created${NC}"
else
    echo -e "${GREEN}✓ Filter file exists at ${FILTER_FILE}${NC}"
fi
echo ""

# Step 4: Create remote directory
echo -e "${YELLOW}Step 4: Creating remote directory structure...${NC}"
if rclone lsd "${REMOTE}" &> /dev/null; then
    echo -e "${GREEN}✓ Remote directory ${REMOTE} already exists${NC}"
else
    echo "Creating ${REMOTE}..."
    rclone mkdir "${REMOTE}"
    echo -e "${GREEN}✓ Created ${REMOTE}${NC}"
fi
echo ""

# Step 5: Create test file for sync verification
echo -e "${YELLOW}Step 5: Creating sync test file...${NC}"
TEST_FILE="${PROJECT_DIR}/RCLONE_TEST"
echo "Bisync test file created $(date -Iseconds)" > "$TEST_FILE"
echo -e "${GREEN}✓ Created ${TEST_FILE}${NC}"
echo ""

# Step 6: Run initial resync
echo -e "${YELLOW}Step 6: Running initial --resync to establish baseline...${NC}"
echo ""
echo -e "${YELLOW}This will sync data files between:${NC}"
echo "  Local:  ${PROJECT_DIR}"
echo "  Remote: ${REMOTE}"
echo ""
read -p "Proceed with initial sync? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Skipped initial sync.${NC}"
    echo "Run manually when ready:"
    echo "  ./scripts/bisync.sh --resync"
else
    echo ""
    rclone bisync \
        "${PROJECT_DIR}" \
        "${REMOTE}" \
        --filter-from "${FILTER_FILE}" \
        --verbose \
        --check-access \
        --resync

    echo ""
    echo -e "${GREEN}✓ Initial sync completed!${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "You can now use bisync with:"
echo "  ./scripts/bisync.sh          # Normal sync"
echo "  ./scripts/bisync.sh --resync # Force resync (after conflicts)"
echo "  ./scripts/bisync.sh --dry-run # Preview changes"
echo ""
echo "Remember to run bisync after:"
echo "  - Adding new data files to data/raw/"
echo "  - Running projections that create output files"
echo "  - Before switching to another computer"
echo ""
