#!/bin/bash
# Complete Pipeline Runner for North Dakota Population Projections
#
# This script runs the complete end-to-end pipeline from raw data to
# dissemination-ready outputs.
#
# Usage:
#   ./run_complete_pipeline.sh [--dry-run] [--resume]

set -e  # Exit immediately if a command exits with a non-zero status

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
DRY_RUN=""
RESUME=""
FAIL_FAST=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --resume)
      RESUME="--resume"
      shift
      ;;
    --fail-fast)
      FAIL_FAST="--fail-fast"
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --dry-run     Show what would be done without doing it"
      echo "  --resume      Resume projections from previous run"
      echo "  --fail-fast   Stop on first error"
      echo "  --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Display banner
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║     North Dakota Population Projection System - Complete Pipeline         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
if [ -n "$DRY_RUN" ]; then
  echo "Mode: DRY RUN (no files will be modified)"
fi
if [ -n "$RESUME" ]; then
  echo "Mode: RESUME (skip already-completed geographies)"
fi
if [ -n "$FAIL_FAST" ]; then
  echo "Mode: FAIL-FAST (stop on first error)"
fi
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Process Demographic Data
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 1/3: Processing Demographic Data                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/pipeline/01_process_demographic_data.py --all $DRY_RUN $FAIL_FAST

if [ $? -ne 0 ]; then
  echo ""
  echo "ERROR: Data processing failed"
  exit 1
fi

echo ""
echo "✓ Data processing completed successfully"
echo ""

# Step 2: Run Projections
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 2/3: Running Population Projections                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/pipeline/02_run_projections.py --all $DRY_RUN $RESUME

if [ $? -ne 0 ]; then
  echo ""
  echo "ERROR: Projection run failed"
  exit 1
fi

echo ""
echo "✓ Projections completed successfully"
echo ""

# Step 3: Export Results
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP 3/3: Exporting Results                                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

python scripts/pipeline/03_export_results.py --all $DRY_RUN

if [ $? -ne 0 ]; then
  echo ""
  echo "ERROR: Export failed"
  exit 1
fi

echo ""
echo "✓ Export completed successfully"
echo ""

# Final summary
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           Pipeline Complete!                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Output Locations:"
echo "  Processed Data:  data/processed/"
echo "  Projections:     data/projections/"
echo "  Exports:         data/exports/"
echo "  Logs:            logs/"
echo ""

if [ -z "$DRY_RUN" ]; then
  echo "Next steps:"
  echo "  1. Review processing reports in data/processed/reports/"
  echo "  2. Check projection summaries in data/projections/*/metadata/"
  echo "  3. Find distribution packages in data/exports/packages/"
else
  echo "This was a dry run. No files were modified."
  echo "Run without --dry-run to execute the pipeline."
fi

echo ""
exit 0
