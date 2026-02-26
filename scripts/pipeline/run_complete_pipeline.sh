#!/bin/bash
# Complete Pipeline Runner for North Dakota Population Projections
#
# This script runs the complete end-to-end pipeline from raw data to
# dissemination-ready outputs.
#
# Usage:
#   ./run_complete_pipeline.sh [--dry-run] [--resume] [--fail-fast]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DRY_RUN=false
RESUME=false
FAIL_FAST=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --fail-fast)
      FAIL_FAST=true
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

print_step_banner() {
  local step="$1"
  local total="$2"
  local title="$3"
  echo "============================================================================="
  echo "STEP ${step}/${total}: ${title}"
  echo "============================================================================="
  echo ""
}

run_step() {
  local step="$1"
  local total="$2"
  local title="$3"
  shift 3

  print_step_banner "$step" "$total" "$title"

  if ! "$@"; then
    echo ""
    echo "ERROR: ${title} failed"
    exit 1
  fi

  echo ""
  echo "OK: ${title} completed successfully"
  echo ""
}

run_step_no_dryrun_support() {
  local step="$1"
  local total="$2"
  local title="$3"
  shift 3

  print_step_banner "$step" "$total" "$title"

  if $DRY_RUN; then
    echo "[DRY RUN] Skipping ${title} (script does not implement --dry-run)."
    echo ""
    return
  fi

  if ! "$@"; then
    echo ""
    echo "ERROR: ${title} failed"
    exit 1
  fi

  echo ""
  echo "OK: ${title} completed successfully"
  echo ""
}

DRY_RUN_ARGS=()
RESUME_ARGS=()
FAIL_FAST_ARGS=()

if $DRY_RUN; then
  DRY_RUN_ARGS+=("--dry-run")
fi
if $RESUME; then
  RESUME_ARGS+=("--resume")
fi
if $FAIL_FAST; then
  FAIL_FAST_ARGS+=("--fail-fast")
fi

echo "============================================================================="
echo "North Dakota Population Projection System - Complete Pipeline"
echo "============================================================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
if $DRY_RUN; then
  echo "Mode: DRY RUN (no files should be modified)"
fi
if $RESUME; then
  echo "Mode: RESUME (skip already-completed geographies)"
fi
if $FAIL_FAST; then
  echo "Mode: FAIL-FAST (stop on first error)"
fi
echo ""

cd "$PROJECT_ROOT"

run_step 1 7 "Preparing Processed Inputs" \
  python scripts/pipeline/00_prepare_processed_data.py "${DRY_RUN_ARGS[@]}"

run_step 2 7 "Processing Demographic Data" \
  python scripts/pipeline/01_process_demographic_data.py --all "${DRY_RUN_ARGS[@]}" "${FAIL_FAST_ARGS[@]}"

run_step_no_dryrun_support 3 7 "Computing Residual Migration Rates" \
  python scripts/pipeline/01a_compute_residual_migration.py

run_step_no_dryrun_support 4 7 "Computing Convergence Interpolation Rates" \
  python scripts/pipeline/01b_compute_convergence.py --all-variants

run_step_no_dryrun_support 5 7 "Computing Mortality Improvement Rates" \
  python scripts/pipeline/01c_compute_mortality_improvement.py

run_step 6 7 "Running Population Projections" \
  python scripts/pipeline/02_run_projections.py --all "${DRY_RUN_ARGS[@]}" "${RESUME_ARGS[@]}"

run_step 7 7 "Exporting Results" \
  python scripts/pipeline/03_export_results.py --all "${DRY_RUN_ARGS[@]}"

echo "============================================================================="
echo "Pipeline Complete"
echo "============================================================================="
echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Output Locations:"
echo "  Processed Data:  data/processed/"
echo "  Projections:     data/projections/"
echo "  Exports:         data/exports/"
echo "  Logs:            logs/"
echo ""

if $DRY_RUN; then
  echo "Dry run completed."
  echo "Note: steps 3-5 were skipped because those scripts do not yet support --dry-run."
else
  echo "Next steps:"
  echo "  1. Review processing reports in data/processed/reports/"
  echo "  2. Check projection summaries in data/projections/*/metadata/"
  echo "  3. Find distribution packages in data/exports/packages/"
fi

echo ""
exit 0
