#!/bin/bash
# Pre-commit hook: warn when core methodology files change without
# a corresponding update to docs/methodology.md.
#
# This is a soft warning — it always exits 0 so it never blocks commits.
# It simply reminds the committer to check whether methodology.md needs
# an update.

CORE_DIRS="cohort_projections/core/ cohort_projections/data/process/ cohort_projections/data/load/"

# Get staged files
STAGED=$(git diff --cached --name-only)

# Check if any core methodology files are staged
CORE_CHANGED=false
for dir in $CORE_DIRS; do
    if echo "$STAGED" | grep -q "^${dir}"; then
        CORE_CHANGED=true
        break
    fi
done

# Check if methodology.md is also staged
METHODOLOGY_CHANGED=false
if echo "$STAGED" | grep -q "^docs/methodology.md"; then
    METHODOLOGY_CHANGED=true
fi

# Warn if core changed but methodology.md didn't
if [ "$CORE_CHANGED" = true ] && [ "$METHODOLOGY_CHANGED" = false ]; then
    echo "=================================================================="
    echo "NOTE: Core methodology files changed without docs/methodology.md"
    echo "------------------------------------------------------------------"
    echo "If this commit changes formulas, rates, data sources, or"
    echo "projection logic, consider updating docs/methodology.md."
    echo "=================================================================="
fi

exit 0
