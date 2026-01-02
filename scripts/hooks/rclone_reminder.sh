#!/bin/bash
# Pre-commit hook to remind user about rclone bisync
# Always passes, just prints information.

echo "=================================================================="
echo "REMINDER: Data Synchronization Protocol"
echo "------------------------------------------------------------------"
echo "1. Code is synced via Git."
echo "2. Data is synced via Rclone."
echo "3. ALWAYS use: ./scripts/bisync.sh"
echo "   (NEVER run raw rclone commands)"
echo "=================================================================="

exit 0
