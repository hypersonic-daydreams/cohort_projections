#!/usr/bin/env python3
"""
Pipeline Verification Script
---------------------------
Safely re-runs the SDC 2024 Replication analysis pipeline to verify
that the current output files match what the code currently generates.

1. Backs up 'output/' to 'output_backup_<timestamp>/'
2. Runs 'run_three_variants.py'
3. Compares new output csvs with backed-up csvs
4. Restores backup (unless --keep-new is specified)
5. Reports differences
"""

import sys
import shutil
import subprocess
import time
from pathlib import Path
import filecmp
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"

def setup_backup():
    """Create a backup of the existing output directory."""
    if not OUTPUT_DIR.exists():
        logger.warning(f"No existing output directory found at {OUTPUT_DIR}")
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = BASE_DIR / f"output_backup_{timestamp}"

    logger.info(f"Backing up {OUTPUT_DIR} to {backup_path}")
    shutil.move(str(OUTPUT_DIR), str(backup_path))
    return backup_path

def restore_backup(backup_path, keep_new=False):
    """Restore the backup directory."""
    if not backup_path or not backup_path.exists():
        return

    if keep_new:
        logger.info(f"Keeping new output. Old output saved at {backup_path}")
        return

    # If we are restoring, we typically want to remove the 'new' output
    # and put the old one back.
    if OUTPUT_DIR.exists():
        logger.info("Removing temporary verification output...")
        shutil.rmtree(str(OUTPUT_DIR))

    logger.info(f"Restoring backup from {backup_path} to {OUTPUT_DIR}")
    shutil.move(str(backup_path), str(OUTPUT_DIR))

def run_pipeline():
    """Run the analysis pipeline."""
    logger.info("Running run_three_variants.py...")
    script_path = SCRIPT_DIR / "run_three_variants.py"

    try:
        # Run in the scripts directory so relative imports work if setup that way,
        # checking the file content previously, it appends parent to sys path or uses relative imports.
        # run_three_variants.py sets up paths relative to __file__, so CWD shouldn't matter too much,
        # but let's run from SCRIPT_DIR to be safe.
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True
        )
        logger.info("Pipeline Execution: SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Pipeline Execution: FAILED")
        logger.error(e.stderr)
        return False

def compare_outputs(backup_path):
    """Compare content of new csvs vs backed up csvs."""
    if not backup_path:
        return

    files_to_compare = [
        "three_variant_comparison.csv",
        "policy_variant_state_totals.csv",
        "original_variant_state_totals.csv",
        "updated_variant_state_totals.csv"
    ]

    logger.info("\n" + "="*40)
    logger.info("VERIFICATION REPORT")
    logger.info("="*40)

    all_match = True

    for filename in files_to_compare:
        old_file = backup_path / filename
        new_file = OUTPUT_DIR / filename

        if not old_file.exists():
            logger.warning(f"[NEW]      {filename} (was not in backup)")
            continue
        if not new_file.exists():
            logger.warning(f"[MISSING]  {filename} (was NOT generated)")
            all_match = False
            continue

        # Use pandas for semantic CSV comparison (ignoring fp precision issues)
        try:
            df_old = pd.read_csv(old_file)
            df_new = pd.read_csv(new_file)

            # Check for exact equality first
            if df_old.equals(df_new):
                logger.info(f"[MATCH]    {filename}")
            else:
                # Try with tolerance
                try:
                    pd.testing.assert_frame_equal(df_old, df_new, check_dtype=False, atol=1e-5)
                    logger.info(f"[MATCH]    {filename} (within tolerance)")
                except AssertionError as e:
                    logger.error(f"[DIFF]     {filename}")
                    logger.error(f"Differences:\n{e}")
                    all_match = False

        except Exception as e:
            logger.error(f"[ERROR]    Could not compare {filename}: {e}")
            all_match = False

    return all_match

def main():
    backup_path = None
    try:
        backup_path = setup_backup()

        success = run_pipeline()

        if success:
            match = compare_outputs(backup_path)
            if match:
                logger.info("\nRESULT: VERIFIED. Code produces identical output to current files.")
            else:
                logger.error("\nRESULT: FAILED. New output differs from existing files.")

    finally:
        # Always restore backup to be safe, unless user explicitly wanted to keep it (not impl here)
        if backup_path:
            restore_backup(backup_path, keep_new=False)

if __name__ == "__main__":
    main()
