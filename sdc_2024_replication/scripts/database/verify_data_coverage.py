import logging
import sys
from pathlib import Path

# Add parent directory to path to import db_utils
sys.path.append(str(Path(__file__).parent))
from db_utils import get_db_cursor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_table_coverage(cursor, table_name, state_column="state_name"):
    logger.info(f"Checking {table_name}...")
    try:
        cursor.execute(f"SELECT COUNT(DISTINCT {state_column}) FROM {table_name}")
        count = cursor.fetchone()["count"]

        cursor.execute(f"SELECT DISTINCT {state_column} FROM {table_name} LIMIT 5")
        examples = [str(row[state_column]) for row in cursor.fetchall()]

        logger.info(f"  Unique States: {count}")
        logger.info(f"  Examples: {examples}")

        if count <= 1:
            logger.warning(
                f"  Warning: Low state count ({count}) in {table_name}. Likely filtered."
            )
        else:
            logger.info(f"  Success: Found {count} states.")

    except Exception as e:
        logger.error(f"  Error checking {table_name}: {e}")


def main():
    logger.info("Starting Data Coverage Verification...")

    with get_db_cursor() as cursor:
        # 1. Census Population
        check_table_coverage(cursor, "census.population_estimates", "state_name")

        # 2. Census Components
        check_table_coverage(cursor, "census.state_components", "state_name")

        # 3. DHS LPR
        check_table_coverage(cursor, "dhs.lpr_arrivals", "state_name")

        # 4. ACS Foreign Born
        check_table_coverage(cursor, "acs.foreign_born", "state_name")

        # 5. RPC Refugees
        check_table_coverage(cursor, "rpc.refugee_arrivals", "destination_state")


if __name__ == "__main__":
    main()
