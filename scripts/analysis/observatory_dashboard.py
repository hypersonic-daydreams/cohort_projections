#!/usr/bin/env python
"""Launch the Projection Observatory interactive dashboard.

Usage
-----
    python scripts/analysis/observatory_dashboard.py          # default port 5006
    python scripts/analysis/observatory_dashboard.py --port 5007
    python scripts/analysis/observatory_dashboard.py --no-open  # don't open browser

Alternatively, use Panel's CLI directly::

    panel serve scripts/analysis/observatory_dashboard.py --show
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is importable when running as a script
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> None:
    """Parse arguments and launch the dashboard server."""
    parser = argparse.ArgumentParser(
        description="Launch the Projection Observatory dashboard."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to serve the dashboard on (default: 5006)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't automatically open a browser tab",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import panel as pn

    pn.extension("plotly", "tabulator")
    pn.config.loading_indicator = True
    pn.config.loading_spinner = "dots"
    pn.config.loading_color = "#0563C1"

    from cohort_projections.analysis.observatory.dashboard.app import create_app
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

    logging.getLogger(__name__).info("Initialising data manager …")
    dm = DashboardDataManager()
    logging.getLogger(__name__).info("Building dashboard …")
    app = create_app(dm)

    pn.serve(
        {"/": app},
        port=args.port,
        show=not args.no_open,
        title="Projection Observatory",
        websocket_origin="*",
    )


# ------------------------------------------------------------------
# Support ``panel serve observatory_dashboard.py``
# ------------------------------------------------------------------
if __name__.startswith("bokeh"):
    # Running via ``panel serve`` — create the template and make it
    # servable so Panel picks it up.
    import panel as pn  # noqa: F811

    pn.extension("plotly", "tabulator")
    pn.config.loading_indicator = True
    pn.config.loading_spinner = "dots"
    pn.config.loading_color = "#0563C1"

    from cohort_projections.analysis.observatory.dashboard.app import create_app
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

    _dm = DashboardDataManager()
    _app = create_app(_dm)
    _app.servable()

elif __name__ == "__main__":
    main()
