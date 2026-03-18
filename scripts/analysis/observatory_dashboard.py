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
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import panel as pn

# Ensure the project root is importable when running as a script
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _configure_panel_runtime() -> None:
    """Enable the Panel extensions and loading indicators used by the dashboard."""
    import panel as pn

    pn.extension("plotly", "tabulator")
    pn.config.loading_indicator = True
    pn.config.loading_spinner = "dots"
    pn.config.loading_color = "#0563C1"


def build_dashboard() -> pn.template.FastListTemplate:
    """Build and return a fresh Observatory dashboard instance.

    Returns
    -------
    pn.template.FastListTemplate
        A new dashboard template backed by a newly created data manager.
    """
    _configure_panel_runtime()

    from cohort_projections.analysis.observatory.dashboard.app import create_app
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

    logging.getLogger(__name__).info("Initialising data manager …")
    dm = DashboardDataManager()
    logging.getLogger(__name__).info("Building dashboard …")
    return create_app(dm)


def main() -> None:
    """Parse arguments and launch the dashboard server."""
    parser = argparse.ArgumentParser(description="Launch the Projection Observatory dashboard.")
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

    open_browser = not args.no_open

    # On WSL, Panel's default show=True opens the WSL-side browser
    # (e.g. Chromium) instead of the Windows host browser.  Detect WSL
    # and handle browser opening ourselves via cmd.exe so it routes to
    # the user's default Windows browser (typically Chrome).
    is_wsl = "microsoft" in platform.uname().release.lower()
    if is_wsl and open_browser:
        import threading

        def _open_windows_browser() -> None:
            url = f"http://localhost:{args.port}/"
            try:
                subprocess.Popen(  # noqa: S603
                    ["cmd.exe", "/c", "start", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                try:
                    subprocess.Popen(  # noqa: S603
                        ["wslview", url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    webbrowser.open(url)

        # Open the Windows browser after a short delay to let the
        # server finish binding the port.
        threading.Timer(1.5, _open_windows_browser).start()

        pn.serve(
            {"/": build_dashboard},
            port=args.port,
            show=False,
            title="Projection Observatory",
            websocket_origin="*",
        )
    else:
        pn.serve(
            {"/": build_dashboard},
            port=args.port,
            show=open_browser,
            title="Projection Observatory",
            websocket_origin="*",
        )


# ------------------------------------------------------------------
# Support ``panel serve observatory_dashboard.py``
# ------------------------------------------------------------------
if __name__.startswith("bokeh"):
    # Running via ``panel serve`` — create the template and make it
    # servable so Panel picks it up.
    _app = build_dashboard()
    _app.servable()

elif __name__ == "__main__":
    main()
