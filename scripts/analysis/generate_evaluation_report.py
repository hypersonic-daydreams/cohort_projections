#!/usr/bin/env python3
"""
Generate an HTML evaluation report from walk-forward validation results.

Created: 2026-03-11
Author: Claude Code

Purpose
-------
Produces a self-contained HTML report summarising projection evaluation
results.  The report embeds all CSS, JavaScript, and chart images inline
so it can be distributed as a single file.

Method
------
1. Load evaluation results from a saved results directory (CSV artefacts
   from a prior EvaluationRunner run) or run a fresh evaluation from
   projection results data.
2. Optionally load evaluation config from evaluation_config.yaml.
3. Call the html_report.generate_html_report() function to produce the
   HTML file.

Key design decisions
--------------------
- **Single-file output**: All assets are inlined (base64 PNGs, CSS, JS)
  so the report can be emailed or archived without auxiliary files.
- **No Jinja2 dependency**: Uses Python f-strings and string templates
  to keep the dependency surface minimal.
- **Two input modes**: Can load pre-computed CSVs or run evaluation
  on-the-fly from a results parquet file.

Inputs
------
Either:
  --results-dir: Directory containing accuracy_diagnostics.csv,
    realism_diagnostics.csv, comparison_diagnostics.csv, and
    scorecard_summary.txt from a prior EvaluationRunner.generate_report()
Or:
  --projection-results: Path to a walk-forward results parquet file
    (will run evaluation on-the-fly)

Output
------
- HTML report file at the path specified by --output

Usage
-----
    python scripts/analysis/generate_evaluation_report.py \\
        --results-dir data/analysis/evaluation/my-run \\
        --output reports/evaluation_report.html

    python scripts/analysis/generate_evaluation_report.py \\
        --projection-results data/analysis/walk_forward/results.parquet \\
        --output reports/evaluation_report.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cohort_projections.analysis.evaluation.data_structures import ScorecardEntry
from cohort_projections.analysis.evaluation.html_report import generate_html_report

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "evaluation_config.yaml"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "reports" / "evaluation_report.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an HTML evaluation report from evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing saved evaluation CSVs from EvaluationRunner.",
    )
    input_group.add_argument(
        "--projection-results",
        type=Path,
        help="Path to walk-forward results parquet file (runs evaluation on-the-fly).",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output HTML file path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to evaluation_config.yaml",
    )
    parser.add_argument(
        "--title",
        default="Population Projection Evaluation Report",
        help="Report title",
    )
    parser.add_argument(
        "--no-appendix",
        action="store_true",
        help="Omit the full diagnostics appendix",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args(argv)


def _load_from_results_dir(results_dir: Path) -> dict:
    """Load evaluation artefacts from a saved results directory.

    Args:
        results_dir: Directory containing CSV and text outputs from
            ``EvaluationRunner.generate_report()``.

    Returns:
        Dictionary matching the ``run_full_evaluation`` output schema.
    """
    out: dict = {
        "accuracy_diagnostics": pd.DataFrame(),
        "realism_diagnostics": pd.DataFrame(),
        "component_diagnostics": None,
        "comparison": None,
        "sensitivity": None,
        "scorecard": None,
        "figures": {},
    }

    acc_path = results_dir / "accuracy_diagnostics.csv"
    if acc_path.exists():
        out["accuracy_diagnostics"] = pd.read_csv(acc_path)
        logger.info("Loaded accuracy diagnostics: %d rows", len(out["accuracy_diagnostics"]))

    real_path = results_dir / "realism_diagnostics.csv"
    if real_path.exists():
        out["realism_diagnostics"] = pd.read_csv(real_path)
        logger.info("Loaded realism diagnostics: %d rows", len(out["realism_diagnostics"]))

    comp_path = results_dir / "comparison_diagnostics.csv"
    if comp_path.exists():
        out["comparison"] = pd.read_csv(comp_path)
        logger.info("Loaded comparison diagnostics: %d rows", len(out["comparison"]))

    # Sensitivity
    pert_path = results_dir / "sensitivity_perturbation.csv"
    stab_path = results_dir / "sensitivity_stability_index.csv"
    if pert_path.exists() or stab_path.exists():
        sens: dict = {}
        if pert_path.exists():
            sens["perturbation"] = pd.read_csv(pert_path)
        if stab_path.exists():
            sens["stability_index"] = pd.read_csv(stab_path)
        out["sensitivity"] = sens

    # Reconstruct a minimal scorecard from accuracy data
    acc = out["accuracy_diagnostics"]
    if not acc.empty:
        model_name = acc["model_name"].iloc[0] if "model_name" in acc.columns else ""
        run_id = acc["run_id"].iloc[0] if "run_id" in acc.columns else ""
        out["scorecard"] = _build_scorecard_from_diagnostics(acc, out["realism_diagnostics"], run_id, model_name)

    # Load figures from figures/ subdirectory if present
    fig_dir = results_dir / "figures"
    if fig_dir.is_dir():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            for png_path in sorted(fig_dir.glob("*.png")):
                fig, ax = plt.subplots()
                img = mpimg.imread(str(png_path))
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout(pad=0)
                out["figures"][png_path.stem] = fig
                logger.info("Loaded figure: %s", png_path.stem)
        except ImportError:
            logger.warning("matplotlib not available; skipping figure loading")

    return out


def _build_scorecard_from_diagnostics(
    accuracy_df: pd.DataFrame,
    realism_df: pd.DataFrame,
    run_id: str,
    model_name: str,
) -> ScorecardEntry:
    """Build a ScorecardEntry from diagnostic DataFrames.

    Uses the same logic as ModelScorecard but avoids importing the full
    class to keep this script lightweight.
    """
    from cohort_projections.analysis.evaluation.scorecard import ModelScorecard

    config_path = _DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
    else:
        config = {}

    sc = ModelScorecard(config)
    return sc.build_scorecard(
        accuracy_diagnostics=accuracy_df,
        realism_diagnostics=realism_df if not realism_df.empty else None,
        run_id=run_id,
        model_name=model_name,
    )


def _run_fresh_evaluation(results_path: Path, config: dict) -> dict:
    """Run a full evaluation from a projection results file.

    Args:
        results_path: Path to a parquet file containing projection results.
        config: Evaluation config dict.

    Returns:
        Dictionary from ``EvaluationRunner.run_full_evaluation()``.
    """
    from cohort_projections.analysis.evaluation.runner import EvaluationRunner

    results_df = pd.read_parquet(results_path)
    logger.info("Loaded projection results: %d rows from %s", len(results_df), results_path)

    runner = EvaluationRunner()
    return runner.run_full_evaluation(results_df)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    if args.config.exists():
        with open(args.config) as fh:
            config = yaml.safe_load(fh)
        logger.info("Loaded config from %s", args.config)
    else:
        logger.warning("Config not found at %s, using defaults", args.config)
        config = {}

    # Load or compute evaluation results
    if args.results_dir is not None:
        if not args.results_dir.is_dir():
            logger.error("Results directory does not exist: %s", args.results_dir)
            sys.exit(1)
        evaluation_results = _load_from_results_dir(args.results_dir)
    else:
        if not args.projection_results.exists():
            logger.error("Projection results file not found: %s", args.projection_results)
            sys.exit(1)
        evaluation_results = _run_fresh_evaluation(args.projection_results, config)

    # Generate report
    output_path = generate_html_report(
        evaluation_results=evaluation_results,
        config=config,
        output_path=args.output,
        title=args.title,
        include_appendix=not args.no_appendix,
    )

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
