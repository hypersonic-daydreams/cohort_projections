#!/usr/bin/env python3
"""
Build versioned journal article artifacts.

This script optionally refreshes statistical analysis outputs, regenerates
publication figures, compiles the LaTeX manuscript, and packages a versioned
bundle containing the PDF, figures, results, and logs.

Usage:
    source .venv/bin/activate
    python sdc_2024_replication/scripts/statistical_analysis/journal_article/build_versioned_artifacts.py \
        --version 0.8.6 --status draft --refresh
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from cohort_projections.utils import setup_logger
from cohort_projections.utils.reproducibility import log_execution

LOGGER = setup_logger(__name__)

STATUS_TO_STAGE = {
    "draft": "working",
    "review": "working",
    "approved": "approved",
    "production": "production",
}

DEFAULT_REFRESH_SCRIPTS = [
    "module_regime_framework.py",
    "module_1_1_descriptive_statistics.py",
    "module_2_1_1_unit_root_tests.py",
    "module_3_1_panel_data.py",
    "module_4_regression_extensions.py",
    "module_6_machine_learning.py",
    "module_7_causal_inference.py",
    "module_7_robustness.py",
    "module_7_robustness_fast.py",
    "module_7b_lssnd_synthetic_control.py",
    "module_8_duration_analysis.py",
    "module_8b_status_durability.py",
    "module_9_scenario_modeling.py",
    "module_9b_policy_scenarios.py",
    "module_10_two_component_estimand.py",
    "module_secondary_migration.py",
    "module_B1_regime_aware_models.py",
    "run_pep_regime_modeling.py",
    "module_B4_bayesian_panel_var.py",
    "run_module_B2_multistate_placebo.py",
]


def ensure_journal_pep_regime_figure(
    analysis_dir: Path,
    journal_dir: Path,
    refresh_enabled: bool,
    logs_dir: Path,
) -> None:
    """
    Ensure the appendix PEP regime diagnostic figure exists in the LaTeX figures directory.

    The manuscript references `figures/fig_app_pep_regime_diagnostic.png`. This helper makes the
    one-command build robust by copying from the statistical-analysis output figure when present,
    and otherwise guiding the user to run with `--refresh`.
    """
    journal_figure = journal_dir / "figures" / "fig_app_pep_regime_diagnostic.png"
    if journal_figure.exists():
        return

    source = analysis_dir / "figures" / "module_B1_pep_regime_modeling__P0.png"
    if source.exists():
        journal_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, journal_figure)
        return

    if refresh_enabled:
        runner = analysis_dir / "run_pep_regime_modeling.py"
        runner_log = logs_dir / "refresh" / "run_pep_regime_modeling.log"
        run_command([sys.executable, str(runner)], runner_log, cwd=analysis_dir)
        if journal_figure.exists():
            return

    raise FileNotFoundError(
        "Missing appendix figure `figures/fig_app_pep_regime_diagnostic.png`. "
        "Run `build_versioned_artifacts.py --refresh` (or run "
        "`python sdc_2024_replication/scripts/statistical_analysis/run_pep_regime_modeling.py`)."
    )


def _project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[4]


def _analysis_dir() -> Path:
    """Return the statistical analysis directory."""
    return _project_root() / "sdc_2024_replication" / "scripts" / "statistical_analysis"


def _journal_dir() -> Path:
    """Return the journal article directory."""
    return _analysis_dir() / "journal_article"


def _config_path() -> Path:
    """Return the main projection config path."""
    return _project_root() / "config" / "projection_config.yaml"


def format_timestamp(timestamp: dt.datetime) -> str:
    """Format a UTC timestamp for filenames."""
    return timestamp.strftime("%Y%m%d_%H%M%S")


def build_version_basename(version: str, status: str, timestamp: str) -> str:
    """Build the versioned base filename (without extension)."""
    return f"article-{version}-{status}_{timestamp}"


def build_published_pdf_name(version: str, timestamp: str) -> str:
    """Build the published (flat) PDF filename for browsing."""
    return f"article-{version}_{timestamp}.pdf"


def stage_from_status(status: str) -> str:
    """Map a status to a bundle stage directory."""
    return STATUS_TO_STAGE[status]


def hash_file(path: Path) -> str:
    """Compute a SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(base: Path, patterns: Sequence[str]) -> list[Path]:
    """Collect files under a base directory matching glob patterns."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return sorted({path.resolve() for path in files if path.is_file()})


def collect_recursive_files(base: Path) -> list[Path]:
    """Collect all files under a directory."""
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*") if path.is_file())


def build_manifest(files: Iterable[Path], base: Path) -> list[dict[str, str]]:
    """Build a file manifest with relative paths and hashes."""
    manifest: list[dict[str, str]] = []
    for path in files:
        manifest.append(
            {
                "path": path.resolve().relative_to(base).as_posix(),
                "sha256": hash_file(path),
            }
        )
    return manifest


def configure_file_logger(logger: logging.Logger, log_path: Path) -> None:
    """Attach a file handler to the logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)


def run_command(command: Sequence[str], log_path: Path, cwd: Path | None = None) -> None:
    """Run a command and stream stdout/stderr to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running command: %s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def write_build_info(journal_dir: Path, version: str, timestamp: dt.datetime) -> Path:
    """Write a small LaTeX include file with build metadata."""
    build_info_path = journal_dir / "build_info.tex"
    timestamp_utc = timestamp.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    content = (
        "% Auto-generated by build_versioned_artifacts.py (do not edit by hand)\n"
        f"\\newcommand{{\\BuildVersion}}{{{version}}}\n"
        f"\\newcommand{{\\BuildTimestampUTC}}{{{timestamp_utc}}}\n"
    )
    build_info_path.write_text(content, encoding="utf-8")
    return build_info_path


def copy_files(
    files: Iterable[Path],
    source_root: Path,
    destination_root: Path,
) -> list[Path]:
    """Copy files preserving relative paths."""
    copied: list[Path] = []
    for path in files:
        relative = path.resolve().relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def update_versions_index(
    versions_path: Path,
    version: str,
    date: str,
    status: str,
    branch: str,
    notes: str,
) -> None:
    """Update the working versions table with a new entry."""
    if not versions_path.exists():
        versions_path.write_text("# Article Version Index\n\n", encoding="utf-8")

    lines = versions_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    inserted = False
    in_working = False
    header_seen = False
    separator_seen = False
    existing_row_prefix = f"| {version} |"

    for line in lines:
        if line.strip() == "## Working Versions":
            in_working = True
            header_seen = False
            separator_seen = False
            updated_lines.append(line)
            continue

        if in_working and line.startswith("## "):
            if not inserted:
                updated_lines.append(
                    f"| {version} | {date} | {status} | {branch} | {notes} |"
                )
                inserted = True
            in_working = False
            updated_lines.append(line)
            continue

        if in_working and line.startswith("| Version |"):
            header_seen = True
            updated_lines.append(line)
            continue

        if in_working and header_seen and line.startswith("|---"):
            separator_seen = True
            updated_lines.append(line)
            if not inserted:
                updated_lines.append(
                    f"| {version} | {date} | {status} | {branch} | {notes} |"
                )
                inserted = True
            continue

        if in_working and line.startswith(existing_row_prefix):
            if not inserted:
                updated_lines.append(
                    f"| {version} | {date} | {status} | {branch} | {notes} |"
                )
                inserted = True
            continue

        updated_lines.append(line)

    if not inserted:
        updated_lines.append("")
        updated_lines.append("## Working Versions")
        updated_lines.append("| Version | Date | Status | Branch | Notes |")
        updated_lines.append("|---------|------|--------|--------|-------|")
        updated_lines.append(f"| {version} | {date} | {status} | {branch} | {notes} |")

    updated_lines = [
        f"*Last Updated: {date}*" if line.startswith("*Last Updated:") else line
        for line in updated_lines
    ]
    versions_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def update_current_version_file(
    current_path: Path, version_label: str, updated_date: str
) -> None:
    """Update CURRENT_VERSION.txt with the latest working version."""
    if not current_path.exists():
        current_path.write_text(
            "# Current Article Versions\n"
            f"# Updated: {updated_date}\n\n"
            "# Current production version\n"
            "# (none - not yet submitted)\n\n"
            "# Current working version\n"
            f"{version_label}\n",
            encoding="utf-8",
        )
        return

    lines = current_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    i = 0
    updated_working = False
    updated_timestamp = False

    while i < len(lines):
        line = lines[i]
        if line.startswith("# Updated:"):
            updated_lines.append(f"# Updated: {updated_date}")
            updated_timestamp = True
            i += 1
            continue

        if line.strip().startswith("# Current working version"):
            updated_lines.append(line)
            i += 1
            while i < len(lines) and lines[i].strip().startswith("#"):
                updated_lines.append(lines[i])
                i += 1
            if i < len(lines):
                updated_lines.append(version_label)
                updated_working = True
                i += 1
            else:
                updated_lines.append(version_label)
                updated_working = True
            continue

        updated_lines.append(line)
        i += 1

    if not updated_timestamp:
        updated_lines.insert(1, f"# Updated: {updated_date}")

    if not updated_working:
        updated_lines.append("")
        updated_lines.append("# Current working version")
        updated_lines.append(version_label)

    updated_lines = [
        f"# Also available with full timestamp: {version_label}"
        if "Also available with full timestamp" in line
        else line
        for line in updated_lines
    ]

    current_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def get_git_commit(project_root: Path) -> str:
    """Return the current git commit SHA."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch(project_root: Path) -> str:
    """Return the current git branch name."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build versioned article artifacts.")
    parser.add_argument("--version", required=True, help="Semantic version (e.g., 0.8.6).")
    parser.add_argument(
        "--status",
        choices=sorted(STATUS_TO_STAGE.keys()),
        default="draft",
        help="Release status label for the artifact.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run statistical analysis modules before building artifacts.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help=(
            "Base directory for versioned bundles (the stage/version folders are created beneath it). "
            "If relative, it is resolved from the repository root. "
            "Default: journal_article/output/versions"
        ),
    )
    parser.add_argument(
        "--publish-pdf-dir",
        default=None,
        help=(
            "Optional flat directory to copy the final PDF into for easy browsing. "
            "Only the PDF is copied (no figures/logs/results), with filename "
            "`article-{version}_{timestamp}.pdf` (status/stage omitted). "
            "If relative, it is resolved from the repository root."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the build pipeline."""
    args = parse_args(argv)
    project_root = _project_root()
    analysis_dir = _analysis_dir()
    journal_dir = _journal_dir()
    config_path = _config_path()

    timestamp = dt.datetime.now(dt.UTC)
    timestamp_label = format_timestamp(timestamp)
    stage = stage_from_status(args.status)
    version_basename = build_version_basename(args.version, args.status, timestamp_label)

    output_base = (
        Path(args.output_base)
        if args.output_base
        else journal_dir / "output" / "versions"
    )
    if not output_base.is_absolute():
        output_base = project_root / output_base

    output_root = output_base / stage / version_basename
    logs_dir = output_root / "logs"
    figures_dir = output_root / "figures"
    results_dir = output_root / "results"

    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    configure_file_logger(LOGGER, logs_dir / "build.log")

    LOGGER.info("Starting versioned artifact build: %s", version_basename)

    required_paths = [
        journal_dir / "create_publication_figures.py",
        journal_dir / "compile.sh",
        journal_dir / "main.tex",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    inputs_for_log = [config_path, journal_dir / "main.tex", journal_dir / "references.bib"]
    outputs_for_log = [output_root]

    with log_execution(
        __file__,
        parameters={
            "version": args.version,
            "status": args.status,
            "stage": stage,
            "refresh": args.refresh,
            "output_base": str(output_base),
            "publish_pdf_dir": args.publish_pdf_dir,
        },
        inputs=[str(path) for path in inputs_for_log if path.exists()],
        outputs=[str(path) for path in outputs_for_log],
    ):
        build_info_path = write_build_info(journal_dir, args.version, timestamp)

        refresh_logs_dir = logs_dir / "refresh"
        if args.refresh:
            refresh_logs_dir.mkdir(parents=True, exist_ok=True)
            for script_name in DEFAULT_REFRESH_SCRIPTS:
                script_path = analysis_dir / script_name
                if not script_path.exists():
                    raise FileNotFoundError(f"Refresh script missing: {script_path}")
                log_path = refresh_logs_dir / f"{script_path.stem}.log"
                run_command([sys.executable, str(script_path)], log_path, cwd=analysis_dir)

        figures_log = logs_dir / "figures.log"
        run_command(
            [sys.executable, str(journal_dir / "create_publication_figures.py")],
            figures_log,
            cwd=journal_dir,
        )

        ensure_journal_pep_regime_figure(
            analysis_dir=analysis_dir,
            journal_dir=journal_dir,
            refresh_enabled=args.refresh,
            logs_dir=logs_dir,
        )

        pdf_output_path = output_root / f"{version_basename}.pdf"
        latex_log = logs_dir / "latex.log"
        run_command(
            [str(journal_dir / "compile.sh"), "--output", str(pdf_output_path)],
            latex_log,
            cwd=journal_dir,
        )

        if args.publish_pdf_dir:
            publish_dir = Path(args.publish_pdf_dir)
            if not publish_dir.is_absolute():
                publish_dir = project_root / publish_dir
            publish_dir.mkdir(parents=True, exist_ok=True)
            published_name = build_published_pdf_name(args.version, timestamp_label)
            shutil.copy2(pdf_output_path, publish_dir / published_name)

        source_figures = collect_recursive_files(journal_dir / "figures")
        if not source_figures:
            raise FileNotFoundError("No figures found after figure generation.")
        copy_files(source_figures, journal_dir / "figures", figures_dir)

        source_results = collect_recursive_files(analysis_dir / "results")
        if not source_results:
            raise FileNotFoundError(
                "No analysis results found. Run with --refresh or generate outputs first."
            )
        copy_files(source_results, analysis_dir / "results", results_dir)

        latex_logs = collect_files(journal_dir, ["*.log", "*.blg"])
        for log_file in latex_logs:
            shutil.copy2(log_file, logs_dir / log_file.name)

        input_files = collect_files(
            journal_dir,
            [
                "main.tex",
                "build_info.tex",
                "preamble.tex",
                "references.bib",
                "sections/*.tex",
                "figures/*.tex",
                "figures/*.pdf",
                "create_publication_figures.py",
                "compile.sh",
            ],
        )
        input_files.extend(source_results)
        input_files = sorted({path.resolve() for path in input_files})

        output_files = collect_recursive_files(output_root)
        output_files = [path for path in output_files if path.name != f"{version_basename}.metadata.json"]

        metadata = {
            "version": {
                "version": args.version,
                "status": args.status,
                "stage": stage,
                "timestamp": timestamp.isoformat(),
                "basename": version_basename,
            },
            "build": {
                "git_commit": get_git_commit(project_root),
                "git_branch": get_git_branch(project_root),
                "config_sha256": hash_file(config_path) if config_path.exists() else "missing",
            },
            "refresh": {
                "enabled": args.refresh,
                "scripts": DEFAULT_REFRESH_SCRIPTS,
            },
            "inputs": {
                "count": len(input_files),
                "files": build_manifest(input_files, project_root),
            },
            "outputs": {
                "count": len(output_files),
                "files": build_manifest(output_files, project_root),
            },
        }

        metadata_path = output_root / f"{version_basename}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        date_label = timestamp.date().isoformat()
        versions_path = journal_dir / "output" / "VERSIONS.md"
        notes = "Automated build (refresh)" if args.refresh else "Automated build"
        update_versions_index(
            versions_path,
            args.version,
            date_label,
            args.status,
            get_git_branch(project_root),
            notes,
        )

        current_path = journal_dir / "output" / "CURRENT_VERSION.txt"
        update_current_version_file(current_path, f"{version_basename}.pdf", date_label)

    LOGGER.info("Build complete: %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
