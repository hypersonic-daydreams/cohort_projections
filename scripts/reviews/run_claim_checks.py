#!/usr/bin/env python3
"""
Claim Verification Runner
=========================

Executes deterministic checks for audit claims defined in a YAML registry and
writes evidence artifacts for later adjudication.

This script is intentionally generic and does not encode any claim-specific
logic. Claims, commands, and pass/fail assertions live in:
`docs/reviews/repo-hygiene-audit/verification/claims_registry.yaml`.

Usage:
    python scripts/reviews/run_claim_checks.py list
    python scripts/reviews/run_claim_checks.py progress
    python scripts/reviews/run_claim_checks.py run --claim-id RHA-001
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "docs/reviews/repo-hygiene-audit/verification/claims_registry.yaml"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "docs/reviews/repo-hygiene-audit/verification/evidence"
DEFAULT_PROGRESS_FILE = REPO_ROOT / "docs/reviews/repo-hygiene-audit/verification/progress.md"
LOGGER = logging.getLogger(__name__)

DEFAULT_ALLOWED_STATUSES = {
    "queued",
    "checks_defined",
    "in_progress",
    "adjudicated",
    "closed",
}
DEFAULT_ALLOWED_VERDICTS = {
    "unreviewed",
    "confirmed",
    "partially_confirmed",
    "refuted",
    "stale",
    "unverifiable",
}
ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "info"}
ALLOWED_CLAIM_TYPES = {"structural", "behavioral", "interpretive"}


@dataclass(frozen=True)
class RegistryValidation:
    """Result of registry schema validation."""

    errors: list[str]
    warnings: list[str]
    allowed_statuses: set[str]
    allowed_verdicts: set[str]


def configure_logging(verbose: bool = False) -> None:
    """Configure CLI logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_registry(registry_path: Path) -> dict[str, Any]:
    """Load registry YAML from disk."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Registry must be a YAML mapping at top level.")
    return data


def ensure_string_list(value: Any, field_name: str) -> tuple[list[str], list[str]]:
    """Normalize a YAML field to a list of strings and collect validation errors."""
    errors: list[str] = []
    if value is None:
        return [], errors
    if isinstance(value, str):
        return [value], errors
    if isinstance(value, list):
        bad = [item for item in value if not isinstance(item, str)]
        if bad:
            errors.append(f"{field_name} must contain only strings.")
            return [], errors
        return value, errors
    errors.append(f"{field_name} must be a string or list of strings.")
    return [], errors


def validate_registry(registry: dict[str, Any]) -> RegistryValidation:
    """Validate registry schema and claim/check metadata."""
    errors: list[str] = []
    warnings: list[str] = []

    claims = registry.get("claims", [])
    if not isinstance(claims, list):
        errors.append("`claims` must be a list.")
        claims = []

    allowed_statuses = set(registry.get("allowed_statuses", [])) or DEFAULT_ALLOWED_STATUSES
    allowed_verdicts = set(registry.get("allowed_verdicts", [])) or DEFAULT_ALLOWED_VERDICTS

    seen_claim_ids: set[str] = set()
    for idx, claim in enumerate(claims, start=1):
        claim_ctx = f"claims[{idx}]"
        if not isinstance(claim, dict):
            errors.append(f"{claim_ctx} must be a mapping.")
            continue

        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            errors.append(f"{claim_ctx}.claim_id is required and must be a non-empty string.")
            claim_id = f"<missing-{idx}>"
        elif claim_id in seen_claim_ids:
            errors.append(f"Duplicate claim_id: {claim_id}")
        else:
            seen_claim_ids.add(claim_id)

        for field in ("title", "claim_text"):
            value = claim.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{claim_ctx}.{field} is required and must be a non-empty string.")

        severity = claim.get("severity")
        if severity not in ALLOWED_SEVERITIES:
            errors.append(f"{claim_ctx}.severity must be one of {sorted(ALLOWED_SEVERITIES)}.")

        claim_type = claim.get("claim_type")
        if claim_type not in ALLOWED_CLAIM_TYPES:
            errors.append(f"{claim_ctx}.claim_type must be one of {sorted(ALLOWED_CLAIM_TYPES)}.")

        status = claim.get("status")
        if status not in allowed_statuses:
            errors.append(f"{claim_ctx}.status must be one of {sorted(allowed_statuses)}.")

        verdict = claim.get("verdict")
        if verdict not in allowed_verdicts:
            errors.append(f"{claim_ctx}.verdict must be one of {sorted(allowed_verdicts)}.")

        source = claim.get("source")
        if not isinstance(source, dict):
            errors.append(f"{claim_ctx}.source must be a mapping.")
        else:
            source_file = source.get("file")
            if not isinstance(source_file, str) or not source_file.strip():
                errors.append(f"{claim_ctx}.source.file must be a non-empty string.")
            else:
                source_path = REPO_ROOT / source_file
                if not source_path.exists():
                    warnings.append(f"{claim_ctx}.source.file does not exist: {source_file}")

        checks = claim.get("checks")
        if not isinstance(checks, list):
            errors.append(f"{claim_ctx}.checks must be a list.")
            continue

        seen_check_ids: set[str] = set()
        for check_idx, check in enumerate(checks, start=1):
            check_ctx = f"{claim_ctx}.checks[{check_idx}]"
            if not isinstance(check, dict):
                errors.append(f"{check_ctx} must be a mapping.")
                continue

            check_id = check.get("check_id")
            if not isinstance(check_id, str) or not check_id.strip():
                errors.append(f"{check_ctx}.check_id is required and must be a non-empty string.")
            elif check_id in seen_check_ids:
                errors.append(f"Duplicate check_id in claim {claim_id}: {check_id}")
            else:
                seen_check_ids.add(check_id)

            for field in ("description", "command"):
                value = check.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{check_ctx}.{field} is required and must be a non-empty string.")

            timeout = check.get("timeout_seconds", 30)
            if not isinstance(timeout, int) or timeout <= 0:
                errors.append(f"{check_ctx}.timeout_seconds must be a positive integer.")

            if "assertion" in check and not isinstance(check["assertion"], dict):
                errors.append(f"{check_ctx}.assertion must be a mapping.")
                continue

            assertion = check.get("assertion", {})
            if isinstance(assertion, dict):
                for field_name in (
                    "stdout_contains",
                    "stdout_not_contains",
                    "stderr_contains",
                    "stderr_not_contains",
                    "regex_match",
                ):
                    _, field_errors = ensure_string_list(assertion.get(field_name), f"{check_ctx}.{field_name}")
                    errors.extend(field_errors)

    return RegistryValidation(
        errors=errors,
        warnings=warnings,
        allowed_statuses=allowed_statuses,
        allowed_verdicts=allowed_verdicts,
    )


def normalize_claim_selector(values: list[str] | None) -> set[str]:
    """Normalize repeated/CSV CLI selection values into a set."""
    if not values:
        return set()
    output: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                output.add(item)
    return output


def select_claims(
    claims: list[dict[str, Any]],
    claim_ids: set[str],
    statuses: set[str],
) -> list[dict[str, Any]]:
    """Filter claims by optional claim ID and status selectors."""
    selected = claims
    if claim_ids:
        selected = [claim for claim in selected if claim.get("claim_id") in claim_ids]
    if statuses:
        selected = [claim for claim in selected if claim.get("status") in statuses]
    return selected


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate long command output for evidence artifacts."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]..."


def run_shell_command(command: str, timeout_seconds: int) -> dict[str, Any]:
    """Execute a shell command and capture structured output."""
    started = utc_now_iso()
    start_time = time.perf_counter()

    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        timed_out = False
        return_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

    duration_seconds = round(time.perf_counter() - start_time, 3)

    return {
        "started_at_utc": started,
        "duration_seconds": duration_seconds,
        "return_code": return_code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def evaluate_assertion(
    command_result: dict[str, Any],
    assertion: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Evaluate check assertions against command output."""
    failures: list[str] = []
    stdout = command_result["stdout"]
    stderr = command_result["stderr"]
    return_code = command_result["return_code"]
    timed_out = command_result["timed_out"]

    expect_exit_code = assertion.get("expect_exit_code", 0)
    if expect_exit_code is not None and return_code != expect_exit_code:
        failures.append(f"Expected exit code {expect_exit_code}, got {return_code}.")

    if timed_out:
        failures.append("Command timed out.")

    stdout_contains, _ = ensure_string_list(assertion.get("stdout_contains"), "assertion.stdout_contains")
    stdout_not_contains, _ = ensure_string_list(
        assertion.get("stdout_not_contains"),
        "assertion.stdout_not_contains",
    )
    stderr_contains, _ = ensure_string_list(assertion.get("stderr_contains"), "assertion.stderr_contains")
    stderr_not_contains, _ = ensure_string_list(
        assertion.get("stderr_not_contains"),
        "assertion.stderr_not_contains",
    )
    regex_match, _ = ensure_string_list(assertion.get("regex_match"), "assertion.regex_match")

    failures.extend([f"stdout missing required token: {token}" for token in stdout_contains if token not in stdout])
    failures.extend([f"stdout contains forbidden token: {token}" for token in stdout_not_contains if token in stdout])
    failures.extend([f"stderr missing required token: {token}" for token in stderr_contains if token not in stderr])
    failures.extend([f"stderr contains forbidden token: {token}" for token in stderr_not_contains if token in stderr])
    failures.extend(
        [f"stdout does not match regex: {pattern}" for pattern in regex_match if re.search(pattern, stdout, flags=re.MULTILINE) is None]
    )

    return len(failures) == 0, failures


def sanitize_filename(value: str) -> str:
    """Make a filesystem-safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def run_checks_for_claim(
    claim: dict[str, Any],
    max_output_chars: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Run all deterministic checks for a single claim."""
    claim_id = claim["claim_id"]
    checks = claim.get("checks", [])
    required_total = 0
    required_passed = 0
    checks_passed = 0
    results: list[dict[str, Any]] = []

    for check in checks:
        check_id = check["check_id"]
        required = bool(check.get("required", True))
        if required:
            required_total += 1

        if dry_run:
            results.append(
                {
                    "check_id": check_id,
                    "description": check["description"],
                    "required": required,
                    "command": check["command"],
                    "timeout_seconds": check.get("timeout_seconds", 30),
                    "assertion": check.get("assertion", {"expect_exit_code": 0}),
                    "executed": False,
                    "passed": None,
                    "failure_reasons": [],
                    "stdout": "",
                    "stderr": "",
                }
            )
            continue

        command_result = run_shell_command(check["command"], timeout_seconds=check.get("timeout_seconds", 30))
        assertion = check.get("assertion", {})
        if not assertion:
            assertion = {"expect_exit_code": 0}

        passed, failures = evaluate_assertion(command_result=command_result, assertion=assertion)

        if passed:
            checks_passed += 1
            if required:
                required_passed += 1

        results.append(
            {
                "check_id": check_id,
                "description": check["description"],
                "required": required,
                "command": check["command"],
                "timeout_seconds": check.get("timeout_seconds", 30),
                "assertion": assertion,
                "executed": True,
                "passed": passed,
                "failure_reasons": failures,
                "started_at_utc": command_result["started_at_utc"],
                "duration_seconds": command_result["duration_seconds"],
                "return_code": command_result["return_code"],
                "timed_out": command_result["timed_out"],
                "stdout": truncate_text(command_result["stdout"], max_output_chars),
                "stderr": truncate_text(command_result["stderr"], max_output_chars),
            }
        )

    claim_passed: bool | None = None if dry_run else required_passed == required_total

    return {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "claim_id": claim_id,
        "claim_title": claim["title"],
        "claim_status_at_run": claim["status"],
        "claim_verdict_at_run": claim["verdict"],
        "results": results,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": checks_passed if not dry_run else None,
            "required_total": required_total,
            "required_passed": required_passed if not dry_run else None,
            "claim_passed": claim_passed,
            "dry_run": dry_run,
        },
    }


def write_evidence_artifact(output_dir: Path, artifact: dict[str, Any]) -> Path:
    """Write a JSON evidence artifact to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    claim_slug = sanitize_filename(artifact["claim_id"])
    output_file = output_dir / f"{timestamp}_{claim_slug}.json"
    output_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_file


def load_latest_evidence(output_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Load the latest evidence artifact per claim ID."""
    latest_by_claim: dict[str, tuple[Path, dict[str, Any]]] = {}
    if not output_dir.exists():
        return latest_by_claim

    for file_path in sorted(output_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        claim_id = payload.get("claim_id")
        if not isinstance(claim_id, str):
            continue
        prior = latest_by_claim.get(claim_id)
        if prior is None or file_path.stat().st_mtime > prior[0].stat().st_mtime:
            latest_by_claim[claim_id] = (file_path, payload)

    return latest_by_claim


def write_progress_markdown(
    registry: dict[str, Any],
    registry_path: Path,
    evidence_dir: Path,
    progress_path: Path,
) -> None:
    """Render progress dashboard markdown from registry + latest evidence."""
    claims = registry.get("claims", [])
    latest_evidence = load_latest_evidence(evidence_dir)

    status_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for claim in claims:
        status = claim.get("status", "unknown")
        verdict = claim.get("verdict", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    lines = [
        "# Repo Hygiene Audit Claim Verification Progress",
        "",
        f"**Generated (UTC):** {utc_now_iso()}",
        f"**Registry:** `{registry_path.relative_to(REPO_ROOT)}`",
        f"**Evidence Directory:** `{evidence_dir.relative_to(REPO_ROOT)}`",
        "",
        "## Summary",
        "",
        f"- Total claims: **{len(claims)}**",
        "- Status counts:",
    ]

    if status_counts:
        lines.extend([f"  - `{status}`: {status_counts[status]}" for status in sorted(status_counts)])
    else:
        lines.append("  - _(no claims registered)_")

    lines.append("- Verdict counts:")
    if verdict_counts:
        lines.extend([f"  - `{verdict}`: {verdict_counts[verdict]}" for verdict in sorted(verdict_counts)])
    else:
        lines.append("  - _(no claims registered)_")

    lines.extend(
        [
            "",
            "## Claim Tracker",
            "",
            "| Claim ID | Severity | Type | Status | Verdict | Checks | Last Run (UTC) | Evidence |",
            "|---|---|---|---|---|---:|---|---|",
        ]
    )

    if not claims:
        lines.append("| _(none)_ | - | - | - | - | 0 | - | - |")
    else:
        for claim in claims:
            claim_id = claim["claim_id"]
            evidence_link = "-"
            last_run = "-"
            checks_summary = f"{len(claim.get('checks', []))}"

            latest = latest_evidence.get(claim_id)
            if latest is not None:
                evidence_file, payload = latest
                summary = payload.get("summary", {})
                checks_passed = summary.get("checks_passed")
                checks_total = summary.get("checks_total")
                if isinstance(checks_passed, int) and isinstance(checks_total, int):
                    checks_summary = f"{checks_passed}/{checks_total}"
                elif isinstance(checks_total, int):
                    checks_summary = f"{checks_total}"

                last_run = payload.get("generated_at_utc", "-")
                try:
                    relative = evidence_file.relative_to(progress_path.parent)
                    evidence_link = f"[artifact]({relative.as_posix()})"
                except ValueError:
                    evidence_link = evidence_file.as_posix()

            lines.append(
                "| "
                f"{claim_id} | "
                f"{claim.get('severity', '-')} | "
                f"{claim.get('claim_type', '-')} | "
                f"{claim.get('status', '-')} | "
                f"{claim.get('verdict', '-')} | "
                f"{checks_summary} | "
                f"{last_run} | "
                f"{evidence_link} |"
            )

    lines.extend(
        [
            "",
            "## Agent Workflow",
            "",
            "1. Claim Extractor Agent registers an atomic claim in `claims_registry.yaml`.",
            "2. Check Designer Agent adds deterministic checks and assertions.",
            "3. Evidence Runner Agent executes checks via `run_claim_checks.py run`.",
            "4. Adjudicator Agent updates `verdict`, `confidence`, and `notes` in registry.",
            "5. Tracker Agent regenerates this file via `run_claim_checks.py progress`.",
            "",
            "## Notes",
            "",
            "- This tracker is generated from the registry and latest evidence artifacts.",
            "- Edit claims in YAML, then regenerate progress to reflect updates.",
        ]
    )

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_claims(claims: list[dict[str, Any]]) -> None:
    """Log a concise claim list."""
    if not claims:
        LOGGER.info("No claims registered.")
        return

    LOGGER.info("Registered claims: %s", len(claims))
    for claim in claims:
        LOGGER.info(
            "%s | %-8s | %-12s | %-14s | checks=%s",
            claim["claim_id"],
            claim.get("severity", "-"),
            claim.get("claim_type", "-"),
            claim.get("status", "-"),
            len(claim.get("checks", [])),
        )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run deterministic checks for claim verification.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to claims registry YAML.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory for JSON evidence artifacts.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=DEFAULT_PROGRESS_FILE,
        help="Path to generated progress markdown.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List claims from registry.")
    subparsers.add_parser("progress", help="Regenerate progress markdown without running checks.")

    run_parser = subparsers.add_parser("run", help="Execute checks for selected claims.")
    run_parser.add_argument(
        "--claim-id",
        action="append",
        help="Claim ID(s) to run (repeatable or comma-separated). Defaults to all selected status claims.",
    )
    run_parser.add_argument(
        "--status",
        action="append",
        help="Claim status filter(s), repeatable or comma-separated (e.g., checks_defined,in_progress).",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Resolve selections but do not execute checks.")
    run_parser.add_argument(
        "--max-output-chars",
        type=int,
        default=4000,
        help="Max chars stored per stdout/stderr field in evidence JSON.",
    )
    run_parser.add_argument(
        "--fail-on-check-failure",
        action="store_true",
        help="Exit non-zero if any selected claim fails required checks.",
    )

    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    try:
        registry = load_registry(args.registry)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    validation = validate_registry(registry)
    for warning in validation.warnings:
        LOGGER.warning("%s", warning)
    if validation.errors:
        LOGGER.error("Registry validation failed with %s error(s):", len(validation.errors))
        for error in validation.errors:
            LOGGER.error("  - %s", error)
        return 1

    claims: list[dict[str, Any]] = registry.get("claims", [])

    if args.command == "list":
        list_claims(claims)
        return 0

    if args.command == "progress":
        write_progress_markdown(
            registry=registry,
            registry_path=args.registry.resolve(),
            evidence_dir=args.evidence_dir.resolve(),
            progress_path=args.progress_file.resolve(),
        )
        LOGGER.info("Wrote progress tracker: %s", args.progress_file)
        return 0

    claim_ids = normalize_claim_selector(args.claim_id)
    statuses = normalize_claim_selector(args.status)

    unknown_statuses = statuses - validation.allowed_statuses
    if unknown_statuses:
        LOGGER.error("Unknown status filter(s): %s", ", ".join(sorted(unknown_statuses)))
        return 1

    selected = select_claims(claims=claims, claim_ids=claim_ids, statuses=statuses)
    if not selected:
        LOGGER.info("No claims matched selection filters.")
        write_progress_markdown(
            registry=registry,
            registry_path=args.registry.resolve(),
            evidence_dir=args.evidence_dir.resolve(),
            progress_path=args.progress_file.resolve(),
        )
        LOGGER.info("Wrote progress tracker: %s", args.progress_file)
        return 0

    LOGGER.info("Selected %s claim(s) for execution.", len(selected))

    any_required_failures = False
    for claim in selected:
        artifact = run_checks_for_claim(
            claim=claim,
            max_output_chars=args.max_output_chars,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            LOGGER.info("Dry run: %s (%s checks)", claim["claim_id"], len(claim.get("checks", [])))
            continue

        output_file = write_evidence_artifact(args.evidence_dir, artifact)
        summary = artifact["summary"]
        LOGGER.info(
            "Claim %s: required passed %s/%s -> %s",
            claim["claim_id"],
            summary["required_passed"],
            summary["required_total"],
            "PASS" if summary["claim_passed"] else "FAIL",
        )
        LOGGER.info("Evidence: %s", output_file)
        if not summary["claim_passed"]:
            any_required_failures = True

    write_progress_markdown(
        registry=registry,
        registry_path=args.registry.resolve(),
        evidence_dir=args.evidence_dir.resolve(),
        progress_path=args.progress_file.resolve(),
    )
    LOGGER.info("Wrote progress tracker: %s", args.progress_file)

    if args.fail_on_check_failure and any_required_failures:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
