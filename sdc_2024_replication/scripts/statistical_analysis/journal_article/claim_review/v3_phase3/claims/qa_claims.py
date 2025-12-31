#!/usr/bin/env python3
"""QA checks for claims_manifest.jsonl.

Checks for schema-like integrity issues, common extraction artifacts, and
section-specific coverage. Supports comparison against section parsers
for any document section.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

CLAIM_ID_RE = re.compile(r"^C[0-9]{4}$")
VALID_CLAIM_TYPES = {
    "descriptive",
    "comparative",
    "causal",
    "forecast",
    "methodological",
    "definition",
    "normative",
}
VALID_STATUSES = {
    "unassigned",
    "assigned",
    "in_review",
    "verified",
    "disputed",
    "needs_revision",
}

SUSPICIOUS_STARTS = (
    "and ",
    "but ",
    "or ",
    "however ",
    "however,",
    "therefore ",
    "because ",
    "with ",
)

# Section expected page ranges for validation
SECTION_PAGES = {
    "Abstract": [1, 2],
    "Introduction": [3, 4, 5],
    "Data and Methods": list(range(5, 20)),
    "Results": list(range(19, 32)),
    "Discussion": list(range(31, 46)),
    "Conclusion": list(range(45, 48)),
    "Appendix": list(range(51, 62)),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run QA checks on claims manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("claims/claims_manifest.jsonl"),
        help="Path to claims_manifest.jsonl.",
    )
    parser.add_argument(
        "--section",
        type=str,
        default=None,
        help="Restrict checks to a section label (e.g., Introduction).",
    )
    parser.add_argument(
        "--expected-pages",
        nargs="*",
        type=int,
        default=None,
        help="Expected pdf_page values for the chosen section.",
    )
    parser.add_argument(
        "--compare-intro",
        action="store_true",
        help="Compare Introduction claims against build_intro_claims output (legacy).",
    )
    parser.add_argument(
        "--compare-section",
        action="store_true",
        help="Compare section claims against build_section_claims output.",
    )
    parser.add_argument(
        "--intro-pages",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Pages to use for Introduction comparison (legacy).",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("extracted"),
        help="Directory containing extracted page-*.txt files.",
    )
    parser.add_argument(
        "--all-sections",
        action="store_true",
        help="Run QA on all sections with default page expectations.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if errors are detected.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Tuple[List[dict], List[str]]:
    """Load JSONL manifest records."""

    records: List[dict] = []
    errors: List[str] = []
    if not path.exists():
        errors.append(f"Manifest not found: {path}")
        return records, errors
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_no}: invalid JSON ({exc}).")
    return records, errors


def normalize_text(text: str) -> str:
    """Normalize whitespace for comparisons."""

    return re.sub(r"\s+", " ", text).strip()


def check_required_fields(record: dict) -> List[str]:
    """Check required fields and basic types."""

    issues: List[str] = []
    claim_id = record.get("claim_id")
    if not isinstance(claim_id, str) or not CLAIM_ID_RE.match(claim_id):
        issues.append("Invalid or missing claim_id.")
    claim_text = record.get("claim_text")
    if not isinstance(claim_text, str) or not claim_text.strip():
        issues.append("Missing or empty claim_text.")
    claim_type = record.get("claim_type")
    if claim_type not in VALID_CLAIM_TYPES:
        issues.append("Invalid or missing claim_type.")
    status = record.get("status")
    if status not in VALID_STATUSES:
        issues.append("Invalid or missing status.")
    source = record.get("source")
    if not isinstance(source, dict):
        issues.append("Missing or invalid source object.")
    return issues


def check_text_quality(record: dict) -> List[str]:
    """Heuristic warnings for likely parsing artifacts."""

    warnings: List[str] = []
    text = record.get("claim_text", "")
    if not isinstance(text, str):
        return warnings
    normalized = text.strip()
    if normalized != text:
        warnings.append("Claim text has leading/trailing whitespace.")
    if "  " in text:
        warnings.append("Claim text contains double spaces.")
    lowered = normalized.lower()
    if normalized and normalized[0].islower():
        warnings.append("Claim text starts with a lowercase letter.")
    for prefix in SUSPICIOUS_STARTS:
        if lowered.startswith(prefix):
            warnings.append(f"Claim text starts with '{prefix.strip()}'.")
            break
    if normalized and normalized[-1] not in ".!?":
        warnings.append("Claim text does not end with sentence punctuation.")
    if normalized.count("(") != normalized.count(")"):
        warnings.append("Unbalanced parentheses in claim text.")
    numeric_ratio = sum(ch.isdigit() for ch in normalized) / max(len(normalized), 1)
    if numeric_ratio > 0.3 and len(normalized) > 25:
        warnings.append("High numeric density (possible axis label or table artifact).")
    if re.search(r"\b(?:ACF|Density|Residual|Panel|Q-Q)\b", normalized):
        warnings.append("Possible figure/axis label fragment detected.")
    return warnings


def check_duplicates(records: Iterable[dict]) -> List[str]:
    """Warn on duplicate claim text within the same section."""

    warnings: List[str] = []
    by_section: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        section = record.get("source", {}).get("section", "Unknown")
        text = normalize_text(record.get("claim_text", ""))
        if text:
            by_section[section][text] += 1
    for section, counts in by_section.items():
        for text, count in counts.items():
            if count > 1:
                warnings.append(
                    f"Duplicate claim_text in section '{section}' ({count}x): {text}"
                )
    return warnings


def check_section_pages(
    records: Sequence[dict],
    section: str,
    expected_pages: Optional[Sequence[int]],
) -> List[str]:
    """Check page coverage for a given section."""

    warnings: List[str] = []
    pages = sorted({rec.get("source", {}).get("pdf_page") for rec in records})
    pages = [page for page in pages if isinstance(page, int)]
    if not pages:
        warnings.append(f"No pdf_page values found for section '{section}'.")
        return warnings
    if expected_pages:
        missing = sorted(set(expected_pages) - set(pages))
        extra = sorted(set(pages) - set(expected_pages))
        if missing:
            warnings.append(f"Section '{section}' missing expected pages: {missing}.")
        if extra:
            warnings.append(f"Section '{section}' has unexpected pages: {extra}.")
    return warnings


def compare_introduction(
    records: Sequence[dict],
    extracted_dir: Path,
    pages: Sequence[int],
) -> List[str]:
    """Compare Introduction claims against build_intro_claims output (legacy)."""

    issues: List[str] = []
    try:
        from build_intro_claims import build_intro_claims  # type: ignore
    except ImportError as exc:
        issues.append(f"Failed to import build_intro_claims: {exc}")
        return issues

    intro_records = [
        rec for rec in records if rec.get("source", {}).get("section") == "Introduction"
    ]
    expected = build_intro_claims(
        extracted_dir=extracted_dir,
        pages=pages,
        section="Introduction",
        source_file="",
    )
    expected_texts = [normalize_text(rec["claim_text"]) for rec in expected]
    actual_texts = [normalize_text(rec.get("claim_text", "")) for rec in intro_records]
    expected_counts = Counter(expected_texts)
    actual_counts = Counter(actual_texts)

    missing = list((expected_counts - actual_counts).elements())
    extra = list((actual_counts - expected_counts).elements())

    if missing:
        issues.append(f"Introduction missing {len(missing)} claims from parser output.")
    if extra:
        issues.append(f"Introduction has {len(extra)} extra claims vs parser output.")

    if not missing and not extra and len(expected) == len(intro_records):
        for idx, (expected_rec, actual_rec) in enumerate(
            zip(expected, intro_records), start=1
        ):
            if expected_rec["claim_text"] != actual_rec.get("claim_text"):
                issues.append(f"Introduction claim text mismatch at index {idx}.")
                break
            if expected_rec["source"]["pdf_page"] != actual_rec.get("source", {}).get(
                "pdf_page"
            ):
                issues.append(f"Introduction page mismatch at index {idx}.")
                break
    return issues


def compare_section(
    records: Sequence[dict],
    section: str,
    extracted_dir: Path,
) -> Tuple[List[str], List[str]]:
    """Compare section claims against build_section_claims output.

    Returns (errors, warnings) where errors are critical issues and
    warnings are informational discrepancies.
    """

    errors: List[str] = []
    warnings: List[str] = []

    try:
        from build_section_claims import build_section_claims  # type: ignore
    except ImportError as exc:
        errors.append(f"Failed to import build_section_claims: {exc}")
        return errors, warnings

    section_records = [
        rec for rec in records if rec.get("source", {}).get("section") == section
    ]

    if not section_records:
        errors.append(f"No claims found for section '{section}'.")
        return errors, warnings

    try:
        parser_claims = build_section_claims(
            extracted_dir=extracted_dir,
            section=section,
            source_file="",
        )
    except Exception as exc:
        errors.append(f"Parser failed for section '{section}': {exc}")
        return errors, warnings

    # Count comparison
    manifest_count = len(section_records)
    parser_count = len(parser_claims)

    if manifest_count != parser_count:
        warnings.append(
            f"{section}: claim count differs (manifest={manifest_count}, "
            f"parser={parser_count}, delta={manifest_count - parser_count})"
        )

    # Text coverage analysis
    manifest_texts = {normalize_text(r.get("claim_text", "")) for r in section_records}
    parser_texts = {normalize_text(c["claim_text"]) for c in parser_claims}

    only_in_parser = parser_texts - manifest_texts

    if only_in_parser:
        # Parser found claims not in manifest - might indicate missing claims
        warnings.append(
            f"{section}: parser found {len(only_in_parser)} unique claims "
            f"not in manifest (possible missing claims)"
        )

    # Page coverage check
    manifest_pages = sorted(
        {
            r.get("source", {}).get("pdf_page")
            for r in section_records
            if isinstance(r.get("source", {}).get("pdf_page"), int)
        }
    )
    parser_pages = sorted({c["source"]["pdf_page"] for c in parser_claims})

    if manifest_pages != parser_pages:
        warnings.append(
            f"{section}: page coverage differs "
            f"(manifest={manifest_pages}, parser={parser_pages})"
        )

    return errors, warnings


def run_section_qa(
    all_records: Sequence[dict],
    section: str,
    expected_pages: Optional[Sequence[int]],
    extracted_dir: Path,
    compare: bool = False,
) -> Tuple[List[str], List[str]]:
    """Run QA checks for a single section.

    Returns (errors, warnings).
    """

    errors: List[str] = []
    warnings: List[str] = []

    # Filter to section
    records = [
        rec for rec in all_records if rec.get("source", {}).get("section") == section
    ]

    if not records:
        errors.append(f"No records found for section '{section}'.")
        return errors, warnings

    LOGGER.info("Section '%s': %d claims", section, len(records))

    # Check required fields and text quality
    for idx, record in enumerate(records, start=1):
        claim_id = record.get("claim_id", f"unknown-{idx}")
        field_issues = check_required_fields(record)
        for issue in field_issues:
            errors.append(f"{claim_id}: {issue}")
        for issue in check_text_quality(record):
            warnings.append(f"{claim_id}: {issue}")

    # Check page coverage
    if expected_pages is None and section in SECTION_PAGES:
        expected_pages = SECTION_PAGES[section]

    if expected_pages:
        warnings.extend(check_section_pages(records, section, expected_pages))

    # Check duplicates within section
    dup_warnings = check_duplicates(records)
    warnings.extend(dup_warnings)

    # Compare with parser output if requested
    if compare:
        cmp_errors, cmp_warnings = compare_section(records, section, extracted_dir)
        errors.extend(cmp_errors)
        warnings.extend(cmp_warnings)

    return errors, warnings


def main() -> None:
    """Run QA checks and report results."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_records, load_errors = load_manifest(args.manifest)
    errors: List[str] = []
    warnings: List[str] = []
    errors.extend(load_errors)

    # Check for global duplicate claim_ids
    claim_id_counts = Counter()
    for record in all_records:
        claim_id = record.get("claim_id")
        if isinstance(claim_id, str):
            claim_id_counts[claim_id] += 1
    for claim_id, count in claim_id_counts.items():
        if count > 1:
            errors.append(f"Duplicate claim_id detected: {claim_id} ({count}x).")

    if args.all_sections:
        # Run QA on all sections
        for section in SECTION_PAGES.keys():
            section_errors, section_warnings = run_section_qa(
                all_records=all_records,
                section=section,
                expected_pages=SECTION_PAGES.get(section),
                extracted_dir=args.extracted_dir,
                compare=args.compare_section,
            )
            errors.extend(section_errors)
            warnings.extend(section_warnings)

    elif args.section:
        # Run QA on a single section
        section_errors, section_warnings = run_section_qa(
            all_records=all_records,
            section=args.section,
            expected_pages=args.expected_pages,
            extracted_dir=args.extracted_dir,
            compare=args.compare_section,
        )
        errors.extend(section_errors)
        warnings.extend(section_warnings)

        # Legacy intro comparison
        if args.compare_intro and args.section == "Introduction":
            errors.extend(
                compare_introduction(all_records, args.extracted_dir, args.intro_pages)
            )

    else:
        # Run basic checks on all records
        records = all_records
        for idx, record in enumerate(records, start=1):
            claim_id = record.get("claim_id")
            field_issues = check_required_fields(record)
            for issue in field_issues:
                errors.append(f"Record {idx}: {issue}")
            warnings.extend(
                f"Record {idx}: {issue}" for issue in check_text_quality(record)
            )

        warnings.extend(check_duplicates(records))

        if args.compare_intro:
            errors.extend(
                compare_introduction(records, args.extracted_dir, args.intro_pages)
            )

    # Report results
    if errors:
        for error in errors:
            LOGGER.error(error)
    if warnings:
        for warning in warnings:
            LOGGER.warning(warning)

    LOGGER.info("QA summary: %s error(s), %s warning(s).", len(errors), len(warnings))
    if args.strict and errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
