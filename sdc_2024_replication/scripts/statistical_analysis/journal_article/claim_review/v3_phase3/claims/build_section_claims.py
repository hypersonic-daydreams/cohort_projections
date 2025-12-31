#!/usr/bin/env python3
"""Build claims from extracted PDF text for any section.

This script regenerates claims for a specified section in claims_manifest.jsonl,
preserving existing claim IDs and leaving all other sections untouched.

Generalizes build_intro_claims.py to handle all document sections.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

# Document title used as repeated header
TITLE_LINE = "Forecasting International Migration to North Dakota"

# Section configuration: section_name -> (start_heading, stop_heading, pages)
# Pages are inclusive ranges; stop_heading is exclusive
SECTION_CONFIG: Dict[str, Dict] = {
    "Abstract": {
        "start_heading": "Abstract",
        "stop_heading": "Introduction",
        "pages": [1, 2],
        "start_after_heading": True,
    },
    "Introduction": {
        "start_heading": "Introduction",
        "stop_heading": "Data and Methods",
        "pages": [3, 4, 5],
        "start_after_heading": True,
    },
    "Data and Methods": {
        "start_heading": "Data and Methods",
        "stop_heading": "Results",
        "pages": list(range(5, 20)),  # Pages 5-19
        "start_after_heading": True,
    },
    "Results": {
        "start_heading": "Results",
        "stop_heading": "Discussion",
        "pages": list(range(19, 32)),  # Pages 19-31
        "start_after_heading": True,
    },
    "Discussion": {
        "start_heading": "Discussion",
        "stop_heading": "Conclusion",
        "pages": list(range(31, 46)),  # Pages 31-45
        "start_after_heading": True,
    },
    "Conclusion": {
        "start_heading": "Conclusion",
        "stop_heading": "References",
        "pages": list(range(45, 48)),  # Pages 45-47
        "start_after_heading": True,
    },
    "Appendix": {
        "start_heading": "Appendix",
        "stop_heading": None,  # End of document
        "pages": list(range(51, 62)),  # Pages 51-61
        "start_after_heading": True,
    },
}

# Common abbreviations to protect during sentence splitting
ABBREVIATIONS = [
    "U.S.",
    "U.K.",
    "et al.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Jr.",
    "Sr.",
    "Inc.",
    "Ltd.",
    "Fig.",
    "Eq.",
    "Vol.",
    "No.",
    "pp.",
    "ca.",
    "cf.",
    "approx.",
]

# Regex patterns
SENTENCE_END_RE = re.compile(r'([.!?]["\')\]]?)\s+')
LIST_ITEM_RE = re.compile(r"^\d+\.\s+")
SUBSECTION_RE = re.compile(r"^\d+\.\d+(\.\d+)?\s+")
TABLE_HEADER_RE = re.compile(r"^Table\s+\d+:")
FIGURE_HEADER_RE = re.compile(r"^Figure\s+\d+:")
EQUATION_RE = re.compile(r"^[A-Za-z_]+\s*[=≈≤≥<>]")

# Lines to filter out (axis labels, plot artifacts, etc.)
FILTER_PATTERNS = [
    re.compile(r"^\d+$"),  # Standalone page numbers
    re.compile(r"^[-−–—]+$"),  # Horizontal rules
    re.compile(r"^\d{4}$"),  # Years alone
    re.compile(r"^[0-9.,\s]+$"),  # Numeric-only lines (table data)
    re.compile(r"^(ACF|Density|Residual|Panel|Q-Q|Lag)\s*$", re.IGNORECASE),
    re.compile(r"^(Year|Count|Rate|Value|Total)\s*$", re.IGNORECASE),
]


@dataclass(frozen=True)
class Line:
    """Represents a single extracted line with page metadata."""

    page: int
    text: str


@dataclass
class SectionConfig:
    """Configuration for a document section."""

    name: str
    start_heading: str
    stop_heading: Optional[str]
    pages: List[int]
    start_after_heading: bool = True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Regenerate claims for a specific section from extracted pages."
    )
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        choices=list(SECTION_CONFIG.keys()),
        help="Section to regenerate claims for.",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("extracted"),
        help="Directory containing extracted page-*.txt files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("claims/claims_manifest.jsonl"),
        help="Path to claims_manifest.jsonl.",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default=(
            "sdc_2024_replication/scripts/statistical_analysis/"
            "journal_article/claim_review/v3_phase3/source/"
            "article_draft_v5_p305_complete.pdf"
        ),
        help="Source PDF path recorded in claims.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updated claims to claims_manifest.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print claims without modifying manifest (implies no --write).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output.",
    )
    return parser.parse_args()


def load_page_lines(extracted_dir: Path, pages: Sequence[int]) -> List[Line]:
    """Load extracted text lines for selected pages."""

    lines: List[Line] = []
    for page in pages:
        path = extracted_dir / f"page-{page}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing extracted page: {path}")
        for raw in path.read_text().splitlines():
            text = raw.strip()
            if not text:
                continue
            lines.append(Line(page=page, text=text))
    return lines


def filter_headers_and_noise(lines: Iterable[Line]) -> List[Line]:
    """Remove repeated headers, page numbers, and noise lines."""

    filtered: List[Line] = []
    for line in lines:
        # Skip document title (repeated header)
        if line.text == TITLE_LINE:
            continue
        # Skip lines matching filter patterns
        if any(pattern.match(line.text) for pattern in FILTER_PATTERNS):
            continue
        filtered.append(line)
    return filtered


def slice_section(
    lines: Sequence[Line],
    config: SectionConfig,
) -> List[Line]:
    """Slice lines between section headings."""

    start_idx = None
    end_idx = None

    for idx, line in enumerate(lines):
        # Find start heading
        if start_idx is None:
            if line.text == config.start_heading or line.text.startswith(
                config.start_heading + " "
            ):
                start_idx = idx + 1 if config.start_after_heading else idx
                continue

        # Find stop heading (if specified)
        if start_idx is not None and config.stop_heading is not None:
            if line.text == config.stop_heading or line.text.startswith(
                config.stop_heading + " "
            ):
                end_idx = idx
                break

    if start_idx is None:
        raise ValueError(f"Start heading '{config.start_heading}' not found.")

    if config.stop_heading is not None and end_idx is None:
        # Stop heading might be on a page not in our range; take all remaining
        LOGGER.warning(
            "Stop heading '%s' not found; taking all remaining lines.",
            config.stop_heading,
        )

    return list(lines[start_idx:end_idx])


def normalize_lines(lines: Sequence[Line]) -> List[Line]:
    """Normalize list numbering and colon-based list lead-ins."""

    normalized: List[Line] = []
    for idx, line in enumerate(lines):
        text = line.text

        # Look ahead for list pattern
        next_text = None
        for j in range(idx + 1, len(lines)):
            candidate = lines[j].text
            if candidate:
                next_text = candidate
                break

        # Convert list lead-in colon to period
        if text.endswith(":") and next_text and LIST_ITEM_RE.match(next_text):
            text = text[:-1] + "."

        # Remove list numbering
        text = LIST_ITEM_RE.sub("", text)

        normalized.append(Line(page=line.page, text=text))
    return normalized


def merge_lines_with_offsets(
    lines: Sequence[Line],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Merge lines into a single text blob and track page offsets."""

    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []  # (char_offset, page)
    current_len = 0

    for line in lines:
        text = line.text.strip()
        if not text:
            continue

        if parts:
            last_piece = parts[-1]
            # Handle hyphenation at line breaks
            if last_piece.endswith("-"):
                # Check if it's a real hyphen (compound word) vs line-break artifact
                # Heuristic: if next text starts lowercase, likely line-break
                if text and text[0].islower():
                    parts[-1] = last_piece[:-1]
                    current_len -= 1
                # else keep hyphen for compound words
            elif last_piece.endswith(("\u2014", "\u2013")):
                # Em/en dash - don't add space
                pass
            else:
                parts.append(" ")
                current_len += 1

        offsets.append((current_len, line.page))
        parts.append(text)
        current_len += len(text)

    return "".join(parts), offsets


def protect_abbreviations(text: str) -> str:
    """Protect abbreviations from sentence splitting."""

    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "~"))
    return protected


def restore_abbreviations(text: str) -> str:
    """Restore protected abbreviations."""

    return text.replace("~", ".")


def split_sentences(text: str) -> List[Tuple[str, int]]:
    """Split text into sentences with start offsets."""

    sentences: List[Tuple[str, int]] = []
    start = 0

    for match in SENTENCE_END_RE.finditer(text):
        end = match.end(1)
        sentence = text[start:end].strip()
        if sentence:
            sentences.append((sentence, start))
        start = match.end()

    # Handle trailing text
    tail = text[start:].strip()
    if tail:
        sentences.append((tail, start))

    return sentences


def assign_pages(
    sentences: Sequence[Tuple[str, int]],
    offsets: Sequence[Tuple[int, int]],
) -> List[Tuple[str, int]]:
    """Assign pdf_page values to sentences based on line offsets."""

    if not offsets:
        raise ValueError("No offsets available for page assignment.")

    positions = [pos for pos, _ in offsets]
    pages = [page for _, page in offsets]

    assigned: List[Tuple[str, int]] = []
    for sentence, start in sentences:
        page = None
        for idx in range(len(positions) - 1, -1, -1):
            if positions[idx] <= start:
                page = pages[idx]
                break
        if page is None:
            page = pages[0]
        assigned.append((sentence, page))

    return assigned


def normalize_sentence(text: str) -> str:
    """Normalize spacing inside a sentence."""

    return re.sub(r"\s+", " ", text).strip()


def infer_claim_type(text: str) -> str:
    """Infer claim type using lightweight heuristics."""

    lowered = text.lower()

    # Research questions
    if "?" in text:
        return "methodological"

    # Normative claims
    if any(keyword in lowered for keyword in [" should ", " must ", " need to "]):
        return "normative"

    # Definitions
    if any(keyword in lowered for keyword in [" is defined", " refers to", " means "]):
        return "definition"

    # Causal claims
    if any(
        keyword in lowered
        for keyword in [
            "effect",
            "effects",
            "impact",
            "causal",
            "policy-associated",
            "policy intervention",
            "policy interventions",
            "bound on causal",
            "difference-in-differences",
            "treatment effect",
        ]
    ):
        return "causal"

    # Comparative claims
    if any(
        keyword in lowered
        for keyword in [
            "relative",
            "smaller than",
            "larger than",
            "higher",
            "lower",
            "more",
            "less",
            "different from",
            "disproportionately",
            "dominant",
            "outsized",
            "exceeds",
            "exceeding",
            "outperforms",
            "compared to",
            "comparison",
        ]
    ):
        return "comparative"

    # Forecast claims
    if any(
        keyword in lowered
        for keyword in [
            "forecast",
            "projection",
            "scenario",
            "future",
            "prediction interval",
            "uncertainty",
            "monte carlo",
            "through 2045",
        ]
    ):
        return "forecast"

    # Methodological claims
    if any(
        keyword in lowered
        for keyword in [
            "analysis",
            "study",
            "method",
            "framework",
            "module",
            "estimation",
            "regression",
            "time series",
            "gravity",
            "machine learning",
            "data sources",
            "research question",
            "section ",
            "literature",
            "paradigm",
            "design",
            "approach",
            "model",
            "specification",
            "robustness",
            "diagnostic",
            "test",
            "coefficient",
            "variable",
            "equation",
        ]
    ):
        return "methodological"

    return "descriptive"


def build_section_claims(
    extracted_dir: Path,
    section: str,
    source_file: str,
) -> List[dict]:
    """Build claims from extracted text for a given section."""

    if section not in SECTION_CONFIG:
        raise ValueError(f"Unknown section: {section}")

    cfg = SECTION_CONFIG[section]
    config = SectionConfig(
        name=section,
        start_heading=cfg["start_heading"],
        stop_heading=cfg.get("stop_heading"),
        pages=cfg["pages"],
        start_after_heading=cfg.get("start_after_heading", True),
    )

    # Load and filter lines
    raw_lines = load_page_lines(extracted_dir, config.pages)
    filtered = filter_headers_and_noise(raw_lines)

    # Slice to section boundaries
    try:
        section_lines = slice_section(filtered, config)
    except ValueError as e:
        LOGGER.error("Failed to slice section '%s': %s", section, e)
        raise

    # Normalize and merge
    normalized = normalize_lines(section_lines)
    text, offsets = merge_lines_with_offsets(normalized)

    # Split into sentences
    protected = protect_abbreviations(text)
    sentences_with_offsets = split_sentences(protected)
    sentences_with_pages = assign_pages(sentences_with_offsets, offsets)

    # Build claim records
    claims: List[dict] = []
    for sentence, page in sentences_with_pages:
        cleaned = normalize_sentence(restore_abbreviations(sentence))

        # Skip very short or likely artifact sentences
        if len(cleaned) < 10:
            continue
        if cleaned.startswith(("Table ", "Figure ", "Note:", "Notes:")):
            # Keep these but mark appropriately
            pass

        claims.append(
            {
                "claim_text": cleaned,
                "claim_type": infer_claim_type(cleaned),
                "source": {"pdf_page": page, "section": section},
                "status": "unassigned",
                "source_file": source_file,
            }
        )

    return claims


def load_manifest(manifest_path: Path) -> List[dict]:
    """Load existing manifest records."""

    records: List[dict] = []
    if not manifest_path.exists():
        return records

    with manifest_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def update_manifest(
    manifest_path: Path,
    new_claims: Sequence[dict],
    section: str,
) -> List[dict]:
    """Update the manifest with new claims for the selected section."""

    records = load_manifest(manifest_path)

    # Find indices of existing claims for this section
    section_indices = [
        idx
        for idx, record in enumerate(records)
        if record.get("source", {}).get("section") == section
    ]

    if not section_indices:
        raise ValueError(f"No existing {section} claims found in manifest.")

    if len(section_indices) != len(new_claims):
        LOGGER.warning(
            "%s claim count changed: existing=%d, new=%d",
            section,
            len(section_indices),
            len(new_claims),
        )
        # This is informational; we'll still update

    # Update existing records
    updated = list(records)

    if len(section_indices) == len(new_claims):
        # Same count - update in place
        for idx, new_claim in zip(section_indices, new_claims):
            record = dict(records[idx])
            record.update(new_claim)
            record["claim_id"] = records[idx]["claim_id"]
            updated[idx] = record
    else:
        # Count mismatch - need to handle carefully
        # For now, preserve existing IDs for matching claims
        raise ValueError(
            f"Claim count mismatch for {section}: existing={len(section_indices)}, "
            f"new={len(new_claims)}. Manual reconciliation required."
        )

    return updated


def write_manifest(manifest_path: Path, records: Sequence[dict]) -> None:
    """Write the updated manifest to disk."""

    with manifest_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the section claim regeneration pipeline."""

    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    LOGGER.info("Building claims for section: %s", args.section)

    new_claims = build_section_claims(
        extracted_dir=args.extracted_dir,
        section=args.section,
        source_file=args.source_file,
    )
    LOGGER.info("Generated %d %s claims.", len(new_claims), args.section)

    if args.dry_run:
        for idx, claim in enumerate(new_claims, start=1):
            print(f"\n[{idx}] Page {claim['source']['pdf_page']}")
            print(f"    Type: {claim['claim_type']}")
            print(f"    Text: {claim['claim_text'][:100]}...")
        return

    try:
        updated_records = update_manifest(args.manifest, new_claims, args.section)

        if args.write:
            write_manifest(args.manifest, updated_records)
            LOGGER.info("Updated manifest written to %s.", args.manifest)
        else:
            LOGGER.info("Dry run; re-run with --write to update manifest.")

    except ValueError as e:
        LOGGER.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
