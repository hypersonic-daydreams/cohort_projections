#!/usr/bin/env python3
"""Build Introduction claims from extracted PDF text.

This script regenerates only the Introduction-section claims in
claims_manifest.jsonl, preserving existing claim IDs and leaving
all other sections untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

TITLE_LINE = "Forecasting International Migration to North Dakota"
INTRO_HEADING = "Introduction"
STOP_HEADING = "Data and Methods"

ABBREVIATIONS = [
    "U.S.",
    "U.K.",
    "et al.",
    "e.g.",
    "i.e.",
    "etc.",
]

SENTENCE_END_RE = re.compile(r'([.!?]["\')\]]?)\s+')
LIST_ITEM_RE = re.compile(r"^\d+\.\s+")


@dataclass(frozen=True)
class Line:
    """Represents a single extracted line with page metadata."""

    page: int
    text: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Regenerate Introduction claims from extracted pages."
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("extracted"),
        help="Directory containing extracted page-*.txt files.",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="PDF page numbers to load for the Introduction section.",
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
        help="Source PDF path recorded in claims_manifest.jsonl.",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="Introduction",
        help="Section label to update.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updated Introduction claims to claims_manifest.jsonl.",
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


def filter_headers(lines: Iterable[Line]) -> List[Line]:
    """Remove repeated headers, page numbers, and standalone numerals."""

    filtered: List[Line] = []
    for line in lines:
        if line.text == TITLE_LINE:
            continue
        if line.text.isdigit():
            continue
        filtered.append(line)
    return filtered


def slice_introduction(lines: Sequence[Line]) -> List[Line]:
    """Slice lines between the Introduction header and the next section."""

    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.text == INTRO_HEADING and start_idx is None:
            start_idx = idx + 1
            continue
        if start_idx is not None and line.text == STOP_HEADING:
            end_idx = idx
            break
    if start_idx is None:
        raise ValueError("Introduction heading not found in extracted text.")
    if end_idx is None:
        raise ValueError("Stop heading not found in extracted text.")
    return list(lines[start_idx:end_idx])


def normalize_lines(lines: Sequence[Line]) -> List[Line]:
    """Normalize list numbering and list lead-in colon."""

    normalized: List[Line] = []
    for idx, line in enumerate(lines):
        text = line.text
        next_text = None
        for j in range(idx + 1, len(lines)):
            candidate = lines[j].text
            if candidate:
                next_text = candidate
                break
        if text.endswith(":") and next_text and LIST_ITEM_RE.match(next_text):
            text = text[:-1] + "."
        text = LIST_ITEM_RE.sub("", text)
        normalized.append(Line(page=line.page, text=text))
    return normalized


def merge_lines_with_offsets(
    lines: Sequence[Line],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Merge lines into a single text blob and track page offsets."""

    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    current_len = 0
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if parts:
            last_piece = parts[-1]
            if last_piece.endswith("-"):
                parts[-1] = last_piece[:-1]
                current_len -= 1
            elif last_piece.endswith(("\u2014", "\u2013")):
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
    if "?" in text:
        return "methodological"
    if any(keyword in lowered for keyword in [" should ", " must ", " need to "]):
        return "normative"
    if any(keyword in lowered for keyword in [" is defined", " refers to", " means "]):
        return "definition"
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
        ]
    ):
        return "causal"
    if "forecasting" in lowered and any(
        keyword in lowered for keyword in ["literature", "paradigm"]
    ):
        return "methodological"
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
        ]
    ):
        return "comparative"
    if any(
        keyword in lowered
        for keyword in [
            "forecast",
            "projection",
            "scenario",
            "future",
            "prediction interval",
            "uncertainty",
        ]
    ):
        return "forecast"
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
        ]
    ):
        return "methodological"
    return "descriptive"


def build_intro_claims(
    extracted_dir: Path,
    pages: Sequence[int],
    section: str,
    source_file: str,
) -> List[dict]:
    """Build Introduction claims from extracted text."""

    raw_lines = load_page_lines(extracted_dir, pages)
    filtered = filter_headers(raw_lines)
    intro_lines = slice_introduction(filtered)
    normalized = normalize_lines(intro_lines)
    text, offsets = merge_lines_with_offsets(normalized)
    protected = protect_abbreviations(text)
    sentences_with_offsets = split_sentences(protected)
    sentences_with_pages = assign_pages(sentences_with_offsets, offsets)
    sentences_with_pages = [
        (normalize_sentence(restore_abbreviations(sentence)), page)
        for sentence, page in sentences_with_pages
    ]
    claims: List[dict] = []
    for sentence, page in sentences_with_pages:
        claims.append(
            {
                "claim_text": sentence,
                "claim_type": infer_claim_type(sentence),
                "source": {"pdf_page": page, "section": section},
                "status": "unassigned",
                "source_file": source_file,
            }
        )
    return claims


def update_manifest(
    manifest_path: Path,
    new_claims: Sequence[dict],
    section: str,
) -> List[dict]:
    """Update the manifest with new claims for the selected section."""

    records: List[dict] = []
    with manifest_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    intro_indices = [
        idx
        for idx, record in enumerate(records)
        if record.get("source", {}).get("section") == section
    ]
    if not intro_indices:
        raise ValueError(f"No {section} claims found in manifest.")
    if len(intro_indices) != len(new_claims):
        raise ValueError(
            f"{section} claim count mismatch: existing={len(intro_indices)}, "
            f"new={len(new_claims)}."
        )

    updated = list(records)
    for idx, new_claim in zip(intro_indices, new_claims):
        record = dict(records[idx])
        record.update(new_claim)
        record["claim_id"] = records[idx]["claim_id"]
        updated[idx] = record
    return updated


def write_manifest(manifest_path: Path, records: Sequence[dict]) -> None:
    """Write the updated manifest to disk."""

    with manifest_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the Introduction claim regeneration pipeline."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    new_claims = build_intro_claims(
        extracted_dir=args.extracted_dir,
        pages=args.pages,
        section=args.section,
        source_file=args.source_file,
    )
    LOGGER.info("Generated %s %s claims.", len(new_claims), args.section)

    updated_records = update_manifest(args.manifest, new_claims, args.section)
    if args.write:
        write_manifest(args.manifest, updated_records)
        LOGGER.info("Updated manifest written to %s.", args.manifest)
    else:
        LOGGER.info("Dry run; re-run with --write to update manifest.")


if __name__ == "__main__":
    main()
