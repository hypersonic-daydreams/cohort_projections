#!/usr/bin/env python3
"""LLM-assisted claim extraction from extracted PDF text.

This script provides structured text chunks to an LLM agent for semantic
claim extraction. Claims are identified by meaning/ideas, not by punctuation.

Designed for parallel sub-agent processing:
- Each chunk is sized for detailed analysis (~1500-2000 chars)
- Chunks can be processed independently by parallel agents
- Results are merged with consistent claim ID assignment

Usage:
    # List chunks for a section (for parallel agent dispatch)
    python claims/extract_claims.py --section Introduction --list-chunks

    # Output a specific chunk for agent processing
    python claims/extract_claims.py --section Introduction --chunk 1

    # Output guidance only
    python claims/extract_claims.py --output-guidance

The actual claim extraction is performed by LLM sub-agents using
the guidance and text provided by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Section configuration
SECTION_CONFIG: Dict[str, Dict] = {
    "Abstract": {
        "start_heading": "Abstract",
        "stop_heading": "Introduction",
        "pages": [1, 2],
    },
    "Introduction": {
        "start_heading": "Introduction",
        "stop_heading": "Data and Methods",
        "pages": [3, 4, 5],
    },
    "Data and Methods": {
        "start_heading": "Data and Methods",
        "stop_heading": "Results",
        "pages": list(range(5, 20)),
    },
    "Results": {
        "start_heading": "Results",
        "stop_heading": "Discussion",
        "pages": list(range(19, 32)),
    },
    "Discussion": {
        "start_heading": "Discussion",
        "stop_heading": "Conclusion",
        "pages": list(range(31, 46)),
    },
    "Conclusion": {
        "start_heading": "Conclusion",
        "stop_heading": "References",
        "pages": list(range(45, 48)),
    },
    "Appendix": {
        "start_heading": "Appendix",
        "stop_heading": None,
        "pages": list(range(51, 62)),
    },
}

TITLE_LINE = "Forecasting International Migration to North Dakota"

# Claim extraction guidance for LLM agents
EXTRACTION_GUIDANCE = """
## Claim Extraction Guidelines

A **claim** is a discrete, verifiable assertion. Extract claims based on IDEAS,
not punctuation or grammar.

### What constitutes a single claim:
- One factual assertion (e.g., "North Dakota ranks 47th in population")
- One statistical result (e.g., "coefficient of variation is 82.5%")
- One methodological statement (e.g., "we use difference-in-differences estimation")
- One causal/comparative assertion (e.g., "X has a larger effect than Y")

### When to split into multiple claims:
- Compound sentences with distinct facts: "X is true, and Y is also true" → 2 claims
- Sentences with multiple statistics: "Mean is 5.2 and SD is 1.3" → 2 claims
- Lists of findings: each item is typically its own claim
- Cause and effect stated together: may be 1 claim (causal) or 2 (descriptive + causal)

### When to keep as one claim:
- A complex idea that requires multiple clauses to express
- Comparative statements that need both sides: "X is higher than Y because of Z"
- Methodological descriptions with necessary context

### Claim types:
- `descriptive`: States a fact or observation
- `comparative`: Compares two or more things
- `causal`: Asserts a cause-effect relationship
- `methodological`: Describes methods, data, or analytical approach
- `forecast`: Makes a prediction or projection
- `definition`: Defines a term or concept
- `normative`: States what should or ought to be done

### For tables and figures:
- Table/figure captions are claims (usually methodological or descriptive)
- Key statistics from tables should be individual claims
- Axis labels and formatting notes are NOT claims

### Output format:
Each claim should be a JSON object with:
- claim_text: The verbatim or lightly normalized text
- claim_type: One of the types above
- pdf_page: The page number where the claim appears
- notes: Optional notes about extraction decisions
"""


@dataclass
class TextChunk:
    """A chunk of text with page context."""

    page: int
    text: str
    chunk_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare text for LLM-based claim extraction."
    )
    parser.add_argument(
        "--section",
        type=str,
        choices=list(SECTION_CONFIG.keys()),
        help="Section to extract claims from.",
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
        "--chunk-size",
        type=int,
        default=1500,
        help="Approximate characters per chunk for LLM processing.",
    )
    parser.add_argument(
        "--output-guidance",
        action="store_true",
        help="Output extraction guidance for LLM agent.",
    )
    parser.add_argument(
        "--list-chunks",
        action="store_true",
        help="List available chunks for parallel processing.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        help="Output a specific chunk by number (1-indexed).",
    )
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Output all chunks with separators.",
    )
    return parser.parse_args()


def load_section_text(
    extracted_dir: Path,
    section: str,
) -> List[TextChunk]:
    """Load and chunk text for a section."""

    config = SECTION_CONFIG[section]
    pages = config["pages"]
    start_heading = config["start_heading"]
    stop_heading = config.get("stop_heading")

    # Load all page text
    all_lines: List[tuple] = []  # (page, line)
    for page in pages:
        path = extracted_dir / f"page-{page}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            text = line.strip()
            if text and text != TITLE_LINE and not text.isdigit():
                all_lines.append((page, text))

    # Find section boundaries
    start_idx = None
    end_idx = None

    for idx, (page, text) in enumerate(all_lines):
        if start_idx is None and text == start_heading:
            start_idx = idx + 1
            continue
        if start_idx is not None and stop_heading and text == stop_heading:
            end_idx = idx
            break

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(all_lines)

    section_lines = all_lines[start_idx:end_idx]

    # Create chunks
    chunks: List[TextChunk] = []
    current_text = []
    current_page = section_lines[0][0] if section_lines else pages[0]
    chunk_id = 1

    for page, line in section_lines:
        current_text.append(line)

        # Create new chunk when we hit size limit or page boundary
        combined = "\n".join(current_text)
        if len(combined) >= 2000 or (current_text and page != current_page):
            chunks.append(
                TextChunk(
                    page=current_page,
                    text=combined,
                    chunk_id=chunk_id,
                )
            )
            current_text = []
            current_page = page
            chunk_id += 1

    # Don't forget the last chunk
    if current_text:
        chunks.append(
            TextChunk(
                page=current_page,
                text="\n".join(current_text),
                chunk_id=chunk_id,
            )
        )

    return chunks


def get_next_claim_id(manifest_path: Path) -> int:
    """Get the next available claim ID number."""

    if not manifest_path.exists():
        return 1

    max_id = 0
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            claim_id = record.get("claim_id", "")
            if claim_id.startswith("C") and claim_id[1:].isdigit():
                max_id = max(max_id, int(claim_id[1:]))
        except json.JSONDecodeError:
            continue

    return max_id + 1


def output_chunk_for_extraction(
    chunk: TextChunk,
    section: str,
    start_claim_id: int,
) -> None:
    """Output a chunk formatted for LLM extraction."""

    print(f"SECTION: {section}")
    print(f"CHUNK: {chunk.chunk_id}")
    print(f"PAGE: {chunk.page}")
    print(f"START_CLAIM_ID: C{start_claim_id:04d}")
    print("=" * 60)
    print()
    print(chunk.text)
    print()
    print("=" * 60)
    print()
    print("Extract discrete claims from the text above.")
    print("Output each claim as a JSON object, one per line.")
    print()
    print("Required fields:")
    print("  - claim_id: Sequential ID starting from START_CLAIM_ID above")
    print("  - claim_text: The claim text (verbatim or lightly normalized)")
    print(
        "  - claim_type: descriptive|comparative|causal|methodological|forecast|definition|normative"
    )
    print("  - source: {pdf_page: <page>, section: <section>}")
    print("  - status: 'unassigned'")
    print()
    print("Example output:")
    print(
        json.dumps(
            {
                "claim_id": f"C{start_claim_id:04d}",
                "claim_text": "Example claim text here.",
                "claim_type": "descriptive",
                "source": {"pdf_page": chunk.page, "section": section},
                "status": "unassigned",
            }
        )
    )


def main() -> None:
    args = parse_args()

    # Output guidance only
    if args.output_guidance:
        print(EXTRACTION_GUIDANCE)
        return

    # Need section for other operations
    if not args.section:
        print(
            "Error: --section required unless using --output-guidance", file=sys.stderr
        )
        sys.exit(1)

    chunks = load_section_text(args.extracted_dir, args.section)
    next_id = get_next_claim_id(args.manifest)

    # List chunks for parallel dispatch
    if args.list_chunks:
        print(f"Section: {args.section}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Next claim ID: C{next_id:04d}")
        print()
        for chunk in chunks:
            print(
                f"  Chunk {chunk.chunk_id}: Page {chunk.page}, {len(chunk.text)} chars"
            )
        return

    # Output specific chunk
    if args.chunk is not None:
        if args.chunk < 1 or args.chunk > len(chunks):
            print(
                f"Error: chunk {args.chunk} out of range (1-{len(chunks)})",
                file=sys.stderr,
            )
            sys.exit(1)
        chunk = chunks[args.chunk - 1]

        # Estimate claim ID offset based on chunk position
        # (rough estimate: ~5 claims per chunk)
        estimated_prior_claims = (args.chunk - 1) * 5
        chunk_start_id = next_id + estimated_prior_claims

        output_chunk_for_extraction(chunk, args.section, chunk_start_id)
        return

    # Output all chunks
    if args.all_chunks:
        print(f"=== SECTION: {args.section} ===")
        print(f"=== CHUNKS: {len(chunks)} ===")
        print(f"=== STARTING ID: C{next_id:04d} ===")
        print()

        for chunk in chunks:
            print(f"\n{'#' * 70}")
            print(f"# CHUNK {chunk.chunk_id} / PAGE {chunk.page}")
            print(f"{'#' * 70}\n")
            print(chunk.text)
            print()

        return

    # Default: show summary
    print(f"Section: {args.section}")
    print(f"Chunks: {len(chunks)}")
    print(f"Next claim ID: C{next_id:04d}")
    print()
    print("Use --list-chunks, --chunk N, or --all-chunks to get content.")


if __name__ == "__main__":
    main()
