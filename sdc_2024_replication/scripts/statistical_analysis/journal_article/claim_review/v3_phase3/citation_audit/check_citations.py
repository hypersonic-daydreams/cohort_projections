#!/usr/bin/env python3
"""Check LaTeX citations against BibTeX entries and APA 7th completeness."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

APA_EDITION = "7th"

CITE_PATTERN = re.compile(
    r"\\(?P<cmd>[A-Za-z]*cite[a-zA-Z*]*)\s*"
    r"(?:\[[^\]]*\]\s*){0,2}"
    r"\{(?P<keys>[^}]*)\}"
)

IGNORE_BIB_TYPES = {"comment", "preamble", "string"}

ENTRY_TYPE_CANONICAL = {
    "article": "article",
    "articleinpress": "article",
    "book": "book",
    "incollection": "chapter",
    "inbook": "chapter",
    "inproceedings": "conference",
    "conference": "conference",
    "proceedings": "proceedings",
    "techreport": "report",
    "report": "report",
    "manual": "report",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "thesis": "thesis",
    "online": "online",
    "misc": "online",
    "webpage": "online",
}

FIELD_GROUPS = {
    "creator": ["author", "editor", "organization", "institution"],
    "year": ["year", "date", "issued"],
    "title": ["title"],
    "journal": ["journal", "journaltitle"],
    "booktitle": ["booktitle", "maintitle"],
    "publisher": ["publisher"],
    "institution": ["institution", "organization", "school", "university"],
    "school": ["school", "institution", "university"],
    "volume": ["volume"],
    "number": ["number", "issue"],
    "pages": ["pages", "pagetotal"],
    "edition": ["edition"],
    "url": ["url"],
    "doi": ["doi"],
    "report_number": ["number", "report_number"],
}

APA_REQUIREMENTS = {
    "article": {
        "required": ["creator", "year", "title", "journal"],
        "recommended": ["volume", "number", "pages", "doi_or_url"],
    },
    "book": {
        "required": ["creator", "year", "title", "publisher"],
        "recommended": ["edition", "doi_or_url"],
    },
    "chapter": {
        "required": ["creator", "year", "title", "booktitle", "editor", "publisher"],
        "recommended": ["pages", "doi_or_url"],
    },
    "conference": {
        "required": ["creator", "year", "title", "booktitle"],
        "recommended": ["editor", "publisher", "pages", "doi_or_url"],
    },
    "proceedings": {
        "required": ["creator", "year", "title"],
        "recommended": ["publisher", "doi_or_url"],
    },
    "report": {
        "required": ["creator", "year", "title", "institution"],
        "recommended": ["report_number", "url"],
    },
    "thesis": {
        "required": ["creator", "year", "title", "school"],
        "recommended": ["url"],
    },
    "online": {
        "required": ["creator", "year", "title", "url"],
        "recommended": [],
    },
    "default": {
        "required": ["creator", "year", "title"],
        "recommended": ["url"],
    },
}


def configure_logging(verbose: bool) -> None:
    """Configure logging output.

    Args:
        verbose: Whether to enable debug logging.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def strip_comments(text: str) -> str:
    """Strip LaTeX comments while preserving escaped percent signs.

    Args:
        text: LaTeX content.

    Returns:
        LaTeX content without comments.
    """

    return re.sub(r"(?<!\\)%.*", "", text)


def extract_citations_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract citation commands and key lists from text.

    Args:
        text: LaTeX content.

    Returns:
        List of (command, keys) tuples.
    """

    citations: List[Tuple[str, str]] = []
    for match in CITE_PATTERN.finditer(text):
        cmd = match.group("cmd")
        keys = match.group("keys")
        citations.append((cmd, keys))
    return citations


def parse_tex_files(tex_files: Iterable[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Parse TeX files and return citation occurrences by key.

    Args:
        tex_files: Iterable of TeX file paths.

    Returns:
        Dictionary keyed by citation key with occurrence metadata.
    """

    occurrences: Dict[str, List[Dict[str, Any]]] = {}
    for tex_file in tex_files:
        if not tex_file.exists():
            logging.warning("TeX file not found: %s", tex_file)
            continue
        with tex_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = strip_comments(line)
                for cmd, keys in extract_citations_from_text(line):
                    for key in split_keys(keys):
                        if not key:
                            continue
                        occurrences.setdefault(key, []).append(
                            {
                                "file": str(tex_file),
                                "line": line_number,
                                "command": cmd,
                            }
                        )
    return occurrences


def split_keys(keys: str) -> List[str]:
    """Split a citation key list into individual keys.

    Args:
        keys: Citation key list string.

    Returns:
        List of citation keys.
    """

    return [key.strip() for key in keys.split(",") if key.strip()]


def extract_all_citations(text: str) -> Tuple[Set[str], Set[str], bool]:
    """Extract citation keys, nocite keys, and wildcard status.

    Args:
        text: LaTeX content.

    Returns:
        Tuple of cited keys, nocite keys, and whether nocite{*} is present.
    """

    cited: Set[str] = set()
    nocite: Set[str] = set()
    nocite_all = False

    for cmd, keys in extract_citations_from_text(text):
        split = split_keys(keys)
        if cmd == "nocite":
            if "*" in split:
                nocite_all = True
            nocite.update(split)
        else:
            cited.update(split)

    return cited, nocite, nocite_all


def read_tex_content(tex_files: Iterable[Path]) -> str:
    """Read and concatenate TeX file content.

    Args:
        tex_files: Iterable of TeX file paths.

    Returns:
        Combined content string.
    """

    contents: List[str] = []
    for tex_file in tex_files:
        if not tex_file.exists():
            continue
        contents.append(strip_comments(tex_file.read_text(encoding="utf-8")))
    return "\n".join(contents)


def parse_bib_entries(bib_path: Path) -> List[Dict[str, Any]]:
    """Parse BibTeX entries from a file.

    Args:
        bib_path: Path to a .bib file.

    Returns:
        List of BibTeX entries with keys and fields.
    """

    if not bib_path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {bib_path}")

    text = bib_path.read_text(encoding="utf-8")
    return parse_bib_text(text)


def parse_bib_text(text: str) -> List[Dict[str, Any]]:
    """Parse raw BibTeX text into entry dictionaries.

    Args:
        text: Raw BibTeX file content.

    Returns:
        List of entry dictionaries containing type, key, and fields.
    """

    entries: List[Dict[str, Any]] = []
    index = 0
    length = len(text)

    while index < length:
        if text[index] != "@":
            index += 1
            continue

        entry_type, start_index = parse_entry_type(text, index + 1)
        if not entry_type:
            index += 1
            continue

        entry_type = entry_type.lower()
        index = start_index

        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] not in {"{", "("}:
            index += 1
            continue

        open_char = text[index]
        close_char = "}" if open_char == "{" else ")"
        index += 1

        body, index = extract_entry_body(text, index, open_char, close_char)
        if entry_type in IGNORE_BIB_TYPES:
            continue

        key, fields_text = split_key_fields(body)
        key = key.strip()
        if not key:
            logging.warning("Skipping entry without a key (type=%s)", entry_type)
            continue

        raw_fields = parse_bib_fields(fields_text)
        fields = clean_fields(raw_fields)

        entries.append(
            {
                "entry_type": entry_type,
                "key": key,
                "fields": fields,
            }
        )

    return entries


def parse_entry_type(text: str, start_index: int) -> Tuple[str, int]:
    """Parse the BibTeX entry type starting at a given index.

    Args:
        text: Raw BibTeX content.
        start_index: Index after the '@' character.

    Returns:
        Tuple of entry type string and index after the type.
    """

    index = start_index
    while index < len(text) and text[index].isalpha():
        index += 1
    return text[start_index:index], index


def extract_entry_body(
    text: str,
    start_index: int,
    open_char: str,
    close_char: str,
) -> Tuple[str, int]:
    """Extract the body of a BibTeX entry with balanced braces.

    Args:
        text: Raw BibTeX content.
        start_index: Index just after the opening brace/paren.
        open_char: Opening character for the entry body.
        close_char: Closing character for the entry body.

    Returns:
        Tuple of body string and index after the closing brace/paren.
    """

    depth = 1
    index = start_index
    in_quote = False
    escape = False

    while index < len(text) and depth > 0:
        char = text[index]
        if char == '"' and not escape:
            in_quote = not in_quote
        elif not in_quote:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
        escape = char == "\\" and not escape
        index += 1

    return text[start_index : index - 1].strip(), index


def split_key_fields(body: str) -> Tuple[str, str]:
    """Split a BibTeX entry body into key and fields text.

    Args:
        body: Entry body text without outer braces.

    Returns:
        Tuple of (key, fields_text).
    """

    depth = 0
    in_quote = False
    escape = False
    for index, char in enumerate(body):
        if char == '"' and not escape:
            in_quote = not in_quote
        elif not in_quote:
            if char == "{":
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)
        if char == "," and depth == 0 and not in_quote:
            return body[:index], body[index + 1 :]
        escape = char == "\\" and not escape

    return body, ""


def parse_bib_fields(fields_text: str) -> Dict[str, str]:
    """Parse BibTeX fields from the entry body.

    Args:
        fields_text: Raw fields section of a BibTeX entry.

    Returns:
        Dictionary of raw field values keyed by field name.
    """

    parts = split_top_level(fields_text, ",")
    fields: Dict[str, str] = {}

    for part in parts:
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip().lower()
        value = value.strip()
        if not name:
            continue
        fields[name] = value

    return fields


def split_top_level(text: str, delimiter: str) -> List[str]:
    """Split text on a delimiter, ignoring nested braces and quotes.

    Args:
        text: Input string to split.
        delimiter: Delimiter character.

    Returns:
        List of split parts.
    """

    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_quote = False
    escape = False

    for char in text:
        if char == '"' and not escape:
            in_quote = not in_quote
        elif not in_quote:
            if char == "{":
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)

        if char == delimiter and depth == 0 and not in_quote:
            segment = "".join(current).strip()
            if segment:
                parts.append(segment)
            current = []
        else:
            current.append(char)

        escape = char == "\\" and not escape

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return parts


def clean_fields(raw_fields: Dict[str, str]) -> Dict[str, str]:
    """Normalize field values and drop empty fields.

    Args:
        raw_fields: Raw field values keyed by field name.

    Returns:
        Cleaned field dictionary with empty values removed.
    """

    cleaned: Dict[str, str] = {}
    for name, value in raw_fields.items():
        normalized = normalize_field_value(value)
        if normalized:
            cleaned[name] = normalized
    return cleaned


def normalize_field_value(value: str) -> str:
    """Normalize a BibTeX field value by trimming wrappers and whitespace.

    Args:
        value: Raw field value.

    Returns:
        Normalized field value.
    """

    cleaned = value.strip()
    while cleaned:
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1].strip()
            continue
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
            continue
        break
    return cleaned.strip()


def canonicalize_entry_type(entry_type: str) -> str:
    """Map a BibTeX entry type to a canonical APA category.

    Args:
        entry_type: Raw BibTeX entry type.

    Returns:
        Canonical entry type.
    """

    return ENTRY_TYPE_CANONICAL.get(entry_type.lower(), "default")


def field_present(fields: Dict[str, str], field_name: str) -> bool:
    """Check if a field or field group is present.

    Args:
        fields: Cleaned field dictionary.
        field_name: Field or field-group name.

    Returns:
        True if any matching field is present.
    """

    if field_name == "doi_or_url":
        return field_present(fields, "doi") or field_present(fields, "url")

    candidates = FIELD_GROUPS.get(field_name, [field_name])
    return any(fields.get(candidate) for candidate in candidates)


def evaluate_apa_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a BibTeX entry against APA completeness rules.

    Args:
        entry: Entry dictionary with type, key, and fields.

    Returns:
        APA audit result for the entry.
    """

    entry_type = entry["entry_type"].lower()
    canonical = canonicalize_entry_type(entry_type)
    rules = APA_REQUIREMENTS.get(canonical, APA_REQUIREMENTS["default"])
    fields = entry["fields"]

    missing_required = [
        field for field in rules["required"] if not field_present(fields, field)
    ]
    missing_recommended = [
        field for field in rules["recommended"] if not field_present(fields, field)
    ]

    if missing_required:
        status = "missing_required"
    elif missing_recommended:
        status = "missing_recommended"
    else:
        status = "ok"

    notes: List[str] = []
    if canonical == "default":
        notes.append("unknown_entry_type")

    return {
        "citation_key": entry["key"],
        "entry_type": entry_type,
        "entry_type_canonical": canonical,
        "fields_present": sorted(fields.keys()),
        "fields": fields,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "status": status,
        "apa_edition": APA_EDITION,
        "notes": notes,
    }


def audit_apa_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate BibTeX entries for APA completeness.

    Args:
        entries: Parsed BibTeX entries.

    Returns:
        List of APA audit results.
    """

    return [evaluate_apa_entry(entry) for entry in entries]


def render_report_md(report: Dict[str, Any]) -> str:
    """Render a Markdown summary of the citation audit.

    Args:
        report: Citation audit report dictionary.

    Returns:
        Markdown string.
    """

    lines: List[str] = [
        "# Citation Audit Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        f"- TeX files scanned: {len(report['files_scanned'])}",
        f"- BibTeX entries: {report['summary']['bib_entry_count']}",
        f"- Citation keys found: {report['summary']['citation_key_count']}",
        f"- Missing in BibTeX: {report['summary']['missing_in_bib_count']}",
        f"- Uncited BibTeX entries: {report['summary']['uncited_bib_count']}",
        "",
        f"## APA {report['apa_edition']} Completeness",
        f"- Entries evaluated: {report['apa_audit']['entries_total']}",
        f"- Missing required fields: {report['apa_audit']['entries_missing_required']}",
        f"- Missing recommended fields: {report['apa_audit']['entries_missing_recommended']}",
        "",
    ]

    if report.get("notes"):
        lines.append("## Notes")
        for note in report["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Missing In BibTeX")
    if report["missing_in_bib"]:
        for item in report["missing_in_bib"]:
            lines.append(f"- {item['key']}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Uncited BibTeX Entries")
    if report["uncited_in_bib"]:
        for key in report["uncited_in_bib"]:
            lines.append(f"- {key}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## APA Entries Missing Required Fields")
    if report["apa_audit"]["entries_missing_required_keys"]:
        for key in report["apa_audit"]["entries_missing_required_keys"]:
            lines.append(f"- {key}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def gather_tex_files(base_dir: Path, extra_tex: List[Path]) -> List[Path]:
    """Collect TeX files to scan.

    Args:
        base_dir: Root directory containing main.tex.
        extra_tex: Additional TeX paths to include.

    Returns:
        List of TeX file paths.
    """

    tex_files: List[Path] = []
    for relative in ["main.tex", "preamble.tex"]:
        candidate = base_dir / relative
        if candidate.exists():
            tex_files.append(candidate)

    for subdir in ["sections", "revision_sections"]:
        path = base_dir / subdir
        if path.exists():
            tex_files.extend(sorted(path.rglob("*.tex")))

    tex_files.extend([path for path in extra_tex if path.exists()])
    return tex_files


def build_report(
    bib_keys: Set[str],
    cited_keys: Set[str],
    nocite_keys: Set[str],
    nocite_all: bool,
    occurrences: Dict[str, List[Dict[str, Any]]],
    files_scanned: List[str],
    apa_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a structured citation audit report.

    Args:
        bib_keys: Keys found in the BibTeX file.
        cited_keys: Keys found in the LaTeX content.
        nocite_keys: Keys referenced by \nocite.
        nocite_all: Whether \nocite{*} was found.
        occurrences: Citation occurrences with file/line metadata.
        files_scanned: List of scanned TeX files.
        apa_entries: APA audit results for BibTeX entries.

    Returns:
        Citation audit report dictionary.
    """

    missing_in_bib = sorted(cited_keys - bib_keys)

    if nocite_all:
        uncited_in_bib: List[str] = []
        notes = ["nocite{*} detected; uncited BibTeX entries not evaluated."]
    else:
        uncited_in_bib = sorted(bib_keys - cited_keys)
        notes = []

    missing_required_keys = [
        entry["citation_key"] for entry in apa_entries if entry["missing_required"]
    ]
    missing_recommended_keys = [
        entry["citation_key"]
        for entry in apa_entries
        if entry["missing_recommended"] and not entry["missing_required"]
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "apa_edition": APA_EDITION,
        "summary": {
            "bib_entry_count": len(bib_keys),
            "citation_key_count": len(cited_keys),
            "missing_in_bib_count": len(missing_in_bib),
            "uncited_bib_count": len(uncited_in_bib),
        },
        "missing_in_bib": [
            {"key": key, "occurrences": occurrences.get(key, [])}
            for key in missing_in_bib
        ],
        "uncited_in_bib": uncited_in_bib,
        "nocite_keys": sorted(nocite_keys),
        "nocite_all": nocite_all,
        "files_scanned": files_scanned,
        "notes": notes,
        "apa_audit": {
            "entries_total": len(apa_entries),
            "entries_missing_required": len(missing_required_keys),
            "entries_missing_recommended": len(missing_recommended_keys),
            "entries_missing_required_keys": missing_required_keys,
            "entries_missing_recommended_keys": missing_recommended_keys,
        },
    }

    return report


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to a JSONL file.

    Args:
        path: Output file path.
        records: Records to write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    """Run the citation audit."""

    parser = argparse.ArgumentParser(
        description="Check LaTeX citations against BibTeX entries"
    )
    parser.add_argument(
        "--tex-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Root directory containing main.tex and sections/",
    )
    parser.add_argument(
        "--bib-file",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "references.bib",
        help="BibTeX file path",
    )
    parser.add_argument(
        "--extra-tex",
        type=Path,
        nargs="*",
        default=[],
        help="Additional TeX files to scan",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "citation_audit_report.json",
        help="Output JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parent / "citation_audit_report.md",
        help="Output Markdown report",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "citation_entries.jsonl",
        help="Output JSONL entry audit",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if missing/uncited/APA-required fields are found",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(args.verbose)

    tex_root = args.tex_root
    bib_file = args.bib_file

    tex_files = gather_tex_files(tex_root, args.extra_tex)
    if not tex_files:
        raise FileNotFoundError(f"No TeX files found under {tex_root}")

    content = read_tex_content(tex_files)
    cited_keys, nocite_keys, nocite_all = extract_all_citations(content)
    occurrences = parse_tex_files(tex_files)

    entries = parse_bib_entries(bib_file)
    bib_keys = {entry["key"] for entry in entries}

    apa_entries = audit_apa_entries(entries)

    report = build_report(
        bib_keys=bib_keys,
        cited_keys=cited_keys,
        nocite_keys=nocite_keys,
        nocite_all=nocite_all,
        occurrences=occurrences,
        files_scanned=[str(path) for path in tex_files],
        apa_entries=apa_entries,
    )

    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_report_md(report), encoding="utf-8")
    write_jsonl(args.output_jsonl, apa_entries)

    logging.info("Wrote JSON report to %s", args.output_json)
    logging.info("Wrote Markdown report to %s", args.output_md)
    logging.info("Wrote JSONL entry audit to %s", args.output_jsonl)

    if args.strict:
        if report["summary"]["missing_in_bib_count"] > 0:
            raise SystemExit(2)
        if report["summary"]["uncited_bib_count"] > 0 and not report["nocite_all"]:
            raise SystemExit(3)
        if report["apa_audit"]["entries_missing_required"] > 0:
            raise SystemExit(4)


if __name__ == "__main__":
    main()
