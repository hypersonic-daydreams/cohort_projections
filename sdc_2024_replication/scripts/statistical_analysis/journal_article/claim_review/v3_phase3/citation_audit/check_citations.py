#!/usr/bin/env python3
"""Check LaTeX citations against BibTeX entries and APA 7th completeness."""

from __future__ import annotations

import argparse
import json
import logging
import re
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

APA_EDITION = "7th"

INCLUDE_PATTERN = re.compile(r"\\(?P<cmd>input|include|subfile)\s*\{(?P<path>[^}]+)\}")
IMPORT_PATTERN = re.compile(
    r"\\(?P<cmd>import|subimport)\s*\{(?P<dir>[^}]+)\}\s*\{(?P<path>[^}]+)\}"
)

IGNORE_BIB_TYPES = {"comment", "preamble"}
EXCLUDED_CITE_COMMANDS = {"setcitestyle", "citestyle"}

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


def build_cite_pattern(extra_commands: Sequence[str]) -> re.Pattern[str]:
    """Build a citation regex pattern including extra command names.

    Args:
        extra_commands: Additional LaTeX command names to treat as citations.

    Returns:
        Compiled regex pattern.
    """

    normalized: List[str] = []
    for command in extra_commands:
        command = command.strip()
        if not command:
            continue
        if command.startswith("\\"):
            command = command[1:]
        normalized.append(re.escape(command))

    if normalized:
        command_pattern = r"(?:[A-Za-z]*cite[a-zA-Z*]*|" + "|".join(normalized) + ")"
    else:
        command_pattern = r"[A-Za-z]*cite[a-zA-Z*]*"

    return re.compile(
        rf"\\(?P<cmd>{command_pattern})\s*(?:\[[^\]]*\]\s*)*\{{(?P<keys>[^}}]*)\}}",
        re.DOTALL,
    )


def build_line_index(text: str) -> List[int]:
    """Build a list of line start offsets for a text blob.

    Args:
        text: Text content to index.

    Returns:
        List of 0-based line start offsets.
    """

    return [0] + [match.end() for match in re.finditer(r"\n", text)]


def line_number_for_index(index: int, line_starts: List[int]) -> int:
    """Convert a string index to a 1-based line number.

    Args:
        index: Character index in the text.
        line_starts: Line start offsets for the text.

    Returns:
        1-based line number.
    """

    return bisect_right(line_starts, index)


def resolve_tex_path(raw_path: str, base_dir: Path, tex_root: Path) -> Path | None:
    """Resolve a TeX include path to an existing file.

    Args:
        raw_path: Path value from an include directive.
        base_dir: Directory of the file containing the include.
        tex_root: Root TeX directory used as a fallback.

    Returns:
        Resolved Path if found, otherwise None.
    """

    path = Path(raw_path)
    candidates: List[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(base_dir / path)
        candidates.append(tex_root / path)

    for candidate in candidates:
        if candidate.suffix:
            if candidate.exists():
                return candidate.resolve()
            continue
        with_suffix = candidate.with_suffix(".tex")
        if with_suffix.exists():
            return with_suffix.resolve()
        if candidate.exists():
            return candidate.resolve()

    return None


def extract_includes(text: str, base_dir: Path, tex_root: Path) -> List[Path]:
    """Extract included TeX file paths from a TeX document.

    Args:
        text: TeX content with comments stripped.
        base_dir: Directory of the TeX file being scanned.
        tex_root: Root directory for resolving includes.

    Returns:
        List of resolved include Paths.
    """

    includes: List[Path] = []
    for match in INCLUDE_PATTERN.finditer(text):
        raw_path = match.group("path").strip()
        resolved = resolve_tex_path(raw_path, base_dir, tex_root)
        if resolved is None:
            logging.warning(
                "Included TeX file not found: %s (from %s)", raw_path, base_dir
            )
            continue
        includes.append(resolved)

    for match in IMPORT_PATTERN.finditer(text):
        dir_part = match.group("dir").strip()
        file_part = match.group("path").strip()
        import_base = Path(dir_part)
        if not import_base.is_absolute():
            import_base = base_dir / import_base
        combined = import_base / file_part
        resolved = resolve_tex_path(str(combined), base_dir, tex_root)
        if resolved is None:
            logging.warning(
                "Imported TeX file not found: %s/%s (from %s)",
                dir_part,
                file_part,
                base_dir,
            )
            continue
        includes.append(resolved)

    return includes


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


def extract_citations_from_text(
    text: str, cite_pattern: re.Pattern[str]
) -> List[Tuple[str, str]]:
    """Extract citation commands and key lists from text.

    Args:
        text: LaTeX content.
        cite_pattern: Compiled regex used to locate citation commands.

    Returns:
        List of (command, keys) tuples.
    """

    citations: List[Tuple[str, str]] = []
    for match in cite_pattern.finditer(text):
        cmd = match.group("cmd")
        if cmd in EXCLUDED_CITE_COMMANDS:
            continue
        keys = match.group("keys")
        citations.append((cmd, keys))
    return citations


def parse_tex_files(
    tex_files: Iterable[Path], cite_pattern: re.Pattern[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse TeX files and return citation occurrences by key.

    Args:
        tex_files: Iterable of TeX file paths.
        cite_pattern: Compiled regex used to locate citation commands.

    Returns:
        Dictionary keyed by citation key with occurrence metadata.
    """

    occurrences: Dict[str, List[Dict[str, Any]]] = {}
    for tex_file in tex_files:
        if not tex_file.exists():
            logging.warning("TeX file not found: %s", tex_file)
            continue
        text = strip_comments(tex_file.read_text(encoding="utf-8"))
        line_starts = build_line_index(text)
        for match in cite_pattern.finditer(text):
            cmd = match.group("cmd")
            keys = match.group("keys")
            line_number = line_number_for_index(match.start(), line_starts)
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


def extract_all_citations(
    text: str, cite_pattern: re.Pattern[str]
) -> Tuple[Set[str], Set[str], bool]:
    """Extract citation keys, nocite keys, and wildcard status.

    Args:
        text: LaTeX content.
        cite_pattern: Compiled regex used to locate citation commands.

    Returns:
        Tuple of cited keys, nocite keys, and whether nocite{*} is present.
    """

    cited: Set[str] = set()
    nocite: Set[str] = set()
    nocite_all = False

    for cmd, keys in extract_citations_from_text(text, cite_pattern):
        split = split_keys(keys)
        if cmd == "nocite":
            if "*" in split:
                nocite_all = True
                split = [key for key in split if key != "*"]
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


def parse_bib_entries(bib_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Parse BibTeX entries from a file.

    Args:
        bib_path: Path to a .bib file.

    Returns:
        Tuple of BibTeX entries with keys/fields and string macros.
    """

    if not bib_path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {bib_path}")

    text = bib_path.read_text(encoding="utf-8")
    return parse_bib_text(text)


def parse_bib_text(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Parse raw BibTeX text into entry dictionaries.

    Args:
        text: Raw BibTeX file content.

    Returns:
        Tuple of entry dictionaries containing type, key, and fields, plus string macros.
    """

    raw_entries: List[Dict[str, Any]] = []
    string_macros: Dict[str, str] = {}
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
        if entry_type == "string":
            string_macros.update(parse_string_macros(body))
            continue
        if entry_type in IGNORE_BIB_TYPES:
            continue

        key, fields_text = split_key_fields(body)
        key = key.strip()
        if not key:
            logging.warning("Skipping entry without a key (type=%s)", entry_type)
            continue

        raw_fields = parse_bib_fields(fields_text)
        raw_entries.append(
            {
                "entry_type": entry_type,
                "key": key,
                "fields": raw_fields,
            }
        )

    entries: List[Dict[str, Any]] = []
    for entry in raw_entries:
        fields = clean_fields(entry["fields"], string_macros)
        entries.append(
            {
                "entry_type": entry["entry_type"],
                "key": entry["key"],
                "fields": fields,
            }
        )

    return entries, string_macros


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


def parse_string_macros(body: str) -> Dict[str, str]:
    """Parse @string macro definitions from a BibTeX entry body.

    Args:
        body: Raw @string entry body.

    Returns:
        Dictionary of macro name to value.
    """

    raw_fields = parse_bib_fields(body)
    macros: Dict[str, str] = {}
    for name, value in raw_fields.items():
        normalized = normalize_field_value(value)
        if normalized:
            macros[name.lower()] = normalized
    return macros


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


def clean_fields(
    raw_fields: Dict[str, str], string_macros: Dict[str, str]
) -> Dict[str, str]:
    """Normalize field values and drop empty fields.

    Args:
        raw_fields: Raw field values keyed by field name.
        string_macros: Parsed @string macros for expansion.

    Returns:
        Cleaned field dictionary with empty values removed.
    """

    cleaned: Dict[str, str] = {}
    for name, value in raw_fields.items():
        expanded = expand_bib_value(value, string_macros)
        if expanded:
            cleaned[name] = expanded
    return cleaned


def expand_bib_value(value: str, string_macros: Dict[str, str]) -> str:
    """Expand BibTeX values with @string macros and concatenation.

    Args:
        value: Raw BibTeX field value.
        string_macros: Parsed @string macros for expansion.

    Returns:
        Expanded field value.
    """

    parts = split_top_level(value, "#")
    expanded_parts: List[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        normalized = normalize_field_value(part)
        if not normalized:
            continue
        macro_key = normalized.lower()
        expanded_parts.append(string_macros.get(macro_key, normalized))

    return "".join(expanded_parts).strip()


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
        f"- Duplicate BibTeX keys: {report['summary']['duplicate_bib_count']}",
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

    lines.append("## Duplicate BibTeX Keys")
    if report.get("duplicate_bib_keys"):
        for key in report["duplicate_bib_keys"]:
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

    fixes = report.get("fixes", {})
    applied = fixes.get("applied", [])
    unmatched = fixes.get("unmatched", [])
    if applied or unmatched:
        lines.append("")
        lines.append("## Fixes Applied")
        if applied:
            for key in applied:
                lines.append(f"- {key}")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("## Fixes Unmatched")
        if unmatched:
            for key in unmatched:
                lines.append(f"- {key}")
        else:
            lines.append("- None")

    return "\n".join(lines) + "\n"


def html_escape(text: str) -> str:
    """Escape text for HTML rendering."""

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def get_field_values(fields: Dict[str, str], field_name: str) -> List[str]:
    """Collect values for a field group or field name."""

    candidates = FIELD_GROUPS.get(field_name, [field_name])
    values = [fields.get(candidate, "").strip() for candidate in candidates]
    return [value for value in values if value]


def render_field_span(
    label: str,
    values: List[str],
    required: bool,
) -> str:
    """Render a field line with highlighting for missing data."""

    if values:
        rendered = "; ".join(html_escape(value) for value in values)
        return (
            f'<span class="field-label">{html_escape(label)}:</span> '
            f'<span class="field-value ok">{rendered}</span>'
        )

    placeholder = "&nbsp;" * 10
    missing_class = "missing-required" if required else "missing-recommended"
    missing_label = "REQUIRED" if required else "RECOMMENDED"
    return (
        f'<span class="field-label">{html_escape(label)}:</span> '
        f'<span class="field-value {missing_class}">{placeholder}</span> '
        f'<span class="missing-note">{missing_label}</span>'
    )


def render_reference_entry(entry: Dict[str, Any]) -> str:
    """Render a reference entry section with highlighted field status."""

    entry_type = entry["entry_type"].lower()
    canonical = entry["entry_type_canonical"]
    fields = entry["fields"]
    status = entry["status"]
    rules = APA_REQUIREMENTS.get(canonical, APA_REQUIREMENTS["default"])

    status_class = {
        "ok": "status-ok",
        "missing_required": "status-missing-required",
        "missing_recommended": "status-missing-recommended",
    }.get(status, "status-unknown")

    lines = [
        f"<div class=\"entry\" data-key=\"{html_escape(entry['citation_key'])}\">",
        f'<div class="entry-header {status_class}">',
        f"<span class=\"entry-key\">{html_escape(entry['citation_key'])}</span>",
        f'<span class="entry-type">{html_escape(entry_type)}</span>',
        f'<span class="entry-canonical">{html_escape(canonical)}</span>',
        "</div>",
        '<div class="entry-fields">',
        '<div class="entry-section-title">Required fields</div>',
    ]

    for field_name in rules["required"]:
        label = field_name.replace("_", " ").title()
        values = get_field_values(fields, field_name)
        lines.append(
            f'<div class="entry-field">{render_field_span(label, values, True)}</div>'
        )

    lines.append('<div class="entry-section-title">Recommended fields</div>')
    for field_name in rules["recommended"]:
        label = field_name.replace("_", " ").title()
        values = get_field_values(fields, field_name)
        lines.append(
            f'<div class="entry-field">{render_field_span(label, values, False)}</div>'
        )

    lines.extend(["</div>", "</div>"])
    return "\n".join(lines)


def load_line_excerpt(path: Path, line_number: int) -> str:
    """Load a single line from a file for context."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    if line_number <= 0 or line_number > len(lines):
        return ""
    return lines[line_number - 1].strip()


def render_report_html(
    report: Dict[str, Any],
    apa_entries: List[Dict[str, Any]],
) -> str:
    """Render a highlighted HTML summary of the citation audit."""

    uncited = set(report.get("uncited_in_bib", []))

    entry_blocks: List[str] = []
    for entry in apa_entries:
        block = render_reference_entry(entry)
        if entry["citation_key"] in uncited:
            block = block.replace(
                'class="entry"',
                'class="entry entry-uncited"',
                1,
            )
        entry_blocks.append(block)

    missing_blocks: List[str] = []
    for item in report.get("missing_in_bib", []):
        key = item["key"]
        occurrences = item.get("occurrences", [])
        occ_lines = []
        for occ in occurrences:
            file_path = Path(occ["file"])
            line = int(occ.get("line", 0))
            snippet = load_line_excerpt(file_path, line)
            occ_lines.append(
                "<div class=\"occurrence\">"
                f"<span class=\"occurrence-file\">{html_escape(occ['file'])}</span>"
                f"<span class=\"occurrence-line\">line {line}</span>"
                f"<span class=\"occurrence-cmd\">{html_escape(occ.get('command', ''))}</span>"
                f"<span class=\"occurrence-snippet\">{html_escape(snippet)}</span>"
                "</div>"
            )
        missing_blocks.append(
            '<div class="missing-entry">'
            f'<div class="missing-key">{html_escape(key)}</div>'
            + "\n".join(occ_lines)
            + "</div>"
        )

    duplicate_blocks = [
        f'<div class="duplicate-key">{html_escape(key)}</div>'
        for key in report.get("duplicate_bib_keys", [])
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Citation Audit Report</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      color: #1b1b1b;
      margin: 24px;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .summary {{
      background: #f5f5f5;
      padding: 12px;
      border-radius: 6px;
    }}
    .summary span {{
      display: inline-block;
      margin-right: 16px;
    }}
    .entry {{
      border: 1px solid #d0d0d0;
      border-radius: 6px;
      margin: 12px 0;
      padding: 12px;
    }}
    .entry-uncited {{
      background: #f8f8f8;
      border-color: #c0c0c0;
    }}
    .entry-header {{
      display: flex;
      gap: 12px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .entry-key {{
      font-family: "Courier New", Courier, monospace;
    }}
    .entry-section-title {{
      margin-top: 8px;
      font-weight: 600;
      color: #333;
    }}
    .entry-field {{
      margin: 4px 0;
    }}
    .field-label {{
      font-weight: 600;
    }}
    .field-value {{
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 4px;
    }}
    .field-value.ok {{
      background: #e8f5e9;
    }}
    .field-value.missing-required {{
      background: #ffebee;
    }}
    .field-value.missing-recommended {{
      background: #fff3e0;
    }}
    .missing-note {{
      margin-left: 6px;
      font-size: 0.85em;
      color: #666;
    }}
    .status-ok {{ color: #2e7d32; }}
    .status-missing-required {{ color: #c62828; }}
    .status-missing-recommended {{ color: #ef6c00; }}
    .missing-entry {{
      border-left: 4px solid #c62828;
      padding: 8px 12px;
      margin-bottom: 12px;
      background: #fff8f8;
    }}
    .missing-key {{
      font-weight: 600;
      margin-bottom: 6px;
      font-family: "Courier New", Courier, monospace;
    }}
    .occurrence {{
      margin-bottom: 6px;
      font-size: 0.9em;
    }}
    .occurrence-file {{
      display: block;
      color: #444;
    }}
    .occurrence-line,
    .occurrence-cmd {{
      margin-right: 10px;
      color: #666;
    }}
    .occurrence-snippet {{
      display: block;
      margin-top: 4px;
      font-family: "Courier New", Courier, monospace;
      background: #f5f5f5;
      padding: 4px 6px;
      border-radius: 4px;
    }}
    .duplicate-key {{
      padding: 4px 6px;
      background: #fce4ec;
      border-radius: 4px;
      margin-bottom: 6px;
      font-family: "Courier New", Courier, monospace;
    }}
  </style>
</head>
<body>
  <h1>Citation Audit Report</h1>
  <div class="summary">
    <span>Generated: {html_escape(report['generated_at'])}</span>
    <span>BibTeX entries: {report['summary']['bib_entry_count']}</span>
    <span>Citation keys: {report['summary']['citation_key_count']}</span>
    <span>Missing in BibTeX: {report['summary']['missing_in_bib_count']}</span>
    <span>Uncited entries: {report['summary']['uncited_bib_count']}</span>
    <span>Duplicate keys: {report['summary']['duplicate_bib_count']}</span>
    <span>APA missing required: {report['apa_audit']['entries_missing_required']}</span>
    <span>APA missing recommended: {report['apa_audit']['entries_missing_recommended']}</span>
  </div>

  <h2>Missing In BibTeX</h2>
  {"".join(missing_blocks) if missing_blocks else "<div>None</div>"}

  <h2>Duplicate BibTeX Keys</h2>
  {"".join(duplicate_blocks) if duplicate_blocks else "<div>None</div>"}

  <h2>Reference Entry Details</h2>
  {"".join(entry_blocks)}
</body>
</html>
"""


def collect_tex_files(start_files: Iterable[Path], tex_root: Path) -> List[Path]:
    """Collect TeX files by following include directives.

    Args:
        start_files: Initial TeX files to scan.
        tex_root: Root directory for resolving includes.

    Returns:
        List of resolved TeX file paths.
    """

    visited: Set[Path] = set()
    queue: List[Path] = list(start_files)

    while queue:
        path = queue.pop()
        if not path.exists():
            logging.warning("TeX file not found: %s", path)
            continue
        resolved = path.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)

        text = strip_comments(resolved.read_text(encoding="utf-8"))
        for include_path in extract_includes(text, resolved.parent, tex_root):
            if include_path not in visited:
                queue.append(include_path)

    return sorted(visited, key=lambda item: str(item))


def gather_tex_files(
    tex_root: Path,
    extra_tex: List[Path],
    main_tex: List[Path],
    scan_all: bool,
) -> List[Path]:
    """Collect TeX files to scan.

    Args:
        tex_root: Root directory containing main.tex.
        extra_tex: Additional TeX paths to include.
        main_tex: Explicit main TeX file paths.
        scan_all: Whether to scan all .tex files under tex_root.

    Returns:
        List of TeX file paths.
    """

    start_files: List[Path] = []
    existing_main = [path for path in main_tex if path.exists()]
    if main_tex and not existing_main:
        logging.warning("No provided --main-tex files exist; falling back to scan-all.")
        scan_all = True
    start_files.extend(existing_main)

    if not main_tex:
        default_main = tex_root / "main.tex"
        if default_main.exists():
            start_files.append(default_main)

    preamble = tex_root / "preamble.tex"
    if preamble.exists():
        start_files.append(preamble)

    start_files.extend([path for path in extra_tex if path.exists()])

    if scan_all or not start_files:
        tex_files = {path.resolve() for path in tex_root.rglob("*.tex")}
        tex_files.update(path.resolve() for path in extra_tex if path.exists())
        return sorted(tex_files, key=lambda item: str(item))

    return collect_tex_files(start_files, tex_root)


def build_report(
    bib_keys: Set[str],
    cited_keys: Set[str],
    nocite_keys: Set[str],
    nocite_all: bool,
    occurrences: Dict[str, List[Dict[str, Any]]],
    files_scanned: List[str],
    apa_entries: List[Dict[str, Any]],
    duplicate_bib_keys: List[str],
    fixes_applied: List[str],
    fixes_unmatched: List[str],
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
        duplicate_bib_keys: Duplicate keys found in the BibTeX entries.
        fixes_applied: Fix records applied to entries.
        fixes_unmatched: Fix records that did not match any entry.

    Returns:
        Citation audit report dictionary.
    """

    mentioned_keys = cited_keys | nocite_keys
    missing_in_bib = sorted(mentioned_keys - bib_keys)

    if nocite_all:
        uncited_in_bib: List[str] = []
        notes = ["nocite{*} detected; uncited BibTeX entries not evaluated."]
    else:
        uncited_in_bib = sorted(bib_keys - mentioned_keys)
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
            "duplicate_bib_count": len(duplicate_bib_keys),
        },
        "missing_in_bib": [
            {"key": key, "occurrences": occurrences.get(key, [])}
            for key in missing_in_bib
        ],
        "uncited_in_bib": uncited_in_bib,
        "duplicate_bib_keys": duplicate_bib_keys,
        "nocite_keys": sorted(nocite_keys),
        "nocite_all": nocite_all,
        "files_scanned": files_scanned,
        "notes": notes,
        "fixes": {
            "applied": sorted(set(fixes_applied)),
            "unmatched": sorted(set(fixes_unmatched)),
        },
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read records from a JSONL file.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of parsed records.
    """

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def apply_fixes(
    entries: List[Dict[str, Any]], fixes: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Apply fix records to BibTeX entries.

    Args:
        entries: Parsed BibTeX entries to mutate in place.
        fixes: Fix records with citation_key and field updates.

    Returns:
        Tuple of (applied_keys, unmatched_keys).
    """

    entries_by_key = {entry["key"]: entry for entry in entries}
    applied: List[str] = []
    unmatched: List[str] = []

    for fix in fixes:
        key = str(fix.get("citation_key", "")).strip()
        if not key:
            continue
        entry = entries_by_key.get(key)
        if entry is None:
            unmatched.append(key)
            continue

        entry_type = fix.get("entry_type")
        if entry_type:
            entry["entry_type"] = str(entry_type).lower()

        fields = entry.get("fields", {})

        for field in fix.get("remove_fields", []) or []:
            fields.pop(str(field).lower(), None)

        for name, value in (fix.get("fields") or {}).items():
            if value is None:
                continue
            normalized_value = normalize_field_value(str(value))
            if not normalized_value:
                continue
            fields[str(name).lower()] = normalized_value

        entry["fields"] = fields
        applied.append(key)

    return applied, unmatched


def find_duplicate_bib_keys(entries: List[Dict[str, Any]]) -> List[str]:
    """Find duplicate BibTeX citation keys.

    Args:
        entries: Parsed BibTeX entries.

    Returns:
        Sorted list of duplicate keys.
    """

    counts = Counter(entry["key"] for entry in entries)
    return sorted([key for key, count in counts.items() if count > 1])


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
        "--main-tex",
        type=Path,
        nargs="*",
        default=[],
        help="Explicit main TeX files to scan for includes",
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
        "--scan-all",
        action="store_true",
        help="Scan all .tex files under the TeX root",
    )
    parser.add_argument(
        "--extra-cite-commands",
        nargs="*",
        default=[],
        help="Additional citation command names (without leading backslash)",
    )
    parser.add_argument(
        "--fixes-file",
        type=Path,
        help="Optional JSONL file with citation fixes to apply",
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
        "--output-html",
        type=Path,
        default=Path(__file__).resolve().parent / "citation_audit_report.html",
        help="Output HTML report with highlighted issues",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if missing/uncited/APA-required fields or duplicate "
            "BibTeX keys are found"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(args.verbose)

    tex_root = args.tex_root
    bib_file = args.bib_file

    cite_pattern = build_cite_pattern(args.extra_cite_commands)
    tex_files = gather_tex_files(tex_root, args.extra_tex, args.main_tex, args.scan_all)
    if not tex_files:
        raise FileNotFoundError(f"No TeX files found under {tex_root}")

    content = read_tex_content(tex_files)
    cited_keys, nocite_keys, nocite_all = extract_all_citations(content, cite_pattern)
    occurrences = parse_tex_files(tex_files, cite_pattern)

    entries, _ = parse_bib_entries(bib_file)

    fixes_applied: List[str] = []
    fixes_unmatched: List[str] = []
    if args.fixes_file:
        if not args.fixes_file.exists():
            raise FileNotFoundError(f"Fixes file not found: {args.fixes_file}")
        fixes = read_jsonl(args.fixes_file)
        fixes_applied, fixes_unmatched = apply_fixes(entries, fixes)

    bib_keys = {entry["key"] for entry in entries}
    duplicate_bib_keys = find_duplicate_bib_keys(entries)

    apa_entries = audit_apa_entries(entries)

    report = build_report(
        bib_keys=bib_keys,
        cited_keys=cited_keys,
        nocite_keys=nocite_keys,
        nocite_all=nocite_all,
        occurrences=occurrences,
        files_scanned=[str(path) for path in tex_files],
        apa_entries=apa_entries,
        duplicate_bib_keys=duplicate_bib_keys,
        fixes_applied=fixes_applied,
        fixes_unmatched=fixes_unmatched,
    )

    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_report_md(report), encoding="utf-8")
    args.output_html.write_text(
        render_report_html(report, apa_entries), encoding="utf-8"
    )
    write_jsonl(args.output_jsonl, apa_entries)

    logging.info("Wrote JSON report to %s", args.output_json)
    logging.info("Wrote Markdown report to %s", args.output_md)
    logging.info("Wrote HTML report to %s", args.output_html)
    logging.info("Wrote JSONL entry audit to %s", args.output_jsonl)

    if args.strict:
        if report["summary"]["missing_in_bib_count"] > 0:
            raise SystemExit(2)
        if report["summary"]["uncited_bib_count"] > 0 and not report["nocite_all"]:
            raise SystemExit(3)
        if report["apa_audit"]["entries_missing_required"] > 0:
            raise SystemExit(4)
        if report["summary"]["duplicate_bib_count"] > 0:
            raise SystemExit(5)


if __name__ == "__main__":
    main()
