#!/usr/bin/env python3
"""
Generate a polished PDF from the immigration policy demographic impact article.

Uses markdown + weasyprint to create a professional journal-style PDF.
"""

import re
from pathlib import Path

import markdown  # type: ignore[import-untyped]
from weasyprint import CSS, HTML

# Paths
DOCS_DIR = Path(__file__).parent.parent / "docs" / "research"
INPUT_FILE = DOCS_DIR / "2025_immigration_policy_demographic_impact.md"
OUTPUT_FILE = DOCS_DIR / "2025_immigration_policy_demographic_impact.pdf"

# Professional journal-style CSS
CSS_STYLE = """
@page {
    size: letter;
    margin: 1in 1in 1in 1in;
    @top-center {
        content: "The Demographic Shock of 2025";
        font-size: 9pt;
        color: #666;
        font-style: italic;
    }
    @bottom-center {
        content: counter(page);
        font-size: 10pt;
    }
}

@page:first {
    @top-center {
        content: none;
    }
}

body {
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}

h1 {
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 24pt;
    line-height: 1.3;
    color: #000;
}

h2 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 24pt;
    margin-bottom: 12pt;
    color: #000;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4pt;
}

h3 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 10pt;
    color: #222;
}

h4 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 8pt;
    color: #333;
}

p {
    margin-bottom: 10pt;
    text-indent: 0;
}

/* First paragraph after heading - no indent */
h1 + p, h2 + p, h3 + p, h4 + p {
    text-indent: 0;
}

/* Subsequent paragraphs - indent */
p + p {
    text-indent: 24pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 16pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #333;
    padding: 6pt 8pt;
    text-align: left;
    vertical-align: top;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

ul, ol {
    margin: 10pt 0;
    padding-left: 24pt;
}

li {
    margin-bottom: 6pt;
}

blockquote {
    margin: 16pt 24pt;
    padding-left: 12pt;
    border-left: 3px solid #ccc;
    font-style: italic;
    color: #444;
}

a {
    color: #0066cc;
    text-decoration: none;
}

/* Citation superscripts */
sup {
    font-size: 8pt;
    vertical-align: super;
    line-height: 0;
}

/* Works cited section */
h4:last-of-type ~ ol {
    font-size: 9pt;
    line-height: 1.4;
}

/* Inline subheadings (bolded paragraph starts) */
strong:first-child {
    display: inline;
}

/* Image handling */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16pt auto;
}

/* Abstract/Executive Summary styling */
h2:first-of-type + p {
    font-style: italic;
}

/* Page breaks before major sections */
h2 {
    page-break-before: auto;
    page-break-after: avoid;
}

h3, h4 {
    page-break-after: avoid;
}

/* Keep tables and figures together */
table, figure, img {
    page-break-inside: avoid;
}
"""


def clean_markdown(content: str) -> str:
    """Clean up markdown formatting issues."""

    # Remove escaped periods (artifact from conversion)
    content = re.sub(r"(\d)\\\.", r"\1.", content)

    # Remove bold from headers (they're already styled)
    content = re.sub(r"^(#+)\s*\*\*(.+?)\*\*\s*$", r"\1 \2", content, flags=re.MULTILINE)

    # Fix inline subheadings that run together with following text
    # Pattern: Title Case Text followed directly by more text without newline
    # e.g., "The Asylum Suspension and Processing Pauses\nIn a decisive move..."
    # These should become proper h4 headers

    # First, identify lines that look like subheadings (Title Case, short, followed by paragraph)
    lines = content.split("\n")
    cleaned_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a standalone line that looks like a subheading
        # (Title case, reasonable length, not a list item, not already a header)
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.startswith("*")
            and not stripped.startswith("-")
            and not stripped.startswith("|")
            and len(stripped) < 80
            and len(stripped) > 10
            and
            # Title case check: first letter cap, contains spaces, most words start with caps
            stripped[0].isupper()
            and " " in stripped
            and not stripped.endswith(".")
            and not stripped.endswith(",")
            and
            # Check if it looks like a title (multiple capitalized words)
            sum(1 for word in stripped.split() if word[0].isupper()) >= len(stripped.split()) * 0.6
        ):
            # Check context: previous line empty, next line is content
            prev_empty = (i == 0) or (cleaned_lines and cleaned_lines[-1].strip() == "")
            next_is_content = (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].strip().startswith("#")
            )

            if prev_empty and next_is_content:
                # This is a subheading - make it h4
                cleaned_lines.append(f"#### {stripped}")
                cleaned_lines.append("")  # Add blank line after header
                i += 1
                continue

        cleaned_lines.append(line)
        i += 1

    content = "\n".join(cleaned_lines)

    # Handle image references - remove broken image refs for now
    # They reference base64 data at the end which may not render properly
    content = re.sub(r"!\[\]\[image\d+\]", "", content)

    # Clean up the base64 image definitions at the end (they're huge and may cause issues)
    # Keep the document cleaner by removing these for the PDF
    content = re.sub(r"\[image\d+\]:\s*<data:image/png;base64,[^>]+>", "", content)

    return content


def convert_to_html(md_content: str) -> str:
    """Convert markdown to HTML."""
    md = markdown.Markdown(
        extensions=[
            "tables",
            "toc",
            "smarty",  # Smart quotes
        ]
    )
    html_body = md.convert(md_content)

    # Wrap in full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>The Demographic Shock of 2025</title>
</head>
<body>
{html_body}
</body>
</html>
"""
    return html


def generate_pdf(html_content: str, output_path: Path) -> None:
    """Generate PDF from HTML using WeasyPrint."""
    html = HTML(string=html_content)
    css = CSS(string=CSS_STYLE)
    html.write_pdf(output_path, stylesheets=[css])
    print(f"PDF generated: {output_path}")


def main():
    """Main entry point."""
    print(f"Reading: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Read markdown
    md_content = INPUT_FILE.read_text(encoding="utf-8")
    print(f"Read {len(md_content):,} characters")

    # Clean up formatting
    print("Cleaning markdown...")
    cleaned_md = clean_markdown(md_content)

    # Convert to HTML
    print("Converting to HTML...")
    html_content = convert_to_html(cleaned_md)

    # Generate PDF
    print("Generating PDF...")
    generate_pdf(html_content, OUTPUT_FILE)

    print(f"\nSuccess! PDF saved to:\n  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
