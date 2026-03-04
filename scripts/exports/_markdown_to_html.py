"""
Shared utilities for converting methodology Markdown documents to standalone HTML.

Provides the CSS theme, HTML wrapper, and Markdown-to-HTML conversion pipeline
used by both ``build_methodology_html.py`` and
``build_methodology_comparison_html.py``.

Design choices:
    - Uses the ``markdown`` library with tables, fenced_code, toc, smarty
      extensions for robust Markdown-to-HTML conversion.
    - LaTeX math is rendered client-side via MathJax 3 CDN (the only external
      dependency in the output HTML).  Inline ``$...$`` and display ``$$...$$``
      are both supported.
    - CSS is fully embedded (no external stylesheets) for self-contained output.
    - The visual theme mirrors the project's interactive report:
      Navy #1F3864, Blue #0563C1, Teal #00B0F0, Segoe UI font stack.
    - A collapsible table of contents is generated from Markdown headings.
    - Print-friendly styles via @media print.

Dependencies:
    - markdown (pip package, version 3.x)
"""

from __future__ import annotations

import re

import markdown


# ---------------------------------------------------------------------------
# Markdown → HTML Conversion
# ---------------------------------------------------------------------------
def convert_markdown_to_html(md_text: str) -> tuple[str, str]:
    """Convert Markdown text to HTML body content and a TOC string.

    Parameters
    ----------
    md_text : str
        Raw Markdown source text.

    Returns
    -------
    tuple[str, str]
        (html_body, toc_html) where html_body is the converted content and
        toc_html is the table-of-contents markup generated from headings.
    """
    # Protect LaTeX math from Markdown processing.
    # We replace $$...$$ and $...$ with placeholders, convert Markdown,
    # then restore them.
    math_blocks: list[str] = []
    math_inlines: list[str] = []

    def _save_block(m: re.Match) -> str:
        math_blocks.append(m.group(0))
        return f"\x00MATHBLOCK{len(math_blocks) - 1}\x00"

    def _save_inline(m: re.Match) -> str:
        math_inlines.append(m.group(0))
        return f"\x00MATHINLINE{len(math_inlines) - 1}\x00"

    # Display math first (greedy across lines)
    protected = re.sub(
        r"\$\$(.+?)\$\$", _save_block, md_text, flags=re.DOTALL
    )
    # Inline math (no newlines allowed inside)
    protected = re.sub(r"\$([^\$\n]+?)\$", _save_inline, protected)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "smarty"],
        extension_configs={
            "toc": {
                "title": "",
                "toc_depth": "2-3",
                "permalink": False,
            },
        },
    )
    html_body = md.convert(protected)
    toc_html = getattr(md, "toc", "")

    # Restore math placeholders
    for i, block in enumerate(math_blocks):
        html_body = html_body.replace(f"\x00MATHBLOCK{i}\x00", block)
        toc_html = toc_html.replace(f"\x00MATHBLOCK{i}\x00", block)
    for i, inline in enumerate(math_inlines):
        html_body = html_body.replace(f"\x00MATHINLINE{i}\x00", inline)
        toc_html = toc_html.replace(f"\x00MATHINLINE{i}\x00", inline)

    return html_body, toc_html


# ---------------------------------------------------------------------------
# CSS Theme (Embedded)
# ---------------------------------------------------------------------------
DOCUMENT_CSS = r"""
/* ============================================================
   CSS Reset & Base
   ============================================================ */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    scroll-behavior: smooth;
    font-size: 15px;
}

body {
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #333;
    background: #f8f9fa;
    line-height: 1.7;
}

/* ============================================================
   Header
   ============================================================ */
.doc-header {
    background: linear-gradient(135deg, #1F3864 0%, #0563C1 100%);
    color: white;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
}

.doc-header h1 {
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}

.doc-header .doc-meta {
    font-size: 0.9rem;
    opacity: 0.85;
    font-weight: 400;
}

.doc-header .doc-meta span {
    margin: 0 0.5rem;
}

/* ============================================================
   Layout: Content + TOC Sidebar
   ============================================================ */
.doc-wrapper {
    display: flex;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem;
    gap: 2rem;
    align-items: flex-start;
}

.doc-content {
    flex: 1;
    min-width: 0;
    max-width: 900px;
}

.toc-sidebar {
    width: 260px;
    flex-shrink: 0;
    position: sticky;
    top: 1rem;
    max-height: calc(100vh - 2rem);
    overflow-y: auto;
    background: white;
    border-radius: 8px;
    padding: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    font-size: 0.82rem;
    line-height: 1.5;
}

.toc-sidebar h2 {
    font-size: 0.85rem;
    font-weight: 700;
    color: #1F3864;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #0563C1;
}

.toc-sidebar ul {
    list-style: none;
    padding-left: 0;
}

.toc-sidebar li {
    margin-bottom: 0.25rem;
}

.toc-sidebar li li {
    padding-left: 0.8rem;
}

.toc-sidebar a {
    color: #595959;
    text-decoration: none;
    display: block;
    padding: 0.15rem 0;
    transition: color 0.15s;
}

.toc-sidebar a:hover {
    color: #0563C1;
}

/* ============================================================
   Mobile TOC (collapsible)
   ============================================================ */
.toc-mobile {
    display: none;
    background: white;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

.toc-mobile summary {
    font-weight: 700;
    color: #1F3864;
    cursor: pointer;
    font-size: 0.95rem;
}

.toc-mobile ul {
    list-style: none;
    padding-left: 0;
    margin-top: 0.5rem;
    font-size: 0.85rem;
}

.toc-mobile li {
    margin-bottom: 0.2rem;
}

.toc-mobile li li {
    padding-left: 0.8rem;
}

.toc-mobile a {
    color: #595959;
    text-decoration: none;
}

.toc-mobile a:hover {
    color: #0563C1;
}

/* ============================================================
   Content Card
   ============================================================ */
.content-card {
    background: white;
    border-radius: 8px;
    padding: 2rem 2.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ============================================================
   Typography
   ============================================================ */
.content-card h1 {
    color: #1F3864;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 2rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #0563C1;
}

.content-card h2 {
    color: #1F3864;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #0563C1;
}

.content-card h3 {
    color: #1F3864;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.8rem 0 0.8rem;
}

.content-card h4 {
    color: #1F3864;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 1.5rem 0 0.6rem;
}

.content-card p {
    margin-bottom: 0.8rem;
}

.content-card ul, .content-card ol {
    padding-left: 1.8rem;
    margin-bottom: 0.8rem;
}

.content-card li {
    margin-bottom: 0.3rem;
}

.content-card strong {
    font-weight: 600;
    color: #1F3864;
}

.content-card em {
    font-style: italic;
}

.content-card hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 2rem 0;
}

.content-card a {
    color: #0563C1;
    text-decoration: none;
}

.content-card a:hover {
    text-decoration: underline;
}

/* ============================================================
   Tables
   ============================================================ */
.content-card table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0 1.5rem;
    font-size: 0.88rem;
}

.content-card thead th {
    background: #1F3864;
    color: white;
    padding: 0.6rem 0.8rem;
    font-weight: 600;
    text-align: left;
    white-space: nowrap;
}

.content-card tbody td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #eee;
}

.content-card tbody tr:nth-child(even) {
    background: #f8f9fa;
}

.content-card tbody tr:hover {
    background: #e8f0fe;
}

/* ============================================================
   Code Blocks
   ============================================================ */
.content-card code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.88em;
    background: #f0f2f5;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    color: #c00000;
}

.content-card pre {
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    overflow-x: auto;
    margin: 1rem 0 1.5rem;
    font-size: 0.85rem;
    line-height: 1.5;
}

.content-card pre code {
    background: none;
    color: inherit;
    padding: 0;
    border-radius: 0;
}

/* ============================================================
   Math (MathJax)
   ============================================================ */
.MathJax {
    font-size: 1.05em !important;
}

/* ============================================================
   Back-to-Top Button
   ============================================================ */
.back-to-top {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: #0563C1;
    color: white;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    font-size: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    opacity: 0;
    transition: opacity 0.3s;
    z-index: 100;
}

.back-to-top.visible {
    opacity: 1;
}

.back-to-top:hover {
    background: #1F3864;
}

/* ============================================================
   Footer
   ============================================================ */
.doc-footer {
    max-width: 1200px;
    margin: 2rem auto 0;
    padding: 1.5rem;
    text-align: center;
    color: #808080;
    font-size: 0.8rem;
    border-top: 1px solid #ddd;
}

.doc-footer p {
    margin: 0.2rem 0;
}

/* ============================================================
   Print Styles
   ============================================================ */
@media print {
    .doc-header {
        background: white !important;
        color: #1F3864 !important;
        border-bottom: 2px solid #1F3864;
    }

    .toc-sidebar {
        display: none !important;
    }

    .toc-mobile {
        display: block !important;
    }

    .back-to-top {
        display: none !important;
    }

    .doc-wrapper {
        display: block;
    }

    .content-card {
        box-shadow: none;
        border: none;
        padding: 0;
    }

    body {
        background: white;
        font-size: 11pt;
    }

    .content-card h2 {
        page-break-before: always;
    }

    .content-card h2:first-of-type {
        page-break-before: avoid;
    }

    .content-card table {
        page-break-inside: avoid;
    }

    .content-card pre {
        background: #f0f2f5 !important;
        color: #333 !important;
        border: 1px solid #ddd;
    }
}

/* ============================================================
   Responsive
   ============================================================ */
@media (max-width: 900px) {
    .toc-sidebar {
        display: none;
    }

    .toc-mobile {
        display: block;
    }

    .doc-wrapper {
        flex-direction: column;
        padding: 1rem;
    }

    .content-card {
        padding: 1.2rem 1rem;
    }

    .doc-header h1 {
        font-size: 1.3rem;
    }
}
"""

# ---------------------------------------------------------------------------
# MathJax Configuration Script
# ---------------------------------------------------------------------------
MATHJAX_SCRIPT = r"""
<script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']],
            processEscapes: true,
            processEnvironments: true
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }
    };
</script>
<script id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
"""

# ---------------------------------------------------------------------------
# JavaScript for Back-to-Top and TOC Highlighting
# ---------------------------------------------------------------------------
DOCUMENT_JS = r"""
<script>
(function() {
    'use strict';

    // --- Back-to-top button ---
    var btn = document.querySelector('.back-to-top');
    if (btn) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 400) {
                btn.classList.add('visible');
            } else {
                btn.classList.remove('visible');
            }
        });
    }

    // --- Active TOC highlighting ---
    var tocLinks = document.querySelectorAll('.toc-sidebar a');
    var headings = [];
    for (var i = 0; i < tocLinks.length; i++) {
        var href = tocLinks[i].getAttribute('href');
        if (href && href.charAt(0) === '#') {
            var el = document.getElementById(href.substring(1));
            if (el) headings.push({ el: el, link: tocLinks[i] });
        }
    }

    function highlightToc() {
        var scrollPos = window.scrollY + 80;
        var current = null;
        for (var j = 0; j < headings.length; j++) {
            if (headings[j].el.offsetTop <= scrollPos) {
                current = headings[j];
            }
        }
        for (var k = 0; k < headings.length; k++) {
            headings[k].link.style.color = '';
            headings[k].link.style.fontWeight = '';
        }
        if (current) {
            current.link.style.color = '#0563C1';
            current.link.style.fontWeight = '600';
        }
    }

    if (headings.length > 0) {
        window.addEventListener('scroll', highlightToc);
        highlightToc();
    }
})();
</script>
"""


# ---------------------------------------------------------------------------
# Full HTML Assembly
# ---------------------------------------------------------------------------
def build_html_document(
    title: str,
    version: str,
    date_str: str,
    md_source: str,
    subtitle: str = "",
) -> str:
    """Convert Markdown source to a complete standalone HTML document.

    Parameters
    ----------
    title : str
        Document title shown in the header and <title> tag.
    version : str
        Version string displayed in the header.
    date_str : str
        Date string displayed in the header (e.g. "February 2026").
    md_source : str
        Raw Markdown text to convert.
    subtitle : str, optional
        Optional subtitle shown below the title.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    html_body, toc_html = convert_markdown_to_html(md_source)

    meta_parts = []
    if version:
        meta_parts.append(f"<span>Version {version}</span>")
    if date_str:
        meta_parts.append(f"<span>{date_str}</span>")
    meta_line = " &bull; ".join(meta_parts)

    subtitle_html = f'<p class="doc-meta">{subtitle}</p>' if subtitle else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {MATHJAX_SCRIPT}
    <style>
{DOCUMENT_CSS}
    </style>
</head>
<body id="top">

<div class="doc-header">
    <h1>{title}</h1>
    {subtitle_html}
    <p class="doc-meta">{meta_line}</p>
</div>

<div class="doc-wrapper">
    <aside class="toc-sidebar">
        <h2>Contents</h2>
        {toc_html}
    </aside>

    <div class="doc-content">
        <details class="toc-mobile">
            <summary>Table of Contents</summary>
            {toc_html}
        </details>

        <div class="content-card">
            {html_body}
        </div>
    </div>
</div>

<a href="#top" class="back-to-top" title="Back to top">&#8679;</a>

<div class="doc-footer">
    <p>Generated {date_str} &bull; North Dakota State Data Center</p>
    <p>North Dakota Cohort-Component Population Projection Model</p>
</div>

{DOCUMENT_JS}
</body>
</html>"""

    return html
