#!/bin/bash
# ==============================================================================
# LaTeX Compilation Script
# Forecasting International Migration to North Dakota: A Multi-Method Analysis
# ==============================================================================
# This script compiles the LaTeX document to PDF using pdflatex and bibtex.
#
# Requirements:
#   - TeX Live or similar LaTeX distribution with pdflatex
#   - bibtex for bibliography processing
#   - Required LaTeX packages (see preamble.tex for full list):
#     - amsmath, amssymb, amsthm (mathematics)
#     - graphicx, float, subcaption (figures)
#     - booktabs, array, multirow, threeparttable, tabularx (tables)
#     - natbib (citations)
#     - hyperref (hyperlinks)
#     - enumitem, xcolor, fancyhdr, titlesec (formatting)
#     - geometry, setspace (page layout)
#
# Usage:
#   ./compile.sh [--clean] [--quick]
#
# Options:
#   --clean    Remove auxiliary files before compilation
#   --quick    Run only pdflatex once (skip bibtex and multiple passes)
#
# Output:
#   main.pdf   Compiled PDF document
#   output/article_draft.pdf   Copy for distribution
# ==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
CLEAN=false
QUICK=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=true
            ;;
        --quick)
            QUICK=true
            ;;
    esac
done

# Clean auxiliary files if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning auxiliary files..."
    rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot
fi

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install a LaTeX distribution."
    echo ""
    echo "Installation options:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  Fedora/RHEL:   sudo dnf install texlive-scheme-full"
    echo "  macOS:         brew install --cask mactex"
    echo "  Windows:       Download MiKTeX from https://miktex.org/"
    echo ""
    echo "Alternatively, upload the files to Overleaf (https://www.overleaf.com/)"
    exit 1
fi

echo "============================================================"
echo "Compiling: Forecasting International Migration to North Dakota"
echo "============================================================"

if [ "$QUICK" = true ]; then
    echo "Running quick compilation (single pdflatex pass)..."
    pdflatex -interaction=nonstopmode main.tex
else
    echo "Pass 1: Running pdflatex..."
    pdflatex -interaction=nonstopmode main.tex

    echo ""
    echo "Running bibtex for bibliography..."
    bibtex main || echo "Warning: bibtex returned non-zero exit code"

    echo ""
    echo "Pass 2: Running pdflatex..."
    pdflatex -interaction=nonstopmode main.tex

    echo ""
    echo "Pass 3: Running pdflatex (final pass for references)..."
    pdflatex -interaction=nonstopmode main.tex
fi

# Check if PDF was created
if [ -f "main.pdf" ]; then
    echo ""
    echo "============================================================"
    echo "Compilation successful!"
    echo "============================================================"

    # Copy to output directory
    mkdir -p output
    cp main.pdf output/article_draft_v0.8.5.pdf
    mkdir -p ../../../revisions/v0.8.5
    cp main.pdf ../../../revisions/v0.8.5/article_draft_v0.8.5.pdf
    echo "Output: $(pwd)/main.pdf"
    echo "Copy 1: $(pwd)/output/article_draft_v0.8.5.pdf"
    echo "Copy 2: $(pwd)/../../../revisions/v0.8.5/article_draft_v0.8.5.pdf"

    # Report PDF info
    if command -v pdfinfo &> /dev/null; then
        echo ""
        echo "PDF Information:"
        pdfinfo main.pdf | grep -E "^(Pages|File size):"
    fi
else
    echo ""
    echo "ERROR: Compilation failed. Check main.log for details."
    exit 1
fi
