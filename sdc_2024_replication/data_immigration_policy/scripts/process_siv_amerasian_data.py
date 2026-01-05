# mypy: ignore-errors
"""
Process RPC Amerasian & SIV arrivals by nationality and state.

Extracts monthly arrivals from RPC PDF archives and produces:
- Annual fiscal-year totals by state and nationality
- Monthly series (Oct-Sep)
- PEP-year aligned totals using Jul–Jun crosswalk
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from cohort_projections.utils.reproducibility import log_execution

logger = logging.getLogger(__name__)

MONTH_ORDER = [
    "Oct",
    "Nov",
    "Dec",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
]
MONTH_TO_PEP_OFFSET = {
    "Oct": 0,
    "Nov": 0,
    "Dec": 0,
    "Jan": 0,
    "Feb": 0,
    "Mar": 0,
    "Apr": 0,
    "May": 0,
    "Jun": 0,
    "Jul": 1,
    "Aug": 1,
    "Sep": 1,
}

SIV_PDF_CANDIDATES = {
    2021: ["FY2021 Amerasian & SIV Arrivals by Nationality and State.pdf"],
    2022: ["FY 2022 Amerasian & SIV Arrivals by Nationality and State.pdf"],
    2023: ["FY 2023 Amerasian & SIV Arrivals by Nationality and State.pdf"],
    2024: ["FY 2024 Amerasian & SIV Arrivals by Nationality and State_updated_2025_01_14.pdf"],
}

NATIONALITY_STANDARDIZATION = {
    "dem. rep. congo": "Democratic Republic of the Congo",
    "dem. rep. of the congo": "Democratic Republic of the Congo",
    "dem. rep. con..": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "zaire": "Democratic Republic of the Congo",
    "unknown": "Unknown",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "republic of sou..": "Republic of South Sudan",
    "republic of sou.": "Republic of South Sudan",
}


def _parse_numeric_cell(cell: str) -> int:
    """Parse numeric cells from PDF text."""
    if not cell:
        return 0
    cleaned = cell.replace(",", "").strip()
    if cleaned in {".", "-", "D"}:
        return 0
    digits = re.sub(r"[^0-9]", "", cleaned)
    return int(digits) if digits else 0


def clean_nationality_name(nat: str) -> str:
    """Standardize nationality names."""
    if nat is None:
        return nat
    nat_lower = nat.lower().strip()
    if nat_lower in NATIONALITY_STANDARDIZATION:
        return NATIONALITY_STANDARDIZATION[nat_lower]
    return nat.title()


def _group_words_by_line(words: list[dict]) -> list[list[dict]]:
    """Group extracted words into lines by y-position."""
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if not lines or abs(word["top"] - lines[-1][0]["top"]) > 1.5:
            lines.append([word])
        else:
            lines[-1].append(word)
    return lines


def _find_month_header(words: list[dict]) -> dict[str, float] | None:
    """Locate month header positions from extracted words."""
    month_words = [w for w in words if w["text"] in MONTH_ORDER + ["Total"]]
    for top in sorted({round(w["top"], 1) for w in month_words}):
        line = [w for w in month_words if abs(w["top"] - top) < 0.5]
        texts = {w["text"] for w in line}
        if "Oct" in texts and "Sep" in texts:
            return {w["text"]: w["x0"] for w in line}
    return None


def _find_label_split(words: list[dict]) -> float | None:
    """Find x-position that splits nationality and state columns."""
    header_words = [w for w in words if w["text"] in {"PA", "Nationality", "State", "Name"}]
    for top in sorted({round(w["top"], 1) for w in header_words}):
        line = [w for w in header_words if abs(w["top"] - top) < 0.5]
        texts = {w["text"] for w in line}
        if {"PA", "Nationality", "State"}.issubset(texts):
            nationality_x = min(w["x0"] for w in line if w["text"] == "Nationality")
            state_x = min(w["x0"] for w in line if w["text"] == "State")
            return (nationality_x + state_x) / 2
    return None


def extract_monthly_from_pdf(pdf_path: Path, fiscal_year: int) -> pd.DataFrame:
    """Extract monthly Amerasian/SIV arrivals by state and nationality."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; skipping %s", pdf_path.name)
        return pd.DataFrame()

    rows: list[dict] = []
    current_nationality: str | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            if not words:
                continue

            month_x = _find_month_header(words)
            split_x = _find_label_split(words)
            if not month_x or split_x is None:
                continue

            min_month_x = min(month_x[m] for m in MONTH_ORDER if m in month_x)

            for line_words in _group_words_by_line(words):
                texts = {w["text"] for w in line_words}
                if "Oct" in texts and "Sep" in texts:
                    continue
                if any(
                    marker in texts
                    for marker in ("Amerasian", "Arrivals", "Nationality", "Fiscal", "October")
                ):
                    continue

                label_words = [
                    w
                    for w in line_words
                    if w["x0"] < min_month_x - 5 and not any(ch.isdigit() for ch in w["text"])
                ]
                value_words = [
                    w
                    for w in line_words
                    if w["x0"] >= min_month_x - 5 and any(ch.isdigit() for ch in w["text"])
                ]
                if not label_words or not value_words:
                    continue

                label = " ".join(w["text"] for w in label_words).strip()
                if not label or label.startswith("Grand Total"):
                    continue

                nationality_words = [w for w in label_words if w["x0"] < split_x]
                state_words = [w for w in label_words if split_x <= w["x0"] < min_month_x - 5]

                if nationality_words:
                    current_nationality = " ".join(w["text"] for w in nationality_words).strip()
                if current_nationality is None:
                    continue

                state = " ".join(w["text"] for w in state_words).strip()
                if not state:
                    continue
                if "settlement" in state.lower() or current_nationality.lower().startswith(
                    "based on"
                ):
                    continue

                values: dict[str, str] = {}
                for w in value_words:
                    text = w["text"]
                    if not any(ch.isdigit() for ch in text):
                        continue
                    closest = min(month_x.items(), key=lambda kv: abs(w["x0"] - kv[1]))[0]
                    values[closest] = text

                if not values:
                    continue

                for month in MONTH_ORDER:
                    arrivals = _parse_numeric_cell(values.get(month, ""))
                    rows.append(
                        {
                            "state": state,
                            "nationality": clean_nationality_name(current_nationality),
                            "fiscal_year": fiscal_year,
                            "month": month,
                            "arrivals": arrivals,
                            "data_source": "RPC Archives (Amerasian/SIV PDF)",
                        }
                    )

    return pd.DataFrame(rows)


def build_pep_year_series(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly arrivals to PEP-year aligned totals."""
    df = monthly_df.copy()
    df["pep_year"] = df["fiscal_year"] + df["month"].map(MONTH_TO_PEP_OFFSET)
    pep_df = (
        df.groupby(["state", "nationality", "pep_year"], as_index=False)["arrivals"]
        .sum()
        .assign(data_source="RPC Archives (Amerasian/SIV monthly -> PEP year)")
    )
    return pep_df


def process_siv_pdfs(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process Amerasian/SIV PDFs and return annual and monthly dataframes."""
    monthly_frames = []
    annual_frames = []

    for fy, candidates in SIV_PDF_CANDIDATES.items():
        pdf_path = None
        for filename in candidates:
            candidate_path = source_dir / filename
            if candidate_path.exists():
                pdf_path = candidate_path
                break
        if pdf_path is None:
            continue

        monthly_df = extract_monthly_from_pdf(pdf_path, fy)
        if monthly_df.empty:
            continue

        monthly_frames.append(monthly_df)
        annual_df = (
            monthly_df.groupby(["state", "nationality", "fiscal_year"], as_index=False)["arrivals"]
            .sum()
            .assign(data_source="RPC Archives (Amerasian/SIV PDF)")
        )
        annual_frames.append(annual_df)

    annual_df = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    monthly_df = (
        pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    )
    return annual_df, monthly_df


def main() -> pd.DataFrame:
    """Run Amerasian/SIV processing and write outputs."""
    project_root = Path(__file__).resolve().parents[3]
    source_dir = project_root / "data" / "raw" / "immigration" / "refugee_arrivals"
    output_dir = project_root / "data" / "processed" / "immigration" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    annual_df, monthly_df = process_siv_pdfs(source_dir)
    if annual_df.empty:
        logger.warning("No Amerasian/SIV data extracted from PDFs.")
        return annual_df

    annual_file = output_dir / "amerasian_siv_arrivals_by_state_nationality.parquet"
    annual_df.to_parquet(annual_file, index=False)
    annual_csv = output_dir / "amerasian_siv_arrivals_by_state_nationality.csv"
    annual_df.to_csv(annual_csv, index=False)

    if not monthly_df.empty:
        monthly_file = output_dir / "amerasian_siv_arrivals_by_state_nationality_monthly.parquet"
        monthly_df.to_parquet(monthly_file, index=False)
        monthly_csv = output_dir / "amerasian_siv_arrivals_by_state_nationality_monthly.csv"
        monthly_df.to_csv(monthly_csv, index=False)

        pep_df = build_pep_year_series(monthly_df)
        pep_file = output_dir / "amerasian_siv_arrivals_by_state_nationality_pep_year.parquet"
        pep_df.to_parquet(pep_file, index=False)
        pep_csv = output_dir / "amerasian_siv_arrivals_by_state_nationality_pep_year.csv"
        pep_df.to_csv(pep_csv, index=False)

    return annual_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with log_execution(__file__, parameters={"series": "amerasian_siv"}):
        main()
