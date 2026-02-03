#!/usr/bin/env python3
"""
Extract Census PEP (POPEST) technical documentation text + tables.

This is Phase 4 of ADR-034. It:
  - extracts per-page text and best-effort tables from PDFs in `raw/docs/`
  - writes derived artifacts to `derived/docs/` with PDF+page references
  - generates per-dataset rich metadata JSON files in `metadata/`
  - links derived artifacts from `catalog.yaml`
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pdfplumber
import pyarrow.parquet as pq

from cohort_projections.data.popest_shared import (
    configure_logging,
    load_catalog,
    resolve_popest_paths,
    write_catalog,
)
from cohort_projections.utils.reproducibility import log_execution

LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_raw_bytes(
    *,
    raw_path: Path,
    raw_rel: str,
    vintage: str | None,
    raw_archives_dir: Path,
) -> bytes:
    if raw_path.exists():
        return raw_path.read_bytes()

    if not vintage:
        raise FileNotFoundError(f"Missing raw file and no vintage to locate archive: {raw_path}")

    archive_path = raw_archives_dir / f"{vintage}-raw.zip"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing raw file and archive not found: {raw_path} ({archive_path})"
        )

    with ZipFile(archive_path, "r") as zf:
        try:
            return zf.read(raw_rel)
        except KeyError as e:
            raise FileNotFoundError(f"File not found in archive {archive_path}: {raw_rel}") from e


def _extract_pdf(
    *,
    pdf_bytes: bytes,
    source_raw_file: str,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    if index_path.exists() and not overwrite:
        return json.loads(index_path.read_text())

    pages_dir = output_dir / "pages"
    tables_dir = output_dir / "tables"
    pages_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    doc_md5 = hashlib.md5(pdf_bytes).hexdigest()
    doc_sha256 = f"sha256:{_sha256_bytes(pdf_bytes)}"

    pages_index: list[dict[str, Any]] = []
    fulltext_parts: list[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            text_rel = f"pages/page_{i:04d}.txt"
            (output_dir / text_rel).write_text(page_text)
            fulltext_parts.append(page_text)

            tables: list[dict[str, Any]] = []
            try:
                extracted_tables = page.extract_tables()
            except Exception:
                extracted_tables = []

            for t_i, table in enumerate(extracted_tables or [], start=1):
                if not table:
                    continue
                df = pd.DataFrame(table)
                table_rel = f"tables/page_{i:04d}_table_{t_i:02d}.csv"
                (output_dir / table_rel).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / table_rel, index=False, header=False)
                tables.append(
                    {
                        "page": i,
                        "table_index": t_i,
                        "csv_file": table_rel,
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    }
                )

            pages_index.append(
                {
                    "page": i,
                    "text_file": text_rel,
                    "tables": tables,
                    "source_pdf": source_raw_file,
                    "source_page": i,
                }
            )

    (output_dir / "fulltext.txt").write_text("\n\n".join(fulltext_parts))

    index = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "source_pdf_raw_file": source_raw_file,
        "source_pdf_md5": doc_md5,
        "source_pdf_sha256": doc_sha256,
        "pages": pages_index,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def _docs_for_dataset(
    dataset: dict[str, Any],
    doc_datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset_norm = _normalize_for_match(str(dataset.get("id", "")))
    matches: list[dict[str, Any]] = []
    for doc in doc_datasets:
        if doc.get("vintage") != dataset.get("vintage"):
            continue
        hay = _normalize_for_match(
            str(doc.get("raw_file", "")) + " " + str(doc.get("description", ""))
        )
        if dataset_norm and dataset_norm in hay:
            matches.append(doc)

    if matches:
        return matches

    return [d for d in doc_datasets if d.get("vintage") == dataset.get("vintage")]


def _dataset_schema_summary(
    *,
    raw_dir: Path,
    parquet_dir: Path,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    parquet_file = dataset.get("parquet_file")
    raw_file = dataset.get("raw_file")
    schema: dict[str, Any] = {"columns": None, "arrow_schema": None, "source": None}

    if parquet_file:
        parquet_path = parquet_dir / parquet_file
        if parquet_path.exists():
            pf = pq.ParquetFile(parquet_path)
            arrow_schema = pf.schema_arrow
            schema["source"] = f"parquet:{parquet_file}"
            schema["arrow_schema"] = [
                {"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in arrow_schema
            ]
            schema["columns"] = [f.name for f in arrow_schema]
            return schema

    if raw_file and str(raw_file).lower().endswith(".csv"):
        raw_path = raw_dir / raw_file
        if raw_path.exists():
            header = raw_path.open("r").readline().strip()
            schema["source"] = f"csv_header:{raw_file}"
            schema["columns"] = header.split(",")
            return schema

    schema["source"] = "unknown"
    return schema


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract POPEST docs + generate dataset metadata (ADR-034 Phase 4)"
    )
    parser.add_argument(
        "--popest-dir", help="Override CENSUS_POPEST_DIR for the shared data directory."
    )
    parser.add_argument("--dataset-id", help="Generate metadata only for a single dataset id.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing derived artifacts."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write files or update catalog."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    configure_logging(args.verbose)
    paths = resolve_popest_paths(args.popest_dir)
    catalog = load_catalog(paths.catalog_path)

    datasets: list[dict[str, Any]] = list(catalog.get("datasets", []))
    doc_datasets = [d for d in datasets if str(d.get("raw_file", "")).lower().endswith(".pdf")]
    non_doc_datasets = [d for d in datasets if d.get("level") != "docs"]

    if args.dataset_id:
        non_doc_datasets = [d for d in non_doc_datasets if d.get("id") == args.dataset_id]

    derived_docs_root = paths.derived_dir / "docs"
    metadata_root = paths.metadata_dir

    if not args.dry_run:
        derived_docs_root.mkdir(parents=True, exist_ok=True)
        metadata_root.mkdir(parents=True, exist_ok=True)

    with log_execution(
        __file__,
        parameters={
            "dataset_id": args.dataset_id,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
        },
        inputs=[paths.catalog_path],
        outputs=[paths.catalog_path],
    ):
        for doc in doc_datasets:
            raw_rel = doc.get("raw_file")
            if not raw_rel:
                continue
            raw_path = paths.raw_dir / raw_rel
            doc_slug = _slugify(Path(raw_rel).stem)
            out_dir = derived_docs_root / doc_slug

            if args.dry_run:
                LOGGER.info("Dry run: would extract %s -> %s", raw_path, out_dir)
                continue

            pdf_bytes = _read_raw_bytes(
                raw_path=raw_path,
                raw_rel=raw_rel,
                vintage=doc.get("vintage"),
                raw_archives_dir=paths.raw_archives_dir,
            )
            _extract_pdf(
                pdf_bytes=pdf_bytes,
                source_raw_file=raw_rel,
                output_dir=out_dir,
                overwrite=args.overwrite,
            )
            doc["derived_doc_index_json"] = f"derived/docs/{doc_slug}/index.json"

        for ds in non_doc_datasets:
            dataset_id = ds.get("id")
            if not dataset_id:
                continue

            schema_summary = _dataset_schema_summary(
                raw_dir=paths.raw_dir,
                parquet_dir=paths.parquet_dir,
                dataset=ds,
            )

            docs = _docs_for_dataset(ds, doc_datasets)
            docs_payload = []
            for doc in docs:
                raw_rel = doc.get("raw_file")
                if not raw_rel:
                    continue
                doc_slug = _slugify(Path(raw_rel).stem)
                docs_payload.append(
                    {
                        "catalog_id": doc.get("id"),
                        "raw_pdf": raw_rel,
                        "source_url": doc.get("source_url"),
                        "derived_index_json": f"derived/docs/{doc_slug}/index.json",
                    }
                )

            metadata = {
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_id": dataset_id,
                "vintage": ds.get("vintage"),
                "level": ds.get("level"),
                "description": ds.get("description"),
                "source_url": ds.get("source_url"),
                "raw_file": ds.get("raw_file"),
                "raw_md5": ds.get("md5"),
                "raw_file_size_bytes": ds.get("file_size_bytes"),
                "parquet_file": ds.get("parquet_file"),
                "parquet_md5": ds.get("parquet_md5"),
                "parquet_file_size_bytes": ds.get("parquet_file_size_bytes"),
                "parquet_row_count": ds.get("parquet_row_count"),
                "parquet_schema_fingerprint": ds.get("parquet_schema_fingerprint"),
                "schema": schema_summary,
                "docs": docs_payload,
            }

            metadata_rel = f"metadata/{dataset_id}.json"
            metadata_path = paths.base_dir / metadata_rel

            if args.dry_run:
                LOGGER.info("Dry run: would write %s", metadata_path)
            else:
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

            ds["metadata_json"] = metadata_rel

        if not args.dry_run:
            write_catalog(paths.catalog_path, catalog)
            LOGGER.info("Updated catalog: %s", paths.catalog_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
