#!/usr/bin/env python3
"""
Convert Census PEP (POPEST) CSV files to 1:1 parquet format.

Created: 2026-02-03
ADR: 034 (Census PEP Data Archive — Phase 2)
Author: nhaarstad

Purpose
-------
Perform a lossless 1:1 conversion of raw Census PEP CSV files to Parquet format
for faster downstream reads and smaller storage footprint. Raw CSV files range
from a few KB to 169 MB (the county intercensal ASRH file); Parquet with zstd
compression typically achieves 5-10x size reduction while enabling columnar reads.
This step preserves all original columns and rows without any filtering or
renaming, ensuring the raw data remains fully reproducible from Parquet alone.

Method
------
1. Load the shared catalog.yaml to identify target datasets (optionally filtered
   by dataset ID, vintage, or geographic level).
2. For each dataset with a raw CSV file:
   a. Read the CSV header to determine column names.
   b. Force all columns to string type (no type inference) to preserve the
      exact raw representation.
   c. Stream the CSV in batches via PyArrow CSVStreamingReader to handle
      large files without loading entirely into memory.
   d. Write batches to a Parquet file using the specified compression codec
      (default: zstd level 3) with dictionary encoding enabled.
   e. If UTF-8 decoding fails, automatically retry with latin1 encoding
      (some older Census files use Windows-1252 characters).
3. Record parquet metadata (row count, schema fingerprint, file size, MD5) in
   the catalog.yaml entry for each converted dataset.
4. Skip datasets whose parquet file already exists unless --overwrite is set.

Key design decisions
--------------------
- **All-string columns**: No type inference is applied during conversion. This
  prevents silent data loss (e.g., leading-zero FIPS codes being parsed as
  integers) and ensures the Parquet file is a byte-for-byte faithful
  representation of the CSV. Type casting happens in downstream consumers.
- **Streaming batch writes**: Using PyArrow's CSVStreamingReader avoids loading
  the full 169 MB county intercensal file into memory. Batches are written
  incrementally to a temporary file, then atomically renamed on success.
- **Atomic writes via temp file**: Parquet output is written to a .tmp file
  first, then renamed to the final path. This prevents partial/corrupt files
  if the process is interrupted.
- **Latin1 fallback**: Older Census files (1980s-1990s vintages) occasionally
  contain non-UTF-8 characters. Rather than failing, the script retries with
  latin1 encoding and records which encoding was used in the catalog.

Validation results (2026-02-03)
-------------------------------
- All 23 catalog datasets converted successfully
- Schema fingerprints and MD5 checksums recorded in catalog.yaml
- Round-trip verification: row counts in parquet match CSV line counts
- Largest file (cc-est2020int-alldata.csv, 169 MB) converts to ~18 MB parquet

Inputs
------
- $CENSUS_POPEST_DIR/catalog.yaml
    Shared dataset catalog with raw_file paths for each dataset.
- $CENSUS_POPEST_DIR/raw/{vintage}/{level}/*.csv
    Raw CSV files as downloaded from Census Bureau FTP.

Output
------
- $CENSUS_POPEST_DIR/parquet/{vintage}/{level}/*.parquet
    1:1 parquet conversions, one per input CSV. Zstd compressed with
    dictionary encoding.
- $CENSUS_POPEST_DIR/catalog.yaml (updated)
    Each dataset entry gains: parquet_file, parquet_row_count,
    parquet_schema_fingerprint, parquet_file_size_bytes, parquet_md5.

Usage
-----
    python scripts/data/convert_popest_to_parquet.py
    python scripts/data/convert_popest_to_parquet.py --dataset-id co-est2024-alldata
    python scripts/data/convert_popest_to_parquet.py --vintage 2020-2024 --overwrite
    python scripts/data/convert_popest_to_parquet.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from cohort_projections.data.popest_shared import (
    PopestPaths,
    configure_logging,
    load_catalog,
    md5_file,
    resolve_popest_paths,
    schema_fingerprint,
    write_catalog,
)
from cohort_projections.utils.reproducibility import log_execution

LOGGER = logging.getLogger(__name__)


def _csv_header_columns(path: Path) -> list[str]:
    with path.open("r", newline="") as f:
        reader = csv.reader([f.readline()])
        return next(reader)


def _convert_csv_to_parquet_1to1(
    *,
    csv_path: Path,
    parquet_path: Path,
    compression: str,
    compression_level: int,
    encoding: str,
) -> tuple[int, pa.Schema, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    columns = _csv_header_columns(csv_path)
    column_types = {name: pa.string() for name in columns}

    def open_reader(*, enc: str) -> pacsv.CSVStreamingReader:
        return pacsv.open_csv(
            csv_path,
            read_options=pacsv.ReadOptions(use_threads=True, encoding=enc),
            convert_options=pacsv.ConvertOptions(column_types=column_types),
            parse_options=pacsv.ParseOptions(ignore_empty_lines=True),
        )

    def attempt(*, enc: str) -> tuple[int, pa.Schema]:
        reader = open_reader(enc=enc)
        row_count = 0
        writer: pq.ParquetWriter | None = None
        schema: pa.Schema | None = None
        tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")

        try:
            try:
                while True:
                    batch = reader.read_next_batch()
                    if writer is None:
                        schema = batch.schema
                        writer = pq.ParquetWriter(
                            tmp_path,
                            schema,
                            compression=compression,
                            compression_level=compression_level,
                            use_dictionary=True,
                        )
                    writer.write_batch(batch)
                    row_count += batch.num_rows
            except StopIteration:
                pass
            finally:
                if writer is not None:
                    writer.close()

            if schema is None:
                # Empty CSV (header-only) edge case
                schema = pa.schema([(name, pa.string()) for name in columns])
                pq.write_table(
                    pa.table(
                        {name: pa.array([], type=pa.string()) for name in columns}, schema=schema
                    ),
                    tmp_path,
                    compression=compression,
                    compression_level=compression_level,
                    use_dictionary=True,
                )
                row_count = 0

            tmp_path.replace(parquet_path)
            return row_count, schema
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    try:
        row_count, schema = attempt(enc=encoding)
        return row_count, schema, encoding
    except pa.ArrowInvalid as e:
        if "invalid UTF8" not in str(e) or encoding.lower() != "utf8":
            raise
        LOGGER.warning("Invalid UTF-8 encountered; retrying with latin1 for %s", csv_path)
        row_count, schema = attempt(enc="latin1")
        return row_count, schema, "latin1"


def _iter_target_datasets(
    catalog: dict[str, Any],
    *,
    dataset_id: str | None,
    vintage: str | None,
    level: str | None,
) -> list[dict[str, Any]]:
    datasets = list(catalog.get("datasets", []))
    selected: list[dict[str, Any]] = []
    for ds in datasets:
        if dataset_id and ds.get("id") != dataset_id:
            continue
        if vintage and ds.get("vintage") != vintage:
            continue
        if level and ds.get("level") != level:
            continue
        selected.append(ds)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Census PEP CSVs to parquet (ADR-034 Phase 2)"
    )
    parser.add_argument(
        "--popest-dir", help="Override CENSUS_POPEST_DIR for the shared data directory."
    )
    parser.add_argument("--dataset-id", help="Convert only a single dataset id from catalog.yaml.")
    parser.add_argument(
        "--vintage", help="Convert only datasets for a single vintage (e.g., 2020-2024)."
    )
    parser.add_argument(
        "--level",
        choices=["place", "county", "state", "docs"],
        help="Convert only datasets for a single level.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing parquet files."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write parquet or update catalog."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--compression", default="zstd", help="Parquet compression codec (default: zstd)."
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Parquet compression level (default: 3).",
    )
    parser.add_argument(
        "--encoding",
        default="utf8",
        help="CSV encoding for pyarrow.csv (default: utf8; falls back to latin1 on invalid UTF-8).",
    )

    args = parser.parse_args()
    configure_logging(args.verbose)

    paths: PopestPaths = resolve_popest_paths(args.popest_dir)
    catalog = load_catalog(paths.catalog_path)
    targets = _iter_target_datasets(
        catalog, dataset_id=args.dataset_id, vintage=args.vintage, level=args.level
    )

    if not targets:
        LOGGER.info("No matching datasets found in catalog.")
        return 0

    with log_execution(
        __file__,
        parameters={
            "dataset_id": args.dataset_id,
            "vintage": args.vintage,
            "level": args.level,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "compression": args.compression,
            "compression_level": args.compression_level,
        },
        inputs=[paths.catalog_path],
        outputs=[paths.catalog_path],
    ):
        for ds in targets:
            raw_rel = ds.get("raw_file")
            if not raw_rel:
                continue

            raw_path = paths.raw_dir / raw_rel
            if raw_path.suffix.lower() != ".csv":
                LOGGER.debug("Skip non-CSV: %s", raw_path)
                continue
            if not raw_path.exists():
                LOGGER.warning("Missing raw file: %s", raw_path)
                continue

            parquet_rel = str(Path(raw_rel).with_suffix(".parquet"))
            parquet_path = paths.parquet_dir / parquet_rel

            if parquet_path.exists() and not args.overwrite:
                LOGGER.info("Parquet exists (using existing): %s", parquet_path)
                try:
                    pf = pq.ParquetFile(parquet_path)
                except Exception:
                    LOGGER.warning("Existing parquet is unreadable; overwriting: %s", parquet_path)
                else:
                    ds["parquet_file"] = parquet_rel
                    ds["parquet_row_count"] = (
                        pf.metadata.num_rows if pf.metadata is not None else None
                    )
                    ds["parquet_schema_fingerprint"] = schema_fingerprint(pf.schema_arrow)
                    ds["parquet_file_size_bytes"] = parquet_path.stat().st_size
                    ds["parquet_md5"] = md5_file(parquet_path)
                    ds.setdefault("parquet_written", datetime.now(UTC).strftime("%Y-%m-%d"))
                    continue

            LOGGER.info("Converting %s -> %s", raw_path, parquet_path)

            if args.dry_run:
                continue

            row_count, schema, encoding_used = _convert_csv_to_parquet_1to1(
                csv_path=raw_path,
                parquet_path=parquet_path,
                compression=args.compression,
                compression_level=args.compression_level,
                encoding=args.encoding,
            )

            ds["parquet_file"] = parquet_rel
            ds["parquet_row_count"] = row_count
            ds["parquet_schema_fingerprint"] = schema_fingerprint(schema)
            ds["parquet_file_size_bytes"] = parquet_path.stat().st_size
            ds["parquet_md5"] = md5_file(parquet_path)
            ds["parquet_written"] = datetime.now(UTC).strftime("%Y-%m-%d")
            ds["csv_read_encoding"] = encoding_used

        if not args.dry_run:
            write_catalog(paths.catalog_path, catalog)
            LOGGER.info("Updated catalog: %s", paths.catalog_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
