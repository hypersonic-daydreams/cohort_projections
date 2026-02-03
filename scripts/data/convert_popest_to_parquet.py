#!/usr/bin/env python3
"""
Convert Census PEP (POPEST) CSV files to 1:1 parquet.

This is Phase 2 of ADR-034. It performs a 1:1 conversion (no column renames, no
row filtering) and records parquet row counts + a schema fingerprint in the
shared `catalog.yaml`.

By default, this script reads the shared data directory from the
`CENSUS_POPEST_DIR` environment variable.
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
