from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pytest

from cohort_projections.data.popest_shared import schema_fingerprint
from scripts.data.archive_popest_raw_by_vintage import _verify_zip_against_manifest


def test_schema_fingerprint_is_deterministic() -> None:
    schema1 = pa.schema([("STATE", pa.string()), ("POPESTIMATE2020", pa.string())])
    fp1 = schema_fingerprint(schema1)
    fp2 = schema_fingerprint(schema1)
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_schema_fingerprint_changes_on_type_change() -> None:
    schema1 = pa.schema([("A", pa.string()), ("B", pa.int64())])
    schema2 = pa.schema([("A", pa.string()), ("B", pa.int32())])
    assert schema_fingerprint(schema1) != schema_fingerprint(schema2)


def test_verify_zip_against_manifest_success(tmp_path: Path) -> None:
    raw_rel = "2020-2024/place/example.csv"
    content = b"col1,col2\n1,2\n"
    expected_md5 = hashlib.md5(content).hexdigest()

    manifest = {
        "schema_version": "1.0",
        "vintage": "2020-2024",
        "created_at": "2026-02-03 00:00:00",
        "files": [{"raw_file": raw_rel, "md5": expected_md5, "file_size_bytes": len(content)}],
    }

    zip_path = tmp_path / "2020-2024-raw.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest).encode())
        zf.writestr(raw_rel, content)

    _verify_zip_against_manifest(zip_path, manifest)


def test_verify_zip_against_manifest_fails_on_md5(tmp_path: Path) -> None:
    raw_rel = "2020-2024/place/example.csv"
    content = b"col1,col2\n1,2\n"
    manifest = {
        "schema_version": "1.0",
        "vintage": "2020-2024",
        "created_at": "2026-02-03 00:00:00",
        "files": [{"raw_file": raw_rel, "md5": "deadbeef", "file_size_bytes": len(content)}],
    }

    zip_path = tmp_path / "2020-2024-raw.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest).encode())
        zf.writestr(raw_rel, content)

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        _verify_zip_against_manifest(zip_path, manifest)
