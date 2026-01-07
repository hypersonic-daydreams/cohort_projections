from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def versioning_module():
    root = Path(__file__).resolve().parents[2]
    module_path = (
        root
        / "sdc_2024_replication"
        / "scripts"
        / "statistical_analysis"
        / "journal_article"
        / "build_versioned_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("build_versioned_artifacts", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_from_status(versioning_module):
    assert versioning_module.stage_from_status("draft") == "working"
    assert versioning_module.stage_from_status("review") == "working"
    assert versioning_module.stage_from_status("approved") == "approved"
    assert versioning_module.stage_from_status("production") == "production"


def test_build_version_basename(versioning_module):
    assert (
        versioning_module.build_version_basename("0.8.6", "draft", "20260106_120000")
        == "article-0.8.6-draft_20260106_120000"
    )


def test_hash_file(versioning_module, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello", encoding="utf-8")
    assert (
        versioning_module.hash_file(target)
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_collect_files(versioning_module, tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.txt").write_text("c", encoding="utf-8")

    files = versioning_module.collect_files(tmp_path, ["*.txt", "nested/*.txt"])
    names = {path.name for path in files}
    assert names == {"a.txt", "c.txt"}


def test_update_versions_index(versioning_module, tmp_path):
    versions_path = tmp_path / "VERSIONS.md"
    versions_path.write_text(
        "# Article Version Index\n\n"
        "## Working Versions\n"
        "| Version | Date | Status | Branch | Notes |\n"
        "|---------|------|--------|--------|-------|\n"
        "| 0.6.0 | 2026-01-01 | draft | master | Phase B complete |\n\n"
        "*Last Updated: 2026-01-01*\n",
        encoding="utf-8",
    )

    versioning_module.update_versions_index(
        versions_path,
        version="0.8.6",
        date="2026-01-06",
        status="draft",
        branch="main",
        notes="Automated build",
    )

    updated = versions_path.read_text(encoding="utf-8")
    assert "| 0.8.6 | 2026-01-06 | draft | main | Automated build |" in updated
    assert "*Last Updated: 2026-01-06*" in updated


def test_update_current_version_file(versioning_module, tmp_path):
    current_path = tmp_path / "CURRENT_VERSION.txt"
    current_path.write_text(
        "# Current Article Versions\n"
        "# Updated: 2026-01-01\n\n"
        "# Current production version\n"
        "# (none - not yet submitted)\n\n"
        "# Current working version\n"
        "article-0.6.0-draft_20260101_153746.pdf\n"
        "# Also available with full timestamp: article-0.6.0-draft_20260101_153746.pdf\n",
        encoding="utf-8",
    )

    versioning_module.update_current_version_file(
        current_path,
        version_label="article-0.8.6-draft_20260106_120000.pdf",
        updated_date="2026-01-06",
    )

    updated = current_path.read_text(encoding="utf-8")
    assert "# Updated: 2026-01-06" in updated
    assert "article-0.8.6-draft_20260106_120000.pdf" in updated
    assert (
        "# Also available with full timestamp: article-0.8.6-draft_20260106_120000.pdf" in updated
    )
