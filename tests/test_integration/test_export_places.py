"""Integration tests for PP-003 IMP-16 export pipeline place wiring."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

export_mod = importlib.import_module("scripts.pipeline.03_export_results")


def _build_config(projection_root: Path, export_root: Path) -> dict[str, Any]:
    """Create minimal config payload for export pipeline tests."""
    return {
        "pipeline": {
            "projection": {"output_dir": str(projection_root)},
            "export": {
                "output_dir": str(export_root),
                "create_packages": True,
                "formats": ["csv"],
            },
        },
        "scenarios": {"baseline": {"active": True}},
    }


def _write_place_summary(projection_root: Path, scenario: str) -> Path:
    """Write minimal places_summary.csv source artifact."""
    place_dir = projection_root / scenario / "place"
    place_dir.mkdir(parents=True, exist_ok=True)
    summary_path = place_dir / "places_summary.csv"
    summary_path.write_text(
        "place_fips,name,county_fips,row_type,confidence_tier,base_population,final_population,growth_rate\n"
        "3825700,Fargo,38017,place,HIGH,100.0,120.0,0.20\n",
        encoding="utf-8",
    )
    return summary_path


def _ok_result(component: str) -> Any:
    """Return successful ExportResult placeholder."""
    result = export_mod.ExportResult(component)
    result.success = True
    return result


def _patch_non_place_steps(
    monkeypatch: pytest.MonkeyPatch,
    convert_override: Callable[..., Any] | None = None,
) -> None:
    """Patch non-place export steps so tests isolate IMP-16 wiring."""
    monkeypatch.setattr(
        export_mod,
        "convert_projection_formats",
        convert_override if convert_override is not None else lambda *args, **kwargs: _ok_result("formats"),
    )
    monkeypatch.setattr(
        export_mod,
        "create_summary_tables",
        lambda *args, **kwargs: _ok_result("summaries"),
    )
    monkeypatch.setattr(
        export_mod,
        "generate_data_dictionary",
        lambda *args, **kwargs: _ok_result("dictionary"),
    )
    monkeypatch.setattr(
        export_mod,
        "package_for_distribution",
        lambda *args, **kwargs: _ok_result("packaging"),
    )


def test_places_flag_exports_place_summary_and_workbook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`--places` exports place summary CSV and workbook artifacts."""
    projection_root = tmp_path / "projections"
    export_root = tmp_path / "exports"
    config = _build_config(projection_root, export_root)
    source_summary = _write_place_summary(projection_root, "baseline")

    _patch_non_place_steps(monkeypatch)
    monkeypatch.setattr(export_mod, "load_projection_config", lambda _: config)

    def _fake_build_place_workbook(scenario: str, config: dict[str, Any]) -> Path:
        workbook_path = Path(config["pipeline"]["export"]["output_dir"]) / f"nd_projections_{scenario}_places_20260301.xlsx"
        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        workbook_path.write_text("placeholder workbook", encoding="utf-8")
        return workbook_path

    monkeypatch.setattr(export_mod, "_build_place_workbook", _fake_build_place_workbook)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_export_results.py",
            "--places",
            "--scenarios",
            "baseline",
            "--formats",
            "csv",
            "--no-package",
        ],
    )

    assert export_mod.main() == 0

    exported_summary = export_root / "baseline" / "place" / "places_summary.csv"
    workbook_path = export_root / "nd_projections_baseline_places_20260301.xlsx"
    assert exported_summary.exists()
    assert exported_summary.read_text(encoding="utf-8") == source_summary.read_text(encoding="utf-8")
    assert workbook_path.exists()


def test_all_flag_includes_place_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`--all` passes place level through export pipeline and emits place artifacts."""
    projection_root = tmp_path / "projections"
    export_root = tmp_path / "exports"
    config = _build_config(projection_root, export_root)
    _write_place_summary(projection_root, "baseline")

    seen_levels: list[str] = []

    def _fake_convert_projection_formats(*args: Any, **kwargs: Any) -> Any:
        levels = kwargs.get("levels", [])
        seen_levels.extend(levels)
        return _ok_result("formats")

    _patch_non_place_steps(monkeypatch, convert_override=_fake_convert_projection_formats)
    monkeypatch.setattr(export_mod, "load_projection_config", lambda _: config)

    def _fake_build_place_workbook(scenario: str, config: dict[str, Any]) -> Path:
        workbook_path = Path(config["pipeline"]["export"]["output_dir"]) / f"{scenario}_places.xlsx"
        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        workbook_path.write_text("placeholder workbook", encoding="utf-8")
        return workbook_path

    monkeypatch.setattr(export_mod, "_build_place_workbook", _fake_build_place_workbook)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_export_results.py",
            "--all",
            "--scenarios",
            "baseline",
            "--formats",
            "csv",
            "--no-package",
        ],
    )

    assert export_mod.main() == 0
    assert "place" in seen_levels
    assert (export_root / "baseline" / "place" / "places_summary.csv").exists()
    assert (export_root / "baseline_places.xlsx").exists()


def test_places_dry_run_creates_no_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`--dry-run --places` does not emit place files or report artifacts."""
    projection_root = tmp_path / "projections"
    export_root = tmp_path / "exports"
    config = _build_config(projection_root, export_root)
    _write_place_summary(projection_root, "baseline")

    _patch_non_place_steps(monkeypatch)
    monkeypatch.setattr(export_mod, "load_projection_config", lambda _: config)

    def _unexpected_build_workbook(scenario: str, config: dict[str, Any]) -> Path:
        raise AssertionError("Workbook builder should not run in dry-run mode")

    monkeypatch.setattr(export_mod, "_build_place_workbook", _unexpected_build_workbook)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "03_export_results.py",
            "--places",
            "--scenarios",
            "baseline",
            "--formats",
            "csv",
            "--no-package",
            "--dry-run",
        ],
    )

    assert export_mod.main() == 0
    assert not any(path.is_file() for path in export_root.rglob("*"))
