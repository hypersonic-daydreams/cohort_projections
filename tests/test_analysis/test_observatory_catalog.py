"""Tests for VariantCatalog (cohort_projections.analysis.observatory.variant_catalog).

Covers catalog loading, variant listing, untested detection, spec generation,
grid expansion (cartesian + zip), pending spec generation, helper functions,
and edge cases (missing files, malformed YAML, empty logs).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from cohort_projections.analysis.observatory.variant_catalog import (
    VariantCatalog,
    _config_delta_summary,
    _match_config_delta,
    _normalize_config_delta,
    _slugify_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CATALOG: dict = {
    "base_method": "m2026r1",
    "base_config": "cfg-20260309-college-fix-v1",
    "variants": {
        "exp-b": {
            "name": "Blend Factor 0.7",
            "parameter": "college_blend_factor",
            "value": 0.7,
            "tier": 1,
            "config_only": True,
            "hypothesis": "Higher blend reduces college county volatility.",
            "expected_improvement": ["county_mape_urban_college"],
            "risk_areas": ["Over-smoothing"],
            "slug": "college-blend-70",
        },
        "exp-g": {
            "name": "Blend Factor 0.8",
            "parameter": "college_blend_factor",
            "value": 0.8,
            "tier": 2,
            "config_only": True,
            "hypothesis": "Extend blend sweep beyond 0.7.",
            "slug": "college-blend-80",
        },
        "exp-d": {
            "name": "Rate Cap 0.15",
            "parameter": "rate_cap_general",
            "value": 0.15,
            "tier": 2,
            "config_only": False,
            "hypothesis": "Tighter rate cap reduces outlier projections.",
            "slug": "rate-cap-015",
        },
    },
    "grids": {
        "blend-factor-sweep": {
            "parameters": {
                "college_blend_factor": [0.6, 0.7, 0.8, 0.9],
            },
            "hypothesis": "Monotonic improvement expected.",
            "mode": "cartesian",
        },
    },
}


def _catalog_with_results() -> dict:
    """Catalog where one variant has inline results."""
    cat = {**MINIMAL_CATALOG}
    cat["variants"] = dict(cat["variants"])
    cat["variants"]["exp-b"] = {
        **cat["variants"]["exp-b"],
        "results": {"status": "passed_all_gates", "overall_mape_delta": -0.09},
    }
    return cat


@pytest.fixture()
def catalog_path(tmp_path: Path) -> Path:
    """Write minimal catalog YAML and return the path."""
    p = tmp_path / "observatory_variants.yaml"
    p.write_text(yaml.safe_dump(MINIMAL_CATALOG, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture()
def catalog_with_results_path(tmp_path: Path) -> Path:
    """Catalog YAML where exp-b has inline results."""
    p = tmp_path / "observatory_variants.yaml"
    p.write_text(
        yaml.safe_dump(_catalog_with_results(), sort_keys=False), encoding="utf-8"
    )
    return p


@pytest.fixture()
def experiment_log() -> pd.DataFrame:
    """An experiment log marking exp-b as tested."""
    return pd.DataFrame({
        "experiment_id": ["exp-20260310-blend-70"],
        "config_delta_summary": ["college_blend_factor=0.7"],
        "outcome": ["passed_all_gates"],
    })


@pytest.fixture()
def catalog(catalog_path: Path) -> VariantCatalog:
    """VariantCatalog with no experiment log."""
    return VariantCatalog(catalog_path=catalog_path, experiment_log=pd.DataFrame())


@pytest.fixture()
def catalog_with_log(catalog_path: Path, experiment_log: pd.DataFrame) -> VariantCatalog:
    """VariantCatalog cross-referenced with an experiment log."""
    return VariantCatalog(catalog_path=catalog_path, experiment_log=experiment_log)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_slugify_value_float(self) -> None:
        assert _slugify_value(0.7) == "0p7"

    def test_slugify_value_int(self) -> None:
        assert _slugify_value(3) == "3"

    def test_slugify_value_string(self) -> None:
        assert _slugify_value("hello world") == "hello-world"

    def test_slugify_value_special_chars(self) -> None:
        result = _slugify_value("a/b.c")
        assert "/" not in result

    def test_normalize_config_delta(self) -> None:
        result = _normalize_config_delta("college_blend_factor", 0.7)
        assert result == {"college_blend_factor": 0.7}

    def test_config_delta_summary_scalar(self) -> None:
        assert _config_delta_summary({"alpha": 0.5}) == "alpha=0.5"

    def test_config_delta_summary_dict(self) -> None:
        result = _config_delta_summary({"boom": {"2005-2010": 0.5}})
        assert "boom" in result
        assert "2005-2010=0.5" in result

    def test_config_delta_summary_list(self) -> None:
        result = _config_delta_summary({"fips": [38017, 38105]})
        assert "fips=" in result

    def test_config_delta_summary_multiple(self) -> None:
        result = _config_delta_summary({"a": 1, "b": 2})
        assert "a=1" in result
        assert "b=2" in result
        assert "; " in result

    def test_match_config_delta_exact(self) -> None:
        assert _match_config_delta("college_blend_factor=0.7", {"college_blend_factor": 0.7})

    def test_match_config_delta_no_match(self) -> None:
        assert not _match_config_delta("college_blend_factor=0.5", {"college_blend_factor": 0.7})

    def test_match_config_delta_non_string(self) -> None:
        assert not _match_config_delta(None, {"a": 1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestCatalogInit
# ---------------------------------------------------------------------------


class TestCatalogInit:
    """Tests for VariantCatalog construction."""

    def test_basic_construction(self, catalog: VariantCatalog) -> None:
        assert catalog.base_method == "m2026r1"
        assert catalog.base_config == "cfg-20260309-college-fix-v1"

    def test_variant_ids(self, catalog: VariantCatalog) -> None:
        ids = catalog.variant_ids
        assert "exp-b" in ids
        assert "exp-g" in ids
        assert "exp-d" in ids

    def test_grid_ids(self, catalog: VariantCatalog) -> None:
        assert "blend-factor-sweep" in catalog.grid_ids

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            VariantCatalog(catalog_path=tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("just a string", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            VariantCatalog(catalog_path=p, experiment_log=pd.DataFrame())


# ---------------------------------------------------------------------------
# TestListVariants
# ---------------------------------------------------------------------------


class TestListVariants:
    """Tests for list_variants."""

    def test_returns_dataframe(self, catalog: VariantCatalog) -> None:
        df = catalog.list_variants()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_has_expected_columns(self, catalog: VariantCatalog) -> None:
        df = catalog.list_variants()
        for col in ("variant_id", "name", "parameter", "value", "tier", "config_only", "tested"):
            assert col in df.columns

    def test_sorted_by_tier(self, catalog: VariantCatalog) -> None:
        df = catalog.list_variants()
        tiers = df["tier"].tolist()
        assert tiers == sorted(tiers)

    def test_all_untested_without_log(self, catalog: VariantCatalog) -> None:
        df = catalog.list_variants()
        assert df["tested"].sum() == 0

    def test_tested_with_log(self, catalog_with_log: VariantCatalog) -> None:
        df = catalog_with_log.list_variants()
        tested = df[df["tested"]]
        assert len(tested) == 1
        assert tested.iloc[0]["variant_id"] == "exp-b"

    def test_tested_with_inline_results(self, catalog_with_results_path: Path) -> None:
        cat = VariantCatalog(catalog_path=catalog_with_results_path, experiment_log=pd.DataFrame())
        df = cat.list_variants()
        tested = df[df["tested"]]
        assert len(tested) >= 1


# ---------------------------------------------------------------------------
# TestGetUntested
# ---------------------------------------------------------------------------


class TestGetUntested:
    """Tests for get_untested."""

    def test_all_untested_no_log(self, catalog: VariantCatalog) -> None:
        untested = catalog.get_untested()
        # Only config_only variants are returned
        assert len(untested) == 2  # exp-b and exp-g (exp-d is not config_only)
        ids = {u["variant_id"] for u in untested}
        assert "exp-d" not in ids

    def test_filters_tested(self, catalog_with_log: VariantCatalog) -> None:
        untested = catalog_with_log.get_untested()
        ids = {u["variant_id"] for u in untested}
        assert "exp-b" not in ids
        assert "exp-g" in ids


# ---------------------------------------------------------------------------
# TestGetVariant / GetGrid
# ---------------------------------------------------------------------------


class TestGetVariantAndGrid:
    """Tests for get_variant and get_grid."""

    def test_get_variant(self, catalog: VariantCatalog) -> None:
        v = catalog.get_variant("exp-b")
        assert v["variant_id"] == "exp-b"
        assert v["parameter"] == "college_blend_factor"
        assert v["value"] == 0.7

    def test_get_variant_includes_tested_flag(self, catalog: VariantCatalog) -> None:
        v = catalog.get_variant("exp-b")
        assert "tested" in v

    def test_get_variant_not_found(self, catalog: VariantCatalog) -> None:
        with pytest.raises(KeyError, match="not found"):
            catalog.get_variant("exp-zzz")

    def test_get_grid(self, catalog: VariantCatalog) -> None:
        g = catalog.get_grid("blend-factor-sweep")
        assert g["grid_id"] == "blend-factor-sweep"
        assert "parameters" in g
        assert "college_blend_factor" in g["parameters"]

    def test_get_grid_not_found(self, catalog: VariantCatalog) -> None:
        with pytest.raises(KeyError, match="not found"):
            catalog.get_grid("nonexistent-grid")


# ---------------------------------------------------------------------------
# TestGenerateSpec
# ---------------------------------------------------------------------------


class TestGenerateSpec:
    """Tests for generate_spec (single variant)."""

    def test_generates_yaml_file(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        spec_path = catalog.generate_spec("exp-b", output_dir=tmp_path)
        assert spec_path.exists()
        assert spec_path.suffix == ".yaml"

    def test_spec_content(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        spec_path = catalog.generate_spec("exp-b", output_dir=tmp_path)
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert "experiment_id" in data
        assert data["base_method"] == "m2026r1"
        assert data["config_delta"]["college_blend_factor"] == 0.7
        assert "hypothesis" in data

    def test_spec_has_benchmark_label(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        spec_path = catalog.generate_spec("exp-b", output_dir=tmp_path)
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert data["benchmark_label"] == "college-blend-70"

    def test_creates_output_dir(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "specs"
        catalog.generate_spec("exp-b", output_dir=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# TestGenerateGridSpecs
# ---------------------------------------------------------------------------


class TestGenerateGridSpecs:
    """Tests for generate_grid_specs."""

    def test_cartesian_count(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        specs = catalog.generate_grid_specs("blend-factor-sweep", output_dir=tmp_path)
        # 4 values for college_blend_factor
        assert len(specs) == 4

    def test_all_specs_valid_yaml(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        specs = catalog.generate_grid_specs("blend-factor-sweep", output_dir=tmp_path)
        for sp in specs:
            data = yaml.safe_load(sp.read_text(encoding="utf-8"))
            assert "experiment_id" in data
            assert "config_delta" in data

    def test_zip_mode(self, tmp_path: Path) -> None:
        cat_data = {
            "base_method": "m2026r1",
            "base_config": "cfg-test",
            "variants": {},
            "grids": {
                "zip-grid": {
                    "parameters": {
                        "alpha": [0.1, 0.2],
                        "beta": [10, 20],
                    },
                    "mode": "zip",
                    "hypothesis": "Zip test.",
                },
            },
        }
        p = tmp_path / "cat.yaml"
        p.write_text(yaml.safe_dump(cat_data), encoding="utf-8")
        cat = VariantCatalog(catalog_path=p, experiment_log=pd.DataFrame())

        out = tmp_path / "specs"
        specs = cat.generate_grid_specs("zip-grid", output_dir=out)
        assert len(specs) == 2

    def test_zip_unequal_raises(self, tmp_path: Path) -> None:
        cat_data = {
            "base_method": "m2026r1",
            "base_config": "cfg-test",
            "variants": {},
            "grids": {
                "bad-zip": {
                    "parameters": {
                        "alpha": [0.1, 0.2],
                        "beta": [10],
                    },
                    "mode": "zip",
                    "hypothesis": "Should fail.",
                },
            },
        }
        p = tmp_path / "cat.yaml"
        p.write_text(yaml.safe_dump(cat_data), encoding="utf-8")
        cat = VariantCatalog(catalog_path=p, experiment_log=pd.DataFrame())
        with pytest.raises(ValueError, match="equal-length"):
            cat.generate_grid_specs("bad-zip", output_dir=tmp_path / "specs")

    def test_multi_param_cartesian(self, tmp_path: Path) -> None:
        cat_data = {
            "base_method": "m2026r1",
            "base_config": "cfg-test",
            "variants": {},
            "grids": {
                "multi": {
                    "parameters": {
                        "alpha": [0.1, 0.2],
                        "beta": [10, 20],
                    },
                    "hypothesis": "Cartesian product.",
                },
            },
        }
        p = tmp_path / "cat.yaml"
        p.write_text(yaml.safe_dump(cat_data), encoding="utf-8")
        cat = VariantCatalog(catalog_path=p, experiment_log=pd.DataFrame())

        out = tmp_path / "specs"
        specs = cat.generate_grid_specs("multi", output_dir=out)
        assert len(specs) == 4  # 2 x 2


# ---------------------------------------------------------------------------
# TestGenerateAllPending
# ---------------------------------------------------------------------------


class TestGenerateAllPending:
    """Tests for generate_all_pending_specs."""

    def test_generates_for_untested_config_only(self, catalog: VariantCatalog, tmp_path: Path) -> None:
        specs = catalog.generate_all_pending_specs(output_dir=tmp_path)
        # exp-b and exp-g are config_only and untested; exp-d is not config_only
        assert len(specs) == 2

    def test_skips_tested_variants(self, catalog_with_log: VariantCatalog, tmp_path: Path) -> None:
        specs = catalog_with_log.generate_all_pending_specs(output_dir=tmp_path)
        # exp-b is tested, exp-g is untested config_only
        assert len(specs) == 1

    def test_empty_catalog(self, tmp_path: Path) -> None:
        cat_data = {
            "base_method": "m2026r1",
            "base_config": "cfg-test",
            "variants": {},
            "grids": {},
        }
        p = tmp_path / "empty_cat.yaml"
        p.write_text(yaml.safe_dump(cat_data), encoding="utf-8")
        cat = VariantCatalog(catalog_path=p, experiment_log=pd.DataFrame())
        specs = cat.generate_all_pending_specs(output_dir=tmp_path / "out")
        assert specs == []
