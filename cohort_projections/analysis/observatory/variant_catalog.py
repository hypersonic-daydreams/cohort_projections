"""Programmatic variant catalog for the Projection Observatory.

Created: 2026-03-12
Author: Claude Code / N. Haarstad

Purpose:
    Provides machine-readable access to all testable projection variants and
    can generate experiment spec YAML files for the benchmarking pipeline.
    The catalog YAML (``config/observatory_variants.yaml``) is the single
    source of truth for what variants exist; this module is the programmatic
    interface to it.

Method:
    1. Load variant and grid definitions from a YAML catalog file.
    2. Cross-reference with the experiment log CSV to determine tested status
       using *parameter-level* matching (not just experiment IDs).
    3. Generate experiment spec YAML files following the schema defined in
       ``config/experiment_spec_schema.yaml``.

Inputs:
    - ``config/observatory_variants.yaml`` — variant and grid definitions
    - ``data/analysis/experiments/experiment_log.csv`` — tested experiment log

Outputs:
    - Experiment spec YAML files written to a specified directory
      (default: ``data/analysis/experiments/pending/``)
"""

from __future__ import annotations

import datetime as dt
import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cohort_projections.analysis.experiment_log import (
    _match_config_delta,
)
from cohort_projections.analysis.observatory.runtime_contract import (
    get_runtime_injectable_parameters,
)
from cohort_projections.analysis.observatory.status import (
    resolve_observatory_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CATALOG_PATH = PROJECT_ROOT / "config" / "observatory_variants.yaml"
DEFAULT_PENDING_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "pending"
DEFAULT_LOG_PATH = (
    PROJECT_ROOT / "data" / "analysis" / "experiments" / "experiment_log.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify_value(value: Any) -> str:
    """Convert a parameter value to a filesystem-safe slug fragment.

    Args:
        value: A scalar or small-collection parameter value.

    Returns:
        Lowercase string safe for use in filenames and experiment IDs.
    """
    s = str(value).lower().replace(".", "p").replace(" ", "-")
    return "".join(c for c in s if c.isalnum() or c in ("-", "_"))


def _normalize_config_delta(parameter: str, value: Any) -> dict[str, Any]:
    """Build a config_delta dict from a single parameter name and value.

    Handles the special case of ``boom_period_dampening`` where the YAML
    stores string-keyed period dicts (e.g. ``{"2005-2010": 0.5}``).

    Args:
        parameter: The MethodConfig parameter name.
        value: The value from the catalog YAML.

    Returns:
        A dict suitable for the ``config_delta`` field in a spec file.
    """
    return {parameter: value}


# ---------------------------------------------------------------------------
# VariantCatalog
# ---------------------------------------------------------------------------


class VariantCatalog:
    """Programmatic interface to the observatory variant catalog.

    Loads variant definitions from YAML and cross-references them with the
    experiment log to determine tested/untested status.

    Args:
        catalog_path: Path to the ``observatory_variants.yaml`` file.
        experiment_log: Pre-loaded experiment log DataFrame, or None to
            load from the default CSV path.

    Raises:
        FileNotFoundError: If the catalog YAML file does not exist.
        ValueError: If the catalog is malformed (missing required keys).
    """

    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        experiment_log: pd.DataFrame | None = None,
    ) -> None:
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Catalog must be a YAML mapping: {catalog_path}")

        self._raw = raw
        self._catalog_path = catalog_path
        self._base_method: str = raw.get("base_method", "m2026r1")
        self._base_config: str = raw.get("base_config", "cfg-20260309-college-fix-v1")
        self._variants: dict[str, dict[str, Any]] = raw.get("variants", {})
        self._grids: dict[str, dict[str, Any]] = raw.get("grids", {})
        self._parameter_bounds: dict[str, dict[str, Any]] = raw.get(
            "parameter_bounds", {}
        )
        self._runtime_params: frozenset[str] = get_runtime_injectable_parameters()

        # Load or accept the experiment log
        if experiment_log is not None:
            self._log = experiment_log
        else:
            self._log = self._load_default_log()

    @staticmethod
    def _load_default_log() -> pd.DataFrame:
        """Load the experiment log from the default path.

        Returns:
            DataFrame with experiment log contents, or empty DataFrame
            if the log file does not exist.
        """
        if not DEFAULT_LOG_PATH.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(DEFAULT_LOG_PATH, dtype=str)
        except Exception:
            return pd.DataFrame()

    def _matching_log_rows(self, config_delta: dict[str, Any]) -> pd.DataFrame:
        """Return experiment-log rows that match a given config delta."""
        if self._log.empty or "config_delta_summary" not in self._log.columns:
            return pd.DataFrame()
        mask = self._log["config_delta_summary"].fillna("").map(
            lambda summary: _match_config_delta(summary, config_delta)
        )
        return self._log.loc[mask].copy()

    def _runtime_validation(self, parameter: str) -> tuple[bool, str]:
        """Validate that a catalog parameter is injectable at runtime."""
        if parameter in self._runtime_params:
            return True, ""
        return (
            False,
            (
                f"Parameter '{parameter}' is not present in the live MethodConfig "
                "runtime contract."
            ),
        )

    def _variant_record(self, variant_id: str, vdef: dict[str, Any]) -> dict[str, Any]:
        """Build the normalized record used by list/query methods."""
        config_delta = _normalize_config_delta(vdef["parameter"], vdef["value"])
        matching_log = self._matching_log_rows(config_delta)
        log_outcome = ""
        if not matching_log.empty and "outcome" in matching_log.columns:
            outcomes = matching_log["outcome"].dropna().astype(str)
            if not outcomes.empty:
                log_outcome = outcomes.iloc[-1]

        catalog_status = ""
        if "results" in vdef:
            catalog_status = str(vdef["results"].get("status", "") or "")

        tested = bool(catalog_status) or not matching_log.empty
        runtime_injectable, validation_error = self._runtime_validation(
            str(vdef.get("parameter", ""))
        )
        config_only = bool(vdef.get("config_only", True))
        runnable_without_code_change = config_only and runtime_injectable
        requires_code_change = not runnable_without_code_change
        resolved_status = resolve_observatory_status(
            experiment_outcome=log_outcome,
            catalog_status=catalog_status,
        )

        return {
            "variant_id": variant_id,
            "name": vdef.get("name", ""),
            "parameter": vdef.get("parameter", ""),
            "value": vdef.get("value"),
            "tier": vdef.get("tier", 99),
            "config_only": config_only,
            "runtime_injectable": runtime_injectable,
            "runnable_without_code_change": runnable_without_code_change,
            "requires_code_change": requires_code_change,
            "tested": tested,
            "status": catalog_status,
            "resolved_status": resolved_status,
            "log_outcome": log_outcome,
            "validation_error": validation_error,
            "hypothesis": (vdef.get("hypothesis", "") or "").strip(),
            **vdef,
        }

    def validate_grid(self, grid_id: str) -> None:
        """Validate that every parameter in a grid is runtime-injectable."""
        gdef = self.get_grid(grid_id)
        invalid = sorted(
            parameter
            for parameter in gdef.get("parameters", {})
            if parameter not in self._runtime_params
        )
        if invalid:
            raise ValueError(
                "Grid contains non-injectable parameter(s): "
                + ", ".join(invalid)
            )

    # -----------------------------------------------------------------
    # Query methods
    # -----------------------------------------------------------------

    @property
    def base_method(self) -> str:
        """The base method ID that all variants modify."""
        return self._base_method

    @property
    def base_config(self) -> str:
        """The base config ID that all variants build upon."""
        return self._base_config

    @property
    def variant_ids(self) -> list[str]:
        """Sorted list of all variant IDs in the catalog."""
        return sorted(self._variants.keys())

    @property
    def grid_ids(self) -> list[str]:
        """Sorted list of all grid IDs in the catalog."""
        return sorted(self._grids.keys())

    def _is_tested(self, variant: dict[str, Any]) -> bool:
        """Check if a variant has been tested by matching its config_delta.

        Args:
            variant: A variant definition dict from the catalog.

        Returns:
            True if the variant's parameter/value combination appears
            in the experiment log.
        """
        # If results are recorded directly in the catalog, consider it tested
        if "results" in variant and variant["results"].get("status"):
            return True

        config_delta = _normalize_config_delta(variant["parameter"], variant["value"])
        return not self._matching_log_rows(config_delta).empty

    def list_variants(self) -> pd.DataFrame:
        """Return all variants as a DataFrame.

        Columns include candidate metadata, reconciled status, and runtime
        validation fields used to distinguish total untested entries from
        runnable config-only variants.

        Returns:
            DataFrame with one row per variant, sorted by tier then ID.
        """
        rows = [self._variant_record(vid, vdef) for vid, vdef in self._variants.items()]

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["tier", "variant_id"]).reset_index(drop=True)
        return df

    def get_untested(self) -> list[dict[str, Any]]:
        """Return variant definitions that have not been tested.

        Only includes variants that are both untested and runnable via the
        current runtime injection contract.

        Returns:
            List of variant definition dicts for untested, runnable variants.
        """
        variants_df = self.list_variants()
        if variants_df.empty:
            return []

        runnable = variants_df[
            variants_df["resolved_status"].eq("untested")
            & variants_df["runnable_without_code_change"].eq(True)
        ]
        return runnable.to_dict("records")

    def get_variant(self, variant_id: str) -> dict[str, Any]:
        """Return the full definition for a single variant.

        Args:
            variant_id: The variant ID (e.g. ``"exp-b"``).

        Returns:
            Dict with variant definition including hypothesis, risk areas,
            tier, config_only flag, and results (if tested).

        Raises:
            KeyError: If the variant ID is not found in the catalog.
        """
        if variant_id not in self._variants:
            raise KeyError(
                f"Variant '{variant_id}' not found. "
                f"Available: {sorted(self._variants.keys())}"
            )
        vdef = self._variants[variant_id]
        return self._variant_record(variant_id, vdef)

    def get_grid(self, grid_id: str) -> dict[str, Any]:
        """Return the full definition for a grid.

        Args:
            grid_id: The grid ID (e.g. ``"blend-factor-sweep"``).

        Returns:
            Dict with grid definition including parameters and hypothesis.

        Raises:
            KeyError: If the grid ID is not found in the catalog.
        """
        if grid_id not in self._grids:
            raise KeyError(
                f"Grid '{grid_id}' not found. "
                f"Available: {sorted(self._grids.keys())}"
            )
        return {"grid_id": grid_id, **self._grids[grid_id]}

    def get_inventory_summary(self) -> dict[str, Any]:
        """Return tested/untested inventory counts for CLI status reporting."""
        variants = self.list_variants()
        if variants.empty:
            return {
                "total": 0,
                "tested": 0,
                "untested_total": 0,
                "untested_runnable": 0,
                "untested_requires_code_change": 0,
                "untested_ids": [],
                "untested_runnable_ids": [],
                "untested_requires_code_change_ids": [],
            }

        untested = variants[variants["resolved_status"] == "untested"]
        runnable = untested[untested["runnable_without_code_change"]]
        requires_code = untested[~untested["runnable_without_code_change"]]
        return {
            "total": int(len(variants)),
            "tested": int((variants["resolved_status"] != "untested").sum()),
            "untested_total": int(len(untested)),
            "untested_runnable": int(len(runnable)),
            "untested_requires_code_change": int(len(requires_code)),
            "untested_ids": sorted(untested["variant_id"].astype(str).str.upper().tolist()),
            "untested_runnable_ids": sorted(
                runnable["variant_id"].astype(str).str.upper().tolist()
            ),
            "untested_requires_code_change_ids": sorted(
                requires_code["variant_id"].astype(str).str.upper().tolist()
            ),
        }

    # -----------------------------------------------------------------
    # Parameter bounds
    # -----------------------------------------------------------------

    def get_bounds(self, parameter: str) -> dict[str, Any] | None:
        """Return min/max/description bounds for a parameter, or None.

        Args:
            parameter: The MethodConfig parameter name.

        Returns:
            A dict with ``min``, ``max``, and ``description`` keys, or
            ``None`` if no bounds are defined for the parameter.
        """
        bounds = self._parameter_bounds.get(parameter)
        if bounds is None:
            return None
        return dict(bounds)  # defensive copy

    def clamp_value(self, parameter: str, value: float) -> float:
        """Clamp *value* to the defined bounds for *parameter*.

        If no bounds are defined, the value is returned unchanged.

        Args:
            parameter: The MethodConfig parameter name.
            value: The numeric value to clamp.

        Returns:
            The clamped value.
        """
        bounds = self._parameter_bounds.get(parameter)
        if bounds is None:
            return value
        lo = bounds.get("min")
        hi = bounds.get("max")
        if lo is not None and value < lo:
            return float(lo)
        if hi is not None and value > hi:
            return float(hi)
        return value

    # -----------------------------------------------------------------
    # Spec generation
    # -----------------------------------------------------------------

    def _build_spec(
        self,
        experiment_id: str,
        hypothesis: str,
        config_delta: dict[str, Any],
        benchmark_label: str,
        expected_improvement: list[str] | None = None,
        risk_areas: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build an experiment spec dict following the canonical schema.

        Args:
            experiment_id: Unique experiment identifier.
            hypothesis: Falsifiable hypothesis statement.
            config_delta: Parameter overrides from the base config.
            benchmark_label: Human-readable label for the run.
            expected_improvement: Metrics expected to improve.
            risk_areas: Known risk areas.
            notes: Additional context.

        Returns:
            A dict ready to be serialized to YAML as an experiment spec.
        """
        spec: dict[str, Any] = {
            "experiment_id": experiment_id,
            "hypothesis": hypothesis.strip(),
            "base_method": self._base_method,
            "base_config": self._base_config,
            "config_delta": config_delta,
            "scope": "county",
            "benchmark_label": benchmark_label,
            "requested_by": "agent",
        }
        if expected_improvement:
            spec["expected_improvement"] = expected_improvement
        if risk_areas:
            spec["risk_areas"] = risk_areas
        if notes:
            spec["notes"] = notes
        return spec

    def generate_spec(
        self,
        variant_id: str,
        output_dir: Path = DEFAULT_PENDING_DIR,
    ) -> Path:
        """Write a single experiment spec YAML for the given variant.

        Args:
            variant_id: The variant ID from the catalog.
            output_dir: Directory where the spec YAML will be written.

        Returns:
            Path to the written spec file.

        Raises:
            KeyError: If the variant ID is not found.
        """
        vdef = self.get_variant(variant_id)
        if not vdef.get("runnable_without_code_change", False):
            raise ValueError(
                vdef.get(
                    "validation_error",
                    f"Variant '{variant_id}' is not runnable without code changes.",
                )
            )
        slug = vdef.get("slug", variant_id)
        today = dt.datetime.now(tz=dt.UTC).date().strftime("%Y%m%d")
        experiment_id = f"exp-{today}-{slug}"

        config_delta = _normalize_config_delta(vdef["parameter"], vdef["value"])

        spec = self._build_spec(
            experiment_id=experiment_id,
            hypothesis=vdef.get("hypothesis", ""),
            config_delta=config_delta,
            benchmark_label=slug,
            expected_improvement=vdef.get("expected_improvement"),
            risk_areas=vdef.get("risk_areas"),
            notes=[
                f"Catalog variant {variant_id} - Tier {vdef.get('tier', '?')}",
            ],
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        spec_path = output_dir / f"{experiment_id}.yaml"
        spec_path.write_text(
            yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return spec_path

    def generate_grid_specs(
        self,
        grid_id: str,
        output_dir: Path = DEFAULT_PENDING_DIR,
    ) -> list[Path]:
        """Generate all experiment specs for a grid definition.

        Creates one spec per combination (cartesian product by default,
        or zip if the grid specifies ``mode: zip``).

        Args:
            grid_id: The grid ID from the catalog.
            output_dir: Directory where spec YAML files will be written.

        Returns:
            List of paths to generated spec files.

        Raises:
            KeyError: If the grid ID is not found.
        """
        self.validate_grid(grid_id)
        gdef = self.get_grid(grid_id)
        parameters: dict[str, list[Any]] = gdef["parameters"]
        mode = gdef.get("mode", "cartesian")
        hypothesis_base = gdef.get("hypothesis", "Grid sweep")

        # Generate combinations
        param_names = list(parameters.keys())
        param_values = [parameters[k] for k in param_names]

        if mode == "zip":
            lengths = {len(v) for v in param_values}
            if len(lengths) > 1:
                raise ValueError(
                    f"Zip mode requires equal-length parameter lists. "
                    f"Got: {[len(v) for v in param_values]}"
                )
            combinations = [
                dict(zip(param_names, combo, strict=True))
                for combo in zip(*param_values, strict=True)
            ]
        else:
            combinations = [
                dict(zip(param_names, combo, strict=True))
                for combo in itertools.product(*param_values)
            ]

        today = dt.datetime.now(tz=dt.UTC).date().strftime("%Y%m%d")
        slug_base = grid_id

        output_dir.mkdir(parents=True, exist_ok=True)
        spec_paths: list[Path] = []

        for combo in combinations:
            # Build a value-based suffix for uniqueness
            value_suffix = "-".join(_slugify_value(v) for v in combo.values())
            experiment_id = f"exp-{today}-{slug_base}-{value_suffix}"
            benchmark_label = f"{slug_base}-{value_suffix}"

            param_desc = ", ".join(f"{k}={v}" for k, v in combo.items())
            hypothesis = f"{hypothesis_base} Testing {param_desc}."

            spec = self._build_spec(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                config_delta=dict(combo),
                benchmark_label=benchmark_label,
                notes=[f"Grid '{grid_id}' — {param_desc}"],
            )

            spec_path = output_dir / f"{experiment_id}.yaml"
            spec_path.write_text(
                yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            spec_paths.append(spec_path)

        return spec_paths

    def generate_all_pending_specs(
        self,
        output_dir: Path = DEFAULT_PENDING_DIR,
    ) -> list[Path]:
        """Generate specs for all untested, config-only variants.

        Skips variants that have already been tested (determined by
        parameter-level matching against the experiment log).

        Args:
            output_dir: Directory where spec YAML files will be written.

        Returns:
            List of paths to generated spec files.
        """
        untested = self.get_untested()
        paths: list[Path] = []
        for entry in untested:
            vid = entry["variant_id"]
            path = self.generate_spec(vid, output_dir=output_dir)
            paths.append(path)
        return paths
