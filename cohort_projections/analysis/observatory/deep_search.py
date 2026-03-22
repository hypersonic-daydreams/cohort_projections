"""Deep-search policy, search-pack loading, and shared control-plane helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def resolve_parallelism(cpu_budget: int) -> tuple[int, int]:
    """Map a CPU budget to ``(parallel_runs, workers_per_run)``.

    The allocation intentionally favors enough workers per benchmark so nested
    benchmark parallelism remains effective while still allowing multiple
    candidate runs when more cores are available.
    """
    if cpu_budget < 1:
        raise ValueError("cpu_budget must be positive.")
    if cpu_budget < 8:
        parallel_runs = 1
    elif cpu_budget < 16:
        parallel_runs = 2
    elif cpu_budget < 32:
        parallel_runs = 3
    elif cpu_budget < 64:
        parallel_runs = 4
    else:
        parallel_runs = min(8, cpu_budget // 8)
    workers_per_run = max(1, cpu_budget // parallel_runs)
    return parallel_runs, workers_per_run


def default_batch_size(parallel_runs: int) -> int:
    """Return the hidden default batch size for a deep-search session."""
    return max(parallel_runs, min(8, 2 * parallel_runs))


@dataclass(frozen=True)
class CodeMutator:
    """Declarative sandbox-only code-mutator definition for a search pack."""

    mutator_id: str
    kind: str
    hypothesis: str
    candidate_id: str
    benchmark_label: str
    enabled: bool = True
    execution_mode: str = "code_recipe"
    method_id_override: str = ""
    config_delta: dict[str, Any] = field(default_factory=dict)
    recipe_steps: tuple[dict[str, Any], ...] = ()
    targeted_tests: tuple[str, ...] = ()
    prompt_template: str = ""
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CodeMutator:
        """Build one mutator from YAML data."""
        mutator_id = str(raw.get("mutator_id", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        candidate_id = str(raw.get("candidate_id", "")).strip()
        benchmark_label = str(raw.get("benchmark_label", "")).strip()
        hypothesis = str(raw.get("hypothesis", "")).strip()
        if not mutator_id or not kind or not candidate_id or not benchmark_label or not hypothesis:
            raise ValueError(
                "Each code mutator requires mutator_id, kind, candidate_id, "
                "benchmark_label, and hypothesis."
            )
        raw_steps = raw.get("recipe_steps", [])
        if raw_steps and not isinstance(raw_steps, list):
            raise ValueError("code_mutators.recipe_steps must be a list when present.")
        raw_tests = raw.get("targeted_tests", [])
        if raw_tests and not isinstance(raw_tests, list):
            raise ValueError("code_mutators.targeted_tests must be a list when present.")
        raw_notes = raw.get("notes", [])
        if raw_notes and not isinstance(raw_notes, list):
            raise ValueError("code_mutators.notes must be a list when present.")
        return cls(
            mutator_id=mutator_id,
            kind=kind,
            hypothesis=hypothesis,
            candidate_id=candidate_id,
            benchmark_label=benchmark_label,
            enabled=bool(raw.get("enabled", True)),
            execution_mode=str(raw.get("execution_mode", "code_recipe")),
            method_id_override=str(raw.get("method_id_override", "")).strip(),
            config_delta=dict(raw.get("config_delta", {}) or {}),
            recipe_steps=tuple(step for step in raw_steps if isinstance(step, dict)),
            targeted_tests=tuple(str(item) for item in raw_tests),
            prompt_template=str(raw.get("prompt_template", "")).strip(),
            notes=tuple(str(item) for item in raw_notes),
        )


@dataclass(frozen=True)
class SearchPack:
    """One deep-search pack describing objectives, seeds, and mutators."""

    pack_id: str
    label: str
    scope: str
    base_method: str
    base_config: str
    objective_order: tuple[str, ...]
    hard_guardrails: dict[str, Any]
    parameter_bounds: dict[str, Any]
    interaction_rules: tuple[dict[str, Any], ...]
    seed_candidates: tuple[dict[str, Any], ...]
    stop_policy: dict[str, Any]
    code_mutators: tuple[CodeMutator, ...] = ()
    enabled: bool = True
    default: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SearchPack:
        """Build one search pack from YAML data."""
        required = [
            "pack_id",
            "scope",
            "base_method",
            "base_config",
            "objective_order",
            "hard_guardrails",
            "parameter_bounds",
            "interaction_rules",
            "seed_candidates",
            "stop_policy",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Search pack missing required keys: {missing}")
        objective_order = tuple(str(item) for item in raw.get("objective_order", []) if item)
        if not objective_order:
            raise ValueError("Search pack objective_order cannot be empty.")
        raw_interactions = raw.get("interaction_rules", [])
        if raw_interactions and not isinstance(raw_interactions, list):
            raise ValueError("interaction_rules must be a list.")
        raw_seeds = raw.get("seed_candidates", [])
        if raw_seeds and not isinstance(raw_seeds, list):
            raise ValueError("seed_candidates must be a list.")
        raw_mutators = raw.get("code_mutators", [])
        if raw_mutators and not isinstance(raw_mutators, list):
            raise ValueError("code_mutators must be a list.")
        return cls(
            pack_id=str(raw["pack_id"]).strip(),
            label=str(raw.get("label", raw["pack_id"])).strip(),
            scope=str(raw["scope"]).strip(),
            base_method=str(raw["base_method"]).strip(),
            base_config=str(raw["base_config"]).strip(),
            objective_order=objective_order,
            hard_guardrails=dict(raw.get("hard_guardrails", {}) or {}),
            parameter_bounds=dict(raw.get("parameter_bounds", {}) or {}),
            interaction_rules=tuple(item for item in raw_interactions if isinstance(item, dict)),
            seed_candidates=tuple(item for item in raw_seeds if isinstance(item, dict)),
            stop_policy=dict(raw.get("stop_policy", {}) or {}),
            code_mutators=tuple(
                CodeMutator.from_dict(item) for item in raw_mutators if isinstance(item, dict)
            ),
            enabled=bool(raw.get("enabled", True)),
            default=bool(raw.get("default", False)),
            description=str(raw.get("description", "")).strip(),
        )


@dataclass(frozen=True)
class DeepSearchPolicy:
    """Resolved deep-search execution defaults."""

    cpu_budget: int
    parallel_runs: int
    workers_per_run: int
    batch_size: int
    time_budget_hours: float
    total_candidate_cap: int
    plateau_rounds: int
    min_improvement_pp: float
    operational_repeat_limit: int
    operational_failure_window: int
    operational_failure_rate: float
    search_pack_root: Path
    max_patch_files: int
    max_patch_lines: int

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        cpu_budget: int,
        search_pack_root: Path,
        time_budget_hours: float | None = None,
    ) -> DeepSearchPolicy:
        """Resolve a deep-search policy from configuration."""
        deep = dict(raw or {})
        parallel_runs, workers_per_run = resolve_parallelism(cpu_budget)
        resolved_time_budget = (
            float(time_budget_hours)
            if time_budget_hours is not None
            else float(deep.get("time_budget_hours", 6.0))
        )
        return cls(
            cpu_budget=cpu_budget,
            parallel_runs=parallel_runs,
            workers_per_run=workers_per_run,
            batch_size=int(deep.get("batch_size", default_batch_size(parallel_runs))),
            time_budget_hours=resolved_time_budget,
            total_candidate_cap=int(deep.get("total_candidate_cap", 48)),
            plateau_rounds=int(deep.get("plateau_rounds", 2)),
            min_improvement_pp=float(deep.get("min_improvement_pp", 0.02)),
            operational_repeat_limit=int(deep.get("operational_repeat_limit", 3)),
            operational_failure_window=int(deep.get("operational_failure_window", 10)),
            operational_failure_rate=float(deep.get("operational_failure_rate", 0.20)),
            search_pack_root=search_pack_root,
            max_patch_files=int(deep.get("max_patch_files", 5)),
            max_patch_lines=int(deep.get("max_patch_lines", 300)),
        )


def load_search_packs(pack_root: Path) -> dict[str, SearchPack]:
    """Load all enabled search packs from a directory or single YAML file."""
    if not pack_root.exists():
        return {}

    files: list[Path]
    if pack_root.is_file():
        files = [pack_root]
    else:
        files = sorted(path for path in pack_root.glob("*.yaml") if path.is_file())

    packs: dict[str, SearchPack] = {}
    for path in files:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Search pack file must contain a mapping: {path}")
        pack = SearchPack.from_dict(raw)
        if pack.enabled:
            packs[pack.pack_id] = pack
    return packs


def default_search_pack(
    packs: dict[str, SearchPack], *, scope: str | None = None
) -> SearchPack | None:
    """Return the default enabled search pack."""
    scoped = [pack for pack in packs.values() if scope is None or pack.scope == scope]
    if not scoped:
        return None
    explicit = [pack for pack in scoped if pack.default]
    if explicit:
        return sorted(explicit, key=lambda pack: pack.pack_id)[0]
    return sorted(scoped, key=lambda pack: pack.pack_id)[0]


def objective_sort_columns(pack: SearchPack) -> list[str]:
    """Return the preferred delta/value columns for pack objective ordering."""
    columns: list[str] = []
    for metric in pack.objective_order:
        delta_col = f"delta_{metric}"
        columns.extend([delta_col, metric])
    return columns


def build_frontier_frame(
    candidates: pd.DataFrame,
    *,
    pack: SearchPack,
    limit: int = 10,
) -> pd.DataFrame:
    """Return the candidate frontier sorted by the pack objective order."""
    if candidates.empty:
        return pd.DataFrame()
    sortable = candidates.copy()
    for column in objective_sort_columns(pack):
        if column not in sortable.columns:
            sortable[column] = pd.NA
    sort_cols = ["decision_priority"]
    ascending = [True]
    sortable["decision_priority"] = (
        sortable.get("decision_state", pd.Series(dtype=object))
        .map(
            {
                "recommended": 0,
                "ready_for_review": 1,
                "mixed_signal": 2,
                "blocked_by_data_or_runtime": 3,
                "failed_hard_gate": 4,
                "not_executed": 5,
            }
        )
        .fillna(6)
    )
    for metric in pack.objective_order:
        delta_col = f"delta_{metric}"
        if delta_col in sortable.columns:
            sort_cols.append(delta_col)
            ascending.append(True)
        elif metric in sortable.columns:
            sort_cols.append(metric)
            ascending.append(True)
    if "candidate_id" in sortable.columns:
        sort_cols.append("candidate_id")
        ascending.append(True)
    ordered = sortable.sort_values(sort_cols, ascending=ascending, na_position="last")
    return ordered.head(limit).reset_index(drop=True)
