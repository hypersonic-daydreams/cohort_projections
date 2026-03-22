"""Deterministic, worktree-isolated search orchestration for the Observatory."""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from cohort_projections.analysis.benchmarking import append_benchmark_index
from cohort_projections.analysis.experiment_log import (
    append_experiment_entry,
    get_tested_hypotheses,
    read_experiment_log,
)
from cohort_projections.analysis.observatory.ai_synthesis import (
    build_evidence_payload,
    synthesize_observatory_summary,
)
from cohort_projections.analysis.observatory.comparator import ObservatoryComparator
from cohort_projections.analysis.observatory.decision_support import (
    build_search_candidate_rows,
    build_search_session_summary,
)
from cohort_projections.analysis.observatory.deep_search import (
    DeepSearchPolicy,
    SearchPack,
    build_frontier_frame,
    default_search_pack,
    load_search_packs,
)
from cohort_projections.analysis.observatory.recipe_registry import RecipeRegistry
from cohort_projections.analysis.observatory.recommender import ObservatoryRecommender
from cohort_projections.analysis.observatory.sandbox_manager import SandboxManager
from cohort_projections.analysis.observatory.search_policy import (
    PROJECT_ROOT,
    SearchPolicy,
    load_search_policy,
)
from cohort_projections.analysis.observatory.variant_catalog import VariantCatalog

logger = logging.getLogger(__name__)


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat()


def _slugify(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value.lower()
    ).strip("-_")


def _predict_method_id(spec: dict[str, Any]) -> str:
    return str(spec.get("method_id_override") or spec["base_method"])


def _predict_config_id(spec: dict[str, Any]) -> str:
    if spec.get("config_id_override"):
        return str(spec["config_id_override"])
    today = dt.datetime.now(tz=dt.UTC).date().strftime("%Y%m%d")
    experiment_id = str(spec["experiment_id"])
    parts = experiment_id.split("-", 3)
    if len(parts) >= 4:
        slug = parts[3]
    elif len(parts) == 3:
        slug = parts[2]
    else:
        slug = experiment_id
    return f"cfg-{today}-{slug}"


def _spec_fingerprint(
    *,
    execution_mode: str,
    spec: dict[str, Any],
    recipe_id: str | None = None,
) -> str:
    payload = {
        "execution_mode": execution_mode,
        "base_method": spec.get("base_method"),
        "method_id_override": spec.get("method_id_override"),
        "config_delta": spec.get("config_delta", {}),
        "recipe_id": recipe_id or "",
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _recommendation_config_delta(rec: Any) -> dict[str, Any] | None:
    suggested_value = getattr(rec, "suggested_value", None)
    parameter = getattr(rec, "parameter", "")
    if isinstance(suggested_value, dict):
        return dict(suggested_value)
    if suggested_value is None or not parameter:
        return None
    return {str(parameter): suggested_value}


def _recommendation_spec(
    rec: Any,
    *,
    base_method: str,
    base_config: str,
    requested_by: str,
) -> dict[str, Any] | None:
    config_delta = _recommendation_config_delta(rec)
    if not config_delta:
        return None
    today = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d")
    fragments = [
        f"{_slugify(str(key))}-{_slugify(str(value))}"
        for key, value in sorted(config_delta.items(), key=lambda item: str(item[0]))
    ]
    slug = "-".join(fragments) or "recommendation"
    notes = [
        "Generated from deterministic Observatory recommendation planning.",
        f"Recommendation parameter: {getattr(rec, 'parameter', 'unknown')}",
    ]
    return {
        "experiment_id": f"exp-{today}-search-{slug}",
        "hypothesis": str(getattr(rec, "rationale", "")).strip()
        or "Recommendation generated from prior benchmark results.",
        "base_method": base_method,
        "base_config": base_config,
        "config_delta": config_delta,
        "scope": "county",
        "benchmark_label": f"search-{slug}",
        "requested_by": requested_by,
        "notes": notes,
    }


class AutonomousSearchController:
    """Plan, execute, and report deterministic autonomous search sessions."""

    def __init__(
        self,
        *,
        store: Any | None,
        observatory_config: dict[str, Any] | None,
        policy_path: Path | None = None,
        project_root: Path = PROJECT_ROOT,
        source_repo: Path | None = None,
        synthesis_provider: Any | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.store = store
        self.observatory_config = observatory_config or {}
        self.synthesis_provider = synthesis_provider
        self.policy: SearchPolicy = load_search_policy(policy_path, project_root=self.project_root)
        self.recipe_registry = RecipeRegistry(self.policy.recipe_catalog)
        self.sandbox_manager = SandboxManager(
            self.policy, source_repo=(source_repo or self.project_root)
        )
        self.search_packs = load_search_packs(self.policy.search_pack_root)
        self.history_dir = _resolve_project_path(
            self.project_root,
            self.observatory_config.get("history_dir", "data/analysis/benchmark_history"),
        )
        self.experiment_log_path = _resolve_project_path(
            self.project_root,
            self.observatory_config.get(
                "experiment_log",
                "data/analysis/experiments/experiment_log.csv",
            ),
        )
        self.variant_catalog_path = _resolve_project_path(
            self.project_root,
            self.observatory_config.get(
                "variant_catalog",
                "config/observatory_variants.yaml",
            ),
        )

    def resolve_search_pack(
        self, search_pack_id: str | None, *, scope: str = "county"
    ) -> SearchPack | None:
        """Resolve the requested search pack or the default pack for the scope."""
        if search_pack_id:
            return self.search_packs.get(search_pack_id)
        return default_search_pack(self.search_packs, scope=scope)

    def build_deep_search_policy(
        self,
        *,
        cpu_budget: int | None = None,
        time_budget_hours: float | None = None,
    ) -> DeepSearchPolicy:
        """Resolve the deep-search policy for one session."""
        observed_cores = max(2, int(self.observatory_config.get("default_cpu_budget", 12)))
        resolved_budget = max(2, int(cpu_budget or observed_cores))
        return DeepSearchPolicy.from_mapping(
            self.policy.deep_search,
            cpu_budget=resolved_budget,
            search_pack_root=self.policy.search_pack_root,
            time_budget_hours=time_budget_hours,
        )

    def plan_session(
        self,
        *,
        search_id: str,
        base_revision: str = "HEAD",
        search_pack_id: str | None = None,
        cpu_budget: int | None = None,
        time_budget_hours: float | None = None,
        max_pending: int | None = None,
        max_recommended: int | None = None,
        include_recipe_catalog: bool | None = None,
        overwrite: bool = False,
        allow_ai_synthesis: bool = False,
    ) -> dict[str, Any]:
        """Create a deterministic search session from the current Observatory state."""
        session_dir = self._session_dir(search_id)
        if session_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Search session already exists: {session_dir}")
            shutil.rmtree(session_dir)
        planned_specs_dir = session_dir / "planned_specs"
        planned_specs_dir.mkdir(parents=True, exist_ok=True)

        catalog = self._load_variant_catalog()
        search_pack = self.resolve_search_pack(search_pack_id, scope="county")
        deep_policy = self.build_deep_search_policy(
            cpu_budget=cpu_budget,
            time_budget_hours=time_budget_hours,
        )
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        requested_by = "deep_search"
        logged_experiment_ids = get_tested_hypotheses(self.experiment_log_path)

        resolved_base_revision = self._resolve_git_revision(base_revision)
        pending_limit = self.policy.default_max_pending if max_pending is None else max_pending
        recommended_limit = (
            self.policy.default_max_recommended if max_recommended is None else max_recommended
        )
        include_recipes = (
            self.policy.include_recipe_catalog
            if include_recipe_catalog is None
            else include_recipe_catalog
        )
        seed_variant_ids = self._seed_source_ids(search_pack, source="variant_catalog")
        seed_recipe_ids = self._seed_source_ids(search_pack, source="recipe_catalog")

        if catalog is not None and pending_limit > 0:
            ordered = sorted(
                catalog.get_untested(),
                key=lambda variant: (
                    0 if str(variant.get("variant_id", "")).lower() in seed_variant_ids else 1,
                    int(variant.get("tier", 999)),
                    str(variant.get("variant_id", "")),
                ),
            )
            for variant in ordered[:pending_limit]:
                spec_path = catalog.generate_spec(
                    str(variant["variant_id"]),
                    output_dir=planned_specs_dir,
                    requested_by=requested_by,
                    extra_notes=[
                        "Generated from deterministic Observatory search planning.",
                    ],
                )
                spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
                if str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                fingerprint = _spec_fingerprint(
                    execution_mode="config_only",
                    spec=spec,
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                candidates.append(
                    self._build_candidate_record(
                        candidate_id=f"variant-{variant['variant_id'].lower()}",
                        source="variant_catalog",
                        source_id=str(variant["variant_id"]),
                        execution_mode="config_only",
                        spec=spec,
                        spec_path=spec_path,
                    )
                )

        if self.store is not None and recommended_limit > 0:
            comparator = ObservatoryComparator(store=self.store, config=self.observatory_config)
            catalog_for_bounds = catalog or self._load_variant_catalog()
            recommender = ObservatoryRecommender(
                store=self.store,
                comparator=comparator,
                config=self.observatory_config,
                bounds_catalog=catalog_for_bounds,
            )
            base_method = getattr(
                catalog_for_bounds, "base_method", None
            ) or self.observatory_config.get("challenger_base_method", "m2026r1")
            base_config = getattr(
                catalog_for_bounds, "base_config", None
            ) or self.observatory_config.get("base_config", "cfg-20260309-college-fix-v1")
            recommendations = [
                rec
                for rec in recommender.suggest_next_experiments(n=recommended_limit * 3)
                if not getattr(rec, "requires_code_change", False)
            ]
            for rec in recommendations:
                spec = _recommendation_spec(
                    rec,
                    base_method=base_method,
                    base_config=base_config,
                    requested_by=requested_by,
                )
                if spec is None:
                    continue
                if str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                fingerprint = _spec_fingerprint(execution_mode="config_only", spec=spec)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                spec_path = planned_specs_dir / f"{spec['experiment_id']}.yaml"
                spec_path.write_text(
                    yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                candidates.append(
                    self._build_candidate_record(
                        candidate_id=f"rec-{_slugify(spec['experiment_id'])}",
                        source="recommendation",
                        source_id=str(getattr(rec, "parameter", "recommendation")),
                        execution_mode="config_only",
                        spec=spec,
                        spec_path=spec_path,
                    )
                )
                if (
                    len([c for c in candidates if c["source"] == "recommendation"])
                    >= recommended_limit
                ):
                    break

        if include_recipes:
            context = {
                "search_id": search_id,
                "date": dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d"),
            }
            enabled_recipes = sorted(
                self.recipe_registry.list_enabled(),
                key=lambda item: (0 if item[0].lower() in seed_recipe_ids else 1, item[0]),
            )
            for recipe_id, recipe in enabled_recipes:
                execution_mode = str(recipe.get("execution_mode", "code_recipe"))
                spec = self.recipe_registry.build_candidate_spec(
                    recipe_id,
                    recipe,
                    requested_by=requested_by,
                    context=context,
                )
                if execution_mode == "code_recipe" and not (
                    spec.get("method_id_override") or recipe.get("allow_shared_method_patch", False)
                ):
                    continue
                if str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                fingerprint = _spec_fingerprint(
                    execution_mode=execution_mode,
                    spec=spec,
                    recipe_id=recipe_id,
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                spec_path = planned_specs_dir / f"{spec['experiment_id']}.yaml"
                spec_path.write_text(
                    yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                candidates.append(
                    self._build_candidate_record(
                        candidate_id=str(recipe.get("candidate_id", f"recipe-{recipe_id.lower()}")),
                        source="recipe_catalog",
                        source_id=recipe_id,
                        execution_mode=execution_mode,
                        spec=spec,
                        spec_path=spec_path,
                        recipe_id=recipe_id,
                        recipe_parameters=recipe.get("parameters", {}),
                    )
                )

        if search_pack is not None:
            for mutator in search_pack.code_mutators:
                if not mutator.enabled:
                    continue
                if mutator.kind not in {"deterministic", "model_generated"}:
                    continue
                spec = {
                    "experiment_id": f"exp-{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d')}-{mutator.candidate_id}",
                    "hypothesis": mutator.hypothesis,
                    "base_method": search_pack.base_method,
                    "base_config": search_pack.base_config,
                    "config_delta": dict(mutator.config_delta),
                    "scope": search_pack.scope,
                    "benchmark_label": mutator.benchmark_label,
                    "requested_by": requested_by,
                    "method_id_override": mutator.method_id_override or None,
                    "notes": list(mutator.notes),
                    "targeted_tests": list(mutator.targeted_tests),
                    "code_mutator_kind": mutator.kind,
                    "code_mutator_id": mutator.mutator_id,
                    "code_mutator_steps": [dict(step) for step in mutator.recipe_steps],
                    "code_mutator_prompt_template": mutator.prompt_template,
                }
                spec_path = planned_specs_dir / f"{spec['experiment_id']}.yaml"
                spec_path.write_text(
                    yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                fingerprint = _spec_fingerprint(
                    execution_mode="code_recipe",
                    spec=spec,
                    recipe_id=mutator.mutator_id,
                )
                if fingerprint in seen or str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                seen.add(fingerprint)
                candidates.append(
                    self._build_candidate_record(
                        candidate_id=mutator.candidate_id,
                        source="search_pack",
                        source_id=mutator.mutator_id,
                        execution_mode="code_recipe",
                        spec=spec,
                        spec_path=spec_path,
                        recipe_id=mutator.mutator_id,
                        recipe_parameters={
                            "steps": [dict(step) for step in mutator.recipe_steps],
                            "targeted_tests": list(mutator.targeted_tests),
                            "kind": mutator.kind,
                        },
                    )
                )

        summary = self._summarize_candidates(candidates)
        session = {
            "search_id": search_id,
            "mode": "deep_search",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "base_revision": base_revision,
            "resolved_base_revision": resolved_base_revision,
            "policy_path": str(self.policy.policy_path),
            "history_dir": str(self.history_dir),
            "experiment_log": str(self.experiment_log_path),
            "search_pack_id": search_pack.pack_id if search_pack is not None else "",
            "search_pack_scope": search_pack.scope if search_pack is not None else "",
            "search_pack_objective_order": list(search_pack.objective_order) if search_pack else [],
            "cpu_budget": deep_policy.cpu_budget,
            "parallel_runs": deep_policy.parallel_runs,
            "workers_per_run": deep_policy.workers_per_run,
            "time_budget_hours": deep_policy.time_budget_hours,
            "allow_ai_synthesis": allow_ai_synthesis,
            "stop_reason": "",
            "replan_rounds": 0,
            "planner": {
                "max_pending": pending_limit,
                "max_recommended": recommended_limit,
                "include_recipe_catalog": include_recipes,
            },
            "deep_search_policy": {
                "batch_size": deep_policy.batch_size,
                "total_candidate_cap": deep_policy.total_candidate_cap,
                "plateau_rounds": deep_policy.plateau_rounds,
                "min_improvement_pp": deep_policy.min_improvement_pp,
                "operational_repeat_limit": deep_policy.operational_repeat_limit,
                "operational_failure_window": deep_policy.operational_failure_window,
                "operational_failure_rate": deep_policy.operational_failure_rate,
                "max_patch_files": deep_policy.max_patch_files,
                "max_patch_lines": deep_policy.max_patch_lines,
            },
            "candidates": candidates,
            "summary": summary,
        }
        session["status"] = "finished" if int(summary.get("planned", 0)) == 0 else "planned"
        self._write_session(search_id, session)
        self._append_journal(
            search_id,
            {
                "event": "planned",
                "candidate_count": summary["total"],
                "search_pack_id": session["search_pack_id"],
                "cpu_budget": session["cpu_budget"],
            },
        )
        return session

    def run_session(
        self,
        *,
        search_id: str,
        run_budget: int | None = None,
        dry_run: bool = False,
        keep_worktrees: bool | None = None,
        workers_per_run: int = 0,
        parallel_runs: int | None = None,
    ) -> dict[str, Any]:
        """Execute queued candidates inside isolated worktrees.

        When *parallel_runs* > 1, candidates are executed concurrently using
        a thread pool.  Each candidate gets its own worktree, so filesystem
        isolation is maintained.  Session state is guarded by a lock.
        """
        session = self.load_session(search_id)
        session_policy = dict(session.get("deep_search_policy", {}) or {})
        session_parallel = int(session.get("parallel_runs", 0) or 0)
        session_workers = int(session.get("workers_per_run", 0) or 0)
        budget = (
            run_budget
            if run_budget is not None
            else int(session_policy.get("batch_size", self.policy.default_run_budget))
        )
        keep = self.policy.keep_worktrees if keep_worktrees is None else keep_worktrees
        effective_parallel = (
            parallel_runs
            if parallel_runs is not None
            else (session_parallel or self.policy.default_parallel_runs)
        )
        effective_parallel = max(1, effective_parallel)
        effective_workers = workers_per_run if workers_per_run > 0 else session_workers

        runnable = [
            candidate for candidate in session["candidates"] if candidate["status"] == "planned"
        ]
        if dry_run:
            return {
                "search_id": search_id,
                "mode": "dry-run",
                "runnable": len(runnable),
                "run_budget": budget,
                "preview": [candidate["candidate_id"] for candidate in runnable[:budget]],
            }

        self.sandbox_manager.assert_live_checkout_clean()
        live_signature = self.sandbox_manager.live_checkout_signature()

        # Pre-refresh the bare mirror once so parallel worktree creation
        # doesn't redundantly fetch from the source repo.
        self.sandbox_manager.ensure_mirror()

        # --- Phase 1: skip already-logged candidates (sequential, fast) ---
        to_execute: list[dict[str, Any]] = []
        for candidate in session["candidates"]:
            if candidate["status"] != "planned":
                continue
            existing_log_row = self._latest_logged_experiment_entry(
                str(candidate["spec"]["experiment_id"])
            )
            if existing_log_row is not None:
                candidate["status"] = "completed"
                candidate["completed_at"] = _utc_now()
                candidate["result"] = self._logged_candidate_result(
                    candidate=candidate,
                    log_row=existing_log_row,
                )
                candidate["last_error"] = ""
                self._persist_session_update(session)
                continue
            if budget is not None and len(to_execute) >= budget:
                break
            to_execute.append(candidate)

        executed = len(to_execute)

        # Mark all candidates as running before execution starts
        session_lock = threading.Lock()
        for candidate in to_execute:
            candidate["status"] = "running"
            candidate["attempts"] = int(candidate.get("attempts", 0)) + 1
            candidate["started_at"] = _utc_now()
        self._persist_session_update(session)

        # --- Phase 2: execute candidates ---
        def _execute_candidate(candidate: dict[str, Any]) -> None:
            context = self.sandbox_manager.create_worktree(
                search_id=search_id,
                candidate_id=str(candidate["candidate_id"]),
                base_revision=str(session["resolved_base_revision"]),
            )
            try:
                applied_recipe: dict[str, Any] | None = None
                if candidate["execution_mode"] == "code_recipe" and candidate.get("recipe_id"):
                    applied = self.recipe_registry.apply(
                        str(candidate["recipe_id"]),
                        worktree_root=context.worktree_path,
                        policy=self.policy,
                        parameters=dict(candidate.get("recipe_parameters") or {}),
                    )
                    applied_recipe = {
                        "recipe_id": applied.recipe_id,
                        "steps_applied": applied.steps_applied,
                        "parameters": applied.parameters,
                    }

                changed_files, diff_patch = self.sandbox_manager.capture_diff(
                    worktree_path=context.worktree_path,
                    base_revision=context.base_revision,
                )
                if self.policy.compile_changed_python:
                    self.sandbox_manager.compile_changed_python(
                        context.worktree_path, changed_files
                    )
                spec_path = self._write_spec_to_worktree(context.worktree_path, candidate["spec"])
                targeted_tests = list(candidate.get("spec", {}).get("targeted_tests", []) or [])
                self._validate_code_changes(
                    changed_files=changed_files,
                    diff_patch=diff_patch,
                    max_patch_files=int(session_policy.get("max_patch_files", 5)),
                    max_patch_lines=int(session_policy.get("max_patch_lines", 300)),
                )
                self._run_targeted_tests(
                    worktree_path=context.worktree_path,
                    targeted_tests=targeted_tests,
                )
                validation_report = {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "execution_mode": str(candidate.get("execution_mode", "") or ""),
                    "changed_files": list(changed_files),
                    "patch_file_count": len(changed_files),
                    "patch_line_count": self._count_changed_lines(diff_patch),
                    "targeted_tests": list(targeted_tests),
                    "status": "passed",
                }
                command_result = self._run_worktree_experiment(
                    context.worktree_path,
                    spec_path,
                    workers_per_run=effective_workers,
                )
                artifacts = self._harvest_candidate(
                    search_id=search_id,
                    candidate=candidate,
                    context=context,
                    changed_files=changed_files,
                    diff_patch=diff_patch,
                    command_result=command_result,
                    applied_recipe=applied_recipe,
                    validation_report=validation_report,
                )
                candidate["status"] = "completed"
                candidate["completed_at"] = _utc_now()
                candidate["result"] = artifacts
                candidate["last_error"] = ""
            except Exception as exc:
                candidate["status"] = "completed"
                candidate["completed_at"] = _utc_now()
                candidate["last_error"] = str(exc)
                candidate["result"] = self._build_operational_blocker_result(
                    search_id=search_id,
                    candidate=candidate,
                    error=str(exc),
                )
            finally:
                with session_lock:
                    self._persist_session_update(session)
                if not keep:
                    self.sandbox_manager.remove_worktree(context)
                self.sandbox_manager.assert_live_checkout_unchanged(live_signature)

        if effective_parallel <= 1 or len(to_execute) <= 1:
            for candidate in to_execute:
                _execute_candidate(candidate)
        else:
            logger.info(
                "Running %d candidates with %d parallel workers.",
                len(to_execute),
                effective_parallel,
            )
            with ThreadPoolExecutor(max_workers=effective_parallel) as pool:
                futures = {pool.submit(_execute_candidate, c): c for c in to_execute}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        cand = futures[future]
                        logger.error(
                            "Unhandled error in candidate %s: %s",
                            cand.get("candidate_id", "?"),
                            exc,
                        )

        session["updated_at"] = _utc_now()
        if all(
            candidate["status"] in {"completed", "failed"} for candidate in session["candidates"]
        ):
            session["status"] = "finished"
        else:
            session["status"] = "planned"
        session["summary"] = self._summarize_candidates(session["candidates"])
        self._write_session(search_id, session)
        return {
            "search_id": search_id,
            "executed": executed,
            "summary": session["summary"],
        }

    def run_to_completion(
        self,
        *,
        search_id: str,
        base_revision: str = "HEAD",
        search_pack_id: str | None = None,
        cpu_budget: int | None = None,
        time_budget_hours: float | None = None,
        max_pending: int | None = None,
        max_recommended: int | None = None,
        include_recipe_catalog: bool | None = None,
        overwrite: bool = False,
        batch_run_budget: int | None = None,
        max_batches: int | None = None,
        max_total_runs: int | None = None,
        keep_worktrees: bool | None = None,
        workers_per_run: int = 0,
        parallel_runs: int | None = None,
        allow_ai_synthesis: bool = False,
    ) -> dict[str, Any]:
        """Plan and execute a search session until exhausted or budget-limited."""
        self.plan_session(
            search_id=search_id,
            base_revision=base_revision,
            search_pack_id=search_pack_id,
            cpu_budget=cpu_budget,
            time_budget_hours=time_budget_hours,
            max_pending=max_pending,
            max_recommended=max_recommended,
            include_recipe_catalog=include_recipe_catalog,
            overwrite=overwrite,
            allow_ai_synthesis=allow_ai_synthesis,
        )
        session = self.load_session(search_id)
        session_policy = dict(session.get("deep_search_policy", {}) or {})
        deep_policy = self.build_deep_search_policy(
            cpu_budget=int(session.get("cpu_budget", 0) or cpu_budget or 12),
            time_budget_hours=float(
                session.get("time_budget_hours", 0) or time_budget_hours or 6.0
            ),
        )
        search_pack = self.resolve_search_pack(str(session.get("search_pack_id", "") or "") or None)
        run_budget = (
            batch_run_budget
            if batch_run_budget is not None
            else int(session_policy.get("batch_size", deep_policy.batch_size))
        )
        total_executed = 0
        rounds: list[dict[str, Any]] = []
        batch_number = 0
        best_primary_delta: float | None = None
        stalled_rounds = 0
        stop_reason = ""
        started_at = time.monotonic()
        candidate_cap = (
            max_total_runs if max_total_runs is not None else deep_policy.total_candidate_cap
        )

        while True:
            current = self.load_session(search_id)
            planned_remaining = int(current["summary"]["planned"])
            if planned_remaining <= 0:
                added = self._replan_session(
                    search_id=search_id,
                    search_pack=search_pack,
                    max_pending=max_pending,
                    max_recommended=max_recommended,
                    include_recipe_catalog=include_recipe_catalog,
                    candidate_cap=candidate_cap,
                )
                current = self.load_session(search_id)
                planned_remaining = int(current["summary"]["planned"])
                if planned_remaining <= 0 and added <= 0:
                    stop_reason = stop_reason or "search_exhausted"
                    break
            if max_batches is not None and batch_number >= max_batches:
                stop_reason = "max_batches_reached"
                break
            if candidate_cap is not None and total_executed >= candidate_cap:
                stop_reason = "candidate_cap_reached"
                break
            if time.monotonic() - started_at >= deep_policy.time_budget_hours * 3600:
                stop_reason = "time_budget_reached"
                break

            this_budget = run_budget
            if candidate_cap is not None:
                remaining_budget = candidate_cap - total_executed
                this_budget = min(this_budget, remaining_budget)
            if this_budget <= 0:
                stop_reason = "candidate_cap_reached"
                break

            batch_number += 1
            result = self.run_session(
                search_id=search_id,
                run_budget=this_budget,
                dry_run=False,
                keep_worktrees=keep_worktrees,
                workers_per_run=workers_per_run or deep_policy.workers_per_run,
                parallel_runs=parallel_runs or deep_policy.parallel_runs,
            )
            executed = int(result.get("executed", 0))
            total_executed += executed
            candidate_rows = self.candidate_summary_frame(search_id=search_id)
            primary_metric = (
                search_pack.objective_order[0] if search_pack is not None else "county_mape_overall"
            )
            primary_best = self._best_primary_delta(candidate_rows, metric_name=primary_metric)
            if primary_best is not None and (
                best_primary_delta is None
                or primary_best <= best_primary_delta - deep_policy.min_improvement_pp
            ):
                best_primary_delta = primary_best
                stalled_rounds = 0
            else:
                stalled_rounds += 1
            rounds.append(
                {
                    "batch": batch_number,
                    "executed": executed,
                    "planned_remaining": int(result["summary"]["planned"]),
                    "completed": int(result["summary"]["completed"]),
                    "failed": int(result["summary"]["failed"]),
                    "best_primary_delta": primary_best,
                }
            )
            self._append_journal(
                search_id,
                {
                    "event": "batch_completed",
                    "batch": batch_number,
                    "executed": executed,
                    "planned_remaining": int(result["summary"]["planned"]),
                    "best_primary_delta": primary_best,
                },
            )
            if executed <= 0:
                stop_reason = "no_progress"
                break
            if stalled_rounds >= deep_policy.plateau_rounds:
                stop_reason = "plateau"
                break
            operational_state = self._operational_stop_state(
                candidate_rows,
                repeat_limit=deep_policy.operational_repeat_limit,
                failure_window=deep_policy.operational_failure_window,
                failure_rate=deep_policy.operational_failure_rate,
            )
            if operational_state is not None:
                stop_reason = operational_state
                break

        final_session = self.load_session(search_id)
        final_session["stop_reason"] = stop_reason or final_session.get("stop_reason", "")
        final_session["status"] = "finished"
        final_session["updated_at"] = _utc_now()
        self._write_session(search_id, final_session)
        artifacts = self.write_session_artifacts(search_id=search_id)
        return {
            "search_id": search_id,
            "status": final_session["status"],
            "executed_total": total_executed,
            "batches": rounds,
            "summary": final_session["summary"],
            "artifacts": artifacts,
        }

    def status(self, *, search_id: str | None = None) -> dict[str, Any]:
        """Return one session summary or a summary across all sessions."""
        if search_id is not None:
            session = self.load_session(search_id)
            return {
                "search_id": search_id,
                "status": session["status"],
                "summary": self._summarize_candidates(session["candidates"]),
                "updated_at": session["updated_at"],
            }

        sessions = []
        for session_file in sorted(self.policy.session_root.glob("*/session.yaml")):
            session = yaml.safe_load(session_file.read_text(encoding="utf-8")) or {}
            sessions.append(
                {
                    "search_id": session.get("search_id", session_file.parent.name),
                    "status": session.get("status", "unknown"),
                    "updated_at": session.get("updated_at", ""),
                    "summary": self._summarize_candidates(session.get("candidates", [])),
                }
            )
        return {"sessions": sessions}

    def render_report(self, *, search_id: str) -> str:
        """Render a Markdown report for one search session."""
        session = self.load_session(search_id)
        candidate_rows = build_search_candidate_rows(session, project_root=self.project_root)
        completed_rows = [row for row in candidate_rows if row["status"] == "completed"]
        primary_metric = str(
            self.observatory_config.get("comparison", {}).get(
                "primary_metric",
                "county_mape_overall",
            )
        )
        history_index_present = (self.history_dir / "index.csv").exists()
        complete_bundle_count = 0
        incomplete_bundle_count = 0
        if self.history_dir.exists():
            complete_bundle_count = sum(
                1
                for run_dir in self.history_dir.iterdir()
                if run_dir.is_dir()
                and (run_dir / "summary_scorecard.csv").exists()
                and (run_dir / "manifest.json").exists()
            )
            incomplete_bundle_count = sum(
                1
                for run_dir in self.history_dir.iterdir()
                if run_dir.is_dir()
                and not (
                    (run_dir / "summary_scorecard.csv").exists()
                    and (run_dir / "manifest.json").exists()
                )
            )
        session_summary = build_search_session_summary(
            pd.DataFrame(candidate_rows),
            search_id=search_id,
            status=str(session.get("status", "")),
            history_index_present=history_index_present,
            complete_bundle_count=complete_bundle_count,
            incomplete_bundle_count=incomplete_bundle_count,
        )
        lines = [
            f"# Deep Search Report: {search_id}",
            "",
            f"- Status: `{session['status']}`",
            f"- Created: `{session['created_at']}`",
            f"- Updated: `{session['updated_at']}`",
            f"- Base revision: `{session['resolved_base_revision']}`",
            f"- Stop reason: `{session.get('stop_reason', '') or 'search_exhausted'}`",
            "",
            "## Summary",
            "",
        ]
        summary = self._summarize_candidates(session["candidates"])
        lines.extend(
            f"- {key.replace('_', ' ').title()}: `{summary.get(key, 0)}`"
            for key in ("total", "planned", "running", "completed", "failed")
        )
        lines.extend(
            [
                "",
                "## Decision Brief",
                "",
                f"- State: `{session_summary['session_decision_state']}`",
                f"- Headline: {session_summary['session_headline']}",
                f"- Recommendation: {session_summary['session_recommendation']}",
            ]
        )
        blocker_summary = str(session_summary.get("session_blocker_summary", "") or "")
        if blocker_summary:
            lines.append(f"- Blocker: {blocker_summary}")
        outcome_counts: dict[str, int] = {}
        for row in completed_rows:
            outcome = str(row.get("outcome", "") or "unknown")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if outcome_counts:
            lines.extend(["", "## Outcomes", ""])
            for outcome, count in sorted(outcome_counts.items()):
                lines.append(f"- `{outcome}`: `{count}`")

        sortable_rows = [
            row
            for row in completed_rows
            if isinstance(row.get(f"delta_{primary_metric}"), (int, float))
        ]
        sortable_rows.sort(key=lambda row: float(row.get(f"delta_{primary_metric}", 0.0)))
        if sortable_rows:
            lines.extend(["", "## Best Completed Candidates", ""])
            for row in sortable_rows[:5]:
                delta_value = float(row.get(f"delta_{primary_metric}", 0.0))
                metric_value = row.get(primary_metric, "")
                lines.append(
                    f"- `{row['candidate_id']}` | outcome=`{row.get('outcome', '')}` | "
                    f"{primary_metric}=`{metric_value}` | "
                    f"delta_vs_champion=`{delta_value:+.4f}` | "
                    f"run_id=`{row.get('run_id', '')}`"
                )
        lines.extend(["", "## Candidates", ""])
        lines.extend(
            f"- `{row['candidate_id']}` | `{row['status']}` | "
            f"decision=`{row.get('decision_state', '')}` | "
            f"source=`{row['source']}` | outcome=`{row.get('outcome', '')}` | "
            f"run_id=`{row.get('run_id', '')}` | "
            f"headline={row.get('headline', '')}"
            for row in candidate_rows
        )
        return "\n".join(lines).rstrip() + "\n"

    def candidate_summary_frame(self, *, search_id: str) -> pd.DataFrame:
        """Return a flat per-candidate summary for one search session."""
        session = self.load_session(search_id)
        rows = build_search_candidate_rows(session, project_root=self.project_root)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def write_session_artifacts(self, *, search_id: str) -> dict[str, str]:
        """Write session-level summary artifacts and return their paths."""
        session_dir = self._session_dir(search_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        summary_frame = self.candidate_summary_frame(search_id=search_id)
        summary_csv = session_dir / "candidate_summary.csv"
        summary_json = session_dir / "candidate_summary.json"
        frontier_csv = session_dir / "frontier.csv"
        deep_brief_json = session_dir / "deep_search_brief.json"
        deep_brief_md = session_dir / "deep_search_brief.md"
        ai_brief_json = session_dir / "ai_brief.json"
        ai_brief_md = session_dir / "ai_brief.md"
        report_path = session_dir / "search_report.md"

        if summary_frame.empty:
            summary_frame = pd.DataFrame(
                columns=["candidate_id", "source", "execution_mode", "status"]
            )
        summary_frame.to_csv(summary_csv, index=False)
        summary_json.write_text(
            summary_frame.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
        session = self.load_session(search_id)
        search_pack = self.resolve_search_pack(str(session.get("search_pack_id", "") or "") or None)
        frontier = (
            build_frontier_frame(summary_frame, pack=search_pack, limit=10)
            if search_pack is not None
            else summary_frame.head(10)
        )
        frontier.to_csv(frontier_csv, index=False)
        brief = self._build_deep_search_brief(search_id=search_id)
        deep_brief_json.write_text(json.dumps(brief, indent=2, sort_keys=True), encoding="utf-8")
        deep_brief_md.write_text(self._render_deep_search_brief_markdown(brief), encoding="utf-8")
        if brief.get("ai_summary"):
            ai_brief_json.write_text(
                json.dumps(
                    {
                        "summary": brief.get("ai_summary"),
                        "validation": brief.get("ai_validation", {}),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            ai_brief_md.write_text(
                "# AI Brief\n\n" + str(brief.get("ai_summary", "")).strip() + "\n",
                encoding="utf-8",
            )
        report_path.write_text(self.render_report(search_id=search_id), encoding="utf-8")
        return {
            "candidate_summary_csv": str(summary_csv.relative_to(self.project_root)),
            "candidate_summary_json": str(summary_json.relative_to(self.project_root)),
            "frontier_csv": str(frontier_csv.relative_to(self.project_root)),
            "deep_search_brief_json": str(deep_brief_json.relative_to(self.project_root)),
            "deep_search_brief_markdown": str(deep_brief_md.relative_to(self.project_root)),
            "ai_brief_json": str(ai_brief_json.relative_to(self.project_root))
            if ai_brief_json.exists()
            else "",
            "ai_brief_markdown": str(ai_brief_md.relative_to(self.project_root))
            if ai_brief_md.exists()
            else "",
            "search_report_markdown": str(report_path.relative_to(self.project_root)),
        }

    def load_session(self, search_id: str) -> dict[str, Any]:
        """Load a persisted session YAML file."""
        session_path = self._session_file(search_id)
        if not session_path.exists():
            raise FileNotFoundError(f"Search session not found: {session_path}")
        data = yaml.safe_load(session_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid search session file: {session_path}")
        return data

    def _build_candidate_record(
        self,
        *,
        candidate_id: str,
        source: str,
        source_id: str,
        execution_mode: str,
        spec: dict[str, Any],
        spec_path: Path,
        recipe_id: str | None = None,
        recipe_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "source": source,
            "source_id": source_id,
            "execution_mode": execution_mode,
            "recipe_id": recipe_id or "",
            "recipe_parameters": recipe_parameters or {},
            "status": "planned",
            "attempts": 0,
            "spec_path": str(spec_path.relative_to(self.project_root)),
            "spec": spec,
        }

    def _run_worktree_experiment(
        self,
        worktree_path: Path,
        spec_path: Path,
        workers_per_run: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        script_path = worktree_path / "scripts" / "analysis" / "run_experiment.py"
        profile_dir = worktree_path / "config" / "method_profiles"
        cmd = [
            sys.executable,
            str(script_path),
            "--spec",
            str(spec_path),
            "--profile-dir",
            str(profile_dir),
        ]
        if workers_per_run > 0:
            cmd.extend(["--workers", str(workers_per_run)])
        result = subprocess.run(
            cmd,
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Worktree experiment failed: "
                f"{result.stderr.strip() or result.stdout.strip() or 'no output'}"
            )
        return result

    def _harvest_candidate(
        self,
        *,
        search_id: str,
        candidate: dict[str, Any],
        context: Any,
        changed_files: list[str],
        diff_patch: str,
        command_result: subprocess.CompletedProcess[str],
        applied_recipe: dict[str, Any] | None,
        validation_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        session_dir = self._session_dir(search_id)
        completed_specs_dir = session_dir / "completed_specs"
        patches_dir = session_dir / "patches"
        code_candidates_dir = session_dir / "code_candidates"
        logs_dir = session_dir / "logs"
        profiles_dir = session_dir / "profiles"
        for directory in (
            completed_specs_dir,
            patches_dir,
            code_candidates_dir,
            logs_dir,
            profiles_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        experiment_id = str(candidate["spec"]["experiment_id"])
        # The experiment subprocess may write to either the worktree's log
        # or the live repo's log (due to editable-install import paths).
        # Check both locations.
        local_log_path = (
            context.worktree_path / "data" / "analysis" / "experiments" / "experiment_log.csv"
        )
        local_log = read_experiment_log(local_log_path)
        matching = local_log[local_log["experiment_id"] == experiment_id]
        if matching.empty:
            canonical_log = read_experiment_log(self.experiment_log_path)
            matching = canonical_log[canonical_log["experiment_id"] == experiment_id]
        if matching.empty:
            raise RuntimeError(f"Experiment log entry not found for {experiment_id}")
        row = cast(dict[str, Any], matching.iloc[-1].to_dict())

        local_completed_spec = (
            context.worktree_path
            / "data"
            / "analysis"
            / "experiments"
            / "completed"
            / f"{experiment_id}.yaml"
        )
        # Completed spec may also land in the live repo due to editable install.
        if not local_completed_spec.exists():
            canonical_completed_spec = (
                self.project_root
                / "data"
                / "analysis"
                / "experiments"
                / "completed"
                / f"{experiment_id}.yaml"
            )
            if canonical_completed_spec.exists():
                local_completed_spec = canonical_completed_spec
            else:
                raise RuntimeError(f"Completed spec not found: {local_completed_spec}")

        copied_spec_path = completed_specs_dir / local_completed_spec.name
        shutil.copy2(local_completed_spec, copied_spec_path)

        stdout_path = logs_dir / f"{candidate['candidate_id']}.stdout.log"
        stderr_path = logs_dir / f"{candidate['candidate_id']}.stderr.log"
        stdout_path.write_text(command_result.stdout, encoding="utf-8")
        stderr_path.write_text(command_result.stderr, encoding="utf-8")

        is_code_candidate = bool(
            changed_files
            or candidate.get("execution_mode") == "code_recipe"
            or str(candidate.get("source", "") or "") == "search_pack"
            or str(candidate.get("spec", {}).get("code_mutator_id", "") or "")
        )
        code_candidate_path = code_candidates_dir / str(candidate["candidate_id"])
        if is_code_candidate:
            code_candidate_path.mkdir(parents=True, exist_ok=True)
        patch_path = (
            code_candidate_path / "patch.diff"
            if is_code_candidate
            else patches_dir / f"{candidate['candidate_id']}.patch"
        )
        patch_path.write_text(diff_patch, encoding="utf-8")
        manifest_path = (
            code_candidate_path / "patch_manifest.json"
            if is_code_candidate
            else patches_dir / f"{candidate['candidate_id']}.json"
        )
        patch_manifest = {
            "candidate_id": candidate["candidate_id"],
            "recipe_id": candidate.get("recipe_id", ""),
            "execution_mode": candidate["execution_mode"],
            "changed_files": changed_files,
            "applied_recipe": applied_recipe or {},
            "base_revision": context.base_revision,
        }
        manifest_path.write_text(
            json.dumps(patch_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validation_report_path = ""
        if validation_report is not None and is_code_candidate:
            validation_file = code_candidate_path / "validation_report.json"
            validation_file.write_text(
                json.dumps(validation_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            validation_report_path = str(validation_file.relative_to(self.project_root))

        copied_spec_relative = copied_spec_path.relative_to(self.project_root)
        row["spec_path"] = str(copied_spec_relative)
        if experiment_id not in get_tested_hypotheses(self.experiment_log_path):
            append_experiment_entry(row, log_path=self.experiment_log_path)

        predicted_method = _predict_method_id(candidate["spec"])
        predicted_config = _predict_config_id(candidate["spec"])
        run_id = str(row.get("run_id", "") or "")
        if run_id and run_id != "not_run":
            issue = self._validate_run_bundle(
                worktree_path=context.worktree_path,
                run_id=run_id,
                method_id=predicted_method,
            )
            if issue is not None:
                return self._build_operational_blocker_result(
                    search_id=search_id,
                    candidate=candidate,
                    error=issue,
                )
            self._harvest_run_bundle(context.worktree_path, run_id)

        profile_artifact = ""
        local_profile = (
            context.worktree_path
            / "config"
            / "method_profiles"
            / f"{predicted_method}__{predicted_config}.yaml"
        )
        if local_profile.exists():
            copied_profile = profiles_dir / local_profile.name
            shutil.copy2(local_profile, copied_profile)
            profile_artifact = str(copied_profile.relative_to(self.project_root))

        benchmark_summary = self._extract_benchmark_summary(
            worktree_path=context.worktree_path,
            run_id=run_id,
            method_id=predicted_method,
            config_id=predicted_config,
        )

        return {
            "outcome": row.get("outcome", ""),
            "run_id": run_id,
            "method_id": predicted_method,
            "config_id": predicted_config,
            "spec_path": str(copied_spec_relative),
            "patch_path": str(patch_path.relative_to(self.project_root)),
            "patch_manifest_path": str(manifest_path.relative_to(self.project_root)),
            "stdout_log": str(stdout_path.relative_to(self.project_root)),
            "stderr_log": str(stderr_path.relative_to(self.project_root)),
            "profile_path": profile_artifact,
            "validation_report_path": validation_report_path,
            "changed_files": changed_files,
            "classification_details": dict(row.get("classification_details") or {}),
            "key_metrics_summary": row.get("key_metrics_summary", ""),
            "interpretation": row.get("interpretation", ""),
            "next_action": row.get("next_action", ""),
            "benchmark_summary": benchmark_summary,
        }

    def _validate_run_bundle(
        self,
        *,
        worktree_path: Path,
        run_id: str,
        method_id: str,
    ) -> str | None:
        """Return an operational issue string for a run bundle, if any."""
        local_run_dir = worktree_path / "data" / "analysis" / "benchmark_history" / run_id
        if not local_run_dir.exists():
            return f"Operational blocker: benchmark bundle missing for run_id {run_id}."
        required = [
            local_run_dir / "manifest.json",
            local_run_dir / "summary_scorecard.csv",
            local_run_dir / "runtime_summary.json",
        ]
        missing = [str(path.name) for path in required if not path.exists()]
        if missing:
            return (
                "Operational blocker: benchmark bundle is incomplete. Missing "
                + ", ".join(missing)
                + "."
            )
        target_run_dir = self.history_dir / run_id
        if target_run_dir.exists():
            return (
                f"Operational blocker: benchmark run_id {run_id} already exists in the canonical "
                "history directory."
            )
        scorecard = pd.read_csv(local_run_dir / "summary_scorecard.csv")
        method_rows = scorecard[scorecard["method_id"].astype(str) == method_id]
        if method_rows.empty:
            return f"Operational blocker: scorecard missing method row for {method_id}."
        row = method_rows.iloc[-1]
        if "runtime_summary_present" in row and not bool(row.get("runtime_summary_present", True)):
            return "Operational blocker: runtime summary flag is false in the scorecard."
        if "reproducibility_logging_flag" in row and not bool(
            row.get("reproducibility_logging_flag", True)
        ):
            return "Operational blocker: reproducibility logging flag is false in the scorecard."
        return None

    def _harvest_run_bundle(self, worktree_path: Path, run_id: str) -> None:
        local_history_dir = worktree_path / "data" / "analysis" / "benchmark_history"
        local_run_dir = local_history_dir / run_id
        if not local_run_dir.exists():
            return
        target_run_dir = self.history_dir / run_id
        target_run_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_run_dir.exists():
            shutil.copytree(local_run_dir, target_run_dir)

        canonical_index = self.history_dir / "index.csv"
        if canonical_index.exists():
            existing = pd.read_csv(canonical_index, dtype=str)
            if "run_id" in existing.columns and run_id in set(existing["run_id"].dropna()):
                return

        local_index = pd.read_csv(local_history_dir / "index.csv", dtype=str)
        run_rows = local_index[local_index["run_id"] == run_id]
        if run_rows.empty:
            return
        scorecard = pd.read_csv(target_run_dir / "summary_scorecard.csv")
        first_row = run_rows.iloc[0]
        champions = run_rows[run_rows["is_champion_at_run"].astype(str).str.lower() == "true"]
        champion_method = (
            str(champions.iloc[0]["method_id"])
            if not champions.empty
            else str(run_rows.iloc[0]["method_id"])
        )
        append_benchmark_index(
            index_path=canonical_index,
            scorecard=scorecard,
            manifest_path=target_run_dir / "manifest.json",
            benchmark_label=str(first_row["benchmark_label"]),
            benchmark_contract_version=str(first_row["benchmark_contract_version"]),
            git_commit=str(first_row["git_commit"]),
            champion_method_id=champion_method,
            decision_id=str(first_row.get("decision_id", "")) or None,
            decision_status=str(first_row.get("decision_status", "pending")),
        )

    def _load_variant_catalog(self) -> VariantCatalog | None:
        if not self.variant_catalog_path.exists():
            return None
        return VariantCatalog(
            self.variant_catalog_path,
            experiment_log=self._read_canonical_experiment_log(),
        )

    def _read_canonical_experiment_log(self) -> pd.DataFrame:
        return read_experiment_log(self.experiment_log_path, dedupe_by_experiment_id=True)

    def _latest_logged_experiment_entry(self, experiment_id: str) -> dict[str, Any] | None:
        """Return the latest canonical experiment-log row for one experiment."""
        canonical_log = self._read_canonical_experiment_log()
        if canonical_log.empty or "experiment_id" not in canonical_log.columns:
            return None
        matching = canonical_log[canonical_log["experiment_id"].astype(str) == experiment_id]
        if matching.empty:
            return None
        return cast(dict[str, Any], matching.iloc[-1].to_dict())

    def _logged_candidate_result(
        self,
        *,
        candidate: dict[str, Any],
        log_row: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a candidate-result payload from an existing experiment-log row."""
        spec = cast(dict[str, Any], candidate["spec"])
        method_id = _predict_method_id(spec)
        config_id = _predict_config_id(spec)
        raw_run_id = str(log_row.get("run_id", "") or "")
        run_id = "" if raw_run_id == "not_run" else raw_run_id
        benchmark_summary = {}
        if run_id:
            benchmark_summary = self._extract_benchmark_summary_from_history_dir(
                history_dir=self.history_dir,
                run_id=run_id,
                method_id=method_id,
                config_id=config_id,
            )
        return {
            "outcome": str(log_row.get("outcome", "") or ""),
            "run_id": run_id,
            "method_id": method_id,
            "config_id": config_id,
            "spec_path": str(log_row.get("spec_path", "") or candidate.get("spec_path", "")),
            "patch_path": "",
            "patch_manifest_path": "",
            "stdout_log": "",
            "stderr_log": "",
            "profile_path": "",
            "changed_files": [],
            "classification_details": {},
            "key_metrics_summary": str(log_row.get("key_metrics_summary", "") or ""),
            "interpretation": str(log_row.get("interpretation", "") or ""),
            "next_action": str(log_row.get("next_action", "") or ""),
            "benchmark_summary": benchmark_summary,
        }

    def _session_dir(self, search_id: str) -> Path:
        return self.policy.session_root / search_id

    def _session_file(self, search_id: str) -> Path:
        return self._session_dir(search_id) / "session.yaml"

    def _journal_file(self, search_id: str) -> Path:
        return self._session_dir(search_id) / "search_journal.jsonl"

    @staticmethod
    def _seed_source_ids(search_pack: SearchPack | None, *, source: str) -> set[str]:
        """Return lower-cased source IDs from a search pack seed list."""
        if search_pack is None:
            return set()
        return {
            str(seed.get("source_id", "")).lower()
            for seed in search_pack.seed_candidates
            if str(seed.get("source", "")).lower() == source.lower()
            and str(seed.get("source_id", "")).strip()
        }

    def _append_journal(self, search_id: str, payload: dict[str, Any]) -> None:
        """Append one structured journal entry for the session."""
        entry = {"timestamp": _utc_now(), **payload}
        journal_path = self._journal_file(search_id)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _write_session(self, search_id: str, session: dict[str, Any]) -> None:
        session_dir = self._session_dir(search_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_file(search_id).write_text(
            yaml.safe_dump(session, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

    def _persist_session_update(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _utc_now()
        session["summary"] = self._summarize_candidates(session["candidates"])
        self._write_session(str(session["search_id"]), session)

    @staticmethod
    def _count_changed_lines(diff_patch: str) -> int:
        """Count non-header added/removed lines in a unified diff."""
        count = 0
        for line in diff_patch.splitlines():
            if not line:
                continue
            if line.startswith(("+++", "---", "@@")):
                continue
            if line.startswith(("+", "-")):
                count += 1
        return count

    def _validate_code_changes(
        self,
        *,
        changed_files: list[str],
        diff_patch: str,
        max_patch_files: int,
        max_patch_lines: int,
    ) -> None:
        """Apply deep-search patch-size guardrails before benchmarking."""
        if len(changed_files) > max_patch_files:
            raise RuntimeError(
                f"Operational blocker: patch touched {len(changed_files)} files "
                f"(limit {max_patch_files})."
            )
        changed_lines = self._count_changed_lines(diff_patch)
        if changed_lines > max_patch_lines:
            raise RuntimeError(
                f"Operational blocker: patch changed {changed_lines} lines "
                f"(limit {max_patch_lines})."
            )

    def _run_targeted_tests(self, *, worktree_path: Path, targeted_tests: list[str]) -> None:
        """Run pack-declared targeted tests before the benchmark when provided."""
        if not targeted_tests:
            return
        cmd = [sys.executable, "-m", "pytest", *targeted_tests, "-q"]
        result = subprocess.run(
            cmd,
            cwd=worktree_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Operational blocker: targeted tests failed before benchmark. "
                f"{result.stderr.strip() or result.stdout.strip() or 'no output'}"
            )

    def _build_operational_blocker_result(
        self,
        *,
        search_id: str,
        candidate: dict[str, Any],
        error: str,
    ) -> dict[str, Any]:
        """Persist a quarantine artifact and return a structured blocker payload."""
        quarantine_dir = self._session_dir(search_id) / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        quarantine_path = quarantine_dir / f"{candidate['candidate_id']}.json"
        payload = {
            "candidate_id": candidate["candidate_id"],
            "source": candidate.get("source", ""),
            "execution_mode": candidate.get("execution_mode", ""),
            "error": error,
            "spec": dict(candidate.get("spec", {}) or {}),
        }
        quarantine_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validation_report_path = ""
        is_code_candidate = bool(
            candidate.get("execution_mode") == "code_recipe"
            or str(candidate.get("source", "") or "") == "search_pack"
            or str(candidate.get("spec", {}).get("code_mutator_id", "") or "")
        )
        if is_code_candidate:
            code_candidate_dir = (
                self._session_dir(search_id) / "code_candidates" / str(candidate["candidate_id"])
            )
            code_candidate_dir.mkdir(parents=True, exist_ok=True)
            validation_file = code_candidate_dir / "validation_report.json"
            validation_file.write_text(
                json.dumps(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "execution_mode": candidate.get("execution_mode", ""),
                        "status": "failed",
                        "error": error,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            validation_report_path = str(validation_file.relative_to(self.project_root))
        self._append_journal(
            str(search_id),
            {
                "event": "operational_blocker",
                "candidate_id": str(candidate.get("candidate_id", "")),
                "error": error,
                "quarantine_path": str(quarantine_path.relative_to(self.project_root)),
            },
        )
        return {
            "outcome": "operational_blocker",
            "run_id": "",
            "method_id": _predict_method_id(cast(dict[str, Any], candidate["spec"])),
            "config_id": _predict_config_id(cast(dict[str, Any], candidate["spec"])),
            "spec_path": str(candidate.get("spec_path", "")),
            "patch_path": "",
            "patch_manifest_path": "",
            "stdout_log": "",
            "stderr_log": "",
            "profile_path": "",
            "validation_report_path": validation_report_path,
            "changed_files": [],
            "classification_details": {
                "classification": "operational_blocker",
                "reasons": [error],
            },
            "key_metrics_summary": "operational_quality: blocked",
            "interpretation": "The search candidate was blocked by orchestration or sandbox validation.",
            "next_action": "flag_for_review",
            "benchmark_summary": {},
            "quarantine_path": str(quarantine_path.relative_to(self.project_root)),
        }

    def _resolve_git_revision(self, revision: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", revision],
            cwd=self.project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _write_spec_to_worktree(self, worktree_path: Path, spec: dict[str, Any]) -> Path:
        pending_dir = worktree_path / "data" / "analysis" / "experiments" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        spec_path = pending_dir / f"{spec['experiment_id']}.yaml"
        spec_path.write_text(
            yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return spec_path

    @staticmethod
    def _summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, int]:
        summary = {
            "total": len(candidates),
            "planned": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for candidate in candidates:
            status = str(candidate.get("status", "planned"))
            if status in summary:
                summary[status] += 1
        return summary

    def _extract_benchmark_summary(
        self,
        *,
        worktree_path: Path,
        run_id: str,
        method_id: str,
        config_id: str,
    ) -> dict[str, Any]:
        """Extract decision-useful metrics for one completed benchmark run."""
        history_dir = worktree_path / "data" / "analysis" / "benchmark_history"
        return self._extract_benchmark_summary_from_history_dir(
            history_dir=history_dir,
            run_id=run_id,
            method_id=method_id,
            config_id=config_id,
        )

    def _extract_benchmark_summary_from_history_dir(
        self,
        *,
        history_dir: Path,
        run_id: str,
        method_id: str,
        config_id: str,
    ) -> dict[str, Any]:
        """Extract decision-useful metrics for one completed benchmark run."""
        if not run_id:
            return {}
        run_dir = history_dir / run_id
        scorecard_path = run_dir / "summary_scorecard.csv"
        comparison_path = run_dir / "comparison_to_champion.json"
        if not scorecard_path.exists():
            return {}

        scorecard = pd.read_csv(scorecard_path)
        method_rows = scorecard[scorecard["method_id"].astype(str) == method_id]
        if "config_id" in scorecard.columns:
            exact_rows = method_rows[method_rows["config_id"].astype(str) == config_id]
            if not exact_rows.empty:
                method_rows = exact_rows
        if method_rows.empty:
            return {}
        row = method_rows.iloc[-1].to_dict()

        comparison_cfg = dict(self.observatory_config.get("comparison", {}))
        metric_names: list[str] = []
        primary_metric = str(comparison_cfg.get("primary_metric", "county_mape_overall"))
        metric_names.append(primary_metric)
        for metric in comparison_cfg.get("secondary_metrics", []):
            metric_str = str(metric)
            if metric_str not in metric_names:
                metric_names.append(metric_str)
        for metric in (
            "state_ape_recent_short",
            "state_ape_recent_medium",
            "county_mape_overall",
            "county_mape_urban_college",
            "county_mape_rural",
            "county_mape_bakken",
        ):
            if metric not in metric_names:
                metric_names.append(metric)

        metric_summary = {
            metric: float(row[metric])
            for metric in metric_names
            if metric in row and pd.notna(row[metric])
        }
        for flag in (
            "negative_population_violations",
            "scenario_order_violations",
            "aggregation_violations",
            "sensitivity_instability_flag",
        ):
            if flag in row and pd.notna(row[flag]):
                metric_summary[flag] = row[flag]

        delta_summary: dict[str, Any] = {}
        champion_method_id = ""
        champion_config_id = ""
        hard_constraint_regression: bool | None = None
        if comparison_path.exists():
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            champion_method_id = str(comparison.get("champion_method_id", "") or "")
            champion_config_id = str(comparison.get("champion_config_id", "") or "")
            for challenger in comparison.get("challengers", []):
                if not isinstance(challenger, dict):
                    continue
                if str(challenger.get("method_id", "")) != method_id:
                    continue
                if str(challenger.get("config_id", "")) != config_id:
                    continue
                raw_deltas = challenger.get("deltas", {})
                if isinstance(raw_deltas, dict):
                    delta_summary = {
                        str(metric): float(value)
                        for metric, value in raw_deltas.items()
                        if pd.notna(value)
                    }
                hard_constraint_regression = bool(
                    challenger.get("hard_constraint_regression", False)
                )
                break

        return {
            "primary_metric": primary_metric,
            "metrics": metric_summary,
            "deltas": delta_summary,
            "hard_constraint_regression": hard_constraint_regression,
            "champion_method_id": champion_method_id,
            "champion_config_id": champion_config_id,
        }

    @staticmethod
    def _candidate_summary_rows(session: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten candidate/session data into one row per candidate."""
        return build_search_candidate_rows(session)

    @staticmethod
    def _best_primary_delta(
        candidates: pd.DataFrame,
        *,
        metric_name: str,
    ) -> float | None:
        """Return the best available primary-metric delta across completed candidates."""
        if candidates.empty:
            return None
        delta_col = f"delta_{metric_name}"
        if delta_col not in candidates.columns:
            return None
        completed = candidates[
            candidates.get("status", pd.Series(dtype=object)).astype(str) == "completed"
        ]
        if completed.empty:
            return None
        deltas = pd.to_numeric(completed[delta_col], errors="coerce").dropna()
        if deltas.empty:
            return None
        return float(deltas.min())

    @staticmethod
    def _operational_stop_state(
        candidates: pd.DataFrame,
        *,
        repeat_limit: int,
        failure_window: int,
        failure_rate: float,
    ) -> str | None:
        """Return a stop-reason code when operational blockers accumulate."""
        if candidates.empty or "outcome" not in candidates.columns:
            return None
        completed = candidates[
            candidates.get("status", pd.Series(dtype=object)).astype(str) == "completed"
        ]
        if completed.empty:
            return None
        recent = completed.tail(failure_window)
        if recent.empty:
            return None
        blockers = recent[recent["outcome"].astype(str) == "operational_blocker"]
        if blockers.empty:
            return None
        blocker_rate = len(blockers) / len(recent)
        if blocker_rate > failure_rate:
            return "operational_failure_rate_exceeded"
        reasons = blockers.get("last_error", pd.Series(dtype=object)).astype(str)
        if not reasons.empty:
            top_count = int(reasons.value_counts().iloc[0])
            if top_count >= repeat_limit:
                return "repeated_operational_blocker"
        return None

    def _build_deep_search_brief(self, *, search_id: str) -> dict[str, Any]:
        """Build the deterministic deep-search brief for one session."""
        session = self.load_session(search_id)
        candidates = self.candidate_summary_frame(search_id=search_id)
        search_pack = self.resolve_search_pack(str(session.get("search_pack_id", "") or "") or None)
        session_summary = build_search_session_summary(
            candidates,
            search_id=search_id,
            status=str(session.get("status", "") or ""),
            history_index_present=(self.history_dir / "index.csv").exists(),
            complete_bundle_count=int(
                sum(
                    1
                    for run_dir in self.history_dir.iterdir()
                    if run_dir.is_dir()
                    and (run_dir / "summary_scorecard.csv").exists()
                    and (run_dir / "manifest.json").exists()
                )
            )
            if self.history_dir.exists()
            else 0,
            incomplete_bundle_count=int(
                sum(
                    1
                    for run_dir in self.history_dir.iterdir()
                    if run_dir.is_dir()
                    and not (
                        (run_dir / "summary_scorecard.csv").exists()
                        and (run_dir / "manifest.json").exists()
                    )
                )
            )
            if self.history_dir.exists()
            else 0,
        )
        if search_pack is not None:
            frontier = build_frontier_frame(candidates, pack=search_pack, limit=10)
        else:
            frontier = (
                candidates.head(10).reset_index(drop=True)
                if not candidates.empty
                else pd.DataFrame()
            )
        winner = frontier.iloc[0].to_dict() if not frontier.empty else {}
        objective_metrics: list[dict[str, Any]] = []
        citations: list[dict[str, str]] = []
        if search_pack is not None and winner:
            for metric in search_pack.objective_order:
                metric_value = winner.get(metric)
                delta_value = winner.get(f"delta_{metric}")
                try:
                    metric_missing = bool(pd.isna(metric_value))  # type: ignore[call-overload]
                except TypeError:
                    metric_missing = metric_value is None
                try:
                    delta_missing = bool(pd.isna(delta_value))  # type: ignore[call-overload]
                except TypeError:
                    delta_missing = delta_value is None
                if metric_missing and delta_missing:
                    continue
                objective_metrics.append(
                    {
                        "name": metric,
                        "value": metric_value,
                        "delta_vs_champion": delta_value,
                        "run_id": str(winner.get("run_id", "") or ""),
                        "lower_is_better": True,
                    }
                )
                run_ref = str(winner.get("run_id", "") or "")
                if run_ref:
                    citations.append(
                        {
                            "kind": "run",
                            "ref_id": run_ref,
                            "label": metric,
                        }
                    )
        coverage = {
            "total_candidates": int(len(candidates)),
            "completed_candidates": int(
                (candidates.get("status", pd.Series(dtype=object)).astype(str) == "completed").sum()
            )
            if not candidates.empty
            else 0,
            "reviewable_candidates": int(
                candidates.get("decision_state", pd.Series(dtype=object))
                .astype(str)
                .isin(["recommended", "ready_for_review"])
                .sum()
            )
            if not candidates.empty
            else 0,
            "operational_blockers": int(
                (
                    candidates.get("outcome", pd.Series(dtype=object)).astype(str)
                    == "operational_blocker"
                ).sum()
            )
            if not candidates.empty
            else 0,
        }
        planned = candidates[
            candidates.get("status", pd.Series(dtype=object)).astype(str) == "planned"
        ].head(3)
        next_experiments = [
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "hypothesis": str(row.get("hypothesis", row.get("headline", "")) or ""),
            }
            for _, row in planned.iterrows()
        ]
        verification_checklist = [
            {
                "label": "Winner is benchmark-backed",
                "detail": "Confirm the top candidate has a real run bundle and is not only a planned spec.",
                "status": "yes" if str(winner.get("run_id", "") or "").strip() else "no",
            },
            {
                "label": "Operational blockers are acceptable",
                "detail": "Review any quarantined candidates before trusting search coverage.",
                "status": "yes" if coverage["operational_blockers"] == 0 else "no",
            },
            {
                "label": "Stop condition makes sense",
                "detail": f"Session stop reason: {session.get('stop_reason', 'search_exhausted') or 'search_exhausted'}.",
                "status": "yes",
            },
        ]
        brief: dict[str, Any] = {
            "search_id": search_id,
            "search_pack_id": str(session.get("search_pack_id", "") or ""),
            "stop_reason": str(session.get("stop_reason", "") or "search_exhausted"),
            "decision_brief": {
                "state": str(session_summary.get("session_decision_state", "")),
                "subject_label": str(
                    winner.get("primary_subject_label")
                    or winner.get("candidate_id")
                    or session_summary.get("primary_subject_label")
                    or "Deep search session"
                ),
                "raw_subject_id": str(
                    winner.get("run_id")
                    or winner.get("candidate_id")
                    or session_summary.get("recommendation_candidate_id")
                    or ""
                ),
                "headline": str(session_summary.get("session_headline", "") or ""),
                "main_reason": str(
                    winner.get("headline")
                    or winner.get("explanation")
                    or session_summary.get("session_headline", "")
                ),
                "next_action": str(session_summary.get("session_recommendation", "") or ""),
                "evidence_quality": str(
                    winner.get("evidence_quality") or session_summary.get("evidence_quality", "")
                ),
                "operational_evidence_summary": (
                    "Operational blockers were quarantined."
                    if coverage["operational_blockers"] > 0
                    else "No operational blockers were detected in the final frontier."
                ),
                "review_checklist": verification_checklist,
                "search_id": search_id,
            },
            "metrics": objective_metrics,
            "coverage": coverage,
            "frontier": frontier.to_dict(orient="records") if not frontier.empty else [],
            "citations": citations,
            "next_experiments": next_experiments,
        }
        synthesis_cfg = dict(self.observatory_config.get("ai_synthesis", {}) or {})
        synthesis_enabled = bool(session.get("allow_ai_synthesis", False)) and bool(
            synthesis_cfg.get("enabled", False)
        )
        payload = build_evidence_payload(
            decision_brief=brief["decision_brief"],
            candidate_rows=candidates,
            comparison_rows=objective_metrics,
            recommendation_rows=next_experiments,
            session_summary={
                "search_id": search_id,
                "stop_reason": brief["stop_reason"],
            },
            context={"search_pack_id": brief["search_pack_id"]},
        )
        synthesis = synthesize_observatory_summary(
            payload,
            enabled=synthesis_enabled,
            provider=self.synthesis_provider,
        )
        brief["deterministic_summary"] = synthesis.deterministic_summary
        brief["final_summary"] = synthesis.final_summary
        brief["ai_summary"] = synthesis.ai_summary
        brief["ai_validation"] = {
            "accepted": synthesis.validation.accepted,
            "suppressed": synthesis.validation.suppressed,
            "issues": list(synthesis.validation.issues),
            "used_provider": synthesis.used_provider,
            "provider_enabled": synthesis.provider_enabled,
        }
        return brief

    @staticmethod
    def _render_deep_search_brief_markdown(brief: dict[str, Any]) -> str:
        """Render a human-readable Markdown brief."""
        decision_brief = dict(brief.get("decision_brief", {}) or {})
        coverage = dict(brief.get("coverage", {}) or {})
        lines = [
            f"# Deep Search Brief: {brief.get('search_id', '')}",
            "",
            f"- Search pack: `{brief.get('search_pack_id', '')}`",
            f"- Stop reason: `{brief.get('stop_reason', '')}`",
            f"- State: `{decision_brief.get('state', '')}`",
            "",
            "## Summary",
            "",
            str(brief.get("final_summary", "") or "No summary available."),
            "",
            "## Coverage",
            "",
        ]
        for key, value in coverage.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")
        frontier = brief.get("frontier", [])
        if isinstance(frontier, list) and frontier:
            lines.extend(["", "## Frontier", ""])
            for row in frontier[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"- `{row.get('candidate_id', '')}` | "
                    f"decision=`{row.get('decision_state', '')}` | "
                    f"run_id=`{row.get('run_id', '')}`"
                )
        next_experiments = brief.get("next_experiments", [])
        if isinstance(next_experiments, list) and next_experiments:
            lines.extend(["", "## Next Experiments", ""])
            for item in next_experiments:
                if not isinstance(item, dict):
                    continue
                lines.append(f"- `{item.get('candidate_id', '')}` | {item.get('hypothesis', '')}")
        validation = brief.get("ai_validation", {})
        if isinstance(validation, dict) and validation.get("suppressed"):
            lines.extend(
                [
                    "",
                    "## AI Synthesis",
                    "",
                    "- AI synthesis was suppressed because the deterministic claim checker found contradictions.",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def _replan_session(
        self,
        *,
        search_id: str,
        search_pack: SearchPack | None,
        max_pending: int | None,
        max_recommended: int | None,
        include_recipe_catalog: bool | None,
        candidate_cap: int,
    ) -> int:
        """Append new candidates to an existing session from fresh evidence."""
        session = self.load_session(search_id)
        current_total = len(session.get("candidates", []))
        if current_total >= candidate_cap:
            return 0

        session_dir = self._session_dir(search_id)
        planned_specs_dir = session_dir / "planned_specs"
        planned_specs_dir.mkdir(parents=True, exist_ok=True)
        logged_experiment_ids = get_tested_hypotheses(self.experiment_log_path)
        seen = {
            _spec_fingerprint(
                execution_mode=str(candidate.get("execution_mode", "config_only")),
                spec=cast(dict[str, Any], candidate.get("spec", {})),
                recipe_id=str(candidate.get("recipe_id", "") or ""),
            )
            for candidate in session.get("candidates", [])
            if isinstance(candidate, dict) and isinstance(candidate.get("spec"), dict)
        }
        added = 0
        requested_by = "deep_search_replan"
        catalog = self._load_variant_catalog()
        seed_variant_ids = self._seed_source_ids(search_pack, source="variant_catalog")
        seed_recipe_ids = self._seed_source_ids(search_pack, source="recipe_catalog")
        pending_limit = self.policy.default_max_pending if max_pending is None else max_pending
        recommended_limit = (
            self.policy.default_max_recommended if max_recommended is None else max_recommended
        )
        include_recipes = (
            self.policy.include_recipe_catalog
            if include_recipe_catalog is None
            else include_recipe_catalog
        )

        if catalog is not None and pending_limit > 0 and current_total + added < candidate_cap:
            ordered = sorted(
                catalog.get_untested(),
                key=lambda variant: (
                    0 if str(variant.get("variant_id", "")).lower() in seed_variant_ids else 1,
                    int(variant.get("tier", 999)),
                    str(variant.get("variant_id", "")),
                ),
            )
            for variant in ordered:
                if current_total + added >= candidate_cap:
                    break
                spec_path = catalog.generate_spec(
                    str(variant["variant_id"]),
                    output_dir=planned_specs_dir,
                    requested_by=requested_by,
                    extra_notes=["Generated from deep-search replanning."],
                )
                spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
                fingerprint = _spec_fingerprint(execution_mode="config_only", spec=spec)
                if fingerprint in seen or str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                seen.add(fingerprint)
                session["candidates"].append(
                    self._build_candidate_record(
                        candidate_id=f"variant-{variant['variant_id'].lower()}",
                        source="variant_catalog",
                        source_id=str(variant["variant_id"]),
                        execution_mode="config_only",
                        spec=spec,
                        spec_path=spec_path,
                    )
                )
                added += 1
                if added >= pending_limit:
                    break

        if (
            self.store is not None
            and recommended_limit > 0
            and current_total + added < candidate_cap
        ):
            comparator = ObservatoryComparator(store=self.store, config=self.observatory_config)
            bounds_catalog = catalog or self._load_variant_catalog()
            recommender = ObservatoryRecommender(
                store=self.store,
                comparator=comparator,
                config=self.observatory_config,
                bounds_catalog=bounds_catalog,
            )
            base_method = (
                search_pack.base_method
                if search_pack is not None
                else self.observatory_config.get("challenger_base_method", "m2026r1")
            )
            base_config = (
                search_pack.base_config
                if search_pack is not None
                else self.observatory_config.get("base_config", "cfg-20260309-college-fix-v1")
            )
            recommendations = recommender.suggest_next_experiments(n=recommended_limit * 3)
            for rec in recommendations:
                if getattr(rec, "requires_code_change", False):
                    continue
                if current_total + added >= candidate_cap:
                    break
                spec = _recommendation_spec(
                    rec,
                    base_method=base_method,
                    base_config=base_config,
                    requested_by=requested_by,
                )
                if spec is None:
                    continue
                fingerprint = _spec_fingerprint(execution_mode="config_only", spec=spec)
                if fingerprint in seen or str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                spec_path = planned_specs_dir / f"{spec['experiment_id']}.yaml"
                spec_path.write_text(
                    yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                seen.add(fingerprint)
                session["candidates"].append(
                    self._build_candidate_record(
                        candidate_id=f"rec-{_slugify(spec['experiment_id'])}",
                        source="recommendation",
                        source_id=str(getattr(rec, "parameter", "recommendation")),
                        execution_mode="config_only",
                        spec=spec,
                        spec_path=spec_path,
                    )
                )
                added += 1
                if added >= recommended_limit:
                    break

        if include_recipes and current_total + added < candidate_cap:
            context = {
                "search_id": search_id,
                "date": dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d"),
            }
            for recipe_id, recipe in sorted(
                self.recipe_registry.list_enabled(),
                key=lambda item: (0 if item[0].lower() in seed_recipe_ids else 1, item[0]),
            ):
                if current_total + added >= candidate_cap:
                    break
                spec = self.recipe_registry.build_candidate_spec(
                    recipe_id,
                    recipe,
                    requested_by=requested_by,
                    context=context,
                )
                fingerprint = _spec_fingerprint(
                    execution_mode=str(recipe.get("execution_mode", "code_recipe")),
                    spec=spec,
                    recipe_id=recipe_id,
                )
                if fingerprint in seen or str(spec["experiment_id"]) in logged_experiment_ids:
                    continue
                spec_path = planned_specs_dir / f"{spec['experiment_id']}.yaml"
                spec_path.write_text(
                    yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                seen.add(fingerprint)
                session["candidates"].append(
                    self._build_candidate_record(
                        candidate_id=str(recipe.get("candidate_id", f"recipe-{recipe_id.lower()}")),
                        source="recipe_catalog",
                        source_id=recipe_id,
                        execution_mode=str(recipe.get("execution_mode", "code_recipe")),
                        spec=spec,
                        spec_path=spec_path,
                        recipe_id=recipe_id,
                        recipe_parameters=dict(recipe.get("parameters", {}) or {}),
                    )
                )
                added += 1

        if added > 0:
            session["replan_rounds"] = int(session.get("replan_rounds", 0) or 0) + 1
            session["status"] = "planned"
            self._persist_session_update(session)
            self._append_journal(
                search_id,
                {
                    "event": "replanned",
                    "added_candidates": added,
                    "replan_round": session["replan_rounds"],
                },
            )
        return added
