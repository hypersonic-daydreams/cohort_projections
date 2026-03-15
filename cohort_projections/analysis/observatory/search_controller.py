"""Deterministic, worktree-isolated search orchestration for the Observatory."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cohort_projections.analysis.benchmarking import append_benchmark_index
from cohort_projections.analysis.experiment_log import (
    append_experiment_entry,
    get_tested_hypotheses,
    read_experiment_log,
)
from cohort_projections.analysis.observatory.comparator import ObservatoryComparator
from cohort_projections.analysis.observatory.recipe_registry import RecipeRegistry
from cohort_projections.analysis.observatory.recommender import ObservatoryRecommender
from cohort_projections.analysis.observatory.sandbox_manager import SandboxManager
from cohort_projections.analysis.observatory.search_policy import (
    PROJECT_ROOT,
    SearchPolicy,
    load_search_policy,
)
from cohort_projections.analysis.observatory.variant_catalog import VariantCatalog


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
    ) -> None:
        self.project_root = project_root.resolve()
        self.store = store
        self.observatory_config = observatory_config or {}
        self.policy: SearchPolicy = load_search_policy(
            policy_path, project_root=self.project_root
        )
        self.recipe_registry = RecipeRegistry(self.policy.recipe_catalog)
        self.sandbox_manager = SandboxManager(
            self.policy, source_repo=(source_repo or self.project_root)
        )
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

    def plan_session(
        self,
        *,
        search_id: str,
        base_revision: str = "HEAD",
        max_pending: int | None = None,
        max_recommended: int | None = None,
        include_recipe_catalog: bool | None = None,
        overwrite: bool = False,
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
        seen: set[str] = set()
        candidates: list[dict[str, Any]] = []
        requested_by = "system"

        resolved_base_revision = self._resolve_git_revision(base_revision)
        pending_limit = max_pending or self.policy.default_max_pending
        recommended_limit = max_recommended or self.policy.default_max_recommended
        include_recipes = (
            self.policy.include_recipe_catalog
            if include_recipe_catalog is None
            else include_recipe_catalog
        )

        if catalog is not None and pending_limit > 0:
            ordered = sorted(
                catalog.get_untested(),
                key=lambda variant: (
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
            base_method = (
                getattr(catalog_for_bounds, "base_method", None)
                or self.observatory_config.get("challenger_base_method", "m2026r1")
            )
            base_config = (
                getattr(catalog_for_bounds, "base_config", None)
                or self.observatory_config.get("base_config", "cfg-20260309-college-fix-v1")
            )
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
                if len([c for c in candidates if c["source"] == "recommendation"]) >= recommended_limit:
                    break

        if include_recipes:
            context = {
                "search_id": search_id,
                "date": dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d"),
            }
            for recipe_id, recipe in self.recipe_registry.list_enabled():
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

        session = {
            "search_id": search_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "planned",
            "base_revision": base_revision,
            "resolved_base_revision": resolved_base_revision,
            "policy_path": str(self.policy.policy_path),
            "history_dir": str(self.history_dir),
            "experiment_log": str(self.experiment_log_path),
            "planner": {
                "max_pending": pending_limit,
                "max_recommended": recommended_limit,
                "include_recipe_catalog": include_recipes,
            },
            "candidates": candidates,
            "summary": self._summarize_candidates(candidates),
        }
        self._write_session(search_id, session)
        return session

    def run_session(
        self,
        *,
        search_id: str,
        run_budget: int | None = None,
        dry_run: bool = False,
        keep_worktrees: bool | None = None,
    ) -> dict[str, Any]:
        """Execute queued candidates inside isolated worktrees."""
        session = self.load_session(search_id)
        budget = run_budget if run_budget is not None else self.policy.default_run_budget
        keep = self.policy.keep_worktrees if keep_worktrees is None else keep_worktrees
        runnable = [candidate for candidate in session["candidates"] if candidate["status"] == "planned"]
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

        executed = 0
        for candidate in session["candidates"]:
            if candidate["status"] != "planned":
                continue
            if budget is not None and executed >= budget:
                break
            executed += 1
            candidate["status"] = "running"
            candidate["attempts"] = int(candidate.get("attempts", 0)) + 1
            candidate["started_at"] = _utc_now()
            self._persist_session_update(session)

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
                command_result = self._run_worktree_experiment(
                    context.worktree_path,
                    spec_path,
                )
                artifacts = self._harvest_candidate(
                    search_id=search_id,
                    candidate=candidate,
                    context=context,
                    changed_files=changed_files,
                    diff_patch=diff_patch,
                    command_result=command_result,
                    applied_recipe=applied_recipe,
                )
                candidate["status"] = "completed"
                candidate["completed_at"] = _utc_now()
                candidate["result"] = artifacts
                candidate["last_error"] = ""
            except Exception as exc:
                candidate["status"] = "failed"
                candidate["completed_at"] = _utc_now()
                candidate["last_error"] = str(exc)
            finally:
                self._persist_session_update(session)
                if not keep:
                    self.sandbox_manager.remove_worktree(context)
                self.sandbox_manager.assert_live_checkout_unchanged(live_signature)

        session["updated_at"] = _utc_now()
        if all(candidate["status"] in {"completed", "failed"} for candidate in session["candidates"]):
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
        lines = [
            f"# Autonomous Search Report: {search_id}",
            "",
            f"- Status: `{session['status']}`",
            f"- Created: `{session['created_at']}`",
            f"- Updated: `{session['updated_at']}`",
            f"- Base revision: `{session['resolved_base_revision']}`",
            "",
            "## Summary",
            "",
        ]
        summary = self._summarize_candidates(session["candidates"])
        lines.extend(
            f"- {key.replace('_', ' ').title()}: `{summary.get(key, 0)}`"
            for key in ("total", "planned", "running", "completed", "failed")
        )
        lines.extend(["", "## Candidates", ""])
        for candidate in session["candidates"]:
            result = candidate.get("result", {})
            lines.append(
                f"- `{candidate['candidate_id']}` | `{candidate['status']}` | "
                f"source=`{candidate['source']}` | outcome=`{result.get('outcome', '')}` | "
                f"run_id=`{result.get('run_id', '')}`"
            )
        return "\n".join(lines).rstrip() + "\n"

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
    ) -> subprocess.CompletedProcess[str]:
        script_path = worktree_path / "scripts" / "analysis" / "run_experiment.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--spec", str(spec_path)],
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
    ) -> dict[str, Any]:
        session_dir = self._session_dir(search_id)
        completed_specs_dir = session_dir / "completed_specs"
        patches_dir = session_dir / "patches"
        logs_dir = session_dir / "logs"
        profiles_dir = session_dir / "profiles"
        for directory in (completed_specs_dir, patches_dir, logs_dir, profiles_dir):
            directory.mkdir(parents=True, exist_ok=True)

        experiment_id = str(candidate["spec"]["experiment_id"])
        local_log_path = (
            context.worktree_path / "data" / "analysis" / "experiments" / "experiment_log.csv"
        )
        local_log = read_experiment_log(local_log_path)
        matching = local_log[local_log["experiment_id"] == experiment_id]
        if matching.empty:
            raise RuntimeError(f"Experiment log entry not found for {experiment_id}")
        row = matching.iloc[-1].to_dict()

        local_completed_spec = (
            context.worktree_path
            / "data"
            / "analysis"
            / "experiments"
            / "completed"
            / f"{experiment_id}.yaml"
        )
        if not local_completed_spec.exists():
            raise RuntimeError(f"Completed spec not found: {local_completed_spec}")

        copied_spec_path = completed_specs_dir / local_completed_spec.name
        shutil.copy2(local_completed_spec, copied_spec_path)

        stdout_path = logs_dir / f"{candidate['candidate_id']}.stdout.log"
        stderr_path = logs_dir / f"{candidate['candidate_id']}.stderr.log"
        stdout_path.write_text(command_result.stdout, encoding="utf-8")
        stderr_path.write_text(command_result.stderr, encoding="utf-8")

        patch_path = patches_dir / f"{candidate['candidate_id']}.patch"
        patch_path.write_text(diff_patch, encoding="utf-8")
        manifest_path = patches_dir / f"{candidate['candidate_id']}.json"
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

        copied_spec_relative = copied_spec_path.relative_to(self.project_root)
        row["spec_path"] = str(copied_spec_relative)
        if experiment_id not in get_tested_hypotheses(self.experiment_log_path):
            append_experiment_entry(row, log_path=self.experiment_log_path)

        run_id = str(row.get("run_id", "") or "")
        if run_id and run_id != "not_run":
            self._harvest_run_bundle(context.worktree_path, run_id)

        profile_artifact = ""
        predicted_method = _predict_method_id(candidate["spec"])
        predicted_config = _predict_config_id(candidate["spec"])
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

        return {
            "outcome": row.get("outcome", ""),
            "run_id": run_id,
            "spec_path": str(copied_spec_relative),
            "patch_path": str(patch_path.relative_to(self.project_root)),
            "patch_manifest_path": str(manifest_path.relative_to(self.project_root)),
            "stdout_log": str(stdout_path.relative_to(self.project_root)),
            "stderr_log": str(stderr_path.relative_to(self.project_root)),
            "profile_path": profile_artifact,
            "changed_files": changed_files,
        }

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
        return read_experiment_log(self.experiment_log_path)

    def _session_dir(self, search_id: str) -> Path:
        return self.policy.session_root / search_id

    def _session_file(self, search_id: str) -> Path:
        return self._session_dir(search_id) / "session.yaml"

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
