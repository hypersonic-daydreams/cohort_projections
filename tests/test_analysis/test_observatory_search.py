"""Tests for deterministic Observatory autonomous-search infrastructure."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cohort_projections.analysis.observatory.deep_search import (
    DeepSearchPolicy,
    SearchPack,
    load_search_packs,
    resolve_parallelism,
)
from cohort_projections.analysis.observatory.recipe_registry import RecipeRegistry
from cohort_projections.analysis.observatory.sandbox_manager import SandboxManager
from cohort_projections.analysis.observatory.search_controller import (
    AutonomousSearchController,
)
from cohort_projections.analysis.observatory.search_policy import load_search_policy


def _git(repo: Path, *args: str) -> None:
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture()
def sandbox_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "analysis").mkdir(parents=True)
    (repo / "cohort_projections" / "analysis").mkdir(parents=True)
    (repo / "config").mkdir()
    (repo / "data" / "analysis" / "experiments").mkdir(parents=True)
    (repo / "scripts" / "analysis" / "example.py").write_text(
        "VALUE = 'before'\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("sandbox\n", encoding="utf-8")

    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


@pytest.fixture()
def search_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
    (project / "config" / "observatory_search_packs").mkdir(parents=True)
    (project / "data" / "analysis" / "experiments").mkdir(parents=True)
    (project / "data" / "analysis" / "benchmark_history").mkdir(parents=True)
    (project / "config" / "observatory_variants.yaml").write_text(
        yaml.safe_dump(
            {
                "base_method": "m2026r1",
                "base_config": "cfg-20260309-college-fix-v1",
                "variants": {
                    "EXP-A": {
                        "name": "College blend 1.0",
                        "parameter": "college_blend_factor",
                        "value": 1.0,
                        "tier": 1,
                        "config_only": True,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "config" / "observatory_recipes.yaml").write_text(
        yaml.safe_dump(
            {
                "recipes": {
                    "RX-TEST": {
                        "enabled": True,
                        "execution_mode": "code_recipe",
                        "candidate_id": "recipe-safe",
                        "experiment_id": "exp-{date}-recipe-safe",
                        "hypothesis": "Safe deterministic recipe candidate",
                        "base_method": "m2026r1",
                        "base_config": "cfg-20260309-college-fix-v1",
                        "scope": "county",
                        "benchmark_label": "recipe-safe",
                        "method_id_override": "m2026r1_recipe_safe",
                        "steps": [],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "config" / "observatory_search_policy.yaml").write_text(
        yaml.safe_dump(
            {
                "search": {
                    "runtime_root": "runtime",
                    "session_root": "data/analysis/experiments/search_runs",
                    "mirror_repo": "runtime/repos/cohort_projections.git",
                    "worktree_root": "runtime/worktrees",
                    "recipe_catalog": "config/observatory_recipes.yaml",
                    "search_pack_root": "config/observatory_search_packs",
                    "protected_paths": ["README.md"],
                    "allowed_recipe_roots": ["scripts/analysis", "cohort_projections"],
                    "deep_search": {
                        "time_budget_hours": 4.0,
                        "total_candidate_cap": 12,
                        "plateau_rounds": 2,
                        "min_improvement_pp": 0.02,
                        "operational_repeat_limit": 3,
                        "operational_failure_window": 10,
                        "operational_failure_rate": 0.2,
                        "max_patch_files": 5,
                        "max_patch_lines": 300,
                    },
                    "planner": {
                        "max_pending": 1,
                        "max_recommended": 1,
                        "include_recipe_catalog": True,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "config" / "observatory_search_packs" / "cf001.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "cf001",
                "label": "CF-001",
                "default": True,
                "scope": "county",
                "base_method": "m2026r1",
                "base_config": "cfg-20260309-college-fix-v1",
                "objective_order": [
                    "state_ape_recent_short",
                    "county_mape_overall",
                ],
                "hard_guardrails": {"negative_population_violations": 0},
                "parameter_bounds": {"college_blend_factor": {"min": 0.5, "max": 1.2}},
                "interaction_rules": [{"parameters": ["college_blend_factor"]}],
                "seed_candidates": [
                    {"source": "variant_catalog", "source_id": "EXP-A"},
                    {"source": "recipe_catalog", "source_id": "RX-TEST"},
                ],
                "stop_policy": {"plateau_rounds": 2},
                "code_mutators": [
                    {
                        "mutator_id": "mut-disabled",
                        "kind": "deterministic",
                        "candidate_id": "mut-disabled",
                        "benchmark_label": "mut-disabled",
                        "hypothesis": "Disabled placeholder mutator.",
                        "enabled": False,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "data" / "analysis" / "experiments" / "experiment_log.csv").write_text(
        ",".join(
            [
                "experiment_id",
                "run_date",
                "hypothesis",
                "base_method",
                "config_delta_summary",
                "run_id",
                "outcome",
                "key_metrics_summary",
                "interpretation",
                "next_action",
                "agent_or_human",
                "spec_path",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _git(project, "init", "-b", "main")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test User")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "initial")
    return project


class TestSearchPolicy:
    def test_policy_resolves_relative_paths(self, sandbox_repo: Path) -> None:
        policy_path = sandbox_repo / "config" / "observatory_search_policy.yaml"
        policy_path.write_text(
            yaml.safe_dump(
                {
                    "search": {
                        "runtime_root": "runtime",
                        "session_root": "sessions",
                        "mirror_repo": "runtime/repos/repo.git",
                        "worktree_root": "runtime/worktrees",
                        "recipe_catalog": "config/recipes.yaml",
                        "search_pack_root": "config/search-packs",
                        "protected_paths": ["README.md"],
                        "allowed_recipe_roots": ["scripts/analysis"],
                    }
                }
            ),
            encoding="utf-8",
        )
        policy = load_search_policy(policy_path, project_root=sandbox_repo)
        assert policy.runtime_root == sandbox_repo / "runtime"
        assert policy.search_pack_root == sandbox_repo / "config" / "search-packs"
        assert policy.is_protected_path(Path("README.md")) is True
        assert policy.is_allowed_recipe_target(Path("scripts/analysis/example.py")) is True
        assert policy.is_allowed_recipe_target(Path("README.md")) is False


class TestDeepSearchHelpers:
    def test_resolve_parallelism_matches_shared_cpu_allocator(self) -> None:
        assert resolve_parallelism(4) == (1, 4)
        assert resolve_parallelism(8) == (2, 4)
        assert resolve_parallelism(16) == (3, 5)
        assert resolve_parallelism(32) == (4, 8)
        assert resolve_parallelism(64) == (8, 8)

    def test_load_search_packs_validates_required_fields(self, tmp_path: Path) -> None:
        pack_root = tmp_path / "packs"
        pack_root.mkdir()
        (pack_root / "valid.yaml").write_text(
            yaml.safe_dump(
                {
                    "pack_id": "cf001",
                    "scope": "county",
                    "base_method": "m2026r1",
                    "base_config": "cfg-20260309-college-fix-v1",
                    "objective_order": ["county_mape_overall"],
                    "hard_guardrails": {},
                    "parameter_bounds": {},
                    "interaction_rules": [],
                    "seed_candidates": [],
                    "stop_policy": {},
                }
            ),
            encoding="utf-8",
        )

        packs = load_search_packs(pack_root)

        assert isinstance(packs["cf001"], SearchPack)

    def test_deep_search_policy_uses_leaf_mapping_defaults(self, tmp_path: Path) -> None:
        policy = DeepSearchPolicy.from_mapping(
            {
                "time_budget_hours": 4.5,
                "total_candidate_cap": 12,
                "plateau_rounds": 3,
            },
            cpu_budget=12,
            search_pack_root=tmp_path,
        )

        assert policy.time_budget_hours == pytest.approx(4.5)
        assert policy.total_candidate_cap == 12
        assert policy.plateau_rounds == 3
        assert policy.parallel_runs == 2


class TestRecipeRegistry:
    def test_apply_replace_text(self, sandbox_repo: Path) -> None:
        catalog_path = sandbox_repo / "config" / "recipes.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "recipes": {
                        "RX": {
                            "enabled": True,
                            "steps": [
                                {
                                    "type": "replace_text",
                                    "path": "scripts/analysis/example.py",
                                    "old": "before",
                                    "new": "after",
                                }
                            ],
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        policy_path = sandbox_repo / "config" / "observatory_search_policy.yaml"
        policy_path.write_text(
            yaml.safe_dump(
                {
                    "search": {
                        "runtime_root": "runtime",
                        "session_root": "sessions",
                        "mirror_repo": "runtime/repos/repo.git",
                        "worktree_root": "runtime/worktrees",
                        "recipe_catalog": "config/recipes.yaml",
                        "protected_paths": ["README.md"],
                        "allowed_recipe_roots": ["scripts/analysis"],
                    }
                }
            ),
            encoding="utf-8",
        )
        policy = load_search_policy(policy_path, project_root=sandbox_repo)
        registry = RecipeRegistry(catalog_path)
        applied = registry.apply(
            "RX",
            worktree_root=sandbox_repo,
            policy=policy,
        )
        assert applied.changed_files == ("scripts/analysis/example.py",)
        assert "after" in (sandbox_repo / "scripts" / "analysis" / "example.py").read_text(
            encoding="utf-8"
        )


class TestSandboxManager:
    def test_mirror_and_worktree_are_isolated(self, sandbox_repo: Path) -> None:
        policy_path = sandbox_repo / "config" / "observatory_search_policy.yaml"
        policy_path.write_text(
            yaml.safe_dump(
                {
                    "search": {
                        "runtime_root": "runtime",
                        "session_root": "sessions",
                        "mirror_repo": "runtime/repos/repo.git",
                        "worktree_root": "runtime/worktrees",
                        "recipe_catalog": "config/recipes.yaml",
                        "protected_paths": ["README.md"],
                        "allowed_recipe_roots": ["scripts/analysis"],
                    }
                }
            ),
            encoding="utf-8",
        )
        policy = load_search_policy(policy_path, project_root=sandbox_repo)
        manager = SandboxManager(policy, source_repo=sandbox_repo)
        signature = manager.live_checkout_signature()
        context = manager.create_worktree(
            search_id="search-one",
            candidate_id="cand-a",
            base_revision="main",
        )
        changed_file = context.worktree_path / "scripts" / "analysis" / "example.py"
        changed_file.write_text("VALUE = 'changed'\n", encoding="utf-8")
        changed_files, diff_patch = manager.capture_diff(
            worktree_path=context.worktree_path,
            base_revision="main",
        )
        assert "scripts/analysis/example.py" in changed_files
        assert "changed" in diff_patch
        manager.assert_live_checkout_unchanged(signature)
        manager.remove_worktree(context)
        assert not context.worktree_path.exists()
        assert manager.live_checkout_signature() == signature

    def test_data_symlink_overlay_preserves_tracked_data_docs(
        self, sandbox_repo: Path
    ) -> None:
        (sandbox_repo / ".gitignore").write_text("data/raw/**/*.csv\n", encoding="utf-8")
        notes_path = sandbox_repo / "data" / "raw" / "census" / "DATA_SOURCE_NOTES.md"
        notes_path.parent.mkdir(parents=True)
        notes_path.write_text("tracked notes\n", encoding="utf-8")
        _git(sandbox_repo, "add", ".gitignore", str(notes_path.relative_to(sandbox_repo)))
        _git(sandbox_repo, "commit", "-m", "add tracked data docs")

        ignored_data = sandbox_repo / "data" / "raw" / "census" / "source.csv"
        ignored_data.write_text("value\n1\n", encoding="utf-8")
        policy_path = sandbox_repo / "config" / "observatory_search_policy.yaml"
        policy_path.write_text(
            yaml.safe_dump(
                {
                    "search": {
                        "runtime_root": "runtime",
                        "session_root": "sessions",
                        "mirror_repo": "runtime/repos/repo.git",
                        "worktree_root": "runtime/worktrees",
                        "recipe_catalog": "config/recipes.yaml",
                        "protected_paths": ["README.md"],
                        "allowed_recipe_roots": ["scripts/analysis"],
                    }
                }
            ),
            encoding="utf-8",
        )
        policy = load_search_policy(policy_path, project_root=sandbox_repo)
        manager = SandboxManager(policy, source_repo=sandbox_repo)

        context = manager.create_worktree(
            search_id="search-data",
            candidate_id="cand-data",
            base_revision="main",
        )

        worktree_notes = context.worktree_path / notes_path.relative_to(sandbox_repo)
        worktree_data = context.worktree_path / ignored_data.relative_to(sandbox_repo)
        assert worktree_notes.read_text(encoding="utf-8") == "tracked notes\n"
        assert worktree_data.is_symlink()
        assert worktree_data.resolve() == ignored_data

        changed_files, diff_patch = manager.capture_diff(
            worktree_path=context.worktree_path,
            base_revision="main",
        )
        assert changed_files == []
        assert diff_patch == ""

        manager.remove_worktree(context)


class TestSearchController:
    def test_plan_session_collects_variants_and_recipes(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )
        session = controller.plan_session(search_id="search-one")
        assert session["summary"]["total"] == 2
        assert session["summary"]["planned"] == 2
        assert session["mode"] == "deep_search"
        assert session["search_pack_id"] == "cf001"
        assert session["cpu_budget"] == 12
        assert session["parallel_runs"] == 2
        assert session["workers_per_run"] == 6
        assert session["time_budget_hours"] == pytest.approx(4.0)
        assert (
            search_project
            / "data"
            / "analysis"
            / "experiments"
            / "search_runs"
            / "search-one"
            / "planned_specs"
        ).exists()
        assert (
            search_project
            / "data"
            / "analysis"
            / "experiments"
            / "search_runs"
            / "search-one"
            / "search_journal.jsonl"
        ).exists()

        status = controller.status(search_id="search-one")
        assert status["summary"]["planned"] == 2

        report = controller.render_report(search_id="search-one")
        assert "variant-exp-a" in report
        assert "recipe-safe" in report

    def test_run_session_dry_run(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )
        controller.plan_session(search_id="search-two")
        result = controller.run_session(search_id="search-two", dry_run=True, run_budget=1)
        assert result["mode"] == "dry-run"
        assert result["runnable"] == 2
        assert len(result["preview"]) == 1

    def test_write_session_artifacts_creates_candidate_summary(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )
        controller.plan_session(search_id="search-artifacts")
        session = controller.load_session("search-artifacts")
        session["candidates"][0]["status"] = "completed"
        session["candidates"][0]["result"] = {
            "outcome": "passed_all_gates",
            "run_id": "br-test",
            "method_id": "m2026r1",
            "config_id": "cfg-test",
            "benchmark_summary": {
                "primary_metric": "county_mape_overall",
                "metrics": {"county_mape_overall": 8.1},
                "deltas": {"county_mape_overall": -0.2},
            },
        }
        controller._write_session("search-artifacts", session)

        artifacts = controller.write_session_artifacts(search_id="search-artifacts")
        summary_csv = search_project / artifacts["candidate_summary_csv"]
        frontier_csv = search_project / artifacts["frontier_csv"]
        deep_brief_json = search_project / artifacts["deep_search_brief_json"]
        deep_brief_md = search_project / artifacts["deep_search_brief_markdown"]
        report_md = search_project / artifacts["search_report_markdown"]
        assert summary_csv.exists()
        assert frontier_csv.exists()
        assert deep_brief_json.exists()
        assert deep_brief_md.exists()
        assert report_md.exists()
        summary_text = summary_csv.read_text(encoding="utf-8")
        assert "candidate_id" in summary_text
        assert "delta_county_mape_overall" in summary_text
        assert "search-artifacts" in deep_brief_md.read_text(encoding="utf-8")
        assert "Best Completed Candidates" in report_md.read_text(encoding="utf-8")

    def test_run_to_completion_batches_until_finished(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )

        call_count = {"n": 0}

        def _fake_run_session(
            *,
            search_id: str,
            run_budget: int | None = None,
            dry_run: bool = False,
            keep_worktrees: bool | None = None,
            workers_per_run: int = 0,
            parallel_runs: int | None = None,
        ) -> dict[str, object]:
            del dry_run, keep_worktrees, run_budget, parallel_runs
            session = controller.load_session(search_id)
            planned = [c for c in session["candidates"] if c["status"] == "planned"]
            if planned:
                planned[0]["status"] = "completed"
                planned[0]["result"] = {
                    "outcome": "passed_all_gates",
                    "run_id": f"run-{call_count['n']}",
                    "method_id": "m2026r1",
                    "config_id": f"cfg-{call_count['n']}",
                    "benchmark_summary": {
                        "primary_metric": "county_mape_overall",
                        "metrics": {"county_mape_overall": 8.0 - call_count["n"]},
                        "deltas": {"county_mape_overall": -0.1},
                    },
                }
            call_count["n"] += 1
            session["summary"] = controller._summarize_candidates(session["candidates"])
            if session["summary"]["planned"] == 0:
                session["status"] = "finished"
            controller._write_session(search_id, session)
            return {
                "search_id": search_id,
                "executed": 1 if planned else 0,
                "summary": session["summary"],
            }

        controller.run_session = _fake_run_session  # type: ignore[method-assign]
        result = controller.run_to_completion(
            search_id="search-loop",
            overwrite=True,
            batch_run_budget=1,
            max_total_runs=2,
        )
        assert result["executed_total"] == 2
        assert len(result["batches"]) == 2
        assert "candidate_summary_csv" in result["artifacts"]

    def test_plan_session_respects_zero_limits(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )

        session = controller.plan_session(
            search_id="search-zero-limits",
            max_pending=0,
            max_recommended=0,
            include_recipe_catalog=False,
        )

        assert session["status"] == "finished"
        assert session["summary"]["total"] == 0
        assert session["summary"]["planned"] == 0
        assert session["candidates"] == []

    def test_plan_session_skips_recipe_candidates_already_logged(
        self, search_project: Path
    ) -> None:
        log_path = search_project / "data" / "analysis" / "experiments" / "experiment_log.csv"
        today = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d")
        pd.DataFrame(
            [
                {
                    "experiment_id": f"exp-{today}-recipe-safe",
                    "run_date": "2026-03-19",
                    "hypothesis": "already tested",
                    "base_method": "m2026r1",
                    "config_delta_summary": "",
                    "run_id": "not_run",
                    "outcome": "inconclusive",
                    "key_metrics_summary": "",
                    "interpretation": "prior attempt",
                    "next_action": "flag_for_review",
                    "agent_or_human": "system",
                    "spec_path": "data/analysis/experiments/completed/exp-test.yaml",
                }
            ]
        ).to_csv(log_path, mode="a", header=False, index=False)

        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )
        session = controller.plan_session(
            search_id="search-skip-logged-recipe",
            max_pending=0,
            max_recommended=0,
            include_recipe_catalog=True,
        )

        assert session["summary"]["total"] == 0
        assert session["candidates"] == []

    def test_run_session_marks_logged_candidates_completed_without_execution(
        self,
        search_project: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )
        controller.plan_session(
            search_id="search-runtime-skip",
            max_pending=0,
            max_recommended=0,
            include_recipe_catalog=True,
        )

        log_path = search_project / "data" / "analysis" / "experiments" / "experiment_log.csv"
        today = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d")
        pd.DataFrame(
            [
                {
                    "experiment_id": f"exp-{today}-recipe-safe",
                    "run_date": "2026-03-19",
                    "hypothesis": "already tested",
                    "base_method": "m2026r1",
                    "config_delta_summary": "",
                    "run_id": "not_run",
                    "outcome": "inconclusive",
                    "key_metrics_summary": "existing result",
                    "interpretation": "prior attempt",
                    "next_action": "flag_for_review",
                    "agent_or_human": "system",
                    "spec_path": "data/analysis/experiments/completed/exp-test.yaml",
                }
            ]
        ).to_csv(log_path, mode="a", header=False, index=False)

        def _unexpected_create_worktree(*args: object, **kwargs: object) -> None:
            raise AssertionError("create_worktree should not be called for logged candidates")

        monkeypatch.setattr(controller.sandbox_manager, "assert_live_checkout_clean", lambda: None)
        monkeypatch.setattr(controller.sandbox_manager, "live_checkout_signature", lambda: "")
        monkeypatch.setattr(
            controller.sandbox_manager,
            "assert_live_checkout_unchanged",
            lambda signature: None,
        )
        monkeypatch.setattr(
            controller.sandbox_manager, "create_worktree", _unexpected_create_worktree
        )

        result = controller.run_session(search_id="search-runtime-skip", run_budget=1)
        session = controller.load_session("search-runtime-skip")

        assert result["executed"] == 0
        assert session["candidates"][0]["status"] == "completed"
        assert session["candidates"][0]["result"]["outcome"] == "inconclusive"
        assert session["candidates"][0]["result"]["key_metrics_summary"] == "existing result"

    def test_validate_code_changes_enforces_patch_limits(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )

        with pytest.raises(RuntimeError, match="touched 2 files"):
            controller._validate_code_changes(
                changed_files=["a.py", "b.py"],
                diff_patch="+one\n+two\n",
                max_patch_files=1,
                max_patch_lines=10,
            )

        with pytest.raises(RuntimeError, match="changed 4 lines"):
            controller._validate_code_changes(
                changed_files=["a.py"],
                diff_patch="--- a.py\n+++ a.py\n+one\n-two\n+three\n-four\n",
                max_patch_files=5,
                max_patch_lines=3,
            )

    def test_operational_stop_state_detects_repeated_blockers(self, search_project: Path) -> None:
        controller = AutonomousSearchController(
            store=None,
            observatory_config={
                "history_dir": "data/analysis/benchmark_history",
                "experiment_log": "data/analysis/experiments/experiment_log.csv",
                "variant_catalog": "config/observatory_variants.yaml",
            },
            policy_path=search_project / "config" / "observatory_search_policy.yaml",
            project_root=search_project,
            source_repo=search_project,
        )
        candidates = pd.DataFrame(
            [
                {
                    "status": "completed",
                    "outcome": "operational_blocker",
                    "last_error": "bundle incomplete",
                },
                {
                    "status": "completed",
                    "outcome": "operational_blocker",
                    "last_error": "bundle incomplete",
                },
                {
                    "status": "completed",
                    "outcome": "operational_blocker",
                    "last_error": "bundle incomplete",
                },
            ]
        )

        stop_reason = controller._operational_stop_state(
            candidates,
            repeat_limit=3,
            failure_window=10,
            failure_rate=1.0,
        )

        assert stop_reason == "repeated_operational_blocker"


def test_repo_recipe_catalog_has_unique_search_lattice() -> None:
    project_root = Path(__file__).resolve().parents[2]
    registry = RecipeRegistry(project_root / "config" / "observatory_recipes.yaml")
    enabled = registry.list_enabled()

    assert len(enabled) >= 9

    candidate_ids: set[str] = set()
    method_ids: set[str] = set()
    saw_recent_window = False
    saw_mortality = False
    saw_interaction = False

    for _, recipe in enabled:
        candidate_id = str(recipe["candidate_id"])
        method_id = str(recipe["method_id_override"])

        assert candidate_id not in candidate_ids
        assert method_id not in method_ids
        assert recipe["execution_mode"] == "code_recipe"
        assert recipe["base_method"] == "m2026r1"
        assert recipe["base_config"] == "cfg-20260309-college-fix-v1"
        assert recipe.get("steps", []) == []

        candidate_ids.add(candidate_id)
        method_ids.add(method_id)

        config_delta = recipe.get("config_delta", {})
        if {
            "convergence_recent_period_count",
            "convergence_medium_period_count",
        }.issubset(config_delta):
            saw_recent_window = True
        if "mortality_improvement_rate" in config_delta:
            saw_mortality = True
        if {
            "convergence_recent_period_count",
            "convergence_medium_period_count",
            "mortality_improvement_rate",
        }.issubset(config_delta):
            saw_interaction = True

    assert saw_recent_window is True
    assert saw_mortality is True
    assert saw_interaction is True
