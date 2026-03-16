"""Tests for deterministic Observatory autonomous-search infrastructure."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from cohort_projections.analysis.observatory.recipe_registry import RecipeRegistry
from cohort_projections.analysis.observatory.sandbox_manager import SandboxManager
from cohort_projections.analysis.observatory.search_controller import (
    AutonomousSearchController,
)
from cohort_projections.analysis.observatory.search_policy import load_search_policy


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
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

    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


@pytest.fixture()
def search_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "config").mkdir(parents=True)
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
                    "protected_paths": ["README.md"],
                    "allowed_recipe_roots": ["scripts/analysis", "cohort_projections"],
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
    _git(project, "init")
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
                        "protected_paths": ["README.md"],
                        "allowed_recipe_roots": ["scripts/analysis"],
                    }
                }
            ),
            encoding="utf-8",
        )
        policy = load_search_policy(policy_path, project_root=sandbox_repo)
        assert policy.runtime_root == sandbox_repo / "runtime"
        assert policy.is_protected_path(Path("README.md")) is True
        assert policy.is_allowed_recipe_target(Path("scripts/analysis/example.py")) is True
        assert policy.is_allowed_recipe_target(Path("README.md")) is False


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
            base_revision="HEAD",
        )
        changed_file = context.worktree_path / "scripts" / "analysis" / "example.py"
        changed_file.write_text("VALUE = 'changed'\n", encoding="utf-8")
        changed_files, diff_patch = manager.capture_diff(
            worktree_path=context.worktree_path,
            base_revision="HEAD",
        )
        assert "scripts/analysis/example.py" in changed_files
        assert "changed" in diff_patch
        manager.assert_live_checkout_unchanged(signature)
        manager.remove_worktree(context)
        assert not context.worktree_path.exists()
        assert manager.live_checkout_signature() == signature


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
        assert (search_project / "data" / "analysis" / "experiments" / "search_runs" / "search-one" / "planned_specs").exists()

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

    def test_write_session_artifacts_creates_candidate_summary(
        self, search_project: Path
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
        report_md = search_project / artifacts["search_report_markdown"]
        assert summary_csv.exists()
        assert report_md.exists()
        summary_text = summary_csv.read_text(encoding="utf-8")
        assert "candidate_id" in summary_text
        assert "delta_county_mape_overall" in summary_text
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
        ) -> dict[str, object]:
            del dry_run, keep_worktrees, run_budget
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
        if (
            {
                "convergence_recent_period_count",
                "convergence_medium_period_count",
                "mortality_improvement_rate",
            }.issubset(config_delta)
        ):
            saw_interaction = True

    assert saw_recent_window is True
    assert saw_mortality is True
    assert saw_interaction is True
