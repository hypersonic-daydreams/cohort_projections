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
