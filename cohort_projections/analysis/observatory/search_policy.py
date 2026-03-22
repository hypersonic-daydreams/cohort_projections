"""Policy loader and path guardrails for deterministic Observatory search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "observatory_search_policy.yaml"


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


@dataclass(frozen=True)
class SearchPolicy:
    """Resolved search policy with filesystem guardrails."""

    project_root: Path
    policy_path: Path
    runtime_root: Path
    session_root: Path
    mirror_repo: Path
    worktree_root: Path
    recipe_catalog: Path
    search_pack_root: Path
    protected_paths: tuple[Path, ...]
    allowed_recipe_roots: tuple[Path, ...]
    compile_changed_python: bool
    keep_worktrees: bool
    default_run_budget: int
    default_parallel_runs: int
    default_max_pending: int
    default_max_recommended: int
    include_recipe_catalog: bool
    deep_search: dict[str, Any]

    def relative_to_project(self, path: Path) -> Path:
        """Return *path* relative to project root when possible."""
        try:
            return path.resolve().relative_to(self.project_root.resolve())
        except ValueError:
            return path

    def is_protected_path(self, path: Path) -> bool:
        """Return whether *path* is blocked from recipe mutation."""
        absolute = path if path.is_absolute() else self.project_root / path
        resolved = absolute.resolve()
        return any(
            resolved == protected or resolved.is_relative_to(protected)
            for protected in self.protected_paths
        )

    def is_allowed_recipe_target(self, path: Path) -> bool:
        """Return whether *path* is inside an allowed recipe root."""
        absolute = path if path.is_absolute() else self.project_root / path
        resolved = absolute.resolve()
        if self.is_protected_path(resolved):
            return False
        return any(
            resolved == root or resolved.is_relative_to(root) for root in self.allowed_recipe_roots
        )


def load_search_policy(
    policy_path: Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> SearchPolicy:
    """Load the deterministic search policy from YAML."""
    resolved_policy_path = _resolve_project_path(project_root, policy_path or DEFAULT_POLICY_PATH)
    if not resolved_policy_path.exists():
        raise FileNotFoundError(f"Search policy file not found: {resolved_policy_path}")

    raw = yaml.safe_load(resolved_policy_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Search policy must be a YAML mapping: {resolved_policy_path}")

    section = raw.get("search", raw)
    if not isinstance(section, dict):
        raise ValueError("Search policy 'search' section must be a mapping.")

    planner = section.get("planner", {})
    if planner and not isinstance(planner, dict):
        raise ValueError("Search policy 'planner' section must be a mapping.")

    runtime_root = _resolve_project_path(
        project_root, section.get("runtime_root", "data/analysis/observatory_runtime")
    )
    session_root = _resolve_project_path(
        project_root,
        section.get("session_root", "data/analysis/experiments/search_runs"),
    )
    mirror_repo = _resolve_project_path(
        project_root,
        section.get(
            "mirror_repo",
            runtime_root / "repos" / "cohort_projections.git",
        ),
    )
    worktree_root = _resolve_project_path(
        project_root,
        section.get("worktree_root", runtime_root / "worktrees"),
    )
    recipe_catalog = _resolve_project_path(
        project_root,
        section.get("recipe_catalog", "config/observatory_recipes.yaml"),
    )
    search_pack_root = _resolve_project_path(
        project_root,
        section.get("search_pack_root", "config/observatory_search_packs"),
    )

    protected_paths = tuple(
        _resolve_project_path(project_root, raw_path).resolve()
        for raw_path in section.get(
            "protected_paths",
            [
                "config/method_profiles/aliases.yaml",
                "DEVELOPMENT_TRACKER.md",
                "docs/governance/adrs",
            ],
        )
    )
    allowed_recipe_roots = tuple(
        _resolve_project_path(project_root, raw_path).resolve()
        for raw_path in section.get(
            "allowed_recipe_roots",
            [
                "cohort_projections",
                "scripts/analysis",
                "config/method_profiles",
            ],
        )
    )

    return SearchPolicy(
        project_root=project_root.resolve(),
        policy_path=resolved_policy_path.resolve(),
        runtime_root=runtime_root.resolve(),
        session_root=session_root.resolve(),
        mirror_repo=mirror_repo.resolve(),
        worktree_root=worktree_root.resolve(),
        recipe_catalog=recipe_catalog.resolve(),
        search_pack_root=search_pack_root.resolve(),
        protected_paths=protected_paths,
        allowed_recipe_roots=allowed_recipe_roots,
        compile_changed_python=bool(section.get("compile_changed_python", True)),
        keep_worktrees=bool(section.get("keep_worktrees", False)),
        default_run_budget=int(section.get("default_run_budget", 3)),
        default_parallel_runs=int(section.get("default_parallel_runs", 1)),
        default_max_pending=int(planner.get("max_pending", 3)),
        default_max_recommended=int(planner.get("max_recommended", 5)),
        include_recipe_catalog=bool(planner.get("include_recipe_catalog", True)),
        deep_search=dict(section.get("deep_search", {}) or {}),
    )
