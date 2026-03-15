"""Declarative, deterministic file mutation recipes for Observatory search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

from cohort_projections.analysis.observatory.search_policy import SearchPolicy


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return Formatter().vformat(value, (), _SafeFormatDict(context))
    if isinstance(value, list):
        return [_render_template(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_template(subvalue, context) for key, subvalue in value.items()}
    return value


@dataclass(frozen=True)
class AppliedRecipe:
    """Summary of one recipe application."""

    recipe_id: str
    changed_files: tuple[str, ...]
    steps_applied: int
    parameters: dict[str, Any]


class RecipeRegistry:
    """Load and apply deterministic mutation recipes."""

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path
        if not catalog_path.exists():
            self._recipes: dict[str, dict[str, Any]] = {}
            return
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Recipe catalog must be a YAML mapping: {catalog_path}")
        recipes = raw.get("recipes", raw)
        if not isinstance(recipes, dict):
            raise ValueError("Recipe catalog 'recipes' section must be a mapping.")
        self._recipes = recipes

    def list_enabled(self) -> list[tuple[str, dict[str, Any]]]:
        """Return enabled recipe records."""
        return [
            (recipe_id, recipe)
            for recipe_id, recipe in self._recipes.items()
            if isinstance(recipe, dict) and bool(recipe.get("enabled", False))
        ]

    def get(self, recipe_id: str) -> dict[str, Any]:
        """Return one recipe definition."""
        recipe = self._recipes.get(recipe_id)
        if not isinstance(recipe, dict):
            raise KeyError(f"Recipe '{recipe_id}' not found in {self.catalog_path}")
        return recipe

    def apply(
        self,
        recipe_id: str,
        *,
        worktree_root: Path,
        policy: SearchPolicy,
        parameters: dict[str, Any] | None = None,
    ) -> AppliedRecipe:
        """Apply a recipe to a worktree and return the patch manifest."""
        recipe = self.get(recipe_id)
        steps = recipe.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError(f"Recipe '{recipe_id}' steps must be a list.")

        context = {"recipe_id": recipe_id, **(parameters or {})}
        changed_files: list[str] = []

        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"Recipe '{recipe_id}' has a non-mapping step.")
            step_type = str(step.get("type", "")).strip()
            rendered = _render_template(step, context)
            relative_path = Path(str(rendered.get("path", "")))
            if not relative_path.parts:
                raise ValueError(f"Recipe '{recipe_id}' step is missing 'path'.")
            target_path = (worktree_root / relative_path).resolve()
            if not policy.is_allowed_recipe_target(policy.relative_to_project(target_path)):
                raise ValueError(
                    f"Recipe '{recipe_id}' attempted to edit protected path: {relative_path}"
                )

            if step_type == "replace_text":
                self._replace_text(target_path, rendered)
            elif step_type == "append_text":
                self._append_text(target_path, rendered)
            elif step_type == "write_file":
                self._write_file(target_path, rendered)
            else:
                raise ValueError(f"Unsupported recipe step type: {step_type!r}")

            rel_changed = str(target_path.relative_to(worktree_root))
            if rel_changed not in changed_files:
                changed_files.append(rel_changed)

        return AppliedRecipe(
            recipe_id=recipe_id,
            changed_files=tuple(changed_files),
            steps_applied=len(steps),
            parameters=parameters or {},
        )

    @staticmethod
    def build_candidate_spec(
        recipe_id: str,
        recipe: dict[str, Any],
        *,
        requested_by: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render a recipe definition into an experiment spec."""
        rendered = _render_template(recipe, context or {})
        required = [
            "experiment_id",
            "hypothesis",
            "base_method",
            "base_config",
            "scope",
            "benchmark_label",
        ]
        missing = [key for key in required if key not in rendered]
        if missing:
            raise ValueError(f"Recipe '{recipe_id}' is missing required spec fields: {missing}")

        spec = {
            "experiment_id": rendered["experiment_id"],
            "hypothesis": rendered["hypothesis"],
            "base_method": rendered["base_method"],
            "base_config": rendered["base_config"],
            "config_delta": rendered.get("config_delta", {}),
            "scope": rendered["scope"],
            "benchmark_label": rendered["benchmark_label"],
            "requested_by": requested_by,
        }
        for optional in (
            "expected_improvement",
            "risk_areas",
            "method_id_override",
            "config_id_override",
            "notes",
        ):
            if optional in rendered:
                spec[optional] = rendered[optional]
        return spec

    @staticmethod
    def _replace_text(path: Path, step: dict[str, Any]) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Recipe replace_text target does not exist: {path}")
        old = str(step.get("old", ""))
        new = str(step.get("new", ""))
        count = int(step.get("count", 1))
        required = bool(step.get("required", True))
        contents = path.read_text(encoding="utf-8")
        if old not in contents:
            if required:
                raise ValueError(f"Expected text not found in {path}")
            return
        updated = contents.replace(old, new, count)
        path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _append_text(path: Path, step: dict[str, Any]) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Recipe append_text target does not exist: {path}")
        content = str(step.get("content", ""))
        after = step.get("after")
        contents = path.read_text(encoding="utf-8")
        if after is None:
            updated = contents + content
        else:
            marker = str(after)
            if marker not in contents:
                raise ValueError(f"Append marker not found in {path}")
            updated = contents.replace(marker, marker + content, 1)
        path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _write_file(path: Path, step: dict[str, Any]) -> None:
        overwrite = bool(step.get("overwrite", False))
        if path.exists() and not overwrite:
            raise ValueError(f"Recipe write_file target already exists: {path}")
        content = step.get("content", "")
        if isinstance(content, (dict, list)):
            rendered = json.dumps(content, indent=2, sort_keys=True)
        else:
            rendered = str(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
