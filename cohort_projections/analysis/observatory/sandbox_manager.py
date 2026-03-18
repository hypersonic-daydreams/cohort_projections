"""Git-mirror and worktree sandbox management for Observatory search."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cohort_projections.analysis.observatory.search_policy import SearchPolicy


def _clean_git_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with ``GIT_*`` variables removed.

    Pre-commit hooks and other callers may set ``GIT_DIR``,
    ``GIT_WORK_TREE``, ``GIT_INDEX_FILE``, etc.  These leak into child
    processes and cause git operations on temporary repos or bare mirrors
    to resolve paths incorrectly.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


@dataclass(frozen=True)
class WorktreeContext:
    """Metadata for one disposable experiment worktree."""

    search_id: str
    candidate_id: str
    base_revision: str
    branch_name: str
    worktree_path: Path


class SandboxManager:
    """Manage a bare mirror and disposable worktrees for search candidates."""

    def __init__(self, policy: SearchPolicy, *, source_repo: Path) -> None:
        self.policy = policy
        self.source_repo = source_repo.resolve()

    def live_checkout_signature(self) -> str:
        """Return the porcelain status for the live checkout."""
        raw = self._git(self.source_repo, "status", "--porcelain").stdout.splitlines()
        filtered = [
            line for line in raw if not self._is_runtime_artifact(self._extract_status_path(line))
        ]
        return "\n".join(filtered).strip()

    def assert_live_checkout_clean(self) -> None:
        """Fail if the live checkout is dirty before an autonomous run."""
        if self.live_checkout_signature():
            raise RuntimeError("Refusing autonomous search while the live checkout is dirty.")

    def assert_live_checkout_unchanged(self, prior_signature: str) -> None:
        """Fail if the live checkout changed during a search run."""
        current = self.live_checkout_signature()
        if current != prior_signature:
            raise RuntimeError(
                "Live checkout changed during autonomous search; stopping for safety."
            )

    def ensure_mirror(self) -> Path:
        """Create or refresh the bare mirror used for search worktrees."""
        mirror = self.policy.mirror_repo
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if not mirror.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--bare",
                    "--no-hardlinks",
                    str(self.source_repo),
                    str(mirror),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=_clean_git_env(),
            )
            subprocess.run(
                [
                    "git",
                    f"--git-dir={mirror}",
                    "remote",
                    "set-url",
                    "--push",
                    "origin",
                    "DISABLED",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=_clean_git_env(),
            )
        else:
            subprocess.run(
                ["git", f"--git-dir={mirror}", "fetch", "origin"],
                check=True,
                capture_output=True,
                text=True,
                env=_clean_git_env(),
            )
        return mirror

    def create_worktree(
        self,
        *,
        search_id: str,
        candidate_id: str,
        base_revision: str,
    ) -> WorktreeContext:
        """Create a fresh worktree for one candidate."""
        mirror = self.ensure_mirror()
        slug = self._slugify(candidate_id)
        branch_name = f"experiment/{self._slugify(search_id)}/{slug}"
        worktree_path = self.policy.worktree_root / self._slugify(search_id) / slug
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        self._remove_branch_if_present(mirror, branch_name)
        if worktree_path.exists():
            shutil.rmtree(worktree_path)

        subprocess.run(
            [
                "git",
                f"--git-dir={mirror}",
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree_path),
                base_revision,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=_clean_git_env(),
        )

        return WorktreeContext(
            search_id=search_id,
            candidate_id=candidate_id,
            base_revision=base_revision,
            branch_name=branch_name,
            worktree_path=worktree_path,
        )

    def remove_worktree(self, context: WorktreeContext) -> None:
        """Remove a disposable worktree and its local branch."""
        mirror = self.policy.mirror_repo
        if context.worktree_path.exists():
            subprocess.run(
                [
                    "git",
                    f"--git-dir={mirror}",
                    "worktree",
                    "remove",
                    "--force",
                    str(context.worktree_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=_clean_git_env(),
            )
        self._remove_branch_if_present(mirror, context.branch_name)
        if context.worktree_path.exists():
            shutil.rmtree(context.worktree_path)

    def capture_diff(
        self,
        *,
        worktree_path: Path,
        base_revision: str,
    ) -> tuple[list[str], str]:
        """Return changed files and a patch relative to the base revision."""
        changed = self._git(
            worktree_path,
            "diff",
            "--name-only",
            base_revision,
            "--",
        ).stdout.splitlines()
        patch = self._git(
            worktree_path,
            "diff",
            "--binary",
            base_revision,
            "--",
        ).stdout
        return [line.strip() for line in changed if line.strip()], patch

    def compile_changed_python(self, worktree_path: Path, changed_files: list[str]) -> None:
        """Run ``py_compile`` against changed Python files."""
        py_files = [
            str(worktree_path / rel_path) for rel_path in changed_files if rel_path.endswith(".py")
        ]
        if not py_files:
            return
        subprocess.run(
            [sys.executable, "-m", "py_compile", *py_files],
            check=True,
            capture_output=True,
            text=True,
            cwd=worktree_path,
        )

    def _remove_branch_if_present(self, mirror: Path, branch_name: str) -> None:
        existing = subprocess.run(
            ["git", f"--git-dir={mirror}", "show-ref", "--verify", f"refs/heads/{branch_name}"],
            check=False,
            capture_output=True,
            text=True,
            env=_clean_git_env(),
        )
        if existing.returncode == 0:
            subprocess.run(
                ["git", f"--git-dir={mirror}", "branch", "-D", branch_name],
                check=True,
                capture_output=True,
                text=True,
                env=_clean_git_env(),
            )

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in value.lower()
        ).strip("-_")

    def _is_runtime_artifact(self, path: str) -> bool:
        if not path:
            return False
        candidate = self.source_repo / path
        for root in (self.policy.runtime_root, self.policy.session_root):
            if candidate.resolve() == root or candidate.resolve().is_relative_to(root):
                return True
        return False

    @staticmethod
    def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=_clean_git_env(),
        )

    @staticmethod
    def _extract_status_path(line: str) -> str:
        payload = line[3:].strip() if len(line) >= 4 else line.strip()
        if " -> " in payload:
            return payload.split(" -> ", 1)[1].strip()
        return payload
