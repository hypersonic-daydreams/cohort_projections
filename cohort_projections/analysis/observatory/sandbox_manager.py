"""Git-mirror and worktree sandbox management for Observatory search."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from cohort_projections.analysis.observatory.search_policy import SearchPolicy

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CheckoutStatusEntry:
    """One porcelain Git status entry from the live checkout."""

    status: str
    paths: tuple[str, ...]

    @property
    def line(self) -> str:
        """Return a stable, human-readable status line."""
        if len(self.paths) == 2 and self.status[0] in {"R", "C"}:
            return f"{self.status} {self.paths[1]} -> {self.paths[0]}"
        return f"{self.status} {' -> '.join(self.paths)}"


class SandboxManager:
    """Manage a bare mirror and disposable worktrees for search candidates."""

    def __init__(self, policy: SearchPolicy, *, source_repo: Path) -> None:
        self.policy = policy
        self.source_repo = source_repo.resolve()

    def live_checkout_signature(self) -> str:
        """Return material dirty status lines for the live checkout."""
        return self._format_status_entries(
            entry for entry in self._live_checkout_entries() if self._is_blocking_entry(entry)
        )

    def live_checkout_nonblocking_signature(self) -> str:
        """Return non-material dirty status lines for the live checkout."""
        return self._format_status_entries(
            entry
            for entry in self._live_checkout_entries()
            if not self._is_blocking_entry(entry)
        )

    def assert_live_checkout_clean(self) -> None:
        """Fail if material live-checkout paths are dirty before a search run."""
        blocking_signature = self.live_checkout_signature()
        if blocking_signature:
            raise RuntimeError(
                "Refusing deep search while material checkout paths are dirty.\n"
                "Commit or stash the blocking changes first, then try again.\n\n"
                f"Blocking changes:\n{blocking_signature}"
            )
        nonblocking_signature = self.live_checkout_nonblocking_signature()
        if nonblocking_signature:
            logger.warning(
                "Deep search starting with non-blocking dirty files:\n%s",
                nonblocking_signature,
            )

    def assert_live_checkout_unchanged(self, prior_signature: str) -> None:
        """Fail if material live-checkout paths changed during a search run."""
        current = self.live_checkout_signature()
        if current != prior_signature:
            raise RuntimeError(
                "Material checkout paths changed during deep search; stopping for safety."
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

        # Symlink gitignored data directories from the source repo into the
        # worktree so that benchmark scripts can find raw data files and
        # existing analysis outputs.  The worktree only has tracked files;
        # without these symlinks the benchmark suite fails immediately.
        self._provision_data_symlinks(worktree_path)

        return WorktreeContext(
            search_id=search_id,
            candidate_id=candidate_id,
            base_revision=base_revision,
            branch_name=branch_name,
            worktree_path=worktree_path,
        )

    def _provision_data_symlinks(self, worktree_path: Path) -> None:
        """Symlink gitignored data directories from the source repo.

        Git worktrees only contain tracked files.  Data directories like
        ``data/raw/`` are mostly gitignored and must be made available via
        symlinks so that benchmark scripts can load census and projection data.

        Some documentation and ``.gitkeep`` files inside those directories are
        tracked.  Preserve those tracked directories and overlay symlinks for
        missing untracked data entries; replacing the directory itself would
        make Git report tracked documentation files as deleted candidate code.
        """
        data_dirs = ["raw", "processed", "interim", "metadata", "backtesting"]
        for subdir in data_dirs:
            source = self.source_repo / "data" / subdir
            target = worktree_path / "data" / subdir
            if not source.is_dir():
                continue
            if target.is_symlink():
                continue  # Already symlinked
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
                continue
            if target.is_dir():
                self._overlay_data_symlinks(source, target)

    def _overlay_data_symlinks(self, source_dir: Path, target_dir: Path) -> None:
        """Symlink missing data entries while preserving tracked files."""
        for source_child in source_dir.iterdir():
            if source_child.name == ".git":
                continue
            target_child = target_dir / source_child.name
            if target_child.exists() or target_child.is_symlink():
                if (
                    source_child.is_dir()
                    and target_child.is_dir()
                    and not target_child.is_symlink()
                ):
                    self._overlay_data_symlinks(source_child, target_child)
                continue
            target_child.symlink_to(source_child.resolve())

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

    def _is_blocking_entry(self, entry: CheckoutStatusEntry) -> bool:
        return any(self.policy.is_dirty_blocking_path(Path(path)) for path in entry.paths)

    def _live_checkout_entries(self) -> list[CheckoutStatusEntry]:
        raw = self._git(
            self.source_repo,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ).stdout
        if not raw:
            return []
        records = [record for record in raw.split("\0") if record]
        entries: list[CheckoutStatusEntry] = []
        index = 0
        while index < len(records):
            token = records[index]
            index += 1
            if len(token) < 4:
                continue
            status = token[:2]
            path = token[3:].strip()
            paths = [path] if path else []
            if status[0] in {"R", "C"} and index < len(records):
                paths.append(records[index].strip())
                index += 1
            if not paths or all(self._is_runtime_artifact(status_path) for status_path in paths):
                continue
            entries.append(CheckoutStatusEntry(status=status, paths=tuple(paths)))
        return entries

    @staticmethod
    def _format_status_entries(entries: Iterable[CheckoutStatusEntry]) -> str:
        return "\n".join(entry.line for entry in entries).rstrip()

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
