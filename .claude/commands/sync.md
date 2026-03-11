---
description: Sync code to GitHub and data to Google Drive using dual-track versioning
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# Dual-Track Sync: Git + rclone

You are executing the `/sync` command for the cohort_projections project. This project uses **dual-track versioning**:
- **Code** (cohort_projections/, scripts/, tests/, docs/, configs) → Git → GitHub
- **Data** (data/, sdc_2024_replication/output/, large files) → rclone → Google Drive

## Reference Documentation

For full procedures, edge cases, and troubleshooting, see: `docs/GIT_RCLONE_SYNC.md`

## Your Task

Execute the sync protocol by following these steps:

### Step 1: Assess Current State

Run `git status` to see what files have changed. Categorize them:

**Code files** (Git):
- Python source: `cohort_projections/**/*.py`, `scripts/**/*.py`, `tests/**/*.py`
- Config files: `config/**/*.yaml`, `*.toml`, `*.cfg`
- Documentation: `docs/**/*.md`, `*.md` (README, AGENTS, etc.)
- Shell scripts: `scripts/**/*.sh`
- Test fixtures: `tests/**` (non-data)

**Data files** (rclone):
- Raw data: `data/raw/**` (csv, parquet, xlsx, json, pdf)
- Processed data: `data/processed/**`, `data/interim/**`
- Projections output: `data/projections/**`
- SDC replication: `sdc_2024_replication/output/**`, `sdc_2024_replication/data_immigration_policy/scripts/b05006_*`
- Generated figures: `**/*.png`, `**/*.pdf` (in output directories)
- Database backups: `data/metadata/**/*.sql`

### Step 2: Handle Code Changes (Git)

If there are code changes:

1. **Check for mis-categorized files** - If data files appear in git status that shouldn't be tracked:
   - Add appropriate patterns to `.gitignore` first
   - Commit the .gitignore change

2. **Stage code files**: `git add <files>`

3. **Create a commit** with conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `refactor:` for restructuring
   - `docs:` for documentation
   - `chore:` for maintenance
   - `test:` for test additions/changes

4. **Push to GitHub**: `git push`

### Step 3: Handle Data Changes (rclone)

If there are data files that need to sync to Google Drive:

1. **Verify they are properly gitignored** (not showing in `git status` after Step 2)

2. **Run the bisync wrapper script**:
   ```bash
   ./scripts/bisync.sh
   ```

   **IMPORTANT**: Always use the wrapper script, never raw rclone commands. The script:
   - Uses the correct filter file from `~/.config/rclone/cohort_projections-bisync-filter.txt`
   - Backs up the manifest database if it exists
   - Handles IPv4 binding for WSL2 compatibility

3. **If bisync reports conflicts**, assess whether local or remote is correct:
   - Local correct: `./scripts/bisync.sh --resync`
   - Remote correct: Download from Drive first, then resync

### Step 4: Report Results

Summarize what was synced:

**Git**:
- List files committed and pushed, or "No code changes"
- Include commit hash if a commit was made

**rclone**:
- Confirm data sync completed, or "No data changes"
- Note any warnings or conflicts encountered

## Important Rules

1. **Never commit data files to git** - They belong in rclone. Check `.gitignore` patterns.

2. **Never skip the gitignore check** - Data files appearing in git status need to be added to `.gitignore`

3. **Never run raw rclone commands** - Always use `./scripts/bisync.sh`

4. **Use conventional commits** - Follow the project's commit message format

5. **Include the commit trailer**:
   ```
   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
   ```

6. **Respect the filter file** - If new file types need syncing, update both:
   - `.gitignore` (to exclude from git)
   - `scripts/setup/bisync-filter.txt` then run `./scripts/setup_rclone_bisync.sh`

## File Type Decision Tree

```
Is it source code (*.py, *.sh, *.yaml config)?
├── YES → Git (commit and push)
└── NO → Is it generated/output data?
    ├── YES → Is it in .gitignore?
    │   ├── YES → rclone (./scripts/bisync.sh)
    │   └── NO → Add to .gitignore first, then rclone
    └── NO → Is it documentation (*.md)?
        ├── In docs/ or project root → Git
        └── Generated report in output/ → rclone
```

## Edge Cases

### New file types not covered
If you encounter a new file type:
1. Check `docs/GIT_RCLONE_SYNC.md` for guidance
2. If large/binary → rclone
3. If human-authored source → Git
4. If machine-generated output → rclone
5. Update both `.gitignore` and filter file as needed

### SDC 2024 Replication files
This subdirectory has special patterns:
- Code (`scripts/statistical_analysis/*.py`) → Git
- Outputs (`output/`, figures in `journal_article/figures/`) → Check .gitignore
- Census data (`b05006_*.csv`, `b05006_*.json`) → rclone only

### Pre-commit hook failures
If `git commit` fails due to pre-commit hooks:
1. Fix the issues reported (ruff, mypy, etc.)
2. Re-stage fixed files
3. Commit again
4. Do NOT use `--no-verify` to bypass hooks

Now execute the sync protocol.
