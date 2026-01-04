# Git and Rclone Bisync: Dual-Track Versioning

This document defines the procedures for managing code (via Git/GitHub) and data (via rclone/Google Drive) in the `cohort_projections` project.

---

## Philosophy: Two Sources of Truth

This project maintains **two separate sources of truth** for different types of artifacts:

| Artifact Type | Source of Truth | Sync Tool | Rationale |
|---------------|-----------------|-----------|-----------|
| **Code** | GitHub | Git | Version control, collaboration, CI/CD |
| **Data** | Google Drive | rclone bisync | Large files, binary outputs, research results |

### The Golden Rule

> **If it's source code, it goes in Git. If it's generated output or data, it goes in rclone.**

---

## Configuration Files

Two configuration files control what goes where:

### 1. `.gitignore` - What Git Ignores

Located at project root. Files matching these patterns are NOT tracked by Git.

```gitignore
# Currently ignored (synced via rclone instead):
*.csv, *.parquet, *.xlsx, *.json (except config_*.json)
*.pdf, *.db, *.sqlite
data/
output/
logs/
__pycache__/
.venv/
```

### 2. `cohort_projections-bisync-filter.txt` - What rclone Syncs

**Location:** `~/.config/rclone/cohort_projections-bisync-filter.txt`
**Managed by:** `scripts/setup_rclone_bisync.sh` (installs from `scripts/setup/bisync-filter.txt`)

```text
# Excluded from rclone (tracked by Git instead):
- .git/**
- .github/**
- core/**
- scripts/**
- tests/**
- docs/**
- config/**
- *.py
- *.sh
- *.md
- .gitignore

# Included in rclone (data files):
+ data/**
+ output/**
+ * (files in root not matched above)
```

### The Complementary Relationship

These two files should be **complementary inverses**:

| File Type | In `.gitignore`? | In rclone (included)? |
|-----------|------------------|----------------------|
| Python source (`*.py`) | NO | NO (excluded) |
| SQL Database (`demography_db`) | YES | YES |
| Analysis outputs (`data/output/`) | YES | YES |
| Generated data (`*.csv`, `*.parquet`) | YES | YES |
| Config files (`*.yaml`) | NO | NO |

---

## Standard Procedures

### Procedure 1: After Creating Code Changes

When you've written or modified source code:

```bash
# 1. Check what's changed
git status

# 2. Stage and commit code changes
git add <files>
git commit -m "feat: description of change"

# 3. Push to GitHub
git push

# NO rclone needed - code doesn't sync to Drive
```

### Procedure 2: After Generating Data Outputs

When analysis or projection pipelines generate output files:

```bash
# 1. Verify files are gitignored (should show nothing or just .gitignore changes)
git status

# 2. If data files appear in git status, they need to be added to .gitignore
# Edit .gitignore, then:
git add .gitignore
git commit -m "chore: Add <pattern> to gitignore"
git push

# 3. Sync data to Google Drive
./scripts/bisync.sh
```

### Procedure 3: After Both Code and Data Changes

Common after running analysis pipelines that create both code and outputs:

```bash
# 1. Handle code first
git status
git add <code files only>
git commit -m "feat: Add analysis module"
git push

# 2. Verify data files are gitignored and NOT staged
git status  # Should be clean

# 3. Sync data to Google Drive
./scripts/bisync.sh
```

### Procedure 4: Initial Sync or Recovery (--resync)

Use `--resync` when setting up on a new machine or resolving conflicts.

```bash
# CAUTION: --resync forces a full check and can overwrite files.
# It makes LOCAL the source of truth if conflicts exist.
./scripts/bisync.sh --resync
```

---

## Edge Cases and Nuances

### Edge Case 1: Sensitive Files
**Files containing secrets (`.env`, API keys)**:
- MUST be in `.gitignore` (never commit secrets).
- Should NOT be in rclone either (don't sync to cloud).
- Add explicit exclusion to `scripts/setup/bisync-filter.txt` and reinstall filters.

### Edge Case 2: rclone Shows Conflicts
**Symptom**: `bisync.sh` reports path 1 and path 2 differ.
**Resolution**:
1. Determine which version is correct.
2. If **Local** is correct: `./scripts/bisync.sh --resync`
3. If **Remote** (Drive) is correct: Delete local file, run `./scripts/bisync.sh --resync` (or manually download).

---

## Quick Reference Commands

```bash
# Git workflow
git status                          # See what's changed
git add <files>                     # Stage changes
git commit -m "type: message"       # Commit
git push                            # Push to GitHub

# rclone workflow (using wrapper script)
./scripts/bisync.sh                 # Normal sync
./scripts/bisync.sh --resync        # Force resync
./scripts/bisync.sh --dry-run       # Preview changes

# List what's on Drive
rclone ls wsdrive:cohort_projections
```
