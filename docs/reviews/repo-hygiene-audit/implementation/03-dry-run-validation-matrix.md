# Repo Hygiene Audit Dry-Run Validation Matrix

## Purpose

Define deterministic preflight and dry-run gates for each remediation batch before
implementation. This matrix is execution-ready but not executed as part of this
planning scaffold.

## Run Context

- Run from repository root.
- Activate venv first:

```bash
source .venv/bin/activate
```

## Profile Index

| Profile ID | Used By Batches | Goal |
|---|---|---|
| `DRY-BASELINE` | `B00` | Pin baseline metrics before edits |
| `DRY-DOCS` | `B01` | Ensure docs-only change set stays docs-only |
| `DRY-PIPELINE` | `B02` | Validate pipeline command wiring without full production write path |
| `DRY-CONFIG` | `B03` | Validate config/path behavior with explicit checks |
| `DRY-REPO-CLEANUP` | `B05` | Preview move/archive/delete effects safely |
| `DRY-LINT-TYPE` | `B03`, `B04`, `B06` | Static quality checks |
| `DRY-TESTS` | `B02`, `B03`, `B04`, `B06` | Behavioral regression checks |
| `DRY-CHECK-REPLAY` | `B00-B06` | Re-run claim checks and compare verdict drift |

## Profiles

### `DRY-BASELINE`

Commands:

```bash
python scripts/reviews/run_claim_checks.py list
python scripts/reviews/run_claim_checks.py progress
python - <<'PY'
import yaml
from pathlib import Path
reg = yaml.safe_load(Path("docs/reviews/repo-hygiene-audit/verification/claims_registry.yaml").read_text())
print("claims", len(reg["claims"]))
print("adjudicated", sum(1 for c in reg["claims"] if c["status"] == "adjudicated"))
PY
```

Pass signals:
- Claim count matches expected planning baseline.
- All claims remain adjudicated before starting a batch.

### `DRY-DOCS`

Commands:

```bash
git diff --name-only
python scripts/reviews/run_claim_checks.py run --claim-id RHA-009 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-019 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-020 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-021 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-022 --dry-run
```

Pass signals:
- Changed paths are documentation/planning paths only.
- Dry-run claim selection resolves correctly for all targeted doc claims.

### `DRY-PIPELINE`

Commands:

```bash
bash scripts/pipeline/run_complete_pipeline.sh --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-001 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-002 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-003 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-004 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-027 --dry-run
```

Pass signals:
- Runner dry-run executes without immediate shell errors.
- Claim dry-runs resolve target checks after file renames/refactors.

### `DRY-CONFIG`

Commands:

```bash
python scripts/reviews/run_claim_checks.py run --claim-id RHA-005 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-006 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-007 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-016 --dry-run
```

Pass signals:
- Config/path claims still have deterministic check coverage.
- No missing check commands after refactors.

### `DRY-REPO-CLEANUP`

Commands:

```bash
git status --short
git ls-files | rg "RCLONE_TEST|journal_article_pdfs|2025_popest_data" || true
python scripts/reviews/run_claim_checks.py run --claim-id RHA-011 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-012 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-015 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-017 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-018 --dry-run
python scripts/reviews/run_claim_checks.py run --claim-id RHA-026 --dry-run
```

Pass signals:
- Cleanup candidates are visible and explicit before changes.
- Check selectors remain stable for footprint claims.

### `DRY-LINT-TYPE`

Commands:

```bash
ruff check .
mypy cohort_projections
```

Pass signals:
- Lint and type checks pass for targeted remediation branch.

### `DRY-TESTS`

Commands:

```bash
pytest tests/ -q
```

Pass signals:
- Test status is no worse than baseline.
- Any known skips remain intentional and documented.

### `DRY-CHECK-REPLAY`

Commands:

```bash
python scripts/reviews/run_claim_checks.py run --status adjudicated
python scripts/reviews/run_claim_checks.py progress
```

Pass signals:
- Evidence artifacts regenerate successfully.
- Claim verdict drift is expected and explicitly adjudicated.

## Escalation Rule

If a dry-run profile fails:
- Stop batch execution.
- Capture failure output in the dashboard.
- Re-scope batch or split into smaller changes before retrying.
