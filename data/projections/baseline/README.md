# `baseline/` — CURRENT LOCKED PRODUCTION RUN

| Attribute | Value |
|-----------|-------|
| **Status** | **locked (current public run)** |
| **Last Updated** | 2026-06-23 |
| **Run alias** | `m2026r1` (method/config alias `county_champion`) |
| **Locked config** | `cfg-20260611-production-lock` — config-sha per `final-run-metadata.md` (`RUN_CONFIG_SHA16`) |
| **Decision record** | ADR-068 (2026-06-15, amended 2026-06-16) |

---

## This is the public release baseline. Do not confuse it with archived scenarios.

These are the **CURRENT LOCKED production outputs** — the **PUB-2026 public
release baseline** (run `m2026r1`, locked config
`cfg-20260611-production-lock`, corrected under ADR-068). This is the **only**
active, public, production projection.

- **53 counties + a bottom-up state total.** State = county sum by construction
  (ADR-054).
- **Base population:** Census PEP Vintage 2025 (799,358 at 2025; ADR-066).
- **Config-sha:** the authoritative `RUN_CONFIG_SHA16` is recorded in the public
  workbook README and in `final-run-metadata.md`. The ADR-068 amendment
  re-stamped it post-correction; the config delta versus the prior locked sha
  `cca42fb42be76680` is **comment-only** (functional values identical).

**Do NOT confuse this with the archived stale scenarios.** `high_growth/` and
`restricted_growth/` were deprecated 2026-02-26 runs (pre-ADR-065/066) and have
been moved to `data_archives/projections/`. The `sensitivity_*` siblings are
internal what-if runs. None of those are public, and none are comparable to this
baseline.

## Provenance

For the full run-identity record, the locked state trajectory, the corrected
ADR-068 figures, and the change attribution:

- `docs/plans/2026-public-projection-release-handoff/` — the PUB-2026 handoff,
  including `final-run-metadata.md` (run identity, config-sha, trajectory) and
  the release QA checklists.
- `DEVELOPMENT_TRACKER.md` — canonical current-state record.

> **Note:** `data/` is gitignored; this marker is on-disk per machine and git
> tracking is handled separately.
