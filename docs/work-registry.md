# Work-Item Registry

| Attribute | Value |
|-----------|-------|
| **Status** | current |
| **Last Updated** | 2026-06-23 |
| **Owner** | Repository maintainers |

The repository uses five identifier schemes for tracked work. This registry is
the single place to see all of them: what each is for, where its canonical
records live, and how to get an up-to-date count. It does **not** track status —
for active vs. complete work, see `DEVELOPMENT_TRACKER.md`.

Always cite identifiers with a hyphen and no space (`ADR-061`, not `ADR 061`) —
see `docs/naming-conventions.md` §5.

---

## Schemes

| Scheme | Purpose | Canonical location | How to find the count |
|--------|---------|--------------------|-----------------------|
| **ADR** | Architecture Decision Records — *why* the system is built as it is (context, decision, consequences, alternatives). Numbered `NNN`, kebab-case file, with child suffixes like `020a`. | `docs/governance/adrs/` (one `NNN-kebab.md` per ADR; see `docs/governance/adrs/README.md`). | `ls docs/governance/adrs/ | grep -E '^[0-9]{3}' | wc -l` — currently ~70 files; the next free number is tracked in `MEMORY`/the ADR README. |
| **SOP** | Standard Operating Procedures — repeatable process rules (data-processing docs, prose voice, benchmarking, AI integration). | `docs/governance/sops/` (`SOP-NNN-…md`; index in `docs/governance/sops/README.md`). | `ls docs/governance/sops/SOP-*.md | wc -l` — currently 5 (SOP-001…SOP-005). |
| **PP** | Projection Packages — major development workstreams (publication QA, place projections, test coverage, evaluation framework, Observatory). | Tracked in `DEVELOPMENT_TRACKER.md` (the "Projection Development Backlog" / Current Snapshot tables); per-package detail rows link out to plans, ADRs, and reviews. | `grep -oE 'PP-[0-9]+' DEVELOPMENT_TRACKER.md | sort -u | wc -l` — currently 9 (PP-001…PP-009), all complete. |
| **CF** | College Fix work items — the college-county model revision (ADR-061 family). | `DEVELOPMENT_TRACKER.md` (active work items) plus `docs/governance/adrs/061-*` and related ADRs/reviews. | `grep -oE 'CF-[0-9]+' DEVELOPMENT_TRACKER.md | sort -u | wc -l` — currently CF-001. |
| **PUB** | Public-release work — the 2026 public projection release handoff to marketing. | `DEVELOPMENT_TRACKER.md` (active work items) + the plan subtree `docs/plans/2026-public-projection-release-handoff/`. | `grep -roE 'PUB-[0-9]+' DEVELOPMENT_TRACKER.md docs/plans/ | grep -oE 'PUB-[0-9]+' | sort -u` — currently PUB-2026. |

---

## Notes

- **ADR and SOP** are the two schemes with their own dedicated directory and a
  filename convention (`NNN-…`). Count them by listing files.
- **PP, CF, and PUB** are *backlog* identifiers, not per-file schemes — they are
  defined and tracked in `DEVELOPMENT_TRACKER.md`, with supporting artifacts
  (plans, ADRs, reviews) linked from their rows. Count them by grepping the
  tracker.
- When opening a new work item, register it in its canonical location **before**
  ending the session, per the source-of-truth rule in `DEVELOPMENT_TRACKER.md`.
- For the next available ADR number and recent disposition notes, the agent
  `MEMORY` file and the ADR README carry the running detail.
