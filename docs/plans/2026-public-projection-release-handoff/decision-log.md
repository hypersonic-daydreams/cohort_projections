# PUB-2026 Decision Log

Internal repo record. Do not include this file in the marketing handoff packet.
It exists to preserve scope and packaging decisions for future agents and data
production work.

| Date | Decision | Rationale | Consequence |
|------|----------|-----------|-------------|
| 2026-05-27 | Create a dedicated handoff folder under `docs/plans/2026-public-projection-release-handoff/`. | Public release decisions need a git-tracked home separate from model-development work. | Future agents should use this folder before recreating marketing-handoff context. |
| 2026-05-27 | Build a standalone 12-16 page public PDF. | The 2024 SDC PDF was a compact public-facing report, and the 2026 release should remain readable without requiring the downloadable data files. | The PDF includes narrative, core charts, core tables, caveats, methodology, and a county appendix. |
| 2026-05-27 | Use baseline-led framing, with restricted growth and high growth as scenario context. | Stakeholders need a primary thread for readability, but ADR-042 requires baseline context and scenario discipline. | Baseline headlines must be paired with restricted-growth context; high growth is secondary planning context. |
| 2026-05-27 | Exclude city/place projections from public PDF and public downloads. | City/place projections remain an experimental, internal-facing layer for this release. | Public geography coverage is state, 8 regions, and 53 counties only. |
| 2026-05-27 | Provide one consolidated public Excel workbook and one consolidated public CSV. | A single public pair is easier for general users than the technical per-geography package set. | Technical internal exports can remain available separately, but they are not the public download shape. |
| 2026-05-27 | Do not use the existing March 1 ZIP packages as public downloads. | Inspection showed the county package can mix current `2025_2055` files with stale `2025_2045` files. | Final downloads must come from a clean staged export with stale-horizon checks. |
| 2026-05-27 | Treat current March 2026 exports as draft structure only. | CF-001 remains an active methodology gate. | Final public numbers are targeted for the week of 2026-06-01 after CF-001 disposition and any approved rerun. |
| 2026-05-27 | Marketing owns the final designed PDF; this repo owns content intent, source data specs, and QA record. | The repo should preserve analytical decisions while allowing design production outside the codebase. | Handoff artifacts should specify what to communicate and what data to use, not final visual design. |
| 2026-05-27 | Govern public-facing language by ADR-042. | Scenario outputs should not be read as a single guaranteed path. | Use `projection`, `scenario`, and `trend continuation`; avoid ADR-042 disallowed point-estimate wording in public text. |
| 2026-05-27 | Create a provisional number snapshot and draft public PDF copy for marketing layout. | Marketing can begin design work before final numbers are locked, and draft copy is stronger when anchored to current outputs. | Draft copy and numbers must be refreshed after CF-001 disposition and final production rerun if needed. |
