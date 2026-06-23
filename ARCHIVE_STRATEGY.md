# Archive Strategy

| Attribute | Value |
|-----------|-------|
| **Status** | current |
| **Last Updated** | 2026-06-23 |
| **Owner** | Repository maintainers |
| **Source** | `docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md` §7A (recorded decisions) |

One archival convention for the whole repository. Archived material is
classified by **what it is**, and each type has exactly one home. This lets any
agent or newcomer tell, at a glance, the difference between "deprecated output,"
"deprecated documentation," "transient scratch," and "irreplaceable reference
data that must never be deleted."

The enumerated inventory of everything currently archived lives in
**`docs/ARCHIVE_MANIFEST.md`**.

---

## The Three Archive Zones

| Zone | What goes here | Tracked? | Per-subdir marker | Index |
|------|----------------|----------|-------------------|-------|
| `docs/archive/` | Deprecated/superseded **documentation**. | git-tracked | metadata header on each file | `docs/archive/README.md` |
| `data_archives/` (repo root) | Deprecated/superseded **data outputs**. Renamed from `archived/`. | gitignored (on-disk only) | `README_STALE.md` in each subdir | `docs/ARCHIVE_MANIFEST.md` |
| `scratch/archive/` | **Transient** working / evidence artifacts. | gitignored | none required | none — no preservation guarantee |

### 1. `docs/archive/` — deprecated documentation

Git-tracked. Keep the existing index format in `docs/archive/README.md`:
`file | archived-date | reason | original-location`. Each archived doc retains
its original content below an archive header so it can be restored by copying it
back and removing the header.

### 2. `data_archives/` — deprecated data outputs

Gitignored, on-disk only (data files are excluded from git). **Renamed from the
former `archived/`.** Every subdirectory gets a `README_STALE.md` carrying:

- **Status** (e.g. quarantined / retired)
- **Reason** with the ADR reference that deprecated it
- **Last-valid-date**
- **Disposition** (retire / regenerate / keep + date)

This generalizes the existing `data/projections/*/README_STALE.md` pattern. The
`README_STALE.md` files that live in `data/projections/high_growth/` and
`data/projections/restricted_growth/` travel **with** those directories when
they move into `data_archives/` (hygiene Decision 1).

### 3. `scratch/archive/` — transient artifacts

Gitignored, auto-cleanable, **no preservation guarantee**. Use it for one-off
experiment evidence and superseded working files. Do not put anything here that
must survive.

---

## Lifecycle

A consistent three-step lifecycle moves something from "active" to "archived":

```
README_STALE marker  →  ARCHIVE_MANIFEST entry  →  move to the correct zone
   (mark in place)        (record reason/date)       (once disposition decided)
```

1. **Mark.** Drop a `README_STALE.md` (data) or set `Status: deprecated` in the
   metadata header (docs) where the material currently lives, stating the reason
   and ADR reference.
2. **Record.** Add an entry to `docs/ARCHIVE_MANIFEST.md` (item, reason/ADR,
   last-valid-date, disposition).
3. **Move.** Once the disposition is decided, move the material into the correct
   zone above. Carry the `README_STALE.md` with it.

For documentation, moving tracked files breaks cross-references — do a
grep-for-references pass first and land each move as its own commit with a
redirect note.

---

## Source Vaults Are Not Archives

Some directories under `data/raw/` hold **immutable source/reference data that
must never be deleted**. These are explicitly **outside** the archive taxonomy
so an agent never treats them as expendable:

- `data/raw/immigration/rpc_archives/`
- `data/raw/nd_sdc_2024_projections/source_files/backup/`

Convention for these vaults:

- Rename to a `*_vault` suffix to signal immutability (e.g. `rpc_archives` →
  `rpc_archives_vault`) — **verify no loader path globs the current name first.**
- Add a `README_SOURCES.md` recording: source URL, access date, and why the data
  is retained.
- **Never delete.** They are reference data, not archive output.

---

## Related Documents

- `docs/ARCHIVE_MANIFEST.md` — enumeration of everything currently archived.
- `docs/archive/README.md` — index for the documentation archive zone.
- `data_archives/*/README_STALE.md` — per-subdir stale markers for data outputs.
- `docs/naming-conventions.md` §4 — "status lives in the directory, not the
  filename" (the rule this strategy operationalizes).
- `docs/NAVIGATION.md` — repo map, including the archive zones and source vaults.
