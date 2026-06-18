# SOP-005: Public-Facing Prose Voice and Style

## Document Information

| Field | Value |
|-------|-------|
| SOP ID | 005 |
| Status | Active |
| Created | 2026-06-18 |
| Last Updated | 2026-06-18 |
| Owner | Project Lead |

---

## 1. Purpose

This SOP defines the **house writing voice for public-facing narrative documents** produced by the
cohort projections project — the public PDF report, plain-language explainers, FAQs, narrative
appendices, and any prose intended for residents, planners, elected officials, or journalists.

The goal is a single, recognizable register: the **reserved, well-made voice of a university-press
trade book**. It writes for a slightly broader audience than a technical methodology document, and
it earns and holds the reader's interest through clear sentences and forward motion — not through
marketing energy, and not through dry technical recitation. The two failure modes this SOP exists to
prevent are (a) prose that reads like promotional copy, and (b) prose that reads like a methods
section bolted onto a public document.

**Origin**: Derived from the PUB-2026 public-release work (2026-06-18), where the
`how-these-projections-work.md` methods companion established the approved register. That document is
the reference exemplar for this SOP.

---

## 2. Scope

### In Scope
- The public PDF report copy (`docs/plans/.../draft-public-pdf-copy.md`)
- Plain-language explainers and methods companions (e.g. `how-these-projections-work.md`)
- Public FAQs, narrative appendices, and download-page / landing-page prose
- Any new public-facing narrative document authored or revised by an agent

### Out of Scope
- `docs/methodology.md` and the SDC comparison — these stay formal and technical
- ADRs, SOPs, and other governance documents — these use their own structured formats
- Code, docstrings, comments, commit messages, and internal tooling output
- Tables, data dictionaries, and figure captions (informational, not narrative)

---

## 3. Prerequisites

### Required Knowledge
- Familiarity with project structure (see [AGENTS.md](../../../AGENTS.md))
- The locked production numbers the prose must reconcile to (see the PUB-2026 handoff
  `final-run-metadata.md` and the locked public CSV)
- [ADR-042](../adrs/) terminology rules ("projection," never "forecast/prediction/expected")

### Reference Exemplar
The following document exemplifies the approved voice and should be read before writing or revising
any public-facing prose:
- `docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`

---

## 4. Procedure

When writing or revising a public-facing narrative document, apply the following style standard.

### Phase 1: Set the register

**Objective**: Establish the trade-book voice before drafting.

**Standard**:

1. **Aim for the university-press trade book**, not the press release and not the journal article.
   Reserved and tame, but alive — the reader should want to keep going.
2. **Restraint is mandatory.** No exclamation points. No salesmanship, hype, or superlatives. No
   second-person hard sell. Confidence comes from clear prose, not from intensifiers.
3. **Assume an intelligent general reader** with no demography background. Explain terms of art in
   ordinary language on first use; never assume the reader already knows what a term like
   "cohort-component" or "residual migration" means.

### Phase 2: Shape the narrative

**Objective**: Build a document that carries the reader forward.

**Standard**:

1. **Section titles carry the argument**, they do not merely label it. Prefer
   "Why a trend line is not enough" over "Methodology"; "What the statewide number conceals" over
   "County Detail."
2. **Open by framing the idea**, often with a gentle, restrained hook that gives the reader a reason
   to care (e.g. the quiet paradox that most of 2055's residents are already here, so the task is
   following the present forward rather than inventing a future).
3. **Make the case with a worked contrast or a concrete image**, not bare assertion — two same-size
   counties with different age structures; age as a "ladder"; fertility as "the quiet engine of the
   long run." One well-chosen image beats a paragraph of abstraction.
4. **Vary sentence rhythm.** Mix longer explanatory sentences with short declarative ones. Let an
   occasional short sentence land a point.
5. **Close sections and the document with a plain, earned summary** rather than a flourish.

### Phase 3: Protect accuracy and required framing

**Objective**: Voice never costs correctness.

**Standard**:

1. **Voice changes connective prose only — never the numbers.** Every figure, date, and county
   callout must remain exact and reconcile to the locked run. Re-verify against
   `final-run-metadata.md` / the locked public CSV on every projection change.
2. **Honor [ADR-042](../adrs/):** call it a "projection," never a "forecast," "prediction,"
   "expected outcome," or "most likely" path. Keep the required baseline caveats intact even when
   smoothing the surrounding prose.
3. **Push technical detail into set-aside boxes.** Use blockquoted **"Technical note"** boxes for
   formulas, data sources, and method specifics so the main narrative stays readable to a general
   audience. The narrative should be complete and intelligible even if every box is skipped.

**Checkpoint**: A reader with no demography background can follow the main narrative end to end; a
technical reader can get the mechanics from the boxes; every number matches the locked run; and the
prose reads like a thoughtful book, not a brochure or a methods section.

---

## 5. Quality Gates

| Gate | Criteria | Responsible |
|------|----------|-------------|
| Voice | Reserved trade-book register; no hype, no dry recitation; titles carry the argument | Author |
| Accessibility | Main narrative intelligible with every Technical-note box removed | Author |
| Accuracy | All figures/dates/county callouts reconcile to the locked run | Author + release-QA |
| Terminology | ADR-042 language honored; required caveats intact | Author + release-QA |

---

## 6. Related Documentation

- [SOP-002](./SOP-002-data-processing-documentation.md) — documentation standard for data processing scripts (technical register)
- [ADR-042](../adrs/) — projection terminology rules
- `docs/methodology.md` — the full technical methodology (formal register; the counterpoint to this SOP)
- `docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md` — reference exemplar

---

## 7. Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-06-18 | 1.0 | Project Lead | Initial version; codifies the PUB-2026 public-prose voice |

---

*Template Version: 1.0*
