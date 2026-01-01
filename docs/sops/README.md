# Standard Operating Procedures (SOPs)

This directory contains standard operating procedures for the cohort_projections project. SOPs document repeatable workflows that transform inputs into outputs through defined steps.

## SOP Index

| ID | Title | Status | Last Updated |
|----|-------|--------|--------------|
| [SOP-001](./SOP-001-external-ai-analysis-integration.md) | External AI Analysis Integration | Active | 2026-01-01 |

---

## SOP vs. ADR

| Document Type | Purpose | When to Use |
|---------------|---------|-------------|
| **SOP** | How to execute a repeatable process | Workflow needs documentation |
| **ADR** | Why a decision was made | Architectural/design decision |

**Relationship**: SOPs may reference ADRs for context on *why* certain steps are required. ADRs may reference SOPs for *how* to implement a decision.

---

## Creating a New SOP

1. Copy [TEMPLATE.md](./TEMPLATE.md) to `SOP-NNN-short-title.md`
2. Assign the next sequential ID
3. Fill in all sections
4. Add to the index table above
5. Create any supporting templates in `templates/`

---

## Directory Structure

```
docs/sops/
├── README.md                    # This index
├── TEMPLATE.md                  # SOP template
├── SOP-001-*.md                 # Individual SOPs
└── templates/                   # Reusable templates referenced by SOPs
    ├── adr-report-structure.md  # ADR report directory template
    ├── planning-synthesis.md    # Planning document template
    └── module-package.md        # Python module package template
```

---

## Related Documentation

- [AGENTS.md](../../AGENTS.md) - AI agent governance and autonomy rules
- [docs/adr/](../adr/) - Architecture Decision Records
- [DEVELOPMENT_TRACKER.md](../../DEVELOPMENT_TRACKER.md) - Current project status

---

*Last Updated: 2026-01-01*
