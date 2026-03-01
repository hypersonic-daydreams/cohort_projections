# 2026-03-01 PP3 IMP-21 Implementation Results: Publication Methodology Consistency

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-01 |
| **Scope** | Final publication-facing methodology consistency pass for PP-003 |
| **Status** | Completed |
| **Primary Files** | `docs/methodology.md`, `scripts/exports/_methodology.py`, `tests/test_output/test_place_workbook.py` |

## Summary

IMP-21 is complete. Publication-facing place methodology wording is now aligned across the narrative methodology document and export-side methodology text used by place workbook outputs.

## Changes Applied

1. Updated `docs/methodology.md` Section 7.5 to explicitly reflect:
   - ADR-033 accepted/implemented state (`Accepted`, implemented `2026-03-01` via IMP-20),
   - winner variant phrasing alignment (`B-II`, weighted least squares `wls` with cap-and-redistribute constraints),
   - IMP-19 validation/sign-off status (`PASS`, human sign-off dated `2026-03-01`).
2. Updated `scripts/exports/_methodology.py` `PLACE_METHODOLOGY_LINE` to align with the same publication-facing wording:
   - ADR-033 accepted/implemented status,
   - winner variant language,
   - IMP-19 validation + sign-off statement,
   - county-constrained accounting and tier framing retained.
3. Updated place workbook contract test (`tests/test_output/test_place_workbook.py`) to assert the methodology worksheet contains the IMP-19 validation statement, ensuring this publication-facing dependency remains enforced.

## Consistency Checklist (Final State)

| Topic | Final Source State | Result |
|------|------|------|
| Method family | `share-of-county trending` phrasing aligned in docs + export constants | PASS |
| Winner variant naming | `B-II` with weighted least squares (`wls`) + cap-and-redistribute constraints in both surfaces | PASS |
| ADR status reflection | ADR-033 presented as accepted/implemented (post-IMP-20) | PASS |
| Validation framing | IMP-19 pass + human sign-off date communicated in publication-facing text | PASS |
| County constraint statement | Place + balance-of-county accounting retained and consistent | PASS |
| Place workbook dependency coverage | Test asserts methodology worksheet includes IMP-19 validation wording | PASS |

## Evidence

- `docs/methodology.md`
- `scripts/exports/_methodology.py`
- `tests/test_output/test_place_workbook.py`
- `docs/reviews/2026-03-01-pp3-imp20-results.md`
- `docs/reviews/pp3-end-to-end-validation.md`

## Verification

- `source .venv/bin/activate && ruff check scripts/exports/_methodology.py tests/test_output/test_place_workbook.py` -> `All checks passed!`
- `source .venv/bin/activate && mypy scripts/exports/_methodology.py tests/test_output/test_place_workbook.py` -> `Success: no issues found in 2 source files`
- `source .venv/bin/activate && pytest tests/test_output/test_place_workbook.py` -> `1 passed`
