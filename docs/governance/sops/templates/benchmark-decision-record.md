# Benchmark Decision Record

## Metadata

| Field | Value |
|-------|-------|
| Decision ID | YYYY-MM-DD-<challenger>-vs-<champion> |
| Date | YYYY-MM-DD |
| Scope | county / place / other |
| Champion Method | `<method_id>` |
| Champion Config | `<config_id>` |
| Challenger Method | `<method_id>` |
| Challenger Config | `<config_id>` |
| Benchmark Run ID | `<run_id>` |
| Reviewer | |
| Status | Draft / Approved / Rejected |

---

## 1. Context

Describe the problem being addressed, the hypothesis behind the challenger, and the ADR or review documents that motivated the comparison.

---

## 2. Runs and Evidence

- Manifest: `data/analysis/benchmark_history/<run_id>/manifest.json`
- Scorecard: `data/analysis/benchmark_history/<run_id>/summary_scorecard.csv`
- Comparison artifact: `data/analysis/benchmark_history/<run_id>/comparison_to_champion.json`
- Related ADRs:
- Related reviews:

---

## 3. Primary Metrics

| Metric | Champion | Challenger | Delta | Notes |
|--------|----------|------------|-------|-------|
| Recent-origin state APE (short) | | | | |
| Recent-origin state APE (medium) | | | | |
| Recent-origin signed bias | | | | |
| County MAPE overall | | | | |
| County MAPE urban/college | | | | |
| County MAPE rural | | | | |
| County MAPE Bakken | | | | |

---

## 4. Sentinel Counties

| County | Champion | Challenger | Delta | Interpretation |
|--------|----------|------------|-------|----------------|
| Cass | | | | |
| Grand Forks | | | | |
| Ward | | | | |
| Burleigh | | | | |
| Williams | | | | |
| McKenzie | | | | |

---

## 5. Hard Constraints and Risks

- Negative population violations:
- Scenario ordering violations:
- Aggregation violations:
- Sensitivity instability flags:
- Other regressions:

---

## 6. Decision

Select one:

- `promote`
- `accept_but_do_not_promote`
- `retain_champion`
- `reject`
- `needs_more_segmentation`

Decision rationale:

---

## 7. Promotion / Follow-Up Actions

- Alias change required:
- Champion after decision:
- Method lifecycle updates:
- Additional analyses required:
- Tracker updates required:

---

## 8. Reversion Plan

If the promoted method later needs to be reverted, identify the prior immutable champion method/config pair and the alias that should be restored.
