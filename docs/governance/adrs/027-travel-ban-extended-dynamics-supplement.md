# ADR-027: Supplemental Travel Ban Regime-Dynamics Extension (FY2002--FY2024)

## Status
Accepted

## Date
2026-01-06

## Context

The SDC 2024 replication analysis uses a nationality-level difference-in-differences (DiD) and
event-study design to characterize the effect of the 2017 Travel Ban on refugee arrivals
(`sdc_2024_replication/scripts/statistical_analysis/module_7_causal_inference.py`).

Historically, this causal design has been restricted to FY2002--FY2019 to avoid confounding from
the COVID-19 pandemic and related policy shifts starting in 2020. The v0.8.6 data extension work
adds FY2021--FY2024 refugee arrivals with improved (near-complete) state coverage and
month-level reports that enable exact fiscal-month to PEP-year alignment in other parts of the
pipeline.

This creates a natural temptation to "extend the Travel Ban post-period" to include FY2020--FY2024
to gain more power and learn about persistence. However, post-2020 years overlap multiple major
structural changes that directly affect refugee arrivals and that are not plausibly orthogonal to
Travel Ban exposure:

- COVID-19 travel restrictions and operational disruptions (beginning in 2020).
- USRAP ceiling/operations collapse and subsequent rebuild (2020--2022).
- Formal rescission of the Travel Ban (early 2021), after which the "treatment" is no longer active.
- Large humanitarian shocks with strong nationality composition (e.g., Afghanistan and Ukraine),
  which change the treated/control composition and can induce differential changes unrelated to the
  Travel Ban itself.

Therefore, a single post indicator ("post >= 2018") cannot simultaneously represent:
1) "Travel Ban is active" and 2) "the only major relevant change is the Travel Ban."

At the same time, for maximum knowledge production we still want to characterize how treated vs.
control nationalities diverge across these overlapping regimes, provided we label the estimand
correctly and avoid accidentally re-framing a descriptive regime comparison as a causal Travel Ban
effect.

### Requirements

- Preserve a clearly-identified primary causal estimand for the Travel Ban that is not dominated by
  post-2020 confounding.
- Enable an explicit, well-labeled extension that shows treated/control divergence through FY2024,
  without claiming it is a causal Travel Ban effect.
- Maintain reproducibility: results must be regenerated from the pipeline, not manually edited.
- Avoid data-quality pitfalls: do not treat aggregate pseudo-nationalities (e.g., "Total") as
  nationality units in DiD/event-study estimation.
- Keep the extension figure supplemental (appendix) to prevent accidental reinterpretation as the
  main causal result.

### Challenges

1. **Time-varying treatment**: The Travel Ban is not a permanent policy. A static post indicator
   conflates "while active" and "after rescission."
2. **Major confounding shocks**: COVID and USRAP collapse/rebuild are large, contemporaneous shocks.
3. **Compositional shifts**: Post-2021 refugee composition changes sharply for humanitarian reasons
   unrelated to the Travel Ban; treated countries may move differently for reasons unrelated to the
   original intervention.
4. **Data hygiene**: The refugee arrivals panel includes non-country pseudo-nationalities such as
   `"Total"` (and FY2021 `"Fy Refugee Admissions"`). Including these units violates the analysis
   unit definition (country-of-nationality) and can distort estimates.

## Decision

### Decision 1: Keep the pre-COVID Travel Ban DiD/event study as the primary causal analysis

**Decision**: Maintain the core Travel Ban DiD/event-study identification window (FY2002--FY2019)
as the primary causal analysis and main figure/table outputs.

**Rationale**:
- FY2020+ contains strong, multi-channel confounding that is not plausibly addressable with the
  simple treated/control nationality design.
- The Travel Ban is rescinded in 2021, so "post-2021" cannot be interpreted as an ongoing treatment
  effect.
- The pre-2020 window remains the cleanest setting for interpreting the treated/control contrast as
  a policy-associated divergence.

**Implementation**:
- Continue to estimate primary DiD/event-study with `year < 2020` in Module 7.
- Continue to treat FY2018 as the first full post-treatment year (FY2017 treated as the reference
  period due to partial/contested implementation in 2017).

**Alternatives Considered**:
- Treat 2017 as the first post year: rejected (partial-year and injunctions create ambiguous
  exposure; inconsistent with annual aggregation).
- Include FY2020+ as "more post": rejected as a primary causal design due to confounding and
  post-rescission treatment invalidity.

### Decision 2: Add a supplemental "regime-dynamics" extension through FY2024 (explicitly non-causal)

**Decision**: Add an explicitly-labeled supplemental extension that estimates:
1) an **extended event study** through FY2024 and
2) a **period-collapsed treated×period model** that summarizes treated/control divergence across
   three post-2017 regimes.

The extension is descriptive: it reports treated-control divergence conditional on nationality and
year fixed effects, without interpreting these differences as causal Travel Ban effects.

**Rationale**:
- Extending the event study provides transparent visualization of dynamics (persistence, rebound,
  or reversal) across later regimes.
- Collapsing post-2017 years into interpretable regime blocks allows communication without
  overfitting year-by-year coefficients.
- Explicit non-causal labeling allows knowledge production while avoiding invalid causal claims.

**Regime blocks (FY, descriptive)**:
- **Ban era (pre-COVID)**: FY2018--FY2019 (policy active; minimal pandemic confounding).
- **COVID/USRAP collapse + rescission transition**: FY2020--FY2021 (pandemic + operational shock;
  rescission occurs in early 2021).
- **USRAP rebuilding + humanitarian shocks**: FY2022--FY2024 (post-rescission, rapid scaling and
  major composition shifts).

**Implementation**:
- Add a new estimator that fits:
  \[
  \log(y_{ct}+1) = \alpha_c + \lambda_t + \sum_{p \in \{2018\text{--}19, 2020\text{--}21, 2022\text{--}24\}}
  \delta_p \cdot \mathbf{1}\{c \in \text{treated}\}\mathbf{1}\{t \in p\} + \varepsilon_{ct}
  \]
  and reports each $\delta_p$ as a regime-specific treated/control divergence relative to the
  pre-2018 period.
- Save a separate results payload and (optional) appendix figure; do not replace the main Figure 6.

**Alternatives Considered**:
- Full-year event study through 2024 as main figure: rejected to avoid encouraging causal
  misinterpretation.
- Model post-2020 with additional controls (e.g., ceilings): rejected for now because ceilings are
  endogenous to policy regimes and would shift the estimand; requires a dedicated design ADR.

### Decision 3: Enforce nationality-unit hygiene by excluding pseudo-nationalities from estimation

**Decision**: Exclude non-country pseudo-nationalities from DiD/event-study estimation:
`"Total"` and `"Fy Refugee Admissions"` (case-insensitive match).

**Rationale**:
- These rows are aggregates, not nationality units. Treating them as controls violates the unit of
  analysis and can distort coefficients.
- The exclusion is a correctness fix: it restores the intended estimand (country-nationality unit).

**Implementation**:
- Filter pseudo-nationalities during Travel Ban DiD data preparation in Module 7 so all downstream
  estimators (primary and supplemental) operate on consistent units.

**Alternatives Considered**:
- Leave as-is: rejected; unit definition violation.
- Filter only for the extension: rejected; would produce inconsistent estimands across primary vs
  supplemental results.

## Consequences

### Positive
1. Preserves a clean primary causal analysis while enabling extended learning.
2. Makes the post-2020 dynamics visible without overstating causal claims.
3. Prevents silent unit-definition errors (pseudo-nationalities treated as countries).
4. Creates a structured template for other regime-dynamics supplements (e.g., other policy shifts).

### Negative
1. Adds additional outputs and complexity to the causal module.
2. Post-2020 extension may be misread as causal despite labeling.
3. Filtering pseudo-nationalities may change previously reported coefficients.

### Risks and Mitigations

**Risk**: Readers interpret post-2020 coefficients as "the Travel Ban effect."
- **Mitigation**: Place the figure in the appendix; label as "regime-dynamics extension"; avoid
  causal language in captions and surrounding text.

**Risk**: Coefficient instability from compositional shifts after 2021.
- **Mitigation**: Report the extension alongside explicit regime markers and note that post-2021 is
  post-rescission and composition-driven.

**Risk**: Silent data artifacts (aggregate rows) re-enter the dataset.
- **Mitigation**: Unit test asserts pseudo-nationalities are excluded from prepared DiD dataset.

## Alternatives Considered

### Alternative 1: Treat Travel Ban as a single "god-tier" structural break across all years

**Description**: Keep one treatment and one post indicator through 2024, interpret as the Travel Ban
effect.

**Pros**:
- Simple narrative.
- More post years.

**Cons**:
- Invalid estimand (treatment rescinded; multiple confounders).
- Encourages over-interpretation.

**Why Rejected**: Violates identification assumptions and conflates multiple regimes.

### Alternative 2: Drop Travel Ban causal analysis entirely and treat all shifts as regime dummies

**Description**: Replace the nationality-level DiD with a fully descriptive regime-dummy framework.

**Pros**:
- Avoids causal claims entirely.

**Cons**:
- Loses the one setting with cross-sectional policy variation suited to DiD.
- Reduces inferential content where it is most defensible (pre-2020).

**Why Rejected**: Sacrifices a relatively well-identified policy contrast.

## Implementation Notes

### Key Functions/Classes
- `prepare_travel_ban_did_data()`: filters pseudo-nationalities and constructs treatment/post fields.
- `estimate_did_travel_ban()`: primary DiD, restricted to `year < 2020`.
- `estimate_event_study()`: primary event study, restricted to `year < 2020`.
- `estimate_event_study_extended()` (new): extended event study through FY2024 (supplemental).
- `estimate_travel_ban_regime_dynamics()` (new): period-collapsed treated×period regressions
  (supplemental).

### Configuration Integration
No changes to `config/projection_config.yaml`.

### Testing Strategy
- Add unit tests verifying pseudo-nationalities are excluded from DiD prep.
- Add unit tests verifying extended event study includes years through FY2024 when input data does.

## References

1. `docs/governance/adrs/025-refugee-coverage-missing-state-handling.md` (post-2020 coverage policy)
2. `sdc_2024_replication/scripts/statistical_analysis/module_7_causal_inference.py`
3. `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet`

## Revision History

- **2026-01-06**: Initial version (ADR-027) - Supplemental regime-dynamics extension and unit hygiene.

## Related ADRs

- ADR-020: Extended Time Series Methodology Analysis
- ADR-024: Immigration Data Extension and Fusion Strategy
- ADR-025: Post-2020 Refugee Coverage and Missing-State Handling
- ADR-026: Amerasian/SIV Handling in Status Decomposition and Scenario Forecasts
