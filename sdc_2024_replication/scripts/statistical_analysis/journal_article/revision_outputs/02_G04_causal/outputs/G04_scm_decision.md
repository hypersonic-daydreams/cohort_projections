# SCM Decision: KEEP (DESCRIPTIVE ONLY)

## Executive Summary
**Definitive recommendation:** Keep the synthetic-control output **only as a descriptive “synthetic comparator”**, and **drop SCM as a causal estimator** of the Travel Ban effect.

Reason: the Travel Ban is a **national** shock, so a standard treated-vs-untreated synthetic control is not identified without an explicit **treatment-intensity / differential exposure** framing—something the current SCM does not implement.

## Identification Problem Analysis
A canonical SCM requires that the treated unit(s) experience a treatment that donor units **do not**. In your setting, the Travel Ban is a national executive policy: every state is exposed to the same federal policy regime. This matches the critique you received: “synthetic ND” built from other states is not a clean counterfactual unless treatment is redefined as *differential exposure* (e.g., ND is high-exposure because of its pre-2017 refugee composition, donors are low-exposure), and even then standard SCM needs adaptation.

A second (often overlooked) issue is estimand mismatch: your SCM outcome is a **state-level international migration rate (PEP-based)**, while your Travel Ban DiD is **refugee arrivals by nationality**. Even if SCM were identified, it would not be estimating the same causal parameter, which invites confusion and weakens the policy section.

## Recommended Action
### A) Re-label and re-locate SCM (what you should do now)
1. **Rename** the SCM output from “Synthetic Control Method (causal)” to **“Synthetic comparator (descriptive benchmark)”**.
2. **Move** it out of the causal identification core of the paper:
   - either to a short descriptive subsection (clearly separated from causal claims), or
   - to an appendix.
3. **Remove causal language and inference**:
   - Do not call gaps “treatment effects.”
   - Do not use placebo RMSPE ratios as “significance” evidence.
   - Do not claim the donor pool is “untreated.”

### B) What to do instead if you want a state-level causal design
Replace SCM with a design that matches the only credible cross-state identifying variation here: **exposure intensity**.

**Intensity DiD (recommended):**
- Define exposure for each state using *pre-policy* composition:
  - e.g., share of 2014–2016 refugee arrivals coming from Travel-Ban countries, or more broadly, a “refugee dependence” index.
- Estimate:
  \[
  Y_{st} = \alpha_s + \gamma_t + \beta (Exposure_s \times Post_t) + \varepsilon_{st}
  \]
  where Post_t is 2018+ (first full year), and Y_st is a state-year outcome aligned to your narrative (international migration rate, refugee arrivals, or a refugee-driven component).

This is clean, transparent, and directly addresses the reviewer’s “define treatment intensity” point.

### C) Only if you insist on an SCM-like estimator
If you truly need an SCM-style counterfactual *for multiple high-exposure units*, use a method built for that framing:
- **Generalized synthetic control / interactive fixed effects** (Xu 2017), where identification is about differential deviations after controlling for latent common factors.
- Still: you must define exposure, report sensitivity to that definition, and interpret results as “differential impact by exposure,” not “treated vs untreated.”

## Paper Language
You can paste (or lightly adapt) the following language:

> **Why we do not use synthetic control for causal inference.** The 2017–2018 Travel Ban was a national policy shock, so no U.S. state is plausibly “untreated.” A standard synthetic control design therefore cannot identify a counterfactual trajectory for North Dakota absent the policy, because the donor pool is also exposed to the same federal policy environment. We consequently do not interpret synthetic-control gaps as causal effects.

> **Descriptive synthetic comparator.** As a descriptive benchmark only, we construct a weighted “synthetic comparator” trajectory for North Dakota’s international migration rate by choosing weights on other states to match North Dakota’s pre-2017 trajectory. The resulting post-2017 divergence is presented as a descriptive pattern, not a causal estimate. Our causal claims about the Travel Ban rely instead on nationality-level difference-in-differences evidence.

Suggested caption text:

> “Synthetic comparator for North Dakota (descriptive): weighted combination of other states chosen to match ND’s pre-2017 international migration rate. Post-2017 gaps are **not interpreted causally** because the Travel Ban is a national shock affecting all states.”

## Residual Concerns
Even with correct labeling, SCM can still be misread by reviewers (and readers) as causal. To reduce that risk:
- Keep SCM out of the main causal table and treat it as an appendix figure.
- Avoid language like “counterfactual,” “treated,” “impact,” or “effect.”
- Do not use the RMSPE ratio to imply statistical significance.

If you need an additional “triangulation” result in the causal section, add an exposure-intensity design (state-level) or a DDD design (state × nationality × year) rather than trying to rescue SCM.
