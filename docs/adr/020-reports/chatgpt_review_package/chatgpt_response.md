## Executive Assessment

Extending to **2000–2024 (n=25)** can be methodologically defensible **in a top-tier demography methods paper**, but only if you treat the result as a *spliced, multi-instrument series* rather than a single, homogeneous measurement process. The Census Bureau’s “don’t combine vintages” warning is real and should be taken at face value—but it’s guidance aimed at preventing *naïve* time-series comparisons that ignore changing benchmarks and methods. Researchers can still build an extended series, as long as they explicitly model (or at minimum, bracket) the resulting discontinuities and show that substantive conclusions do not depend on cross-vintage level differences.

Crucially, your strongest statistical “red flag” (the significant decade-to-decade mean difference at the 2009–2010 boundary) does **not** look like a clean *step change at 2010* in North Dakota: 2009→2010 is a small change (521 to 468), while the large increases occur later in the 2010s and then again post‑2021. That pattern is consistent with **real economic and policy-era dynamics** (Bakken + later post‑COVID surge), even though it remains *compatible* with methodology differences contributing some portion of the shift. Meanwhile, the extreme variance ratio (29:1) is a warning sign mainly about **non-constant variance and regime changes**, not necessarily proof of “bad data.”

Bottom line: the most publication-defensible path is **Option C (hybrid)**: keep primary inference anchored in the more comparable window, and use the extended series for robustness and carefully specified models that acknowledge regime/vintage changes.

---

## Answer to Primary Question 1

### Is extension methodologically defensible, and under what conditions?

**Yes—conditionally.** Extending the series is defensible **if you frame the 2000–2024 series as a constructed splice across measurement regimes**, and you avoid letting conclusions hinge on across-vintage level comparisons that the Census itself cautions against. The necessary conditions are:

1. **Declare the data-generating reality honestly:** you do not have one measurement system from 2000–2024; you have three. Vintage 2009 (residual/decennial foreign-born anchoring), Vintage 2020 (ACS ROYA-based), Vintage 2024 (ROYA plus DHS-based adjustments for humanitarian migration) are not just cosmetic relabels—they are different instruments.

2. **Treat vintage transitions as “regime boundaries,” not as substantive demographic shocks by default.** The Census Bureau’s warning (“data from separate vintages should not be combined”) becomes, in your paper, a *design constraint you explicitly manage*, not something you pretend doesn’t exist.

3. **Make identification explicit:** without overlap years measured under both methods, you **cannot nonparametrically identify** a unique “correction factor” that converts Vintage 2009 levels into Vintage 2020 levels. Any “correction” is therefore model-based (assumption-driven). That’s fine in a methods paper—but only if you show sensitivity to those assumptions.

4. **Keep primary substantive claims anchored in within-regime variation,** and use cross-regime comparisons as secondary / robustness evidence.

If you do those four things, reviewers may still push you (because reviewers are reviewers), but your approach is defensible.

---

### How to interpret the discrepancy between the significant t-test and non-significant Chow test?

This is not a paradox; it’s two tests answering **different questions** under **different model assumptions**.

* The **t-test/Welch/Mann–Whitney** results (p≈0.003, d≈1.56) show that the **average level** in 2010–2019 is much higher than 2000–2009. That is a statement about *two decade blocks*, not about a discrete jump at exactly 2010.

* The **Chow test** (p=0.45 at 2009–2010) asks whether a **specific regression model** (here, a linear trend model) has **different parameters** before vs. after the breakpoint. If the series is not well-described by a single linear trend, or if the main changes happen *after* 2010 rather than *at* 2010, the Chow test at 2010 won’t “light up.” Low power with small samples and heteroskedasticity also push in that direction.

The key practical implication: **the significant t-test does not prove a “methodology-induced jump at 2010.”** It proves the 2010s were different from the 2000s in mean level—which could be (a) real change, (b) methodology, or (c) both. The non-significant Chow test, plus the fact that 2009→2010 itself is smooth in ND, is evidence *against* an abrupt discontinuity exactly at the vintage boundary.

---

### Does the shared pattern across similar states suggest real regional effects vs. measurement artifacts?

It suggests **real regional effects are plausibly large**, but it does not exonerate measurement artifacts.

What it *does* buy you:

* If ND, SD, and MT all show large increases from the 2000s to the 2010s while a comparison state like WY does not, that heterogeneity is more consistent with **regional economic/policy drivers** than with a purely mechanical measurement shift that should affect everyone similarly.

What it does *not* fully settle:

* A methodology change could still raise measured migration in many small states, and differences in state trajectories could reflect how the new method interacts with small denominators, ACS sampling error, and state allocation procedures. Agent 1’s documentation that state-level allocation for the DHS humanitarian adjustment may not reflect true settlement patterns is exactly the kind of nuance that keeps “artifact vs reality” from being a clean binary.

So: the cross-state evidence is supportive of real signal, but you should still treat vintage boundaries as potential measurement regime shifts in modeling and interpretation.

---

### What documentation and caveats are required for defensibility?

To make extension publication-safe, I would expect:

* **A dedicated “Data comparability” subsection** (not just a footnote) explaining:

  * What “vintage” means in PEP,
  * The major methodological differences across your three segments,
  * The Census Bureau’s warning about combining vintages,
  * Why you are combining anyway (statistical necessity + transparent sensitivity plan).

* **An explicit analysis plan that separates inference from sensitivity:**

  * “Primary models estimated on 2010–2024; extended 2000–2024 used for robustness / exploratory time-series diagnostics.”
  * “No substantive interpretation is based solely on the level difference between 2000s and 2010s.”

* **Visible robustness outputs:**

  * Same headline conclusion under (i) n=15, (ii) n=25 with vintage controls, (iii) excluding 2020, (iv) excluding 2000–2009, etc.

* **A clear naming convention** like “spliced PEP-vintage series” or “constructed vintage-bridged series,” so readers never confuse it with a single official vintage.

That’s the kind of transparency that disarms (most) reviewer objections.

---

## Answer to Primary Question 2

### Which option should you take?

**Option C (Hybrid) is the best-supported choice.**

It dominates the others on “reviewer defensibility per unit of pain”:

* **Option B (extend with caveats only)** is risky because you’d be asking readers to accept that a multi-regime measurement series can be treated as a single stochastic process *without controls*. Given (i) the Census’s explicit warning and (ii) strong evidence of cross-vintage mean and variance differences, that’s an easy target in peer review.

* **Option A (extend with statistical corrections)** sounds tempting, but without overlap years, “corrections” are mostly assumption-driven and can create a *false sense of comparability*. In a methods journal, reviewers will ask: “How do you identify the splice factor?” If the honest answer is “we can’t; we assume,” that’s not fatal, but it’s rarely worth making the corrected series the star of the show.

* **Option D (stay n=15)** is defensible but leaves you with exactly the constraints you listed—and n=15 is not just small; it’s also a period with COVID shock and a major methodological update in the 2020s, so it’s not “clean” in the way people sometimes imagine.

**Option C** is the sweet spot: it lets you keep your main inferential claims on the most coherent window, while still extracting value from the longer series as robustness and for certain diagnostics.

---

### Is the 29:1 variance ratio a serious concern requiring correction?

It’s a serious concern requiring **modeling respect**, not necessarily a single “correction.”

* The raw variance ratio is huge largely because the *level of the series rises* and because 2020–2024 contains extreme shocks (COVID suppression then rebound/surge). That combination will mechanically inflate variance.

What you should do about it:

* Avoid methods that silently assume homoskedasticity.
* Consider modeling **rates** (per 1,000 population) or using a variance-stabilizing transform (with care because you have a negative value in 2003).
* Use **heteroskedasticity-robust or regime-specific variance** approaches (see next steps).

So yes, it matters—but not in the simplistic “your data are invalid” sense.

---

### Would corrections actually improve on “document and report sensitivity”?

**Lightweight corrections (controls) can help; heavy splicing is harder to defend.**

A good compromise—especially in a demography methods paper—is:

* Use **vintage/regime indicators** (and possibly interactions with trend) in the extended-series models.
* Treat 2020 as an **intervention/outlier** year (pandemic shock) rather than forcing it into the same error structure.
* Report results both with and without these adjustments.

That’s “correction” in the statistical sense (you’re controlling for known regime changes), without pretending you can convert one measurement system into another perfectly.

---

### Should you proceed to Phase B (correction methods), or decide now?

You can decide now on **Option C** as the publication strategy, because the Phase A evidence is already enough to rule out “just pool everything and hope.”

But I would still do a **targeted, bounded Phase B**—not an open-ended quest for The One True Splice Factor. Focus Phase B on a small set of defensible specifications:

* vintage/regime dummy + piecewise trend
* intervention for 2020
* robust standard errors / regime-specific variance
* sensitivity to excluding 2000–2009

That gives you concrete, publishable robustness checks without overpromising measurement comparability.

---

## Agent Report Validation

### Agent 1

Strong and methodologically grounded. The key contribution is establishing that the vintages reflect **real methodological shifts** and that the Census explicitly discourages combining vintages. That’s exactly the institutional constraint you must address directly in a paper.

Two caveats I’d add:

* The Census warning is often written for general users; in research contexts, it becomes a prompt to **model discontinuities**, not necessarily to abandon all multi-vintage work.
* It would be worth checking whether **intercensal** series (where available) offer a better “within-decade” reconstruction than the particular vintage slices you’re using—this could reduce benchmarking artifacts (see next steps).

### Agent 2

The testing suite is reasonable for Phase A triage, and it correctly flags that evidence is mixed.

My main concern is interpretive labeling: calling the decade mean difference a “level shift at the transition” invites readers to imagine a discontinuity at 2010 that the ND point values don’t actually show. A more careful phrasing is:

* “A significant difference in mean level between the 2000s and 2010s” (true), rather than
* “A discrete jump at the 2009–2010 boundary” (not demonstrated).

Also, the Chow test depends on the specified linear trend model; with obvious nonlinearity and shocks, it’s a limited lens. That’s fine—as long as it’s presented as such.

### Agent 3

The comparability checks are thoughtful and helpful, especially the cross-state comparison and the attempt to validate against ACS directionality.

Key limitations to keep front-and-center:

* The **n=5** correlation in the 2024 period (r≈0.998) is almost certainly dominated by the shared COVID suppression and rebound dynamics; it’s informative but fragile.
* ACS “stock vs flow” validation can be suggestive but is not a direct measurement comparison; it’s more of a sanity check than a calibration.

Overall, Agent 3 supports the view that the series is not nonsense and likely tracks real dynamics, but it doesn’t eliminate the need to treat regime changes explicitly.

---

## Alternative Interpretations

1. **Benchmarking/closure effects masquerading as migration change.** Some of what looks like a decade shift can come from how each decade’s estimates are “closed” to a decennial census benchmark. That can produce level differences even if underlying flows were stable. This is adjacent to methodology change but conceptually distinct: it’s about reconciliation to benchmarks, not just measuring flows differently.

2. **Scale effects explain much of the variance ratio.** Absolute variance rising with the mean is expected for count processes. The 29:1 ratio is eye-catching, but the more relevant question is whether *relative* variability or model residual variance (after scaling/transforming) changes as sharply.

3. **The 2010 “method change” may have altered composition more than totals.** Even if totals don’t jump at 2010, the ROYA approach can change *who* is counted (and how) in ways that matter for covariate models, especially in small states where ACS sampling error is nontrivial.

4. **2021–2024 may contain a state-allocation artifact even if national adjustment is correct.** Agent 1 notes Census concerns about distributing the DHS adjustment down to states using “usual methods” that may not reflect true humanitarian migrant settlement patterns. That means your most recent segment could have a different kind of measurement uncertainty than either earlier segment.

---

## Recommended Next Steps

1. **Lock in Option C as the paper’s framing.**

   * Primary inference: 2010–2024 (or even 2010–2019 plus a separate post‑2020 analysis, depending on your research focus).
   * Extended 2000–2024: robustness / diagnostics / exploratory time-series tests with regime controls.

2. **Add a “regime-aware” modeling layer (minimal Phase B).**
   Run a small set of pre-registered (or at least clearly enumerated) specifications on the n=25 series:

   * vintage dummies (2000s vs 2010s vs 2020s),
   * piecewise trends (allow slopes to differ by regime),
   * an intervention/outlier term for 2020,
   * heteroskedasticity-robust inference (or regime-specific error variance).
     Keep it simple enough that reviewers can follow, and show that your substantive conclusions don’t hinge on the splice.

3. **Do a “many-states placebo” check (if feasible).**
   If you can access the same series for all states, quantify how unusual ND’s 2000s→2010s shift is relative to the national distribution. If *everyone* jumps similarly, that screams “methodology.” If ND (and oil-adjacent states) are outliers, that supports a real driver story. Agent 3’s state comparison is a good start; scaling it up makes it much more convincing.

4. **Consider reframing your time-series ambition.**
   Even n=25 is still small for some of the things people want to do with VAR/cointegration, especially with regime changes and heteroskedasticity. If those analyses remain central, consider:

   * Bayesian/shrinkage VAR approaches, or
   * panel approaches (multi-state) to increase information without relying on a single stitched series.

5. **Write the “comparability disclosure” like you mean it.**
   Put the vintage warning, the key methodological shifts, and the sensitivity design in the main text (not buried). That’s how you turn a potential reviewer objection into a methodological feature: *“we treat instrument changes as part of the uncertainty structure.”*

6. **Treat 2020 explicitly as a shock year in any inference about long-run dynamics.**
   Whether you keep it, down-weight it, or model it as an intervention, don’t let it silently contaminate stationarity/break diagnostics.

---

If you follow this path, you’re not “breaking the rules” of the Census warning—you’re acknowledging that the rules exist because measurement changes are real, then doing what good methods papers do: modeling those changes as part of the inferential problem rather than pretending they aren’t there.
