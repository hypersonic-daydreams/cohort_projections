# Integrated Causal Inference Recommendations

## Triangulation Assessment
### Do DiD and SCM tell a consistent story?
- **DiD/event study:** Consistent evidence of a sharp post-2018 reduction in refugee arrivals from the targeted nationalities, relative to other nationalities, with no strong evidence of differential pre-trends in the few years leading up to treatment.
- **SCM:** Not directly comparable (state-level international migration rate vs nationality-level refugee arrivals) and not causally identified under a national shock; keeping it as causal triangulation risks undermining the entire section.

Bottom line: **DiD is your causal workhorse**; SCM cannot be used as causal corroboration in its current form.

### Does Bartik provide independent corroboration?
As implemented, Bartik provides **predictive relevance** (a strong relationship between a refugee-based shift-share index and international migration), not an independent causal estimate of the Travel Ban. With modern shift-share inference and a Ban-aligned shock definition, it can support the claim that ND’s international migration is meaningfully tied to refugee-flow drivers, which is compatible with the policy-sensitivity narrative.

## Strongest Defensible Claims
1. **Short-run Travel Ban effect on targeted-origin refugee arrivals:**
   Refugee arrivals to North Dakota from Travel-Ban nationalities fell substantially in the first full post years (2018–2019) relative to other origin countries, under a standard parallel-trends assumption for nationalities.
2. **Policy sensitivity as a forecasting-relevant mechanism:**
   Because ND’s international migration has a large refugee-driven component, policies that restrict specific origin groups can materially move ND’s realized flows—meaning forecasts that ignore policy regimes are miscalibrated.
3. **What you cannot claim (without more work):**
   - A single-state “ND would have been X absent the Travel Ban” counterfactual based on SCM.
   - A precise Travel Ban effect on total ND international migration (PEP net) without bridging from refugee arrivals to net international migration and showing the mapping.

## Required Caveats (must appear in the paper)
1. **National shock / spillovers:** The Travel Ban affects all states, so cross-state donor designs require an exposure/intensity framework; we do not treat other states as untreated.
2. **Outcome mismatch:** Refugee arrivals (gross inflow, FY) and PEP net international migration (net, CY) are not the same object; the paper must state the estimand and the mapping.
3. **Inference and small treated group:** Only 7 treated nationalities; clustered inference and placebo/randomization checks are required.
4. **Post-period length:** The clean post period before COVID is short (2018–2019); claims should be explicitly framed as short-run effects.

## Revision Priority List
1. **SCM triage (highest priority):** Relabel SCM as descriptive only (or move to appendix) and remove causal language; replace any “SCM causal triangulation” with an exposure-intensity panel design if needed.
2. **Re-estimate DiD with modern inference:** PPML FE + clustering by nationality; add event-study plot and joint pretrend test; run placebo and timing sensitivity checks.
3. **Bartik modernization:** Clarify units, add leave-one-out shifts, and compute shift-share-robust SE (AKM/BHJ). Reframe as exposure index unless you add a defensible second stage.
4. **Narrative tightening:** Align estimand (refugee arrivals vs net international migration) and explicitly position DiD as the key policy-identification result supporting the forecasting argument.
