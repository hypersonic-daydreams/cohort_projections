# G04 ChatGPT Response

**Session:** G04 - Causal Inference Deep Dive (SCM Decision)
**Date:** [YYYY-MM-DD]
**Status:** Pending

---

**Definitive SCM call:** **KEEP SCM only as a descriptive “synthetic comparator” and DROP it as a causal estimator of the Travel Ban.** A standard SCM needs untreated donors; with a national shock, other states are not untreated, so the SCM gap can’t be interpreted as “ND absent the policy” unless you *explicitly* redesign around exposure intensity—exactly the reviewer’s point.

Your **nationality-level DiD is the causal backbone** and it’s already telling a clean story: ATT = **−1.38** (SE 0.48, p=0.004), which you’ve interpreted as about a **75% reduction** in arrivals from the banned-origin group, with a pre-trend test p≈0.18.  The event study reinforces “big immediate hit”: 2018 is strongly negative (≈ −2.69, p≈0.001) and the joint pre-trend test is not significant (p≈0.149).

Your current SCM output looks “impressive” numerically (tiny pre-RMSPE, huge RMSPE ratio), but that’s exactly the trap: with only 7 pre-years, excellent fit is easy, and the donor pool is still policy-exposed. Plus, the donor weights (e.g., **Wyoming ~0.42**) scream “compositional mismatch risk” in a refugee-driven story.  Treat it as a descriptive time-series comparator **at most**, not triangulating causal identification.

**Bartik/shift-share:** you do have a strong first-stage relationship (F≈**22.46**, coef≈**4.36**),  but (i) as currently implemented it’s *only* first stage, and (ii) shift-share designs need modern inference and explicit identifying assumptions (Goldsmith‑Pinkham et al. 2020; Adão et al. 2019; Borusyak et al. 2022; plus the 2025 JEP guide).

**Strongest defensible causal claim:** “In North Dakota, refugee arrivals from Travel-Ban-targeted origins fell sharply in the first full post-ban years (2018–2019) relative to other origins, consistent with a large immediate policy effect,” with caveats about functional form, clustered inference, and the short clean post window. **Top 3 priority actions:** (1) relabel/move SCM to descriptive-only (or appendix) and purge causal language;  (2) re-estimate DiD using PPML+FE with clustering by nationality (and add wild cluster bootstrap / placebo checks);  (3) modernize shift-share inference + clarify units/interpretation (or reframe Bartik as an exposure index, not an IV estimate).

## Downloadable files

1. [Download G04_scm_decision.md](sandbox:/mnt/data/G04_scm_decision.md)
2. [Download G04_did_specification.md](sandbox:/mnt/data/G04_did_specification.md)
3. [Download G04_bartik_specification.md](sandbox:/mnt/data/G04_bartik_specification.md)
4. [Download G04_recommendations.md](sandbox:/mnt/data/G04_recommendations.md)
