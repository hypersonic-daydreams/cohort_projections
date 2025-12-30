# G03 ChatGPT Response

**Session:** G03 - Gravity Model & Network Analysis
**Date:** [YYYY-MM-DD]
**Status:** Pending

---

The core gravity-specification problem is that the current “PPML gravity” exercise is neither (i) a gravity model in the modern sense **nor** (ii) a clean predictive allocation model. Most importantly, the script **drops all zero flows** (`flow > 0`) before estimation, which changes the estimand to “conditional on a positive admission cell” and undercuts the usual PPML rationale (“handles zeros”). On top of that, the “full” model discusses distance conceptually but **omits any bilateral cost term**, and it relies on stock-based “mass” proxies rather than the standard origin/destination fixed-effect structure that soaks up multilateral resistance and unobserved attractiveness. The clean fix is: rebuild the estimation dataset to include the full origin–state grid (including zeros), then estimate PPML with **origin FE + state FE** and add **log distance** (or explicitly justify exclusion, ideally as a robustness check).

The implausibly tiny standard errors are almost certainly an inference artifact. Right now the GLM is fit with the **default Poisson MLE covariance** (non-robust) and **no clustering**, which effectively assumes equidispersion and independent cells. Your own diagnostics scream “not Poisson”: the full model’s Pearson χ² is ~2.63 million on ~2,676 df (dispersion ≈ 983), i.e., orders-of-magnitude overdispersion; that alone can shrink naïve SE by ~√983 ≈ 31 relative to a quasi/robust treatment. Reviewers will immediately ask for sandwich SE and clustering by **state and origin** (two-way clustering preferred). Add ACS measurement error (MOE) and the true uncertainty is larger still.

On identification: with a **single FY2023 cross-section**, you cannot defend “causal network effects” or dynamic diaspora multipliers—only contemporaneous **associations useful for prediction**. The paper should treat the diaspora coefficient (~0.096 in the current “full” model) as a **predictive diaspora–flow elasticity (cross-sectional)**, not evidence that increasing diaspora stock would increase admissions. Also avoid mixing objects: the “panel elasticity” you report (stock-on-stock persistence) is not comparable to a flow-on-stock gravity elasticity, and the ND-only cross-section estimate is negative in your output—exactly the kind of fragile, composition-driven result (refugee placement vs. other pathways) that should be framed cautiously.

Top three priority changes for revision are: (1) **stop dropping zero flows** and estimate PPML on the full origin–state grid; (2) move to a gravity-correct predictive spec with **origin FE + destination FE + (log) distance** and report **robust + (two-way) clustered** inference; (3) rewrite “network effects” language to be explicitly **predictive/descriptive** and add a clear limitations paragraph tying the gravity module to the forecasting workflow rather than to causal mechanism testing.

[Download G03_gravity_specification.md](sandbox:/mnt/data/G03_gravity_specification.md)
[Download G03_se_analysis.md](sandbox:/mnt/data/G03_se_analysis.md)
[Download G03_recommendations.md](sandbox:/mnt/data/G03_recommendations.md)
