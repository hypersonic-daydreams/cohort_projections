# Data Source Mapping and Harmonization (for “Forecasting International Migration to North Dakota”)

## 1. Source mapping matrix (recommended Table in Data section)

| Source | What it measures | Flow/Stock | Net/Gross | Population universe | Time basis | Geo detail | Coverage in your paper | Role relative to estimand (PEP ND net int’l migration) | Key limitations / caveats |
|---|---|---|---|---|---|---|---|---|---|
| **Census PEP Components of Change** | State **net international migration** component of population change (persons) | Flow (annual component) | **Net** | All movers across international boundary (citizens + noncitizens), estimated as part of population accounting | **PEP estimate year** (annual component used for July 1 estimates; not a strict Jan–Dec calendar year) | State (ND; also all states + US) | 2010–2024 (ND and US totals; panel across states) | **Primary forecast target \(Y_t\)** (dependent variable for projection scenarios) | Model-based estimates; subject to revision; includes multiple migration channels (not only LPR/refugee); 2020 is an extreme disruption and may mix true shock + measurement artifacts |
| **RPC Refugee Arrivals** | Refugee (and related humanitarian) **arrivals / initial placement** by nationality and state | Flow | **Gross inflow** | Refugees resettled via USRAP (initial placement) | **Fiscal year** (Oct–Sep) | State × nationality | 2002–2020 (used for DiD, event study, wave/duration) | Mechanism + scenario input: informs refugee component and policy responsiveness; **not the target** | Initial placement ≠ final residence (secondary migration); policy/national ceiling drives totals; FY timing mismatch to PEP; many zeros at state×nationality level |
| **DHS LPR Admissions** | Lawful permanent resident **admissions** (state of intended residence; by origin) | Flow | **Gross inflow** | LPRs (green card recipients; includes adjustments of status) | **Fiscal year** | State × origin | FY2023 cross-section (gravity/concentration) | Composition/predictor module: origin structure + gravity-style predictors; **not the target** | Intended residence may differ from actual; one-year cross-section limits identification; excludes temporary/other statuses; not net of emigration |
| **ACS Foreign-Born Stock** | Resident **foreign-born population stock** by origin (survey estimate; MOE) | **Stock** | N/A | People residing in state at survey time who are foreign-born | Survey year (1-year or 5-year depending on table; you must specify) | State × origin | 2009–2023 (diaspora, LQs, stock persistence) | Predictor/validation: diaspora proxies, concentration diagnostics, network features | Sampling error (MOE); undercoverage for some groups; stock conflates past flows + retention + internal migration; measurement error can attenuate relationships |

---

## 2. Calendar year vs. fiscal year harmonization strategy (recommended text + practice)

### Step 1: Pick one “analysis-year” convention and stick to it

Because your **forecast target is PEP**, the cleanest convention is:

- Treat “year \(t\)” for the estimand as the **PEP estimate year** used in components-of-change accounting.
- Always label refugee and LPR data as **FY** in text, tables, and figure captions.

Add one sentence in Data:
> “PEP international migration is reported on the PEP estimate-year basis; DHS LPR and RPC refugee series are fiscal-year (FY) counts. We preserve native time bases and harmonize only when using FY series as predictors for the PEP target.”

### Step 2: When FY series are used as predictors for PEP-year outcomes, use an explicit crosswalk

If you need a simple, defensible conversion without monthly data, use overlap weights. PEP-year \(t\) (Jul–Jun) overlaps FY\(t\) (Oct–Sep) for 9 months and overlaps FY\((t-1)\) for 3 months. A transparent approximation is:

\[
X^{\text{PEP-year}}_{t} \approx 0.75\,X^{\text{FY}}_{t} + 0.25\,X^{\text{FY}}_{t-1}.
\]

Where feasible, show sensitivity to ±1-year shifts (because the exact mapping depends on seasonality).

### Step 3: Treatment-year definitions must follow the time base

- Travel Ban begins **Jan 2017** (calendar).
- For **FY** refugee data, FY2017 is a partial exposure year; FY2018 is the first full FY post-period.
- Recommendation: present main results with FY2018 as post, and show robustness with FY2017 treated as transition or post.

---

## 3. Recommended changes to the paper’s Data section (concrete edits)

1. **Add a new subsection: “Estimand, Time Base, and Harmonization.”**
   - Define \(Y_t\) (PEP ND net international migration).
   - Define each auxiliary object (RPC refugees, DHS LPR, ACS stock).
   - State the FY vs PEP-year issue and the chosen crosswalk.

2. **Add (or revise) the data-source summary table to include “Role in analysis.”**
   - One column should literally say: “Primary target / Predictor / Validation / Scenario input.”

3. **Label all flow vs stock objects in captions.**
   - Example: “Refugee arrivals (FY counts)” vs “Net international migration (PEP component, persons).”

4. **Insert a short measurement-error paragraph for ACS.**
   - Explicitly note MOE and how you will treat it (at minimum: robustness; ideally: simulation-based propagation into uncertainty bands where ACS enters predictors).

5. **Insert a short “conceptual alignment” paragraph.**
   - One sentence each on: net vs gross; initial vs eventual residence; stock vs flow; and why these differences matter for identification and for forecasting.
