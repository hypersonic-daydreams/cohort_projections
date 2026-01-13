# Equation Explanation: Shift-Share (Bartik) Instrument

**Number in Paper:** Eq. 15
**Category:** Instrumental Variables
**Paper Section:** 2.7.4 Shift-Share (Bartik) Index

---

## What This Equation Does

The Bartik (or Shift-Share) instrument is a method for isolating exogenous variation in a variable that might otherwise be endogenous. The basic idea is to combine two sources of variation: (1) a destination's historical composition of source groups ("shares"), and (2) national-level changes in flows from those source groups ("shifts"). The key insight is that if the national changes are driven by factors external to any single destination, they can serve as a source of exogenous variation when interacted with pre-determined local shares.

In the context of this paper, the Bartik instrument predicts immigration to a state based on: (1) which origin countries historically sent immigrants to that state (the "shares"), and (2) how national immigration from each origin country changed (the "shifts"). For example, if North Dakota historically received many immigrants from Nepal, and Nepal-to-US immigration increased nationally, the Bartik instrument predicts ND would receive more immigrants--not because of anything ND did, but because of the push factors from Nepal. This predicted variation is arguably exogenous to ND-specific factors.

---

## The Formula

$$
B_{dt} = \sum_{o} \omega_{od,t_0} \cdot g_{o,t}^{\text{US},-d}
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $B_{dt}$ | Bartik index for destination state $d$ at time $t$ | Constructed instrument | Real numbers |
| $o$ | Index for origin country | Index | All origin countries |
| $\omega_{od,t_0}$ | Origin $o$'s share of state $d$'s arrivals at baseline time $t_0$ | Pre-determined share | $[0, 1]$ |
| $t_0$ | Baseline period (e.g., 2010) | Reference time | Fixed year |
| $g_{o,t}^{\text{US},-d}$ | National growth in arrivals from origin $o$, excluding state $d$ | Leave-one-out shift | Real numbers |
| $\sum_{o}$ | Sum over all origin countries | Aggregation | -- |

---

## Step-by-Step Interpretation

1. **Calculate baseline shares $\omega_{od,t_0}$:** At a baseline period (e.g., 2010), determine what fraction of state $d$'s total immigration came from each origin country. For North Dakota in 2010, this might show: 25% from Somalia, 15% from Nepal, 10% from Mexico, etc. These shares capture ND's historical immigrant composition.

2. **Calculate national shifts $g_{o,t}^{\text{US},-d}$:** For each origin country, calculate how immigration to the US changed between baseline and year $t$. Crucially, this uses "leave-one-out" construction: exclude state $d$'s own arrivals when calculating national changes. This prevents mechanical correlation between the instrument and state $d$'s actual immigration.

3. **Multiply shares by shifts:** For each origin, multiply ND's baseline share from that origin by the national (excluding ND) change in immigration from that origin. If ND received 25% of its baseline immigrants from Somalia, and Somali immigration nationally increased by 1,000, this contributes $0.25 \times 1{,}000 = 250$ to ND's Bartik index.

4. **Sum across all origins:** Add up these products across all origin countries to get the total Bartik index. This represents the predicted change in ND's immigration based purely on national trends and ND's historical composition--no ND-specific factors enter.

5. **Use as instrument:** The Bartik index can be used as an instrumental variable in regression analysis. Instead of using actual immigration (which might be endogenous), we use predicted immigration based on the Bartik index. Any relationship between Bartik-predicted immigration and outcomes is plausibly causal because the variation comes from factors external to the destination state.

---

## Worked Example

**Setup:**
Construct Bartik instrument for North Dakota in 2020, using 2010 as baseline.

**Baseline shares (2010) for North Dakota:**

| Origin | ND Arrivals 2010 | Total ND 2010 | Share ($\omega_{od,2010}$) |
|--------|------------------|---------------|----------------------------|
| Somalia | 500 | 2,000 | 0.250 |
| Nepal | 300 | 2,000 | 0.150 |
| Mexico | 400 | 2,000 | 0.200 |
| India | 200 | 2,000 | 0.100 |
| Other | 600 | 2,000 | 0.300 |

**National shifts (2010 to 2020, excluding ND):**

| Origin | US Arrivals 2010 (excl ND) | US Arrivals 2020 (excl ND) | Change ($g_{o,2020}^{\text{US},-\text{ND}}$) |
|--------|----------------------------|----------------------------|---------------------------------------------|
| Somalia | 10,000 | 6,000 | -4,000 |
| Nepal | 8,000 | 12,000 | +4,000 |
| Mexico | 150,000 | 180,000 | +30,000 |
| India | 60,000 | 75,000 | +15,000 |
| Other | 200,000 | 220,000 | +20,000 |

**Calculation:**
```
Step 1: Multiply each share by its corresponding national shift

Somalia: 0.250 x (-4,000) = -1,000
Nepal:   0.150 x (+4,000) = +600
Mexico:  0.200 x (+30,000) = +6,000
India:   0.100 x (+15,000) = +1,500
Other:   0.300 x (+20,000) = +6,000

Step 2: Sum across all origins

B_ND,2020 = -1,000 + 600 + 6,000 + 1,500 + 6,000
          = +13,100

Step 3: Interpret the Bartik index

The Bartik instrument predicts that ND should receive about 13,100 more
immigrants in 2020 than in 2010, based solely on:
  - ND's historical immigrant composition
  - National trends in immigration from those origin countries

This predicted change is driven by push factors in origin countries and
pull factors affecting the entire US--not by ND-specific policies or
economic conditions.
```

**Interpretation:**
Despite the decline in Somali immigration nationally (which hurts ND due to its historical Somali concentration), the overall Bartik prediction is strongly positive because ND's Mexican and "Other" shares, combined with large national increases from those origins, dominate. If ND's actual immigration increased less than 13,100, this suggests ND-specific factors depressed immigration relative to what national trends would predict.

---

## Key Assumptions

1. **Exogeneity of Shares:** The baseline shares $\omega_{od,t_0}$ must be uncorrelated with future shocks to the outcome variable. Using a sufficiently lagged baseline period helps, as does controlling for baseline characteristics.

2. **Exogeneity of Shifts:** The national shifts $g_{o,t}^{\text{US},-d}$ must be driven by factors external to destination $d$. Push factors in origin countries (war, economic conditions) typically satisfy this; pull factors in the US that also affect $d$ specifically do not.

3. **Leave-One-Out Construction:** Excluding state $d$ when calculating national shifts prevents mechanical correlation. If $d$ is a large state (like California), its own immigration substantially affects national totals, making leave-one-out essential.

4. **Stable Industry/Origin Structure:** The shares capture relevant exposure. If the composition of immigrants has changed dramatically since baseline, old shares may not capture current exposure to national shocks.

5. **Quasi-Random Variation in Shifts:** Across origin countries, the shifts should not be systematically correlated with factors that also affect the outcome through other channels. This is the key identifying assumption.

---

## Common Pitfalls

- **Using contaminated shares:** If baseline shares themselves reflect unobserved factors that also affect current outcomes, the instrument is not valid. Use shares from a period far enough in the past that they are plausibly predetermined.

- **Omitting leave-one-out adjustment:** Failing to exclude state $d$ when computing national shifts creates mechanical correlation between the instrument and actual immigration. This is especially problematic for large states.

- **Weak instruments:** If baseline shares are relatively uniform across destinations (all states have similar origin composition), there is little variation in the Bartik instrument to exploit. Check first-stage F-statistics.

- **Shift endogeneity:** If national shifts are themselves driven by factors correlated with destination-specific shocks (e.g., a policy that affects both US pull factors and origin push factors), the instrument is invalid.

- **Many instruments problem:** When there are many origins, and each origin's share-times-shift is used separately, over-identification can lead to bias. The aggregated Bartik (summing across origins) avoids this but requires assuming all origins provide valid variation.

- **Heterogeneous effects:** If the effect of immigration varies across origin countries, the Bartik estimate captures a weighted average of these effects, weighted by the variance of each origin's contribution to the instrument.

---

## Related Tests

- **First-Stage F-Test:** Tests whether the Bartik instrument has sufficient predictive power for actual immigration. F-statistic should exceed 10 (or preferably higher thresholds from recent literature) to avoid weak instrument bias.

- **Overidentification Test (Sargan/Hansen):** If using multiple instruments or disaggregated shift-share components, tests whether all instruments are consistent with the same causal effect. Rejection suggests at least one component is invalid.

- **Balance Tests:** Check whether the Bartik instrument is correlated with pre-treatment characteristics. Significant correlations suggest the instrument may not be as-good-as-randomly assigned.

- **Rotemberg Weights Analysis:** Decomposes the Bartik estimate into contributions from each origin country. Identifies which origins are "driving" the results and allows assessment of whether those origins have plausibly exogenous shifts.

---

## Python Implementation

```python
import pandas as pd
import numpy as np

# Calculate baseline shares (t0 = 2010)
baseline_shares = baseline_data.groupby(['state', 'nationality']).apply(
    lambda g: g['arrivals'].sum() / nat_totals_baseline[g.name[1]]
)

# Calculate leave-one-out national changes
panel['nat_total_excl_state'] = panel['nat_total'] - panel['state_arrivals']
panel['delta_shift'] = panel['nat_total_excl_state'] - panel['nat_total_baseline_excl_state']

# Construct Bartik instrument
bartik_df = panel.groupby(['state', 'year']).apply(
    lambda g: np.sum(g['state_share'] * g['delta_shift'])
).reset_index(name='bartik_instrument')
```

**Extended implementation with IV regression:**

```python
from linearmodels.iv import IV2SLS
import pandas as pd

# Merge Bartik instrument with outcome data
df = df.merge(bartik_df, on=['state', 'year'])

# First stage: Does Bartik predict actual immigration?
first_stage = sm.OLS(
    df['actual_immigration'],
    sm.add_constant(df[['bartik_instrument', 'controls']])
).fit()
print(f"First-stage F-statistic: {first_stage.fvalue:.2f}")

# Two-stage least squares
iv_model = IV2SLS.from_formula(
    'outcome ~ 1 + controls + [actual_immigration ~ bartik_instrument]',
    data=df
)
iv_results = iv_model.fit(cov_type='robust')
print(iv_results.summary)
```

---

## References

- Bartik, Timothy J. (1991). *Who Benefits from State and Local Economic Development Policies?* W.E. Upjohn Institute for Employment Research.

- Goldsmith-Pinkham, Paul, Isaac Sorkin, and Henry Swift (2020). "Bartik Instruments: What, When, Why, and How." *American Economic Review* 110(8): 2586-2624.

- Borusyak, Kirill, Peter Hull, and Xavier Jaravel (2022). "Quasi-Experimental Shift-Share Research Designs." *Review of Economic Studies* 89(1): 181-213.

- Adao, Rodrigo, Michal Kolesar, and Eduardo Morales (2019). "Shift-Share Designs: Theory and Inference." *Quarterly Journal of Economics* 134(4): 1949-2010.
