# Equation Explanation: Kaplan-Meier Survival Estimator

**Number in Paper:** Eq. 16
**Category:** Survival Analysis
**Paper Section:** 2.8 Duration Analysis

---

## What This Equation Does

The Kaplan-Meier estimator calculates the probability that something (in this case, an immigration wave) "survives" past a given point in time. Think of it as answering the question: "What percentage of immigration waves are still ongoing after X years?"

Unlike simple percentages, the Kaplan-Meier method handles a tricky problem: not all observations are followed for the same amount of time. Some immigration waves might still be active when we stop observing them (we do not know when they will end). The Kaplan-Meier estimator correctly accounts for this "censoring" by calculating survival probabilities only from the subjects who could actually experience the event at each time point.

---

## The Formula

$$
\hat{S}(t) = \prod_{t_i \leq t} \left( 1 - \frac{d_i}{n_i} \right)
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $\hat{S}(t)$ | Estimated survival probability at time t | Output | [0, 1] |
| $\prod$ | Product operator (multiply all terms together) | Operator | - |
| $t_i$ | Time when the i-th event occurs | Variable | Positive real numbers |
| $t$ | The time point we want the survival probability for | Variable | Positive real numbers |
| $d_i$ | Number of events (wave terminations) happening at time $t_i$ | Count | Non-negative integers |
| $n_i$ | Number of subjects "at risk" just before time $t_i$ | Count | Positive integers |

---

## Step-by-Step Interpretation

1. **Identify event times ($t_i$):** Look at all the times when immigration waves ended (terminated). These are the "event times."

2. **Count events and at-risk subjects:** At each event time $t_i$:
   - $d_i$ = how many waves ended at this exact time
   - $n_i$ = how many waves were still ongoing (had not yet ended AND had not yet been censored) just before this time

3. **Calculate survival through each event time:** The fraction $(1 - d_i/n_i)$ is the probability of surviving through time $t_i$, given that you made it to just before $t_i$. For example, if 10 waves are at risk and 2 end, the probability of surviving through that moment is $(1 - 2/10) = 0.80$ or 80%.

4. **Multiply probabilities together:** To get the survival probability to time $t$, multiply together all the individual survival probabilities from each event time up to $t$. This follows the logic that to survive to year 5, you must survive year 1, then year 2, then year 3, and so on.

---

## Worked Example

**Setup:**
Suppose we are tracking 10 immigration waves to see how long they last. We observe the following:

| Time (years) | Events (waves ending) | Still at risk before |
|--------------|----------------------|---------------------|
| 2 | 1 wave ended | 10 waves |
| 4 | 2 waves ended | 9 waves |
| 5 | 1 wave ended | 6 waves (one was censored at year 4.5) |
| 7 | 1 wave ended | 5 waves |

**Calculation:**
```
Step 1: Survival through year 2
  S(2) = (1 - 1/10) = 0.90 = 90%

Step 2: Survival through year 4
  S(4) = 0.90 x (1 - 2/9) = 0.90 x 0.778 = 0.70 = 70%

Step 3: Survival through year 5
  S(5) = 0.70 x (1 - 1/6) = 0.70 x 0.833 = 0.583 = 58.3%

Step 4: Survival through year 7
  S(7) = 0.583 x (1 - 1/5) = 0.583 x 0.80 = 0.467 = 46.7%

Result: S(7) = 46.7%
```

**Interpretation:**
The estimated probability that an immigration wave lasts more than 7 years is 46.7%. In other words, about 47% of immigration waves are expected to persist beyond 7 years based on our observed data. Note how the survival probability can only decrease (or stay the same) over time - it never increases.

---

## Key Assumptions

1. **Non-informative censoring:** Waves that we stop observing (censored) are not systematically different from waves we continue to observe. Censoring should be unrelated to the likelihood of the wave ending.

2. **Independence:** Each immigration wave's duration is independent of other waves.

3. **Constant hazard within intervals:** The risk of an event is assumed constant between event times (this is generally not a strong assumption for the Kaplan-Meier estimator).

4. **Accurate event times:** Event times are measured precisely. If there are ties (multiple events at the same time), they are handled correctly.

---

## Common Pitfalls

- **Ignoring censoring:** Simply dividing "number of events" by "total sample size" ignores censored observations and underestimates survival. Always use Kaplan-Meier when some observations are censored.

- **Small sample sizes at tail:** The survival curve becomes unreliable when few subjects remain at risk. Confidence intervals widen dramatically. Be cautious interpreting survival estimates when $n_i$ is small.

- **Assuming causality:** The Kaplan-Meier curve is descriptive. It shows how long things last but does not explain why. Do not interpret differences between groups as causal without additional analysis.

- **Confusing survival with incidence:** Survival analysis tracks how long events take to occur, not how many occur overall. A high survival probability means events happen slowly, not that they are rare.

---

## Related Tests

- **Log-Rank Test:** Used to compare survival curves between two or more groups (e.g., do immigration waves from different regions have different survival patterns?).

- **Confidence Intervals:** Greenwood's formula provides confidence bounds around the survival estimate.

- **Median Survival Time:** The time at which $\hat{S}(t) = 0.50$. Useful for summarizing typical duration.

---

## Python Implementation

```python
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt

# Example data
df_survival = pd.DataFrame({
    'duration': [2, 4, 4, 5, 7, 8, 10, 12, 15, 20],  # Time observed
    'event': [1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  # 1=event occurred, 0=censored
})

# Fit the Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(
    durations=df_survival['duration'],
    event_observed=df_survival['event'],
    label='All Immigration Waves'
)

# Access results
survival_function = kmf.survival_function_  # DataFrame with survival probabilities
median_survival = kmf.median_survival_time_  # Time when S(t) = 0.50
confidence_intervals = kmf.confidence_interval_  # 95% CI bounds

# Plot the survival curve
kmf.plot_survival_function()
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curve for Immigration Waves')
plt.show()

# Print summary statistics
print(f"Median survival time: {median_survival:.1f} years")
print(f"Survival probability at year 5: {kmf.survival_function_at_times(5).values[0]:.2%}")
```

---

## References

- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481.

- Hosmer, D. W., Lemeshow, S., & May, S. (2008). *Applied Survival Analysis: Regression Modeling of Time-to-Event Data* (2nd ed.). Wiley.

- Davidson-Pilon, C. (2019). *lifelines: survival analysis in Python*. Journal of Open Source Software, 4(40), 1317.
