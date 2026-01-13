# Equation Explanation: Location Quotient

**Number in Paper:** Eq. 3
**Category:** Concentration Measure
**Paper Section:** 2.2 Descriptive and Concentration Methods

---

## What This Equation Does

The Location Quotient (LQ) compares the relative concentration of a specific immigrant group in North Dakota to its concentration in the United States as a whole. It answers the question: "Does North Dakota have more or fewer immigrants from Country X than we would expect based on national patterns?"

For example, if 2% of all foreign-born people in the U.S. are from Nepal, but 8% of North Dakota's foreign-born population is from Nepal, then Nepalis are "overrepresented" in North Dakota relative to the national average. The LQ would be 4.0 (8% / 2%), meaning North Dakota has four times the expected share of Nepali immigrants. This helps identify which origin countries have a particularly strong connection to North Dakota compared to the rest of the country.

---

## The Formula

$$
\text{LQ}_{i,\text{ND}} = \frac{(\text{Foreign-born from } i \text{ in ND}) / (\text{Total foreign-born in ND})}{(\text{Foreign-born from } i \text{ in US}) / (\text{Total foreign-born in US})}
$$

Or more compactly:

$$
\text{LQ}_{i,\text{ND}} = \frac{\text{ND share from country } i}{\text{US share from country } i}
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $\text{LQ}_{i,\text{ND}}$ | Location Quotient for country $i$ in North Dakota | Output | 0 to infinity |
| Foreign-born from $i$ in ND | Count of immigrants from country $i$ living in North Dakota | Variable | Non-negative integer |
| Total foreign-born in ND | Total immigrant population in North Dakota | Variable | Positive integer |
| Foreign-born from $i$ in US | Count of immigrants from country $i$ living in United States | Variable | Non-negative integer |
| Total foreign-born in US | Total immigrant population in United States | Variable | Positive integer |

---

## Step-by-Step Interpretation

1. **Calculate North Dakota's Share:** $\frac{\text{Foreign-born from } i \text{ in ND}}{\text{Total foreign-born in ND}}$

   What proportion of North Dakota's immigrant population comes from country $i$? This is the numerator of the LQ ratio. For example, if ND has 25,000 foreign-born residents and 2,000 are from India, the ND share is 2,000/25,000 = 0.08 (or 8%).

2. **Calculate the National Share:** $\frac{\text{Foreign-born from } i \text{ in US}}{\text{Total foreign-born in US}}$

   What proportion of the entire U.S. immigrant population comes from country $i$? This is the denominator and serves as the "benchmark." If the U.S. has 45 million foreign-born and 2.7 million are from India, the national share is 2.7M/45M = 0.06 (or 6%).

3. **Divide to Get the Ratio:** $\text{LQ} = \frac{\text{ND share}}{\text{US share}}$

   The LQ is simply the local share divided by the national share. Using the example above: LQ = 0.08 / 0.06 = 1.33. This means North Dakota has 33% more Indians (proportionally) than the national average.

---

## Worked Example

**Setup:**
Let's calculate the Location Quotient for Bhutanese immigrants in North Dakota.

Given data:
- Foreign-born from Bhutan in ND: 3,500
- Total foreign-born in ND: 28,000
- Foreign-born from Bhutan in US: 92,000
- Total foreign-born in US: 45,000,000

**Calculation:**
```
Step 1: Calculate ND's share from Bhutan
  ND share = 3,500 / 28,000 = 0.125 (12.5%)

Step 2: Calculate US share from Bhutan
  US share = 92,000 / 45,000,000 = 0.00204 (0.204%)

Step 3: Calculate Location Quotient
  LQ = 0.125 / 0.00204 = 61.2

Result: LQ = 61.2
```

**Interpretation:**
A Location Quotient of 61.2 means that Bhutanese immigrants are 61 times more concentrated in North Dakota than in the U.S. as a whole. Put another way, if North Dakota's immigrant composition matched the national average, we'd expect only about 0.2% of its foreign-born population to be from Bhutan, but the actual figure is 12.5%. This dramatic overrepresentation typically indicates:
- Refugee resettlement patterns directing Bhutanese refugees to North Dakota
- Chain migration following initial settlement
- Specific community or employment connections

---

## Key Assumptions

1. **Meaningful comparison:** The national average is an appropriate benchmark. For some analyses, comparing to a regional average (e.g., Midwest states) might be more informative

2. **Data comparability:** Both numerator and denominator use the same definition of "foreign-born" and the same time period

3. **Non-zero denominator:** The country must have some presence in the U.S. overall for the LQ to be calculable. If US share = 0, LQ is undefined

4. **Population-based, not flow-based:** LQ typically uses stock data (people currently living somewhere), not flow data (recent arrivals). Mixing these produces misleading results

---

## Common Pitfalls

- **Small numbers problem:** If a country has very few immigrants in both ND and the US, small fluctuations cause wild swings in LQ. A country with 5 immigrants in ND and 500 in the US would have its LQ change dramatically if even one person moved.

- **Confusing LQ with absolute numbers:** A high LQ does not mean "lots of immigrants." A tiny country might have LQ = 50.0 but only represent 100 actual people in ND. Always report LQ alongside actual counts.

- **Ignoring the base rate:** LQ = 2.0 for a group that's 10% of national immigrants means 20% of ND's immigrants. LQ = 2.0 for a group that's 0.01% nationally means only 0.02% of ND's immigrants. Same LQ, very different practical significance.

- **Temporal mismatch:** Using 2010 Census data for ND counts but 2020 data for US counts produces invalid comparisons.

---

## Related Tests

- **HHI (Eq. 2):** Measures overall concentration/diversity across all origin countries, while LQ examines one country at a time
- **Chi-square test of independence:** Can formally test whether ND's origin distribution differs significantly from the national distribution
- **Shift-share analysis (Eq. 15):** Related technique that decomposes changes over time into national trends vs. local factors

---

## Python Implementation

```python
import numpy as np
import pandas as pd

def calculate_lq(nd_from_i, nd_total, us_from_i, us_total):
    """
    Calculate Location Quotient for a specific origin country.

    Parameters:
    -----------
    nd_from_i : int or float
        Foreign-born from country i in North Dakota
    nd_total : int or float
        Total foreign-born in North Dakota
    us_from_i : int or float
        Foreign-born from country i in United States
    us_total : int or float
        Total foreign-born in United States

    Returns:
    --------
    float
        Location Quotient (LQ > 1 indicates overrepresentation)

    Example:
    --------
    >>> calculate_lq(3500, 28000, 92000, 45000000)
    61.14...
    """
    # Input validation
    if nd_total <= 0 or us_total <= 0:
        raise ValueError("Total populations must be positive")
    if nd_from_i < 0 or us_from_i < 0:
        raise ValueError("Country populations cannot be negative")

    # Calculate shares
    nd_share = nd_from_i / nd_total
    us_share = us_from_i / us_total

    # Handle zero US share (country not present nationally)
    if us_share == 0:
        if nd_share == 0:
            return 1.0  # Neither place has this group
        else:
            return np.nan  # ND has them but US doesn't (undefined)

    # Calculate and return LQ
    lq = nd_share / us_share
    return lq


def interpret_lq(lq):
    """
    Provide interpretation of Location Quotient value.

    Parameters:
    -----------
    lq : float
        Location Quotient value

    Returns:
    --------
    str
        Interpretation string
    """
    if np.isnan(lq):
        return "Undefined (no national presence)"
    elif lq < 0.5:
        return f"Strongly underrepresented (LQ = {lq:.2f})"
    elif lq < 0.8:
        return f"Moderately underrepresented (LQ = {lq:.2f})"
    elif lq <= 1.2:
        return f"Near national average (LQ = {lq:.2f})"
    elif lq <= 2.0:
        return f"Moderately overrepresented (LQ = {lq:.2f})"
    else:
        return f"Strongly overrepresented (LQ = {lq:.2f})"


# Example: Calculate LQ for multiple countries
countries_data = {
    'Bhutan': {'nd': 3500, 'us': 92000},
    'India': {'nd': 2000, 'us': 2700000},
    'Mexico': {'nd': 4000, 'us': 10500000},
    'Canada': {'nd': 3000, 'us': 800000},
    'Somalia': {'nd': 1500, 'us': 150000},
}

nd_total = 28000
us_total = 45000000

print("Location Quotients for North Dakota Foreign-Born Population")
print("=" * 60)

results = []
for country, data in countries_data.items():
    lq = calculate_lq(data['nd'], nd_total, data['us'], us_total)
    nd_share = data['nd'] / nd_total * 100
    us_share = data['us'] / us_total * 100
    results.append({
        'Country': country,
        'ND Count': data['nd'],
        'ND Share (%)': nd_share,
        'US Share (%)': us_share,
        'LQ': lq
    })
    print(f"{country:12} | Count: {data['nd']:5,} | "
          f"ND: {nd_share:5.1f}% | US: {us_share:5.2f}% | LQ: {lq:6.1f}")
    print(f"             | {interpret_lq(lq)}")
    print()

# Convert to DataFrame for further analysis
df_results = pd.DataFrame(results).sort_values('LQ', ascending=False)
print("\nSorted by LQ (highest overrepresentation first):")
print(df_results.to_string(index=False))
```

---

## References

- Florence, P. S. (1939). Report on the Location of Industry. London: Political and Economic Planning. [Original development of LQ concept]
- Isserman, A. M. (1977). The Location Quotient Approach to Estimating Regional Economic Impacts. *Journal of the American Institute of Planners*, 43(1), 33-41.
- Moineddin, R., Beyene, J., & Boyle, E. (2003). On the Location Quotient Confidence Interval. *Geographical Analysis*, 35(3), 249-256. [Statistical properties and confidence intervals]
- U.S. Bureau of Labor Statistics. Location Quotients. [Standard application in labor economics]
