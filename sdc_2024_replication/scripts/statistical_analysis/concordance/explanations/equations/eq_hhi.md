# Equation Explanation: Herfindahl-Hirschman Index

**Number in Paper:** Eq. 2
**Category:** Concentration Measure
**Paper Section:** 2.2 Descriptive and Concentration Methods

---

## What This Equation Does

The Herfindahl-Hirschman Index (HHI) measures how concentrated or spread out a distribution is across different categories. In the context of migration, it answers the question: "Does North Dakota receive immigrants from many countries in roughly equal proportions, or does migration come predominantly from just a few countries?"

Think of it this way: if immigration came equally from 100 countries (1% each), the HHI would be very low (100), indicating high diversity. If all immigration came from just one country, the HHI would be 10,000 (the maximum), indicating complete concentration. The HHI is widely used in economics to measure market concentration (whether an industry is dominated by a few large firms or spread across many competitors), and the same logic applies perfectly to measuring the geographic concentration of migration origins.

---

## The Formula

$$
\text{HHI} = \sum_{i=1}^{N} s_i^2 \times 10{,}000
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| HHI | Herfindahl-Hirschman Index | Output | 0 to 10,000 |
| $s_i$ | Share of migration from country $i$ (as a decimal) | Variable | 0 to 1 |
| $N$ | Number of origin countries | Constant | Positive integer |
| $\sum_{i=1}^{N}$ | Sum over all origin countries | Operator | - |
| 10,000 | Scaling factor to convert to standard HHI scale | Constant | - |

---

## Step-by-Step Interpretation

1. **Calculate Each Country's Share:** $s_i$

   For each origin country, divide its migration count by the total migration. For example, if 200 of 1,000 total immigrants came from Mexico, $s_{Mexico} = 200/1000 = 0.20$ (or 20%).

2. **Square Each Share:** $s_i^2$

   Squaring the shares gives more weight to larger contributors. A country with 50% share contributes $0.50^2 = 0.25$ to the sum, while a country with 1% share contributes only $0.01^2 = 0.0001$. This is what makes HHI sensitive to concentration.

3. **Sum the Squared Shares:** $\sum_{i=1}^{N} s_i^2$

   Add up all the squared shares. If there are many countries with small shares, this sum will be small. If one or two countries dominate, this sum will be large.

4. **Scale by 10,000:** $\times 10{,}000$

   Multiply by 10,000 to convert from a 0-1 scale to a 0-10,000 scale. This is a convention that makes the numbers easier to read and compare (e.g., an HHI of 2,500 is easier to discuss than 0.25).

---

## Worked Example

**Setup:**
Suppose North Dakota receives immigrants from 5 countries with the following distribution:
- Country A: 500 immigrants (50%)
- Country B: 250 immigrants (25%)
- Country C: 150 immigrants (15%)
- Country D: 70 immigrants (7%)
- Country E: 30 immigrants (3%)

Total: 1,000 immigrants

**Calculation:**
```
Step 1: Convert to shares (decimals)
  s_A = 0.50
  s_B = 0.25
  s_C = 0.15
  s_D = 0.07
  s_E = 0.03

Step 2: Square each share
  s_A^2 = 0.50^2 = 0.2500
  s_B^2 = 0.25^2 = 0.0625
  s_C^2 = 0.15^2 = 0.0225
  s_D^2 = 0.07^2 = 0.0049
  s_E^2 = 0.03^2 = 0.0009

Step 3: Sum the squared shares
  Sum = 0.2500 + 0.0625 + 0.0225 + 0.0049 + 0.0009 = 0.3408

Step 4: Scale by 10,000
  HHI = 0.3408 * 10,000 = 3,408

Result: HHI = 3,408
```

**Interpretation:**
An HHI of 3,408 indicates "high concentration" (above 2,500 threshold). This makes sense: Country A alone accounts for half of all immigration, so the origin distribution is quite concentrated rather than diversified. For comparison:
- If all 5 countries had equal shares (20% each): HHI = 2,000
- If all immigration came from one country: HHI = 10,000
- If immigration were spread equally across 100 countries (1% each): HHI = 100

---

## Key Assumptions

1. **Complete enumeration:** All origin countries are included in the calculation (shares should sum to 1.0)

2. **Meaningful categories:** The groupings (countries) are appropriate for the analysis. Using regions instead of countries, or vice versa, would produce different HHI values

3. **Point-in-time measure:** HHI captures concentration at a single moment. Trends over time require calculating HHI for multiple periods

4. **No weighting:** All categories are treated equally. A small country and a large country are both just "one category" in the calculation

---

## Common Pitfalls

- **Percentage vs. decimal confusion:** Make sure shares are expressed as decimals (0.25), not percentages (25). Using percentages would inflate the HHI by a factor of 100.

- **Missing data:** If some origin countries are grouped as "Other," the HHI will be artificially affected. A large "Other" category inflates HHI; many small "Other" components would deflate it.

- **Comparing different N:** An HHI for a state with immigrants from 50 countries is not directly comparable to one with immigrants from 200 countries. The theoretical minimum HHI depends on N (minimum = 10,000/N).

- **Threshold interpretation:** The standard thresholds (unconcentrated < 1,500; moderate 1,500-2,500; high > 2,500) come from antitrust economics and may need adjustment for migration contexts.

---

## Related Tests

- **Location Quotient (Eq. 3):** Complements HHI by identifying which specific countries are over/under-represented rather than overall concentration
- **Entropy measures:** Alternative concentration metrics that are more sensitive to small shares
- **Gini coefficient:** Measures inequality in distribution, related concept but different interpretation

---

## Python Implementation

```python
import numpy as np

def calculate_hhi(shares):
    """
    Calculate Herfindahl-Hirschman Index from shares.

    Parameters:
    -----------
    shares : array-like
        Shares as decimals (should sum to approximately 1.0)

    Returns:
    --------
    float
        HHI on 0-10,000 scale

    Example:
    --------
    >>> shares = [0.50, 0.25, 0.15, 0.07, 0.03]
    >>> calculate_hhi(shares)
    3408.0
    """
    shares = np.array(shares)

    # Validate inputs
    if np.any(shares < 0):
        raise ValueError("Shares cannot be negative")
    if not np.isclose(shares.sum(), 1.0, atol=0.01):
        print(f"Warning: Shares sum to {shares.sum():.4f}, not 1.0")

    # Calculate HHI
    hhi = np.sum(shares ** 2) * 10000

    return hhi


def interpret_hhi(hhi):
    """
    Provide interpretation of HHI value using standard thresholds.

    Parameters:
    -----------
    hhi : float
        HHI value on 0-10,000 scale

    Returns:
    --------
    str
        Interpretation string
    """
    if hhi < 1500:
        return "Unconcentrated (HHI < 1,500): Diverse origins"
    elif hhi <= 2500:
        return "Moderately concentrated (1,500 <= HHI <= 2,500)"
    else:
        return "Highly concentrated (HHI > 2,500): Few dominant origins"


# Example usage
migration_counts = {
    'Mexico': 500,
    'India': 250,
    'China': 150,
    'Philippines': 70,
    'Canada': 30
}

total = sum(migration_counts.values())
shares = [count / total for count in migration_counts.values()]

hhi = calculate_hhi(shares)
print(f"HHI: {hhi:,.0f}")
print(interpret_hhi(hhi))

# Compare with equal distribution
n_countries = len(shares)
equal_shares = [1/n_countries] * n_countries
hhi_equal = calculate_hhi(equal_shares)
print(f"\nIf equal shares: HHI = {hhi_equal:,.0f}")
print(f"Concentration ratio: {hhi / hhi_equal:.2f}x baseline")
```

---

## References

- Herfindahl, O. C. (1950). Concentration in the Steel Industry. Unpublished Ph.D. dissertation, Columbia University.
- Hirschman, A. O. (1964). The Paternity of an Index. *American Economic Review*, 54(5), 761.
- U.S. Department of Justice & Federal Trade Commission (2010). Horizontal Merger Guidelines. [Source of standard thresholds]
- Rhoades, S. A. (1993). The Herfindahl-Hirschman Index. *Federal Reserve Bulletin*, 79, 188-189.
