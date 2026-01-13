# Statistical Test Explanation Template

## Test: [TEST_NAME]

**Full Name:** [FULL_NAME]
**Category:** [CATEGORY]
**Paper Section:** [SECTION]

---

## What This Test Does

[1-2 paragraph plain-English explanation of the test's purpose]

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H₀):** | [Null hypothesis statement] |
| **Alternative (H₁):** | [Alternative hypothesis statement] |

---

## Test Statistic

$$
[FORMULA_FOR_TEST_STATISTIC]
$$

**Distribution under H₀:** [e.g., standard normal, chi-squared with k df, etc.]

---

## Decision Rule

| Significance Level | Critical Value | Decision |
|-------------------|----------------|----------|
| α = 0.01 | [value] | Reject H₀ if test stat [>/<] critical |
| α = 0.05 | [value] | Reject H₀ if test stat [>/<] critical |
| α = 0.10 | [value] | Reject H₀ if test stat [>/<] critical |

**P-value approach:** Reject H₀ if p-value < α

---

## When to Use This Test

**Use when:**
- [Condition 1]
- [Condition 2]

**Don't use when:**
- [Condition that violates assumptions]
- [Better alternative exists for this situation]

---

## Key Assumptions

1. **[Assumption 1]:** [Explanation of why this matters]
2. **[Assumption 2]:** [Explanation]
3. ...

---

## Worked Example

**Data:**
[Describe the sample data]

**Calculation:**
```
Step 1: [calculation]
Step 2: [calculation]
Test statistic = [value]
P-value = [value]
```

**Interpretation:**
[What does this result tell us about H₀?]

---

## Interpreting Results

**If we reject H₀:**
[What this means in practical terms]

**If we fail to reject H₀:**
[What this means - NOT the same as "accepting H₀"]

---

## Common Pitfalls

- **[Pitfall 1]:** [How to avoid]
- **[Pitfall 2]:** [How to avoid]

---

## Related Tests

| Test | Use When |
|------|----------|
| [Alternative test 1] | [Situation where this test is preferred] |
| [Complementary test] | [Situation where both tests are used together] |

---

## Python Implementation

```python
[Full working code example with comments]
```

---

## Output Interpretation

```
[Example output from Python]

- Test statistic: [interpretation]
- P-value: [interpretation]
- Critical values: [interpretation]
```

---

## References

- [Original paper introducing the test]
- [Key methodological reference]
