# Agent Report: [Agent Number] - [Title]

## Metadata

| Field | Value |
|-------|-------|
| Agent | [1/2/3/4/5] |
| Title | [Full descriptive title] |
| Date | YYYY-MM-DD |
| Status | [Draft / Complete / Requires Review] |
| Confidence Level | [High / Medium / Low] |

---

## Executive Summary

*3-5 sentences maximum. State the key finding and its implication for the decision.*

**Bottom Line**: [One sentence stating the main conclusion]

**Recommendation**: [Proceed with extension / Proceed with caution / Do not extend / Requires Phase B investigation]

---

## Scope and Objectives

### Primary Question
*What specific question does this investigation answer?*

### Boundaries
- **In Scope**: [What this report covers]
- **Out of Scope**: [What this report does NOT cover]
- **Dependencies**: [What other agents' findings this relies on, if any]

---

## Methodology

### Data Sources
*List all data sources used with specific identifiers*

| Source | Type | Coverage | Location/Citation |
|--------|------|----------|-------------------|
| [Name] | [Primary/Secondary] | [Years/Scope] | [Path or URL] |

### Analytical Methods
*Describe methods used, with enough detail for replication*

1. **Method 1**: [Brief description]
   - Rationale: [Why this method]
   - Parameters: [Key settings/thresholds]

2. **Method 2**: [Brief description]
   - Rationale: [Why this method]
   - Parameters: [Key settings/thresholds]

### Limitations
*What this methodology cannot detect or may miss*

---

## Findings

### Finding 1: [Title]

**Summary**: [2-3 sentence summary]

**Evidence**:
- [Specific observation or statistic]
- [Specific observation or statistic]

**Uncertainty**: [Low/Medium/High] - [Brief explanation]

**Implications**: [What this means for the extension decision]

---

### Finding 2: [Title]

**Summary**: [2-3 sentence summary]

**Evidence**:
- [Specific observation or statistic]
- [Specific observation or statistic]

**Uncertainty**: [Low/Medium/High] - [Brief explanation]

**Implications**: [What this means for the extension decision]

---

*[Repeat for additional findings]*

---

## Quantitative Summary

*Machine-readable summary of key metrics. Include units and confidence intervals where applicable.*

```json
{
  "agent_id": "[1/2/3/4/5]",
  "report_date": "YYYY-MM-DD",
  "metrics": {
    "metric_1_name": {
      "value": null,
      "unit": "",
      "confidence_interval_95": [null, null],
      "interpretation": ""
    },
    "metric_2_name": {
      "value": null,
      "unit": "",
      "confidence_interval_95": [null, null],
      "interpretation": ""
    }
  },
  "categorical_findings": {
    "finding_1": {
      "conclusion": "",
      "confidence": "",
      "evidence_strength": ""
    }
  },
  "overall_assessment": {
    "recommendation": "",
    "confidence_level": "",
    "key_uncertainties": []
  }
}
```

---

## Uncertainty Quantification

### Epistemic Uncertainty (What We Don't Know)

| Unknown | Impact on Conclusion | Reducible? |
|---------|---------------------|------------|
| [Gap 1] | [Low/Medium/High] | [Yes/No/Partially] |
| [Gap 2] | [Low/Medium/High] | [Yes/No/Partially] |

### Aleatory Uncertainty (Inherent Variability)

| Source | Magnitude | Handling |
|--------|-----------|----------|
| [Source 1] | [Quantify if possible] | [How addressed] |

### Sensitivity to Assumptions

| Assumption | If Wrong, Impact | Alternative Interpretation |
|------------|------------------|---------------------------|
| [Assumption 1] | [Effect on conclusion] | [What we'd conclude instead] |

---

## Areas Flagged for External Review

*Specific items where ChatGPT 5.2 Pro's analysis would be valuable*

### Review Request 1: [Title]

**Question**: [Specific question for ChatGPT 5.2 Pro]

**Context**: [Relevant background]

**Our Tentative Answer**: [What we think, if anything]

**Why External Review Needed**: [Why we're uncertain]

---

### Review Request 2: [Title]

*[Repeat structure]*

---

## Artifacts Produced

*List all supplementary files this agent produced*

| Artifact | Filename | Format | Purpose |
|----------|----------|--------|---------|
| [Description] | `[filename.ext]` | [CSV/JSON/PNG/etc] | [What reviewer can do with it] |

### Artifact Descriptions

#### [Artifact 1 Filename]
- **Contents**: [What's in the file]
- **Schema/Format**: [Column names, JSON structure, etc.]
- **Usage**: [How to interpret/use this artifact]

---

## Conclusion

### Answer to Primary Question
*Direct answer to the question stated in Scope and Objectives*

### Confidence Assessment
*How confident are we in this answer?*

| Aspect | Confidence | Explanation |
|--------|------------|-------------|
| Data Quality | [High/Medium/Low] | [Brief explanation] |
| Method Appropriateness | [High/Medium/Low] | [Brief explanation] |
| Conclusion Robustness | [High/Medium/Low] | [Brief explanation] |
| **Overall** | **[High/Medium/Low]** | **[Summary]** |

### Implications for Extension Decision

| Option | Supported? | Confidence |
|--------|-----------|------------|
| A: Extend with corrections | [Yes/No/Partial] | [High/Medium/Low] |
| B: Extend with caveats | [Yes/No/Partial] | [High/Medium/Low] |
| C: Hybrid approach | [Yes/No/Partial] | [High/Medium/Low] |
| D: Maintain n=15 | [Yes/No/Partial] | [High/Medium/Low] |

---

## Sources and References

### Primary Sources Consulted

1. [Author/Organization]. "[Title]." *Publication*, Year. [URL if available]
2. ...

### Data Files Used

1. `[path/to/file.csv]` - [Description]
2. ...

### Methods References

1. [Citation for statistical method used]
2. ...

---

## Appendix: Technical Details

*Optional section for detailed calculations, code snippets, or extended analysis that supports findings but would clutter the main report*

### A.1 [Technical Detail Title]

[Content]

### A.2 [Technical Detail Title]

[Content]

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| YYYY-MM-DD | 1.0 | [Agent ID] | Initial report |

---

*End of Report*
