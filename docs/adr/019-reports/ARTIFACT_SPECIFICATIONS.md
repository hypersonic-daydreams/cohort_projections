# Artifact Specifications for ADR-019 Investigation

## Purpose

This document specifies the supplementary artifacts that each agent must produce alongside their reports. These artifacts enable:

1. **External verification** by ChatGPT 5.2 Pro
2. **Reproducibility** of analyses
3. **Cross-agent integration** of findings

---

## Universal Artifacts (All Agents)

### 1. Findings Summary JSON

**Filename**: `agent[N]_findings_summary.json`

**Purpose**: Machine-readable summary of key findings for automated synthesis

**Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["agent_id", "report_date", "primary_question", "recommendation", "confidence", "findings"],
  "properties": {
    "agent_id": {
      "type": "integer",
      "enum": [1, 2, 3, 4, 5]
    },
    "report_date": {
      "type": "string",
      "format": "date"
    },
    "primary_question": {
      "type": "string"
    },
    "recommendation": {
      "type": "string",
      "enum": ["proceed_with_extension", "proceed_with_caution", "do_not_extend", "requires_phase_b", "inconclusive"]
    },
    "confidence": {
      "type": "string",
      "enum": ["high", "medium", "low"]
    },
    "option_support": {
      "type": "object",
      "properties": {
        "A_extend_with_corrections": {"type": "boolean"},
        "B_extend_with_caveats": {"type": "boolean"},
        "C_hybrid_approach": {"type": "boolean"},
        "D_maintain_current": {"type": "boolean"}
      }
    },
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "title", "conclusion", "confidence", "uncertainty"],
        "properties": {
          "id": {"type": "string"},
          "title": {"type": "string"},
          "conclusion": {"type": "string"},
          "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
          "uncertainty": {"type": "string"},
          "evidence_type": {"type": "string", "enum": ["quantitative", "qualitative", "mixed"]},
          "key_metrics": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {"type": "string"},
                "value": {"type": ["number", "string", "null"]},
                "unit": {"type": "string"},
                "interpretation": {"type": "string"}
              }
            }
          }
        }
      }
    },
    "review_requests": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "question": {"type": "string"},
          "context": {"type": "string"},
          "priority": {"type": "string", "enum": ["high", "medium", "low"]}
        }
      }
    },
    "key_uncertainties": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

### 2. Sources Bibliography

**Filename**: `agent[N]_sources.bib` (BibTeX format) or `agent[N]_sources.json`

**Purpose**: Complete, machine-readable bibliography of all sources consulted

**JSON Schema**:
```json
{
  "sources": [
    {
      "id": "unique_key",
      "type": "article|report|website|dataset|book",
      "title": "Full title",
      "authors": ["Author 1", "Author 2"],
      "year": 2024,
      "publication": "Journal or publisher name",
      "url": "https://...",
      "accessed_date": "YYYY-MM-DD",
      "relevance": "How this source was used",
      "key_excerpts": ["Direct quote 1", "Direct quote 2"]
    }
  ]
}
```

---

## Agent-Specific Artifacts

### Agent 1: Methodology Documentation

#### 1.1 Methodology Comparison Matrix

**Filename**: `agent1_methodology_matrix.csv`

**Purpose**: Side-by-side comparison of methods across vintages

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `aspect` | string | What is being compared (e.g., "NIM estimation method") |
| `vintage_2009` | string | How Vintage 2009 handles this |
| `vintage_2020` | string | How Vintage 2020 handles this |
| `vintage_2024` | string | How Vintage 2024 handles this |
| `difference_severity` | enum | none / minor / moderate / major |
| `impact_on_comparability` | string | Free text assessment |
| `source` | string | Citation for this information |

#### 1.2 Census Bureau Quotes Extract

**Filename**: `agent1_census_quotes.json`

**Purpose**: Direct quotes from Census Bureau documentation that support findings

**Schema**:
```json
{
  "quotes": [
    {
      "id": "Q001",
      "source_id": "reference to sources.json",
      "quote": "Exact text from source",
      "context": "Where in document this appears",
      "relevance": "Why this quote matters",
      "finding_supported": "Which finding this supports"
    }
  ]
}
```

#### 1.3 Data Source Timeline

**Filename**: `agent1_data_sources_timeline.csv`

**Purpose**: Track which administrative data sources were used when

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `data_source` | string | Name of data source (e.g., "ACS Foreign-Born") |
| `start_year` | int | First year incorporated |
| `end_year` | int | Last year incorporated (or 9999 if ongoing) |
| `vintage_used` | string | Which vintages use this source |
| `notes` | string | Any relevant context |

---

### Agent 2: Statistical Analysis

#### 2.1 Raw Data Extract

**Filename**: `agent2_nd_migration_data.csv`

**Purpose**: Complete data used in analysis (for reviewer to verify/replicate)

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Calendar year |
| `intl_migration` | int | Net international migration count |
| `vintage` | string | Which vintage this comes from |
| `vintage_period` | string | pre_2010 / 2010s / post_2020 |

#### 2.2 Statistical Test Results

**Filename**: `agent2_test_results.csv`

**Purpose**: Complete results of all statistical tests

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `test_id` | string | Unique identifier (e.g., T001) |
| `test_name` | string | Name of statistical test |
| `hypothesis` | string | What is being tested |
| `test_statistic` | float | Computed test statistic |
| `statistic_name` | string | Name of statistic (t, F, chi2, etc.) |
| `df` | string | Degrees of freedom |
| `p_value` | float | P-value |
| `significance_level` | float | Alpha used |
| `reject_null` | boolean | Whether null is rejected |
| `effect_size` | float | Standardized effect size if applicable |
| `interpretation` | string | What this means |
| `notes` | string | Caveats or context |

#### 2.3 Transition Metrics

**Filename**: `agent2_transition_metrics.json`

**Purpose**: Key quantitative metrics at vintage transitions

**Schema**:
```json
{
  "transitions": [
    {
      "transition": "2009_to_2010",
      "pre_period": {"start": 2000, "end": 2009},
      "post_period": {"start": 2010, "end": 2019},
      "metrics": {
        "pre_mean": null,
        "post_mean": null,
        "mean_difference": null,
        "mean_difference_pct": null,
        "pre_variance": null,
        "post_variance": null,
        "variance_ratio": null,
        "level_shift_significant": null,
        "level_shift_pvalue": null
      }
    },
    {
      "transition": "2019_to_2020",
      "pre_period": {"start": 2010, "end": 2019},
      "post_period": {"start": 2020, "end": 2024},
      "metrics": {
        "pre_mean": null,
        "post_mean": null,
        "mean_difference": null,
        "mean_difference_pct": null,
        "pre_variance": null,
        "post_variance": null,
        "variance_ratio": null,
        "level_shift_significant": null,
        "level_shift_pvalue": null
      },
      "notes": "COVID in 2020 complicates interpretation"
    }
  ]
}
```

#### 2.4 Visualizations

**Filenames**:
- `agent2_fig1_timeseries_with_vintages.png`
- `agent2_fig2_variance_by_vintage.png`
- `agent2_fig3_structural_breaks.png`
- `agent2_fig4_acf_by_vintage.png`

**Specifications**:
- Resolution: 300 DPI minimum
- Format: PNG (for upload compatibility)
- Size: At least 800x600 pixels
- Include clear axis labels, titles, legends
- Mark vintage transitions with vertical lines
- Use colorblind-friendly palette

#### 2.5 Calculation Audit Trail

**Filename**: `agent2_calculations.md` or `agent2_calculations.ipynb`

**Purpose**: Step-by-step calculations for key findings, enabling verification

**Required Sections**:
- Import and data loading
- Summary statistics by vintage
- Each statistical test with intermediate steps
- Final synthesis

---

### Agent 3: Comparability Assessment

#### 3.1 External Correlation Data

**Filename**: `agent3_external_correlations.csv`

**Purpose**: Correlation of ND migration with external indicators by period

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `indicator` | string | External indicator name |
| `period` | string | Time period (pre_2010 / 2010s / post_2020 / full) |
| `n` | int | Number of observations |
| `correlation` | float | Pearson correlation coefficient |
| `correlation_95_lower` | float | Lower CI bound |
| `correlation_95_upper` | float | Upper CI bound |
| `p_value` | float | Significance of correlation |
| `interpretation` | string | What this suggests |

#### 3.2 State Comparison Data

**Filename**: `agent3_state_comparison.csv`

**Purpose**: Compare ND patterns to similar states

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `state` | string | State abbreviation |
| `metric` | string | What is being compared |
| `vintage_2009_value` | float | Value in 2000-2009 period |
| `vintage_2020_value` | float | Value in 2010-2019 period |
| `transition_change` | float | Change at transition |
| `similar_to_nd` | boolean | Whether pattern resembles ND |

#### 3.3 Validation Data Summary

**Filename**: `agent3_validation_data.csv`

**Purpose**: Comparison of PEP to alternative data sources

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Calendar year |
| `pep_intl_migration` | int | PEP estimate |
| `dhs_lpr` | int | DHS Legal Permanent Resident count (if available) |
| `acs_foreign_born_change` | int | ACS year-over-year change (if available) |
| `source_agreement` | string | Do sources agree on direction/magnitude? |

#### 3.4 Coherence Check Results

**Filename**: `agent3_coherence_checks.json`

**Purpose**: Results of internal consistency checks

**Schema**:
```json
{
  "checks": [
    {
      "check_id": "C001",
      "check_name": "Components sum to total",
      "description": "Births - Deaths + Net Migration = Population Change",
      "passed": true,
      "max_discrepancy": null,
      "discrepancy_unit": "persons",
      "notes": ""
    },
    {
      "check_id": "C002",
      "check_name": "ND/US ratio stability",
      "description": "Is ND share of US migration stable across vintages?",
      "passed": null,
      "pre_2010_ratio": null,
      "post_2010_ratio": null,
      "ratio_change_pct": null,
      "notes": ""
    }
  ]
}
```

---

## Synthesis Artifacts (Post-Phase A)

These are produced after all agents complete, for the briefing document.

### Combined Findings Matrix

**Filename**: `synthesis_findings_matrix.csv`

**Purpose**: Cross-reference findings across agents

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `topic` | string | Topic area |
| `agent1_finding` | string | What Agent 1 found |
| `agent2_finding` | string | What Agent 2 found |
| `agent3_finding` | string | What Agent 3 found |
| `agreement_level` | enum | full / partial / conflict / n/a |
| `synthesis` | string | Combined interpretation |

### Recommendation Summary

**Filename**: `synthesis_recommendations.json`

**Purpose**: Aggregate recommendations across agents

**Schema**:
```json
{
  "agent_recommendations": [
    {"agent": 1, "recommendation": "...", "confidence": "..."},
    {"agent": 2, "recommendation": "...", "confidence": "..."},
    {"agent": 3, "recommendation": "...", "confidence": "..."}
  ],
  "synthesis": {
    "unanimous": false,
    "majority_recommendation": "...",
    "areas_of_agreement": [],
    "areas_of_disagreement": [],
    "synthesizer_recommendation": "...",
    "synthesizer_confidence": "..."
  }
}
```

---

## File Naming Conventions

All artifacts should follow this pattern:

```
agent[N]_[artifact_type]_[optional_detail].[ext]
```

Examples:
- `agent2_test_results.csv`
- `agent2_fig1_timeseries_with_vintages.png`
- `agent3_state_comparison_detailed.csv`

Synthesis artifacts use prefix `synthesis_`:
- `synthesis_findings_matrix.csv`
- `synthesis_recommendations.json`

---

## Delivery Checklist

### Agent 1 Must Deliver:
- [ ] `AGENT_1_REPORT.md` (using REPORT_TEMPLATE.md)
- [ ] `agent1_findings_summary.json`
- [ ] `agent1_sources.json` or `.bib`
- [ ] `agent1_methodology_matrix.csv`
- [ ] `agent1_census_quotes.json`
- [ ] `agent1_data_sources_timeline.csv`

### Agent 2 Must Deliver:
- [ ] `AGENT_2_REPORT.md` (using REPORT_TEMPLATE.md)
- [ ] `agent2_findings_summary.json`
- [ ] `agent2_sources.json` or `.bib`
- [ ] `agent2_nd_migration_data.csv`
- [ ] `agent2_test_results.csv`
- [ ] `agent2_transition_metrics.json`
- [ ] `agent2_fig1_timeseries_with_vintages.png`
- [ ] `agent2_fig2_variance_by_vintage.png`
- [ ] `agent2_fig3_structural_breaks.png`
- [ ] `agent2_calculations.md` or `.ipynb`

### Agent 3 Must Deliver:
- [ ] `AGENT_3_REPORT.md` (using REPORT_TEMPLATE.md)
- [ ] `agent3_findings_summary.json`
- [ ] `agent3_sources.json` or `.bib`
- [ ] `agent3_external_correlations.csv`
- [ ] `agent3_state_comparison.csv`
- [ ] `agent3_validation_data.csv`
- [ ] `agent3_coherence_checks.json`

### Synthesis Must Deliver:
- [ ] Completed `CHATGPT_BRIEFING.md` (from template)
- [ ] `synthesis_findings_matrix.csv`
- [ ] `synthesis_recommendations.json`

---

## Quality Checklist

Before delivery, each agent should verify:

- [ ] All JSON files are valid (parseable)
- [ ] All CSV files have headers matching specification
- [ ] All visualizations are legible and properly labeled
- [ ] Report follows template structure
- [ ] Findings summary JSON is complete
- [ ] All sources are properly cited
- [ ] Uncertainty is explicitly quantified
- [ ] Review requests are clearly articulated

---

*End of Artifact Specifications*
