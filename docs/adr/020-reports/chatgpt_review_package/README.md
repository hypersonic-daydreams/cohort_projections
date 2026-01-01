# ChatGPT 5.2 Pro Review Package

This folder contains all materials needed for external review by ChatGPT 5.2 Pro.

## How to Use

1. **Open `PROMPT_FOR_CHATGPT.md`** and copy the prompt text (everything after "---BEGIN PROMPT---")

2. **Start a new ChatGPT 5.2 Pro conversation** and paste the prompt

3. **Upload the following files** (in order of importance):

### Essential Files (Always Upload)
| File | Description | Size |
|------|-------------|------|
| `CHATGPT_BRIEFING.md` | Comprehensive context document | 19 KB |
| `AGENT_1_REPORT.md` | Census methodology review | 27 KB |
| `AGENT_2_REPORT.md` | Statistical analysis | 22 KB |
| `AGENT_3_REPORT.md` | Comparability assessment | 23 KB |
| `agent2_nd_migration_data.csv` | Raw ND data (25 rows) | 1 KB |
| `synthesis_findings_matrix.csv` | Cross-agent comparison | 2 KB |

### Recommended Supporting Files
| File | Description |
|------|-------------|
| `agent2_test_results.csv` | Full statistical test results (25 tests) |
| `agent2_fig1_timeseries_with_vintages.png` | Time series visualization |
| `agent2_fig2_variance_by_vintage.png` | Variance comparison chart |

### Optional Detailed Files
| File | Description |
|------|-------------|
| `agent1_methodology_matrix.csv` | Vintage comparison table |
| `agent1_census_quotes.json` | Direct quotes from Census Bureau |
| `agent*_findings_summary.json` | Machine-readable findings |
| `agent3_*.csv` | Validation and correlation data |
| `synthesis_recommendations.json` | Aggregate recommendations |

## The Two Core Questions

The prompt focuses ChatGPT 5.2 Pro on two overarching questions:

1. **Is extension methodologically defensible?**
   - Given Census Bureau warnings and detected level shifts
   - What conditions would make it acceptable for publication?

2. **Which approach should I take?**
   - Option A: Extend with statistical corrections
   - Option B: Extend with caveats only
   - Option C: Hybrid approach (primary n=15, robustness n=25)
   - Option D: Maintain n=15

## Expected Response

ChatGPT 5.2 Pro should provide:
- Executive assessment
- Answers to both primary questions with reasoning
- Validation of agent methodologies
- Alternative interpretations
- Recommended next steps

## After Review

Save ChatGPT 5.2 Pro's response to:
`/docs/adr/020-reports/CHATGPT_RESPONSE.md`

Then use the response to make the final decision on ADR-019.
