# Advanced Scite.ai Workflows for Literature Review

This guide outlines advanced workflows for using Scite.ai to bolster literature reviews and verify the interpretation of sources, going beyond basic reference checking.

## 1. Bolstering Literature Reviews (Discovery)

### Sorting by "Supporting" Citations
When searching for key topics (e.g., "Demographic projection methods"):
1.  Enter your search term in Scite.
2.  **Filter/Sort** results by **"Supporting"** citation count (descending).
    *   **Goal**: Identify empirically validated foundational papers.
    *   **Agent Note**: Prioritize papers with high "Supporting" ratios as primary evidence.

### Citation Chaining & Co-citations
To ensure comprehensive coverage of a topic:
1.  **Forward Chaining**: Check papers that cite your "key" reference. Look for "Supporting" badges to find successful replications or newer applications.
2.  **Co-citations**: In the Scite report for a key paper, look for "References" or "Cited With" data.
    *   **Goal**: Find papers frequently cited alongside your known sources (e.g., "If you cite Paper A, you probably should cite Paper B").

### "Table Mode" for Synthesis
Use the Assistant to generate structured data from multiple papers:
1.  **Prompt**: *"Compare the accuracy of cohort-component methods vs. machine learning models in recent migration studies."*
2.  **Action**: Use the "Table" view (if available) or request a structured comparison.
3.  **Custom Extraction**: Ask for specific columns: *"Sample Size"*, *"Time Horizon"*, *"Key Limitation"*.
    *   **Goal**: Rapidly synthesize metadata across 10-20 papers to identify gaps or consensus.

## 2. Verifying Interpretation (Avoiding Hallucinations)

Ensure that your citation actually supports the claim you are making.

### The "Nuance" Check (Citation Statements)
Before citing a paper for a specific claim:
1.  Search for the paper on Scite.
2.  Read the **Citation Statements** (snippets from *other* papers that cite it).
    *   **Validation**: Do other authors summarize the paper's findings the same way you do?
    *   **Correction**: If other authors consistently mention a limitation you missed, update your text to reflect this nuance.

### Assistant Source Verification
When using Scite Assistant to draft summaries or finding confirmation:
1.  **Hover over citations**: In the Assistant's response, hover over the `[1]` citation markers.
2.  **Read the Snippet**: Verify that the highlighted text in the source actually says what the Assistant claims it says.
    *   **Goal**: Prevent AI "hallucinations" where a real paper is cited for a fake finding.
    *   **Rule**: Never accept an AI summary without checking the source snippet.

## 3. Writing with Precision

Use "Contrasting" data to write more robust academic prose.
*   **Weak Phrasing**: *"Smith (2020) showed that migration follows economic trends."*
*   **Strong Phrasing (Scite-informed)**: *"Smith (2020) provides evidence that migration follows economic trends, a finding supported by Jones (2021) but contested by recent work in post-industrial regions (Lee 2023)."*
    *   **Agent Note**: Search for "Contrasting" citations to find the "Lee 2023" counter-examples.
