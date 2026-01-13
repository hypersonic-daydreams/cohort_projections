# Scite.ai Citation Verification Guide

This guide documents the process for using Scite.ai to verify the veracity and accuracy of citations in our manuscripts, leveraging the premium subscription features.

## 1. Reference Check Tool (Primary workflow)

The **Reference Check** is the most direct way to screen a draft manuscript (PDF or DOCX) for citation issues.

### How to use:
1.  **Log in** to [scite.ai](https://scite.ai).
2.  Navigate to **Product** in the top menu and select **Reference Check**.
3.  **Upload the manuscript**: Upload the PDF version of the draft (e.g., `article-0.9-production...pdf`).
4.  **Review the Report**: Scite will generate a report listing all detected references.

### Interpreting the Results:
*   **Retractions & Editorial Concerns**: The report will immediately flag references that have been retracted, have an expression of concern, or have been withdrawn. **Action**: These must be investigated and likely removed or explicitly noted as retracted in the text.
*   **Smart Citations Breakdown**: each reference shows a tally of:
    *   **Supporting** (Green): Citing papers that provide confirming evidence.
    *   **Contrasting** (Blue): Citing papers that dispute or fail to replicate the findings.
    *   **Mentioning** (Grey): Neutral citations.

**Action Item**: Pay close attention to references with high **Contrasting** counts. Ensure that our use of these sources acknowledges the academic disagreement if it exists.

## 2. Smart Citations Classification

Scite classifies citation statements (not just the link) to determine the context:

*   **Supporting**: "We replicated the findings of Smith et al..."
*   **Contrasting**: "In contrast to Smith et al, we found..."
*   **Mentioning**: "Smith et al previously studied this phenomenon..."

## 3. Investigating Specific Papers

If the Reference Check flags a paper as controversial (high contrasting citations), use these tools to dig deeper:

*   **Scite Assistant**:
    *   Use the Chat interface to ask: *"What are the main criticisms of [Paper Title]?"*
    *   ask: *"Has [Paper Title] been replicated?"*
    *   The Assistant uses full-text access to answer with evidence.
*   **Citation Statement Search**:
    *   Search for the paper title to see a list of snippets showing exactly *how* other authors have successfully cited it. This is useful for finding the specific phrasing of critiques.

## Summary Checklist for Draft Reviews
- [ ] Run **Reference Check** on the final PDF draft.
- [ ] Verify no **Retracted** papers are cited without context.
- [ ] Review any papers with >0 **Contrasting** citations to ensure our usage is robust.
- [ ] Use **Assistant** to clarify any ambiguous "Contrasting" signals.
