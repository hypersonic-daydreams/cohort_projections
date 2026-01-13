# Web Search Verification Workflow

This workflow describes the "fast path" verification process using general web search engines to validate citation claims. This process serves as a rapid calibration step to be compared against the deep-dive Scite.ai audit.

## Overview

1.  **Extract Claim**: Locate a citation in the manuscript.
2.  **Search**: Perform a targeted web search using the author, year, and key phrases from the claim.
3.  **Verify**: Determine if the claim is supported by the abstract, title, or snippets available in search results.
4.  **Track**: Record the finding in `citation_audit.json` with `method="Web Search"`.

## Step-by-Step

### 1. Identify the Claim
*   **Source**: Manuscript text (PDF or extracted text).
*   **Example**: "Refugee resettlement is a key driver of population growth (Smith 2020)."

### 2. Targeted Search
*   **Query Format**: `"[Author]" [Year] "[Key Phrase from Claim]" [Context keywords]`
*   **Example**: `"Smith" 2020 "refugee resettlement" population growth`

### 3. Determine Status
*   **Verified**: Search results (abstracts/snippets) explicitly confirm the claim.
*   **Pending**: Search finds the paper but the specific claim is not visible in the abstract/snippet (requires full text).
*   **Not Found**: Paper or claim cannot be located.

### 4. Record Result
Use `audit_manager.py` with the `--method` flag (once implemented):

```bash
./citation_management/tracking/audit_manager.py add \
  --key "Smith2020" \
  --method "Web Search" \
  --title "Refugee Resettlement Impacts" \
  --claim "Refugee resettlement drives growth" \
  --status "Verified" \
  --notes "Confirmed via abstract on Publisher Site."
```
