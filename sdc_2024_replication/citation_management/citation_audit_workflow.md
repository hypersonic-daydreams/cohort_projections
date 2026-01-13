# Citation Audit Workflow

This workflow describes how to perform a structured audit of citations in the user's paper using Scite.ai and the local JSON tracking system.

## Overview

1.  **Extract Claim**: Locate a citation in the PDF draft.
2.  **Verify**: Use the Scite Assistant to verify if the paper supports the specific claim.
3.  **Track**: Record the finding in `citation_audit.json` using the `audit_manager.py` script.

## Setup
Ensure the script is executable:
```bash
chmod +x citation_management/tracking/audit_manager.py
```

## Batch Manual Verification Workflow

This workflow optimizes for agent-user collaboration: the User performs the high-value manual verification in Scite, and the Agent handles the data entry and tracking.

### 1. Workspace Preparation (Agent)
The Agent creates a file `citation_management/scite_workspace.md` containing pre-generated prompts for all pending citations.

### 2. Batch Execution (User)
The User iterates through the workspace file:
1.  **Copy Prompt**: Copy the prompt for a citation.
2.  **Run in Scite**: Paste into [Scite Assistant](https://scite.ai/assistant).
3.  **Paste Results**: Copy the *entire* raw text response from Scite and paste it back into the `Results` block in `scite_workspace.md`.

### 3. Data Ingestion (Agent)
Once the user completes the batch:
1.  The Agent reads `scite_workspace.md`.
2.  The Agent parses the raw text to extract:
    *   Status (Verified, Nuanced, Contested)
    *   Metrics (Supporting, Mentioning, Contrasting)
    *   Key claim confirmation
3.  The Agent updates `citation_audit.json` in a single operation.
4.  The Agent generates the final report.

### Error Handling
If a result is unclear or a paper is not found, the user should note this in the `Results` block (e.g., "Paper not found" or "Ambiguous"). The agent will mark these as `Pending` or `Review Required`.
