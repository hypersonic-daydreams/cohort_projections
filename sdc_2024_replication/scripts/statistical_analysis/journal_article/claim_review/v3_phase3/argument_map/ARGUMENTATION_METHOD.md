# Argumentation Method (Why / What / How)

## Why
We need to verify not only individual claims but also the logical structure that connects them. Argument mapping exposes missing warrants, unsupported leaps, and weak evidence chains that a claim-only review can miss.

## What
We map the paper's reasoning using a Toulmin-style model with explicit nodes and links:
- **Claim**: Central assertion being argued
- **Grounds**: Evidence that supports the claim (data, statistics, observations)
- **Warrant**: Logical bridge explaining WHY the grounds support the claim
- **Backing**: Additional support for warrants (optional)
- **Qualifier**: Scope limitations (optional)
- **Rebuttal**: Counter-arguments or exceptions (optional)

Each node is stored as a structured JSONL record and linked to claim IDs in the claim manifest.

## Minimum Requirements
**Every argument group MUST have at minimum: Claim + Grounds + Warrant**

This is because:
- Data alone (grounds) is not an argument
- An assertion (claim) without evidence is unsupported
- Evidence without reasoning (warrant) doesn't explain the logical connection

## How
1. Build the claim manifest first.
2. For each key assertion, create an argument group:
   a. Identify the central CLAIM
   b. Find GROUNDS (which claims provide evidence?)
   c. Identify or construct the WARRANT (why does the evidence support the claim?)
   d. Add BACKING, QUALIFIER, REBUTTAL as appropriate
3. Link nodes with `supports_argument_ids` and `rebuts_argument_ids`.
4. Run the audit to verify completeness: `python map_section_arguments.py --audit`
5. Export visualizations: `python generate_graphs.py` and `python build_viewer.py`

## Tools
```bash
# Audit for completeness (must pass before done)
python map_section_arguments.py --audit

# Get mapping guidance for a section
python map_section_arguments.py --section Results

# Get mapping guidance for a specific group
python map_section_arguments.py --group G020

# Generate DOT files
python generate_graphs.py

# Generate interactive HTML viewer
python build_viewer.py
```

## Reuse
This method is designed to be reused for future papers by copying the argument_map folder structure, adapting the claim manifest, and regenerating graphs from JSONL.
