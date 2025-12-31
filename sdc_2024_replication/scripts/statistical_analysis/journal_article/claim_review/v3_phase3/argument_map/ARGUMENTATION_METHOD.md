# Argumentation Method (Why / What / How)

## Why
We need to verify not only individual claims but also the logical structure that connects them. Argument mapping exposes missing warrants, unsupported leaps, and weak evidence chains that a claim-only review can miss.

## What
We map the paper's reasoning using a Toulmin-style model with explicit nodes and links:
- Claim, Grounds, Warrant, Backing, Qualifier, Rebuttal.
Each node is stored as a structured JSONL record and linked to claim IDs in the claim manifest.

## How
1. Build the claim manifest first.
2. Create argument nodes that reference claim IDs and group them by `argument_group_id`.
3. Link nodes with `supports_argument_ids` and `rebuts_argument_ids`.
4. Export Graphviz DOT files for per-argument and full-paper visualization.
5. Use the argument map to prioritize verification and revisions.

## Reuse
This method is designed to be reused for future papers by copying the argument_map folder structure, adapting the claim manifest, and regenerating graphs from JSONL.
