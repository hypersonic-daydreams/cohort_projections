# Argument Map Workspace

Purpose: Map the logical structure of the paper using a Toulmin-style model and link each argument element to claim IDs.

Canonical data file: `argument_map.jsonl`

## Structure
- `argument_schema.json` - JSON Schema for argument map records
- `argument_guidelines.md` - Mapping rules and role definitions
- `ARGUMENTATION_METHOD.md` - Rationale and reusable workflow description
- `graphs/` - Graphviz DOT files (per-argument and full-paper)

## Outputs
- `argument_map.jsonl` - Nodes representing claims/grounds/warrants/backing/qualifiers/rebuttals
- `graphs/argument_graph_all.dot` - Full-paper argument network
- `graphs/argument_graph_template.dot` - Template for per-argument graphs

## Graph Generation
Use `generate_argument_graphs.py` to create Graphviz DOT outputs.

Example:
```bash
source .venv/bin/activate
python sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/argument_map/generate_argument_graphs.py --per-group
```

Notes:
- Default input: `argument_map.jsonl` in this folder.
- Default output: `graphs/argument_graph_all.dot` plus per-group graphs when `--per-group` is set.
