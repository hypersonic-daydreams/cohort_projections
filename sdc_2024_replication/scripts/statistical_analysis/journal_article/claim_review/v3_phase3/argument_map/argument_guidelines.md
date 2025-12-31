# Argument Mapping Guidelines

## Framework
Use a Toulmin-style model:
- Claim: main assertion being advanced.
- Grounds: data or observations offered in support.
- Warrant: inferential link explaining why grounds support the claim.
- Backing: additional support for the warrant.
- Qualifier: scope/strength/conditions of the claim.
- Rebuttal: stated exceptions or counterpoints.

## Linking Rules
- Every argument node should link to at least one `claim_id` when possible.
- If a warrant is implicit but necessary, write it as its own node.
- Group related nodes under a shared `argument_group_id`.
- Use `supports_argument_ids` to connect grounds/warrants/backing to a claim node.
- Use `rebuts_argument_ids` for rebuttals that challenge a claim or warrant.

## Granularity
- Prefer multiple small nodes over one overloaded node.
- Each figure/table caption can be a grounds node if it provides evidence.

## Status Workflow
- `draft`: initial mapping.
- `reviewed`: second pass confirms roles and links.
- `reconciled`: disagreements resolved and stable.

## Graph Conventions
- Claim nodes: box
- Grounds nodes: ellipse
- Warrant nodes: diamond
- Backing nodes: hexagon
- Qualifier nodes: note
- Rebuttal nodes: octagon

These shapes are defined in `graphs/argument_graph_template.dot`.
