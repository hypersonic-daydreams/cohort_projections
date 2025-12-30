# 06 Resubmission Review

## Purpose

This folder contains materials for the second round of AI review after implementing Phase 1 revisions (C01-C10). Two reviewers are used in parallel for broader coverage.

## Reviewers

| Reviewer | Model | Role |
|----------|-------|------|
| ChatGPT 5.2 Pro | OpenAI | Primary reviewer (continuity from Round 1) |
| Gemini 3 Pro "Deep Think" | Google | Secondary reviewer (fresh perspective) |

## Contents

| File | Description |
|------|-------------|
| `prompt.md` | Prompt template to use when submitting to both reviewers |
| `outputs/ChatGPT_5-2-Pro_revision_critique.md` | ChatGPT response (with metadata header) |
| `outputs/Gemini_3_Pro_DeepThink_revision_critique.md` | Gemini response (with metadata header) |

## Workflow

### For Each Reviewer:

1. Open the AI interface (ChatGPT or Gemini)
2. Upload `output/article_draft_v2_revised.pdf`
3. Paste contents of `prompt.md`
4. Copy the complete response
5. Paste into the appropriate `outputs/*.md` file (below the header)
6. Fill in `response_date` in the YAML header

## Version History

| Version | File | Date | Description |
|---------|------|------|-------------|
| v1 | `output/article_draft.pdf` | 2025-12-29 | Original draft submitted for initial review |
| v2 | `output/article_draft_v2_revised.pdf` | 2025-12-30 | Revised draft after C01-C10 implementation |

## Related Files

- Initial critique: `output/ChatGPT_5-2-Pro_article_draft_critique.md`
- Revision plan: `HYBRID_REVISION_PLAN.md`
- Status tracker: `REVISION_STATUS.md`

## After Both Reviews Complete

1. Compare feedback from both reviewers
2. Identify consensus issues (high priority)
3. Identify divergent opinions (requires judgment)
4. Update `REVISION_STATUS.md` with Phase 3 tasks if needed
