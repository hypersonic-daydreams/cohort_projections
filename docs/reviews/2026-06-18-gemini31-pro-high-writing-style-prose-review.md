# Gemini 3.1 Pro (High) Writing Style: Research + Prose Review of `how-these-projections-work.md`

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-06-18 |
| **Author** | Codex research pass |
| **Target reviewed** | [`docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`](../plans/2026-public-projection-release-handoff/how-these-projections-work.md) |
| **Related reviews** | [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md); [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md); [`2026-06-18-gemini35-flash-high-writing-style-prose-review.md`](./2026-06-18-gemini35-flash-high-writing-style-prose-review.md) |
| **Scope** | Analyze the current public Gemini Pro-class text model, Gemini 3.1 Pro Preview at `thinking_level="high"`, for writing-style tells and fit against the public projections explainer. |

This is a reference artifact. It records findings and editorial guidance only. No edits were made to the
target explainer.

---

## 1. Current model status

As of this review, the current public Gemini Pro-class text model in the Google AI developer docs is
**Gemini 3.1 Pro Preview**, model ID `gemini-3.1-pro-preview`. Google's Gemini 3 developer guide says
Gemini 3.1 Pro is best for complex tasks requiring broad world knowledge and advanced reasoning across
modalities. The dedicated model page describes Gemini 3.1 Pro Preview as improving thinking, token
efficiency, groundedness, and fact consistency, with optimization for software engineering behavior,
agentic workflows, precise tool use, and reliable multi-step execution.

Google has announced Gemini 3.5 Pro as forthcoming/internal, but the sources checked for this review did
not provide a public Gemini 3.5 Pro model page or model ID. For this repo's documentation, do not refer
to Gemini 3.5 Pro as the current public Pro model unless a later source verifies that release.

Key public specs for `gemini-3.1-pro-preview`:

- Input types: text, image, video, audio, PDF.
- Output: text.
- Input token limit: 1,048,576.
- Output token limit: 65,536.
- Thinking: supported.
- Tools: code execution, function calling, search grounding, URL context, structured outputs, and other
  Gemini tool features are supported.
- Published knowledge cutoff in the model page checked here: January 2025.

For `high`, the relevant control is Gemini 3's `thinking_level`. Google describes `high` as maximizing
reasoning depth; it may take longer before the first visible token.

---

## 2. Source-weighted findings

### Official Google sources

Google's official descriptions place Gemini 3.1 Pro in the "broad reasoning over large multimodal
context" lane rather than the "final prose stylist" lane:

- Gemini 3.1 Pro is framed as a complex-task model with broad world knowledge and advanced reasoning
  across modalities.
- The model page emphasizes better thinking, token efficiency, groundedness, fact consistency, software
  engineering behavior, and reliable multi-step execution.
- Gemini 3.x prompting guidance says to be concise and direct. It warns that verbose or older
  chain-of-thought-style prompt engineering can make the model over-analyze.
- Google says Gemini 3 and 3.1 are less verbose by default and prefer direct, efficient answers; if a
  conversational tone is needed, the prompt should explicitly steer it.

Implication for this project: Gemini 3.1 Pro High is likely strongest as a source-grounded consistency
and synthesis model. It should be prompted carefully if asked to revise public prose, because its default
mode may become direct, efficient, and explanatory rather than voice-sensitive.

### External writing-style sources

External comparisons remain mixed, but they converge on a useful routing rule:

- Gemini 3.1 Pro is usually praised for context length, multimodal reasoning, technical depth, and
  structured conceptual explanation.
- Claude is still commonly favored for pure writing quality, tone consistency, and nuanced prose.
- Gemini writing can be strong analytically, but reviewers often describe it as less consistently
  natural for voice-sensitive long-form prose than Claude.

Implication for this project: Gemini 3.1 Pro High should be treated as a careful reviewer and
source-synthesis model, not as the default final line editor for SOP-005 prose.

---

## 3. Gemini 3.1 Pro High writing tells to watch

These are editorial risk signals, not proof of authorship.

1. **Broad-context synthesis voice.** Gemini 3.1 Pro is built to absorb large, multimodal inputs. Its
   prose can sound like a synthesis report: accurate, organized, and slightly detached.
2. **Groundedness scaffolding.** Because fact consistency is a model-strength claim, outputs may lean
   toward explicit source framing, qualification, and evidence labels.
3. **Direct efficient explanation.** Gemini 3.x guidance says the models prefer direct, efficient answers.
   In narrative prose, that can strip out texture unless explicitly protected.
4. **High-thinking over-analysis.** `thinking_level="high"` can improve reasoning but can also encourage
   more careful caveats, rationale labels, and exhaustive explanation.
5. **Structured decomposition.** Gemini 3.1 Pro is strong at breaking a complex problem into parts. The
   tell is a document that feels like a well-designed lesson plan or technical synthesis rather than a
   human essay.
6. **Tool/workflow register.** The model page emphasizes multi-step execution, software engineering, tool
   use, and agentic workflows. That operational register can bleed into prose if not constrained.
7. **Multimodal/dashboard instinct.** Gemini often thinks naturally in terms of charts, PDFs, interfaces,
   simulations, and visual organization. Useful for QA packages, less useful for a trade-book register.
8. **Shared LLM surface tells.** Corrective pivots, tidy tricolons, em-dash cadence, polished closers,
   and significance signposts still apply. They are not unique to GPT or Claude.

---

## 4. Target-document measurements

Quick mechanical pass over `how-these-projections-work.md`:

| Signal | Measurement | Interpretation |
|--------|-------------|----------------|
| Word count | ~2,022 words by simple token count | A compact public explainer. |
| Sentences | 66 | Enough length for rhythm signals to matter. |
| Average sentence length | 21.8 words | Moderate-long explanatory prose. |
| Sentences of 18-24 words | 14 / 66 (21.2%) | Good: not metronomic. |
| Em dashes | 37 | High; a shared LLM surface tell. |
| `not` occurrences | 19 | High; supports the contrast-pivot concern. |
| `rather than` occurrences | 5 | High enough to be audible. |
| `Technical note` boxes | 5 | Strong public-methods architecture; also resembles structured synthesis. |
| Action/workflow terms | Almost absent | Good: the document does not sound like an agentic workflow memo. |
| Obvious slop vocabulary | 0 hits for common terms searched | Good: vocabulary is clean. |

The same core risk remains: the target is not lexically generic, but it is structurally very polished.
From a Gemini 3.1 Pro lens, the closest tell is **structured synthesis**: the document is organized into
clean explanatory modules, with technical detail boxed out and each section performing a clear job.

---

## 5. Do Gemini 3.1 Pro tells appear in the target?

### Present: structured synthesis

The explainer's architecture is highly ordered:

- A conceptual opening.
- A rejection of the trend-line shortcut.
- A three-component method explanation.
- A public-assumption section.
- A county-divergence section.
- A reliability section.
- A final procedure-like recap.

That structure is correct for the subject. The editorial risk is that every section feels too perfectly
placed. A Gemini 3.1 Pro High rewrite might preserve this structure but intensify the synthesis-report
feel.

### Present: rationale signposting

Examples:

- "That is the whole idea..."
- "The cohort-component method works precisely because..."
- "This is why the projected path..."
- "That early dip is worth dwelling on..."
- "If there is one point worth carrying away..."

These signposts help readers, but they also resemble the model habit of explaining why the explanation is
important. A human line edit can trust the examples and numbers to carry more of the emphasis.

### Mildly present: caveat architecture

The reliability section is cautious and accurate. It should remain cautious. The risk is not the current
text; the risk is that a Gemini 3.1 Pro High revision pass might add additional qualifiers because the
model is optimizing for groundedness.

### Mostly absent: operational register

The target does not use workflow, agent, execution, scale, tool, pipeline, or dashboard language. That is
good. It means the document avoids the most obvious Gemini product-register bleed.

### Mostly absent: direct-efficient flattening

The target still has an essay voice. It uses metaphor and rhythm, not only efficient explanation. That is
good and should be protected if Gemini is used for review.

---

## 6. Editorial implications for `how-these-projections-work.md`

1. **Use Gemini 3.1 Pro High for consistency, not voice.** It is a good model for checking whether the
   explainer agrees with the PDF copy, methodology, and locked public data package.
2. **Prompt against caveat creep.** If used for revision, explicitly forbid adding new qualifications
   unless they replace existing text or identify a concrete factual inconsistency.
3. **Protect the essay register.** Tell the model to preserve SOP-005 voice and avoid turning the piece
   into a technical memo, FAQ, checklist, or lesson plan.
4. **Keep the technical-note boxes.** They are good public-methods design. The edit should thin repeated
   rhetorical devices, not remove the architecture.
5. **Use source-grounded checks after any human/Claude line edit.** Gemini 3.1 Pro High is well suited to
   verifying that numbers, dates, caveats, and method steps still reconcile.

---

## 7. Model routing for this task

Best uses for Gemini 3.1 Pro High:

- Full-package consistency review across the explainer, public PDF copy, methodology notes, ADR-042
  terminology, and locked public outputs.
- Finding missing or inconsistent assumptions.
- Checking whether a public-reader explanation omits a necessary method step.
- Producing a structured revision checklist.

Weaknesses for this task:

- May over-explain.
- May add caveats.
- May flatten voice into direct efficient synthesis.
- May preserve structure at the cost of human cadence.

Recommended route:

1. Use **Claude Opus 4.8 `max` or `xhigh`** for the final voice-sensitive line edit.
2. Use **GPT-5.5 `xhigh`** for structure, reader comprehension, and constraint retention.
3. Use **Gemini 3.1 Pro `high`** after edits as a broad-context source-grounding and consistency check.
4. Keep a human final pass for taste and numerical verification.

Bottom line: Gemini 3.1 Pro High is likely the strongest current Gemini choice for **large-context
review and grounded synthesis**. It is not the strongest choice for final public-prose cadence.

---

## 8. Sources consulted

Official Google:

- [Gemini 3.1 Pro Preview model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)
- [Gemini 3 developer guide](https://ai.google.dev/gemini-api/docs/gemini-3)
- [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [Gemini API models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini 3.5 launch post](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/)
- [What's new in Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/interactions/whats-new-gemini-3.5)

External commentary and reviews:

- [Vellum - Is Claude Better Than Gemini?](https://www.vellum.ai/blog/is-claude-better-than-gemini)
- [Tactiq - Claude vs ChatGPT vs Gemini for writing](https://tactiq.io/learn/claude-vs-gemini-vs-chatgpt-for-writing)
- [MindStudio - GPT-5.4 vs Claude Opus 4.6 vs Gemini 3.1 Pro](https://www.mindstudio.ai/blog/gpt-54-vs-claude-opus-46-vs-gemini-31-pro-benchmarks)

Related local references:

- [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md)
- [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md)
- [`2026-06-18-gemini35-flash-high-writing-style-prose-review.md`](./2026-06-18-gemini35-flash-high-writing-style-prose-review.md)
- [`SOP-005-public-facing-prose-voice.md`](../governance/sops/SOP-005-public-facing-prose-voice.md)

---

## 9. Source caveats

1. The current public model ID is `gemini-3.1-pro-preview`; it is a preview model, not a stable named
   Gemini 3.5 Pro release.
2. `thinking_level="high"` is a reasoning-depth control, not a prose-quality control.
3. External model comparisons are useful for practitioner signal, not definitive writing benchmarks.
4. AI-writing detection is probabilistic. These tells are editorial heuristics, not evidence of
   authorship.
