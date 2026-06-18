# Gemini 3.5 Flash (High) Writing Style: Research + Prose Review of `how-these-projections-work.md`

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-06-18 |
| **Author** | Codex research pass |
| **Target reviewed** | [`docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`](../plans/2026-public-projection-release-handoff/how-these-projections-work.md) |
| **Related reviews** | [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md); [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md); [`2026-06-18-gemini31-pro-high-writing-style-prose-review.md`](./2026-06-18-gemini31-pro-high-writing-style-prose-review.md) |
| **Scope** | Analyze the current Gemini 3.5 Flash model at `thinking_level="high"` for writing-style tells and fit against the public projections explainer. |

This is a reference artifact. It records findings and editorial guidance only. No edits were made to the
target explainer.

---

## 1. Current model status

As of this review, **Gemini 3.5 Flash** is the current public Gemini 3.5 Flash model. The public model ID
is `gemini-3.5-flash`. Google describes it as the most intelligent Flash model for sustained frontier
performance in agentic and coding tasks.

Key public specs and behavior from the Google docs checked here:

- Model ID: `gemini-3.5-flash`.
- Release state: stable / generally available for scaled production use.
- Input context: 1M tokens.
- Max output: 65k tokens.
- Thinking: supported.
- Default thinking effort: `medium`, changed from `high` in Gemini 3 Flash Preview.
- `high` thinking: maximizes the model's ability to think and use tools; Google recommends it for
  complex reasoning, hard math, and difficult code or agent tasks.
- Tools: Gemini 3.5 Flash supports Gemini 3 family tools such as Google Search, grounding with Google
  Maps, File Search, Code Execution, URL Context, function calling, and combined tool use. Computer Use
  is not supported at this moment.
- Google recommends removing custom `temperature`, `top_p`, and `top_k` settings for Gemini 3.x because
  the models' reasoning capabilities are optimized for defaults.

Do not confuse this with Gemini 3.5 Pro. Google's own launch materials say 3.5 Pro was internal and
forthcoming; the public model docs checked here do not list 3.5 Pro as the current public Pro model.

---

## 2. Source-weighted findings

### Official Google sources

Google frames Gemini 3.5 Flash around speed plus action:

- The Gemini 3.5 launch post says 3.5 Flash delivers frontier performance for agents and coding, with
  emphasis on complex long-horizon tasks.
- The 3.5 Flash API guide says it is designed for sustained frontier performance in agentic execution,
  coding, and long-horizon tasks at scale.
- The behavioral-change notes say default effort is now `medium`; `high` is available for the hardest
  reasoning, math, code, and agent tasks.
- Gemini 3.x prompting guidance says the models respond best to concise, direct instructions and may
  over-analyze if prompts carry verbose legacy scaffolding.
- The same guidance says Gemini 3 and 3.1 are less verbose by default and prefer direct, efficient
  answers; conversational tone must be steered explicitly.

Implication for this project: Gemini 3.5 Flash High is likely a fast, structured, action-oriented
reviewer. It can help test whether the explanation works as a sequence. It is less likely to be the best
model for final public-prose voice.

### External writing-style sources

External hands-on notes focus more on Gemini 3.5 Flash than on Pro:

- TechRadar's tests praised Gemini 3.5 Flash for clear rationale, procedural guidance, practical
  planning, and task-specific adaptation.
- MindStudio characterizes Flash as the speed-optimized model in the 3.5 series, intended for
  low-latency agentic workflows, automation pipelines, and real-time applications.
- Some practitioner commentary says Gemini 3.5 Flash can be strong in structured reasoning but uneven in
  open-ended writing quality. This aligns with treating it as a structured reviewer, not a final
  stylist.

Implication for this project: Flash High can be useful for a quick "does this explainer work?" pass. It
should not be allowed to rewrite the document into a procedural checklist or product-style explainer.

---

## 3. Gemini 3.5 Flash High writing tells to watch

These are editorial risk signals, not proof of authorship.

1. **Action-first structure.** Flash is marketed for agents, coding, and long-horizon tasks. Its prose can
   favor doing, sequencing, and execution over reflective essay movement.
2. **Fast procedural clarity.** The model is good at converting a task into steps. In public prose, that
   can sound like a guide, checklist, or onboarding script.
3. **Direct efficient answer shape.** Gemini 3.x is documented as less verbose by default and more direct.
   That can help clarity but flatten the trade-book voice.
4. **Tool-loop vocabulary.** "Task," "workflow," "execution," "scale," "tool," "function," "agent," and
   "iteration" are natural to the model's product lane. These terms would be wrong for this public
   explainer unless they appear as domain facts.
5. **Rationale labels.** Hands-on reviews note that Gemini often articulates why it chose an approach.
   This can over-label significance in prose.
6. **Parallel-decomposition framing.** Flash can split complex tasks into simultaneous tracks. In prose,
   this may surface as too much explicit sectioning or "component A / component B / component C"
   symmetry.
7. **High-thinking overreach.** `thinking_level="high"` can make Flash more careful and tool-capable, but
   it may also add explanatory bulk or qualifications that the public piece does not need.
8. **Shared LLM surface tells.** Corrective pivots, tidy tricolons, em-dash cadence, polished closers,
   and significance signposts still apply.

---

## 4. Target-document measurements

Quick mechanical pass over `how-these-projections-work.md`:

| Signal | Measurement | Interpretation |
|--------|-------------|----------------|
| Word count | ~2,022 words by simple token count | A compact public explainer. |
| Sentences | 66 | Enough length for rhythm signals to matter. |
| Average sentence length | 21.8 words | Moderate-long explanatory prose. |
| Sentences of 18-24 words | 14 / 66 (21.2%) | Good: not metronomic. |
| Em dashes | 37 | High; shared LLM surface tell. |
| `not` occurrences | 19 | High; supports contrast-pivot concern. |
| `rather than` occurrences | 5 | High enough to be audible. |
| `Technical note` boxes | 5 | Good design; can resemble structured lesson-plan prose. |
| `This is` occurrences | 5 | Mild signpost density. |
| Workflow/product terms | Almost absent | Good: avoids the strongest Flash product-register tell. |

The target is not Flash-like in vocabulary. It is not action/product prose. Its closest Flash-shaped
feature is the clean tutorial sequence: the reader is walked through a system component by component,
with technical-note boxes and a final procedural recap.

---

## 5. Do Gemini 3.5 Flash tells appear in the target?

### Present: procedural clarity

The strongest overlap is the method explanation:

- Lines 68-70 introduce the three things that change a population.
- Survival, fertility, and migration are handled as sequential components.
- Lines 116-118 summarize the annual cycle.
- Lines 192-197 recap the method as a sequence of operations.

This is necessary for a plain-language methods companion. The risk is not the sequence. The risk is that
a Flash High rewrite would likely make the main narrative more procedural and less essayistic.

### Present: rationale labels

Examples:

- "That is the whole idea..."
- "The cohort-component method works precisely because..."
- "This is why..."
- "That early dip is worth dwelling on..."

These are useful, but a Flash High revision might add more labels of this kind. The better edit is to
let examples and facts do more work.

### Mildly present: component symmetry

The survival/fertility/migration section is intentionally symmetrical. It helps readers. It also creates
some model-like regularity: heading, explanation, technical note; heading, explanation, technical note.
Keep the architecture, but vary the local sentence endings and transitions.

### Mostly absent: action/workflow register

The target avoids terms like workflow, agent, execution, scale, tool, and pipeline. That is the clearest
evidence that it does not read like a Gemini 3.5 Flash product-style output.

### Mostly absent: direct-efficient flattening

The target has a reserved essay voice with concrete images: a trend line, a ladder of age, a quiet
engine, comings and goings. A Flash rewrite should be constrained not to remove those.

---

## 6. Editorial implications for `how-these-projections-work.md`

1. **Use Flash High sparingly for prose.** It can check whether the explanation is followable, but it is
   likely to make a rewrite more procedural.
2. **Ask for diagnosis, not rewrite.** Better prompt: "Identify places where the procedural sequence is
   unclear." Riskier prompt: "Improve the prose."
3. **Forbid product-register vocabulary.** If Flash is used, explicitly avoid "workflow," "agent,"
   "execution," "tool," "scale," "dashboard," and similar terms unless present in the source.
4. **Protect the metaphors.** The target's least model-like elements are concrete and domain-grounded.
   Do not let a direct-efficiency pass remove them.
5. **Keep the earlier edits.** Reduce em dashes, thin contrast pivots, and vary section closers. Those
   shared tells remain more important than anything Flash-specific.

---

## 7. Model routing for this task

Best uses for Gemini 3.5 Flash High:

- Fast public-reader comprehension review.
- Checking whether the method sequence is clear.
- Identifying missing transitions in survival/fertility/migration explanation.
- Running a quick second opinion when cost/latency matters.

Weaknesses for this task:

- May turn the piece into a lesson plan.
- May flatten the SOP-005 trade-book voice.
- May add action/workflow register.
- May over-label why each point matters.

Recommended route:

1. Use **Gemini 3.5 Flash `high`** for fast clarity diagnostics on the method sequence.
2. Use **Gemini 3.1 Pro `high`** for broader source-grounded consistency across the full handoff package.
3. Use **GPT-5.5 `xhigh`** for structure and reader-comprehension review.
4. Use **Claude Opus 4.8 `max` or `xhigh`** for final cadence, taste, and voice.

Bottom line: Gemini 3.5 Flash High is useful as a **fast procedural clarity reviewer**. It is not the
right final prose stylist for this explainer.

---

## 8. Sources consulted

Official Google:

- [What's new in Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/interactions/whats-new-gemini-3.5)
- [Gemini 3.5 launch post](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/)
- [Gemini 3 developer guide](https://ai.google.dev/gemini-api/docs/gemini-3)
- [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [Gemini API models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini 3.1 Pro Preview model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)

External commentary and reviews:

- [TechRadar - 5 prompts that show how the new Gemini 3.5 Flash is its best AI model yet](https://www.techradar.com/ai-platforms-assistants/gemini/5-prompts-that-show-how-the-new-gemini-3-5-flash-is-its-best-ai-model-yet)
- [MindStudio - What Is Google Gemini 3.5 Flash?](https://www.mindstudio.ai/blog/what-is-gemini-3-5-flash-4)
- [MindStudio - Gemini 3.5 Flash vs Gemini 3.1 Pro](https://www.mindstudio.ai/blog/gemini-3-5-flash-vs-gemini-3-1-pro-comparison)

Related local references:

- [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md)
- [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md)
- [`2026-06-18-gemini31-pro-high-writing-style-prose-review.md`](./2026-06-18-gemini31-pro-high-writing-style-prose-review.md)
- [`SOP-005-public-facing-prose-voice.md`](../governance/sops/SOP-005-public-facing-prose-voice.md)

---

## 9. Source caveats

1. This file analyzes Gemini 3.5 Flash, not Gemini 3.5 Pro.
2. `thinking_level="high"` is a reasoning/tool-use depth control, not a prose-quality control.
3. External model comparisons are useful for practitioner signal, not definitive writing benchmarks.
4. AI-writing detection is probabilistic. These tells are editorial heuristics, not evidence of
   authorship.
