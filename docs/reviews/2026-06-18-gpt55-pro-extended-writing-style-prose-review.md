# GPT-5.5 Pro Extended Writing Style: Research + Prose Review of `how-these-projections-work.md`

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-06-18 |
| **Author** | Codex research pass |
| **Target reviewed** | [`docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`](../plans/2026-public-projection-release-handoff/how-these-projections-work.md) |
| **Related reviews** | [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md); [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md); [`2026-06-18-gemini31-pro-high-writing-style-prose-review.md`](./2026-06-18-gemini31-pro-high-writing-style-prose-review.md); [`2026-06-18-gemini35-flash-high-writing-style-prose-review.md`](./2026-06-18-gemini35-flash-high-writing-style-prose-review.md) |
| **Scope** | Analyze ChatGPT's GPT-5.5 Pro with the web UI thinking-time setting at **Extended**, focusing on writing-style tells and fit against the public projections explainer. |

This is a reference artifact. It records findings and editorial guidance only. No edits were made to the
target explainer.

---

## 1. Current model and surface status

As of this review, OpenAI's ChatGPT help page describes **GPT-5.5 Pro** as the highest-capability
GPT-5.5 option in ChatGPT for the hardest tasks and long-running workflows. In the model picker, it is
the **Pro** option, described as research-grade intelligence. GPT-5.5 Pro is available on Pro, Business,
Enterprise, and Edu plans.

The web UI setting needs careful wording:

- ChatGPT lets users set thinking effort when **Thinking** or **Pro** is selected.
- The web message composer exposes thinking-time labels. The documented labels are **Light**,
  **Standard**, **Extended**, and **Heavy**.
- **Standard** is the default. **Extended** gives more time for deeper, more comprehensive responses.
- Pro users get two additional options, **Light** and **Heavy**. OpenAI describes Heavy as deeper
  reasoning.
- The setting is available on ChatGPT Web and does not sync to mobile.
- The selected thinking time is saved for later web queries until changed.

Important caveat: **Extended in the ChatGPT web UI is not publicly documented as a one-to-one alias for
API `reasoning.effort="xhigh"`**. The API uses `reasoning.effort` values such as `low`, `medium`,
`high`, and `xhigh`; ChatGPT Web uses product labels such as Standard, Extended, and Heavy. Treat them
as related concepts, not interchangeable names.

Other relevant official facts:

- GPT-5.5 Pro in ChatGPT may begin with a short preamble before reasoning starts.
- While GPT-5.5 Thinking or Pro is still reasoning, the user can add instructions to steer the response
  before it finishes.
- Apps, Memory, Canvas, and image generation are not available with Pro in ChatGPT.
- The API model page for `gpt-5.5-pro` lists a 1,050,000-token context window, 128,000 max output
  tokens, a December 1, 2025 knowledge cutoff, and reasoning token support.
- OpenAI warns that GPT-5.5 Pro is designed for tough problems and some API requests may take several
  minutes.

Implication for this review: GPT-5.5 Pro Extended should be understood as a **deep ChatGPT review
setting**, not simply "GPT-5.5 xhigh under another name." It is likely most useful when the prose task
requires broad context, factual cross-checking, and careful critique. It may be less useful when the only
goal is a light, voice-sensitive line edit.

---

## 2. Source-weighted findings

### Official OpenAI sources

OpenAI frames GPT-5.5 Pro around difficult, long-running, professional work:

- The launch post says GPT-5.5 is especially strong in agentic coding, computer use, knowledge work,
  and scientific research.
- For ChatGPT, OpenAI says GPT-5.5 Thinking is useful for coding, research, information synthesis,
  analysis, and document-heavy tasks.
- The same post says early GPT-5.5 Pro testers found outputs more comprehensive, well-structured,
  accurate, relevant, and useful than GPT-5.4 Pro, especially in business, legal, education, and data
  science.
- OpenAI describes GPT-5.5 Pro research use as multi-pass work: critiquing manuscripts, stress-testing
  technical arguments, proposing analyses, and working with code, notes, and PDF context.
- The reasoning-model guide says reasoning models use internal reasoning tokens to plan, inspect
  alternatives, recover from ambiguity, and solve harder multi-step tasks.

Implication for this project: GPT-5.5 Pro Extended is probably best treated as a **senior reviewer**:
give it the explainer, the methodology constraints, ADR-042 terminology, locked public numbers, and the
existing prose-review findings; ask it to find factual drift, overclaims, missing caveats, and repeated
AI-shaped rhetorical devices. Do not let it freely rewrite the whole piece unless the prompt tightly
protects SOP-005 voice.

### Practitioner signal

External commentary on GPT-5.5 generally converges on a similar shape, with lower evidentiary weight
than official sources:

- It is often praised for clear explanation, orderly transitions, and broad-audience readability.
- It is often described as strong at complex professional work, especially when the task benefits from
  planning, tool use, or long-context synthesis.
- Some user reports suggest Pro can be less satisfying for pure prose style than for analysis, because
  its answers may become very comprehensive or one-line-per-thought unless steered.

Implication for this project: the same quality that makes GPT-5.5 Pro useful for review can become a
writing tell. It may over-complete the assignment: more context, more caveats, more sections, more
explicit reasoning, and more polished closure than the public explainer needs.

---

## 3. GPT-5.5 Pro Extended writing tells to watch

These are editorial risk signals, not proof of authorship.

1. **Research-partner posture.** The model may write like a careful research collaborator: identifying
   assumptions, stress-testing claims, and naming the purpose of each move.
2. **Comprehensive answer architecture.** Pro outputs are likely to be complete, well-structured, and
   exhaustively scoped. In prose, that can become too orderly for a public essay.
3. **Preamble and plan residue.** Because ChatGPT may show a preamble before reasoning, final text can
   inherit plan-like language: "I will examine," "The key issue is," "There are three risks," or
   similarly explicit framing.
4. **Caveat expansion.** A deep review setting may add qualifications to reduce risk. In this project,
   that can bloat the reliability section or weaken clear public language.
5. **Evidence-labeling habit.** The model may prefer source labels, criteria, numbered findings, and
   structured rationale. Useful in an internal review; distracting in a trade-book explainer.
6. **Well-structured professional memo voice.** GPT-5.5 Pro is optimized for high-stakes professional
   work. Without guardrails, it may move the text toward a polished briefing note.
7. **Over-smoothing.** The model may repair every rough edge and produce a high-gloss essay. The result
   can be clear but less human.
8. **Repeated transition rails.** GPT-family writing often leans on "This is why," "What matters is,"
   "The point is," and similar reader-guiding phrases.
9. **Balanced conclusion pressure.** Pro may finish sections with polished, symmetrical takeaways rather
   than letting a fact or example carry the paragraph.
10. **Shared surface tells.** Em-dash density, antithesis, tidy tricolons, and "not X but Y" contrast
    pivots still apply. Extended effort may make the reasoning better but does not automatically remove
    these stylistic habits.

---

## 4. Target-document measurements

Quick mechanical pass over `how-these-projections-work.md`:

| Signal | Measurement | Interpretation |
|--------|-------------|----------------|
| Word count | ~2,022 words by simple token count | A compact public explainer. |
| Sentences | 66 | Enough length for rhythm signals to matter. |
| Average sentence length | 21.8 words | Moderate-long explanatory prose. |
| Sentences of 18-24 words | 14 / 66 (21.2%) | Good: not metronomic. |
| Em dashes | 37 | High; still the loudest surface tell. |
| `not` occurrences | 19 | High; supports the contrast-pivot concern. |
| `rather than` occurrences | 5 | Noticeable repetition. |
| `Technical note` boxes | 5 | Good architecture; also attractive to synthesis-heavy models. |
| Bullet lines | 3 | Good: the main narrative is not list-shaped. |
| Obvious slop vocabulary | 0 hits for common terms searched | Good: vocabulary is clean. |

The target does not read like an unedited GPT-5.5 Pro output. It has specific county examples, real
method constraints, and a deliberate SOP-005 essay register. The GPT-5.5 Pro risk is not generic
vocabulary. It is **reviewerly polish**: a deep model would probably make the piece more comprehensive
and more explicitly reasoned unless instructed otherwise.

---

## 5. Do GPT-5.5 Pro Extended tells appear in the target?

### Present: professional synthesis architecture

The target is built like a controlled public explanation:

- Conceptual opening.
- Trend-line rejection.
- Three-component method explanation.
- Assumption caveat.
- County divergence.
- Reliability limits.
- Final recap.

This is appropriate. It also matches what GPT-5.5 Pro is good at: turning complex material into a
well-ordered explanation. The risk is not the structure itself; the risk is that a Pro Extended rewrite
would make the structure even more explicit and less essay-like.

### Present: evidence-aware caveat management

The document carefully separates projection from forecast, explains migration uncertainty, gives
back-testing context, and warns about small-area and long-range uncertainty. That care is necessary.
It also resembles the model habit of surfacing every qualification. A Pro Extended pass might add more
legalistic or methodological caveats than a public explainer needs.

### Present: transition rails and balanced takeaways

The target uses several phrases that a GPT-5.5 Pro reviewer might preserve or intensify:

- "That is the whole idea..."
- "This is why..."
- "If there is one point worth carrying away..."
- "Two cautions deserve..."
- "In a sentence..."

These are not bad. They are readable and often elegant. But they create a guided-tour feel when repeated
across the document.

### Present: antithesis and contrast pivots

The existing GPT-5.5 review already flagged the repeated "not X / but Y" pattern. From a Pro Extended
perspective, this is especially important because the model may see these turns as strong explanatory
logic and leave them untouched. The human edit should reduce frequency, not remove the required
"projection is not a forecast" caveat.

### Mostly absent: preamble residue

The target does not contain obvious planning-language residue. It does not say "this review will," "the
following sections," or "we will examine." That is good. Preserve it.

### Mostly absent: memo voice

Despite its structure, the target is not a briefing memo. It has metaphor, cadence, concrete examples,
and a restrained essay voice. That is the main thing to protect if GPT-5.5 Pro Extended is used for
review.

---

## 6. Editorial implications for `how-these-projections-work.md`

Use GPT-5.5 Pro Extended for:

- Checking that every public number still matches the locked run and public download package.
- Verifying that ADR-042 terminology is preserved.
- Finding factual overclaims, missing caveats, and places where the text implies more certainty than
  the model supports.
- Reviewing whether technical-note boxes are accurate and understandable to non-specialists.
- Producing a narrow change list after a human or Claude line edit.

Do not use GPT-5.5 Pro Extended as an unconstrained final rewrite model. The likely failure mode is a
document that is cleaner, longer, more careful, and more obviously model-shaped.

If using it anyway, constrain the prompt:

1. Ask for **diagnosis first**, not rewritten prose.
2. Forbid new numbers and new methodology claims unless tied to a cited local source.
3. Require every proposed edit to be shorter than the original unless a factual correction demands more
   words.
4. Preserve SOP-005 voice: reserved, trade-book prose; no FAQ conversion, no memo conversion, no
   marketing tone.
5. Ask it specifically to reduce repeated em dashes, antithesis, and transition rails.
6. Tell it not to add headings, bullets, or technical notes.

The best prompt pattern is: "Find the five highest-leverage edits that reduce AI-shaped rhythm while
preserving all numbers, caveats, and the current structure. Return only the original phrase, proposed
replacement, and reason."

---

## 7. Model routing for this task

Recommended route:

1. Use **GPT-5.5 Pro Extended in ChatGPT Web** for high-stakes review: factual consistency, caveat
   sufficiency, and whether the public explanation underspecifies any method step.
2. Use **GPT-5.5 `xhigh` via API/Codex** for repeatable constraint-checking when the task needs a
   programmable surface or a clear audit trail.
3. Use **Claude Opus 4.8 `max` or `xhigh`** for final voice-sensitive line editing.
4. Use **Gemini 3.1 Pro `high`** as a broad-context source-grounding and consistency check across the
   full handoff package.
5. Use **Gemini 3.5 Flash `high`** for quick procedural clarity checks.

Practical bottom line: GPT-5.5 Pro Extended is probably the strongest OpenAI option for **deep editorial
review and factual pressure-testing in ChatGPT**. It is not automatically the best option for final
public prose. For this document, it should act as a careful reviewer, not as the final stylist.

---

## 8. Suggested GPT-5.5 Pro Extended prompt

Use this only after the target document is ready for review:

```text
Review this public explainer as an internal editor. Do not rewrite the document wholesale.

Goals:
- Preserve ADR-042 terminology: "projection," never "forecast" except when explaining that a projection
  is not a forecast.
- Preserve all numbers exactly unless you identify a specific conflict with the cited source.
- Preserve the SOP-005 voice: reserved, trade-book prose for a broad public audience.
- Reduce AI-shaped rhythm: repeated em dashes, repeated "not X but Y" pivots, repeated section-ending
  maxims, and repeated transition rails.

Return only the 5-8 highest-leverage edits. For each edit, provide:
1. Original phrase or sentence.
2. Proposed replacement.
3. One-sentence reason.

Do not add new headings, bullets, caveats, numbers, methodology claims, or technical notes.
```

---

## 9. Sources consulted

Official OpenAI:

- [GPT-5.5 in ChatGPT](https://help.openai.com/en/articles/11909943)
- [Introducing GPT-5.5](https://openai.com/index/introducing-gpt-5-5/)
- [GPT-5.5 Pro model page](https://developers.openai.com/api/docs/models/gpt-5.5-pro)
- [Using GPT-5.5](https://developers.openai.com/api/docs/guides/latest-model)
- [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)
- [Prompt guidance](https://developers.openai.com/api/docs/guides/prompt-guidance)
- [Model Release Notes](https://help.openai.com/en/articles/9624314-model-release-notes)
- [GPT-5.5 System Card](https://deploymentsafety.openai.com/gpt-5-5)
- [About ChatGPT Pro tiers](https://help.openai.com/en/articles/9793128-about-chatgpt-pro-tiers)

External commentary and practitioner reviews:

- [Every - Vibe Check: GPT-5.5 Has It All](https://every.to/vibe-check/gpt-5-5)
- [MindStudio - GPT-5.5 Review: What It Actually Does Well](https://www.mindstudio.ai/blog/gpt-5-5-review-agentic-model)
- [Composio - Claude Opus 4.8 vs. GPT 5.5](https://composio.dev/content/opus-vs-gpt)

Related local references:

- [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md)
- [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md)
- [`SOP-005-public-facing-prose-voice.md`](../governance/sops/SOP-005-public-facing-prose-voice.md)

---

## 10. Source caveats

1. This review did not directly run GPT-5.5 Pro Extended against the target document. It infers likely
   writing behavior from current official product documentation, the existing GPT-5.5 review, and
   current practitioner commentary.
2. ChatGPT Web **Extended** is not documented as equivalent to API `xhigh`. Do not use those labels
   interchangeably in repo documentation.
3. The web UI setting is user-facing and may change faster than API model IDs. Re-check the ChatGPT help
   page before relying on this analysis in a future release.
4. AI-writing detection is probabilistic. These tells are editorial heuristics, not evidence of
   authorship.
