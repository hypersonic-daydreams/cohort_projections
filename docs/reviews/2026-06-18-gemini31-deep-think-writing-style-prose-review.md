# Gemini 3.1 Deep Think Writing Style: Research + Prose Review of `how-these-projections-work.md`

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-06-18 |
| **Author** | Codex research pass |
| **Target reviewed** | [`docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`](../plans/2026-public-projection-release-handoff/how-these-projections-work.md) |
| **Related reviews** | [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md); [`2026-06-18-gpt55-writing-style-prose-review.md`](./2026-06-18-gpt55-writing-style-prose-review.md); [`2026-06-18-gpt55-pro-extended-writing-style-prose-review.md`](./2026-06-18-gpt55-pro-extended-writing-style-prose-review.md); [`2026-06-18-gemini31-pro-high-writing-style-prose-review.md`](./2026-06-18-gemini31-pro-high-writing-style-prose-review.md); [`2026-06-18-gemini35-flash-high-writing-style-prose-review.md`](./2026-06-18-gemini35-flash-high-writing-style-prose-review.md) |
| **Scope** | Analyze the most current public Gemini **Deep Think** surface, determine the underlying model as closely as public sources allow, and assess likely writing-style tells against the public projections explainer. |

This is a reference artifact. It records findings and editorial guidance only. No edits were made to the
target explainer.

---

## 1. Current model status as of 2026-06-18

The current public DeepMind model page identifies the current Deep Think surface as
**Gemini 3.1 Deep Think** and says it is **built on top of Gemini 3.1 Pro**. That is the clearest current
answer to the underlying-model question.

The naming history matters:

- In December 2025 and February 2026 Google product posts referred to **Gemini 3 Deep Think**.
- In February 2026 Google released a major Deep Think upgrade for science, research, and engineering.
- One week later, Google described **Gemini 3.1 Pro** as the upgraded core intelligence that made those
  Deep Think breakthroughs possible.
- The current DeepMind product page now labels the surface **Gemini 3.1 Deep Think**.

Practical conclusion for this repo: use **Gemini 3.1 Deep Think** when referring to the current Deep
Think surface. Do not describe it as Gemini 3.5 Deep Think unless Google later publishes that name. Do
not treat it as a Gemini 3.5 Flash feature; Google help says Deep Think requires the **Pro** model.

Access and surface details:

- In Gemini Apps, Deep Think requires Google AI Ultra or Google AI Ultra for Business.
- Google Help says to select **Pro**, then select **Thinking Level > Deep Think**.
- Google describes Deep Think as experimental and subject to discontinuation or suspension.
- Google Help says Deep Think provides maximum parallel reasoning and may take a few minutes.
- Google says Deep Think requires the Pro model.
- Google said in February 2026 that select researchers, engineers, and enterprises could express
  interest in early Gemini API access for Deep Think.
- The public Gemini API model list checked for this review does not present Deep Think as a normal
  standalone public model ID. Treat it as a product/API early-access reasoning mode unless a specific
  model string is provided by Google.

---

## 2. Source-weighted findings

### Official Google and DeepMind sources

Google frames Deep Think as a specialized reasoning mode for difficult scientific, mathematical,
research, and engineering work:

- The current DeepMind page says Deep Think is built on Gemini 3.1 Pro and is best for modern challenges
  across science, research, and engineering.
- The February 2026 Google post says the updated Deep Think was built to solve modern challenges in
  science, research, and engineering, including problems with messy data, incomplete information, and no
  single clear solution.
- Google describes Deep Think as using advanced or maximum **parallel reasoning** to explore multiple
  hypotheses.
- DeepMind's science writeup emphasizes iterative generate-verify-revise workflows, natural-language
  verification, admission of failure, web/search grounding, and collaboration under expert direction.
- The model page highlights performance on ARC-AGI-2, Humanity's Last Exam, Codeforces, Olympiad math,
  Olympiad physics and chemistry, and condensed-matter theory benchmarks.
- The Gemini 3.1 Pro model card says Gemini 3.1 Pro is natively multimodal, has up to a 1M-token context
  window, and outputs text up to 64K tokens.

Implication for this project: Gemini 3.1 Deep Think is likely useful when the question resembles
scientific review: "Does this public explanation accurately reduce a complex model without losing a
necessary assumption?" It is not primarily a prose-style model. Its likely editorial failure mode is
turning the explainer into an expert-review memo.

### Evidence about current capability ceiling

Two current-source details matter for routing:

- The June 2026 Gemini 3.5 Audio model card says Google relies on the **Gemini 3.1 Pro with Deep Think**
  frontier-safety evaluation because it is the most generally capable model as of that publication.
- The DeepMind model-card index has newer Gemini 3.5 media/audio model cards, but the current Deep Think
  page still ties Deep Think to Gemini 3.1 Pro.

Implication for this project: even though Gemini 3.5 exists in other product lanes, the current public
evidence points to **Gemini 3.1 Pro + Deep Think** as the highest-reasoning Deep Think configuration,
not a Gemini 3.5 Pro or Gemini 3.5 Flash variant.

---

## 3. Gemini 3.1 Deep Think writing tells to watch

These are editorial risk signals, not proof of authorship.

1. **Scientific-review posture.** Deep Think is tuned and described around research, engineering, and
   scientific problem-solving. Its prose may sound like an expert reviewer assessing assumptions,
   proofs, boundary cases, and validity.
2. **Parallel-hypothesis framing.** Because Deep Think is described as maximum parallel reasoning, it may
   present alternatives, branches, or competing interpretations even when the public explainer needs one
   clean line.
3. **Generate-verify-revise residue.** DeepMind's research examples emphasize verifier loops. In prose,
   that can become "possible issue / check / correction" structure.
4. **Failure-admission and uncertainty discipline.** This is useful for methodology review, but it can
   overpopulate a public explainer with hedges.
5. **Benchmark and rigor register.** The model's public identity is tied to hard benchmarks, Olympiad
   problems, Codeforces, and scientific discovery. It may overvalue rigor language in writing tasks.
6. **Cross-domain synthesis voice.** Deep Think is described as bridging scientific domains. Its prose
   may pull analogies or abstractions from outside the immediate demographic question.
7. **Complexity inflation.** Deep Think may treat a prose problem as more technically intricate than it
   is, especially if asked to "analyze deeply."
8. **Long-form comprehensiveness.** With a large-context Pro base and deep reasoning mode, it may produce
   a complete review package when the editorial need is a small set of line edits.
9. **Structured decomposition.** Expect headings, numbered findings, diagnostic categories, and explicit
   criteria unless the prompt forbids them.
10. **Shared LLM surface tells.** Em-dash density, "not X but Y" contrast pivots, tidy tricolons,
    balanced closers, and self-labeling significance remain relevant.

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
| `Technical note` boxes | 5 | Good public-methods design; also attractive to synthesis-heavy models. |
| Bullet lines | 3 | Good: the main narrative is not list-shaped. |
| Obvious slop vocabulary | 0 hits for common terms searched | Good: vocabulary is clean. |

The target does not read like Gemini 3.1 Deep Think output. It is too restrained and public-facing for
that. The Deep Think risk is mostly prospective: if used as an editor, it may add more verification
machinery and expert caveats than the piece can carry.

---

## 5. Do Gemini 3.1 Deep Think tells appear in the target?

### Present: structured model explanation

The target explains a complex model through orderly conceptual stages: population structure, survival,
fertility, migration, assumptions, county divergence, and reliability. That structure is compatible with
Gemini 3.1 Deep Think's synthesis strengths.

This is a strength. The editorial risk is that a Deep Think revision might make each stage more
proof-like, with extra qualifiers and explicit validation language.

### Present: caveat and uncertainty discipline

The document repeatedly protects the projection from being read as a forecast:

- It uses ADR-042 language.
- It explains assumption sensitivity.
- It gives back-testing context.
- It warns that small-area and long-range results carry more uncertainty.

This is necessary. A Deep Think pass might still overdo it, because the model's research posture rewards
surfacing edge cases and uncertainty.

### Mildly present: verification architecture

The technical-note boxes function as a controlled verification layer. They are good public design. They
also resemble the kind of "main answer plus technical validation" structure a deep reasoning model might
prefer.

The boxes should stay. The issue is to keep them short and factual, not let them expand into mini-methods
appendices.

### Mostly absent: scientific proof register

The target does not sound like a proof, benchmark report, or scientific review. It does not use
"hypothesis," "candidate solution," "verifier," "failure mode," or "benchmark" language. That is good.

### Mostly absent: parallel alternatives

The target follows one public baseline scenario. It does not branch into many possible paths. That is
correct for this release. Deep Think should not be invited to add scenario-analysis prose unless the
public product scope changes.

---

## 6. Editorial implications for `how-these-projections-work.md`

Use Gemini 3.1 Deep Think for:

- Stress-testing whether the public explanation has made a hidden methodology leap.
- Checking whether any caveat is too weak or misleading.
- Verifying whether the technical-note boxes accurately simplify the method.
- Looking for contradictions across the explainer, public PDF copy, final-run metadata, and methodology
  notes.
- Reviewing whether a skeptical technical reader could reasonably object to a public claim.

Do not use Gemini 3.1 Deep Think for:

- Final voice edits.
- Making the explainer warmer or more literary.
- Shortening prose unless the prompt is tightly constrained.
- Producing broad rewrites of the public narrative.
- Replacing human judgment about SOP-005 voice.

If using it, prompt against the likely failure mode:

1. Ask for **at most five** findings.
2. Require each finding to cite the exact sentence and the exact local source that creates the issue.
3. Forbid additional caveats unless the current text is factually misleading.
4. Forbid new headings, new technical-note boxes, and scenario expansion.
5. Require proposed replacements to be no longer than the original.
6. Tell it that the intended audience is a broad public audience, not scientists, engineers, or model
   reviewers.

---

## 7. Model routing for this task

Recommended route:

1. Use **Gemini 3.1 Deep Think** only for the deepest factual/methodological stress test.
2. Use **Gemini 3.1 Pro `high`** for broad-context consistency review when Deep Think is overkill.
3. Use **GPT-5.5 Pro Extended** for high-stakes editorial diagnosis and factual pressure-testing in
   ChatGPT.
4. Use **GPT-5.5 `xhigh`** for repeatable structure and constraint-preservation checks.
5. Use **Claude Opus 4.8 `max` or `xhigh`** for final voice-sensitive public-prose editing.
6. Use **Gemini 3.5 Flash `high`** for quick procedural clarity checks.

Practical bottom line: Gemini 3.1 Deep Think is the best Gemini option to ask, "Could a serious technical
reviewer poke a hole in this?" It is not the best option to ask, "Does this sound like a human public
essay?"

---

## 8. Suggested Gemini 3.1 Deep Think prompt

Use this only after the target document is close to final:

```text
Review this public explainer as a technical stress test, not as a rewrite task.

Audience and voice:
- The intended reader is a broad public audience.
- Preserve the SOP-005 reserved trade-book voice.
- Do not turn the explainer into a technical memo, scientific review, FAQ, or benchmark report.

Task:
- Identify at most five places where the explainer may be factually misleading, under-caveated, or
  inconsistent with the methodology and locked public outputs.
- For each finding, quote only the short phrase or sentence needed to locate the issue.
- Explain the issue in one sentence.
- Propose a replacement that is no longer than the original.

Constraints:
- Preserve all numbers unless you identify a specific source conflict.
- Preserve ADR-042 terminology: use "projection," not "forecast."
- Do not add new headings, bullets, scenario analysis, methodology claims, or technical-note boxes.
- Do not suggest style-only edits unless they reduce a concrete misunderstanding.
```

---

## 9. Sources consulted

Official Google and DeepMind:

- [Gemini 3.1 Deep Think](https://deepmind.google/models/gemini/deep-think/)
- [Gemini 3 Deep Think: Advancing science, research and engineering](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/)
- [Use Deep Think in Gemini Apps](https://support.google.com/gemini/answer/16345172?co=GENIE.Platform%3DDesktop&hl=en)
- [Gemini 3 Deep Think is now available](https://blog.google/products-and-platforms/products/gemini/gemini-3-deep-think/)
- [Gemini Apps limits and upgrades for Google AI subscribers](https://support.google.com/gemini/answer/16275805?hl=en)
- [Gemini 3.1 Pro model card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)
- [Gemini 3.1 Pro: A smarter model for your most complex tasks](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/)
- [Gemini 3 Deep Think evaluation methodology](https://deepmind.google/models/evals-methodology/gemini-3-deep-think)
- [Accelerating Mathematical and Scientific Discovery with Gemini Deep Think](https://deepmind.google/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/)
- [Google DeepMind model cards index](https://deepmind.google/models/model-cards/)
- [Gemini 3.5 Audio model card](https://deepmind.google/models/model-cards/gemini-3-5-audio/)
- [Gemini API models](https://ai.google.dev/gemini-api/docs/models)

Related local references:

- [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md)
- [`2026-06-18-gemini31-pro-high-writing-style-prose-review.md`](./2026-06-18-gemini31-pro-high-writing-style-prose-review.md)
- [`SOP-005-public-facing-prose-voice.md`](../governance/sops/SOP-005-public-facing-prose-voice.md)

---

## 10. Source caveats

1. This review did not directly run Gemini 3.1 Deep Think against the target document. It infers likely
   writing behavior from current Google/DeepMind documentation and the existing local reviews.
2. Google product naming has shifted from "Gemini 3 Deep Think" to "Gemini 3.1 Deep Think" across public
   sources. This review uses the current DeepMind product-page name.
3. The public sources checked here do not provide a normal standalone public API model string for Deep
   Think. Treat it as a Gemini Apps reasoning level and API early-access mode unless Google publishes a
   specific model ID.
4. Deep Think is experimental and can change quickly. Re-check Google Help and DeepMind model pages
   before relying on this in a later release.
5. AI-writing detection is probabilistic. These tells are editorial heuristics, not evidence of
   authorship.
