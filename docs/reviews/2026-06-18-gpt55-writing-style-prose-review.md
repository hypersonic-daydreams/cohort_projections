# GPT-5.5 Writing Style: Research + Prose Review of `how-these-projections-work.md`

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-06-18 |
| **Author** | Codex research pass |
| **Target reviewed** | [`docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`](../plans/2026-public-projection-release-handoff/how-these-projections-work.md) |
| **Prior review** | [`2026-06-18-ai-writing-tells-prose-review.md`](./2026-06-18-ai-writing-tells-prose-review.md) |
| **Scope** | Augment the prior AI-writing-tells review with GPT-5.5-specific style research, assess whether those tells appear in the target, and compare GPT-5.5 (`xhigh`) with Claude Opus 4.8 (`max`) for writing work. |

This is a reference artifact. It records findings and editorial guidance only. No edits were made to the
target explainer.

---

## 1. Source-weighted findings

### Official OpenAI sources

OpenAI's current GPT-5.5 guidance does not describe GPT-5.5 as a literary prose model. It describes a
model tuned for cleaner task execution:

- The GPT-5.5 prompting guide says the default style is "efficient, direct, and task-oriented," with
  less conversational padding. It recommends specifying personality and collaboration style when the
  product needs a customer-facing voice.
- OpenAI's model release notes say the May 28, 2026 GPT-5.5 Instant update improved response style:
  easier to read, more natural in everyday conversation, better paced, and less overly long or
  bullet-heavy.
- The ChatGPT help page describes GPT-5.5 Instant as warmer and conversational for everyday work, and
  GPT-5.5 Thinking as more streamlined with cleaner formatting and less unnecessary header text.
- The GPT-5.5 API guide says GPT-5.5 is more polished but can be more direct, and that migration should
  start with the smallest prompt that preserves the product contract rather than carrying over heavy
  prompt scaffolding.
- The `reasoning.effort` parameter can include `xhigh`; higher effort means more complete thinking, but
  this is a reasoning-depth control, not a direct prose-style control.

Implication for this project: GPT-5.5's default writing strength is clarity, structure, and disciplined
execution. Its weakness, when used for public narrative prose, is that it can make the prose sound like a
highly competent professional memo unless the prompt explicitly protects cadence, texture, and voice.

### External writing-style sources

The outside commentary is directionally consistent with the official guidance, though it is less
authoritative:

- Every's hands-on review says GPT-5.5 explains itself clearly, moves cleanly between ideas, and is
  better suited to broader audiences than Opus 4.7 in the examples tested, though less clever.
- FwdSlash characterizes GPT-5.5 as an efficient professional communicator: concise, well-structured,
  fast, and good at templated content.
- Current AI-writing-tell discussions continue to flag repeated corrective pivots ("not X, but Y"),
  tidy tricolons, cadence uniformity, and boilerplate closers. Several sources caution that em dashes
  alone are weak evidence, but em-dash cadence plus repeated contrast framing remains recognizable.

Implication for this project: the strongest GPT-5.5 tell is not a particular banned word. It is a
document that is too cleanly scaffolded: every paragraph moves efficiently, every transition is explicit,
every section lands a polished takeaway, and every contrast is balanced.

---

## 2. GPT-5.5 writing tells to watch

These are not proof of authorship. They are editorial risk signals for prose that should read as a
human public explainer rather than a model-shaped explanation.

1. **Efficient-professional overstructure.** GPT-5.5 is good at "intro, key points, conclusion" and at
   turning messy material into a clean sequence. In narrative prose, that can become too audible:
   every paragraph has a job, every transition is named, and the document never idles or surprises.
2. **Clean transition rails.** Phrases such as "This is why," "The common thread," "What matters is,"
   "That is the point," and "In a sentence" are useful, but dense use can make the argument feel
   mechanically guided.
3. **Corrective contrast pivots.** GPT-family writing still leans on "not X but Y," "not merely X,"
   "not a guarantee," and "rather than." These are legitimate rhetorical tools, but repeated use is a
   current AI-writing tell.
4. **Balanced aphoristic closers.** GPT-5.5 can produce polished, memorable section endings. Too many
   in sequence make the essay sound generated because every section resolves with the same kind of
   rhetorical click.
5. **Bullet and heading discipline.** GPT-5.5 has been explicitly tuned away from being overly
   bullet-heavy, but its instinct is still to organize. In public prose, this is useful for technical
   notes and dangerous for connective narrative if every idea becomes a neat subheading or list.
6. **Polished directness.** The model's concision helps with readability but can flatten voice. A
   human essayist sometimes hesitates, varies emphasis, leaves one sentence plain, or lets a number end
   the paragraph.
7. **Low lexical slop, high structural regularity.** Newer GPT outputs may avoid older tells like
   "delve," "tapestry," and "robust." The tell moves to rhythm: repeated sentence shapes, repeated
   contrast logic, and tidy section geometry.

---

## 3. Target-document measurements

Quick mechanical pass over `how-these-projections-work.md`:

| Signal | Measurement | Interpretation |
|--------|-------------|----------------|
| Word count | ~2,022 words by simple token count | A compact public explainer. |
| Sentences | 66 | Enough length for rhythm signals to matter. |
| Average sentence length | 21.8 words | In the detector-risk zone, but not conclusive. |
| Sentences of 18-24 words | 14 / 66 (21.2%) | Good: not metronomic. The target has real sentence-length variation. |
| Em dashes | 37 | High. This is the loudest surface tell. |
| `not` occurrences | 19 | High, though some are required caveats. |
| `rather than` occurrences | 5 | Supports the contrast-pivot concern. |
| `not merely` / `not only` | 2 | Mild alone; noticeable in context. |
| Bullet lines | 3 | Good: the main narrative is not bullet-heavy. |
| Obvious slop vocabulary | 0 hits for common terms searched | Good: vocabulary is clean. |

The prior review's diagnosis holds: the target does not read as generic AI prose at the word level. It
is specific, accurate, and often genuinely elegant. The risk is structural density.

---

## 4. Do GPT-5.5 tells appear in the target?

### Present: efficient-professional overstructure

The document's macro-architecture is very GPT-5.5-compatible: framing hook, trend-line contrast,
three-component explanation, assumption caveat, county divergence, accuracy caveats, final summary. That
is good public pedagogy. It also means the piece has almost no roughness. Every section knows exactly
what it is doing.

This is not a problem by itself. But paired with frequent contrast pivots and polished closers, the
structure starts to feel model-shaped.

### Present: clean transition rails

Examples:

- "That is the whole idea..." in the opening.
- "This is why..." before the statewide path explanation.
- "If there is one point worth carrying away..." before the county section.
- "Two cautions deserve..." before the reliability caveats.
- "In a sentence" as the final section title.

These are clear, reader-friendly moves. The edit is not to ban them. The edit is to make some of them
less explicitly signposted, so the reader feels trusted rather than guided down rails.

### Present: corrective contrast pivots

Representative lines:

- Lines 26-28: "not to conjure... It is to take..."
- Lines 42-43: "not really a single number. It is a structure..."
- Line 54: "who... not merely how many"
- Lines 106-107: "not from births and deaths but from people on the move"
- Lines 148-150: "not a forecast... not a guarantee and not a prediction"
- Lines 161-167: "rather than merely assert them... Neither conclusion..."
- Lines 173-186: "No projection... No method... not a flaw..."

Some of these are necessary. ADR-042 almost requires the "projection is not a forecast" caveat. But the
same contrast move appears often enough that it becomes the signature rhythm of the piece.

### Present: balanced closers

Several endings are individually good:

- "a trend line never saw it coming"
- "The arithmetic is patient and unglamorous..."
- "That is a limit of the enterprise, not a flaw..."
- "Its strength is... Its limits are..."

The issue is repetition, not quality. GPT-5.5 is especially good at clean explanatory closure; the
target currently uses that move often enough to be audible.

### Mostly absent: bullet-heavy GPT formatting

This document does not have the most obvious GPT-5.5 presentation tell. It has only one short bullet
list in the assumptions section. That is appropriate.

### Mostly absent: vocabulary slop

The target avoids the old marker words. That is a strength. Do not chase synonyms; change rhythm.

### Mostly absent: cadence uniformity

The target's sentence lengths vary. It uses short sentences, long explanatory sentences, and occasional
fragments. This is the main reason the piece still reads human in many places despite the structural
signals above.

---

## 5. Editorial implications for `how-these-projections-work.md`

Highest-leverage changes:

1. **Cut the em dashes by at least half.** Keep the ones that genuinely create an aside or turn. Convert
   many appositive dashes to commas, parentheses, colons, or periods.
2. **Reduce contrast pivots, not required caveats.** Keep "projection is not a forecast." Keep one or two
   strong "not X / but Y" moves. Convert the rest to direct positive statements.
3. **Let more sections end on facts.** A number, county example, or plain sentence will sound more human
   than another balanced maxim.
4. **Remove a few signposts.** Replace "That early dip is worth dwelling on..." with "The early dip is
   easily misread..." Replace "If there is one point worth carrying away..." with the point itself.
5. **Preserve the concrete demography.** The Williams and Ward examples, the 2027 low, the 53-county
   reconciliation, and the 91 x 2 x 6 cohort detail are the least AI-like elements because they are
   specific and accountable.

Do not flatten the style into a plain FAQ. SOP-005 asks for a university-press trade-book register, and
the current document is the exemplar for that register. The goal is to reduce repeated devices while
keeping the reserved essay voice.

---

## 6. GPT-5.5 (`xhigh`) vs Claude Opus 4.8 (`max`) for writing

The comparison is not apples-to-apples. OpenAI's `xhigh` and Anthropic's `max` are provider-specific
reasoning-effort controls. They increase the amount of reasoning or adaptive thinking the model may use;
they do not directly mean "better prose style."

### GPT-5.5 at `xhigh`

Best writing uses:

- Restructuring messy source material into a clean public explanation.
- Checking whether all constraints, caveats, and facts are preserved.
- Producing clear, audience-aware drafts that do not need much prompt scaffolding.
- Turning notes into a readable first draft or a controlled revision plan.

Likely weaknesses for this explainer:

- May over-regularize the essay into a clean explanatory sequence.
- May prefer visible scaffolding, crisp section logic, and tidy conclusions.
- May improve clarity while making the prose less idiosyncratic.

Practical assessment: GPT-5.5 `xhigh` is excellent for structure, clarity, and constraint retention. It is
not the first model I would use alone for the final "make this sound less AI-shaped" line edit.

### Claude Opus 4.8 at `max`

Best writing uses:

- Preserving and extending a specific voice over a long session.
- Performing taste-sensitive line edits where voice, factual care, and document purpose have to coexist.
- Flagging uncertainty or unsupported claims rather than smoothing over them.
- Reviewing prose for over-explanation, repeated rhetorical devices, and places where a human editor
  would let the sentence breathe.

Likely weaknesses for this explainer:

- At `max`, it may overthink a prose problem and produce too much commentary or too many alternate
  rewrites.
- It may smooth rough edges too aggressively unless told to preserve the house voice.
- It can still share Claude-family tells noted in the prior review: polished structure, high lexical
  competence, and repeated balanced framing.

Practical assessment: Claude Opus 4.8 `max` is likely the stronger choice for final public prose revision
when the goal is voice, taste, and local cadence. For many editing passes, `xhigh` or `high` may be more
efficient than `max`; max should be reserved for especially delicate or high-stakes edits.

### Recommended routing

For this document:

1. Use GPT-5.5 `xhigh` for a constraint-preservation pass: "After edits, did we preserve ADR-042
   terminology, all numbers, all caveats, and the explanatory sequence?"
2. Use Claude Opus 4.8 `max` or `xhigh` for a line-edit pass: "Reduce repeated AI-shaped rhetorical
   devices while preserving SOP-005 voice."
3. Use a human final pass for taste: count dashes, read the section endings aloud, and ensure the piece
   still sounds like a public explainer rather than an AI-detection avoidance exercise.

Bottom line: GPT-5.5 `xhigh` is the stronger organizing and verification instrument. Claude Opus 4.8
`max` is the better candidate for the final voice-sensitive prose pass. Neither should be treated as a
truth oracle for "human-sounding" writing; the target standard is the house voice plus factual fidelity.

---

## 7. Sources consulted

Official OpenAI:

- [Using GPT-5.5 | OpenAI API](https://developers.openai.com/api/docs/guides/latest-model)
- [Prompt guidance | OpenAI API](https://developers.openai.com/api/docs/guides/prompt-guidance)
- [Reasoning models | OpenAI API](https://developers.openai.com/api/docs/guides/reasoning)
- [GPT-5.5 in ChatGPT | OpenAI Help Center](https://help.openai.com/en/articles/11909943)
- [Model Release Notes | OpenAI Help Center](https://help.openai.com/en/articles/9624314-model-release-notes)
- [GPT-5.5 Pro model page | OpenAI API](https://developers.openai.com/api/docs/models/gpt-5.5-pro)
- [Introducing GPT-5.5 | OpenAI](https://openai.com/index/introducing-gpt-5-5/)

Official Anthropic:

- [Introducing Claude Opus 4.8 | Anthropic](https://www.anthropic.com/news/claude-opus-4-8)
- [What's new in Claude Opus 4.8 | Claude API Docs](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8)
- [Choosing the right model | Claude API Docs](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model)
- [Claude Code model configuration](https://code.claude.com/docs/en/model-config)
- [Migration guide | Claude API Docs](https://platform.claude.com/docs/en/about-claude/models/migration-guide)

External commentary and detection heuristics:

- [Every - Vibe Check: GPT-5.5 Has It All](https://every.to/vibe-check/gpt-5-5)
- [FwdSlash - Claude vs GPT 5.5 for Writing](https://www.fwdslash.ai/blog/claude-vs-gpt-5-5-for-writing)
- [Pangram Labs - Can AI detection catch Claude writing styles?](https://www.pangram.com/blog/claude-writing-styles)
- [Duey AI - The Em-Dash Myth](https://www.duey.ai/post/em-dash-ai-writing)
- [Beutler Ink - How to Spot AI Writing, According to Wikipedia](https://www.beutlerink.com/blog/how-to-spot-ai-writing)
- [David Bachman - AI on AI slop](https://profbachman.substack.com/p/ai-on-ai-slop)
- [avoid-ai-writing skill](https://github.com/conorbronsdon/avoid-ai-writing/blob/main/SKILL.md)

---

## 8. Source caveats

1. Official model docs are reliable for product behavior, parameters, and vendor-stated style goals, but
   they are not neutral writing-quality benchmarks.
2. External writing comparisons are useful for practitioner signal, not definitive model ranking.
3. AI-writing detection is probabilistic. These tells are editorial heuristics, not evidence of
   authorship.
4. The target document intentionally uses the SOP-005 trade-book voice. Some antithesis and polished
   cadence are part of the desired register; the actionable issue is frequency.
