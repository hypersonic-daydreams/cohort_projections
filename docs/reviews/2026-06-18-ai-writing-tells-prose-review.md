# AI-Writing Tells: Research + Prose Review of `how-these-projections-work.md`

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-06-18 |
| **Author** | Claude Code (Opus 4.8) research pass |
| **Target reviewed** | [`docs/plans/2026-public-projection-release-handoff/how-these-projections-work.md`](../plans/2026-public-projection-release-handoff/how-these-projections-work.md) |
| **Scope** | Research the current state of AI-prose detection (esp. recent Claude models), then assess the public "How These Projections Work" companion against it. **No edits were made to the target.** |
| **Related** | [SOP-005 Public-Facing Prose Voice](../governance/sops/SOP-005-public-facing-prose-voice.md) |

This is a reference artifact. It records *findings and suggestions only* — the target document was deliberately left unedited so that any changes can be decided and applied separately.

---

## 1. What current research says identifies AI prose (2025–2026)

The discriminators have moved away from obvious vocabulary flags ("delve," "tapestry," "leverage," "robust") toward **structural rhythm**. Vocabulary swaps no longer help much because rhythm is what detectors measure. The strongest tells, across the sources consulted:

1. **Negation / antithesis parallelism** — "not X, but Y" / "It isn't about X, it's about Y." LLMs produce roughly *one per paragraph*; humans use it occasionally. This is currently the single loudest structural signal.
2. **Compulsive rule-of-three** — tidy tricolons with matched rhythm ("fast, reliable, and affordable"). AI adds a third element even when two would do.
3. **Em-dash density** — weak alone (humans love em dashes too), but at high density still the most *recognizable* surface tell. Working budget: ~1 per 1,000 words.
4. **Cadence uniformity** — described as the *strongest statistical* indicator: a run of 18–24-word sentences, paragraph after paragraph, with little length variation.
5. **Self-labeling significance** — "worth dwelling on," "worth carrying away," "what struck me," "here's where it gets interesting." Pre-announcing importance instead of letting the description carry it.
6. **Templated balanced-aphorism closers** — every section landing on a tidy antithetical maxim.
7. **Copula avoidance** — "serves as / represents / features / boasts" instead of plain "is / has."
8. **Pangram's Claude-specific finding** — Claude prose shows *high lexical diversity but rigid structuring*; it is the **structure, not the vocabulary**, that gives it away. Claude outputs "still sound like Claude" even under custom style prompts.

**Universal caveat (in nearly every source):** these patterns also fire on careful technical writing and second-language writers, so no single one is conclusive. They are probability signals, best combined.

### Sources

- [Pangram Labs — Can AI detection catch Claude writing styles?](https://www.pangram.com/blog/claude-writing-styles)
- [avoid-ai-writing SKILL.md (Conor Bronsdon)](https://github.com/conorbronsdon/avoid-ai-writing/blob/main/SKILL.md) — the most concrete pattern/alternative catalog
- [Duey AI — The Em-Dash Myth: What Actually Gives Away AI Writing](https://www.duey.ai/post/em-dash-ai-writing)
- [The Ringer — Stop AI-Shaming Our Precious Em Dashes](https://www.theringer.com/2025/08/20/pop-culture/em-dash-use-ai-artificial-intelligence-chatgpt-google-gemini)
- [Beutler Ink — How to Spot AI Writing, According to Wikipedia](https://www.beutlerink.com/blog/how-to-spot-ai-writing)
- [Knightli — How to Detect Claude 4-Generated Text](https://knightli.com/en/2026/05/08/detect-claude-4-ai-generated-text-tools/)

---

## 2. How the document scores

The document is genuinely well-written and its **vocabulary is clean** — almost none of the Tier-1 slop words appear. But it lands hard on exactly the *structural* tells that current detection keys on. Two are pervasive.

### Tell 1 — Negation parallelism in nearly every paragraph (loudest signal)

- L27 — "The task is not to conjure a future from nothing. It is to take what we know…"
- L42 — "a population is not really a single number. It is a structure"
- L54 — "responds to *who* lives in a place, not merely *how many*"
- L106 — "not from births and deaths but from people on the move"
- L149 — "It is a planning path, not a guarantee and not a prediction"
- L166 — "Neither conclusion is an editorial judgment… each falls out of that county's own numbers"
- L186 — "That is a limit of the enterprise, not a flaw in this particular model"
- plus "rather than merely assert them," "rather than assumed to run at peak rates forever," "rather than projecting the state on its own."

~10+ instances of the same rhetorical move. **A human stylist would never deploy "not X but Y" ten times in ~1,900 words — that repetition, more than any single sentence, is the tell.**

### Tell 2 — Em-dash density far over budget

Roughly **25+ em dashes in ~1,900 words** — about 12× the ~1-per-1,000-words threshold. Many do real work, but clusters like L106–107 ("on the move — a great deal of it from abroad") and L144 ("starting point of some **799,400**. That early dip…") could be commas, periods, or parentheses.

### Tell 3 — Every section closes on a balanced aphorism

- L118 — "The arithmetic is patient and unglamorous, but it is faithful…"
- L195 — "Its strength is that it honors… Its limits are simply the limits of looking thirty years ahead"
- L49 — "a trend line never saw it coming"

Each is good individually; the problem is that *every* section ends this way, so the template becomes audible.

### Tell 4 — Self-labeling significance (three times in close range)

- L145 — "That early dip is worth dwelling on for a moment"
- L156 — "If there is one point worth carrying away"
- L179 — "Two cautions deserve to travel alongside those figures, and they are not fine print"

### Tell 5 — Tricolons to vary (mild)

- L28 — "as people age, as some are born, as some die, and as others arrive or depart"
- L113 — "oil-boom counties, college towns, military and tribal communities"

---

## 3. Suggestions (not applied)

1. **Thin the negation parallelism.** Keep the best two or three (the opening "not to conjure… / to take what we know" earns it); convert the rest to direct positive statements. E.g. L149 → "It is a planning path. Change the assumptions and the path changes with them." Highest-leverage change for de-AI-ing the prose.
2. **Halve the em dashes.** Appositive em dashes → commas; dramatic ones → periods or parentheses. Highest-leverage *surface* change.
3. **Vary the section closers.** Let two or three sections end flatly on a fact or a number instead of a maxim.
4. **Cut the significance-framing.** Start with the substance instead of announcing it ("The early dip is easily misread: it is built into the migration assumption…").
5. **Break up one or two tricolons** (L28, L113). Mild; low priority.

### What *not* to change

- **Sentence-length variation is already good** — the doc mixes short declaratives, long sentences, and fragments. That is the strongest statistical tell and the doc passes it. Do not homogenize it while editing.
- **The concrete county examples** (Williams young/high-birth; Ward post-2020 out-migration) are the most human element — specific, falsifiable, load-bearing. Keep.
- **Vocabulary is clean** — no synonym-hunting needed. The issue is rhythm, not words.

---

## 4. Tension to flag: SOP-005 trade-book voice

This document is deliberately written in the reserved "trade-book voice" mandated by [SOP-005](../governance/sops/SOP-005-public-facing-prose-voice.md). University-press prose *legitimately* uses antithesis and balanced cadence, so some of what reads as an "AI tell" is the intended register. The goal is **not** to strip the voice — it is to **lower the density** of two or three mechanical moves (negation parallelism, em dashes, templated closers) so the register reads as a confident human essayist rather than a model doing literary cosplay. The fix is quantitative (frequency), not categorical (banning the device).

---

## 5. Reusability

The Section 1 tell list and the Section 3 "what not to change" guidance apply to **all** public-facing narrative prose in this repo (public PDF copy, explainers, FAQs), not only the file reviewed here. Consult before drafting or revising any SOP-005 document.
