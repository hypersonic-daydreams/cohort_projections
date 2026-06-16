# GPT-5.5 Pro API Reference: Usage, Limits, and Lessons Learned

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-06-16 |
| **Basis** | OpenAI developer docs (fetched 2026-06-16) + a billed failure during the PUB-2026 ref-intl round-2 review (batch `batch_6a307dd0…`, 2026-06-15/16) |
| **Occasion** | A round-2 GPT-5.5 Pro review of the corrected projection was submitted via the **Batch API**, failed without producing output, and still billed **~$85** through undocumented retries. This document records how to use the model correctly so future agents do not repeat that. |
| **Scope** | How this project calls `gpt-5.5-pro` via the OpenAI API for large document/code reviews (the `run_gpt55pro_*` runners under `docs/reviews/`). |

This is the reference for using OpenAI's `gpt-5.5-pro` reasoning model from the API. Read it before
submitting any large (>100k-token) review to the model, and **before choosing batch vs live**. The
headline lesson: for a big, long-running Pro request, **use the live Responses API in background
mode — not the Batch API.** The reasons are below, grounded in the docs and in a real $85 incident.

---

## TL;DR — the rules

1. **Use live + background mode, not batch, for large Pro requests.** The Batch API has **no
   timeout protection**, and a long Pro request that times out gets **retried and billed each time**
   with no cost ceiling. Background mode (live Responses API) is the documented timeout remedy and
   lets you **cancel for $0**.
2. **A "failed" batch is NOT free.** Empirically it billed ~$85 (6 retry executions) even though the
   batch object reported `completed: 0` and `usage: {}`. **Never trust the batch status object's
   `usage` as the cost** — only the billing/usage CSVs are authoritative.
3. **`max_output_tokens` is a hard wall, not a budget.** Hitting it **truncates** generation to status
   `incomplete` (`reason: max_output_tokens`) — the model does NOT wrap up gracefully. Worse, reasoning
   bills first, so a too-low cap can spend the whole budget on hidden reasoning and return an
   **empty answer you still pay for.** A single live response is already hard-capped at the model's
   **128k output ceiling (~$23)**, so prefer **no cap** (rely on that ceiling + cancel-for-$0) or a
   **high backstop ~115k**. The real runaway protection is using live (no retries), not the cap.
4. **Size is rarely the limit.** Context window is 1,050,000 tokens; a 400k-token package uses ~40%.
   Don't shrink the package for "size" reasons — check the actual limits first.
5. **Cancel = $0.** A live background response cancelled before completion reports `usage: 0` and is
   not billed. This is the main reason live+background beats batch for control.
6. **Prefer one comprehensive run over splitting** for an end-to-end audit (see "Scoping" below).
   Splitting usually costs *more* (re-sent shared context) and loses cross-cutting checks.

---

## Model facts (documented)

Source: [gpt-5.5-pro model page](https://developers.openai.com/api/docs/models/gpt-5.5-pro).

| Property | Value |
|----------|-------|
| Model id | `gpt-5.5-pro` (dated snapshot seen in usage: `gpt-5.5-pro-2026-04-23`) |
| Context window | **1,050,000 tokens** |
| Max output tokens | **128,000** (includes reasoning tokens) |
| Input price | **$30.00 / 1M tokens** |
| Output price | **$180.00 / 1M tokens** (reasoning tokens bill here too) |
| Cached input discount | **None** — Pro does not offer cached-input pricing |
| Reasoning effort | supports reasoning; this project uses `xhigh` for deep review |
| Endpoints | Responses API (`/v1/responses`), including via Batch |
| Latency note | *"Some requests may take several minutes to finish. To avoid timeouts, try using background mode."* |

### Rate limits by tier (gpt-5.5-pro)

| Tier | Qualify (cumulative paid) | RPM | TPM (live) | Batch queue (enqueued tokens) |
|------|---------------------------|-----|-----------|-------------------------------|
| 1 | $5 | 50 | 50K | 500K |
| 2 | $50 | 500 | 200K | 1M |
| 3 | $100 | 500 | **500K** | 10M |
| 4 | $250 | 1,000 | **1M** | 20M |
| 5 | $1,000 | 2,000 | 4M | 1.5B |

**A single review request in this project is ~405k input tokens.** That fits live **TPM** at **Tier 3
and above** (500K/1M) and the **batch queue** at every tier ≥1 (≥500K). So at this project's tier
(Tier 3–4), **rate limits are not the binding constraint** for one review request. (Reasoning tokens
bill as output — [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).)

---

## Live (synchronous) vs Batch — when to use which

| | **Live Responses API + background mode** | **Batch API** |
|---|---|---|
| Cost | full price ($30/$180 per M) | 50% discount *on completed requests* |
| Latency | minutes (returns when done) | async, up to 24h completion window |
| **Timeout protection** | **Yes** — background mode is the documented remedy | **No** — no background mode in batch |
| **Behavior on a long/timed-out request** | keeps running in background; you poll | **retries (undocumented) and bills each attempt** |
| **Cancellable** | **Yes → `usage: 0`, $0** (`POST /v1/responses/{id}/cancel`) | not per-request; batch-level only |
| Cost ceiling | `max_output_tokens` caps it; cancel anytime | **none** — retries can multiply cost |
| Good for | **large/long Pro reviews (this project's use)** | many small, independent, latency-tolerant requests |

**Decision rule:** for a single large Pro review that may take several minutes, **use live +
background mode.** Batch's 50% discount is not worth losing timeout protection and accepting
uncapped retry billing. Batch is appropriate for a high volume of *small* requests where any one
timing out is cheap.

---

## What happened on 2026-06-15/16 (the incident)

A round-2 review (~405k-token package, `xhigh`) was submitted via Batch (`/v1/responses`). Observed:

- Submitted 06-15 5:34pm CT; sat `in_progress` ~6.7h; final state `completed` with
  `request_counts {completed: 0, failed: 1}`, `error_file` = `insufficient_quota` (429), **no output**.
- The batch status object reported `usage: {}` → a live check said **$0**. **This was wrong.**
- The billing/usage CSVs showed the **same single request executed 6 times** (1 on 06-15, 5 on 06-16),
  each re-ingesting ~405,626 input tokens and generating **13–16k output (reasoning) tokens before
  dying** — then retried. Total billed ≈ **$85** (06-15 batch $15.11 + 06-16 batch $73.70), at full
  rate (no cache discount). The user's dashboard line `$60.844` = the 06-16 input cost alone
  (5 × 405,626 × $30/M).

### Why it failed (documentation-grounded)

- **Not context** — 405k ≪ 1.05M window.
- **Not the output cap** — died at ~14k ≪ 128k, and no `max_output_tokens` was set.
- **Not rate limits** — 405k fits TPM at Tier 3 (500K) and Tier 4 (1M); error code was
  `insufficient_quota`, which is **billing/credit**, not the rate-limit code `rate_limit_exceeded`
  ([rate-limits guide](https://developers.openai.com/api/docs/guides/rate-limits)).
- **Timeout signature** — the 6 attempts died at a **tight 13–16k-output cluster** (16,319 / 13,031 /
  14,181 / 15,151 / 15,949 / 13,126). Consistent cutoff at the same token budget = a **fixed timeout**,
  not random errors (which would scatter) or a rate rejection (which would be ~0 tokens). The model
  page warns Pro requests "may take several minutes… use background mode to avoid timeouts" — and
  **batch has no background mode.**

### Causal chain

> 405k `xhigh` Pro request needs several minutes → **batch has no timeout protection** → each attempt
> times out at ~14k reasoning tokens → **retried (undocumented) and billed each time** → ~$85 burned →
> prepaid credit balance hits $0 → final disposition logs `insufficient_quota`.

### Doc-vs-reality discrepancies to remember

- The [batch guide](https://developers.openai.com/api/docs/guides/batch) says *"You will be charged
  for tokens consumed from any completed requests,"* and documents **no automatic retries**. Reality:
  a *failed* batch retried 6× and billed all of it. **Treat batch failures as potentially billable.**
- The batch status object's `usage` is not a reliable cost figure. Reconcile against the **usage/cost
  export CSVs** (Dashboard → Usage → export), which break down by `model`, `batch` flag, and tokens.

---

## Recommended usage pattern

Reference implementation: [`docs/reviews/2026-06-15-ref-intl-sensitivity/run_gpt55pro_round2_batch.py`](../reviews/2026-06-15-ref-intl-sensitivity/run_gpt55pro_round2_batch.py)
(despite the name, it now supports both live and batch). Stdlib-only (`urllib`); key read from
`OPENAI_API_KEY` in the gitignored project `.env`, never printed.

```bash
# Load the key from the gitignored project .env (direnv is not active in the tool shell)
set -a; . ./.env; set +a

# LIVE + background (RECOMMENDED for large Pro reviews) — create only, get the response id fast:
python run_gpt55pro_round2_batch.py --live --no-poll        # confirms credits work; prints resp_...
python run_gpt55pro_round2_batch.py --recover-response resp_xxx   # fetch when done; writes output + cost

# DRY RUN (assemble package + cost estimate, NO call):
python run_gpt55pro_round2_batch.py                          # batch dry run
python run_gpt55pro_round2_batch.py --live                  # live dry run + cost

# BATCH (only for many small requests; avoid for one large review):
python run_gpt55pro_round2_batch.py --send --no-poll        # submit; prints batch id
python run_gpt55pro_round2_batch.py --recover batch_xxx     # fetch when done
```

Hardening to add for live submissions of large packages:

- **`max_output_tokens`** — optional, and a **hard truncation** (→ status `incomplete`,
  `reason: max_output_tokens`), NOT a graceful wrap-up; reasoning counts first, so too-low a cap can
  return an empty-but-billed answer ([reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)).
  A single live response is already bounded at the model's 128k output ceiling (~$23), so either **omit
  the cap** (rely on that ceiling + cancel) or set a **high backstop ~115k** that won't truncate a
  legitimate answer. Do NOT set it tight as a "budget" — round 1 used ~33k output, but a deeper
  `xhigh` audit can legitimately need far more, and truncation wastes the spend.
- **Background mode** (`"background": true`) so the request survives long generation; poll
  `GET /v1/responses/{id}`; the response id is recoverable if the local poller dies (e.g. machine sleep).
- **De-risk further if needed:** lower effort `xhigh → high` (less reasoning time → lower timeout risk
  and cost) before going bigger.

### Scoping: one comprehensive run vs splitting

Prefer **one comprehensive run** over splitting an end-to-end audit into many focused prompts:

- **Splitting usually costs *more*, not less.** Each slice still needs the shared context
  (methodology, config, the framing/disposition), so you re-send 100k+ tokens per slice and the total
  input balloons past a single 405k run.
- **It loses the cross-cutting checks** that are the point of an end-to-end audit — reconciliations
  that span the pipeline (state = Σ counties; numbers consistent across mortality + migration + GQ).
- **The model handles large single prompts fine.** 405k input is ~39% of the 1.05M context, and the
  round-1 single-prompt review produced ~15 substantive findings. "More than it can handle" was a
  *batch-timeout* failure, not a size/quality one.

Use **"comprehensive first, drill-down on demand"**: run the full review once, then do a *narrow*
follow-up only on an area the full pass flags as needing depth. To de-risk the first pass, lower
**effort** (`xhigh → high`) — don't pre-split.

### Cost estimation (live, full price)

```text
input_cost  = input_tokens  / 1e6 * 30
output_cost = output_tokens / 1e6 * 180     # includes reasoning tokens
```

A ~405k-input review with ~40k output ≈ **$12.2 input + $7.2 output ≈ $20** live. Round-1 (195k in /
33k out) was **$11.72**. Background-mode polling does not add cost; cancelling before completion = **$0**.

---

## Watching it think: streaming + reasoning summaries (observability option)

Background mode (above) is **durable but opaque** — you see `status` only, with token counts at the
end (the `usage` object is empty while `in_progress`). When you'd rather watch the model work in real
time, use **streaming + reasoning summaries.** Good for: iterating on a prompt, lower-stakes runs, or
any time the "watch it think" view is worth more than fire-and-forget simplicity.

**Two parameters:**

- `stream: true` — emit Server-Sent Events as the response is generated.
- `reasoning: {summary: "auto"}` — include a human-readable **summary** of the model's reasoning
  (`auto` = most detailed summarizer available for the model). gpt-5.5-pro does **not** expose its raw
  chain-of-thought — only these summaries. (Raw-CoT streaming events like `response.reasoning_text.delta`
  exist, but only for open-weight models such as `gpt-oss`.)

**Key streaming events** ([streaming events reference](https://platform.openai.com/docs/api-reference/responses-streaming)):

| Event | Meaning |
|-------|---------|
| `response.created` / `response.in_progress` | lifecycle |
| `response.reasoning_summary_text.delta` | a chunk of the **reasoning summary** as it is produced |
| `response.output_text.delta` | a chunk of the **visible answer** |
| `response.completed` | terminal; carries final `usage` (input / output / **reasoning** token counts) |
| `error` | failure |

**You still do not get a precise live token *counter*** — you watch summary text and answer text
*flow*, and exact token counts arrive only at `response.completed`. But you do get a real
"it's reasoning about X now" window that background-poll cannot give.

**Durable AND observable (you don't have to choose):** streaming a long job over one connection is
fragile, but combine **`background: true` + `stream: true`**; if the connection drops, **resume** the
event stream with `GET /v1/responses/{id}?stream=true&starting_after=<sequence_number>` — the
`starting_after` cursor replays from the last event you saw. The job keeps running server-side
regardless, so a dropped connection costs nothing; you just reconnect.

**Caveats:**

- Reasoning summaries are **opt-in** and may require **organization verification** before the API
  returns them.
- Summaries are abstractions, not the raw reasoning tokens.
- The runner here uses background+poll, **not** streaming. Adding the "watch it think" view means
  implementing SSE parsing + the `starting_after` reconnect loop — a moderate addition, worth it only
  when live visibility matters.

**Rule of thumb:** fire-and-forget audit where you just want the result → **background + poll**
(simpler, what the runner does). Want to see it reason, or iterating on a prompt →
**stream + `reasoning.summary: "auto"`** (add `background: true` + `starting_after` resume for long jobs).

## Sequential vs parallel reasoning compute (and where to get "Heavy")

gpt-5.5-pro can spend extra inference compute in **two distinct ways**, which are easy to conflate:

- **Sequential test-time compute** — think *longer in one reasoning chain*. Controlled by
  `reasoning.effort` (`low → medium → high → xhigh`). `xhigh` = maximum sequential depth.
- **Parallel test-time compute** — run **many independent reasoning paths** and aggregate them
  (majority-vote / consensus / best-of). This is what **defines "Pro"**: per the
  [model page](https://developers.openai.com/api/docs/models/gpt-5.5-pro), gpt-5.5-pro is "the same
  underlying model" using "scaled but efficient parallel test-time compute."

These are different axes. `reasoning.effort` is the *sequential* dial; the *parallel* breadth is
intrinsic to the pro model.

**What the API exposes — and doesn't:**

- The only documented `reasoning.*` sub-parameters are **`effort`** and **`summary`**
  ([reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)). There is **no documented
  parameter to increase parallel breadth** — no "Heavy" knob.
- You get gpt-5.5-pro's parallel compute simply by **calling the model**; you cannot dial it.
- **UNRESOLVED (undocumented):** whether API `gpt-5.5-pro` + `xhigh` equals the ChatGPT Pro **"Heavy"**
  mode or something lighter. The authoritative guides never tie `effort` to parallel breadth, so
  **do not assume API == web Heavy.**

**ChatGPT Pro (web)** exposes explicit **Light / Heavy** reasoning dials (Heavy = most parallel compute,
long runtime) — the clearest lever to *maximize* parallel reasoning today.

**Codex CLI does NOT offer the pro/parallel compute.** Its selectable models top out at `gpt-5.5`
(sequential; effort to `xhigh`); `gpt-5.5-pro` is **not** a listed Codex model
([Codex models](https://developers.openai.com/codex/models)), and the subscription dispatch path
(`knowledge`/workspace `REFERENCE-codex-cli-subscription-dispatch.md`) runs `gpt-5.5`, not pro. Great
for agentic coding/review on the subscription; **not** a route to Heavy.

### Where to run what

| Goal | Best path | Why |
|------|-----------|-----|
| Max parallel depth on a **focused, hard** problem | **ChatGPT Pro web, "Heavy"** | explicit Heavy dial; the input is small enough to fit |
| **Broad audit over a large corpus** | **API `gpt-5.5-pro` + `xhigh` + background** | full 1M context (reads *everything*); scriptable, reproducible, exact cost |
| Agentic coding/review on the subscription | **Codex CLI (`gpt-5.5`)** | repo-aware; subscription-billed; not pro |

**The core trade-off:** no single path gives **both** max parallel depth **and** full context. Web
"Heavy" maximizes parallelism but **chunks/retrieves** large uploads (~128k effective, ~10-file limit);
the API reads the full context but has **no Heavy dial**. Power move for a hard problem buried in a big
corpus: use the **API to assemble/curate** the large package, then **web Heavy** to hammer the single
hardest sub-question you extract.

## Pre-flight checklist (large Pro review)

- [ ] Package assembled and **dry-run** checked (token estimate; remember `chars/4` under-counts ~15–20% —
      405k actual vs 347k estimated in the incident).
- [ ] **Live + background mode**, not batch.
- [ ] `max_output_tokens` ceiling set (60–80k).
- [ ] Credits present (a 429 `insufficient_quota` = top up the prepaid balance; it is billing, not rate).
- [ ] `--no-poll` create first to confirm admission + capture the `resp_…` id, then poll/recover.
- [ ] If the local poller may die (machine sleep), keep the `resp_…` id — recover anytime; the request
      runs server-side regardless.
- [ ] After completion, reconcile actual cost against the **usage CSV**, not the status object.

---

## Appendix: how the parallel compute likely works (informed speculation)

> **Not OpenAI-confirmed.** OpenAI does not document the internal mechanism of gpt-5.5-pro's parallel
> test-time compute or the ChatGPT "Heavy" mode. Below is the consensus of qualified outside analysis as
> of June 2026, credibility-tiered. Treat it as a working mental model, not fact.

**The mechanism (consensus view).** gpt-5.5-pro is believed to run **multiple independent reasoning
threads in parallel and converge/merge them into one answer** (the self-consistency / best-of-N /
consensus family) — "the same underlying model, not a separate training run." The nuance specific to
this generation: the parallel budget is **adaptive per-request** — it allocates *more* parallel compute
on queries it judges *harder*. That is the likely reason a well-scaffolded verification task finishes
fast and cheap while a genuinely hard problem runs long.

**"Heavy" as a compute multiplier.** ChatGPT Pro's **Light / Heavy** toggle reportedly maps to internal
"juice levels" — **Light ≈ 5, Heavy ≈ 200** — i.e. Heavy is roughly a **~40× test-time-compute budget**
over Light, exposed as a toggle rather than the originally-planned slider. *(The specific numbers are
leak/analysis, not an OpenAI spec — treat as rumor.)*

**Why the field is going parallel.** The most credible current framing — Nathan Lambert (research
scientist, Allen Institute for AI; *Interconnects*) — is that the **sequential** axis (longer single
chains = `reasoning.effort`) is hitting **diminishing returns**, while **parallel** inference is the live
scaling frontier (enabled by new GPU clusters). So gpt-5.5-pro leaning on parallel compute reflects where
the field is heading, not a one-off quirk.

**Where it helps — and where it may not.** Parallel/consensus compute pays off most on **hard,
verifiable, single-answer** problems (math, code, logic), where a majority vote or verifier signal is
meaningful. Evidence is **mixed for broad knowledge/factual work**: some studies show consensus@k helps
factual reasoning, while others argue test-time scaling is "not effective for knowledge-intensive tasks
yet." Practical implication: **reach for Heavy on a focused hard problem, not a broad multi-part audit**
(matches the "where to run what" table above).

**Credibility tiering:**

- *High:* Nathan Lambert / Interconnects; peer-reviewed arXiv on test-time scaling.
- *Medium:* independent model-drop analyses and review sites; the ChatGPT help center for the Light/Heavy facts.
- *Low / rumor:* the specific "juice 5 / 200" numbers; post-launch "silent downgrade" reports.

**Speculation sources:**
[Lambert — rise of thinking models](https://www.understandingai.org/p/nathan-lambert-on-the-rise-of-thinking) ·
[Lambert — GPT-5 and the arc of progress](https://www.interconnects.ai/p/gpt-5-and-bending-the-arc-of-progress) ·
[GPT-5.5 review (buildfastwithai)](https://www.buildfastwithai.com/blogs/gpt-5-5-review-2026) ·
[PromptLayer — GPT-5 Pro vs Thinking](https://blog.promptlayer.com/gpt-5-vs-gpt-5-pro-vs-gpt-5-thinking-mode/) ·
[Jake Handy — Model Drop: GPT-5.5](https://handyai.substack.com/p/model-drop-gpt-55) ·
[GPT-5.5 in ChatGPT (Help Center)](https://help.openai.com/en/articles/11909943-gpt-53-and-54-in-chatgpt) ·
[arXiv — test-time scaling not effective for knowledge-intensive tasks yet](https://arxiv.org/pdf/2509.06861)

## Sources

- [gpt-5.5-pro model page](https://developers.openai.com/api/docs/models/gpt-5.5-pro) — limits, pricing, tiers, background-mode note
- [Batch API guide](https://developers.openai.com/api/docs/guides/batch) — endpoints, 24h window, "completed requests" billing language, no documented retries
- [Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) — reasoning tokens bill as output; `max_output_tokens`/`incomplete`; reserve ≥25k
- [Rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits) — tiers; `insufficient_quota` (billing) vs `rate_limit_exceeded` (rate)
- Incident data: OpenAI usage/cost exports for 2026-06-15/16 (project "ETL Dev"); batch `batch_6a307dd0c6388190addb109576e19e27`
