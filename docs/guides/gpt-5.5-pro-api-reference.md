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
3. **Set a `max_output_tokens` ceiling** (e.g. 60–80k) as a runaway-cost guardrail. Reasoning bills
   as output at $180/M; without a cap a runaway reasoning trace is uncapped spend.
4. **Size is rarely the limit.** Context window is 1,050,000 tokens; a 400k-token package uses ~40%.
   Don't shrink the package for "size" reasons — check the actual limits first.
5. **Cancel = $0.** A live background response cancelled before completion reports `usage: 0` and is
   not billed. This is the main reason live+background beats batch for control.

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
- **`max_output_tokens`** ceiling (e.g. 60–80k): bounds reasoning+answer, caps worst-case output cost.
  Reserve ≥25k for reasoning ([reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)).
  A complete review of this size produced ~33k output in round 1, so 80k is generous headroom.
- **Background mode** (`"background": true`) so the request survives long generation; poll
  `GET /v1/responses/{id}`; the response id is recoverable if the local poller dies (e.g. machine sleep).
- **De-risk further if needed:** lower effort `xhigh → high` (less reasoning time → lower timeout risk
  and cost) before going bigger.

### Cost estimation (live, full price)

```
input_cost  = input_tokens  / 1e6 * 30
output_cost = output_tokens / 1e6 * 180     # includes reasoning tokens
```

A ~405k-input review with ~40k output ≈ **$12.2 input + $7.2 output ≈ $20** live. Round-1 (195k in /
33k out) was **$11.72**. Background-mode polling does not add cost; cancelling before completion = **$0**.

---

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

## Sources

- [gpt-5.5-pro model page](https://developers.openai.com/api/docs/models/gpt-5.5-pro) — limits, pricing, tiers, background-mode note
- [Batch API guide](https://developers.openai.com/api/docs/guides/batch) — endpoints, 24h window, "completed requests" billing language, no documented retries
- [Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) — reasoning tokens bill as output; `max_output_tokens`/`incomplete`; reserve ≥25k
- [Rate limits guide](https://developers.openai.com/api/docs/guides/rate-limits) — tiers; `insufficient_quota` (billing) vs `rate_limit_exceeded` (rate)
- Incident data: OpenAI usage/cost exports for 2026-06-15/16 (project "ETL Dev"); batch `batch_6a307dd0c6388190addb109576e19e27`
