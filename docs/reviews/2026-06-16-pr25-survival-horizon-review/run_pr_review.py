#!/usr/bin/env python3
"""PR-framed GPT-5.5 Pro review of PR #25 (ADR-068 survival-horizon amendment).

Assembles `review_brief.md` + the full PR text diff (`git diff master...HEAD`, binaries
excluded) and submits it to gpt-5.5-pro via the Responses API in **background** mode
(live, NOT batch — see docs/guides/gpt-5.5-pro-api-reference.md for the $85 batch lesson).

Stdlib only (urllib). The API key is read from OPENAI_API_KEY and never printed.

SAFETY: the default action is a DRY RUN — it assembles the package, prints size + cost
estimate, writes the exact package to `pr25_review_input.txt`, and makes NO API call.
A real call happens ONLY with an explicit `--send`.

Usage:
    python run_pr_review.py                 # DRY RUN: assemble + cost, no call
    python run_pr_review.py --send          # submit background review (needs OPENAI_API_KEY)
    python run_pr_review.py --send --no-poll  # submit and exit without polling
    python run_pr_review.py --recover-response <id>   # fetch a completed background response
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

MODEL = "gpt-5.5-pro"
REASONING_EFFORT = "xhigh"   # highest; live falls back to high/default on 400
MAX_OUTPUT = None            # no cap — reasoning bills as output; a cap can truncate (hard wall)
LIVE_IN_PRICE, LIVE_OUT_PRICE = 30.0, 180.0   # $/1M live in/out
API = "https://api.openai.com"
BASE_REF = "master"
BRIEF_PATH = os.path.join(HERE, "review_brief.md")
INPUT_RECORD = os.path.join(HERE, "pr25_review_input.txt")
OUTPUT_PATH = os.path.join(HERE, "gpt55pro_pr25_review_output.md")

# Binaries can't be diffed as text; their numbers live in the included public CSV.
_EXCLUDE = [":(exclude)*.docx", ":(exclude)*.png", ":(exclude)*.xlsx", ":(exclude)*.pdf"]


def _pr_diff() -> str:
    """Full PR text diff (base...HEAD), binaries excluded, generated fresh."""
    out = subprocess.run(
        ["git", "diff", f"{BASE_REF}...HEAD", "--", ".", *_EXCLUDE],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return out.stdout


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def assemble() -> str:
    with open(BRIEF_PATH, encoding="utf-8") as fh:
        brief = fh.read()
    diff = _pr_diff()
    bar = "=" * 80
    pkg = (
        f"{brief}\n\n{bar}\n=== PR #25 DIFF — `git diff {BASE_REF}...HEAD` "
        f"@ {_head()} (binaries excluded) ===\n{bar}\n\n{diff}"
    )
    with open(INPUT_RECORD, "w", encoding="utf-8") as fh:
        fh.write(pkg)
    return pkg


def _post_json(url: str, key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _get_json(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _extract_text(body: dict) -> str:
    if body.get("output_text"):
        return body["output_text"]
    chunks = []
    for item in body.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                chunks.append(c.get("text", ""))
    return "\n".join(chunks)


def _write_output(text: str, usage: dict) -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text or "(empty response)")
    print("review written:", OUTPUT_PATH)
    it, ot = usage.get("input_tokens"), usage.get("output_tokens")
    rt = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    print("usage:", json.dumps(usage))
    if rt is not None:
        print(f"  reasoning tokens: {rt:,} of {ot:,} output")
    if it and ot:
        print(f"ACTUAL cost = ${it/1e6*LIVE_IN_PRICE + ot/1e6*LIVE_OUT_PRICE:.2f} "
              f"(input {it:,} @ ${LIVE_IN_PRICE:.0f}/M, output {ot:,} @ ${LIVE_OUT_PRICE:.0f}/M)")


def _recover(rid: str, key: str) -> None:
    data = _get_json(f"{API}/v1/responses/{rid}", key)
    print(f"response {rid}: status={data.get('status')}")
    if data.get("status") == "completed":
        _write_output(_extract_text(data), data.get("usage", {}))
    elif data.get("status") in ("failed", "cancelled", "incomplete"):
        print("terminal non-success:",
              json.dumps(data.get("error") or data.get("incomplete_details") or {})[:800])
    else:
        print("not complete yet; re-run --recover-response later.")


def _estimate(pkg: str) -> None:
    est_in = len(pkg) // 4
    in_cost = est_in / 1e6 * LIVE_IN_PRICE
    out_guess = 40000
    print(f"package: {len(pkg):,} chars ~= {est_in:,} input tokens "
          f"({est_in*100/1_050_000:.1f}% of the 1.05M window)")
    print(f"LIVE est: input ${in_cost:.2f} + ~{out_guess//1000}k output @ ${LIVE_OUT_PRICE:.0f}/M "
          f"= ~${in_cost + out_guess/1e6*LIVE_OUT_PRICE:.2f} typical total")
    print(f"package recorded at: {INPUT_RECORD}")


def send(no_poll: bool) -> None:
    pkg = assemble()
    _estimate(pkg)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not set — aborting, NO call made.")
        sys.exit(2)
    created, chosen = None, None
    for eff in [REASONING_EFFORT, "high", None]:
        body = {"model": MODEL, "input": pkg, "background": True}
        if eff:
            body["reasoning"] = {"effort": eff}
        if MAX_OUTPUT:
            body["max_output_tokens"] = MAX_OUTPUT
        try:
            print(f"creating background response (effort={eff or 'default'})...")
            created = _post_json(f"{API}/v1/responses", key, body)
            chosen = eff or "default"
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 400 and eff and ("effort" in msg.lower() or "reasoning" in msg.lower()):
                print(f"effort={eff} rejected (400); lowering...")
                continue
            print("create FAILED", e.code, msg[:1500])
            sys.exit(3)
    rid = created.get("id")
    print(f"created response id={rid} status={created.get('status')} effort={chosen}")
    print(f"  recover later:  python {os.path.basename(__file__)} --recover-response {rid}")
    if no_poll:
        print("--no-poll set; response generating in the background, not polling.")
        return
    waited, deadline, status, data = 0, 10800, created.get("status"), created
    while status in ("queued", "in_progress"):
        time.sleep(20)
        waited += 20
        if waited > deadline:
            print(f"poll timeout after {waited}s; recover via --recover-response {rid}")
            return
        try:
            data = _get_json(f"{API}/v1/responses/{rid}", key)
        except urllib.error.HTTPError as e:
            print("poll HTTPError", e.code, e.read().decode("utf-8", "replace")[:800])
            break
        status = data.get("status")
        print(f"  [{waited}s] status={status}")
    print(f"final status={status}")
    if status == "completed":
        _write_output(_extract_text(data), data.get("usage", {}))
    else:
        print("non-success:", json.dumps(data.get("error") or data.get("incomplete_details") or {})[:800])


def main() -> None:
    if "--recover-response" in sys.argv:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(2)
        _recover(sys.argv[sys.argv.index("--recover-response") + 1], key)
        return
    if "--send" in sys.argv:
        send("--no-poll" in sys.argv)
        return
    # Default: DRY RUN — assemble, estimate, record package; make NO API call.
    pkg = assemble()
    _estimate(pkg)
    print("\nDRY RUN — no API call made. Re-run with --send (needs OPENAI_API_KEY) to submit.")


if __name__ == "__main__":
    main()
