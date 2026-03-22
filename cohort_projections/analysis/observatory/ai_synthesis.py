"""Optional AI synthesis helpers for the Projection Observatory.

This module stays self-contained so higher-level Observatory workflows can
wire in an external model later without changing the deterministic
decision spine. It provides three pieces:

1. Structured evidence payload construction.
2. A provider-agnostic synthesis wrapper that can call any compatible model
   client when enabled.
3. A deterministic claim-checking helper that can suppress contradictory
   summaries and fall back to a deterministic brief.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import pandas as pd


class SynthesisProvider(Protocol):
    """Provider-agnostic callable interface for optional synthesis."""

    def __call__(self, prompt: str, payload: dict[str, Any]) -> Any:
        """Return a model response for *payload* using *prompt*."""


@dataclass(frozen=True)
class ClaimCheckResult:
    """Deterministic validation outcome for a synthesized summary."""

    accepted: bool
    suppressed: bool
    issues: tuple[str, ...] = ()
    parsed_output: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynthesisResult:
    """Container for deterministic and optional AI-style Observatory output."""

    payload: dict[str, Any]
    deterministic_summary: str
    ai_summary: str | None
    final_summary: str
    validation: ClaimCheckResult
    used_provider: bool
    provider_enabled: bool


def build_default_claims(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a deterministic claim set from the evidence payload."""
    brief = payload.get("decision_brief", {})
    if not isinstance(brief, dict):
        brief = {}
    claims: list[dict[str, Any]] = []
    subject = _clean_text(brief.get("subject_label"))
    state = _clean_text(brief.get("state"))
    raw_subject_id = _clean_text(brief.get("raw_subject_id"))
    search_id = _clean_text(brief.get("search_id"))

    if state:
        claims.append(
            {
                "claim_id": "state-is",
                "claim_type": "state_is",
                "subject": subject,
                "expected": state,
                "run_id": raw_subject_id,
                "citations": [search_id] if search_id else [],
            }
        )
    if subject or raw_subject_id:
        claims.append(
            {
                "claim_id": "winner-is",
                "claim_type": "winner_is",
                "subject": subject or raw_subject_id,
                "expected": subject or raw_subject_id,
                "run_id": raw_subject_id,
                "citations": [raw_subject_id] if raw_subject_id else [],
            }
        )

    metrics = payload.get("metrics", [])
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_name = _clean_text(metric.get("name"))
            if not metric_name:
                continue
            delta = _as_float(metric.get("delta_vs_champion"))
            if delta is None:
                continue
            expected_direction = (
                "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"
            )
            claims.append(
                {
                    "claim_id": f"metric-{metric_name}",
                    "claim_type": "metric_direction",
                    "subject": subject,
                    "metric": metric_name,
                    "expected": expected_direction,
                    "observed": expected_direction,
                    "run_id": _clean_text(metric.get("run_id")),
                    "value": _as_float(metric.get("value")),
                    "delta_vs_champion": delta,
                    "citations": [metric_name, _clean_text(metric.get("run_id"))]
                    if _clean_text(metric.get("run_id"))
                    else [metric_name],
                }
            )

    citations = payload.get("citations", [])
    if isinstance(citations, list):
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            ref_id = _clean_text(citation.get("ref_id"))
            if ref_id:
                claims.append(
                    {
                        "claim_id": f"citation-{ref_id}",
                        "claim_type": "citation_present",
                        "subject": subject,
                        "expected": ref_id,
                        "run_id": ref_id if citation.get("kind") == "run" else "",
                        "citations": [ref_id],
                    }
                )

    return claims


def _clean_text(value: object) -> str:
    """Return a compact string or an empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return ""
    except TypeError:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _as_float(value: object) -> float | None:
    """Return *value* as a float when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool | None:
    """Return *value* as a boolean when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return None


def _coerce_rows(rows: Any) -> list[dict[str, Any]]:
    """Return a normalized list of row dictionaries."""
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return [cast(dict[str, Any], row.to_dict()) for _, row in rows.iterrows()]
    if isinstance(rows, dict):
        return [cast(dict[str, Any], rows)]
    return [cast(dict[str, Any], row) for row in rows if isinstance(row, dict)]


def _default_citation(kind: str, ref_id: str, label: str = "") -> dict[str, str]:
    """Build a canonical citation record."""
    return {
        "kind": kind,
        "ref_id": ref_id,
        "label": label,
    }


def build_evidence_payload(
    *,
    decision_brief: dict[str, Any] | None = None,
    candidate_rows: Any = None,
    comparison_rows: Any = None,
    recommendation_rows: Any = None,
    session_summary: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured evidence payload for optional synthesis.

    Parameters
    ----------
    decision_brief:
        Canonical dashboard decision brief or a compatible mapping.
    candidate_rows:
        Optional candidate-level rows. These can be a DataFrame or a sequence
        of mappings.
    comparison_rows:
        Optional comparison rows with metric values and deltas.
    recommendation_rows:
        Optional recommendation rows for next-step suggestions.
    session_summary:
        Optional session-level summary.
    context:
        Optional extra context to persist in the payload.
    """
    brief = dict(decision_brief or {})
    comparison_records = _coerce_rows(comparison_rows)
    candidate_records = _coerce_rows(candidate_rows)
    recommendation_records = _coerce_rows(recommendation_rows)
    session = dict(session_summary or {})
    payload_context = dict(context or {})

    subject_label = _clean_text(
        brief.get("primary_subject_label")
        or brief.get("subject_label")
        or session.get("primary_subject_label")
    )
    raw_subject_id = _clean_text(
        brief.get("raw_subject_id")
        or brief.get("recommendation_candidate_id")
        or session.get("recommendation_candidate_id")
        or payload_context.get("raw_subject_id")
    )
    search_id = _clean_text(
        brief.get("search_id") or session.get("search_id") or payload_context.get("search_id")
    )
    state = _clean_text(
        brief.get("state")
        or brief.get("decision_state")
        or brief.get("session_decision_state")
        or session.get("session_decision_state")
    )
    headline = _clean_text(brief.get("headline") or brief.get("session_headline"))
    main_reason = _clean_text(brief.get("main_reason") or brief.get("explanation"))
    main_gain = _clean_text(brief.get("main_gain"))
    main_tradeoff = _clean_text(brief.get("main_tradeoff"))
    next_action = _clean_text(
        brief.get("recommended_next_step")
        or brief.get("recommended_action")
        or session.get("session_recommendation")
    )
    confidence_label = _clean_text(brief.get("confidence_label") or brief.get("confidence"))
    evidence_quality = _clean_text(brief.get("evidence_quality"))
    operational_label = _clean_text(brief.get("operational_evidence_label"))
    operational_summary = _clean_text(brief.get("operational_evidence_summary"))
    safe_to_recommend = _as_bool(brief.get("safe_to_recommend"))

    metrics: list[dict[str, Any]] = []
    citations: list[dict[str, str]] = []
    run_ids: set[str] = set()

    for row in comparison_records:
        metric_name = _clean_text(row.get("metric") or row.get("primary_metric_name"))
        if not metric_name:
            continue
        metric_value = _as_float(
            row.get("value")
            if row.get("value") is not None
            else row.get(metric_name)
            if metric_name in row
            else row.get("selected_value")
        )
        delta = _as_float(
            row.get("delta_vs_champion")
            if row.get("delta_vs_champion") is not None
            else row.get("delta_value")
            if row.get("delta_value") is not None
            else row.get(f"delta_{metric_name}")
        )
        run_id = _clean_text(row.get("run_id"))
        if run_id:
            run_ids.add(run_id)
        lower_is_better = _as_bool(row.get("lower_is_better"))
        if lower_is_better is None:
            lower_is_better = True
        metrics.append(
            {
                "name": metric_name,
                "label": _clean_text(row.get("label") or metric_name.replace("_", " ").title()),
                "value": metric_value,
                "delta_vs_champion": delta,
                "lower_is_better": lower_is_better,
                "run_id": run_id,
                "role": _clean_text(row.get("role") or row.get("status")),
            }
        )
        if run_id:
            citations.append(_default_citation("run", run_id, _clean_text(row.get("label"))))
        citations.append(_default_citation("metric", metric_name, _clean_text(row.get("label"))))

    for row in candidate_records:
        run_id = _clean_text(row.get("run_id"))
        if run_id:
            run_ids.add(run_id)
            citations.append(_default_citation("run", run_id, _clean_text(row.get("candidate_id"))))

    for row in recommendation_records:
        metric_name = _clean_text(row.get("metric") or row.get("parameter"))
        if metric_name:
            citations.append(
                _default_citation("metric", metric_name, _clean_text(row.get("label")))
            )

    if raw_subject_id:
        run_ids.add(raw_subject_id)
        citations.append(_default_citation("run", raw_subject_id, subject_label))

    if search_id:
        citations.append(_default_citation("session", search_id, "search session"))

    payload = {
        "version": 1,
        "generated_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "source": "Projection Observatory",
        "context": payload_context,
        "decision_brief": {
            "subject_label": subject_label,
            "raw_subject_id": raw_subject_id,
            "search_id": search_id,
            "state": state,
            "headline": headline,
            "main_reason": main_reason,
            "main_gain": main_gain,
            "main_tradeoff": main_tradeoff,
            "confidence_label": confidence_label,
            "evidence_quality": evidence_quality,
            "operational_evidence_label": operational_label,
            "operational_evidence_summary": operational_summary,
            "safe_to_recommend": safe_to_recommend,
            "next_action": next_action,
        },
        "metrics": metrics,
        "runs": sorted(run_ids),
        "citations": citations,
        "review_checklist": list(brief.get("review_checklist", []))
        if isinstance(brief.get("review_checklist"), list)
        else [],
        "tradeoffs": list(brief.get("tradeoffs", []))
        if isinstance(brief.get("tradeoffs"), list)
        else [],
        "recommendations": recommendation_records,
        "session_summary": session,
        "candidate_rows": candidate_records,
    }
    payload["claims"] = build_default_claims(payload)
    return payload


def build_deterministic_summary(payload: dict[str, Any]) -> str:
    """Build a concise, deterministic Observatory summary."""
    brief = payload.get("decision_brief", {})
    if not isinstance(brief, dict):
        brief = {}
    metrics = payload.get("metrics", [])
    if not isinstance(metrics, list):
        metrics = []
    recommendations = payload.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    subject = _clean_text(brief.get("subject_label") or "Current candidate")
    headline = _clean_text(brief.get("headline"))
    main_reason = _clean_text(brief.get("main_reason"))
    next_action = _clean_text(brief.get("next_action"))
    evidence_quality = _clean_text(brief.get("evidence_quality"))
    operational_summary = _clean_text(brief.get("operational_evidence_summary"))

    parts = [f"Subject: {subject}."]
    if headline:
        parts.append(headline)
    if main_reason:
        parts.append(f"Reason: {main_reason}")
    if metrics:
        metric_bits: list[str] = []
        for metric in metrics[:4]:
            if not isinstance(metric, dict):
                continue
            metric_name = _clean_text(metric.get("name"))
            delta = _as_float(metric.get("delta_vs_champion"))
            if not metric_name or delta is None:
                continue
            direction = "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"
            metric_bits.append(f"{metric_name} {direction} by {abs(delta):.3f}")
        if metric_bits:
            parts.append("Key metrics: " + "; ".join(metric_bits) + ".")
    if evidence_quality:
        parts.append(f"Evidence quality: {evidence_quality}.")
    if operational_summary:
        parts.append(f"Operational note: {operational_summary}")
    if next_action:
        parts.append(f"Next action: {next_action}")
    if recommendations:
        first = recommendations[0]
        if isinstance(first, dict):
            rec_text = _clean_text(first.get("rationale") or first.get("text"))
            if rec_text:
                parts.append(f"Next experiment: {rec_text}")
    return " ".join(part for part in parts if part).strip()


def build_synthesis_prompt(payload: dict[str, Any]) -> str:
    """Build a model prompt for external AI providers.

    The prompt asks the provider to return JSON only with summary, claims,
    counterarguments, and review questions. The claim checker relies on that
    structure when present.
    """
    schema = {
        "summary": "string",
        "claims": [
            {
                "subject": "string",
                "metric": "string",
                "direction": "improved|regressed|unchanged",
                "run_id": "string",
                "value": "number|null",
                "delta_vs_champion": "number|null",
                "citations": ["run-id-or-metric-key"],
            }
        ],
        "counterarguments": ["string"],
        "review_questions": ["string"],
    }
    return (
        "You are drafting an Observatory review note from structured evidence.\n"
        "Return JSON only. Do not invent facts.\n"
        "Use only the payload below and cite any factual claim with run IDs or metric keys from the payload.\n"
        "If a claim is not supported by the payload, omit it.\n\n"
        f"Required JSON schema: {json.dumps(schema, indent=2)}\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _parse_ai_output(ai_output: Any) -> dict[str, Any]:
    """Normalize provider output to a dictionary."""
    if ai_output is None:
        return {}
    if isinstance(ai_output, dict):
        return cast(dict[str, Any], ai_output)
    if isinstance(ai_output, str):
        text = ai_output.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"summary": text}
        if isinstance(parsed, dict):
            return parsed
        return {"summary": text}
    return {}


def _metric_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a metric lookup keyed by metric name."""
    metrics = payload.get("metrics", [])
    lookup: dict[str, dict[str, Any]] = {}
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            name = _clean_text(metric.get("name"))
            if name:
                lookup[name] = metric
    return lookup


def _payload_run_ids(payload: dict[str, Any]) -> set[str]:
    """Return all known run IDs from the payload."""
    runs = payload.get("runs", [])
    if isinstance(runs, list):
        return {str(run_id) for run_id in runs if _clean_text(run_id)}
    return set()


def _payload_reference_ids(payload: dict[str, Any]) -> set[str]:
    """Return all citation and evidence identifiers that the payload exposes."""
    identifiers = set(_payload_run_ids(payload))
    metrics = payload.get("metrics", [])
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_name = _clean_text(metric.get("name"))
            if metric_name:
                identifiers.add(metric_name)
            run_id = _clean_text(metric.get("run_id"))
            if run_id:
                identifiers.add(run_id)
    citations = payload.get("citations", [])
    if isinstance(citations, list):
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            ref_id = _clean_text(citation.get("ref_id"))
            if ref_id:
                identifiers.add(ref_id)
    brief = payload.get("decision_brief", {})
    if isinstance(brief, dict):
        subject = _clean_text(brief.get("subject_label"))
        raw_subject_id = _clean_text(brief.get("raw_subject_id"))
        search_id = _clean_text(brief.get("search_id"))
        if subject:
            identifiers.add(subject)
        if raw_subject_id:
            identifiers.add(raw_subject_id)
        if search_id:
            identifiers.add(search_id)
    return identifiers


def _summary_mentions(summary: str, candidate: str) -> bool:
    """Return whether *summary* explicitly mentions *candidate*."""
    if not summary or not candidate:
        return False
    return candidate.lower() in summary.lower()


def _direction_supported(summary: str, metric: dict[str, Any]) -> bool:
    """Check whether a summary's directional language matches a metric delta."""
    delta = _as_float(metric.get("delta_vs_champion"))
    if delta is None:
        return True
    lower_is_better = _as_bool(metric.get("lower_is_better"))
    if lower_is_better is None:
        lower_is_better = True

    positive_terms = {"improved", "better", "lower", "reduced", "declined", "decreased"}
    negative_terms = {"regressed", "worse", "higher", "increased", "weaker", "declined"}
    lower_summary = summary.lower()

    improved = delta < 0 if lower_is_better else delta > 0
    regressed = delta > 0 if lower_is_better else delta < 0

    if any(term in lower_summary for term in positive_terms) and regressed:
        return False
    if any(term in lower_summary for term in negative_terms) and improved:
        return False
    if "no regression" in lower_summary or "no regressions" in lower_summary:
        return not regressed
    return True


def validate_synthesis_output(
    payload: dict[str, Any],
    ai_output: dict[str, Any] | str | None,
) -> ClaimCheckResult:
    """Validate provider claims against deterministic Observatory evidence."""
    issues: list[str] = []
    metric_lookup = _metric_lookup(payload)
    payload_run_ids = _payload_run_ids(payload)
    reference_ids = _payload_reference_ids(payload)
    parsed_output = _parse_ai_output(ai_output)
    summary = _clean_text(parsed_output.get("summary"))
    claims = parsed_output.get("claims", [])
    if not claims:
        claims = build_default_claims(payload)

    if isinstance(claims, list):
        for index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                issues.append(f"Claim {index} is not a mapping.")
                continue
            metric_name = _clean_text(claim.get("metric"))
            run_id = _clean_text(claim.get("run_id"))
            claim_type = _clean_text(claim.get("claim_type"))
            expected = _clean_text(claim.get("expected"))
            value = _as_float(claim.get("value"))
            delta = _as_float(claim.get("delta_vs_champion"))
            citations = claim.get("citations", [])
            if metric_name and metric_name not in metric_lookup:
                issues.append(f"Claim {index} references unknown metric '{metric_name}'.")
            if claim_type != "citation_present" and run_id and run_id not in payload_run_ids:
                issues.append(f"Claim {index} references unknown run_id '{run_id}'.")
            if claim_type == "state_is":
                expected_state = _clean_text(expected)
                if expected_state and summary:
                    blocked_terms = {"blocked", "do not promote", "failed hard gate", "not ready"}
                    review_terms = {"recommended", "ready for review", "best candidate"}
                    lower_summary = summary.lower()
                    if expected_state in {"recommended", "ready_for_review"} and any(
                        term in lower_summary for term in blocked_terms
                    ):
                        issues.append(
                            f"Claim {index} contradicts the expected decision state '{expected_state}'."
                        )
                    if expected_state in {"blocked_by_data_or_runtime", "failed_hard_gate"} and any(
                        term in lower_summary for term in review_terms
                    ):
                        issues.append(
                            f"Claim {index} contradicts the expected decision state '{expected_state}'."
                        )
            elif claim_type == "winner_is":
                expected_subject = _clean_text(expected)
                if (
                    expected_subject
                    and summary
                    and not _summary_mentions(summary, expected_subject)
                    and any(
                        term in summary.lower()
                        for term in {"best candidate", "front-runner", "winner", "recommended"}
                    )
                ):
                    issues.append(
                        f"Claim {index} does not match the declared winner '{expected_subject}'."
                    )
            elif metric_name and metric_name in metric_lookup:
                metric = metric_lookup[metric_name]
                if not _direction_supported(summary or expected, metric):
                    issues.append(
                        f"Claim {index} contradicts the metric direction for '{metric_name}'."
                    )
                metric_value = _as_float(metric.get("value"))
                if (
                    value is not None
                    and metric_value is not None
                    and abs(value - metric_value) > 0.0001
                ):
                    issues.append(f"Claim {index} value does not match '{metric_name}'.")
                metric_delta = _as_float(metric.get("delta_vs_champion"))
                if (
                    delta is not None
                    and metric_delta is not None
                    and abs(delta - metric_delta) > 0.0001
                ):
                    issues.append(f"Claim {index} delta does not match '{metric_name}'.")
            elif claim_type == "citation_present":
                expected_ref = _clean_text(expected)
                if expected_ref and expected_ref not in reference_ids:
                    issues.append(f"Claim {index} cites unknown reference '{expected_ref}'.")
            if citations and isinstance(citations, list):
                for citation in citations:
                    text = _clean_text(citation)
                    if not text:
                        continue
                    if text not in reference_ids:
                        issues.append(f"Claim {index} cites unknown reference '{text}'.")
    elif claims:
        issues.append("Claims must be provided as a list.")

    if summary:
        for metric_name, metric in metric_lookup.items():
            if not _summary_mentions(summary, metric_name):
                continue
            if not _direction_supported(summary, metric):
                issues.append(f"Summary contradicts the payload direction for '{metric_name}'.")
        mentioned_ids = {
            ref_id for ref_id in re.findall(r"(?:br|exp)-[A-Za-z0-9._-]+", summary) if ref_id
        }
        unknown_ids = mentioned_ids - reference_ids
        if unknown_ids:
            issues.append(
                "Summary references unknown evidence identifier(s): "
                + ", ".join(sorted(unknown_ids))
            )

    accepted = not issues
    return ClaimCheckResult(
        accepted=accepted,
        suppressed=not accepted,
        issues=tuple(dict.fromkeys(issues)),
        parsed_output=parsed_output,
    )


def synthesize_observatory_summary(
    payload: dict[str, Any],
    *,
    enabled: bool = False,
    provider: SynthesisProvider | None = None,
) -> SynthesisResult:
    """Return a deterministic or optionally AI-generated Observatory summary."""
    deterministic_summary = build_deterministic_summary(payload)
    provider_enabled = bool(enabled)
    if not provider_enabled or provider is None:
        return SynthesisResult(
            payload=payload,
            deterministic_summary=deterministic_summary,
            ai_summary=None,
            final_summary=deterministic_summary,
            validation=ClaimCheckResult(
                accepted=True,
                suppressed=False,
                issues=(),
                parsed_output={},
            ),
            used_provider=False,
            provider_enabled=provider_enabled,
        )

    prompt = build_synthesis_prompt(payload)
    ai_output = _parse_ai_output(provider(prompt, payload))
    validation = validate_synthesis_output(payload, ai_output)
    ai_summary = _clean_text(ai_output.get("summary"))
    if validation.suppressed or not ai_summary:
        return SynthesisResult(
            payload=payload,
            deterministic_summary=deterministic_summary,
            ai_summary=ai_summary or None,
            final_summary=deterministic_summary,
            validation=validation,
            used_provider=True,
            provider_enabled=provider_enabled,
        )

    return SynthesisResult(
        payload=payload,
        deterministic_summary=deterministic_summary,
        ai_summary=ai_summary,
        final_summary=ai_summary,
        validation=validation,
        used_provider=True,
        provider_enabled=provider_enabled,
    )
