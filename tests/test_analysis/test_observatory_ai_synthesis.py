"""Tests for optional Observatory AI synthesis helpers."""

from __future__ import annotations

from cohort_projections.analysis.observatory.ai_synthesis import (
    build_evidence_payload,
    build_synthesis_prompt,
    synthesize_observatory_summary,
    validate_synthesis_output,
)


def _build_payload() -> dict[str, object]:
    """Return a compact synthetic evidence payload."""
    decision_brief = {
        "primary_subject_label": "Candidate A",
        "raw_subject_id": "run-a",
        "search_id": "search-1",
        "decision_state": "recommended",
        "headline": "Candidate A is ready for review.",
        "main_reason": "County error improved without a hard-gate regression.",
        "main_gain": "County MAPE improved.",
        "main_tradeoff": "Rural MAPE regressed slightly.",
        "confidence_label": "High confidence",
        "evidence_quality": "Benchmark-backed and reviewable",
        "safe_to_recommend": True,
        "next_action": "Review before recommending",
        "review_checklist": [
            {"label": "Benchmark complete", "status": "yes", "detail": "Bundle is complete."},
            {"label": "No hard-gate regression", "status": "yes", "detail": "No hard gate failed."},
        ],
    }
    comparison_rows = [
        {
            "run_id": "run-a",
            "metric": "county_mape_overall",
            "value": 8.2,
            "delta_vs_champion": -0.3,
            "label": "County error",
            "lower_is_better": True,
        },
        {
            "run_id": "run-a",
            "metric": "county_mape_rural",
            "value": 7.1,
            "delta_vs_champion": 0.1,
            "label": "Rural county error",
            "lower_is_better": True,
        },
    ]
    candidate_rows = [{"run_id": "run-a", "candidate_id": "cand-a"}]
    recommendation_rows = [{"metric": "county_mape_overall", "label": "Try a narrower window"}]
    return build_evidence_payload(
        decision_brief=decision_brief,
        comparison_rows=comparison_rows,
        candidate_rows=candidate_rows,
        recommendation_rows=recommendation_rows,
        context={"search_id": "search-1"},
    )


def test_build_evidence_payload_normalizes_structured_sections() -> None:
    payload = _build_payload()

    assert payload["decision_brief"]["subject_label"] == "Candidate A"
    assert payload["decision_brief"]["raw_subject_id"] == "run-a"
    assert payload["metrics"][0]["name"] == "county_mape_overall"
    assert any(citation["ref_id"] == "run-a" for citation in payload["citations"])
    assert any(claim["claim_type"] == "state_is" for claim in payload["claims"])
    assert any(claim["claim_type"] == "metric_direction" for claim in payload["claims"])


def test_validate_synthesis_output_accepts_structured_claims() -> None:
    payload = _build_payload()
    output = {
        "summary": "Candidate A is the current front-runner.",
        "claims": [
            {
                "claim_type": "state_is",
                "expected": "recommended",
                "citations": ["search-1"],
            },
            {
                "claim_type": "winner_is",
                "expected": "Candidate A",
                "citations": ["run-a"],
            },
            {
                "claim_type": "metric_direction",
                "metric": "county_mape_overall",
                "expected": "improved",
                "citations": ["county_mape_overall", "run-a"],
            },
        ],
        "counterarguments": ["Rural regression still needs review."],
        "review_questions": ["Should rural counties stay within threshold?"],
    }

    result = validate_synthesis_output(payload, output)

    assert result.accepted is True
    assert result.suppressed is False
    assert result.issues == ()


def test_validate_synthesis_output_suppresses_contradictory_summary_text() -> None:
    payload = _build_payload()

    result = validate_synthesis_output(
        payload,
        "Candidate A is blocked and should not be promoted. County error improved.",
    )

    assert result.accepted is False
    assert result.suppressed is True
    assert any("decision state" in issue.lower() for issue in result.issues)


def test_synthesize_observatory_summary_uses_provider_when_valid() -> None:
    payload = _build_payload()

    def _provider(prompt: str, provider_payload: dict[str, object]) -> dict[str, object]:
        assert "Return JSON only" in prompt
        assert provider_payload["decision_brief"]["raw_subject_id"] == "run-a"
        return {
            "summary": "AI synthesis: Candidate A improved overall and remains reviewable.",
            "claims": [
                {
                    "claim_type": "state_is",
                    "expected": "recommended",
                    "citations": ["search-1"],
                },
                {
                    "claim_type": "metric_direction",
                    "metric": "county_mape_overall",
                    "expected": "improved",
                    "citations": ["county_mape_overall", "run-a"],
                },
            ],
            "counterarguments": ["Rural regression is still visible."],
            "review_questions": ["Is the rural regression acceptable?"],
        }

    result = synthesize_observatory_summary(payload, enabled=True, provider=_provider)

    assert result.used_provider is True
    assert (
        result.final_summary == "AI synthesis: Candidate A improved overall and remains reviewable."
    )
    assert result.validation.accepted is True
    assert result.validation.suppressed is False


def test_synthesize_observatory_summary_falls_back_when_provider_contradicts() -> None:
    payload = _build_payload()

    def _provider(prompt: str, provider_payload: dict[str, object]) -> str:
        del prompt, provider_payload
        return "Candidate A is blocked and not reviewable."

    result = synthesize_observatory_summary(payload, enabled=True, provider=_provider)

    assert result.used_provider is True
    assert result.final_summary == result.deterministic_summary
    assert result.ai_summary == "Candidate A is blocked and not reviewable."
    assert result.validation.suppressed is True


def test_build_synthesis_prompt_mentions_json_only_and_payload() -> None:
    payload = _build_payload()
    prompt = build_synthesis_prompt(payload)

    assert "Return JSON only" in prompt
    assert "claims" in prompt
    assert "Candidate A" in prompt
