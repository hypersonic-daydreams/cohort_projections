"""Decision-support helpers for the Projection Observatory.

Turns raw benchmark/search outputs into plain-language candidate, session, and
benchmark summaries that the dashboard can render without exposing internal
artifact quirks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

_DECISION_META: dict[str, dict[str, str]] = {
    "recommended": {
        "label": "Recommended",
        "confidence": "high",
    },
    "ready_for_review": {
        "label": "Ready for review",
        "confidence": "high",
    },
    "mixed_signal": {
        "label": "Mixed signal",
        "confidence": "medium",
    },
    "blocked_by_data_or_runtime": {
        "label": "Blocked",
        "confidence": "low",
    },
    "failed_hard_gate": {
        "label": "Failed hard gate",
        "confidence": "high",
    },
    "not_executed": {
        "label": "Not executed",
        "confidence": "low",
    },
}

_PATH_PATTERN = re.compile(r"(/[^\s)]+)")
_MISSING_FILE_PATTERN = re.compile(r"FileNotFoundError:\s*.*?not found:\s*(.+)")


def _as_float(value: object) -> float | None:
    """Return *value* as float when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _clean_text(value: object) -> str:
    """Return a compact string for display or empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return ""
    except TypeError:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _best_metric(row: Mapping[str, Any]) -> tuple[str, float | None]:
    """Return the most relevant metric name/delta pair for one candidate row."""
    preferred = _clean_text(row.get("primary_metric_name"))
    if preferred:
        delta = _as_float(row.get(f"delta_{preferred}"))
        if delta is not None:
            return preferred, delta
        direct = _as_float(row.get(preferred))
        if direct is not None:
            return preferred, direct

    for name in (
        "county_mape_overall",
        "state_ape_recent_short",
        "county_mape_rural",
        "county_mape_bakken",
        "county_mape_urban_college",
    ):
        delta = _as_float(row.get(f"delta_{name}"))
        if delta is not None:
            return name, delta
    return preferred or "county_mape_overall", None


def _parse_result_json_from_log(log_text: str) -> dict[str, Any]:
    """Extract the trailing JSON result object from a run-experiment stdout log."""
    if not log_text:
        return {}
    marker = '{\n  "outcome"'
    idx = log_text.rfind(marker)
    if idx == -1:
        marker = '{"outcome"'
        idx = log_text.rfind(marker)
    if idx == -1:
        return {}
    snippet = log_text[idx:].strip()
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_text(path: Path | None) -> str:
    """Read a UTF-8 text file when it exists."""
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_candidate_log(project_root: Path | None, relative_path: str) -> str:
    """Read one session log referenced by a project-relative path."""
    if project_root is None:
        return ""
    path_text = _clean_text(relative_path)
    if not path_text:
        return ""
    return _read_text(project_root / path_text)


def _extract_blocker_detail(text: str) -> tuple[str, str]:
    """Infer a blocker type/detail from free-text failure evidence."""
    message = _clean_text(text)
    if not message:
        return "", ""

    missing_match = _MISSING_FILE_PATTERN.search(text)
    if missing_match:
        path = _clean_text(missing_match.group(1))
        return "missing_input_data", path

    if "not in METHOD_DISPATCH" in text or "code changes required" in text:
        return "code_registration_required", "Method is not registered in METHOD_DISPATCH."

    if "Benchmark suite failed" in text:
        path_match = _PATH_PATTERN.search(text)
        detail = _clean_text(path_match.group(1)) if path_match else "Benchmark subprocess failed."
        return "benchmark_execution_failure", detail

    if "Unhandled error:" in text:
        return "runtime_error", message[:220]

    return "", ""


def _recommended_action(
    state: str,
    *,
    blocker_type: str,
    blocker_detail: str,
    candidate_label: str,
) -> str:
    """Return a next-step sentence for one candidate."""
    if state == "recommended":
        return f"Open Scorecards next and validate whether {candidate_label} should advance to human review."
    if state == "ready_for_review":
        return f"Review the benchmark evidence for {candidate_label} before making a promotion decision."
    if state == "mixed_signal":
        return f"Inspect tradeoffs for {candidate_label} in Scorecards and Horizon & Bias before deciding whether the gain is acceptable."
    if state == "failed_hard_gate":
        return f"Do not promote {candidate_label}; inspect the failed hard-gate evidence before reusing this line of work."
    if blocker_type == "missing_input_data" and blocker_detail:
        return (
            f"Restore the missing input data at `{blocker_detail}` and rerun the blocked candidate."
        )
    if blocker_type == "code_registration_required":
        return "Register the method/config path in the runtime dispatch before rerunning this candidate."
    if blocker_type == "benchmark_execution_failure":
        return "Inspect the benchmark logs, fix the execution failure, and rerun the blocked candidate."
    return "Rerun or replan this candidate after resolving the blocking issue."


def build_candidate_decision_summary(
    row: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Return a canonical plain-language decision summary for one candidate."""
    candidate_id = _clean_text(row.get("candidate_id")) or "candidate"
    outcome = _clean_text(row.get("outcome")).lower()
    status = _clean_text(row.get("status")).lower()
    run_id = _clean_text(row.get("run_id"))
    has_run = bool(run_id and run_id != "not_run")
    stdout_log = _clean_text(row.get("stdout_log"))
    stderr_log = _clean_text(row.get("stderr_log"))
    log_text = _read_candidate_log(project_root, stdout_log) or _read_candidate_log(
        project_root, stderr_log
    )

    classification_details = row.get("classification_details")
    if not isinstance(classification_details, dict):
        classification_details = {}
    if not classification_details and log_text:
        result_json = _parse_result_json_from_log(log_text)
        parsed_details = result_json.get("classification_details")
        if isinstance(parsed_details, dict):
            classification_details = parsed_details

    reasons = classification_details.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    reason_text = "\n".join(str(reason) for reason in reasons if reason)
    interpretation = _clean_text(row.get("interpretation"))
    key_metrics_summary = _clean_text(row.get("key_metrics_summary"))
    blocker_type, blocker_detail = _extract_blocker_detail("\n".join([reason_text, log_text]))

    benchmark_completeness = "full" if has_run else "none"
    decision_state = "not_executed"
    if status in {"planned", "running", "launching"}:
        decision_state = "not_executed"
    elif outcome == "failed_hard_gate":
        decision_state = "failed_hard_gate"
    elif has_run and outcome == "passed_all_gates":
        decision_state = "ready_for_review"
    elif has_run and outcome == "needs_human_review":
        decision_state = "mixed_signal"
    elif outcome == "inconclusive" or (status == "completed" and not has_run):
        decision_state = "blocked_by_data_or_runtime"

    metric_name, metric_delta = _best_metric(row)
    decision_meta = _DECISION_META[decision_state]
    candidate_label = candidate_id.replace("-", " ")

    if decision_state == "ready_for_review":
        headline = f"{candidate_id} produced a usable benchmark and is ready for review."
    elif decision_state == "mixed_signal":
        headline = f"{candidate_id} benchmarked successfully but still has meaningful tradeoffs."
    elif decision_state == "failed_hard_gate":
        headline = f"{candidate_id} failed a hard gate and should not advance."
    elif decision_state == "blocked_by_data_or_runtime":
        detail = blocker_detail or "benchmark execution did not produce a usable run bundle"
        headline = f"{candidate_id} is blocked because {detail}."
    else:
        headline = f"{candidate_id} has not produced reviewable evidence yet."

    explanation_parts: list[str] = []
    if interpretation:
        explanation_parts.append(interpretation)
    if key_metrics_summary and key_metrics_summary != "no tradeoff metrics":
        explanation_parts.append(f"Key metrics: {key_metrics_summary}.")
    if metric_delta is not None and metric_name:
        explanation_parts.append(
            f"Best tracked delta: {metric_name} {metric_delta:+.4f} versus the champion."
        )
    if blocker_type == "missing_input_data" and blocker_detail:
        explanation_parts.append(f"Required input data is missing at `{blocker_detail}`.")
    elif blocker_type == "code_registration_required":
        explanation_parts.append("This candidate needs code registration before it can run.")
    elif blocker_type == "benchmark_execution_failure":
        explanation_parts.append(
            "The benchmark subprocess failed before producing a usable run bundle."
        )
    elif decision_state == "blocked_by_data_or_runtime" and reason_text:
        explanation_parts.append(_clean_text(reason_text)[:320])

    explanation = " ".join(part for part in explanation_parts if part).strip()
    if not explanation:
        explanation = headline

    return {
        "decision_state": decision_state,
        "decision_label": decision_meta["label"],
        "headline": headline,
        "explanation": explanation,
        "recommended_action": _recommended_action(
            decision_state,
            blocker_type=blocker_type,
            blocker_detail=blocker_detail,
            candidate_label=candidate_label,
        ),
        "blocker_type": blocker_type,
        "blocker_detail": blocker_detail,
        "confidence": decision_meta["confidence"],
        "evidence_log_path": stdout_log or stderr_log,
        "benchmark_completeness": benchmark_completeness,
        "best_metric_name": metric_name,
        "best_metric_delta": metric_delta,
    }


def finalize_candidate_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Promote the strongest review-ready candidate to ``recommended``."""
    reviewable = [row for row in rows if row.get("decision_state") == "ready_for_review"]
    if not reviewable:
        return rows

    def _sort_key(row: Mapping[str, Any]) -> tuple[int, float, str]:
        delta = _as_float(row.get("best_metric_delta"))
        return (
            0 if delta is not None else 1,
            delta if delta is not None else 0.0,
            _clean_text(row.get("candidate_id")),
        )

    winner = min(reviewable, key=_sort_key)
    winner["decision_state"] = "recommended"
    winner["decision_label"] = _DECISION_META["recommended"]["label"]
    winner["confidence"] = _DECISION_META["recommended"]["confidence"]
    winner["headline"] = (
        f"{_clean_text(winner.get('candidate_id'))} is the strongest fully benchmarked candidate in this session."
    )
    winner["recommended_action"] = _recommended_action(
        "recommended",
        blocker_type="",
        blocker_detail="",
        candidate_label=_clean_text(winner.get("candidate_id")).replace("-", " "),
    )
    winner["explanation"] = (
        f"{winner.get('explanation', '').strip()} This is the best fully benchmarked candidate available from the current session."
    ).strip()
    return rows


def build_search_candidate_rows(
    session: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Flatten and enrich one search session into candidate-level summaries."""
    seed_map: dict[str, dict[str, Any]] = {}
    if seed_rows is not None and not seed_rows.empty:
        for _, seed_row in seed_rows.iterrows():
            seed_map[_clean_text(seed_row.get("candidate_id"))] = {
                str(key): value for key, value in seed_row.to_dict().items()
            }

    rows: list[dict[str, Any]] = []
    for candidate in session.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        result = candidate.get("result", {})
        spec = candidate.get("spec", {})
        benchmark_summary = result.get("benchmark_summary", {}) if isinstance(result, dict) else {}
        classification_details = (
            result.get("classification_details", {}) if isinstance(result, dict) else {}
        )
        base = dict(seed_map.get(_clean_text(candidate.get("candidate_id")), {}))
        row: dict[str, Any] = {
            **base,
            "candidate_id": _clean_text(candidate.get("candidate_id")),
            "source": _clean_text(candidate.get("source")),
            "source_id": _clean_text(candidate.get("source_id")),
            "execution_mode": _clean_text(candidate.get("execution_mode")),
            "status": _clean_text(candidate.get("status")),
            "attempts": int(candidate.get("attempts", 0) or 0),
            "recipe_id": _clean_text(candidate.get("recipe_id")),
            "experiment_id": _clean_text(spec.get("experiment_id"))
            if isinstance(spec, dict)
            else "",
            "method_id": _clean_text(result.get("method_id")) if isinstance(result, dict) else "",
            "config_id": _clean_text(result.get("config_id")) if isinstance(result, dict) else "",
            "outcome": _clean_text(result.get("outcome")) if isinstance(result, dict) else "",
            "run_id": _clean_text(result.get("run_id")) if isinstance(result, dict) else "",
            "primary_metric_name": _clean_text(benchmark_summary.get("primary_metric"))
            if isinstance(benchmark_summary, dict)
            else "",
            "hard_constraint_regression": (
                benchmark_summary.get("hard_constraint_regression")
                if isinstance(benchmark_summary, dict)
                else None
            ),
            "champion_method_id": _clean_text(benchmark_summary.get("champion_method_id"))
            if isinstance(benchmark_summary, dict)
            else "",
            "champion_config_id": _clean_text(benchmark_summary.get("champion_config_id"))
            if isinstance(benchmark_summary, dict)
            else "",
            "patch_path": _clean_text(result.get("patch_path")) if isinstance(result, dict) else "",
            "profile_path": _clean_text(result.get("profile_path"))
            if isinstance(result, dict)
            else "",
            "stdout_log": _clean_text(result.get("stdout_log")) if isinstance(result, dict) else "",
            "stderr_log": _clean_text(result.get("stderr_log")) if isinstance(result, dict) else "",
            "classification_details": (
                classification_details if isinstance(classification_details, dict) else {}
            ),
            "key_metrics_summary": _clean_text(result.get("key_metrics_summary"))
            if isinstance(result, dict)
            else "",
            "interpretation": _clean_text(result.get("interpretation"))
            if isinstance(result, dict)
            else "",
            "next_action": _clean_text(result.get("next_action"))
            if isinstance(result, dict)
            else "",
        }
        if isinstance(benchmark_summary, dict):
            metrics = benchmark_summary.get("metrics", {})
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    row[str(metric)] = value
            deltas = benchmark_summary.get("deltas", {})
            if isinstance(deltas, dict):
                for metric, value in deltas.items():
                    row[f"delta_{metric}"] = value

        row.update(build_candidate_decision_summary(row, project_root=project_root))
        rows.append(row)

    return finalize_candidate_recommendations(rows)


def build_search_session_summary(
    candidates: pd.DataFrame,
    *,
    search_id: str,
    status: str = "",
    history_index_present: bool | None = None,
    incomplete_bundle_count: int = 0,
) -> dict[str, Any]:
    """Return a session-level decision brief from candidate summaries."""
    total = len(candidates)
    if total == 0:
        return {
            "source": "search_session",
            "search_id": search_id,
            "session_decision_state": "not_executed",
            "session_headline": "No search candidates have been recorded for this session yet.",
            "session_recommendation": "Run or resume a search session to populate decision evidence.",
            "session_blocker_summary": "",
            "successful_benchmark_count": 0,
            "inconclusive_count": 0,
            "recommendation_candidate_id": "",
            "recommended_next_step": "Run a search session.",
        }

    state_counts = candidates["decision_state"].value_counts(dropna=False).to_dict()
    successful_benchmark_count = int((candidates["benchmark_completeness"] == "full").sum())
    inconclusive_count = int(
        (candidates["outcome"].astype(str).str.lower() == "inconclusive").sum()
    )

    recommended_rows = candidates[candidates["decision_state"] == "recommended"]
    review_rows = candidates[candidates["decision_state"] == "ready_for_review"]
    mixed_rows = candidates[candidates["decision_state"] == "mixed_signal"]
    blocked_rows = candidates[candidates["decision_state"] == "blocked_by_data_or_runtime"]

    session_state = "not_executed"
    headline = f"Search session `{search_id}` has not produced decision-ready evidence yet."
    recommendation = "Inspect the candidate list and logs to understand what happened."
    blocker_summary = ""
    recommendation_candidate_id = ""

    if not recommended_rows.empty:
        top = recommended_rows.iloc[0]
        recommendation_candidate_id = _clean_text(top.get("candidate_id"))
        session_state = "recommended"
        headline = f"Search finished. Best available candidate: {recommendation_candidate_id}."
        recommendation = f"Open Scorecards for {recommendation_candidate_id} first, then confirm plausibility in Projections."
    elif not review_rows.empty:
        top = review_rows.sort_values(
            ["best_metric_delta", "candidate_id"],
            ascending=[True, True],
            na_position="last",
        ).iloc[0]
        recommendation_candidate_id = _clean_text(top.get("candidate_id"))
        session_state = "ready_for_review"
        headline = (
            f"Search finished. {len(review_rows)} candidate(s) produced usable benchmark evidence; "
            f"{recommendation_candidate_id} looks strongest so far."
        )
        recommendation = f"Review {recommendation_candidate_id} in Scorecards, then use Horizon & Bias to decide whether the tradeoffs are acceptable."
    elif not mixed_rows.empty:
        top = mixed_rows.sort_values(
            ["best_metric_delta", "candidate_id"],
            ascending=[True, True],
            na_position="last",
        ).iloc[0]
        recommendation_candidate_id = _clean_text(top.get("candidate_id"))
        session_state = "mixed_signal"
        headline = f"Search finished, but the best available candidate still has unresolved tradeoffs: {recommendation_candidate_id}."
        recommendation = f"Inspect tradeoffs for {recommendation_candidate_id} in Scorecards and Horizon & Bias before deciding whether to keep exploring."
    elif len(blocked_rows) == total:
        top = blocked_rows.iloc[0]
        blocker_summary = _clean_text(top.get("blocker_detail")) or _clean_text(top.get("headline"))
        session_state = "blocked_by_data_or_runtime"
        headline = (
            f"Search finished, but none of the {total} candidate(s) produced a usable benchmark."
        )
        recommendation = _clean_text(top.get("recommended_action")) or (
            "Resolve the blocking issue and rerun the affected candidates."
        )
    elif state_counts.get("failed_hard_gate", 0):
        session_state = "failed_hard_gate"
        headline = "Search finished, but the reviewed candidates failed hard gates."
        recommendation = "Inspect the failed hard-gate evidence before continuing this search line."

    if history_index_present is False:
        extra = " Benchmark history index is missing, so benchmark-backed review tabs may be partially unavailable."
        headline += extra
        blocker_summary = (
            f"{blocker_summary} Benchmark history index is missing.".strip()
            if blocker_summary
            else "Benchmark history index is missing."
        )
    if incomplete_bundle_count > 0:
        addition = f" {incomplete_bundle_count} incomplete benchmark bundle(s) were detected."
        headline += addition
        blocker_summary = (
            f"{blocker_summary} {incomplete_bundle_count} incomplete benchmark bundle(s) detected.".strip()
            if blocker_summary
            else f"{incomplete_bundle_count} incomplete benchmark bundle(s) detected."
        )

    if status:
        headline = f"{headline} Session status: {status}."

    return {
        "source": "search_session",
        "search_id": search_id,
        "session_decision_state": session_state,
        "session_headline": headline,
        "session_recommendation": recommendation,
        "session_blocker_summary": blocker_summary,
        "successful_benchmark_count": successful_benchmark_count,
        "inconclusive_count": inconclusive_count,
        "recommendation_candidate_id": recommendation_candidate_id,
        "recommended_next_step": recommendation,
        "state_counts": state_counts,
    }


def build_benchmark_decision_brief(
    run_metadata: pd.DataFrame,
    *,
    champion_id: str | None,
) -> dict[str, Any]:
    """Return a benchmark-backed decision brief from consolidated run metadata."""
    if run_metadata.empty:
        return {
            "source": "benchmark",
            "decision_state": "not_executed",
            "headline": "No benchmark-backed decision evidence is available yet.",
            "explanation": "Benchmark history is empty or incomplete, so the dashboard cannot build a benchmark-backed recommendation.",
            "recommended_action": "Use the latest search session or run a benchmark to populate review evidence.",
            "recommendation_candidate_id": "",
        }

    champion_row = (
        run_metadata[run_metadata["run_id"] == champion_id].iloc[0]
        if champion_id is not None and not run_metadata[run_metadata["run_id"] == champion_id].empty
        else None
    )
    challengers = run_metadata.copy()
    if champion_id is not None:
        challengers = challengers[challengers["run_id"] != champion_id]
    challengers = challengers[challengers["selected_county_mape_overall"].notna()]
    if challengers.empty:
        champion_name = (
            _clean_text(champion_row.get("display_name"))
            if champion_row is not None
            else "Champion"
        )
        return {
            "source": "benchmark",
            "decision_state": "ready_for_review",
            "headline": f"{champion_name} is still the only benchmark-backed option on record.",
            "explanation": "No challenger bundle with readable scorecard data is available yet.",
            "recommended_action": "Run or register a challenger benchmark bundle before making a new decision.",
            "recommendation_candidate_id": "",
        }

    best = challengers.sort_values(
        ["selected_county_mape_overall", "run_date_sort"],
        ascending=[True, False],
        na_position="last",
    ).iloc[0]
    champion_mape = (
        _as_float(champion_row.get("reference_county_mape_overall"))
        if champion_row is not None
        else None
    )
    best_mape = _as_float(best.get("selected_county_mape_overall"))
    delta = (
        best_mape - champion_mape if best_mape is not None and champion_mape is not None else None
    )
    status_code = _clean_text(best.get("status_code")) or "needs_human_review"

    if status_code == "passed_all_gates" and (delta is None or delta <= 0):
        decision_state = "recommended"
    elif status_code in {"passed_all_gates", "needs_human_review"}:
        decision_state = "ready_for_review"
    elif status_code == "failed_hard_gate":
        decision_state = "failed_hard_gate"
    else:
        decision_state = "mixed_signal"

    best_name = _clean_text(best.get("display_name")) or _clean_text(best.get("run_id"))
    champion_name = (
        _clean_text(champion_row.get("display_name"))
        if champion_row is not None
        else "the champion"
    )

    if decision_state == "recommended":
        headline = f"Best benchmark-backed candidate: {best_name}."
        explanation = (
            f"{best_name} is the strongest fully benchmarked challenger and improves county error by {delta:+.2f} points versus {champion_name}."
            if delta is not None
            else f"{best_name} is the strongest fully benchmarked challenger available."
        )
        recommended_action = f"Use Scorecards to confirm the claimed gain for {best_name}, then decide whether it is ready for human approval."
    elif decision_state == "ready_for_review":
        headline = f"Best benchmark-backed candidate awaiting review: {best_name}."
        explanation = f"{best_name} is the strongest challenger in the benchmark archive, but it still needs explicit human review before any promotion decision."
        recommended_action = f"Review {best_name} in Scorecards and Projections before deciding."
    elif decision_state == "failed_hard_gate":
        headline = (
            f"The best available benchmark-backed challenger, {best_name}, failed a hard gate."
        )
        explanation = "A visible challenger exists, but it is not promotion-ready because it violated a hard constraint."
        recommended_action = "Inspect the failed hard-gate evidence before continuing this line."
    else:
        headline = f"Benchmark evidence is mixed; {best_name} is the current front-runner."
        explanation = "A challenger exists, but the benchmark record still contains tradeoffs or incomplete governance status."
        recommended_action = (
            f"Inspect the tradeoffs for {best_name} before deciding whether to keep exploring."
        )

    return {
        "source": "benchmark",
        "decision_state": decision_state,
        "headline": headline,
        "explanation": explanation,
        "recommended_action": recommended_action,
        "recommendation_candidate_id": _clean_text(best.get("run_id")),
        "best_label": best_name,
        "best_delta": delta,
    }
