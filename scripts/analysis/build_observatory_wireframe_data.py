"""Generate observatory_wireframe/data.js from real benchmark history.

The wireframe at observatory_wireframe/index.html consumes a global
``window.OBS_DATA`` object defined in ``data.js``. The version shipped
with the wireframe is hand-authored synthetic data. This script replaces
it with values derived from a real benchmark run bundle.

Usage
-----
    python scripts/analysis/build_observatory_wireframe_data.py
    python scripts/analysis/build_observatory_wireframe_data.py --run-id br-...
    python scripts/analysis/build_observatory_wireframe_data.py --list-runs

Fields without a real data source are filled with documented stubs and
marked ``stub: True`` so the UI can render but the operator can tell what
is real vs placeholder.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = PROJECT_ROOT / "data/analysis/benchmark_history"
WIREFRAME_DIR = PROJECT_ROOT / "observatory_wireframe"
POP_CSV = PROJECT_ROOT / "data/raw/population/nd_county_population.csv"
PROFILES_DIR = PROJECT_ROOT / "config/method_profiles"
DECISION_DIR = PROJECT_ROOT / "docs/reviews/benchmark_decisions"

DEFAULT_DECISION_RECORD = "2026-03-09-m2026r1-vs-m2026.md"
LOGGER = logging.getLogger(__name__)

# Cartogram positions copied verbatim from observatory_wireframe/data.js
# (manual design artifact; no automated source).
COUNTY_LAYOUT: list[tuple[str, str, float, float]] = [
    ("Divide",        "38023", 1,    0),
    ("Burke",         "38013", 2,    0),
    ("Renville",      "38075", 3,    0),
    ("Bottineau",     "38009", 4,    0),
    ("Rolette",       "38079", 5,    0),
    ("Towner",        "38095", 6,    0),
    ("Cavalier",      "38019", 7,    0),
    ("Pembina",       "38067", 8,    0),
    ("Williams",      "38105", 1,    1),
    ("Mountrail",     "38061", 2,    1),
    ("Ward",          "38101", 3,    1),
    ("McHenry",       "38049", 4,    1),
    ("Pierce",        "38069", 5,    1),
    ("Benson",        "38005", 6,    1),
    ("Ramsey",        "38071", 7,    1),
    ("Walsh",         "38099", 8,    1),
    ("McKenzie",      "38053", 1,    2),
    ("Dunn",          "38025", 2,    2),
    ("McLean",        "38055", 3,    2),
    ("Sheridan",      "38083", 4,    2),
    ("Wells",         "38103", 5,    2),
    ("Eddy",          "38027", 6,    2),
    ("Nelson",        "38063", 7,    2),
    ("Grand Forks",   "38035", 8,    2),
    ("Billings",      "38007", 1,    3),
    ("Stark",         "38089", 2,    3),
    ("Mercer",        "38057", 3,    3),
    ("Oliver",        "38065", 4,    3),
    ("Burleigh",      "38015", 5,    3),
    ("Kidder",        "38043", 6,    3),
    ("Foster",        "38031", 5.5,  3.5),
    ("Stutsman",      "38093", 7,    3),
    ("Barnes",        "38003", 8,    3),
    ("Golden Valley", "38033", 0,    4),
    ("Slope",         "38087", 1,    4),
    ("Hettinger",     "38041", 2,    4),
    ("Grant",         "38037", 3,    4),
    ("Morton",        "38059", 4,    4),
    ("Emmons",        "38029", 5,    4),
    ("Logan",         "38047", 6,    4),
    ("LaMoure",       "38045", 7,    4),
    ("Ransom",        "38073", 8,    4),
    ("Steele",        "38091", 8.5,  2.5),
    ("Traill",        "38097", 8.7,  3.7),
    ("Cass",          "38017", 9,    4),
    ("Bowman",        "38011", 1,    5),
    ("Adams",         "38001", 2,    5),
    ("Sioux",         "38085", 3,    5),
    ("McIntosh",      "38051", 5,    5),
    ("Dickey",        "38021", 6,    5),
    ("Sargent",       "38081", 7,    5.5),
    ("Richland",      "38077", 8,    5.5),
    ("Griggs",        "38039", 6.5,  2.5),
]

CATEGORY_TO_GROUP = {
    "Bakken": "bakken",
    "Urban/College": "urban_college",
    "Reservation": "reservation",
    "Rural": "rural",
}


@dataclass
class BenchmarkBundle:
    run_id: str
    scorecard: pd.DataFrame
    county_metrics: pd.DataFrame
    manifest: dict
    champion_row: pd.Series
    challenger_row: pd.Series


def latest_run_id() -> str:
    """Pick the most recent run that has both a champion and a challenger."""
    if not BENCH_DIR.exists():
        raise SystemExit(f"benchmark_history not found: {BENCH_DIR}")
    candidates = sorted(
        [p for p in BENCH_DIR.iterdir() if p.is_dir() and p.name.startswith("br-")],
        reverse=True,
    )
    for run_dir in candidates:
        sc_path = run_dir / "summary_scorecard.csv"
        if not sc_path.exists():
            continue
        sc = pd.read_csv(sc_path)
        statuses = set(sc["status_at_run"])
        if "champion" in statuses and (statuses & {"experiment", "candidate"}):
            return run_dir.name
    raise SystemExit("no benchmark runs with both champion and challenger rows")


def load_bundle(run_id: str) -> BenchmarkBundle:
    run_dir = BENCH_DIR / run_id
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")
    sc = pd.read_csv(run_dir / "summary_scorecard.csv")
    cm = pd.read_csv(run_dir / "county_metrics.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    champ = sc[sc["status_at_run"] == "champion"].iloc[0]
    chall_rows = sc[sc["status_at_run"].isin(["experiment", "candidate"])]
    if chall_rows.empty:
        raise SystemExit(f"no challenger row in {run_id} summary_scorecard.csv")
    chall = chall_rows.iloc[0]
    return BenchmarkBundle(run_id, sc, cm, manifest, champ, chall)


def per_county_metrics(cm: pd.DataFrame, champ_method: str, chall_method: str) -> pd.DataFrame:
    """Aggregate county-level MAPE and signed bias per method."""
    def agg(method: str) -> pd.DataFrame:
        sub = cm[cm["method"] == method].copy()
        sub["county_fips"] = sub["county_fips"].astype(str).str.zfill(5)
        sub["abs_pct"] = sub["pct_error"].abs()
        g = sub.groupby(["county_fips", "county_name", "category"]).agg(
            mape=("abs_pct", "mean"),
            bias=("pct_error", "mean"),
            last_residual_2024=("pct_error", lambda s: s.iloc[-1] if len(s) else 0.0),
        ).reset_index()
        return g

    champ = agg(champ_method).rename(columns={
        "mape": "champion_mape", "bias": "champion_bias",
        "last_residual_2024": "champion_last_resid",
    })
    chall = agg(chall_method).rename(columns={
        "mape": "challenger_mape", "bias": "challenger_bias",
        "last_residual_2024": "challenger_last_resid",
    })
    joined = champ.merge(
        chall[["county_fips", "challenger_mape", "challenger_bias", "challenger_last_resid"]],
        on="county_fips",
    )
    joined["delta_mape"] = joined["challenger_mape"] - joined["champion_mape"]
    return joined


def load_populations() -> dict[str, int]:
    df = pd.read_csv(POP_CSV, dtype={"county_fips": str})
    return dict(zip(df["county_fips"], df["population_2024"].astype(int), strict=True))


def build_counties(per_county: pd.DataFrame, pops: dict[str, int]) -> list[dict]:
    metrics = {row.county_fips: row for row in per_county.itertuples()}
    out: list[dict] = []
    for name, fips, col, row in COUNTY_LAYOUT:
        m = metrics.get(fips)
        if m is None:
            # County missing from benchmark: emit minimal stub.
            out.append({
                "name": name, "fips": fips, "group": "rural",
                "col": col, "row": row, "pop": pops.get(fips, 0),
                "_missing_from_benchmark": True,
            })
            continue
        out.append({
            "name": name,
            "fips": fips,
            "group": CATEGORY_TO_GROUP.get(m.category, "rural"),
            "col": col,
            "row": row,
            "pop": pops.get(fips, 0),
            "champion_mape": round(float(m.champion_mape), 2),
            "challenger_mape": round(float(m.challenger_mape), 2),
            "delta_mape": round(float(m.delta_mape), 3),
            "signed_bias_champion": round(float(m.champion_bias), 2),
            "signed_bias_challenger": round(float(m.challenger_bias), 2),
            "last_residual_2024": round(float(m.challenger_last_resid), 2),
        })
    return out


def build_scorecard(b: BenchmarkBundle, decision: dict) -> dict:
    champ, chall = b.champion_row, b.challenger_row
    metrics = [
        ("state_ape_short",      "Recent-origin state APE (short)",   "state_ape_recent_short",   True, True, False),
        ("state_ape_medium",     "Recent-origin state APE (medium)",  "state_ape_recent_medium",  True, True, False),
        ("signed_bias",          "Recent-origin signed bias",         "state_signed_bias_recent", False, True, True),
        ("county_mape_overall",  "County MAPE - overall",             "county_mape_overall",      True, True, False),
        ("county_mape_urban",    "County MAPE - urban / college",     "county_mape_urban_college", True, True, False),
        ("county_mape_rural",    "County MAPE - rural",               "county_mape_rural",        True, False, False),
        ("county_mape_bakken",   "County MAPE - Bakken",              "county_mape_bakken",       True, False, False),
        ("sentinel_cass",        "Sentinel - Cass county MAPE",       "sentinel_cass_mape",       True, False, False),
        ("sentinel_grand_forks", "Sentinel - Grand Forks MAPE",       "sentinel_grand_forks_mape", True, False, False),
        ("sentinel_burleigh",    "Sentinel - Burleigh MAPE",          "sentinel_burleigh_mape",   True, False, False),
        ("sentinel_williams",    "Sentinel - Williams MAPE",          "sentinel_williams_mape",   True, False, False),
        ("sentinel_mckenzie",    "Sentinel - McKenzie MAPE",          "sentinel_mckenzie_mape",   True, False, False),
    ]
    metric_rows = []
    for key, label, col, lower_is_better, primary, target_zero in metrics:
        if col not in champ.index:
            continue
        row = {
            "key": key,
            "label": label,
            "champion": round(float(champ[col]), 4),
            "challenger": round(float(chall[col]), 4),
            "lower_is_better": lower_is_better,
            "primary": primary,
        }
        if target_zero:
            row["target_zero"] = True
        metric_rows.append(row)

    hard = [
        {"name": "Negative population violations",
         "champion": int(champ["negative_population_violations"]),
         "challenger": int(chall["negative_population_violations"]),
         "gate": "hard"},
        {"name": "Aggregation violations",
         "champion": int(champ["aggregation_violations"]),
         "challenger": int(chall["aggregation_violations"]),
         "gate": "hard"},
        {"name": "Scenario order violations",
         "champion": int(champ["scenario_order_violations"]),
         "challenger": int(chall["scenario_order_violations"]),
         "gate": "hard"},
        {"name": "Sensitivity instability flag",
         "champion": bool(champ["sensitivity_instability_flag"]),
         "challenger": bool(chall["sensitivity_instability_flag"]),
         "gate": "hard"},
    ]

    return {
        "label": b.manifest.get("benchmark_label", "(no label)"),
        "decision_id": decision.get("decision_id", ""),
        "benchmark_run_id": b.run_id,
        "champion": {
            "method": champ["method_id"],
            "config": champ["config_id"],
            "vintage": _vintage_title(champ["config_id"]),
        },
        "challenger": {
            "method": chall["method_id"],
            "config": chall["config_id"],
            "vintage": _vintage_title(chall["config_id"]),
        },
        "metrics": metric_rows,
        "hard_constraints": hard,
        "verdict": decision["verdict"],
    }


def _vintage_title(config_id: str) -> str:
    """Make a friendly name out of a config_id."""
    name = config_id.replace("cfg-", "").replace("-v1", "")
    return name


def parse_decision_record(path: Path) -> dict:
    if not path.exists():
        return {
            "decision_id": "",
            "verdict": _stub_verdict("No decision record found at " + str(path.relative_to(PROJECT_ROOT))),
        }
    txt = path.read_text()
    decision_id_match = re.search(r"Decision ID \|\s*([^\s|]+)", txt)
    status_match = re.search(r"Status \|\s*([^|]+)\|", txt)
    reviewer_match = re.search(r"Reviewer \|\s*([^|]*)\|", txt)
    rationale_match = re.search(r"Decision rationale:\s*\n*(.+?)(?:\n##|\Z)", txt, re.DOTALL)
    decision_id = decision_id_match.group(1).strip() if decision_id_match else ""
    status = status_match.group(1).strip() if status_match else "Unknown"
    reviewer = reviewer_match.group(1).strip() if reviewer_match else ""
    rationale = (rationale_match.group(1).strip() if rationale_match else "")
    verdict = _stub_verdict(
        f"Decision record status: {status}. "
        + (f"Rationale: {rationale[:300]}" if rationale else "Rationale not yet written.")
    )
    verdict["status"] = (status if status != "Draft" else "Draft - awaiting sign-off")
    verdict["reviewer"] = reviewer or None
    return {"decision_id": decision_id, "verdict": verdict}


def _stub_verdict(headline: str) -> dict:
    return {
        "state": "ready_for_review",
        "user_status_label": "Ready for human review",
        "headline": headline,
        "main_reason": "See scorecard metrics and per-county map for evidence.",
        "main_gain": "",
        "main_tradeoff": "",
        "confidence": "",
        "safe_to_recommend": True,
        "escalation_guidance": "Review before recommending",
        "recommended_action": "review",
        "recommended_label": "Review benchmark evidence before recommending",
        "operational_label": "Operationally clean",
        "reviewer": None,
        "status": "Draft - awaiting sign-off",
        "benchmark_completeness": "full",
        "stub": True,
    }


def build_labels(b: BenchmarkBundle, decision: dict) -> dict:
    champ, chall = b.champion_row, b.challenger_row
    return {
        "champion": {
            "role": "Current production",
            "short": "Current",
            "title": _vintage_title(champ["config_id"]),
            "sub": f"in production - method {champ['method_id']}",
            "method_id": champ["method_id"],
            "config_id": champ["config_id"],
        },
        "challenger": {
            "role": "Candidate under review",
            "short": "Candidate",
            "title": _vintage_title(chall["config_id"]),
            "sub": f"method {chall['method_id']}",
            "method_id": chall["method_id"],
            "config_id": chall["config_id"],
        },
        "benchmark": {
            "title": b.manifest.get("benchmark_label", b.run_id),
            "sub": b.manifest.get("benchmark_contract_version", ""),
            "id": b.run_id,
        },
        "decision": {
            "title": decision.get("decision_id", "(no decision record)"),
            "id": decision.get("decision_id", ""),
            "file": (f"docs/reviews/benchmark_decisions/{decision.get('decision_id', '')}.md"
                     if decision.get("decision_id") else ""),
        },
    }


def stub_production_health(labels: dict) -> dict:
    """No real source exists for drift indicators or per-vintage history."""
    return {
        "champion_label": labels["champion"]["title"],
        "champion_method": labels["champion"]["method_id"],
        "champion_config": labels["champion"]["config_id"],
        "alias_set_on": "",
        "days_in_production": None,
        "aliases": {
            "county": labels["champion"]["title"],
            "state":  labels["champion"]["title"],
            "city":   labels["champion"]["title"],
        },
        "drift_indicators": [
            {"label": "No drift monitor wired", "value": "-", "trend": "unknown",
             "detail": "production_health.drift_indicators has no real data source yet"},
        ],
        "last_5_vintages": [],
        "stub": True,
    }


def stub_activity() -> list[dict]:
    return [{
        "when": "(no source)",
        "icon": "alert",
        "text": "Activity feed not yet wired - no real source for cross-system events.",
        "tab": "home",
        "stub": True,
    }]


def stub_deep_search() -> dict:
    return {
        "state": "idle",
        "pack": "",
        "started_at": "",
        "eta": "",
        "cores": {"allocated": 0, "parallel_runs": 0, "workers_per_run": 0},
        "progress": {"completed": 0, "planned": 0, "candidates_found": 0, "candidates_pending_review": 0},
        "journal": [],
        "leaders": [],
        "stub": True,
        "note": "Deep search state is only populated while a search session is running. Wire this to live session journal when implemented.",
    }


def history_from_index() -> list[dict]:
    """Synthesize a promotion log from the benchmark history index."""
    idx_path = BENCH_DIR / "index.csv"
    if not idx_path.exists():
        return []
    idx = pd.read_csv(idx_path)
    # One entry per distinct run, keep most recent first
    rows = (
        idx.drop_duplicates("run_id")
        .sort_values("run_date", ascending=False)
        .head(10)
    )
    out = []
    for r in rows.itertuples():
        out.append({
            "date": str(r.run_date),
            "title": getattr(r, "benchmark_label", "") or r.run_id,
            "method": r.method_id,
            "action": r.decision_status or "pending",
            "note": f"run_id {r.run_id}",
        })
    return out


def emit_data_js(data: dict, out_path: Path) -> None:
    body = json.dumps(data, indent=2, default=str)
    js = (
        "/* Generated by scripts/analysis/build_observatory_wireframe_data.py - "
        "do not edit by hand. */\n"
        f"window.OBS_DATA = {body};\n"
    )
    out_path.write_text(js)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="benchmark run id (default: most recent valid run)")
    parser.add_argument("--decision-record", default=DEFAULT_DECISION_RECORD,
                        help=f"filename under docs/reviews/benchmark_decisions/ "
                             f"(default: {DEFAULT_DECISION_RECORD})")
    parser.add_argument("--output", type=Path, default=WIREFRAME_DIR / "data.js")
    parser.add_argument("--list-runs", action="store_true",
                        help="print available benchmark runs and exit")
    args = parser.parse_args()

    if args.list_runs:
        for p in sorted(BENCH_DIR.iterdir()):
            if p.is_dir() and p.name.startswith("br-"):
                LOGGER.info(p.name)
        return

    run_id = args.run_id or latest_run_id()
    bundle = load_bundle(run_id)
    decision = parse_decision_record(DECISION_DIR / args.decision_record)

    per_county = per_county_metrics(
        bundle.county_metrics,
        champ_method=bundle.champion_row["method_id"],
        chall_method=bundle.challenger_row["method_id"],
    )
    pops = load_populations()
    counties = build_counties(per_county, pops)
    labels = build_labels(bundle, decision)
    scorecard = build_scorecard(bundle, decision)

    data = {
        "counties": counties,
        "scorecard": scorecard,
        "labels": labels,
        "production_health": stub_production_health(labels),
        "activity": stub_activity(),
        "deep_search": stub_deep_search(),
        "history": history_from_index(),
        "_generated": {
            "run_id": run_id,
            "decision_record": args.decision_record,
            "champion_method": bundle.champion_row["method_id"],
            "challenger_method": bundle.challenger_row["method_id"],
        },
    }

    emit_data_js(data, args.output)
    LOGGER.info("wrote %s", args.output.relative_to(PROJECT_ROOT))
    LOGGER.info("  run_id:     %s", run_id)
    LOGGER.info(
        "  champion:   %s / %s",
        bundle.champion_row["method_id"],
        bundle.champion_row["config_id"],
    )
    LOGGER.info(
        "  challenger: %s / %s",
        bundle.challenger_row["method_id"],
        bundle.challenger_row["config_id"],
    )
    LOGGER.info("  counties:   %d (with metrics)", len(counties))
    n_stub = sum(1 for c in counties if c.get("_missing_from_benchmark"))
    if n_stub:
        LOGGER.warning("  WARNING: %d counties missing from county_metrics.csv", n_stub)


if __name__ == "__main__":
    main()
