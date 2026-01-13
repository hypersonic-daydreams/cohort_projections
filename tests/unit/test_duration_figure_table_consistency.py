from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def publication_figures_module():
    root = Path(__file__).resolve().parents[2]
    module_path = (
        root
        / "sdc_2024_replication"
        / "scripts"
        / "statistical_analysis"
        / "journal_article"
        / "create_publication_figures.py"
    )
    spec = importlib.util.spec_from_file_location("create_publication_figures", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_duration_table(latex_text: str) -> tuple[dict[str, dict[str, float]], float, int]:
    table_groups: dict[str, dict[str, float]] = {}

    row_pattern = re.compile(
        r"^(Q[1-4](?: \((?:Low|High)\))?)\s*&\s*(\d+)\s*&\s*([0-9.]+)\s*years",
        re.MULTILINE,
    )
    for label, n, median in row_pattern.findall(latex_text):
        table_groups[label] = {"n": int(n), "median": float(median)}

    logrank_pattern = re.compile(
        r"Log-rank test.*?\\chi\^2\s*=\s*([0-9.]+).*?10\^\{-([0-9]+)\}",
        re.MULTILINE,
    )
    match = logrank_pattern.search(latex_text)
    if not match:  # pragma: no cover
        raise AssertionError("Unable to parse log-rank test row in duration table.")

    chi2 = float(match.group(1))
    power = int(match.group(2))
    return table_groups, chi2, power


def test_survival_figure_defaults_to_p0_duration_results(publication_figures_module, monkeypatch):
    monkeypatch.delenv("SDC_DURATION_TAG", raising=False)
    resolved = publication_figures_module._resolve_duration_results_filename()
    assert resolved == "module_8_duration_analysis__P0.json"


def test_table_14_matches_p0_duration_results():
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "sdc_2024_replication" / "scripts" / "statistical_analysis" / "results"
    latex_path = (
        root
        / "sdc_2024_replication"
        / "scripts"
        / "statistical_analysis"
        / "journal_article"
        / "sections"
        / "03_results.tex"
    )

    table_groups, chi2_table, power_table = _parse_duration_table(
        latex_path.read_text(encoding="utf-8")
    )

    duration_data = json.loads(
        (results_dir / "module_8_duration_analysis__P0.json").read_text(encoding="utf-8")
    )
    km_intensity = duration_data["results"]["kaplan_meier_by_intensity"]
    groups = km_intensity["groups"]

    expected_mapping = {
        "Q1 (Low)": "Q1 (Low)",
        "Q2": "Q2",
        "Q3": "Q3",
        "Q4 (High)": "Q4 (High)",
    }

    for table_label, json_label in expected_mapping.items():
        assert table_label in table_groups
        assert json_label in groups
        assert groups[json_label]["n_subjects"] == table_groups[table_label]["n"]
        assert groups[json_label]["median_survival"] == table_groups[table_label]["median"]

    log_rank = km_intensity["log_rank_test"]
    assert round(float(log_rank["test_statistic"]), 1) == chi2_table
    p_value = float(log_rank["p_value"])
    assert p_value > 0
    assert power_table == int(math.floor(-math.log10(p_value)))
