from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "sdc_2024_replication"
    / "scripts"
    / "statistical_analysis"
    / "journal_article"
    / "claim_review"
    / "v3_phase3"
    / "citation_audit"
    / "check_citations.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("check_citations", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_report_nocite_affects_missing_and_uncited():
    module = load_module()
    report = module.build_report(
        bib_keys={"A"},
        cited_keys=set(),
        nocite_keys={"B"},
        nocite_all=False,
        occurrences={},
        files_scanned=[],
        apa_entries=[],
        duplicate_bib_keys=[],
        fixes_applied=[],
        fixes_unmatched=[],
    )

    missing_keys = [item["key"] for item in report["missing_in_bib"]]
    assert missing_keys == ["B"]
    assert report["uncited_in_bib"] == ["A"]


def test_build_report_nocite_all_suppresses_uncited():
    module = load_module()
    report = module.build_report(
        bib_keys={"A", "B"},
        cited_keys={"A"},
        nocite_keys=set(),
        nocite_all=True,
        occurrences={},
        files_scanned=[],
        apa_entries=[],
        duplicate_bib_keys=[],
        fixes_applied=[],
        fixes_unmatched=[],
    )

    assert report["uncited_in_bib"] == []
    assert report["summary"]["uncited_bib_count"] == 0


def test_parse_bib_text_expands_string_macros():
    module = load_module()
    bib_text = """
@string{jname = "Journal of Tests"}
@article{Key2020,
  author = {Doe, Jane},
  title = "A" # "Study",
  journal = jname,
  year = 2020
}
"""
    entries, macros = module.parse_bib_text(bib_text)
    assert macros["jname"] == "Journal of Tests"
    assert entries[0]["fields"]["journal"] == "Journal of Tests"
    assert entries[0]["fields"]["title"] == "AStudy"


def test_extract_all_citations_handles_extra_commands_and_multiline():
    module = load_module()
    text = "\\source{Key1}\nMore text \\cite{Key2,\nKey3}"
    pattern = module.build_cite_pattern(["source"])
    cited, nocite, nocite_all = module.extract_all_citations(text, pattern)
    assert cited == {"Key1", "Key2", "Key3"}
    assert nocite == set()
    assert nocite_all is False


def test_extract_all_citations_ignores_setcitestyle():
    module = load_module()
    text = "\\setcitestyle{authoryear,round,semicolon}\n\\cite{Key1}"
    pattern = module.build_cite_pattern([])
    cited, nocite, nocite_all = module.extract_all_citations(text, pattern)
    assert cited == {"Key1"}
    assert nocite == set()
    assert nocite_all is False


def test_parse_tex_files_multiline_occurrence_line(tmp_path):
    module = load_module()
    tex_path = tmp_path / "main.tex"
    tex_path.write_text("Text \\\\cite{Key1,\nKey2}\nMore", encoding="utf-8")
    pattern = module.build_cite_pattern([])

    occurrences = module.parse_tex_files([tex_path], pattern)
    assert set(occurrences.keys()) == {"Key1", "Key2"}
    assert occurrences["Key1"][0]["line"] == 1
    assert occurrences["Key2"][0]["line"] == 1


def test_gather_tex_files_follows_input(tmp_path):
    module = load_module()
    root = tmp_path
    sections = root / "sections"
    sections.mkdir()
    main_tex = root / "main.tex"
    intro_tex = sections / "intro.tex"
    main_tex.write_text("\\input{sections/intro}", encoding="utf-8")
    intro_tex.write_text("Intro text", encoding="utf-8")

    tex_files = module.gather_tex_files(root, [], [], False)
    tex_paths = {path.resolve() for path in tex_files}
    assert main_tex.resolve() in tex_paths
    assert intro_tex.resolve() in tex_paths


def test_apply_fixes_updates_fields_and_entry_type():
    module = load_module()
    entries = [{"entry_type": "misc", "key": "Key1", "fields": {"title": "Old"}}]
    fixes = [
        {
            "citation_key": "Key1",
            "entry_type": "report",
            "fields": {"year": "2020", "publisher": "Test Pub"},
        },
        {"citation_key": "MissingKey", "fields": {"year": "1999"}},
    ]

    applied, unmatched = module.apply_fixes(entries, fixes)
    assert applied == ["Key1"]
    assert unmatched == ["MissingKey"]
    assert entries[0]["entry_type"] == "report"
    assert entries[0]["fields"]["year"] == "2020"
    assert entries[0]["fields"]["publisher"] == "Test Pub"


def test_find_duplicate_bib_keys():
    module = load_module()
    entries = [
        {"entry_type": "article", "key": "DupKey", "fields": {}},
        {"entry_type": "book", "key": "DupKey", "fields": {}},
        {"entry_type": "misc", "key": "UniqueKey", "fields": {}},
    ]

    assert module.find_duplicate_bib_keys(entries) == ["DupKey"]


def test_render_report_html_includes_missing_placeholders():
    module = load_module()
    report = {
        "generated_at": "2025-01-01T00:00:00+00:00",
        "summary": {
            "bib_entry_count": 1,
            "citation_key_count": 0,
            "missing_in_bib_count": 0,
            "uncited_bib_count": 0,
            "duplicate_bib_count": 0,
        },
        "missing_in_bib": [],
        "uncited_in_bib": [],
        "duplicate_bib_keys": [],
        "apa_audit": {
            "entries_missing_required": 1,
            "entries_missing_recommended": 0,
        },
    }
    entry = {
        "citation_key": "Key1",
        "entry_type": "report",
        "entry_type_canonical": "report",
        "fields": {"title": "Test"},
        "missing_required": ["creator", "year", "institution"],
        "missing_recommended": [],
        "status": "missing_required",
        "apa_edition": "7th",
        "notes": [],
    }

    html = module.render_report_html(report, [entry])
    assert "citation_audit_report" not in html
    assert "missing-required" in html
    assert "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" in html
