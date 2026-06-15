"""Build a standalone interactive North Dakota population-pyramid explorer.

Produces a single self-contained HTML file with two controls:

- a **geography selector** — the whole state, each of the 8 planning regions,
  each Census CBSA (Metropolitan / Micropolitan area, ND-portion counties), and
  each of the 53 counties; and
- a **year slider** (with Play) spanning the previous ~10 observed years
  (Census PEP) and the locked baseline projection years 2025-2055
  (``m2026r1`` / ``cfg-20260611-production-lock``; ADR-065 CBO-adjusted baseline).

Both eras are binned to the same 18 five-year age groups (0-4 ... 85+) so the
scrub is seamless across the observed->projected boundary. The page embeds the
data as JSON and redraws with ``Plotly.react`` on every control change; open the
HTML in any browser, no server required.

Data sources (single source of truth):
- Projection: ``data/projections/baseline/county/nd_county_*_projection_2025_2055_baseline.parquet``
  (single-year age x sex x race; aggregated over race, binned to 5-year groups).
- Historical: Census PEP county age-sex via
  ``scripts.analysis.walk_forward_validation.load_population_snapshot``.
- Region map: ``REGION_NAMES`` / ``COUNTY_TO_REGION`` in ``build_detail_workbooks``.
- CBSA map: ``data/raw/geographic/metro_crosswalk.csv`` (OMB delineation, ND filter).

Geographies that cross the state line (Fargo, Grand Forks, Wahpeton CBSAs) show
only their North Dakota counties; this is labeled in the UI.

Usage:
    python scripts/exports/build_pyramid_explorer.py
    python scripts/exports/build_pyramid_explorer.py --history-years 10 --output <path.html>
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Make sibling/script-package imports work under `python scripts/...` (own dir is sys.path[0]).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from build_detail_workbooks import COUNTY_TO_REGION, REGION_NAMES  # noqa: E402

COUNTY_PROJ_DIR = PROJECT_ROOT / "data" / "projections" / "baseline" / "county"
COUNTIES_CSV = PROJECT_ROOT / "data" / "raw" / "geographic" / "nd_counties.csv"
METRO_CROSSWALK = PROJECT_ROOT / "data" / "raw" / "geographic" / "metro_crosswalk.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "exports" / "pyramid_explorer" / "nd_pyramid_explorer.html"

PROJ_FIRST_YEAR = 2025
PROJ_LAST_YEAR = 2055
RUN_TAG = "m2026r1 / cfg-20260611-production-lock (2026-06-13)"
MALE_COLOR = "#2c5f8a"
FEMALE_COLOR = "#d99536"

_wfv = importlib.import_module("scripts.analysis.walk_forward_validation")
AGE_GROUPS: list[str] = list(_wfv.AGE_GROUP_LABELS)  # ["0-4", ..., "85+"], bottom -> top


# ---------------------------------------------------------------------------
# Crosswalks
# ---------------------------------------------------------------------------


def _county_names() -> dict[str, str]:
    df = pd.read_csv(COUNTIES_CSV, dtype={"county_fips": str})
    nd = df[df["county_fips"].str.startswith("38")]
    return dict(zip(nd["county_fips"], nd["county_name"], strict=False))


def _cbsa_definitions() -> list[dict]:
    """Return ND CBSAs with their ND-portion member counties, metros first."""
    df = pd.read_csv(METRO_CROSSWALK, dtype=str)
    nd = df[(df["state_fips"] == "38") & df["cbsa_code"].notna()]
    out = []
    for code, grp in nd.groupby("cbsa_code"):
        title = grp["cbsa_title"].iloc[0]
        metro_micro = grp["metro_micro"].iloc[0]  # "...Metropolitan..." / "...Micropolitan..."
        is_metro = "Metropolitan" in str(metro_micro)
        short = title.split(",")[0].strip()
        kind = "Metropolitan Statistical Area" if is_metro else "Micropolitan Statistical Area"
        cross_state = "-MN" in title or "-SD" in title or "-MT" in title
        out.append(
            {
                "code": code,
                "short": short,
                "title": title,
                "is_metro": is_metro,
                "kind": kind,
                "cross_state": cross_state,
                "counties": sorted(grp["county_fips"]),
            }
        )
    # Metros first, then micros; alphabetical within each.
    return sorted(out, key=lambda c: (not c["is_metro"], c["short"]))


# ---------------------------------------------------------------------------
# Population data (county-level, both eras)
# ---------------------------------------------------------------------------


def _age_to_group(age: float) -> str:
    return AGE_GROUPS[min(int(age) // 5, len(AGE_GROUPS) - 1)]


def load_county_history(years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        snap = _wfv.load_population_snapshot(year)  # county_fips x age_group x sex
        snap = snap.copy()
        snap["year"] = year
        frames.append(snap[["county_fips", "year", "age_group", "sex", "population"]])
    out = pd.concat(frames, ignore_index=True)
    out["era"] = "observed"
    return out


def load_county_projection() -> pd.DataFrame:
    files = sorted(
        f for f in COUNTY_PROJ_DIR.glob("nd_county_*_projection_2025_2055_baseline.parquet")
        if "_components" not in f.name
    )
    frames = []
    for path in files:
        fips = path.name.split("_")[2]  # nd_county_<FIPS>_projection_...
        df = pd.read_parquet(path)
        df["county_fips"] = fips
        df["age_group"] = df["age"].map(_age_to_group)
        agg = (
            df.groupby(["county_fips", "year", "age_group", "sex"], observed=True)["population"]
            .sum()
            .reset_index()
        )
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True)
    out["era"] = "projected"
    return out


def load_combined(history_years: int) -> pd.DataFrame:
    hist_years = list(range(PROJ_FIRST_YEAR - history_years, PROJ_FIRST_YEAR))
    combined = pd.concat(
        [load_county_history(hist_years), load_county_projection()], ignore_index=True
    )
    combined["county_fips"] = combined["county_fips"].astype(str)
    return combined


# ---------------------------------------------------------------------------
# Geography registry + aggregation to the embedded JSON payload
# ---------------------------------------------------------------------------


def build_geographies(names: dict[str, str]) -> list[dict]:
    all_fips = sorted(names)
    geos: list[dict] = [
        {"id": "state", "label": "North Dakota (statewide)", "group": "State",
         "kind": "State", "note": None, "counties": all_fips}
    ]

    for code in sorted(REGION_NAMES):
        members = sorted(f for f, r in COUNTY_TO_REGION.items() if r == code)
        member_names = ", ".join(names.get(f, f).replace(" County", "") for f in members)
        geos.append({
            "id": f"region_{code}", "label": f"Region {code} — {REGION_NAMES[code]}",
            "group": "Planning regions", "kind": "Planning region",
            "note": f"Planning Region {code} ({REGION_NAMES[code]}). Counties: {member_names}.",
            "counties": members,
        })

    for c in _cbsa_definitions():
        member_names = ", ".join(names.get(f, f) for f in c["counties"])
        note = f"{c['kind']}. ND counties: {member_names}."
        if c["cross_state"]:
            note += " Shows the North Dakota portion only (excludes the out-of-state counties)."
        geos.append({
            "id": f"cbsa_{c['code']}",
            "label": f"{c['short']} {'MSA' if c['is_metro'] else 'µSA'}"
                     + (" (ND portion)" if c["cross_state"] else ""),
            "group": "Metro / micropolitan areas", "kind": c["kind"], "note": note,
            "counties": c["counties"],
        })

    for fips in all_fips:
        geos.append({
            "id": f"county_{fips}", "label": names.get(fips, fips),
            "group": "Counties", "kind": "County",
            "note": f"{names.get(fips, fips)} (FIPS {fips}).", "counties": [fips],
        })
    return geos


def _aggregate(df: pd.DataFrame, geo: dict) -> dict:
    sub = df[df["county_fips"].isin(geo["counties"])]
    piv = sub.pivot_table(
        index=["year", "age_group"], columns="sex", values="population", aggfunc="sum",
        observed=False,
    ).fillna(0.0)
    years = sorted(sub["year"].unique())
    male: dict[str, list[int]] = {}
    female: dict[str, list[int]] = {}
    total: dict[str, int] = {}
    max_abs = 0
    for year in years:
        m, f = [], []
        for ag in AGE_GROUPS:
            try:
                row = piv.loc[(year, ag)]
            except KeyError:
                mv = fv = 0.0
            else:
                mv = float(row.get("Male", 0.0))
                fv = float(row.get("Female", 0.0))
            m.append(round(mv))
            f.append(round(fv))
            max_abs = max(max_abs, m[-1], f[-1])
        male[str(year)] = m
        female[str(year)] = f
        total[str(year)] = sum(m) + sum(f)
    return {**{k: geo[k] for k in ("id", "label", "group", "kind", "note")},
            "M": male, "F": female, "total": total, "max": max_abs}


def build_payload(history_years: int) -> dict:
    names = _county_names()
    combined = load_combined(history_years)
    geos = build_geographies(names)
    years = sorted(int(y) for y in combined["year"].unique())
    return {
        "age_groups": AGE_GROUPS,
        "years": years,
        "proj_start": PROJ_FIRST_YEAR,
        "proj_last": PROJ_LAST_YEAR,
        "run_tag": RUN_TAG,
        "male_color": MALE_COLOR,
        "female_color": FEMALE_COLOR,
        "geographies": [_aggregate(combined, g) for g in geos],
    }


# ---------------------------------------------------------------------------
# HTML / JS rendering
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>North Dakota Population Pyramid Explorer</title>
<script>__PLOTLYJS__</script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 18px 22px; color: #222; }
  h1 { font-size: 20px; margin: 0 0 2px; }
  .sub { color: #666; font-size: 12px; margin-bottom: 14px; }
  .controls { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; margin-bottom: 6px; }
  .ctl label { font-size: 12px; font-weight: 600; display: block; margin-bottom: 3px; color: #444; }
  select { font-size: 14px; padding: 5px 8px; min-width: 290px; }
  input[type=range] { width: 420px; vertical-align: middle; }
  #yearLabel { font-weight: 700; font-size: 15px; margin-left: 8px; }
  button { font-size: 13px; padding: 5px 12px; cursor: pointer; }
  #note { font-size: 12px; color: #555; margin: 6px 0 0; min-height: 16px; }
  #chart { margin-top: 4px; }
  .era-obs { color: #2c5f8a; font-weight: 700; }
  .era-proj { color: #b6701f; font-weight: 700; }
</style>
</head>
<body>
  <h1>North Dakota Statewide &amp; Sub-State Population Pyramids</h1>
  <div class="sub">Observed (Census PEP) through projected (baseline, CBO-adjusted) &middot; __RUN_TAG__
     &middot; 5-year age groups</div>
  <div class="controls">
    <div class="ctl"><label for="geo">Geography</label><span id="geoWrap"></span></div>
    <div class="ctl"><label for="year">Year <span id="yearLabel"></span></label>
      <input type="range" id="year" min="0" step="1"/>
      <button id="play">&#9654; Play</button>
    </div>
  </div>
  <div id="note"></div>
  <div id="chart" style="width:920px;height:680px;"></div>
<script>
const DATA = __DATA__;
const ages = DATA.age_groups, years = DATA.years, geos = DATA.geographies;

// Build geography dropdown grouped by kind.
const sel = document.createElement('select');
sel.id = 'geo';
const groupsOrder = ['State', 'Metro / micropolitan areas', 'Planning regions', 'Counties'];
const byGroup = {};
geos.forEach((g, i) => { (byGroup[g.group] = byGroup[g.group] || []).push(i); });
groupsOrder.forEach(grp => {
  if (!byGroup[grp]) return;
  const og = document.createElement('optgroup'); og.label = grp;
  byGroup[grp].forEach(i => {
    const o = document.createElement('option'); o.value = i; o.textContent = geos[i].label;
    og.appendChild(o);
  });
  sel.appendChild(og);
});
document.getElementById('geoWrap').appendChild(sel);

const yearSlider = document.getElementById('year');
yearSlider.max = years.length - 1;
yearSlider.value = years.indexOf(DATA.proj_start) >= 0 ? years.indexOf(DATA.proj_start) : 0;

function niceStep(top) {
  const steps = [100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000];
  const target = top / 5;
  for (const s of steps) if (s >= target) return s;
  return 100000;
}
function axis(top) {
  const step = niceStep(top);
  const hi = Math.ceil(top / step) * step;
  const vals = [], txt = [];
  for (let v = -hi; v <= hi + 1; v += step) { vals.push(v); txt.push(Math.abs(v).toLocaleString()); }
  return { range: [-hi * 1.04, hi * 1.04], tickvals: vals, ticktext: txt };
}

function render() {
  const g = geos[+sel.value];
  const yi = +yearSlider.value, year = years[yi], ys = String(year);
  const M = g.M[ys] || ages.map(() => 0), F = g.F[ys] || ages.map(() => 0);
  const observed = year < DATA.proj_start;
  const era = observed ? 'Observed (Census PEP)' : 'Projected (baseline, CBO-adjusted)';
  const ax = axis(g.max || 1);
  const traces = [
    { type: 'bar', orientation: 'h', y: ages, x: M.map(v => -v), name: 'Male',
      marker: { color: DATA.male_color }, customdata: M,
      hovertemplate: 'Age %{y}<br>Male: %{customdata:,}<extra></extra>' },
    { type: 'bar', orientation: 'h', y: ages, x: F, name: 'Female',
      marker: { color: DATA.female_color },
      hovertemplate: 'Age %{y}<br>Female: %{x:,}<extra></extra>' }
  ];
  const layout = {
    barmode: 'overlay', bargap: 0.08, template: 'plotly_white',
    title: { text: '<b>' + g.label + '</b> &mdash; ' + year + '<br><sub>' + era
             + '  &middot;  Total: ' + (g.total[ys] || 0).toLocaleString() + '</sub>', x: 0.5 },
    xaxis: { title: 'Population', range: ax.range, tickvals: ax.tickvals, ticktext: ax.ticktext,
             zeroline: true, zerolinecolor: '#888' },
    yaxis: { title: 'Age group', categoryorder: 'array', categoryarray: ages },
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.13 },
    margin: { l: 64, r: 30, t: 80, b: 70 },
    height: 680
  };
  Plotly.react('chart', traces, layout, { displaylogo: false, responsive: true });
  document.getElementById('yearLabel').textContent = year;
  const eraClass = observed ? 'era-obs' : 'era-proj';
  document.getElementById('note').innerHTML =
    '<span class="' + eraClass + '">' + (observed ? 'OBSERVED' : 'PROJECTED') + '</span> &middot; '
    + (g.note || '');
}

let timer = null;
const playBtn = document.getElementById('play');
playBtn.onclick = function () {
  if (timer) { clearInterval(timer); timer = null; playBtn.innerHTML = '&#9654; Play'; return; }
  playBtn.innerHTML = '&#10073;&#10073; Pause';
  timer = setInterval(() => {
    let v = +yearSlider.value + 1;
    if (v > +yearSlider.max) v = 0;
    yearSlider.value = v; render();
  }, 450);
};
sel.onchange = render;
yearSlider.oninput = render;
render();
</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    return (
        _HTML_TEMPLATE
        .replace("__PLOTLYJS__", get_plotlyjs())
        .replace("__RUN_TAG__", payload["run_tag"])
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-years", type=int, default=10,
                        help="Observed years before 2025 to include (default 10 -> from 2015).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.history_years)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")

    geos = payload["geographies"]
    kinds = pd.Series([g["kind"] for g in geos]).value_counts().to_dict()
    years = payload["years"]
    print(f"Wrote pyramid explorer: {args.output}")
    print(f"  Geographies: {len(geos)} ({kinds})")
    print(f"  Years: {years[0]}-{years[-1]} (observed {years[0]}-{PROJ_FIRST_YEAR - 1}, "
          f"projected {PROJ_FIRST_YEAR}-{years[-1]})")
    print(f"  Open in a browser: file://{args.output.resolve()}")


if __name__ == "__main__":
    main()
