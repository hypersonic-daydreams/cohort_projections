#!/usr/bin/env python3
"""Build state + county comparison CSVs for the reference_intl_migration sensitivity.

Compares the published locked baseline against the in-bundle control run (numerator
unchanged, 10,051) and the corrected run (numerator = 3,350.33). The control should
reproduce the published locked trajectory exactly, proving the corrected-vs-control
delta is attributable solely to the numerator change.

NOT a production artifact. See README.md in this directory.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

BUNDLE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BUNDLE, "..", "..", ".."))

PUBLISHED_STATE = os.path.join(
    REPO, "data/projections/baseline/state/nd_state_38_projection_2025_2055_baseline_summary.csv"
)
CONTROL_STATE = os.path.join(
    BUNDLE, "reference_control/baseline/state/nd_state_38_projection_2025_2055_baseline_summary.csv"
)
CORRECTED_STATE = os.path.join(
    BUNDLE, "corrected_refintl/baseline/state/nd_state_38_projection_2025_2055_baseline_summary.csv"
)


def _state(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.set_index("year")["total_population"]


def main() -> None:
    pub = _state(PUBLISHED_STATE)
    ctl = _state(CONTROL_STATE)
    cor = _state(CORRECTED_STATE)
    base = pub.loc[2025]

    traj = pd.DataFrame({
        "published_locked": pub,
        "control_run": ctl,
        "corrected_run": cor,
    })
    traj["corrected_minus_locked"] = traj["corrected_run"] - traj["published_locked"]
    traj["locked_pct_vs_2025"] = (traj["published_locked"] / base - 1.0) * 100
    traj["corrected_pct_vs_2025"] = (traj["corrected_run"] / base - 1.0) * 100
    traj["control_minus_published"] = traj["control_run"] - traj["published_locked"]
    traj = traj.round(2)
    out_state = os.path.join(BUNDLE, "comparison_state_trajectory.csv")
    traj.to_csv(out_state)

    # Reproducibility check
    max_ctl_diff = traj["control_minus_published"].abs().max()

    # Troughs
    locked_trough_yr = int(pub.idxmin())
    corrected_trough_yr = int(cor.idxmin())

    # County comparison at 2055. Per-county summary files start at 2026, so the
    # 2025 base comes from countys_summary.csv (base_population), identical in both runs.
    names, base_pop = {}, {}
    cs = os.path.join(REPO, "data/projections/baseline/county/countys_summary.csv")
    if os.path.exists(cs):
        cdf = pd.read_csv(cs)
        names = dict(zip(cdf["fips"].astype(str), cdf["name"], strict=False))
        base_pop = dict(zip(cdf["fips"].astype(str), cdf["base_population"], strict=False))

    def county_series(root: str) -> dict[str, pd.Series]:
        out = {}
        for f in glob.glob(os.path.join(root, "nd_county_*_projection_2025_2055_baseline_summary.csv")):
            fips = os.path.basename(f).split("_")[2]
            out[fips] = pd.read_csv(f).set_index("year")["total_population"]
        return out

    locked_c = county_series(os.path.join(REPO, "data/projections/baseline/county"))
    corrected_c = county_series(os.path.join(BUNDLE, "corrected_refintl/baseline/county"))

    rows = []
    for fips in sorted(locked_c):
        if fips not in corrected_c:
            continue
        lk, cr = locked_c[fips], corrected_c[fips]
        b = base_pop.get(str(fips))
        rows.append({
            "fips": fips,
            "county": names.get(str(fips), ""),
            "base_2025": round(b, 1) if b is not None else "",
            "locked_2055": round(lk.loc[2055], 1),
            "corrected_2055": round(cr.loc[2055], 1),
            "delta_2055": round(cr.loc[2055] - lk.loc[2055], 1),
            "locked_growth_pct": round((lk.loc[2055] / b - 1) * 100, 1) if b else "",
            "corrected_growth_pct": round((cr.loc[2055] / b - 1) * 100, 1) if b else "",
        })
    cdf_out = pd.DataFrame(rows).sort_values("delta_2055", ascending=False)
    out_county = os.path.join(BUNDLE, "comparison_county_2055.csv")
    cdf_out.to_csv(out_county, index=False)

    # Reconciliation: corrected state == sum of corrected counties (ADR-054).
    # County files start at 2026, so only reconcile years >= 2026.
    recon = {}
    for yr in (2030, corrected_trough_yr, 2055):
        if yr < 2026:
            continue
        csum = sum(s.loc[yr] for s in corrected_c.values())
        recon[yr] = round(csum - cor.loc[yr], 4)

    # Report
    print("=== REPRODUCIBILITY: control vs published (max abs diff, all years) ===")
    print(f"  {max_ctl_diff:.4f}  (≈0 confirms environment reproduces the locked run)")
    print()
    print("=== STATE TROUGH ===")
    print(f"  Locked/published trough : {locked_trough_yr} = {pub.loc[locked_trough_yr]:,.0f} "
          f"({(pub.loc[locked_trough_yr]/base-1)*100:+.2f}% vs 2025)")
    print(f"  Corrected trough        : {corrected_trough_yr} = {cor.loc[corrected_trough_yr]:,.0f} "
          f"({(cor.loc[corrected_trough_yr]/base-1)*100:+.2f}% vs 2025)")
    print()
    print("=== KEY YEARS (locked → corrected) ===")
    for yr in (2025, 2028, 2030, 2040, 2050, 2055):
        print(f"  {yr}: {pub.loc[yr]:>9,.0f} → {cor.loc[yr]:>9,.0f}  "
              f"(Δ {cor.loc[yr]-pub.loc[yr]:+,.0f}; corrected {(cor.loc[yr]/base-1)*100:+.2f}% vs 2025)")
    print()
    print("=== ADR-054 reconciliation (corrected state − Σcounties; should be ~0) ===")
    for yr, d in recon.items():
        print(f"  {yr}: {d}")
    print()
    print("=== Largest upward county revisions @2055 (top 6) ===")
    for _, r in cdf_out.head(6).iterrows():
        print(f"  {r['county']:<16} {r['locked_2055']:>9,.0f} → {r['corrected_2055']:>9,.0f}  (Δ {r['delta_2055']:+,.0f})")
    print()
    print(f"Wrote: {out_state}")
    print(f"Wrote: {out_county}")


if __name__ == "__main__":
    main()
