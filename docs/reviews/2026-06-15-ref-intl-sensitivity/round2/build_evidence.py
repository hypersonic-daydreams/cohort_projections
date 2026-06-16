#!/usr/bin/env python3
"""Build the curated DATA evidence CSVs for the GPT-5.5 Pro round-2 review.

Round 1 left the reviewer unable to verify several calculations because only CODE
(not the processed DATA) was in the package. This exports small, high-signal CSV
slices of the parquet inputs the reviewer explicitly asked for — full tables where
they are small and decisive (survival, fertility, county base pop), samples/aggregates
where the full table is large (county distribution, GQ), and a reservation-county
filter for PEP. All outputs land in this folder so the runner can inline them as text.

Run:  python docs/reviews/2026-06-15-ref-intl-sensitivity/round2/build_evidence.py
"""

from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def _read(rel: str) -> pd.DataFrame:
    path = os.path.join(REPO, rel)
    df = pd.read_parquet(path)
    print(f"[read] {rel}: {len(df):,} rows, cols={list(df.columns)}")
    return df


def _write(df: pd.DataFrame, name: str, note: str = "") -> None:
    out = os.path.join(HERE, name)
    df.to_csv(out, index=False)
    print(f"[write] {name}: {len(df):,} rows{('  — ' + note) if note else ''}")


def main() -> None:
    # 1) Operative survival table (corrected run). FULL — small (4,242 rows) and decisive:
    #    no race column => proves race-flat (ADR-068 D3); lets reviewer check open-90+ fix by sex/year.
    surv = _read("data/processed/mortality/nd_adjusted_survival_projections.parquet")
    _write(surv, "evidence_survival_table_FULL.csv", "FULL: no race col proves race-flat; check age==90 open-interval")
    # Focused 90+ slice for quick verification of the ADR-068 open-ended fix.
    if "age" in surv.columns:
        hi = surv[surv["age"] >= 85].sort_values([c for c in ["year", "age", "sex"] if c in surv.columns])
        _write(hi, "evidence_survival_85plus.csv", "ages 85+ across all years/sexes (open-90+ fix focus)")

    # 2) Fertility ASFR table. FULL — tiny (49 rows). Reviewer wanted the actual rates + TFR.
    fert = _read("data/processed/fertility_rates.parquet")
    _write(fert, "evidence_fertility_rates_FULL.csv", "FULL ASFR input (before the in-engine -5% scenario cut)")
    # Compute TFR per race group. ASFR here is per-1,000 women in 5-year age groups
    # ('15-19'..'45-49'), single year (2023). TFR = 5 * sum(ASFR)/1000.
    if {"asfr", "age", "race_ethnicity"}.issubset(fert.columns):
        is_5yr = fert["age"].astype(str).str.contains("-").any()
        width = 5 if is_5yr else 1
        scale = 1000.0 if fert["asfr"].max() > 20 else 1.0  # per-1,000 vs proportion
        tfr = (fert.groupby("race_ethnicity")["asfr"].sum() * width / scale).round(4)
        print(f"[TFR] 5yr-groups={is_5yr}, per-1000={scale==1000.0}; TFR by race (before -5% cut):")
        rows = []
        for k, v in tfr.items():
            after = round(v * 0.95, 4)
            print(f"       {k}: TFR={v}  (after -5% scenario: {after})")
            rows.append({"race_ethnicity": k, "tfr_before_cut": v, "tfr_after_minus5pct": after})
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "evidence_fertility_TFR.csv"), index=False)
        print("[write] evidence_fertility_TFR.csv")
        print(f"[NOTE] fertility table year(s) present: {sorted(fert['year'].unique())} "
              f"(config comment says 'pooled CDC WONDER 2020-2023' — reviewer to reconcile)")

    # 3) County base population 2025 (proves the 53 county totals sum to the locked 799,358).
    #    The published export carries a state-total row; drop it so the 53 counties sum cleanly.
    cpop_path = os.path.join(REPO, "data/exports/nd_county_population_2020_2025.csv")
    cpop = pd.read_csv(cpop_path)
    print(f"[read] nd_county_population_2020_2025.csv: {len(cpop):,} rows, cols={list(cpop.columns)}")
    pop_col = "population_2025"
    # Drop any state-total / aggregate row (name like North Dakota / State, or value == state total).
    state_total = 799358
    mask_total = (
        cpop["county_name"].astype(str).str.contains("North Dakota|State|Total", case=False, na=False)
        | (cpop[pop_col] == state_total)
    )
    counties = cpop[~mask_total].copy()
    total = counties[pop_col].sum()
    print(f"[check] {len(counties)} counties, {pop_col} sum = {total:,.0f}  (locked state base = {state_total:,})")
    counties[["county_name", pop_col]].to_csv(
        os.path.join(HERE, "evidence_county_base_pop_2025.csv"), index=False)
    with open(os.path.join(HERE, "evidence_county_base_pop_2025.csv"), "a") as fh:
        fh.write(f"# SUM_OF_{len(counties)}_COUNTIES,{total:.0f}\n")
    print("[write] evidence_county_base_pop_2025.csv (with SUM footer)")

    # 4) County age-sex-race distribution — SAMPLE only (11,448 rows full). The blend question is a
    #    CODE question; a sample shows structure. Include 3 counties (1 small <5000, 2 larger).
    dist = _read("data/processed/county_age_sex_race_distributions.parquet")
    fips_col = "fips" if "fips" in dist.columns else [c for c in dist.columns if "fips" in c.lower()][0]
    sample_fips = sorted(dist[fips_col].unique())[:3]
    dist_s = dist[dist[fips_col].isin(sample_fips)]
    _write(dist_s, "evidence_county_distribution_SAMPLE.csv", f"sample of fips {list(sample_fips)} (full=11,448 rows)")

    # 5) Group Quarters 2025 (held constant). County aggregate (53 rows) + small sample.
    gq = _read("data/processed/gq_county_age_sex_2025.parquet")
    gcol = "county_fips" if "county_fips" in gq.columns else [c for c in gq.columns if "fips" in c.lower()][0]
    pcol = "gq_population" if "gq_population" in gq.columns else [c for c in gq.columns if "pop" in c.lower()][0]
    gq_agg = gq.groupby(gcol)[pcol].sum().reset_index().rename(columns={pcol: "gq_total_2025"})
    print(f"[check] statewide GQ 2025 = {gq_agg['gq_total_2025'].sum():,.0f}")
    _write(gq_agg, "evidence_gq_2025_by_county.csv", "county GQ totals (held constant through horizon)")
    _write(gq[gq[gcol].isin(sorted(gq[gcol].unique())[:2])], "evidence_gq_2025_SAMPLE.csv", "2-county age/sex sample")

    # 6) Historical GQ — by-year aggregate (shows the backward-constant construction) + sample.
    gqh = _read("data/processed/gq_county_age_sex_historical.parquet")
    if "year" in gqh.columns:
        ycol = "gq_population" if "gq_population" in gqh.columns else [c for c in gqh.columns if "pop" in c.lower()][0]
        gqh_year = gqh.groupby("year")[ycol].sum().reset_index().rename(columns={ycol: "gq_statewide_total"})
        _write(gqh_year, "evidence_gq_historical_by_year.csv", "statewide GQ by year (2000-2015 backward-constant from 2020)")

    # 7) PEP components — reservation counties (ADR-045 recalibration target) + state intl totals.
    pep = _read("data/processed/pep_county_components_2000_2025.parquet")
    print(f"[pep cols] {list(pep.columns)}")
    if "county_name" in pep.columns:
        res = pep[pep["county_name"].astype(str).str.contains(
            "Benson|Sioux|Rolette", case=False, na=False)].copy()
        if "is_preferred_estimate" in res.columns and res["is_preferred_estimate"].any():
            res = res[res["is_preferred_estimate"]]
        keep = [c for c in ["county_fips", "county_name", "year", "netmig", "intl_mig",
                            "domestic_mig", "residual"] if c in res.columns]
        _write(res[keep], "evidence_pep_reservation_counties.csv", "Benson/Sioux/Rolette components (ADR-045)")
    # International-migration columns at the state level, to corroborate the 3,158/4,083/2,810 numerator.
    intl_cols = [c for c in pep.columns if "intl" in c.lower() or "international" in c.lower()]
    print(f"[pep intl cols] {intl_cols}")
    if intl_cols and "year" in pep.columns:
        yr = pep[pep["year"].isin([2023, 2024, 2025])]
        if "is_preferred_estimate" in yr.columns and yr["is_preferred_estimate"].any():
            yr = yr[yr["is_preferred_estimate"]]
        agg = yr.groupby("year")[intl_cols].sum().reset_index()
        _write(agg, "evidence_pep_state_intl_2023_2025.csv", "state intl-migration by year (numerator source: sum 10,051 / mean 3,350.33)")
        s = agg["intl_mig"].sum()
        print(f"[check] state intl 2023-2025: by-year={dict(zip(agg['year'], agg['intl_mig'].round(0)))}; "
              f"SUM={s:,.1f} (config 10,051); MEAN={s/3:,.2f} (corrected 3,350.33)")

    print("\nDONE. Evidence CSVs written to:", HERE)


if __name__ == "__main__":
    main()
