"""
Build ND-adjusted survival rates from NVSR 74-12 state life tables.

Created: 2026-02-23
ADR: 053 (Part B — ND-Adjusted Mortality Rates)
Author: Claude Code / N. Haarstad

Purpose
-------
Calibrate national race-specific survival rates to North Dakota's actual
mortality level using the ND/national qx ratio method. This is standard
demographic practice (used by Census Bureau for state projections, PPL-47):
apply a state-level adjustment to preserve race differentials from national
data while matching the state's overall mortality.

Method
------
1. Parse NVSR 74-12 state life tables for ND (2022 data year) and matching
   national 2022 life tables. These are Excel files with 3 header rows, then
   data rows for ages 0-1, 1-2, ..., 99-100, 100+.
2. Compute the ND/national qx ratio by single year of age and sex:
       ratio[age, sex] = ND_qx_2022[age, sex] / National_qx_2022[age, sex]
3. Cap the ratio at [0.5, 2.0] to prevent extreme adjustments at ages where
   small ND death counts produce volatile qx values.
4. Apply the ratio to all race-specific national survival rates (2023 vintage):
       ND_qx[age, sex, race] = National_qx_2023[age, sex, race] × ratio[age, sex]
       ND_survival[age, sex, race] = 1 - ND_qx[age, sex, race]
5. Recompute lx (survivors) from the adjusted qx for each sex × race cohort.

Key finding: ND e0 is ABOVE national
-------------------------------------
The original ADR-053 proposal assumed ND life expectancy was ~0.6 years below
national. NVSR 74-12 data shows the opposite — ND is ~0.5 years ABOVE national:
    ND Male e0:   75.37  (US: 74.76, delta +0.61)
    ND Female e0: 80.76  (US: 80.24, delta +0.52)
    ND Total e0:  77.93  (US: 77.46, delta +0.47)

This means the ratio adjustment slightly *improves* survival rates (ratio < 1
at most ages). The ratio method works correctly regardless of direction.

Key design decisions
--------------------
- **Aggregate ratio, not race-specific**: NVSR 74-12 provides ND life tables
  by sex only, not by race. The aggregate ratio is dominated by the White
  majority (~82% of ND population), so the adjustment primarily captures
  White mortality patterns. AIAN-specific adjustment is minimal (+0.06 to
  +0.45 years), which understates the true AIAN mortality disadvantage.
  Future enhancement: compute AIAN-specific ratio from CDC WONDER death counts.
- **2022 ND tables applied to 2023 national tables**: The NVSR 74-12 state
  tables are from 2022, while the race-specific national tables are from 2023.
  This 1-year mismatch is acceptable because year-to-year mortality improvement
  is <1%, and using the most recent race-specific data (2023) is preferred.
- **Ratio cap [0.5, 2.0]**: At very old ages (95+), small ND death counts
  produce noisy qx estimates. Capping prevents implausible survival rates.
- **Age 100 unmatched**: ND life tables go to age 99; national go to 100.
  12 records at age 100 are left unadjusted (negligible population).

Validation results (2026-02-23)
-------------------------------
- 1,200 of 1,212 survival rate records adjusted (12 at age 100 unmatched)
- Male ratio: mean=0.9370, min=0.5000, max=1.5727
  (ND better at 77 ages, worse at 23, equal at 0)
- Female ratio: mean=0.9372, min=0.5000, max=2.0000
  (ND better at 76 ages, worse at 24, equal at 0)
- Computed ND e0 from adjusted 'total' rates:
    Male: 75.86 (published 75.37, delta 0.49 — expected from 2022/2023 mismatch)
    Female: 81.09 (published 80.76, delta 0.33)
- Life expectancy delta (adjusted vs. original) ranges from +0.06 (AIAN) to
  +0.60 (Asian female), reflecting the aggregate ND advantage

Inputs
------
- data/raw/mortality/nd_lifetable_2022_ND2.xlsx
    NVSR 74-12, ND male life table (2022). 100 ages (0-99), 7 columns
    (age_label, qx, lx, dx, Lx, Tx, ex). Published e0 = 75.37.
    Downloaded from: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/74-12/
    Download date: 2026-02-23.
- data/raw/mortality/nd_lifetable_2022_ND3.xlsx
    NVSR 74-12, ND female life table (2022). Published e0 = 80.76.
- data/raw/mortality/us_lifetable_male_2022.xlsx
    National male life table (2022), used as denominator for ratio computation.
- data/raw/mortality/us_lifetable_female_2022.xlsx
    National female life table (2022), used as denominator for ratio computation.
- data/processed/survival_rates.parquet
    Current national race-specific survival rates (2023 vintage, produced by
    survival_rates.py from NVSR 74-06 life tables). 1,212 rows covering
    ages 0-100, 2 sexes, 6 race categories. This file is overwritten in place.

Output
------
- data/processed/survival_rates.parquet (ND-adjusted, overwritten in place)
- data/processed/survival_rates.csv (companion CSV for inspection)

Usage
-----
    python scripts/data/build_nd_survival_rates.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_life_table_excel(filepath: Path) -> pd.DataFrame:
    """Parse NVSR life table Excel into clean DataFrame.

    These files have a title row, header rows with column names, then data.
    Returns DataFrame with columns [age, qx, lx, dx, Lx, Tx, ex].
    """
    df = pd.read_excel(filepath, header=None, skiprows=3)
    df.columns = ["age_label", "qx", "lx", "dx", "Lx", "Tx", "ex"]

    # Parse age: "0–1" -> 0, "1–2" -> 1, ..., "99–100" -> 99, "100 and over" -> 100
    def parse_age(label):
        s = str(label).strip()
        if "and over" in s:
            return int(s.split()[0])
        if "–" in s:
            return int(s.split("–")[0])
        if "-" in s:
            return int(s.split("-")[0])
        try:
            return int(s)
        except ValueError:
            return None

    df["age"] = df["age_label"].apply(parse_age)
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].astype(int)

    # Ensure qx is numeric
    df["qx"] = pd.to_numeric(df["qx"], errors="coerce")
    df = df.dropna(subset=["qx"])

    return df[["age", "qx", "lx", "dx", "Lx", "Tx", "ex"]].reset_index(drop=True)


def compute_nd_ratio(nd_table: pd.DataFrame, national_table: pd.DataFrame) -> pd.DataFrame:
    """Compute ND/national qx ratio by age.

    Returns DataFrame with [age, nd_qx, national_qx, ratio].
    Ratio is capped at [0.5, 2.0] to prevent extreme adjustments for volatile ages.
    """
    merged = nd_table[["age", "qx"]].merge(
        national_table[["age", "qx"]],
        on="age",
        how="inner",
        suffixes=("_nd", "_national"),
    )

    # Compute ratio, guarding against division by zero
    merged["ratio"] = merged.apply(
        lambda r: r["qx_nd"] / r["qx_national"] if r["qx_national"] > 0 else 1.0,
        axis=1,
    )

    # Cap ratio to prevent extreme adjustments
    merged["ratio"] = merged["ratio"].clip(lower=0.5, upper=2.0)

    return merged


def main():
    print("=" * 70)
    print("Building ND-Adjusted Survival Rates (ADR-053)")
    print("=" * 70)

    mortality_dir = PROJECT_ROOT / "data" / "raw" / "mortality"

    # Step 1: Parse ND and national life tables
    print("\nStep 1: Parsing life tables")
    nd_male = parse_life_table_excel(mortality_dir / "nd_lifetable_2022_ND2.xlsx")
    nd_female = parse_life_table_excel(mortality_dir / "nd_lifetable_2022_ND3.xlsx")
    us_male = parse_life_table_excel(mortality_dir / "us_lifetable_male_2022.xlsx")
    us_female = parse_life_table_excel(mortality_dir / "us_lifetable_female_2022.xlsx")

    print(f"  ND Male: {len(nd_male)} ages, e0={nd_male.iloc[0]['ex']:.2f}")
    print(f"  ND Female: {len(nd_female)} ages, e0={nd_female.iloc[0]['ex']:.2f}")
    print(f"  US Male: {len(us_male)} ages, e0={us_male.iloc[0]['ex']:.2f}")
    print(f"  US Female: {len(us_female)} ages, e0={us_female.iloc[0]['ex']:.2f}")

    # Step 2: Compute ND/national mortality ratios
    print("\nStep 2: Computing ND/national mortality ratios")
    male_ratio = compute_nd_ratio(nd_male, us_male)
    female_ratio = compute_nd_ratio(nd_female, us_female)

    male_ratio["sex"] = "male"
    female_ratio["sex"] = "female"

    ratio_df = pd.concat([male_ratio, female_ratio], ignore_index=True)

    print(f"  Male ratio: mean={male_ratio['ratio'].mean():.4f}, "
          f"min={male_ratio['ratio'].min():.4f}, max={male_ratio['ratio'].max():.4f}")
    print(f"  Female ratio: mean={female_ratio['ratio'].mean():.4f}, "
          f"min={female_ratio['ratio'].min():.4f}, max={female_ratio['ratio'].max():.4f}")

    # Count ages where ND mortality is worse vs better
    for sex_label, sex_ratio in [("Male", male_ratio), ("Female", female_ratio)]:
        worse = (sex_ratio["ratio"] > 1.0).sum()
        better = (sex_ratio["ratio"] < 1.0).sum()
        equal = (sex_ratio["ratio"] == 1.0).sum()
        print(f"  {sex_label}: ND worse at {worse} ages, better at {better}, equal at {equal}")

    # Step 3: Apply ratios to existing race-specific survival rates
    print("\nStep 3: Applying ND adjustment to race-specific survival rates")
    survival_file = PROJECT_ROOT / "data" / "processed" / "survival_rates.parquet"
    survival_df = pd.read_parquet(survival_file)
    print(f"  Loaded {len(survival_df)} survival rate records")
    print(f"  Races: {sorted(survival_df['race_ethnicity'].unique())}")
    print(f"  Sexes: {sorted(survival_df['sex'].unique())}")

    # Build ratio lookup
    ratio_lookup = ratio_df.set_index(["age", "sex"])["ratio"]

    # Apply adjustment
    original_rates = survival_df.copy()
    adjusted_count = 0

    for idx, row in survival_df.iterrows():
        age = row["age"]
        sex = row["sex"]
        key = (age, sex)

        if key in ratio_lookup.index:
            ratio = ratio_lookup[key]
            old_qx = row["qx"]
            new_qx = old_qx * ratio
            new_qx = min(new_qx, 1.0)  # Cap at 1.0
            new_qx = max(new_qx, 0.0)  # Floor at 0.0
            survival_df.at[idx, "qx"] = new_qx
            survival_df.at[idx, "survival_rate"] = 1 - new_qx
            adjusted_count += 1

    print(f"  Adjusted {adjusted_count} records")

    # Recompute lx from adjusted qx
    for (_sex, _race), group in survival_df.groupby(["sex", "race_ethnicity"]):
        sorted_idx = group.sort_values("age").index
        lx = 100000.0
        for i in sorted_idx:
            survival_df.at[i, "lx"] = lx
            qx = survival_df.at[i, "qx"]
            lx = lx * (1 - qx)

    # Step 4: Validate
    print("\nStep 4: Validation")

    # Compare life expectancy before and after
    print(f"\n  {'Race':<35} {'Sex':<8} {'Original e0':>12} {'ND-Adj e0':>12} {'Delta':>8}")
    print("  " + "-" * 80)

    for (sex, race), group in original_rates.groupby(["sex", "race_ethnicity"]):
        orig_e0 = group.sort_values("age")["survival_rate"].cumprod().sum()
        adj_group = survival_df[
            (survival_df["sex"] == sex) & (survival_df["race_ethnicity"] == race)
        ]
        adj_e0 = adj_group.sort_values("age")["survival_rate"].cumprod().sum()
        delta = adj_e0 - orig_e0
        print(f"  {race:<35} {sex:<8} {orig_e0:>12.2f} {adj_e0:>12.2f} {delta:>+8.2f}")

    # Overall check against published ND values
    for sex in ["male", "female"]:
        total = survival_df[
            (survival_df["sex"] == sex) & (survival_df["race_ethnicity"] == "total")
        ].sort_values("age")
        e0 = total["survival_rate"].cumprod().sum()
        nd_published = {"male": 75.37, "female": 80.76}
        print(f"\n  ND {sex.title()} e0 (computed from adjusted 'total'): {e0:.2f} "
              f"(published: {nd_published[sex]})")

    # Step 5: Save
    print("\nStep 5: Saving ND-adjusted survival rates")
    survival_df.to_parquet(survival_file, compression="gzip", index=False)
    print(f"  Saved to {survival_file}")

    # Also save CSV for inspection
    csv_file = survival_file.with_suffix(".csv")
    survival_df.to_csv(csv_file, index=False)
    print(f"  Saved CSV to {csv_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
