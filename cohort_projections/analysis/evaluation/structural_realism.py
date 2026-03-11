"""Module 2: Structural Realism checks for projection evaluation.

Evaluates whether projected demographic outputs are structurally plausible,
independent of accuracy against actuals.  Checks cover component realism,
age-structure divergence, cohort continuity, demographic accounting
identities, and cross-county distributional realism.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    jensen_shannon_divergence,
    kullback_leibler_divergence,
    mae,
    mape,
    mean_signed_percentage_error,
)
from .utils import build_lookup, make_diagnostic_record

logger = logging.getLogger(__name__)

# Default thresholds (overridden by config if supplied)
_DEFAULT_REALISM_CONFIG: dict[str, Any] = {
    "max_jsd_age_distribution": 0.05,
    "max_cohort_survival_residual": 0.10,
    "plausible_fertility_range": [0.0, 0.30],
    "plausible_migration_rate_range": [-0.50, 0.50],
}

# Standard 5-year age groups used by the projection system
_STANDARD_AGE_GROUPS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85+",
]

# Mapping from age group to the next cohort age group (5-year shift)
_COHORT_SUCCESSOR: dict[str, str] = dict(
    zip(_STANDARD_AGE_GROUPS[:-1], _STANDARD_AGE_GROUPS[1:], strict=True)
)


def _realism_diagnostic(
    run_id: str,
    metric_name: str,
    geography: str,
    target: str,
    value: float,
    *,
    geography_group: str = "",
    horizon: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a realism diagnostic record with metric_group='realism'."""
    return make_diagnostic_record(
        run_id=run_id,
        metric_name=metric_name,
        metric_group="realism",
        geography=geography,
        target=target,
        value=value,
        geography_group=geography_group,
        horizon=horizon,
        notes=notes,
    )


def _compute_divergence_metrics(
    run_id: str,
    proj_dist: np.ndarray,
    act_dist: np.ndarray,
    geography: str,
    horizon: int | None,
    jsd_metric_name: str,
    kld_metric_name: str,
    target: str = "population",
    max_jsd: float | None = None,
) -> list[dict[str, Any]]:
    """Compute JSD and KLD for two distributions and return diagnostic records.

    Shared helper used by both :meth:`age_structure_realism` and
    :meth:`distributional_realism` to reduce duplicated divergence logic.
    """
    records: list[dict[str, Any]] = []
    if act_dist.sum() <= 0 or proj_dist.sum() <= 0:
        return records

    jsd = jensen_shannon_divergence(proj_dist, act_dist)
    notes = ""
    if max_jsd is not None:
        notes = "WARN" if jsd > max_jsd else "OK"
    records.append(_realism_diagnostic(
        run_id=run_id,
        metric_name=jsd_metric_name,
        geography=geography,
        target=target,
        value=jsd,
        horizon=horizon,
        notes=notes,
    ))

    kld = kullback_leibler_divergence(act_dist, proj_dist)
    records.append(_realism_diagnostic(
        run_id=run_id,
        metric_name=kld_metric_name,
        geography=geography,
        target=target,
        value=kld,
        horizon=horizon,
    ))

    return records


class StructuralRealismModule:
    """Module 2 of the Evaluation Blueprint: structural realism checks.

    Parameters
    ----------
    realism_config:
        Dictionary of realism thresholds.  Keys should match those in
        ``config/evaluation_config.yaml`` under the ``realism`` section.
        Missing keys fall back to built-in defaults.
    """

    def __init__(self, realism_config: dict[str, Any] | None = None) -> None:
        cfg = dict(_DEFAULT_REALISM_CONFIG)
        if realism_config:
            cfg.update(realism_config)
        self.max_jsd: float = cfg["max_jsd_age_distribution"]
        self.max_cohort_residual: float = cfg["max_cohort_survival_residual"]
        self.fertility_range: tuple[float, float] = tuple(cfg["plausible_fertility_range"])  # type: ignore[assignment]
        self.migration_range: tuple[float, float] = tuple(cfg["plausible_migration_rate_range"])  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def compute_all_checks(
        self,
        results_df: pd.DataFrame,
        components_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run every structural realism check and return a combined diagnostics table.

        Parameters
        ----------
        results_df:
            Tidy projection results (columns matching ``ProjectionResultRecord``).
        components_df:
            Demographic component records (columns matching ``ComponentRecord``).

        Returns
        -------
        pd.DataFrame
            Rows compatible with ``DiagnosticRecord`` schema.
        """
        frames: list[pd.DataFrame] = []

        comp = self.component_realism(components_df)
        if not comp.empty:
            frames.append(comp)

        age = self.age_structure_realism(results_df)
        if not age.empty:
            frames.append(age)

        cohort = self.cohort_continuity(results_df)
        if not cohort.empty:
            frames.append(cohort)

        acct = self.accounting_checks(results_df, components_df)
        if not acct.empty:
            frames.append(acct)

        dist = self.distributional_realism(results_df)
        if not dist.empty:
            frames.append(dist)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 1. Component realism
    # ------------------------------------------------------------------

    def component_realism(self, components_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate projection error for births, deaths, and migration by horizon.

        Parameters
        ----------
        components_df:
            Must contain columns: ``run_id``, ``geography``, ``year``,
            ``horizon``, ``component``, ``projected_component_value``,
            ``actual_component_value``.

        Returns
        -------
        pd.DataFrame
            Diagnostics with MAPE and MSPE per component per horizon.
        """
        if components_df.empty:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        run_id = str(components_df["run_id"].iloc[0])

        for component in components_df["component"].unique():
            comp_data = components_df[components_df["component"] == component]
            for horizon_val in sorted(comp_data["horizon"].unique()):
                hslice = comp_data[comp_data["horizon"] == horizon_val]
                proj = hslice["projected_component_value"].values
                act = hslice["actual_component_value"].values

                mape_val = mape(proj, act)
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name=f"{component}_mape",
                    geography="all",
                    target=component,
                    value=mape_val,
                    horizon=int(horizon_val),
                ))

                mspe_val = mean_signed_percentage_error(proj, act)
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name=f"{component}_mspe",
                    geography="all",
                    target=component,
                    value=mspe_val,
                    horizon=int(horizon_val),
                ))

                mae_val = mae(proj, act)
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name=f"{component}_mae",
                    geography="all",
                    target=component,
                    value=mae_val,
                    horizon=int(horizon_val),
                ))

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 2. Age-structure realism
    # ------------------------------------------------------------------

    def age_structure_realism(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate age-distribution divergence and smoothness.

        Checks performed:
        - Jensen-Shannon and KL divergence of projected vs actual age
          distributions, per geography per year.
        - Five-year age-band accuracy (MAPE per age group).
        - Age-schedule smoothness (second-difference roughness measure).

        Parameters
        ----------
        results_df:
            Projection results with columns matching ``ProjectionResultRecord``.
            Must include rows where ``target == "population"`` and ``age_group``
            is a standard five-year group.

        Returns
        -------
        pd.DataFrame
            Diagnostics rows.
        """
        if results_df.empty:
            return pd.DataFrame()

        pop = results_df[
            (results_df["target"] == "population")
            & (results_df["age_group"].isin(_STANDARD_AGE_GROUPS))
        ].copy()

        if pop.empty:
            return pd.DataFrame()

        run_id = str(results_df["run_id"].iloc[0])
        records: list[dict[str, Any]] = []

        # Group by geography + year to get a full age distribution per snapshot
        for (geo, _year), group in pop.groupby(["geography", "year"]):
            # Ensure we have a reasonably complete age distribution
            if len(group) < 5:
                continue

            # Sort by standard age-group order
            order_map = {ag: i for i, ag in enumerate(_STANDARD_AGE_GROUPS)}
            group = group.copy()
            group["_order"] = group["age_group"].map(order_map)
            group = group.sort_values("_order")

            proj_dist = group["projected_value"].values.astype(float)
            act_dist = group["actual_value"].values.astype(float)

            horizon = int(group["horizon"].iloc[0]) if "horizon" in group.columns else None

            # JSD and KLD via shared helper
            records.extend(_compute_divergence_metrics(
                run_id=run_id,
                proj_dist=proj_dist,
                act_dist=act_dist,
                geography=str(geo),
                horizon=horizon,
                jsd_metric_name="age_jsd",
                kld_metric_name="age_kld",
                target="population",
                max_jsd=self.max_jsd,
            ))

            # Five-year age-band MAPE
            for _, row in group.iterrows():
                if row["actual_value"] != 0:
                    band_ape = abs(
                        (row["projected_value"] - row["actual_value"])
                        / row["actual_value"]
                    ) * 100
                    records.append(_realism_diagnostic(
                        run_id=run_id,
                        metric_name=f"age_band_ape_{row['age_group']}",
                        geography=str(geo),
                        target="population",
                        value=float(band_ape),
                        horizon=horizon,
                    ))

            # Age-schedule smoothness (second-difference roughness)
            if len(proj_dist) >= 3 and proj_dist.sum() > 0:
                normed = proj_dist / proj_dist.sum()
                second_diff = np.diff(normed, n=2)
                roughness = float(np.sqrt(np.mean(second_diff**2)))
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name="age_roughness",
                    geography=str(geo),
                    target="population",
                    value=roughness,
                    horizon=horizon,
                ))

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 3. Cohort continuity
    # ------------------------------------------------------------------

    def cohort_continuity(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Check whether cohorts track plausibly across time steps.

        For each five-year age group at time *T*, the population in the next
        age group at time *T+5* should be close to the original cohort minus
        expected mortality and plus/minus migration.  Large residuals signal
        implausible jumps.

        Parameters
        ----------
        results_df:
            Projection results with ``target == "population"`` rows that
            include ``age_group``, ``year``, ``geography``, ``sex``,
            ``projected_value``.

        Returns
        -------
        pd.DataFrame
            Diagnostics for cohort transition residuals.
        """
        if results_df.empty:
            return pd.DataFrame()

        pop = results_df[
            (results_df["target"] == "population")
            & (results_df["age_group"].isin(_STANDARD_AGE_GROUPS))
        ].copy()

        if pop.empty:
            return pd.DataFrame()

        run_id = str(results_df["run_id"].iloc[0])
        records: list[dict[str, Any]] = []

        # Build lookup: (geography, sex, year, age_group) -> projected_value
        lookup = build_lookup(
            pop,
            key_cols=["geography", "sex", "year", "age_group"],
            value_col="projected_value",
        )

        years = sorted(pop["year"].unique())

        for geo in pop["geography"].unique():
            for sex in pop["sex"].unique():
                for year in years:
                    next_year = year + 5
                    if next_year not in years:
                        continue
                    for age_group, successor in _COHORT_SUCCESSOR.items():
                        val_now = lookup.get((str(geo), str(sex), int(year), age_group))
                        val_next = lookup.get((str(geo), str(sex), int(next_year), successor))
                        if val_now is None or val_next is None or val_now == 0:
                            continue

                        # Cohort survival ratio: next / current
                        # For ages < 85, ratio > 1.0 is possible (net in-migration)
                        # but extreme values are suspicious
                        ratio = val_next / val_now
                        residual = abs(ratio - 1.0)

                        flag = "WARN" if residual > self.max_cohort_residual else "OK"
                        records.append(_realism_diagnostic(
                            run_id=run_id,
                            metric_name="cohort_survival_residual",
                            geography=str(geo),
                            target=f"{age_group}->{successor}",
                            value=residual,
                            horizon=int(next_year - years[0]),
                            notes=f"sex={sex}, ratio={ratio:.4f}, {flag}",
                        ))

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 4. Accounting and coherence checks
    # ------------------------------------------------------------------

    def accounting_checks(
        self,
        results_df: pd.DataFrame,
        components_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Verify demographic accounting identities and internal consistency.

        Checks:
        1. Population accounting: P(t+1) = P(t) + births - deaths + net_migration
        2. Age-sex detail sums to geography total
        3. County totals sum to state total
        4. Rates within plausible bounds

        Parameters
        ----------
        results_df:
            Projection results.
        components_df:
            Demographic component records.

        Returns
        -------
        pd.DataFrame
            Diagnostics rows.
        """
        records: list[dict[str, Any]] = []
        run_id = ""

        if not results_df.empty:
            run_id = str(results_df["run_id"].iloc[0])
        elif not components_df.empty:
            run_id = str(components_df["run_id"].iloc[0])
        else:
            return pd.DataFrame()

        # --- Check 1: Population accounting identity ---
        records.extend(self._check_accounting_identity(results_df, components_df, run_id))

        # --- Check 2: Age-sex totals sum to geography total ---
        records.extend(self._check_age_sex_totals(results_df, run_id))

        # --- Check 3: County totals sum to state total ---
        records.extend(self._check_county_state_totals(results_df, run_id))

        # --- Check 4: Plausible rate bounds ---
        records.extend(self._check_rate_bounds(results_df, components_df, run_id))

        return pd.DataFrame(records)

    def _check_accounting_identity(
        self,
        results_df: pd.DataFrame,
        components_df: pd.DataFrame,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """P(t+1) = P(t) + births - deaths + net_migration."""
        records: list[dict[str, Any]] = []
        if results_df.empty or components_df.empty:
            return records

        # Get total population per geography per year
        total_pop = results_df[
            (results_df["target"] == "population")
            & (results_df["age_group"] == "total")
            & (results_df["sex"] == "total")
        ].copy()

        if total_pop.empty:
            return records

        pop_lookup: dict[tuple[str, int], float] = {}
        for _, row in total_pop.iterrows():
            pop_lookup[(str(row["geography"]), int(row["year"]))] = float(row["projected_value"])

        # Get components per geography per year
        for geo in total_pop["geography"].unique():
            geo_comp = components_df[components_df["geography"] == geo]
            years = sorted(total_pop[total_pop["geography"] == geo]["year"].unique())

            for i in range(len(years) - 1):
                t, t1 = int(years[i]), int(years[i + 1])
                p_t = pop_lookup.get((str(geo), t))
                p_t1 = pop_lookup.get((str(geo), t1))
                if p_t is None or p_t1 is None:
                    continue

                # Sum components for the interval
                interval_comp = geo_comp[geo_comp["year"] == t1]
                births = interval_comp[
                    interval_comp["component"] == "births"
                ]["projected_component_value"].sum()
                deaths = interval_comp[
                    interval_comp["component"] == "deaths"
                ]["projected_component_value"].sum()
                net_mig = interval_comp[
                    interval_comp["component"] == "net_migration"
                ]["projected_component_value"].sum()

                expected = p_t + births - deaths + net_mig
                residual = p_t1 - expected
                rel_residual = abs(residual) / max(abs(p_t), 1.0)

                flag = "FAIL" if rel_residual > 0.001 else "OK"
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name="accounting_identity_residual",
                    geography=str(geo),
                    target="population",
                    value=rel_residual,
                    horizon=t1 - years[0],
                    notes=f"year={t}->{t1}, abs_residual={residual:.1f}, {flag}",
                ))

        return records

    def _check_age_sex_totals(
        self,
        results_df: pd.DataFrame,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Age-sex detail rows must sum to the geography total."""
        records: list[dict[str, Any]] = []
        if results_df.empty:
            return records

        pop = results_df[results_df["target"] == "population"]
        if pop.empty:
            return records

        for (geo, _year, sex), group in pop.groupby(["geography", "year", "sex"]):
            total_row = group[group["age_group"] == "total"]
            detail_rows = group[group["age_group"].isin(_STANDARD_AGE_GROUPS)]

            if total_row.empty or detail_rows.empty:
                continue

            reported_total = float(total_row["projected_value"].iloc[0])
            detail_sum = float(detail_rows["projected_value"].sum())

            if reported_total == 0:
                continue

            diff_pct = abs(detail_sum - reported_total) / abs(reported_total) * 100
            flag = "FAIL" if diff_pct > 0.1 else "OK"
            records.append(_realism_diagnostic(
                run_id=run_id,
                metric_name="age_sex_total_mismatch_pct",
                geography=str(geo),
                target="population",
                value=diff_pct,
                horizon=int(group["horizon"].iloc[0]) if "horizon" in group.columns else None,
                notes=f"sex={sex}, detail_sum={detail_sum:.0f}, "
                      f"total={reported_total:.0f}, {flag}",
            ))

        return records

    def _check_county_state_totals(
        self,
        results_df: pd.DataFrame,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """County totals should sum to state total."""
        records: list[dict[str, Any]] = []
        if results_df.empty:
            return records

        if "geography_type" not in results_df.columns:
            return records

        pop = results_df[
            (results_df["target"] == "population")
            & (results_df["age_group"] == "total")
            & (results_df["sex"] == "total")
        ]

        state_rows = pop[pop["geography_type"] == "state"]
        county_rows = pop[pop["geography_type"] == "county"]

        if state_rows.empty or county_rows.empty:
            return records

        for year in state_rows["year"].unique():
            state_val = float(
                state_rows[state_rows["year"] == year]["projected_value"].sum()
            )
            county_sum = float(
                county_rows[county_rows["year"] == year]["projected_value"].sum()
            )
            if state_val == 0:
                continue
            diff_pct = abs(county_sum - state_val) / abs(state_val) * 100
            flag = "FAIL" if diff_pct > 0.1 else "OK"
            records.append(_realism_diagnostic(
                run_id=run_id,
                metric_name="county_state_total_mismatch_pct",
                geography="state",
                target="population",
                value=diff_pct,
                horizon=int(state_rows[state_rows["year"] == year]["horizon"].iloc[0]),
                notes=f"year={year}, county_sum={county_sum:.0f}, "
                      f"state={state_val:.0f}, {flag}",
            ))

        return records

    def _check_rate_bounds(
        self,
        results_df: pd.DataFrame,
        components_df: pd.DataFrame,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Flag rates that fall outside plausible bounds."""
        records: list[dict[str, Any]] = []
        if results_df.empty or components_df.empty:
            return records

        # Get total population for rate denominators
        total_pop = results_df[
            (results_df["target"] == "population")
            & (results_df["age_group"] == "total")
            & (results_df["sex"] == "total")
        ]
        pop_lookup: dict[tuple[str, int], float] = {}
        for _, row in total_pop.iterrows():
            pop_lookup[(str(row["geography"]), int(row["year"]))] = float(
                row["projected_value"]
            )

        for _, row in components_df.iterrows():
            pop_val = pop_lookup.get((str(row["geography"]), int(row["year"])))
            if pop_val is None or pop_val == 0:
                continue

            rate = float(row["projected_component_value"]) / pop_val
            component = str(row["component"])

            out_of_bounds = False
            if component == "net_migration":
                lo, hi = self.migration_range
                out_of_bounds = rate < lo or rate > hi
            elif component == "births":
                lo, hi = self.fertility_range
                out_of_bounds = rate < lo or rate > hi

            if out_of_bounds:
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name=f"{component}_rate_out_of_bounds",
                    geography=str(row["geography"]),
                    target=component,
                    value=rate,
                    horizon=int(row["horizon"]),
                    notes=f"year={row['year']}, bounds=[{lo:.2f}, {hi:.2f}]",
                ))

        return records

    # ------------------------------------------------------------------
    # 5. Distributional realism
    # ------------------------------------------------------------------

    def distributional_realism(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate cross-county distributional properties.

        Compares the distribution of county-level population sizes and growth
        rates (projected vs actual) using variance ratio, skewness difference,
        and JSD.

        Parameters
        ----------
        results_df:
            Projection results.  Needs ``geography_type == "county"`` rows
            with ``target == "population"``, ``age_group == "total"``,
            ``sex == "total"``.

        Returns
        -------
        pd.DataFrame
            Diagnostics rows.
        """
        if results_df.empty:
            return pd.DataFrame()

        run_id = str(results_df["run_id"].iloc[0])
        records: list[dict[str, Any]] = []

        # Filter to county-level total population
        mask = (
            (results_df["target"] == "population")
            & (results_df["age_group"] == "total")
            & (results_df["sex"] == "total")
        )
        if "geography_type" in results_df.columns:
            mask = mask & (results_df["geography_type"] == "county")

        county_pop = results_df[mask].copy()

        if county_pop.empty:
            return pd.DataFrame()

        for year in sorted(county_pop["year"].unique()):
            yslice = county_pop[county_pop["year"] == year]
            if len(yslice) < 3:
                continue

            proj_sizes = yslice["projected_value"].values.astype(float)
            act_sizes = yslice["actual_value"].values.astype(float)
            horizon = int(yslice["horizon"].iloc[0])

            # Population size distribution JSD via shared helper
            # (only JSD is used here; KLD record is also emitted for consistency)
            records.extend(_compute_divergence_metrics(
                run_id=run_id,
                proj_dist=proj_sizes,
                act_dist=act_sizes,
                geography="all_counties",
                horizon=horizon,
                jsd_metric_name="county_size_dist_jsd",
                kld_metric_name="county_size_dist_kld",
                target="population",
            ))

            # Variance ratio of population sizes
            proj_var = float(np.var(proj_sizes))
            act_var = float(np.var(act_sizes))
            if act_var > 0:
                variance_ratio = proj_var / act_var
                records.append(_realism_diagnostic(
                    run_id=run_id,
                    metric_name="county_size_variance_ratio",
                    geography="all_counties",
                    target="population",
                    value=variance_ratio,
                    horizon=horizon,
                    notes="<1 = under-dispersed, >1 = over-dispersed",
                ))

            # Skewness comparison
            from scipy.stats import skew as scipy_skew  # noqa: PLC0415

            proj_skew = float(scipy_skew(proj_sizes))
            act_skew = float(scipy_skew(act_sizes))
            records.append(_realism_diagnostic(
                run_id=run_id,
                metric_name="county_size_skewness_diff",
                geography="all_counties",
                target="population",
                value=proj_skew - act_skew,
                horizon=horizon,
                notes=f"proj_skew={proj_skew:.3f}, act_skew={act_skew:.3f}",
            ))

            # Growth rate distribution (if base_value available)
            if "base_value" in yslice.columns:
                base = yslice["base_value"].values.astype(float)
                valid = base > 0
                if valid.sum() >= 3:
                    proj_growth = (proj_sizes[valid] - base[valid]) / base[valid]
                    act_growth = (act_sizes[valid] - base[valid]) / base[valid]

                    # Growth rate variance ratio
                    pg_var = float(np.var(proj_growth))
                    ag_var = float(np.var(act_growth))
                    if ag_var > 0:
                        records.append(_realism_diagnostic(
                            run_id=run_id,
                            metric_name="county_growth_variance_ratio",
                            geography="all_counties",
                            target="growth_rate",
                            value=pg_var / ag_var,
                            horizon=horizon,
                            notes="<1 = under-dispersed, >1 = over-dispersed",
                        ))

                    # Growth rate skewness difference
                    pg_skew = float(scipy_skew(proj_growth))
                    ag_skew = float(scipy_skew(act_growth))
                    records.append(_realism_diagnostic(
                        run_id=run_id,
                        metric_name="county_growth_skewness_diff",
                        geography="all_counties",
                        target="growth_rate",
                        value=pg_skew - ag_skew,
                        horizon=horizon,
                    ))

        return pd.DataFrame(records)
