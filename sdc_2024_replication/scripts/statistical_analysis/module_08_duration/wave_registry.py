"""Wave registry and hazard-based duration forecasting utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(slots=True)
class WaveRecord:
    """Single immigration-wave record for forecasting."""

    wave_id: str
    state: str
    origin: str
    start_year: int
    last_observed_year: int
    age: int
    status: str
    baseline: float
    observed_years: list[int]
    observed_arrivals: list[float]
    intensity_ratio: float
    peak_arrivals: float
    covariates: dict[str, float] = field(default_factory=dict)
    survival: dict[str, float] = field(default_factory=dict)
    shape_params: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class LifecycleStats:
    """Aggregate lifecycle parameters for wave shapes."""

    mean_time_to_peak: float
    mean_initiation_share: float
    mean_decline_share: float
    mean_total_duration: float

    @classmethod
    def from_hazard_model(cls, hazard_model: dict) -> "LifecycleStats":
        stats = hazard_model.get("lifecycle_analysis", {}).get(
            "aggregate_statistics", {}
        )
        return cls(
            mean_time_to_peak=float(stats.get("mean_time_to_peak", 2.0)),
            mean_initiation_share=float(stats.get("mean_initiation_share", 0.28)),
            mean_decline_share=float(stats.get("mean_decline_share", 0.35)),
            mean_total_duration=float(stats.get("mean_total_duration", 3.5)),
        )


@dataclass(slots=True)
class DurationModelBundle:
    """Container for hazard-based duration forecasting inputs."""

    predictor: "ConditionalDurationPredictor"
    lifecycle_stats: LifecycleStats


class ConditionalDurationPredictor:
    """Compute conditional duration distributions from Cox + baseline survival."""

    def __init__(
        self,
        beta_hat: dict[str, float],
        baseline_survival: dict[int, float],
        beta_vcov: Optional[np.ndarray] = None,
        peak_arrivals_scale: float = 1000.0,
    ) -> None:
        self.beta_hat = dict(beta_hat)
        self.beta_vcov = beta_vcov
        self.peak_arrivals_scale = peak_arrivals_scale
        self._beta_names = list(self.beta_hat.keys())
        self._beta_vector = np.array([self.beta_hat[name] for name in self._beta_names])
        self._baseline_survival = self._prepare_baseline(baseline_survival)
        self._max_tau = max(self._baseline_survival)
        self._tail_hazard = self._estimate_tail_hazard()

    @staticmethod
    def _prepare_baseline(baseline_survival: dict[int, float]) -> dict[int, float]:
        if not baseline_survival:
            raise ValueError("Baseline survival table is empty.")
        baseline = {int(k): float(v) for k, v in baseline_survival.items()}
        if 0 not in baseline:
            baseline[0] = 1.0
        max_tau = max(baseline)
        for tau in range(1, max_tau + 1):
            if tau not in baseline:
                baseline[tau] = baseline.get(tau - 1, 1.0)
        return baseline

    def _estimate_tail_hazard(self) -> float:
        max_tau = self._max_tau
        prev_tau = max_tau - 1 if max_tau > 0 else max_tau
        if prev_tau not in self._baseline_survival:
            return 0.0
        s_prev = self._baseline_survival[prev_tau]
        s_last = self._baseline_survival[max_tau]
        if s_prev <= 0:
            return 0.0
        hazard = 1.0 - (s_last / s_prev)
        return min(max(hazard, 0.0), 0.95)

    def _baseline_survival_at(self, tau: int) -> float:
        if tau <= 0:
            return 1.0
        if tau in self._baseline_survival:
            return self._baseline_survival[tau]
        if tau <= self._max_tau:
            return self._baseline_survival[self._max_tau]
        if self._baseline_survival[self._max_tau] <= 0:
            return 0.0
        decay = (1.0 - self._tail_hazard) ** (tau - self._max_tau)
        return self._baseline_survival[self._max_tau] * decay

    def _linear_predictor(
        self, covariates: dict[str, float], beta: dict[str, float]
    ) -> float:
        lp = 0.0
        for name, coef in beta.items():
            value = covariates.get(name, 0.0)
            if name == "peak_arrivals" and self.peak_arrivals_scale:
                value = value / self.peak_arrivals_scale
            if not math.isfinite(value):
                value = 0.0
            lp += coef * value
        return lp

    def sample_beta(self, rng: np.random.Generator) -> dict[str, float]:
        if self.beta_vcov is None:
            return dict(self.beta_hat)
        draw = rng.multivariate_normal(self._beta_vector, self.beta_vcov)
        return {
            name: float(val) for name, val in zip(self._beta_names, draw, strict=False)
        }

    def survival(
        self,
        tau: int,
        covariates: dict[str, float],
        beta: Optional[dict[str, float]] = None,
    ) -> float:
        if tau <= 0:
            return 1.0
        beta_use = beta or self.beta_hat
        lp = self._linear_predictor(covariates, beta_use)
        s0 = self._baseline_survival_at(tau)
        if s0 <= 0:
            return 0.0
        return float(s0 ** math.exp(lp))

    def conditional_survival(
        self,
        age: int,
        horizon: int,
        covariates: dict[str, float],
        beta: Optional[dict[str, float]] = None,
    ) -> list[float]:
        age = max(1, age)
        s_age = self.survival(age, covariates, beta=beta)
        if s_age <= 0:
            return [0.0 for _ in range(horizon + 1)]
        return [
            self.survival(age + k, covariates, beta=beta) / s_age
            for k in range(horizon + 1)
        ]

    def expected_remaining_duration(
        self,
        age: int,
        horizon: int,
        covariates: dict[str, float],
        beta: Optional[dict[str, float]] = None,
    ) -> float:
        cond = self.conditional_survival(age, horizon, covariates, beta=beta)
        return float(sum(cond[1:]))

    def sample_total_duration(
        self,
        age: int,
        covariates: dict[str, float],
        rng: np.random.Generator,
        horizon: int,
        beta: Optional[dict[str, float]] = None,
    ) -> int:
        age = max(1, age)
        cond = self.conditional_survival(age, horizon, covariates, beta=beta)
        pmf = [cond[k] - cond[k + 1] for k in range(len(cond) - 1)]
        pmf.append(max(0.0, cond[-1]))
        total = sum(pmf)
        if total <= 0:
            return age
        pmf = [p / total for p in pmf]
        remaining = int(rng.choice(len(pmf), p=pmf))
        return age + remaining

    @classmethod
    def from_hazard_results(
        cls,
        hazard_model: dict,
        baseline_survival: dict[int, float],
        peak_arrivals_scale: float = 1000.0,
    ) -> "ConditionalDurationPredictor":
        coef_table = hazard_model.get("model", {}).get("coefficient_table", {})
        if not coef_table:
            coef_table = hazard_model.get("coefficient_table", {})
        beta_hat = {name: stats["coefficient"] for name, stats in coef_table.items()}
        std_errors = [coef_table[name].get("std_error") for name in beta_hat]
        if any(se is None for se in std_errors):
            beta_vcov = None
        else:
            beta_vcov = np.diag([float(se) ** 2 for se in std_errors])
        return cls(
            beta_hat=beta_hat,
            baseline_survival=baseline_survival,
            beta_vcov=beta_vcov,
            peak_arrivals_scale=peak_arrivals_scale,
        )


class WaveRegistry:
    """Stateful registry of detected waves."""

    def __init__(self, waves: Optional[dict[str, WaveRecord]] = None) -> None:
        self.waves: dict[str, WaveRecord] = waves or {}

    def add_wave(self, wave: WaveRecord) -> None:
        self.waves[wave.wave_id] = wave

    def active_waves(self) -> list[WaveRecord]:
        return [wave for wave in self.waves.values() if wave.status == "active"]

    @classmethod
    def from_series(
        cls,
        *,
        state: str,
        origin: str,
        years: Iterable[int],
        arrivals: Iterable[float],
        baseline: float,
        threshold_pct: float = 50.0,
        min_wave_years: int = 2,
        covariate_overrides: Optional[dict[str, float]] = None,
    ) -> "WaveRegistry":
        years_list = list(years)
        arrivals_list = list(arrivals)
        if len(years_list) != len(arrivals_list):
            raise ValueError("Years and arrivals must be the same length.")
        if not years_list:
            return cls()

        threshold = (1.0 + threshold_pct / 100.0) * baseline
        exceed = [val >= threshold for val in arrivals_list]

        runs = []
        start_idx = None
        for idx, flag in enumerate(exceed):
            if flag and start_idx is None:
                start_idx = idx
            elif not flag and start_idx is not None:
                runs.append((start_idx, idx - 1))
                start_idx = None
        if start_idx is not None:
            runs.append((start_idx, len(exceed) - 1))

        registry = cls()
        last_index = len(years_list) - 1
        for run_idx, (start, end) in enumerate(runs, start=1):
            run_years = years_list[start : end + 1]
            run_arrivals = arrivals_list[start : end + 1]
            duration = len(run_years)
            status = "ended"
            if end == last_index:
                status = "active" if duration >= min_wave_years else "candidate"

            peak_arrivals = max(run_arrivals) if run_arrivals else 0.0
            intensity_ratio = peak_arrivals / baseline if baseline > 0 else 0.0
            covariates = {
                "log_intensity": math.log(intensity_ratio)
                if intensity_ratio > 0
                else 0.0,
                "high_intensity": 1.0 if intensity_ratio > 5 else 0.0,
                "early_wave": 1.0 if run_years[0] <= 2010 else 0.0,
                "peak_arrivals": float(peak_arrivals),
            }
            if covariate_overrides:
                covariates.update(covariate_overrides)

            wave = WaveRecord(
                wave_id=f"{state}_{origin}_wave_{run_idx}",
                state=state,
                origin=origin,
                start_year=run_years[0],
                last_observed_year=run_years[-1],
                age=duration,
                status=status,
                baseline=float(baseline),
                observed_years=run_years,
                observed_arrivals=run_arrivals,
                intensity_ratio=float(intensity_ratio),
                peak_arrivals=float(peak_arrivals),
                covariates=covariates,
            )
            registry.add_wave(wave)

        return registry

    def annotate_survival(
        self, predictor: ConditionalDurationPredictor, horizon: int
    ) -> None:
        for wave in self.waves.values():
            cond = predictor.conditional_survival(wave.age, horizon, wave.covariates)
            wave.survival = {
                "conditional_survival": cond,
                "expected_remaining": predictor.expected_remaining_duration(
                    wave.age, horizon, wave.covariates
                ),
                "one_year_end_prob": 1.0 - cond[1] if len(cond) > 1 else None,
            }


def estimate_peak_age(duration: int, lifecycle_stats: LifecycleStats) -> int:
    if duration <= 0:
        return 1
    if lifecycle_stats.mean_total_duration > 0:
        peak_share = (
            lifecycle_stats.mean_time_to_peak / lifecycle_stats.mean_total_duration
        )
    else:
        peak_share = 0.5
    peak_age = int(round(peak_share * duration))
    return max(1, min(duration, peak_age))


def wave_shape(
    age: int,
    duration: int,
    peak_age: int,
    gamma_up: float = 1.0,
    gamma_down: float = 1.0,
) -> float:
    if duration <= 0:
        return 0.0
    if age <= peak_age:
        return (age / max(peak_age, 1)) ** gamma_up
    return ((duration - age + 1) / max(duration - peak_age + 1, 1)) ** gamma_down


def simulate_wave_contributions(
    wave: WaveRecord,
    predictor: ConditionalDurationPredictor,
    lifecycle_stats: LifecycleStats,
    horizon: int,
    rng: np.random.Generator,
    gamma_up: float = 1.0,
    gamma_down: float = 1.0,
) -> np.ndarray:
    beta_draw = predictor.sample_beta(rng)
    total_duration = predictor.sample_total_duration(
        wave.age, wave.covariates, rng, horizon=horizon, beta=beta_draw
    )
    peak_age = estimate_peak_age(total_duration, lifecycle_stats)
    peak_excess = max(0.0, wave.peak_arrivals - wave.baseline)

    contributions = np.zeros(horizon)
    for k in range(1, horizon + 1):
        wave_age = wave.age + k
        if wave_age <= total_duration:
            contributions[k - 1] = peak_excess * wave_shape(
                wave_age,
                total_duration,
                peak_age,
                gamma_up=gamma_up,
                gamma_down=gamma_down,
            )
    return contributions


def load_duration_model_bundle(
    results_dir: Path, *, tag: str | None = None
) -> DurationModelBundle:
    """Load Module 8 duration + hazard outputs, optionally from a tagged run."""

    def _tagged(filename: str) -> str:
        if not tag:
            return filename
        path = Path(filename)
        return f"{path.stem}__{tag}{path.suffix}"

    hazard_path = results_dir / _tagged("module_8_hazard_model.json")
    duration_path = results_dir / _tagged("module_8_duration_analysis.json")
    with hazard_path.open() as fp:
        hazard_model = json.load(fp)
    with duration_path.open() as fp:
        duration_model = json.load(fp)

    life_table = (
        duration_model.get("results", {}).get("kaplan_meier", {}).get("life_table", [])
    )
    baseline_survival = {
        int(row["time_years"]): float(row["survival_probability"])
        for row in life_table
        if row.get("survival_probability") is not None
    }
    if not baseline_survival:
        raise ValueError("Kaplan-Meier life table missing survival probabilities.")
    predictor = ConditionalDurationPredictor.from_hazard_results(
        hazard_model, baseline_survival
    )
    lifecycle_stats = LifecycleStats.from_hazard_model(hazard_model)
    return DurationModelBundle(predictor=predictor, lifecycle_stats=lifecycle_stats)
