#!/usr/bin/env python3
"""
Module 2.1.3: Rolling-Origin Backtesting
========================================

Implements rolling-origin backtesting for ND international migration with
benchmark models and interval calibration diagnostics.

Models evaluated:
- Naive random walk (last observation)
- Expanding mean
- Driver regression (ND on US migration, ex post driver)
- ARIMA(0,1,0) baseline

Outputs:
- results/backtesting_results.json

Usage:
    source .venv/bin/activate
    python sdc_2024_replication/scripts/statistical_analysis/module_2_1_3_backtesting.py
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"

# Ensure output directory exists
RESULTS_DIR.mkdir(exist_ok=True)

LOGGER = logging.getLogger(__name__)


class ModuleResult:
    """Standard result container for all modules."""

    def __init__(self, module_id: str, analysis_name: str):
        self.module_id = module_id
        self.analysis_name = analysis_name
        self.input_files: list[str] = []
        self.parameters: dict = {}
        self.results: dict = {}
        self.diagnostics: dict = {}
        self.decisions: list[dict] = []
        self.warnings: list[str] = []
        self.next_steps: list[str] = []

    def add_decision(
        self,
        decision_id: str,
        category: str,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
        evidence: str | None = None,
        reversible: bool = True,
    ):
        """Add a documented decision to the log."""
        self.decisions.append(
            {
                "decision_id": decision_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "category": category,
                "decision": decision,
                "rationale": rationale,
                "alternatives_considered": alternatives or [],
                "evidence": evidence,
                "reversible": reversible,
            }
        )

    def to_dict(self) -> dict:
        return {
            "module": self.module_id,
            "analysis": self.analysis_name,
            "generated": datetime.now(UTC).isoformat(),
            "input_files": self.input_files,
            "parameters": self.parameters,
            "results": self.results,
            "diagnostics": self.diagnostics,
            "decisions": self.decisions,
            "warnings": self.warnings,
            "next_steps": self.next_steps,
        }

    def save(self, filename: str) -> Path:
        """Save results to JSON file."""
        output_path = RESULTS_DIR / filename
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        LOGGER.info("Results saved to: %s", output_path)
        return output_path


def load_data(filename: str) -> pd.DataFrame:
    """Load data file from analysis directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    raise ValueError(f"Unsupported file type: {filepath.suffix}")


def _intervals_from_sigma(
    y_hat: float,
    sigma: float,
    levels: tuple[float, ...],
) -> dict[float, tuple[float | None, float | None]]:
    """Compute symmetric prediction intervals from a residual sigma."""
    intervals: dict[float, tuple[float | None, float | None]] = {}
    if sigma is None or np.isnan(sigma):
        for level in levels:
            intervals[level] = (None, None)
        return intervals

    dist = NormalDist()
    for level in levels:
        z = dist.inv_cdf(0.5 + level / 2.0)
        half_width = z * sigma
        intervals[level] = (float(y_hat - half_width), float(y_hat + half_width))
    return intervals


def _safe_mape(actual: float, forecast: float) -> float | None:
    """Compute MAPE with guard for zero actuals."""
    if actual == 0:
        return None
    return float(abs((actual - forecast) / actual))


def _naive_forecast(train: pd.Series) -> tuple[float, float | None]:
    """Random-walk forecast and residual sigma."""
    y_hat = float(train.iloc[-1])
    diffs = train.diff().dropna()
    sigma = float(diffs.std(ddof=1)) if len(diffs) > 1 else None
    return y_hat, sigma


def _mean_forecast(train: pd.Series) -> tuple[float, float | None]:
    """Expanding-mean forecast and residual sigma."""
    y_hat = float(train.mean())
    residuals = train - y_hat
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else None
    return y_hat, sigma


def _driver_ols_forecast(
    train: pd.DataFrame,
    target_col: str,
    driver_col: str,
    driver_next: float,
) -> tuple[float, float | None]:
    """OLS forecast using a national driver and residual sigma."""
    y = train[target_col].to_numpy()
    x = train[driver_col].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = float(beta[0] + beta[1] * driver_next)
        residuals = y - X @ beta
        sigma = float(residuals.std(ddof=1)) if len(residuals) > 2 else None
        return y_hat, sigma
    except np.linalg.LinAlgError:
        return float(train[target_col].mean()), None


def _arima_forecast(train: pd.Series) -> tuple[float, dict[float, tuple[float, float]]]:
    """ARIMA(0,1,0) forecast with prediction intervals."""
    model = ARIMA(train, order=(0, 1, 0), trend="n")
    fit = model.fit()
    forecast_res = fit.get_forecast(steps=1)
    y_hat = float(forecast_res.predicted_mean.iloc[0])

    intervals: dict[float, tuple[float, float]] = {}
    for level in (0.8, 0.95):
        conf = forecast_res.conf_int(alpha=1.0 - level)
        lower = float(conf.iloc[0, 0])
        upper = float(conf.iloc[0, 1])
        intervals[level] = (lower, upper)
    return y_hat, intervals


def rolling_origin_backtest(
    df: pd.DataFrame,
    target_col: str,
    driver_col: str | None,
    start_train_end: int,
    horizons: tuple[int, ...],
    interval_levels: tuple[float, ...],
) -> list[dict]:
    """Run rolling-origin backtesting and return forecast log entries."""
    years = df["year"].to_numpy()
    max_year = int(years.max())
    forecast_log: list[dict] = []

    for train_end in range(start_train_end, max_year):
        train = df[df["year"] <= train_end]
        for horizon in horizons:
            forecast_year = train_end + horizon
            if forecast_year > max_year:
                continue

            y_true = float(df.loc[df["year"] == forecast_year, target_col].iloc[0])

            y_hat_naive, sigma_naive = _naive_forecast(train[target_col])
            naive_intervals = _intervals_from_sigma(
                y_hat_naive, sigma_naive, interval_levels
            )

            y_hat_mean, sigma_mean = _mean_forecast(train[target_col])
            mean_intervals = _intervals_from_sigma(
                y_hat_mean, sigma_mean, interval_levels
            )

            driver_intervals: dict[float, tuple[float | None, float | None]] = {
                level: (None, None) for level in interval_levels
            }
            y_hat_driver = None
            if driver_col is not None:
                driver_next = float(
                    df.loc[df["year"] == forecast_year, driver_col].iloc[0]
                )
                y_hat_driver, sigma_driver = _driver_ols_forecast(
                    train, target_col, driver_col, driver_next
                )
                driver_intervals = _intervals_from_sigma(
                    y_hat_driver, sigma_driver, interval_levels
                )

            try:
                y_hat_arima, arima_intervals = _arima_forecast(train[target_col])
            except Exception:
                y_hat_arima = y_hat_naive
                arima_intervals = {level: (None, None) for level in interval_levels}

            model_specs = [
                ("naive_rw", y_hat_naive, naive_intervals),
                ("expanding_mean", y_hat_mean, mean_intervals),
                ("driver_ols", y_hat_driver, driver_intervals),
                ("arima_010", y_hat_arima, arima_intervals),
            ]

            for model_name, y_hat, intervals in model_specs:
                if y_hat is None:
                    continue
                error = y_true - y_hat
                abs_error = abs(error)
                mape = _safe_mape(y_true, y_hat)

                lower_80, upper_80 = intervals.get(0.8, (None, None))
                lower_95, upper_95 = intervals.get(0.95, (None, None))

                covered_80 = (
                    None if lower_80 is None else float(lower_80 <= y_true <= upper_80)
                )
                covered_95 = (
                    None if lower_95 is None else float(lower_95 <= y_true <= upper_95)
                )
                width_80 = None if lower_80 is None else float(upper_80 - lower_80)
                width_95 = None if lower_95 is None else float(upper_95 - lower_95)

                forecast_log.append(
                    {
                        "origin_year": int(train_end),
                        "forecast_year": int(forecast_year),
                        "horizon": int(horizon),
                        "model": model_name,
                        "actual": y_true,
                        "forecast": float(y_hat),
                        "error": float(error),
                        "abs_error": float(abs_error),
                        "mape": mape,
                        "lower_80": lower_80,
                        "upper_80": upper_80,
                        "lower_95": lower_95,
                        "upper_95": upper_95,
                        "covered_80": covered_80,
                        "covered_95": covered_95,
                        "interval_width_80": width_80,
                        "interval_width_95": width_95,
                    }
                )

    return forecast_log


def summarize_forecasts(forecast_log: list[dict]) -> list[dict]:
    """Summarize forecast accuracy and interval calibration by model."""
    df = pd.DataFrame(forecast_log)
    if df.empty:
        return []

    summaries = []
    group_cols = ["model", "horizon"]

    for (model, horizon), group in df.groupby(group_cols):
        summaries.append(
            {
                "model": model,
                "horizon": int(horizon),
                "n_forecasts": int(len(group)),
                "mae": float(group["abs_error"].mean()),
                "rmse": float(np.sqrt((group["error"] ** 2).mean())),
                "mape": float(group["mape"].mean(skipna=True)),
                "coverage_80": float(group["covered_80"].mean(skipna=True)),
                "coverage_95": float(group["covered_95"].mean(skipna=True)),
                "avg_width_80": float(group["interval_width_80"].mean(skipna=True)),
                "avg_width_95": float(group["interval_width_95"].mean(skipna=True)),
            }
        )

    return summaries


def run_analysis() -> ModuleResult:
    """Run rolling-origin backtesting."""
    result = ModuleResult(module_id="2.1.3", analysis_name="backtesting")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_data("nd_migration_summary.csv")
    result.input_files.append("nd_migration_summary.csv")

    target_col = "nd_intl_migration"
    driver_col = "us_intl_migration"

    result.parameters = {
        "target_col": target_col,
        "driver_col": driver_col,
        "start_train_end": 2016,
        "horizons": [1],
        "interval_levels": [0.8, 0.95],
    }

    result.add_decision(
        decision_id="D001",
        category="validation",
        decision="Rolling-origin evaluation with expanding window",
        rationale="Maintains temporal ordering and avoids look-ahead bias.",
        alternatives=["Leave-one-out", "Random k-fold"],
        reversible=True,
    )
    result.add_decision(
        decision_id="D002",
        category="drivers",
        decision="Use US migration driver as ex post predictor",
        rationale="Provides a benchmark for association strength when the national series is observed.",
        alternatives=["Exclude national driver", "Lagged driver only"],
        reversible=True,
    )

    result.warnings.append(
        "Backtesting uses 8 one-step forecasts (2017-2024); results are indicative."
    )
    result.warnings.append(
        "Driver regression uses contemporaneous US migration (ex post)."
    )

    forecast_log = rolling_origin_backtest(
        df=df,
        target_col=target_col,
        driver_col=driver_col,
        start_train_end=2016,
        horizons=(1,),
        interval_levels=(0.8, 0.95),
    )

    summary = summarize_forecasts(forecast_log)

    result.results = {
        "forecast_log": forecast_log,
        "summary": summary,
    }
    result.diagnostics = {
        "n_forecasts": len(forecast_log),
        "n_models": len({entry["model"] for entry in forecast_log}),
    }

    result.next_steps = [
        "Compare backtest diagnostics to scenario forecast uncertainty bands.",
        "Consider sensitivity checks with and without 2020 as an intervention.",
    ]

    return result


def main() -> int:
    """Entry point."""
    try:
        result = run_analysis()
        result.save("backtesting_results.json")
        return 0
    except Exception as exc:
        LOGGER.error("Backtesting failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
