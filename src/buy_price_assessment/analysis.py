"""0050 最佳買點研究的可重跑分析流程。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from buy_price_assessment.evaluation import (
    moving_block_bootstrap_ci,
    oracle_feature_profile,
    random_strategy_distribution,
    select_fixed_day,
    select_last_day,
    select_rsi_rule,
    strategy_metrics,
    terminal_wealth_proxy,
)
from buy_price_assessment.features import FEATURE_GROUPS, build_features
from buy_price_assessment.labels import add_monthly_labels, monthly_oracle_table
from buy_price_assessment.modeling import select_monthly_purchases
from buy_price_assessment.reporting import generate_figures, render_report
from buy_price_assessment.walk_forward import WalkForwardConfig, walk_forward_predictions

MODEL_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "technical_calendar": FEATURE_GROUPS["technical"] + FEATURE_GROUPS["calendar"],
    "technical_valuation_calendar": FEATURE_GROUPS["technical"]
    + FEATURE_GROUPS["valuation"]
    + FEATURE_GROUPS["calendar"],
    "all": tuple(column for group in FEATURE_GROUPS.values() for column in group),
}
MODEL_CACHE_VERSION = "2"


def last_complete_month(as_of: date) -> str:
    """研究當下只標記已結束月份。"""

    return str(pd.Period(as_of, freq="M") - 1)


def add_action_adjusted_open_gap(daily: pd.DataFrame) -> pd.DataFrame:
    """計算排除當日配息與分割機械效果的 opening gap。"""

    required = {"date", "open", "close", "split_ratio", "cash_dividend"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"opening gap 缺少欄位：{sorted(missing)}")
    result = daily.copy()
    previous_reference = result["close"].shift(1) / result["split_ratio"] - result["cash_dividend"]
    result["open_gap_vs_action_ref"] = result["open"] / previous_reference - 1.0
    return result


def _prepare_frames(
    daily: pd.DataFrame, complete_through: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(daily)
    labeled = add_monthly_labels(features, complete_through=complete_through)
    labeled = labeled.loc[labeled["month"] >= "2003-07"].reset_index(drop=True)
    oracle = monthly_oracle_table(labeled)
    return features, labeled, oracle


def _prediction_signature(
    labeled: pd.DataFrame,
    columns: tuple[str, ...],
    config: WalkForwardConfig,
) -> str:
    selected = labeled.loc[:, ["date", *columns]]
    values_hash = pd.util.hash_pandas_object(selected, index=False).to_numpy(dtype=np.uint64)
    settings = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(MODEL_CACHE_VERSION.encode())
    digest.update("\0".join(columns).encode())
    digest.update(settings.encode())
    digest.update(values_hash.tobytes())
    return digest.hexdigest()


def _cached_prediction_is_valid(
    path: Path,
    expected_dates: pd.Series,
    feature_set: str,
    expected_signature: str,
) -> bool:
    if not path.exists():
        return False
    try:
        cached = pd.read_csv(
            path,
            usecols=["date", "feature_set", "analysis_signature"],
        )
    except (OSError, ValueError):
        return False
    cached_dates = pd.to_datetime(cached["date"], errors="coerce")
    return bool(
        not cached_dates.isna().any()
        and cached_dates.tolist() == expected_dates.tolist()
        and cached["feature_set"].eq(feature_set).all()
        and cached["analysis_signature"].eq(expected_signature).all()
    )


def _load_or_run_prediction(
    labeled: pd.DataFrame,
    feature_set: str,
    columns: tuple[str, ...],
    processed_dir: Path,
    config: WalkForwardConfig,
    reuse_existing: bool,
) -> pd.DataFrame:
    path = processed_dir / f"walk_forward_{feature_set}.csv"
    ordered_months = sorted(labeled["month"].unique())
    first_test_month = ordered_months[config.initial_months]
    expected_dates = labeled.loc[labeled["month"] >= first_test_month, "date"].reset_index(
        drop=True
    )
    signature = _prediction_signature(labeled, columns, config)
    if reuse_existing and _cached_prediction_is_valid(
        path,
        expected_dates,
        feature_set,
        signature,
    ):
        return pd.read_csv(path, parse_dates=["date"])
    prediction = walk_forward_predictions(labeled, feature_columns=columns, config=config)
    prediction["feature_set"] = feature_set
    prediction["analysis_signature"] = signature
    prediction.to_csv(path, index=False, encoding="utf-8-sig")
    return prediction


def _model_outputs(
    labeled: pd.DataFrame,
    processed_dir: Path,
    *,
    reuse_existing: bool,
) -> dict[str, pd.DataFrame]:
    config = WalkForwardConfig()
    return {
        name: _load_or_run_prediction(
            labeled,
            name,
            columns,
            processed_dir,
            config,
            reuse_existing,
        )
        for name, columns in MODEL_FEATURE_SETS.items()
    }


def _baseline_outputs(oos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {f"fixed_day_{day}": select_fixed_day(oos, trading_day=day) for day in (1, 5, 10, 15)}
    outputs["last_day"] = select_last_day(oos)
    outputs["rsi30_or_last"] = select_rsi_rule(oos)
    return outputs


def _metric_with_wealth(purchases: pd.DataFrame, terminal_close: float) -> dict[str, int | float]:
    metrics = strategy_metrics(purchases)
    metrics["terminal_wealth_proxy"] = terminal_wealth_proxy(
        purchases, terminal_adjusted_close=terminal_close
    )
    return metrics


def _evaluate_models(
    predictions: Mapping[str, pd.DataFrame],
    terminal_close: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int | float]]]:
    purchases: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, int | float]] = {}
    for name, prediction in predictions.items():
        selected = select_monthly_purchases(prediction, probability_threshold=0.5)
        selected["strategy"] = name
        purchases[name] = selected
        item = _metric_with_wealth(selected, terminal_close)
        item["brier"] = float(
            brier_score_loss(
                prediction["near_optimal"].astype(bool), prediction["near_probability"]
            )
        )
        item["average_precision"] = float(
            average_precision_score(
                prediction["near_optimal"].astype(bool), prediction["near_probability"]
            )
        )
        metrics[name] = item
    return purchases, metrics


def _paired_improvement(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    block_length: int = 12,
) -> tuple[float, list[float]]:
    paired = candidate.loc[:, ["month", "regret"]].merge(
        baseline.loc[:, ["month", "regret"]],
        on="month",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    improvement = paired["regret_baseline"] - paired["regret_candidate"]
    lower, upper = moving_block_bootstrap_ci(improvement, block_length=block_length)
    return float(improvement.mean() * 10_000), [float(lower * 10_000), float(upper * 10_000)]


def _oracle_distribution(oracle: pd.DataFrame) -> dict[str, Any]:
    day = oracle["oracle_trading_day"]
    return {
        "months": len(oracle),
        "mean_day": float(day.mean()),
        "median_day": float(day.median()),
        "mode_day": int(day.mode().iloc[0]),
        "first_day_rate": float((day == 1).mean()),
        "first_5_rate": float((day <= 5).mean()),
        "first_10_rate": float((day <= 10).mean()),
    }


def _oracle_ranges(labeled: pd.DataFrame, daily: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    gaps = add_action_adjusted_open_gap(daily).loc[:, ["date", "open_gap_vs_action_ref"]]
    oracle = labeled.loc[labeled["is_oracle"].astype(bool)].merge(
        gaps, on="date", validate="one_to_one"
    )
    columns = [
        "open_gap_vs_action_ref",
        "ret_1",
        "ret_5",
        "ret_20",
        "ma_gap_20",
        "ma_gap_60",
        "rsi_14",
        "bollinger_z_20",
        "drawdown_60",
        "dividend_yield_ttm",
        "premium_discount",
        "margin_change_5",
        "institutional_net_ratio",
        "volume_z_20",
    ]
    ranges: dict[str, dict[str, float | int]] = {}
    for column in columns:
        values = oracle[column].dropna().astype(float)
        ranges[column] = {
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "mean": float(values.mean()),
            "n": len(values),
        }
    return ranges


def _holdout_result(
    labeled: pd.DataFrame,
    model_purchases: pd.DataFrame,
) -> dict[str, Any]:
    development = labeled.loc[(labeled["month"] >= "2008-07") & (labeled["month"] < "2023-07")]
    holdout = labeled.loc[(labeled["month"] >= "2023-07") & (labeled["month"] <= "2026-06")]
    candidates = {
        day: select_fixed_day(development, trading_day=day)["regret"].mean()
        for day in (1, 5, 10, 15)
    }
    selected_day = min(candidates, key=lambda day: candidates[day])
    fixed = select_fixed_day(holdout, trading_day=selected_day)
    model = model_purchases.loc[model_purchases["month"] >= "2023-07"]
    improvement, ci = _paired_improvement(model, fixed, block_length=6)
    fixed_metrics = strategy_metrics(fixed)
    model_metrics = strategy_metrics(model)
    return {
        "months": len(fixed),
        "selected_day": int(selected_day),
        "day1_mean_regret": float(fixed_metrics["mean_regret"]),
        "day1_within_rate": float(fixed_metrics["within_0_5pct_rate"]),
        "model_mean_regret": float(model_metrics["mean_regret"]),
        "model_improvement_bps": improvement,
        "model_ci95_bps": ci,
    }


def _data_quality(daily: pd.DataFrame) -> dict[str, Any]:
    issuer_difference = (daily["issuer_market_price"] - daily["close"]).dropna().abs()
    official_sources = sorted(daily["official_close_source"].dropna().astype(str).unique())
    return {
        "rows": len(daily),
        "start": str(daily["date"].min().date()),
        "end": str(daily["date"].max().date()),
        "columns": len(daily.columns),
        "official_max_difference": float(daily["close_difference"].abs().max()),
        "official_missing": int(daily["official_close"].isna().sum()),
        "official_source": ",".join(official_sources),
        "issuer_max_difference": float(issuer_difference.max()),
        "issuer_market_price_missing": int(daily["issuer_market_price"].isna().sum()),
        "nav_missing": int(daily["nav"].isna().sum()),
        "split_events": int((daily["split_ratio"] != 1.0).sum()),
        "dividend_events": int((daily["cash_dividend"] > 0.0).sum()),
    }


def _build_results(
    daily: pd.DataFrame,
    labeled: pd.DataFrame,
    oracle: pd.DataFrame,
    purchases: Mapping[str, pd.DataFrame],
    metrics: dict[str, dict[str, int | float]],
    baselines: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    day1 = baselines["fixed_day_1"]
    model = purchases["all"]
    improvement, model_ci = _paired_improvement(model, day1)
    day_comparison, day_ci = _paired_improvement(day1, baselines["fixed_day_5"])
    return {
        "data_quality": _data_quality(daily),
        "oracle_distribution": _oracle_distribution(oracle),
        "oracle_ranges": _oracle_ranges(labeled, daily),
        "primary": {
            "day1_mean_regret": float(metrics["fixed_day_1"]["mean_regret"]),
            "day1_within_rate": float(metrics["fixed_day_1"]["within_0_5pct_rate"]),
            "model_mean_regret": float(metrics["all"]["mean_regret"]),
            "model_improvement_bps": improvement,
            "model_ci95_bps": model_ci,
        },
        "holdout": _holdout_result(labeled, model),
        "day1_vs_day5": {"improvement_bps": day_comparison, "ci95_bps": day_ci},
        "strategy_metrics": metrics,
    }


def run_analysis(
    *,
    daily_path: Path = Path("data/processed/0050_daily.csv"),
    processed_dir: Path = Path("data/processed"),
    reports_dir: Path = Path("reports"),
    as_of: date | None = None,
    reuse_existing_predictions: bool = True,
) -> dict[str, Any]:
    """執行完整特徵、walk-forward、評估、圖表與報告流程。"""

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(daily_path, parse_dates=["date", "feature_available_date"])
    complete_through = last_complete_month(as_of or date.today())
    features, labeled, oracle = _prepare_frames(daily, complete_through)
    predictions = _model_outputs(labeled, processed_dir, reuse_existing=reuse_existing_predictions)
    first_oos = min(frame["month"].min() for frame in predictions.values())
    oos = labeled.loc[labeled["month"] >= first_oos].reset_index(drop=True)
    terminal_close = float(daily["adjusted_close"].iloc[-1])
    purchases, metrics = _evaluate_models(predictions, terminal_close)
    baselines = _baseline_outputs(oos)
    metrics.update(
        {name: _metric_with_wealth(frame, terminal_close) for name, frame in baselines.items()}
    )
    results = _build_results(daily, labeled, oracle, purchases, metrics, baselines)
    random = random_strategy_distribution(oos)
    results["random_strategy"] = {
        "mean_regret": float(random.mean()),
        "ci95": [float(value) for value in np.quantile(random, [0.025, 0.975])],
    }
    profile = oracle_feature_profile(labeled, MODEL_FEATURE_SETS["all"])

    features.to_csv(processed_dir / "0050_features.csv", index=False, encoding="utf-8-sig")
    labeled.to_csv(processed_dir / "0050_labeled_daily.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv(processed_dir / "0050_monthly_oracle.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(processed_dir / "oracle_feature_profile.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions.values(), ignore_index=True).to_csv(
        processed_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(purchases.values(), ignore_index=True).to_csv(
        processed_dir / "walk_forward_purchases.csv", index=False, encoding="utf-8-sig"
    )
    (reports_dir / "research_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "0050_buy_point_analysis.md").write_text(
        render_report(results), encoding="utf-8"
    )
    generate_figures(daily, oracle, profile, metrics, reports_dir / "figures")
    return results
