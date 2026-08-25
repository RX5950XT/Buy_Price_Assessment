"""0050 最佳買點研究的可重跑分析流程。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
    select_first_true_or_deadline,
    select_fixed_day,
    select_last_day,
    select_rsi_rule,
    strategy_metrics,
    terminal_wealth_proxy,
)
from buy_price_assessment.features import FEATURE_GROUPS, build_features
from buy_price_assessment.labels import add_monthly_labels, monthly_oracle_table
from buy_price_assessment.lead import (
    LEAD_RULES,
    LEAD_SIGNAL_COLUMNS,
    attach_lead_features,
    load_lead_data,
)
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
VT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "technical_calendar": FEATURE_GROUPS["technical"] + FEATURE_GROUPS["calendar"],
}
MODEL_CACHE_VERSION = "2"
HOLDOUT_START = "2023-07"
HOLDOUT_END = "2026-06"


@dataclass(frozen=True)
class ResearchInstrument:
    """單一標的的研究設定；VT 複製 0050 協議但不使用臺灣 NAV／籌碼。"""

    code: str
    display_name: str
    first_complete_month: str
    feature_sets: dict[str, tuple[str, ...]]
    primary_model: str
    primary_label: str
    file_stem: str
    report_filename: str
    results_filename: str
    figure_relpath: str
    currency_label: str
    fill_note: str
    lead_market_note: str
    has_taiwan_crosscheck: bool
    price_column: str
    price_title: str
    price_ylabel: str
    deadline_first_5_rate: float


TW_INSTRUMENT = ResearchInstrument(
    code="0050",
    display_name="0050",
    first_complete_month="2003-07",
    feature_sets=MODEL_FEATURE_SETS,
    primary_model="all",
    primary_label="全特徵模型",
    file_stem="0050",
    report_filename="0050_buy_point_analysis.md",
    results_filename="research_results.json",
    figure_relpath="figures",
    currency_label="新臺幣",
    fill_note="開盤集合競價可確保成交，盤中低點不一定成交。",
    lead_market_note=(
        "美股與匯率只使用臺灣交易日曆日前一日（含）已收盤的資料，當日美股不可用於當日開盤決策。"
    ),
    has_taiwan_crosscheck=True,
    price_column="split_adjusted_close",
    price_title="0050 分割還原收盤價（對數軸）",
    price_ylabel="新臺幣／目前受益權單位",
    deadline_first_5_rate=0.482,
)
VT_INSTRUMENT = ResearchInstrument(
    code="VT",
    display_name="VT（Vanguard Total World Stock ETF）",
    first_complete_month="2008-07",
    feature_sets=VT_FEATURE_SETS,
    primary_model="technical_calendar",
    primary_label="技術＋日曆模型",
    file_stem="VT",
    report_filename="VT_buy_point_analysis.md",
    results_filename="vt_research_results.json",
    figure_relpath="figures/vt",
    currency_label="美元",
    fill_note="成交價採美股常規交易時段官方 Open，不是臺灣開盤集合競價；盤中低點不視為可保證成交。",
    lead_market_note=(
        "TSM／SOX 與 VT 同屬美股交易時段，as-of 仍為目標日 T-1，"
        "用的是前一美股交易日、不是跨市場隔夜資訊；當日美股收盤不可用於當日開盤決策。"
        "USD/TWD 同樣只用 T-1。第 5 日截止沿用 0050 已公布的 oracle 前 5 日占比，不依 VT 重估。"
    ),
    has_taiwan_crosscheck=False,
    price_column="adjusted_close",
    price_title="VT total-return 還原收盤價（對數軸）",
    price_ylabel="美元／股",
    deadline_first_5_rate=0.482,
)


@dataclass(frozen=True)
class PurchasePolicy:
    """預先指定的買入規則；不在樣本外挑選。"""

    name: str
    probability_threshold: float | None
    use_reservation: bool
    fallback_trading_day: int | None


PURCHASE_POLICIES: tuple[PurchasePolicy, ...] = (
    PurchasePolicy("prob_and_res", 0.5, True, None),
    PurchasePolicy("prob_only", 0.5, False, None),
    PurchasePolicy("res_only", None, True, None),
    PurchasePolicy("prob_and_res_deadline5", 0.5, True, 5),
    PurchasePolicy("prob_only_deadline5", 0.5, False, 5),
    PurchasePolicy("res_only_deadline5", None, True, 5),
)


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
    daily: pd.DataFrame,
    complete_through: str,
    *,
    first_complete_month: str = "2003-07",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(daily)
    labeled = add_monthly_labels(features, complete_through=complete_through)
    labeled = labeled.loc[labeled["month"] >= first_complete_month].reset_index(drop=True)
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
    feature_sets: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, pd.DataFrame]:
    config = WalkForwardConfig()
    sets = feature_sets or MODEL_FEATURE_SETS
    return {
        name: _load_or_run_prediction(
            labeled,
            name,
            columns,
            processed_dir,
            config,
            reuse_existing,
        )
        for name, columns in sets.items()
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
    default_policy = PURCHASE_POLICIES[0]
    for name, prediction in predictions.items():
        selected = select_monthly_purchases(
            prediction,
            probability_threshold=default_policy.probability_threshold,
            use_reservation=default_policy.use_reservation,
            fallback_trading_day=default_policy.fallback_trading_day,
        )
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


def _oracle_ranges(
    labeled: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    include_taiwan_features: bool = True,
) -> dict[str, dict[str, float | int]]:
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
        "volume_z_20",
    ]
    if include_taiwan_features:
        columns.extend(
            [
                "dividend_yield_ttm",
                "premium_discount",
                "margin_change_5",
                "institutional_net_ratio",
            ]
        )
    ranges: dict[str, dict[str, float | int]] = {}
    for column in columns:
        if column not in oracle.columns:
            continue
        values = oracle[column].dropna().astype(float)
        if len(values) == 0 or values.nunique() <= 1:
            continue
        ranges[column] = {
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "mean": float(values.mean()),
            "n": len(values),
        }
    return ranges


def _holdout_slice(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[(frame["month"] >= HOLDOUT_START) & (frame["month"] <= HOLDOUT_END)]


def _compare_to_day1(
    purchases: pd.DataFrame,
    day1: pd.DataFrame,
    terminal_close: float,
) -> dict[str, Any]:
    item: dict[str, Any] = dict(_metric_with_wealth(purchases, terminal_close))
    improvement, ci = _paired_improvement(purchases, day1)
    holdout = _holdout_slice(purchases)
    holdout_improvement, holdout_ci = _paired_improvement(
        holdout, _holdout_slice(day1), block_length=6
    )
    holdout_metrics = strategy_metrics(holdout)
    item.update(
        {
            "improvement_bps": improvement,
            "ci95_bps": ci,
            "holdout_mean_regret": float(holdout_metrics["mean_regret"]),
            "holdout_forced_rate": float(holdout_metrics.get("forced_rate", 0.0)),
            "holdout_mean_trading_day": float(holdout_metrics["mean_trading_day"]),
            "holdout_improvement_bps": holdout_improvement,
            "holdout_ci95_bps": holdout_ci,
        }
    )
    return item


def _evaluate_purchase_policies(
    prediction: pd.DataFrame,
    day1: pd.DataFrame,
    terminal_close: float,
) -> dict[str, dict[str, Any]]:
    """用既有樣本外預測評估預先指定的買入規則，不重訓模型。"""

    results: dict[str, dict[str, Any]] = {}
    for policy in PURCHASE_POLICIES:
        purchases = select_monthly_purchases(
            prediction,
            probability_threshold=policy.probability_threshold,
            use_reservation=policy.use_reservation,
            fallback_trading_day=policy.fallback_trading_day,
        )
        results[policy.name] = _compare_to_day1(purchases, day1, terminal_close)
    return results


def _evaluate_lead_rules(
    oos: pd.DataFrame,
    day1: pd.DataFrame,
    terminal_close: float,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, column in LEAD_RULES:
        purchases = select_first_true_or_deadline(oos, column=column, fallback_trading_day=5)
        item = _compare_to_day1(purchases, day1, terminal_close)
        item["signal_rate"] = float(oos[column].mean())
        results[name] = item
    return results


def _lead_coverage(oos: pd.DataFrame) -> dict[str, float]:
    fx_available = oos["usd_twd_up"] if "usd_twd_up" in oos.columns else oos["usd_twd_up_streak3"]
    return {
        "tsm_available_rate": float(oos["tsm_lead_ret"].notna().mean()),
        "sox_available_rate": float(oos["sox_lead_ret"].notna().mean()),
        "fx_available_rate": float(fx_available.notna().mean()),
        "tsm_dump_rate": float(oos["tsm_dump"].mean()),
        "sox_dump_rate": float(oos["sox_dump"].mean()),
        "fx_pause_rate": float((~oos["fx_not_depreciating"]).mean()),
        "tsm_dump_1pct_rate": float(oos["tsm_dump_1pct"].mean()),
        "tsm_adverse_rate": float((~oos["tsm_not_dump"]).mean()),
        "fx_single_pause_rate": float((~oos["fx_not_up"]).mean()),
    }


def _holdout_result(
    labeled: pd.DataFrame,
    model_purchases: pd.DataFrame,
    *,
    development_start: str = "2008-07",
) -> dict[str, Any]:
    development = labeled.loc[
        (labeled["month"] >= development_start) & (labeled["month"] < HOLDOUT_START)
    ]
    holdout = _holdout_slice(labeled)
    candidates = {
        day: select_fixed_day(development, trading_day=day)["regret"].mean()
        for day in (1, 5, 10, 15)
    }
    selected_day = min(candidates, key=lambda day: candidates[day])
    fixed = select_fixed_day(holdout, trading_day=selected_day)
    model = _holdout_slice(model_purchases)
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


def _instrument_meta(
    instrument: ResearchInstrument,
    *,
    oos_start: str,
    oos_end: str,
    oos_months: int,
) -> dict[str, Any]:
    return {
        "instrument": instrument.code,
        "display_name": instrument.display_name,
        "first_complete_month": instrument.first_complete_month,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "oos_months": oos_months,
        "primary_model": instrument.primary_model,
        "primary_label": instrument.primary_label,
        "figure_relpath": instrument.figure_relpath,
        "currency_label": instrument.currency_label,
        "fill_note": instrument.fill_note,
        "lead_market_note": instrument.lead_market_note,
        "has_taiwan_crosscheck": instrument.has_taiwan_crosscheck,
        "report_path": f"reports/{instrument.report_filename}",
        "deadline_first_5_rate": instrument.deadline_first_5_rate,
    }


def _build_results(
    daily: pd.DataFrame,
    labeled: pd.DataFrame,
    oracle: pd.DataFrame,
    purchases: Mapping[str, pd.DataFrame],
    metrics: dict[str, dict[str, int | float]],
    baselines: Mapping[str, pd.DataFrame],
    policy_ablation: Mapping[str, Mapping[str, Any]],
    *,
    instrument: ResearchInstrument = TW_INSTRUMENT,
    oos_start: str = "2008-07",
    oos_end: str = "2026-06",
) -> dict[str, Any]:
    day1 = baselines["fixed_day_1"]
    model = purchases[instrument.primary_model]
    improvement, model_ci = _paired_improvement(model, day1)
    day_comparison, day_ci = _paired_improvement(day1, baselines["fixed_day_5"])
    primary_metrics = metrics[instrument.primary_model]
    return {
        "data_quality": _data_quality(daily),
        "oracle_distribution": _oracle_distribution(oracle),
        "oracle_ranges": _oracle_ranges(
            labeled,
            daily,
            include_taiwan_features=instrument.has_taiwan_crosscheck,
        ),
        "primary": {
            "day1_mean_regret": float(metrics["fixed_day_1"]["mean_regret"]),
            "day1_within_rate": float(metrics["fixed_day_1"]["within_0_5pct_rate"]),
            "model_mean_regret": float(primary_metrics["mean_regret"]),
            "model_improvement_bps": improvement,
            "model_ci95_bps": model_ci,
        },
        "holdout": _holdout_result(labeled, model, development_start=oos_start),
        "day1_vs_day5": {"improvement_bps": day_comparison, "ci95_bps": day_ci},
        "strategy_metrics": metrics,
        "policy_ablation": dict(policy_ablation),
        "meta": _instrument_meta(
            instrument,
            oos_start=oos_start,
            oos_end=oos_end,
            oos_months=int(metrics["fixed_day_1"]["months"]),
        ),
    }


def _run_research(
    *,
    daily_path: Path,
    processed_dir: Path,
    reports_dir: Path,
    raw_dir: Path,
    as_of: date | None,
    reuse_existing_predictions: bool,
    instrument: ResearchInstrument,
) -> dict[str, Any]:
    """執行完整特徵、walk-forward、評估、圖表與報告流程。"""

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(daily_path, parse_dates=["date", "feature_available_date"])
    complete_through = last_complete_month(as_of or daily["date"].max().date())
    features, labeled, oracle = _prepare_frames(
        daily,
        complete_through,
        first_complete_month=instrument.first_complete_month,
    )
    lead = load_lead_data(raw_dir)
    if lead is None:
        raise ValueError(
            f"{instrument.code} 缺少領先序列：data/raw 需有 tsm_us.csv、sox_us.csv、usd_twd.csv"
        )
    labeled = attach_lead_features(labeled, tsm=lead["tsm"], sox=lead["sox"], fx=lead["fx"])
    predictions = _model_outputs(
        labeled,
        processed_dir,
        reuse_existing=reuse_existing_predictions,
        feature_sets=instrument.feature_sets,
    )
    first_oos = min(frame["month"].min() for frame in predictions.values())
    oos = labeled.loc[labeled["month"] >= first_oos].reset_index(drop=True)
    terminal_close = float(daily["adjusted_close"].iloc[-1])
    purchases, metrics = _evaluate_models(predictions, terminal_close)
    baselines = _baseline_outputs(oos)
    metrics.update(
        {name: _metric_with_wealth(frame, terminal_close) for name, frame in baselines.items()}
    )
    policy_ablation = _evaluate_purchase_policies(
        predictions[instrument.primary_model],
        baselines["fixed_day_1"],
        terminal_close,
    )
    if not set(LEAD_SIGNAL_COLUMNS).issubset(oos.columns):
        missing = sorted(set(LEAD_SIGNAL_COLUMNS).difference(oos.columns))
        raise ValueError(f"領先訊號欄位未接到每日表：{missing}")
    lead_rules = _evaluate_lead_rules(oos, baselines["fixed_day_1"], terminal_close)
    lead_coverage = _lead_coverage(oos)
    results = _build_results(
        daily,
        labeled,
        oracle,
        purchases,
        metrics,
        baselines,
        policy_ablation,
        instrument=instrument,
        oos_start=str(first_oos),
        oos_end=complete_through,
    )
    random = random_strategy_distribution(oos)
    results["random_strategy"] = {
        "mean_regret": float(random.mean()),
        "ci95": [float(value) for value in np.quantile(random, [0.025, 0.975])],
    }
    results["lead_rules"] = lead_rules
    results["lead_coverage"] = lead_coverage
    profile_columns: tuple[str, ...] = tuple(
        dict.fromkeys(column for columns in instrument.feature_sets.values() for column in columns)
    )
    profile = oracle_feature_profile(labeled, profile_columns)

    stem = instrument.file_stem
    features.to_csv(processed_dir / f"{stem}_features.csv", index=False, encoding="utf-8-sig")
    labeled.to_csv(processed_dir / f"{stem}_labeled_daily.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv(processed_dir / f"{stem}_monthly_oracle.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(processed_dir / "oracle_feature_profile.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions.values(), ignore_index=True).to_csv(
        processed_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(purchases.values(), ignore_index=True).to_csv(
        processed_dir / "walk_forward_purchases.csv", index=False, encoding="utf-8-sig"
    )
    (reports_dir / instrument.results_filename).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / instrument.report_filename).write_text(render_report(results), encoding="utf-8")
    generate_figures(
        daily,
        oracle,
        profile,
        metrics,
        reports_dir / Path(instrument.figure_relpath),
        policy_ablation=policy_ablation,
        day1_mean_regret=float(metrics["fixed_day_1"]["mean_regret"]),
        lead_rules=lead_rules,
        primary_model=instrument.primary_model,
        primary_label=instrument.primary_label,
        price_column=instrument.price_column,
        price_title=instrument.price_title,
        price_ylabel=instrument.price_ylabel,
    )
    return results


def run_analysis(
    *,
    daily_path: Path = Path("data/processed/0050_daily.csv"),
    processed_dir: Path = Path("data/processed"),
    reports_dir: Path = Path("reports"),
    raw_dir: Path = Path("data/raw"),
    as_of: date | None = None,
    reuse_existing_predictions: bool = True,
) -> dict[str, Any]:
    """0050 完整研究流程。"""

    return _run_research(
        daily_path=daily_path,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        raw_dir=raw_dir,
        as_of=as_of,
        reuse_existing_predictions=reuse_existing_predictions,
        instrument=TW_INSTRUMENT,
    )


def run_vt_analysis(
    *,
    daily_path: Path = Path("data/processed/VT_daily.csv"),
    processed_dir: Path = Path("data/processed/vt"),
    reports_dir: Path = Path("reports"),
    raw_dir: Path = Path("data/raw"),
    as_of: date | None = None,
    reuse_existing_predictions: bool = True,
) -> dict[str, Any]:
    """VT 複製同一套月內買一次協議；產物不覆寫 0050。"""

    return _run_research(
        daily_path=daily_path,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        raw_dir=raw_dir,
        as_of=as_of,
        reuse_existing_predictions=reuse_existing_predictions,
        instrument=VT_INSTRUMENT,
    )
