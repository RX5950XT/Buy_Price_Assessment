"""月度策略基準、統計與不確定性評估。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_monthly_rows(frame: pd.DataFrame) -> None:
    required = {"month", "date", "trading_day", "days_in_month", "regret", "adjusted_open"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"評估資料缺少欄位：{sorted(missing)}")
    if frame.empty:
        raise ValueError("評估資料不可為空")


def select_fixed_day(frame: pd.DataFrame, *, trading_day: int) -> pd.DataFrame:
    """選每月第 N 個交易日；不足 N 日時選月底。"""

    _validate_monthly_rows(frame)
    if isinstance(trading_day, bool) or not isinstance(trading_day, int) or trading_day < 1:
        raise ValueError("trading_day 必須是正整數")
    ordered = frame.sort_values(["month", "trading_day"], ignore_index=True)
    targets = ordered["days_in_month"].clip(upper=trading_day)
    selected = ordered.loc[ordered["trading_day"] == targets]
    if selected["month"].duplicated().any():
        raise ValueError("固定日基準每月選到多筆資料")
    return selected.reset_index(drop=True)


def select_last_day(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_monthly_rows(frame)
    selected = frame.loc[frame["trading_day"] == frame["days_in_month"]]
    return selected.sort_values("month", ignore_index=True)


def select_rsi_rule(frame: pd.DataFrame, *, threshold: float = 30.0) -> pd.DataFrame:
    """每月首次前日 RSI 低於門檻時買，否則月底買。"""

    _validate_monthly_rows(frame)
    if "rsi_14" not in frame.columns:
        raise ValueError("RSI 基準缺少 rsi_14")
    selected: list[pd.Series] = []
    for _, month in frame.sort_values("date").groupby("month", sort=True):
        triggered = month.loc[month["rsi_14"] < threshold]
        row = month.iloc[-1] if triggered.empty else triggered.iloc[0]
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def strategy_metrics(purchases: pd.DataFrame) -> dict[str, int | float]:
    """以月份為觀測單位計算策略結果。"""

    _validate_monthly_rows(purchases)
    if purchases["month"].duplicated().any():
        raise ValueError("策略每月必須恰好一筆買入")
    regret = purchases["regret"].astype(float)
    metrics: dict[str, int | float] = {
        "months": len(purchases),
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "p75_regret": float(regret.quantile(0.75)),
        "p90_regret": float(regret.quantile(0.90)),
        "p95_regret": float(regret.quantile(0.95)),
        "within_0_5pct_rate": float((regret <= 0.005).mean()),
        "within_1pct_rate": float((regret <= 0.01).mean()),
        "mean_trading_day": float(purchases["trading_day"].mean()),
    }
    if "forced" in purchases.columns:
        metrics["forced_rate"] = float(purchases["forced"].astype(bool).mean())
    return metrics


def moving_block_bootstrap_ci(
    differences: pd.Series | np.ndarray,
    *,
    block_length: int = 12,
    simulations: int = 5_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """以循環 moving-block bootstrap 估計平均差異的 95% CI。"""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap 差異必須是一維有限數列且至少兩筆")
    if block_length < 1 or simulations < 1:
        raise ValueError("block_length 與 simulations 必須大於零")
    block = min(block_length, len(values))
    block_count = math.ceil(len(values) / block)
    rng = np.random.default_rng(random_state)
    starts = rng.integers(0, len(values), size=(simulations, block_count))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % len(values)
    indices = indices.reshape(simulations, -1)[:, : len(values)]
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def random_strategy_distribution(
    frame: pd.DataFrame,
    *,
    simulations: int = 5_000,
    random_state: int = 42,
) -> np.ndarray:
    """每月均勻隨機選一天，回傳各模擬的平均 regret。"""

    _validate_monthly_rows(frame)
    rng = np.random.default_rng(random_state)
    totals = np.zeros(simulations)
    month_count = 0
    for _, month in frame.groupby("month", sort=True):
        values = month["regret"].to_numpy(dtype=float)
        totals += values[rng.integers(0, len(values), size=simulations)]
        month_count += 1
    return totals / month_count


def terminal_wealth_proxy(
    purchases: pd.DataFrame,
    *,
    terminal_adjusted_close: float,
    monthly_contribution: float = 10_000.0,
    commission_rate: float = 0.001425,
) -> float:
    """以 fractional total-return units 比較同額月投資的期末價值。"""

    _validate_monthly_rows(purchases)
    invested = monthly_contribution / (1.0 + commission_rate)
    units = (invested / purchases["adjusted_open"].astype(float)).sum()
    return float(units * terminal_adjusted_close)


def oracle_feature_profile(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """計算 oracle 日特徵在各自月份內的平均 percentile rank。"""

    if "is_oracle" not in frame.columns:
        raise ValueError("特徵描述缺少 is_oracle")
    rows: list[dict[str, float | str | int]] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        ranks = frame.groupby("month")[feature].rank(pct=True)
        oracle_ranks = ranks.loc[frame["is_oracle"].astype(bool)].dropna()
        if oracle_ranks.empty:
            continue
        mean_rank = float(oracle_ranks.mean())
        rows.append(
            {
                "feature": feature,
                "oracle_mean_percentile": mean_rank,
                "distance_from_median": mean_rank - 0.5,
                "months": len(oracle_ranks),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.reindex(
        result["distance_from_median"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
