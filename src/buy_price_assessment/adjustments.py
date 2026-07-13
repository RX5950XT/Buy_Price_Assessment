"""現金配息與分割的價格還原。"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("date", "open", "high", "low", "close")


def _validate_prices(prices: pd.DataFrame) -> None:
    missing = set(PRICE_COLUMNS).difference(prices.columns)
    if missing:
        raise ValueError(f"價格資料缺少欄位：{sorted(missing)}")
    if prices.empty:
        raise ValueError("價格資料不可為空")
    if prices["date"].duplicated().any():
        raise ValueError("價格資料含重複日期")
    numeric = prices.loc[:, PRICE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise ValueError("OHLC 必須是正數")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("最高價低於其他 OHLC 價格")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("最低價高於其他 OHLC 價格")


def _aggregate_actions(
    frame: pd.DataFrame,
    value_column: str,
    default: float,
    aggregation: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", value_column])
    if not {"date", value_column}.issubset(frame.columns):
        raise ValueError(f"企業行動資料缺少 date 或 {value_column}")
    actions = frame.loc[:, ["date", value_column]].copy()
    actions["date"] = pd.to_datetime(actions["date"], errors="raise")
    actions[value_column] = pd.to_numeric(actions[value_column], errors="raise")
    if (actions[value_column] <= 0).any() and default == 1.0:
        raise ValueError("分割比率必須大於零")
    grouped = actions.groupby("date", as_index=False)[value_column]
    aggregated = grouped.sum() if aggregation == "sum" else grouped.prod()
    return cast(pd.DataFrame, aggregated)


def _future_split_factor(split_ratio: pd.Series) -> pd.Series:
    reverse_product = split_ratio.iloc[::-1].cumprod().iloc[::-1]
    return reverse_product / split_ratio


def build_adjusted_prices(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """回傳原始、分割還原與 total-return adjusted OHLC。"""

    _validate_prices(prices)
    result = prices.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values("date", ignore_index=True)

    cash = _aggregate_actions(dividends, "cash_dividend", 0.0, "sum")
    split = _aggregate_actions(splits, "split_ratio", 1.0, "product")
    result = result.merge(cash, on="date", how="left").merge(split, on="date", how="left")
    result["cash_dividend"] = result["cash_dividend"].fillna(0.0)
    result["split_ratio"] = result["split_ratio"].fillna(1.0)

    previous_close = result["close"].shift(1)
    gross = (result["close"] + result["cash_dividend"]) * result["split_ratio"] / previous_close
    result["gross_return"] = gross.fillna(1.0)
    if (result["gross_return"] <= 0).any() or not np.isfinite(result["gross_return"]).all():
        raise ValueError("企業行動產生無效總報酬")
    result["total_return_index"] = 100.0 * result["gross_return"].cumprod()

    terminal_close = float(result["close"].iloc[-1])
    terminal_index = float(result["total_return_index"].iloc[-1])
    result["adjusted_close"] = terminal_close * result["total_return_index"] / terminal_index
    result["adjustment_factor"] = result["adjusted_close"] / result["close"]
    for column in ("open", "high", "low"):
        result[f"adjusted_{column}"] = result[column] * result["adjustment_factor"]

    future_splits = _future_split_factor(result["split_ratio"])
    for column in ("open", "high", "low", "close"):
        result[f"split_adjusted_{column}"] = result[column] / future_splits
    return result
