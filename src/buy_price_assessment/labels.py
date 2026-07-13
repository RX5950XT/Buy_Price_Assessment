"""每月事後最佳買點標籤。

標籤只描述事後結果, 不能直接當成當日可用特徵。最佳買點使用還原開盤價,
同價時固定選擇最早交易日, 確保結果可重現。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_DAILY_COLUMNS = frozenset({"date", "open", "adjusted_open", "adjusted_close"})
REQUIRED_LABELED_COLUMNS = frozenset(
    {
        "month",
        "date",
        "open",
        "adjusted_open",
        "trading_day",
        "days_in_month",
        "regret",
    }
)


def _parse_month(value: str | pd.Period) -> pd.Period:
    """將月份輸入正規化為月頻 Period。"""

    try:
        period = pd.Period(value, freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError(f"無效月份: {value!r}") from error
    if str(period) == "NaT":
        raise ValueError("月份不可為空")
    return period


def _validated_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("每日資料必須是 pandas DataFrame")
    missing = REQUIRED_DAILY_COLUMNS.difference(daily.columns)
    if missing:
        raise ValueError(f"每日資料缺少欄位: {sorted(missing)}")
    if daily.empty:
        raise ValueError("每日資料不可為空")

    frame = daily.copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("每日資料含無效日期") from error
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("每日資料日期不可為空或重複")

    numeric_columns = ["open", "adjusted_open", "adjusted_close"]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if numeric.isna().any().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("開盤價與還原價格必須是有限正數")
    frame.loc[:, numeric_columns] = numeric
    return frame.sort_values("date", kind="mergesort", ignore_index=True)


def _empty_labeled(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["month"] = pd.Series(dtype="string")
    result["trading_day"] = pd.Series(dtype="int64")
    result["days_in_month"] = pd.Series(dtype="int64")
    result["oracle_date"] = pd.Series(dtype="datetime64[ns]")
    result["oracle_adjusted_open"] = pd.Series(dtype="float64")
    result["regret"] = pd.Series(dtype="float64")
    result["is_oracle"] = pd.Series(dtype="bool")
    result["near_optimal"] = pd.Series(dtype="bool")
    result["remaining_min_adjusted_open"] = pd.Series(dtype="float64")
    result["previous_adjusted_close"] = pd.Series(dtype="float64")
    result["remaining_min_log_ratio"] = pd.Series(dtype="float64")
    return result


def add_monthly_labels(
    daily: pd.DataFrame,
    *,
    complete_through: str | pd.Period,
    near_threshold: float = 0.005,
) -> pd.DataFrame:
    """加入每月最低還原開盤價、後悔值與交易日標籤。

    ``complete_through`` 是最後一個已確認收盤的完整月份; 較新的資料會被排除,
    避免把尚未結束月份的暫時低點誤當最終標籤。
    """

    if not np.isfinite(near_threshold) or not 0.0 <= near_threshold <= 1.0:
        raise ValueError("near_threshold 必須介於 0 與 1")
    frame = _validated_daily(daily)
    cutoff = _parse_month(complete_through)
    month_periods = frame["date"].dt.to_period("M")
    result = frame.loc[month_periods <= cutoff].copy()
    if result.empty:
        return _empty_labeled(result)

    result["month"] = result["date"].dt.to_period("M").astype(str)
    grouped = result.groupby("month", sort=False)
    result["trading_day"] = (grouped.cumcount() + 1).astype("int64")
    result["days_in_month"] = grouped["date"].transform("size").astype("int64")
    result["oracle_adjusted_open"] = grouped["adjusted_open"].transform("min")

    oracle_rows = (
        result.sort_values(["month", "adjusted_open", "date"], kind="mergesort")
        .drop_duplicates("month", keep="first")
        .set_index("month")
    )
    result["oracle_date"] = result["month"].map(oracle_rows["date"])
    result["regret"] = result["adjusted_open"] / result["oracle_adjusted_open"] - 1.0
    result["is_oracle"] = result["date"].eq(result["oracle_date"])
    result["near_optimal"] = result["regret"] <= near_threshold
    result["remaining_min_adjusted_open"] = grouped["adjusted_open"].transform(
        lambda values: values.iloc[::-1].cummin().iloc[::-1]
    )
    result["previous_adjusted_close"] = result["adjusted_close"].shift(1)
    result["remaining_min_log_ratio"] = np.log(
        result["remaining_min_adjusted_open"] / result["previous_adjusted_close"]
    )
    reset_result: pd.DataFrame = result.reset_index(drop=True)
    return reset_result


def _validated_labeled(labeled: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(labeled, pd.DataFrame):
        raise TypeError("標籤資料必須是 pandas DataFrame")
    missing = REQUIRED_LABELED_COLUMNS.difference(labeled.columns)
    if missing:
        raise ValueError(f"標籤資料缺少欄位: {sorted(missing)}")

    frame = labeled.copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        month_periods = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError("標籤資料含無效日期或月份") from error
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("標籤資料日期不可為空或重複")
    if not frame.empty and not (frame["date"].dt.to_period("M") == month_periods).all():
        raise ValueError("month 與 date 所屬月份不一致")
    frame["month"] = month_periods.astype(str)
    return frame


def monthly_oracle_table(labeled: pd.DataFrame) -> pd.DataFrame:
    """將每日標籤彙整成每月唯一、同價取最早日的事後最佳買點。"""

    frame = _validated_labeled(labeled)
    columns = [
        "month",
        "oracle_date",
        "oracle_open",
        "oracle_adjusted_open",
        "oracle_trading_day",
        "days_in_month",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    oracle = (
        frame.sort_values(["month", "adjusted_open", "date"], kind="mergesort")
        .drop_duplicates("month", keep="first")
        .loc[:, ["month", "date", "open", "adjusted_open", "trading_day", "days_in_month"]]
        .rename(
            columns={
                "date": "oracle_date",
                "open": "oracle_open",
                "adjusted_open": "oracle_adjusted_open",
                "trading_day": "oracle_trading_day",
            }
        )
    )
    return oracle.loc[:, columns].sort_values("month", ignore_index=True)
