"""無前視偏誤的月度 walk-forward 切分與買入規則。"""

from __future__ import annotations

import math
from collections.abc import Iterator
from numbers import Integral, Real
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

REQUIRED_PURCHASE_COLUMNS = frozenset(
    {
        "month",
        "date",
        "trading_day",
        "days_in_month",
        "near_probability",
        "adjusted_open",
        "reservation_adjusted",
    }
)


@runtime_checkable
class WalkForwardEstimator(Protocol):
    """後續 walk-forward runner 可替換的最小機率模型介面。"""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        """只以當期切分的訓練資料擬合模型。"""

        ...

    def predict_near_probability(self, features: pd.DataFrame) -> pd.Series:
        """回傳各交易日成為近最佳買點的機率。"""

        ...


def _month_periods(frame: pd.DataFrame) -> pd.PeriodIndex:
    if "month" not in frame.columns:
        raise ValueError("資料缺少欄位: month")
    try:
        periods = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError("month 含無效月份") from error
    if np.asarray(periods.isna()).any():
        raise ValueError("month 不可為空")
    return periods


def expanding_month_splits(
    frame: pd.DataFrame,
    *,
    initial_months: int,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """逐月產生 expanding-window 的 index label, 測試集固定為下一個月。"""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("切分資料必須是 pandas DataFrame")
    if frame.empty:
        raise ValueError("切分資料不可為空")
    if not frame.index.is_unique:
        raise ValueError("切分資料 index 必須唯一")
    validated_initial_months = _validated_initial_months(initial_months)
    if validated_initial_months < 1:
        raise ValueError("initial_months 必須大於零")

    periods = _month_periods(frame)
    ordered_months = periods.unique().sort_values()
    if validated_initial_months >= len(ordered_months):
        raise ValueError("initial_months 必須小於資料的月份數")

    for test_position in range(validated_initial_months, len(ordered_months)):
        test_month = ordered_months[test_position]
        train_months = ordered_months[:test_position]
        train_index = frame.index[periods.isin(train_months)]
        test_index = frame.index[periods == test_month]
        yield train_index, test_index


def _validated_initial_months(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("initial_months 必須是整數")
    return int(value)


def _validated_probability_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("probability_threshold 必須是數值")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("probability_threshold 必須介於 0 與 1")
    return threshold


def _validated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("預測資料必須是 pandas DataFrame")
    missing = REQUIRED_PURCHASE_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"預測資料缺少欄位: {sorted(missing)}")

    frame = predictions.copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("預測資料含無效日期") from error
    periods = _month_periods(frame)
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("預測日期不可為空或重複")
    if not frame.empty and not (frame["date"].dt.to_period("M") == periods).all():
        raise ValueError("month 與 date 所屬月份不一致")

    numeric_columns = [
        "trading_day",
        "days_in_month",
        "near_probability",
        "adjusted_open",
        "reservation_adjusted",
    ]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if numeric.isna().any().any() or not np.isfinite(values).all():
        raise ValueError("預測資料的決策欄位必須是有限數值")
    frame.loc[:, numeric_columns] = numeric
    frame["month"] = periods.astype(str)
    _validate_prediction_ranges(frame)
    return frame.sort_values(["month", "date"], kind="mergesort", ignore_index=True)


def _validate_prediction_ranges(frame: pd.DataFrame) -> None:
    probability = frame["near_probability"]
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("near_probability 必須介於 0 與 1")
    if (frame[["adjusted_open", "reservation_adjusted"]] <= 0.0).any().any():
        raise ValueError("還原開盤價與保留價必須大於零")

    trading_day = frame["trading_day"]
    days_in_month = frame["days_in_month"]
    if ((trading_day % 1 != 0) | (days_in_month % 1 != 0)).any():
        raise ValueError("交易日序號必須是整數")
    if ((trading_day < 1) | (days_in_month < trading_day)).any():
        raise ValueError("交易日序號超出月份範圍")

    for _, month_frame in frame.groupby("month", sort=False):
        expected_days = int(month_frame["days_in_month"].iloc[0])
        if month_frame["days_in_month"].nunique() != 1:
            raise ValueError("同月份的 days_in_month 必須一致")
        actual_days = month_frame["trading_day"].astype(int).sort_values().tolist()
        if actual_days != list(range(1, expected_days + 1)):
            raise ValueError("每個月份必須包含完整且不重複的交易日")


def select_monthly_purchases(
    predictions: pd.DataFrame,
    *,
    probability_threshold: float,
) -> pd.DataFrame:
    """每月選擇首個同時通過機率與保留價門檻的交易日。

    若整月都未觸發, 則在該月最後交易日強制買入, 確保每月恰好買一次。
    """

    threshold = _validated_probability_threshold(probability_threshold)
    frame = _validated_predictions(predictions)
    if frame.empty:
        result = frame.copy()
        result["forced"] = pd.Series(dtype="bool")
        return result

    selected_rows: list[pd.Series] = []
    for _, month_frame in frame.groupby("month", sort=True):
        eligible = month_frame.loc[
            (month_frame["near_probability"] >= threshold)
            & (month_frame["adjusted_open"] <= month_frame["reservation_adjusted"])
        ]
        forced = eligible.empty
        selected = month_frame.iloc[-1] if forced else eligible.iloc[0]
        selected = selected.copy()
        selected["forced"] = forced
        selected_rows.append(selected)

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result["forced"] = result["forced"].astype(bool)
    return result
