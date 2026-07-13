"""只使用決策時點以前資訊的每日特徵。"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "technical": (
        "ret_1",
        "ret_5",
        "ret_20",
        "ret_60",
        "ma_gap_20",
        "ma_gap_60",
        "rsi_14",
        "bollinger_z_20",
        "drawdown_60",
        "volatility_20",
        "volume_z_20",
    ),
    "valuation": ("premium_discount", "dividend_yield_ttm"),
    "chip": (
        "margin_change_5",
        "short_change_5",
        "institutional_net_ratio",
        "institutional_data_available",
        "fund_flow_5",
    ),
    "calendar": ("trading_day", "month_progress", "weekday_sin", "weekday_cos"),
}

REQUIRED_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "total_return_index",
    "nav",
    "cash_dividend",
    "margin_balance",
    "short_balance",
    "institutional_net",
    "outstanding_units",
}


def _validate_daily(daily: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(daily.columns)
    if missing:
        raise ValueError(f"每日資料缺少欄位：{sorted(missing)}")
    if daily.empty:
        raise ValueError("每日資料不可為空")
    if daily["date"].duplicated().any():
        raise ValueError("每日資料含重複日期")


def _rsi(price: pd.Series, period: int = 14) -> pd.Series:
    change = price.diff()
    gain = change.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = -change.clip(upper=0).rolling(period, min_periods=period).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative_strength)
    return result.mask((loss == 0.0) & (gain > 0.0), 100.0).mask(
        (loss == 0.0) & (gain == 0.0), 50.0
    )


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    deviation = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / deviation.replace(0.0, np.nan)


def _cumulative_split(frame: pd.DataFrame) -> pd.Series:
    if "split_ratio" not in frame.columns:
        return pd.Series(1.0, index=frame.index)
    return frame["split_ratio"].fillna(1.0).cumprod()


def _technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    price = frame["total_return_index"].astype(float)
    returns = price.pct_change()
    features = pd.DataFrame(index=frame.index)
    for window in (1, 5, 20, 60):
        features[f"ret_{window}"] = price.pct_change(window)
    for window in (20, 60):
        features[f"ma_gap_{window}"] = price / price.rolling(window).mean() - 1.0
    features["rsi_14"] = _rsi(price)
    features["bollinger_z_20"] = _rolling_zscore(price, 20)
    features["drawdown_60"] = price / price.rolling(60).max() - 1.0
    features["volatility_20"] = returns.rolling(20).std(ddof=0) * np.sqrt(252.0)
    comparable_volume = frame["volume"].astype(float) / _cumulative_split(frame)
    features["volume_z_20"] = _rolling_zscore(comparable_volume, 20)
    return features


def _valuation_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["premium_discount"] = frame["close"] / frame["nav"] - 1.0
    if "distribution_yield_ttm" in frame.columns:
        features["dividend_yield_ttm"] = frame["distribution_yield_ttm"]
    else:
        trailing_cash = frame.set_index("date")["cash_dividend"].rolling("365D").sum().to_numpy()
        features["dividend_yield_ttm"] = trailing_cash / frame["close"].to_numpy()
    return features


def _chip_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    cumulative_split = _cumulative_split(frame)
    comparable_margin = frame["margin_balance"] / cumulative_split
    comparable_short = frame["short_balance"] / cumulative_split
    comparable_units = frame["outstanding_units"] / cumulative_split
    features["margin_change_5"] = comparable_margin.pct_change(5)
    features["short_change_5"] = comparable_short.pct_change(5)
    features["institutional_net_ratio"] = frame["institutional_net"] / frame["volume"].replace(
        0, np.nan
    )
    available = (
        frame["institutional_available"]
        if "institutional_available" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    features["institutional_data_available"] = available.astype(float)
    features["fund_flow_5"] = comparable_units.pct_change(5)
    return features


def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    months = frame["date"].dt.to_period("M")
    trading_day = frame.groupby(months).cumcount() + 1
    weekday = frame["date"].dt.weekday
    return pd.DataFrame(
        {
            "trading_day": trading_day,
            "month_progress": frame["date"].dt.day / frame["date"].dt.days_in_month,
            "weekday_sin": np.sin(2.0 * np.pi * weekday / 5.0),
            "weekday_cos": np.cos(2.0 * np.pi * weekday / 5.0),
        },
        index=frame.index,
    )


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """建立隔日開盤前可用特徵；所有日終資料自動 lag 一日。"""

    _validate_daily(daily)
    result = daily.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values("date", ignore_index=True)
    end_of_day = pd.concat(
        [_technical_features(result), _valuation_features(result), _chip_features(result)], axis=1
    ).shift(1)
    calendar = _calendar_features(result)
    features = pd.concat([end_of_day, calendar], axis=1).replace([np.inf, -np.inf], np.nan)
    return pd.concat([result, features], axis=1)
