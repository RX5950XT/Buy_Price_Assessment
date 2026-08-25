import numpy as np
import pandas as pd
import pytest

from buy_price_assessment.modeling import expanding_month_splits, select_monthly_purchases


def test_expanding_splits_never_train_on_test_or_future_months() -> None:
    months = pd.period_range("2020-01", periods=8, freq="M").astype(str)
    frame = pd.DataFrame({"month": np.repeat(months, 2)})

    splits = list(expanding_month_splits(frame, initial_months=3))

    assert len(splits) == 5
    for train_index, test_index in splits:
        assert frame.loc[train_index, "month"].max() < frame.loc[test_index, "month"].min()
        assert frame.loc[test_index, "month"].nunique() == 1


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["2024-01"] * 3 + ["2024-02"] * 2,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-02-01", "2024-02-02"]
            ),
            "trading_day": [1, 2, 3, 1, 2],
            "days_in_month": [3, 3, 3, 2, 2],
            "near_probability": [0.2, 0.8, 0.9, 0.1, 0.2],
            "adjusted_open": [10.0, 9.0, 8.5, 20.0, 21.0],
            "reservation_adjusted": [9.0, 9.1, 8.6, 18.0, 19.0],
            "regret": [0.2, 0.05, 0.0, 0.0, 0.05],
            "open": [10.0, 9.0, 8.5, 20.0, 21.0],
        }
    )


def test_purchase_policy_triggers_once_or_forces_last_day() -> None:
    selected = select_monthly_purchases(_predictions(), probability_threshold=0.5)

    assert selected["date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-02")]
    assert selected["forced"].tolist() == [False, True]


def test_probability_only_ignores_reservation_price() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-03"] * 3,
            "date": pd.to_datetime(["2024-03-01", "2024-03-04", "2024-03-05"]),
            "trading_day": [1, 2, 3],
            "days_in_month": [3, 3, 3],
            "near_probability": [0.8, 0.2, 0.9],
            "adjusted_open": [10.0, 8.0, 7.0],
            "reservation_adjusted": [9.0, 9.0, 9.0],
        }
    )
    selected = select_monthly_purchases(frame, probability_threshold=0.5, use_reservation=False)
    assert selected["date"].tolist() == [pd.Timestamp("2024-03-01")]
    assert selected["forced"].tolist() == [False]


def test_reservation_only_ignores_probability() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-03"] * 3,
            "date": pd.to_datetime(["2024-03-01", "2024-03-04", "2024-03-05"]),
            "trading_day": [1, 2, 3],
            "days_in_month": [3, 3, 3],
            "near_probability": [0.1, 0.2, 0.9],
            "adjusted_open": [10.0, 8.0, 7.0],
            "reservation_adjusted": [9.0, 9.0, 9.0],
        }
    )
    selected = select_monthly_purchases(frame, probability_threshold=None, use_reservation=True)
    assert selected["date"].tolist() == [pd.Timestamp("2024-03-04")]
    assert selected["forced"].tolist() == [False]


def test_deadline_buys_fallback_day_when_no_signal() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-04"] * 6,
            "date": pd.to_datetime([f"2024-04-0{day}" for day in range(1, 7)]),
            "trading_day": [1, 2, 3, 4, 5, 6],
            "days_in_month": [6] * 6,
            "near_probability": [0.1] * 6,
            "adjusted_open": [10.0] * 6,
            "reservation_adjusted": [9.0] * 6,
        }
    )
    selected = select_monthly_purchases(frame, probability_threshold=0.5, fallback_trading_day=5)
    assert selected["trading_day"].tolist() == [5]
    assert selected["forced"].tolist() == [True]


def test_deadline_still_takes_early_signal() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-04"] * 6,
            "date": pd.to_datetime([f"2024-04-0{day}" for day in range(1, 7)]),
            "trading_day": [1, 2, 3, 4, 5, 6],
            "days_in_month": [6] * 6,
            "near_probability": [0.1, 0.8, 0.9, 0.9, 0.9, 0.9],
            "adjusted_open": [10.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            "reservation_adjusted": [9.0] * 6,
        }
    )
    selected = select_monthly_purchases(frame, probability_threshold=0.5, fallback_trading_day=5)
    assert selected["trading_day"].tolist() == [2]
    assert selected["forced"].tolist() == [False]


def test_deadline_ignores_signal_after_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-04"] * 6,
            "date": pd.to_datetime([f"2024-04-0{day}" for day in range(1, 7)]),
            "trading_day": [1, 2, 3, 4, 5, 6],
            "days_in_month": [6] * 6,
            "near_probability": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            "adjusted_open": [10.0, 10.0, 10.0, 10.0, 10.0, 8.0],
            "reservation_adjusted": [9.0] * 6,
        }
    )
    selected = select_monthly_purchases(frame, probability_threshold=0.5, fallback_trading_day=5)
    assert selected["trading_day"].tolist() == [5]
    assert selected["forced"].tolist() == [True]


def test_deadline_caps_at_last_day_in_short_month() -> None:
    selected = select_monthly_purchases(
        _predictions(), probability_threshold=0.5, fallback_trading_day=5
    )
    assert selected["date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-02")]
    assert selected["forced"].tolist() == [False, True]


def test_deadline_with_probability_only_stops_at_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-04"] * 6,
            "date": pd.to_datetime([f"2024-04-0{day}" for day in range(1, 7)]),
            "trading_day": [1, 2, 3, 4, 5, 6],
            "days_in_month": [6] * 6,
            "near_probability": [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            "adjusted_open": [8.0] * 6,
            "reservation_adjusted": [9.0] * 6,
        }
    )
    selected = select_monthly_purchases(
        frame, probability_threshold=0.5, use_reservation=False, fallback_trading_day=5
    )
    assert selected["trading_day"].tolist() == [5]
    assert selected["forced"].tolist() == [True]


def test_purchase_policy_rejects_invalid_deadline_and_reservation_flag() -> None:
    frame = _predictions()
    with pytest.raises(ValueError, match="正整數"):
        select_monthly_purchases(frame, fallback_trading_day=0)
    with pytest.raises(TypeError, match="use_reservation"):
        select_monthly_purchases(frame, use_reservation=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fallback_trading_day"):
        select_monthly_purchases(frame, fallback_trading_day=True)
