import pandas as pd
import pytest

from buy_price_assessment.evaluation import (
    moving_block_bootstrap_ci,
    select_first_true_or_deadline,
    select_fixed_day,
    strategy_metrics,
)


def _labeled() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["2024-01"] * 3 + ["2024-02"] * 2,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-02-01", "2024-02-02"]
            ),
            "trading_day": [1, 2, 3, 1, 2],
            "days_in_month": [3, 3, 3, 2, 2],
            "regret": [0.1, 0.0, 0.2, 0.05, 0.0],
            "adjusted_open": [10.0, 9.0, 11.0, 20.0, 19.0],
            "near_optimal": [False, True, False, False, True],
        }
    )


def test_fixed_day_caps_at_last_trading_day() -> None:
    selected = select_fixed_day(_labeled(), trading_day=3)
    assert selected["date"].tolist() == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-02-02")]


def test_strategy_metrics_use_month_as_observation_unit() -> None:
    selected = select_fixed_day(_labeled(), trading_day=1)
    metrics = strategy_metrics(selected)
    assert metrics["months"] == 2
    assert metrics["mean_regret"] == pytest.approx(0.075)
    assert metrics["within_0_5pct_rate"] == pytest.approx(0.0)


def test_pause_flag_skips_day1_when_adverse_else_buys_day1() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-01"] * 5 + ["2024-02"] * 5,
            "date": pd.to_datetime(
                [f"2024-01-0{day}" for day in range(1, 6)]
                + [f"2024-02-0{day}" for day in range(1, 6)]
            ),
            "trading_day": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "days_in_month": [5] * 10,
            "regret": [0.0] * 10,
            "adjusted_open": [10.0] * 10,
            "buy_unless_adverse": [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )
    selected = select_first_true_or_deadline(
        frame, column="buy_unless_adverse", fallback_trading_day=5
    )
    assert selected["trading_day"].tolist() == [2, 1]
    assert selected["forced"].tolist() == [False, False]


def test_first_true_or_deadline_ignores_signal_after_deadline() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-01"] * 8,
            "date": pd.to_datetime([f"2024-01-{day:02d}" for day in range(1, 9)]),
            "trading_day": list(range(1, 9)),
            "days_in_month": [31] * 8,
            "regret": [0.1, 0.2, 0.2, 0.2, 0.3, 0.0, 0.4, 0.5],
            "adjusted_open": [10.0] * 8,
            "signal": [False, False, False, False, False, True, True, False],
        }
    )
    selected = select_first_true_or_deadline(frame, column="signal", fallback_trading_day=5)
    assert selected["trading_day"].tolist() == [5]
    assert selected["forced"].tolist() == [True]


def test_first_true_or_deadline_buys_signal_then_caps_at_day_five() -> None:
    frame = pd.DataFrame(
        {
            "month": ["2024-01"] * 6 + ["2024-02"] * 3,
            "date": pd.to_datetime(
                [f"2024-01-0{day}" for day in range(1, 7)]
                + ["2024-02-01", "2024-02-02", "2024-02-03"]
            ),
            "trading_day": [1, 2, 3, 4, 5, 6, 1, 2, 3],
            "days_in_month": [6] * 6 + [3, 3, 3],
            "regret": [0.1, 0.0, 0.2, 0.2, 0.3, 0.4, 0.05, 0.0, 0.1],
            "adjusted_open": [10.0] * 6 + [20.0, 19.0, 21.0],
            "signal": [False, True, True, False, False, True, False, False, False],
        }
    )
    selected = select_first_true_or_deadline(frame, column="signal", fallback_trading_day=5)
    assert selected["trading_day"].tolist() == [2, 3]
    assert selected["forced"].tolist() == [False, True]


def test_moving_block_bootstrap_constant_difference_has_exact_interval() -> None:
    lower, upper = moving_block_bootstrap_ci(pd.Series([0.01] * 24), simulations=200)
    assert lower == pytest.approx(0.01)
    assert upper == pytest.approx(0.01)
