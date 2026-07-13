import pandas as pd
import pytest

from buy_price_assessment.evaluation import (
    moving_block_bootstrap_ci,
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


def test_moving_block_bootstrap_constant_difference_has_exact_interval() -> None:
    lower, upper = moving_block_bootstrap_ci(pd.Series([0.01] * 24), simulations=200)
    assert lower == pytest.approx(0.01)
    assert upper == pytest.approx(0.01)
