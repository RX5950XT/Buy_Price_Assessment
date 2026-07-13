import numpy as np
import pandas as pd

from buy_price_assessment.modeling import expanding_month_splits, select_monthly_purchases


def test_expanding_splits_never_train_on_test_or_future_months() -> None:
    months = pd.period_range("2020-01", periods=8, freq="M").astype(str)
    frame = pd.DataFrame({"month": np.repeat(months, 2)})

    splits = list(expanding_month_splits(frame, initial_months=3))

    assert len(splits) == 5
    for train_index, test_index in splits:
        assert frame.loc[train_index, "month"].max() < frame.loc[test_index, "month"].min()
        assert frame.loc[test_index, "month"].nunique() == 1


def test_purchase_policy_triggers_once_or_forces_last_day() -> None:
    predictions = pd.DataFrame(
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

    selected = select_monthly_purchases(predictions, probability_threshold=0.5)

    assert selected["date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-02")]
    assert selected["forced"].tolist() == [False, True]
