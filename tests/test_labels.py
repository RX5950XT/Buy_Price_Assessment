import math

import pandas as pd
import pytest

from buy_price_assessment.labels import add_monthly_labels, monthly_oracle_table


def test_monthly_oracle_uses_adjusted_open_and_earliest_tie() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-02-01", "2024-02-02"]
            ),
            "open": [10.0, 9.0, 9.0, 20.0, 19.0],
            "adjusted_open": [10.0, 9.0, 9.0, 20.0, 19.0],
            "adjusted_close": [10.0, 9.1, 9.2, 20.1, 19.1],
        }
    )

    labeled = add_monthly_labels(daily, complete_through="2024-02")
    oracle = monthly_oracle_table(labeled)

    assert oracle.loc[oracle["month"] == "2024-01", "oracle_date"].item() == pd.Timestamp(
        "2024-01-03"
    )
    jan_first = labeled.loc[labeled["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert jan_first["regret"] == pytest.approx(10.0 / 9.0 - 1)
    assert not bool(jan_first["near_optimal"])
    jan_last = labeled.loc[labeled["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert jan_last["remaining_min_adjusted_open"] == pytest.approx(9.0)
    assert jan_last["remaining_min_log_ratio"] == pytest.approx(math.log(9.0 / 9.1))
    assert labeled.groupby("month")["trading_day"].max().to_dict() == {
        "2024-01": 3,
        "2024-02": 2,
    }


def test_labels_exclude_incomplete_months() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-02-01"]),
            "open": [10.0, 11.0],
            "adjusted_open": [10.0, 11.0],
            "adjusted_close": [10.0, 11.0],
        }
    )
    result = add_monthly_labels(daily, complete_through="2024-01")
    assert result["month"].unique().tolist() == ["2024-01"]
