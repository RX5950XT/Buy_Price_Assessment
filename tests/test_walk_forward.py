import numpy as np
import pandas as pd

from buy_price_assessment.walk_forward import WalkForwardConfig, walk_forward_predictions


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_number, month in enumerate(pd.period_range("2020-01", periods=7, freq="M")):
        dates = pd.bdate_range(month.start_time, periods=4)
        opens = np.array([10.0, 9.5, 9.0, 9.8]) + month_number
        minimum = float(opens.min())
        for day, (date, adjusted_open) in enumerate(zip(dates, opens, strict=True), start=1):
            rows.append(
                {
                    "month": str(month),
                    "date": date,
                    "trading_day": day,
                    "days_in_month": 4,
                    "open": adjusted_open,
                    "adjusted_open": adjusted_open,
                    "adjusted_close": adjusted_open + 0.1,
                    "previous_adjusted_close": adjusted_open + 0.2,
                    "remaining_min_log_ratio": np.log(
                        min(opens[day - 1 :]) / (adjusted_open + 0.2)
                    ),
                    "near_optimal": adjusted_open / minimum - 1 <= 0.005,
                    "regret": adjusted_open / minimum - 1,
                    "signal": -abs(day - 3),
                    "month_progress": day / 4,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_predictions_only_use_prior_months() -> None:
    result = walk_forward_predictions(
        _training_frame(),
        feature_columns=("signal", "month_progress"),
        config=WalkForwardConfig(initial_months=3, quantile_estimators=10),
    )

    assert result["month"].nunique() == 4
    assert result["month"].min() == "2020-04"
    assert (result["training_through"] < result["month"]).all()
    assert result["near_probability"].between(0, 1).all()
    assert (result["reservation_adjusted"] > 0).all()
