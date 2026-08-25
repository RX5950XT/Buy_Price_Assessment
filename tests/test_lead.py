import pandas as pd
import pytest

from buy_price_assessment.evaluation import select_first_true_or_deadline
from buy_price_assessment.lead import (
    OVERNIGHT_DUMP_THRESHOLD,
    align_prior_session,
    attach_lead_features,
)


def _us_prices(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "adj_close": closes})


def _fx_prices(dates: list[str], rates: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "usd_twd": rates})


def _taiwan(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates)})


def _with_month(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    result = frame.copy()
    n = len(result)
    result["month"] = month
    result["trading_day"] = list(range(1, n + 1))
    result["days_in_month"] = n
    result["regret"] = 0.0
    result["adjusted_open"] = 100.0
    return result


def test_overnight_dump_threshold_is_pre_specified_one_percent() -> None:
    assert OVERNIGHT_DUMP_THRESHOLD == -0.01


def test_align_prior_session_uses_friday_for_monday() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-08", "2024-03-11"]),
            "ret": [-0.02, 0.01],
        }
    )
    target = pd.Series(pd.to_datetime(["2024-03-11", "2024-03-12"]))
    aligned = align_prior_session(source, target, "ret")
    assert aligned.tolist() == pytest.approx([-0.02, 0.01])


def test_align_prior_session_never_uses_same_calendar_day() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-11", "2024-03-12"]),
            "ret": [0.05, -0.09],
        }
    )
    target = pd.Series(pd.to_datetime(["2024-03-12"]))
    aligned = align_prior_session(source, target, "ret")
    assert aligned.tolist() == pytest.approx([0.05])


def test_attach_lead_features_marks_dump_and_fx_streak() -> None:
    taiwan = _taiwan(["2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15"])
    us = _us_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [100.0, 99.0, 98.0, 100.0, 101.0, 100.0],
    )
    fx = _fx_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [31.0, 31.2, 31.4, 31.6, 31.5, 31.4],
    )
    result = attach_lead_features(taiwan, tsm=us, sox=us, fx=fx)
    assert result["tsm_dump"].tolist() == [True, True, False, False, True]
    assert result["fx_not_depreciating"].tolist() == [True, True, False, True, True]


def test_small_negative_overnight_is_not_rare_dump_and_forces_day_five() -> None:
    taiwan = _with_month(
        _taiwan(["2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15"]),
        "2024-03",
    )
    us = _us_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [100.0, 99.5, 99.0, 98.5, 98.0, 97.5],
    )
    fx = _fx_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
    )
    featured = attach_lead_features(taiwan, tsm=us, sox=us, fx=fx)
    assert featured["tsm_dump"].all()
    assert not featured["tsm_dump_1pct"].any()
    selected = select_first_true_or_deadline(
        featured, column="tsm_dump_1pct", fallback_trading_day=5
    )
    assert selected["trading_day"].tolist() == [5]
    assert selected["forced"].tolist() == [True]


def test_one_percent_overnight_dump_buys_on_first_rare_drop() -> None:
    taiwan = _with_month(
        _taiwan(["2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15"]),
        "2024-03",
    )
    us = _us_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [100.0, 99.5, 99.0, 97.0, 97.0, 97.0],
    )
    fx = _fx_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
    )
    featured = attach_lead_features(taiwan, tsm=us, sox=us, fx=fx)
    assert featured["tsm_dump_1pct"].tolist() == [False, False, True, False, False]
    selected = select_first_true_or_deadline(
        featured, column="tsm_dump_1pct", fallback_trading_day=5
    )
    assert selected["trading_day"].tolist() == [3]
    assert selected["forced"].tolist() == [False]


def test_pause_rule_skips_day1_when_overnight_down_else_buys_day1() -> None:
    adverse_month = _with_month(
        _taiwan(["2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15"]),
        "2024-03",
    )
    calm_month = _with_month(
        _taiwan(["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04", "2024-04-05"]),
        "2024-04",
    )
    taiwan = pd.concat([adverse_month, calm_month], ignore_index=True)
    us = _us_prices(
        [
            "2024-03-07",
            "2024-03-08",
            "2024-03-11",
            "2024-03-12",
            "2024-03-13",
            "2024-03-14",
            "2024-03-28",
            "2024-03-29",
            "2024-04-01",
            "2024-04-02",
            "2024-04-03",
            "2024-04-04",
        ],
        [100.0, 98.0, 100.0, 101.0, 101.0, 101.0, 100.0, 102.0, 103.0, 103.0, 103.0, 103.0],
    )
    fx = _fx_prices(
        [
            "2024-03-07",
            "2024-03-08",
            "2024-03-11",
            "2024-03-12",
            "2024-03-13",
            "2024-03-14",
            "2024-03-28",
            "2024-03-29",
            "2024-04-01",
            "2024-04-02",
            "2024-04-03",
            "2024-04-04",
        ],
        [31.0] * 12,
    )
    featured = attach_lead_features(taiwan, tsm=us, sox=us, fx=fx)
    selected = select_first_true_or_deadline(
        featured, column="tsm_not_dump", fallback_trading_day=5
    )
    assert selected["trading_day"].tolist() == [2, 1]
    assert selected["forced"].tolist() == [False, False]


def test_single_day_fx_pause_differs_from_three_day_streak_on_short_up_path() -> None:
    taiwan = _with_month(
        _taiwan(["2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14", "2024-03-15"]),
        "2024-03",
    )
    us = _us_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    )
    fx = _fx_prices(
        ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13", "2024-03-14"],
        [31.0, 31.1, 31.2, 31.0, 31.0, 31.0],
    )
    featured = attach_lead_features(taiwan, tsm=us, sox=us, fx=fx)
    three_day = select_first_true_or_deadline(
        featured, column="fx_not_depreciating", fallback_trading_day=5
    )
    one_day = select_first_true_or_deadline(featured, column="fx_not_up", fallback_trading_day=5)
    assert three_day["trading_day"].tolist() == [1]
    assert one_day["trading_day"].tolist() == [3]
    assert three_day["forced"].tolist() == [False]
    assert one_day["forced"].tolist() == [False]
