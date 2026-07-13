import numpy as np
import pandas as pd
import pytest

from buy_price_assessment.features import FEATURE_GROUPS, build_features


def _daily_fixture(rows: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    close = pd.Series(np.arange(100.0, 100.0 + rows))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.arange(1_000, 1_000 + rows),
            "trading_value": np.arange(100_000, 100_000 + rows),
            "total_return_index": close,
            "nav": close - 0.1,
            "cash_dividend": 0.0,
            "margin_balance": np.arange(500, 500 + rows),
            "short_balance": np.arange(50, 50 + rows),
            "institutional_net": np.arange(rows),
            "institutional_available": True,
            "outstanding_units": np.arange(10_000, 10_000 + rows),
        }
    )


def test_features_are_lagged_to_previous_close() -> None:
    source = _daily_fixture()
    result = build_features(source)

    current = result.iloc[10]
    expected_prior_return = source["total_return_index"].pct_change().iloc[9]
    assert current["ret_1"] == pytest.approx(expected_prior_return)
    assert current["premium_discount"] == pytest.approx(
        source.loc[9, "close"] / source.loc[9, "nav"] - 1
    )


def test_feature_groups_are_nonempty_and_known() -> None:
    assert set(FEATURE_GROUPS) == {"technical", "valuation", "chip", "calendar"}
    assert all(FEATURE_GROUPS[group] for group in FEATURE_GROUPS)
    assert build_features(_daily_fixture()).columns.is_unique


def test_calendar_features_do_not_depend_on_future_trading_days() -> None:
    source = _daily_fixture()
    cutoff = 10

    full = build_features(source).iloc[:cutoff]
    truncated = build_features(source.iloc[:cutoff])

    assert truncated["month_progress"].tolist() == pytest.approx(full["month_progress"].tolist())


def test_volume_feature_is_comparable_across_split() -> None:
    base = _daily_fixture()
    base["split_ratio"] = 1.0
    split = base.copy()
    split.loc[60, "split_ratio"] = 4.0
    split.loc[60:, "volume"] *= 4

    base_feature = build_features(base)["volume_z_20"]
    split_feature = build_features(split)["volume_z_20"]

    pd.testing.assert_series_equal(base_feature, split_feature)
