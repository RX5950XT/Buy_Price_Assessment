import pandas as pd
import pytest

from buy_price_assessment.ingestion import assemble_daily_data


def _frames() -> tuple[pd.DataFrame, ...]:
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    prices = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 50.0, 48.0],
            "high": [101.0, 51.0, 49.0],
            "low": [99.0, 49.0, 47.0],
            "close": [100.0, 50.0, 48.0],
            "volume": [1000, 1100, 1200],
            "trading_value": [100000, 55000, 57600],
            "price_change": [0.0, 0.0, -2.0],
            "transactions": [100, 110, 120],
        }
    )
    dividends = pd.DataFrame({"date": dates[[2]], "cash_dividend": [2.0]})
    splits = pd.DataFrame({"date": dates[[1]], "split_ratio": [2.0]})
    nav = pd.DataFrame(
        {
            "date": dates,
            "nav": [99.5, 49.8, 47.9],
            "issuer_market_price": [100.0, 50.0, 48.0],
            "fund_size": [1000, 1000, 1000],
            "outstanding_units": [10, 20, 20],
        }
    )
    margin = pd.DataFrame({"date": dates, "margin_balance": [1, 2, 3], "short_balance": [0, 1, 1]})
    institutions = pd.DataFrame(
        {
            "date": dates,
            "institutional_net": [1, 2, 3],
            "foreign_net": [1, 2, 3],
            "trust_net": [0, 0, 0],
            "dealer_net": [0, 0, 0],
        }
    )
    official = pd.DataFrame({"date": dates, "official_close": [100.0, 50.0, 48.0]})
    return prices, dividends, splits, nav, margin, institutions, official


def test_assemble_daily_data_adjusts_and_validates_sources() -> None:
    result = assemble_daily_data(*_frames())
    assert result["total_return_index"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert result["close_difference"].max() == pytest.approx(0.0)
    assert result["institutional_available"].all()
    assert result["distribution_yield_ttm"].iloc[-1] == pytest.approx(2.0 / 48.0)


def test_assemble_daily_data_fails_on_cross_source_price_mismatch() -> None:
    frames = list(_frames())
    frames[-1] = frames[-1].assign(official_close=[100.0, 50.0, 49.0])
    with pytest.raises(ValueError, match="官方收盤價"):
        assemble_daily_data(*frames)


def test_assemble_daily_data_marks_unavailable_optional_source() -> None:
    frames = list(_frames())
    frames[5] = pd.DataFrame()
    result = assemble_daily_data(*frames)
    assert not result["institutional_available"].any()
    assert result["institutional_net"].eq(0).all()
