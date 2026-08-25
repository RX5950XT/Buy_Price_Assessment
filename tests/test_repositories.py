import pandas as pd
import pytest

from buy_price_assessment.clients import DataSourceError
from buy_price_assessment.repositories import (
    parse_finmind_institutional,
    parse_finmind_margin,
    parse_finmind_prices,
    parse_twse_monthly_closes,
    parse_us_prices,
    parse_usd_twd,
    parse_yuanta_nav,
)


def test_parse_us_prices_keeps_adjusted_close() -> None:
    result = parse_us_prices(
        [
            {
                "date": "2003-06-02",
                "stock_id": "TSM",
                "Adj_Close": 3.76,
                "Close": 7.61,
            }
        ],
        source="FinMind USStockPrice TSM",
    )
    assert result.loc[0, "date"] == pd.Timestamp("2003-06-02")
    assert result.loc[0, "adj_close"] == pytest.approx(3.76)


def test_parse_usd_twd_drops_invalid_spot_quotes() -> None:
    result = parse_usd_twd(
        [
            {
                "date": "2006-01-02",
                "currency": "USD",
                "spot_buy": -99,
                "spot_sell": -99,
            },
            {
                "date": "2006-01-03",
                "currency": "USD",
                "spot_buy": 32.595,
                "spot_sell": 32.695,
            },
        ]
    )
    assert result["date"].tolist() == [pd.Timestamp("2006-01-03")]
    assert result.loc[0, "usd_twd"] == pytest.approx(32.645)


def test_parse_finmind_prices_normalizes_schema() -> None:
    rows = [
        {
            "date": "2003-06-30",
            "stock_id": "0050",
            "Trading_Volume": 100,
            "Trading_money": 3_700,
            "open": 37.1,
            "max": 37.4,
            "min": 36.9,
            "close": 37.08,
            "spread": 0.12,
            "Trading_turnover": 20,
        }
    ]
    result = parse_finmind_prices(rows)
    assert result.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "price_change",
        "transactions",
    ]
    assert result.loc[0, "date"] == pd.Timestamp("2003-06-30")
    assert result.loc[0, "high"] == pytest.approx(37.4)


def test_parse_twse_monthly_closes_skips_month_average() -> None:
    payload = {
        "stat": "OK",
        "data": [[" 92/06/30", "37.08"], ["月平均收盤價", "37.08"]],
    }
    result = parse_twse_monthly_closes(payload)
    assert result.to_dict("records") == [
        {"date": pd.Timestamp("2003-06-30"), "official_close": 37.08}
    ]


def test_parse_twse_monthly_closes_rejects_api_error() -> None:
    with pytest.raises(DataSourceError, match="TWSE"):
        parse_twse_monthly_closes({"stat": "查無資料"})


def test_parse_yuanta_nav_validates_result_and_sorts() -> None:
    payload = {
        "ResultCode": 0,
        "Data": [
            {
                "UPDATE_T": "2025-01-03T00:00:00",
                "NOW_NAV": 50.1,
                "NOW_PRICE": 50.2,
                "FUND_SIZE": 1000,
                "OS_UNIT": 20,
            },
            {
                "UPDATE_T": "2025-01-02T00:00:00",
                "NOW_NAV": 49.9,
                "NOW_PRICE": 50.0,
                "FUND_SIZE": 980,
                "OS_UNIT": 19,
            },
            {
                "UPDATE_T": "2025-01-01T00:00:00",
                "NOW_NAV": 0,
                "NOW_PRICE": 0,
                "FUND_SIZE": 0,
                "OS_UNIT": 0,
            },
        ],
    }
    result = parse_yuanta_nav(payload)
    assert result["date"].tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert result.loc[0, "nav"] == pytest.approx(49.9)


def test_parse_chip_data_aggregates_institutions_and_margin() -> None:
    institutional = parse_finmind_institutional(
        [
            {"date": "2025-01-02", "name": "Foreign_Investor", "buy": 100, "sell": 40},
            {"date": "2025-01-02", "name": "Investment_Trust", "buy": 10, "sell": 20},
            {"date": "2025-01-02", "name": "Dealer_self", "buy": 15, "sell": 5},
            {"date": "2025-01-02", "name": "Dealer", "buy": 9, "sell": 2},
            {
                "date": "2025-01-02",
                "name": "Foreign_Dealer_Self",
                "buy": 1_000,
                "sell": 0,
            },
        ]
    )
    assert institutional.loc[0, "institutional_net"] == pytest.approx(67)
    assert institutional.loc[0, "foreign_net"] == pytest.approx(60)
    assert institutional.loc[0, "dealer_net"] == pytest.approx(17)

    margin = parse_finmind_margin(
        [
            {
                "date": "2025-01-02",
                "MarginPurchaseTodayBalance": 123,
                "ShortSaleTodayBalance": 45,
            }
        ]
    )
    assert margin.loc[0, "margin_balance"] == 123
    assert margin.loc[0, "short_balance"] == 45
