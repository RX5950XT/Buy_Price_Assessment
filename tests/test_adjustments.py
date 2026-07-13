import pandas as pd
import pytest

from buy_price_assessment.adjustments import build_adjusted_prices


def test_total_return_adjustment_handles_split_and_dividend() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [100.0, 50.0, 48.0],
            "high": [101.0, 51.0, 49.0],
            "low": [99.0, 49.0, 47.0],
            "close": [100.0, 50.0, 48.0],
        }
    )
    dividends = pd.DataFrame({"date": pd.to_datetime(["2025-01-03"]), "cash_dividend": [2.0]})
    splits = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "split_ratio": [2.0]})

    result = build_adjusted_prices(prices, dividends, splits)

    assert result["gross_return"].iloc[1:].tolist() == pytest.approx([1.0, 1.0])
    assert result["total_return_index"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert result["adjusted_close"].tolist() == pytest.approx([48.0, 48.0, 48.0])


def test_adjustment_rejects_duplicate_dates_and_invalid_prices() -> None:
    duplicate = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="重複"):
        build_adjusted_prices(duplicate, pd.DataFrame(), pd.DataFrame())
