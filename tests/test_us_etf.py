import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from buy_price_assessment.analysis import (
    VT_FEATURE_SETS,
    _prepare_frames,
    last_complete_month,
)
from buy_price_assessment.cli import build_parser
from buy_price_assessment.clients import DataSourceError
from buy_price_assessment.ingestion import assemble_us_etf_daily, infer_us_actions
from buy_price_assessment.lead import align_prior_session
from buy_price_assessment.repositories import parse_us_ohlc


def test_parse_us_ohlc_keeps_raw_and_adjusted_close() -> None:
    result = parse_us_ohlc(
        [
            {
                "date": "2008-06-26",
                "stock_id": "VT",
                "Open": 49.92,
                "High": 49.99,
                "Low": 49.34,
                "Close": 49.58,
                "Adj_Close": 33.87,
                "Volume": 15800,
            }
        ],
        source="FinMind USStockPrice VT",
    )
    assert result.loc[0, "date"] == pd.Timestamp("2008-06-26")
    assert result.loc[0, "open"] == pytest.approx(49.92)
    assert result.loc[0, "close"] == pytest.approx(49.58)
    assert result.loc[0, "adj_close"] == pytest.approx(33.87)
    assert result.loc[0, "volume"] == 15800


def test_parse_us_ohlc_rejects_missing_adj_close() -> None:
    with pytest.raises(DataSourceError, match="Adj_Close"):
        parse_us_ohlc(
            [{"date": "2008-06-26", "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}],
            source="FinMind USStockPrice VT",
        )


def test_infer_us_actions_separates_dividend_from_split() -> None:
    close = pd.Series([100.0, 100.0, 98.0, 50.0])
    adj_close = pd.Series([49.0, 49.0, 49.0, 50.0])
    cash, split = infer_us_actions(close, adj_close)
    assert cash.tolist() == pytest.approx([0.0, 0.0, 2.0, 0.0], abs=0.05)
    assert split.tolist() == pytest.approx([1.0, 1.0, 1.0, 2.0])


def test_assemble_us_etf_daily_uses_vendor_factor_for_adjusted_open() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2008-07-01", "2008-07-02", "2008-07-03"]),
            "open": [100.0, 100.0, 98.0],
            "high": [101.0, 101.0, 99.0],
            "low": [99.0, 99.0, 97.0],
            "close": [100.0, 100.0, 98.0],
            "adj_close": [98.0, 98.0, 98.0],
            "volume": [1000, 1100, 1200],
        }
    )
    result = assemble_us_etf_daily(prices)
    assert result["adjusted_open"].tolist() == pytest.approx([98.0, 98.0, 98.0])
    assert result["adjusted_close"].tolist() == pytest.approx([98.0, 98.0, 98.0])
    assert result["total_return_index"].iloc[0] == pytest.approx(100.0)
    assert result["nav"].tolist() == result["close"].tolist()
    assert not result["institutional_available"].any()


def test_prepare_frames_starts_vt_from_first_complete_month() -> None:
    dates = pd.to_datetime(["2008-06-26", "2008-06-27", "2008-07-01", "2008-07-02", "2008-07-03"])
    close = pd.Series([50.0, 51.0, 52.0, 51.5, 53.0])
    daily = assemble_us_etf_daily(
        pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "adj_close": close,
                "volume": [1000, 1000, 1000, 1000, 1000],
            }
        )
    )
    _features, labeled, oracle = _prepare_frames(
        daily,
        complete_through="2008-07",
        first_complete_month="2008-07",
    )
    assert labeled["month"].min() == "2008-07"
    assert oracle["month"].tolist() == ["2008-07"]


def test_vt_feature_sets_are_technical_and_calendar_only() -> None:
    columns = VT_FEATURE_SETS["technical_calendar"]
    assert "ret_1" in columns
    assert "trading_day" in columns
    assert "premium_discount" not in columns
    assert "institutional_net_ratio" not in columns
    assert list(VT_FEATURE_SETS) == ["technical_calendar"]


def test_vt_open_decision_does_not_use_same_day_us_close() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-11", "2024-03-12"]),
            "ret": [0.05, -0.09],
        }
    )
    target = pd.Series(pd.to_datetime(["2024-03-12"]))
    aligned = align_prior_session(source, target, "ret")
    assert aligned.tolist() == pytest.approx([0.05])


def test_cli_accepts_vt_symbol() -> None:
    parser = build_parser()
    args = parser.parse_args(["analyze", "--symbol", "VT", "--force-models"])
    assert args.symbol == "VT"
    assert args.force_models is True
    fetch = parser.parse_args(["fetch", "--symbol", "VT"])
    assert fetch.symbol == "VT"


def test_cli_fetch_vt_writes_separate_daily_path(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from buy_price_assessment import cli

    daily = pd.DataFrame({"date": pd.to_datetime(["2008-06-26"]), "close": [49.58]})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "download_us_etf_data", lambda **_: daily)
    fetched = cli._fetch(False, symbol="VT")
    assert fetched["path"] == "data/processed/VT_daily.csv"
    assert (tmp_path / "data/processed/VT_daily.csv").exists()
    assert not (tmp_path / "data/processed/0050_daily.csv").exists()


def test_cli_rejects_twse_validation_for_vt() -> None:
    from buy_price_assessment import cli

    with pytest.raises(ValueError, match="TWSE"):
        cli._fetch(True, symbol="VT")


def test_vt_analysis_fails_closed_without_lead_files(tmp_path: Path) -> None:
    from buy_price_assessment.analysis import run_vt_analysis

    with pytest.raises(ValueError, match="領先序列"):
        run_vt_analysis(
            daily_path=Path("data/processed/VT_daily.csv"),
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
            raw_dir=tmp_path / "raw",
            reuse_existing_predictions=True,
        )


def test_last_complete_month_still_uses_data_end() -> None:
    assert last_complete_month(date(2026, 7, 10)) == "2026-06"


def test_cli_analyze_vt_does_not_overwrite_0050_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from buy_price_assessment import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "run_vt_analysis",
        lambda **_: {
            "oracle_distribution": {"months": 216},
            "primary": {"day1_mean_regret": 0.02, "model_mean_regret": 0.03},
            "meta": {"report_path": "reports/VT_buy_point_analysis.md"},
        },
    )
    analyzed = cli._analyze(False, symbol="VT")
    assert analyzed["report"] == "reports/VT_buy_point_analysis.md"
    assert json.dumps(analyzed)
