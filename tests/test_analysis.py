from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from buy_price_assessment.analysis import (
    _cached_prediction_is_valid,
    add_action_adjusted_open_gap,
    last_complete_month,
    run_analysis,
)


def test_last_complete_month_never_labels_current_month() -> None:
    assert last_complete_month(date(2026, 7, 10)) == "2026-06"
    assert last_complete_month(date(2026, 1, 1)) == "2025-12"


def test_action_adjusted_open_gap_removes_split_and_dividend_mechanics() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [100.0, 50.0, 48.0],
            "close": [100.0, 50.0, 48.0],
            "split_ratio": [1.0, 2.0, 1.0],
            "cash_dividend": [0.0, 0.0, 2.0],
        }
    )
    result = add_action_adjusted_open_gap(frame)
    assert result["open_gap_vs_action_ref"].iloc[1:].tolist() == pytest.approx([0.0, 0.0])


def test_prediction_cache_requires_matching_analysis_signature(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    dates = pd.Series(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    pd.DataFrame(
        {
            "date": dates,
            "feature_set": ["all", "all"],
            "analysis_signature": ["current", "current"],
        }
    ).to_csv(path, index=False)

    assert _cached_prediction_is_valid(path, dates, "all", "current")
    assert not _cached_prediction_is_valid(path, dates, "all", "changed")


def test_full_analysis_reuses_verified_predictions_and_renders_report(tmp_path: Path) -> None:
    results = run_analysis(
        daily_path=Path("data/processed/0050_daily.csv"),
        processed_dir=Path("data/processed"),
        reports_dir=tmp_path,
        as_of=date(2026, 7, 10),
        reuse_existing_predictions=True,
    )
    assert results["data_quality"]["rows"] == 5665
    assert results["oracle_distribution"]["months"] == 276
    assert results["primary"]["day1_mean_regret"] < results["primary"]["model_mean_regret"]
    assert (tmp_path / "0050_buy_point_analysis.md").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
