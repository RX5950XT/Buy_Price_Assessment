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
    assert last_complete_month(date(2026, 7, 9)) == "2026-06"
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
    assert results["policy_ablation"]["prob_and_res"]["mean_regret"] == pytest.approx(
        results["strategy_metrics"]["all"]["mean_regret"]
    )
    assert set(results["policy_ablation"]) == {
        "prob_and_res",
        "prob_only",
        "res_only",
        "prob_and_res_deadline5",
        "prob_only_deadline5",
        "res_only_deadline5",
    }
    assert (tmp_path / "0050_buy_point_analysis.md").exists()
    report = (tmp_path / "0050_buy_point_analysis.md").read_text(encoding="utf-8")
    assert "決策規則拆解" in report
    pngs = {path.name for path in (tmp_path / "figures").glob("*.png")}
    assert {
        "price_history.png",
        "oracle_day_distribution.png",
        "oracle_feature_profile.png",
        "strategy_comparison.png",
        "policy_ablation.png",
    } <= pngs
    assert results.get("lead_rules"), "缺少領先資料或未接到領先規則"
    assert "外部領先規則" in report
    assert "lead_rules.png" in pngs
    assert set(results["lead_rules"]) == {
        "tsm_neg_or_day5",
        "sox_neg_or_day5",
        "fx_pause_or_day5",
        "tsm_dump1pct_or_day5",
        "tsm_buy_unless_adverse",
        "fx_single_pause_or_day5",
    }
    required = {
        "mean_regret",
        "forced_rate",
        "mean_trading_day",
        "improvement_bps",
        "ci95_bps",
        "holdout_mean_regret",
        "holdout_improvement_bps",
        "holdout_ci95_bps",
        "signal_rate",
    }
    for item in results["lead_rules"].values():
        assert required <= set(item)
    coverage = results["lead_coverage"]
    assert {
        "tsm_dump_rate",
        "tsm_dump_1pct_rate",
        "tsm_adverse_rate",
        "fx_pause_rate",
        "fx_single_pause_rate",
    } <= set(coverage)
    assert "1%" in report
    assert "單日貶值" in report
    assert "強制率" in report
