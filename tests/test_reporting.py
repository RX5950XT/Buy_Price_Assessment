from buy_price_assessment.reporting import render_report


def test_render_report_leads_with_evidence_backed_conclusion() -> None:
    report = render_report(
        {
            "data_quality": {"rows": 5665, "start": "2003-06-30", "end": "2026-07-09"},
            "oracle_distribution": {
                "months": 276,
                "mode_day": 1,
                "median_day": 6.0,
                "first_day_rate": 0.268,
                "first_5_rate": 0.482,
            },
            "primary": {
                "day1_mean_regret": 0.032,
                "day1_within_rate": 0.389,
                "model_mean_regret": 0.045,
                "model_improvement_bps": -126.0,
                "model_ci95_bps": [-205.0, -53.0],
            },
            "holdout": {
                "months": 36,
                "day1_mean_regret": 0.0258,
                "day1_within_rate": 0.528,
                "model_mean_regret": 0.0591,
            },
            "oracle_ranges": {
                "ret_5": {"median": -0.0159},
                "ma_gap_20": {"median": -0.0142},
                "rsi_14": {"median": 41.1},
                "drawdown_60": {"median": -0.052},
                "open_gap_vs_action_ref": {"q25": -0.0074, "median": -0.0013, "q75": 0.0009},
                "premium_discount": {"median": 0.00033},
            },
            "day1_vs_day5": {"improvement_bps": 18.0, "ci95_bps": [-38.0, 73.0]},
        }
    )
    assert "第 1 個交易日" in report
    assert "沒有可靠證據" in report
    assert "5,665" in report
    assert "41.1" in report
