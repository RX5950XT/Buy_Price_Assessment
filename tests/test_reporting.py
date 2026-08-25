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
    assert "決策規則拆解" in report


def test_render_report_lists_all_pre_specified_policies() -> None:
    report = render_report(
        {
            "policy_ablation": {
                "prob_and_res": {
                    "mean_regret": 0.0436,
                    "forced_rate": 0.532,
                    "mean_trading_day": 15.6,
                    "improvement_bps": -116.0,
                    "ci95_bps": [-196.0, -39.0],
                    "holdout_mean_regret": 0.0579,
                    "holdout_improvement_bps": -321.0,
                    "holdout_ci95_bps": [-498.0, -151.0],
                },
                "prob_only": {
                    "mean_regret": 0.0333,
                    "forced_rate": 0.079,
                    "mean_trading_day": 4.0,
                    "improvement_bps": -12.0,
                    "ci95_bps": [-47.0, 21.0],
                    "holdout_mean_regret": 0.0358,
                    "holdout_improvement_bps": -100.0,
                    "holdout_ci95_bps": [-210.0, -10.0],
                },
                "prob_and_res_deadline5": {
                    "mean_regret": 0.0335,
                    "forced_rate": 0.819,
                    "mean_trading_day": 4.6,
                    "improvement_bps": -15.0,
                    "ci95_bps": [-65.0, 39.0],
                    "holdout_mean_regret": 0.035,
                    "holdout_improvement_bps": -92.0,
                    "holdout_ci95_bps": [-178.0, -6.0],
                },
                "prob_only_deadline5": {
                    "mean_regret": 0.0325,
                    "forced_rate": 0.181,
                    "mean_trading_day": 2.1,
                    "improvement_bps": -5.0,
                    "ci95_bps": [-27.0, 21.0],
                    "holdout_mean_regret": 0.0322,
                    "holdout_improvement_bps": -64.0,
                    "holdout_ci95_bps": [-115.0, -20.0],
                },
            }
        }
    )
    assert "機率＋保留價／月底" in report
    assert "僅機率／第5日截止" in report
    assert "沒有證據顯示改門檻或改截止日優於第 1 交易日" in report
    assert "拿掉保留價後" in report
    assert "點估計仍全為負" in report


def test_render_report_omits_holdout_all_negative_when_any_positive() -> None:
    report = render_report(
        {
            "oracle_distribution": {"first_5_rate": 0.482},
            "policy_ablation": {
                "prob_and_res": {
                    "mean_regret": 0.04,
                    "forced_rate": 0.5,
                    "mean_trading_day": 15.0,
                    "improvement_bps": -80.0,
                    "ci95_bps": [-100.0, -10.0],
                    "holdout_mean_regret": 0.03,
                    "holdout_improvement_bps": 20.0,
                    "holdout_ci95_bps": [-10.0, 40.0],
                },
                "prob_only": {
                    "mean_regret": 0.033,
                    "forced_rate": 0.08,
                    "mean_trading_day": 4.0,
                    "improvement_bps": -10.0,
                    "ci95_bps": [-40.0, 20.0],
                    "holdout_mean_regret": 0.03,
                    "holdout_improvement_bps": -10.0,
                    "holdout_ci95_bps": [-30.0, 10.0],
                },
                "prob_and_res_deadline5": {
                    "mean_regret": 0.034,
                    "forced_rate": 0.8,
                    "mean_trading_day": 4.6,
                    "improvement_bps": -15.0,
                    "ci95_bps": [-50.0, 20.0],
                    "holdout_mean_regret": 0.03,
                    "holdout_improvement_bps": -5.0,
                    "holdout_ci95_bps": [-20.0, 10.0],
                },
            },
        }
    )
    assert "拿掉保留價後" in report
    assert "點估計仍全為負" not in report


def test_render_report_lead_rules_require_ci_to_beat_day1() -> None:
    report = render_report(
        {
            "oracle_distribution": {"first_5_rate": 0.482},
            "lead_rules": {
                "tsm_neg_or_day5": {
                    "mean_regret": 0.033,
                    "forced_rate": 0.4,
                    "mean_trading_day": 3.0,
                    "improvement_bps": -10.0,
                    "ci95_bps": [-30.0, 10.0],
                    "holdout_mean_regret": 0.04,
                    "holdout_improvement_bps": -50.0,
                    "holdout_ci95_bps": [-80.0, -10.0],
                }
            },
            "lead_coverage": {
                "tsm_available_rate": 1.0,
                "sox_available_rate": 1.0,
                "fx_available_rate": 1.0,
            },
        }
    )
    assert "外部領先規則" in report
    assert "TSM ADR 下跌／第5日截止" in report
    assert "沒有證據顯示優於第 1 交易日" in report
    assert "任意下跌" in report
    assert "1%" in report
    assert "單日" in report


def test_render_report_classifies_filter_versus_delayed_day1() -> None:
    report = render_report(
        {
            "oracle_distribution": {"first_5_rate": 0.482},
            "lead_rules": {
                "tsm_neg_or_day5": {
                    "mean_regret": 0.033,
                    "forced_rate": 0.0,
                    "mean_trading_day": 2.2,
                    "improvement_bps": -13.0,
                    "ci95_bps": [-40.0, 10.0],
                    "holdout_mean_regret": 0.04,
                    "holdout_improvement_bps": -20.0,
                    "holdout_ci95_bps": [-80.0, 10.0],
                    "signal_rate": 0.47,
                },
                "tsm_dump1pct_or_day5": {
                    "mean_regret": 0.034,
                    "forced_rate": 0.22,
                    "mean_trading_day": 3.4,
                    "improvement_bps": -20.0,
                    "ci95_bps": [-50.0, 5.0],
                    "holdout_mean_regret": 0.05,
                    "holdout_improvement_bps": -30.0,
                    "holdout_ci95_bps": [-90.0, 10.0],
                    "signal_rate": 0.18,
                },
                "fx_pause_or_day5": {
                    "mean_regret": 0.032,
                    "forced_rate": 0.0,
                    "mean_trading_day": 1.1,
                    "improvement_bps": -1.0,
                    "ci95_bps": [-20.0, 15.0],
                    "holdout_mean_regret": 0.03,
                    "holdout_improvement_bps": -23.0,
                    "holdout_ci95_bps": [-40.0, -5.0],
                    "signal_rate": 0.87,
                },
            },
            "lead_coverage": {
                "tsm_available_rate": 1.0,
                "sox_available_rate": 1.0,
                "fx_available_rate": 1.0,
                "tsm_dump_rate": 0.47,
                "tsm_dump_1pct_rate": 0.18,
                "fx_pause_rate": 0.13,
                "fx_single_pause_rate": 0.48,
            },
        }
    )
    assert "延後買入" in report
    assert "幾乎就是第 1 日" in report
    assert "屬過濾" in report
    assert "TSM 隔夜大跌≥1%" in report
    assert "單日臺幣貶值" in report


def test_render_report_freezes_remaining_hypotheses() -> None:
    report = render_report({})
    assert "月初體制閘門" in report
    assert "真正的隔夜期貨" in report
    assert "月度金額" in report
    assert "不太可能贏、不要再做" in report
    assert "臺指夜盤" in report
    assert "每個月都等" in report
    assert "再掃 dump 門檻" in report
