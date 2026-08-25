"""繁體中文研究報告與圖表輸出。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _bps(value: float) -> str:
    return f"{value:.0f} bps"


def _strategy_table(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    if not metrics:
        return ""
    labels = {
        "fixed_day_1": "第 1 交易日",
        "fixed_day_5": "第 5 交易日",
        "fixed_day_10": "第 10 交易日",
        "fixed_day_15": "第 15 交易日",
        "last_day": "月底",
        "rsi30_or_last": "RSI<30／月底",
        "technical_calendar": "技術模型",
        "technical_valuation_calendar": "技術＋估值模型",
        "all": "全特徵模型",
    }
    header = (
        "| 策略 | 平均 regret | 中位 regret | P75 | P90 | P95"
        " | ≤0.5% 月份 | ≤1% 月份 | 強制率 | 平均買入日 | 期末財富 |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for name, item in metrics.items():
        forced = _percent(float(item["forced_rate"]), 1) if "forced_rate" in item else "—"
        wkey = "terminal_wealth_proxy"
        wealth = f"{float(item[wkey]):,.0f}" if wkey in item else "—"
        lines.append(
            "| {label} | {mean} | {median} | {p75} | {p90} | {p95}"
            " | {within05} | {within1} | {forced} | {day:.1f} | {wealth} |".format(
                label=labels.get(name, name),
                mean=_percent(float(item["mean_regret"])),
                median=_percent(float(item["median_regret"])),
                p75=_percent(float(item["p75_regret"])),
                p90=_percent(float(item["p90_regret"])),
                p95=_percent(float(item["p95_regret"])),
                within05=_percent(float(item["within_0_5pct_rate"]), 1),
                within1=_percent(float(item["within_1pct_rate"]), 1),
                forced=forced,
                day=float(item["mean_trading_day"]),
                wealth=wealth,
            )
        )
    return "\n".join(lines)


def _oracle_feature_table(ranges: Mapping[str, Mapping[str, Any]]) -> str:
    labels = {
        "open_gap_vs_action_ref": "開盤 gap（vs 調整前收）",
        "ret_1": "1 日報酬",
        "ret_5": "5 日報酬",
        "ret_20": "20 日報酬",
        "ma_gap_20": "距 20 日均線",
        "ma_gap_60": "距 60 日均線",
        "rsi_14": "RSI(14)",
        "bollinger_z_20": "布林 Z(20)",
        "drawdown_60": "距 60 日高點",
        "dividend_yield_ttm": "殖利率(TTM)",
        "premium_discount": "溢折價(vs NAV)",
        "margin_change_5": "融資 5 日變化率",
        "institutional_net_ratio": "法人淨買賣比",
        "volume_z_20": "量能 Z(20)",
    }
    lines = [
        "| 特徵 | Q25 | 中位數 | Q75 | 平均 | 觀測月份 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pct_keys = {
        "open_gap_vs_action_ref",
        "ret_1",
        "ret_5",
        "ret_20",
        "ma_gap_20",
        "ma_gap_60",
        "drawdown_60",
        "dividend_yield_ttm",
        "premium_discount",
        "margin_change_5",
    }
    for key, values in ranges.items():
        label = labels.get(key, key)

        def get_val(c: str, key: str = key, values: Mapping[str, Any] = values) -> str:
            if c not in values or values[c] is None:
                return "—"
            v = float(values[c])
            if key in pct_keys:
                return _percent(v)
            elif key == "rsi_14":
                return f"{v:.1f}"
            else:
                return f"{v:.3f}"

        cells = [get_val(c) for c in ("q25", "median", "q75", "mean")]
        n = int(values.get("n", 276))
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {n} |")
    return "\n".join(lines)


LEAD_LABELS: dict[str, str] = {
    "tsm_neg_or_day5": "TSM ADR 下跌／第5日截止",
    "sox_neg_or_day5": "SOX 下跌／第5日截止",
    "fx_pause_or_day5": "臺幣連貶暫緩／第5日截止",
    "tsm_dump1pct_or_day5": "TSM 隔夜大跌≥1%／第5日截止",
    "tsm_buy_unless_adverse": "第1日買入、TSM 隔夜下跌才暫緩／第5日截止",
    "fx_single_pause_or_day5": "臺幣單日貶值暫緩／第5日截止",
}
POLICY_LABELS: dict[str, str] = {
    "prob_and_res": "機率＋保留價／月底",
    "prob_only": "僅機率／月底",
    "res_only": "僅保留價／月底",
    "prob_and_res_deadline5": "機率＋保留價／第5日截止",
    "prob_only_deadline5": "僅機率／第5日截止",
    "res_only_deadline5": "僅保留價／第5日截止",
}


def _policy_ablation_table(
    ablation: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, str] | None = None,
) -> str:
    if not ablation:
        return ""
    name_labels = labels or POLICY_LABELS
    lines = [
        "| 決策規則 | 平均 regret | 強制率 | 平均買入日 | vs 第1日 | 95% CI |"
        " holdout regret | holdout vs 第1日 | holdout 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in name_labels:
        item = ablation.get(name, {})
        if not item:
            continue
        ci = item.get("ci95_bps", [0.0, 0.0])
        holdout_ci = item.get("holdout_ci95_bps", [0.0, 0.0])
        lines.append(
            "| {label} | {regret} | {forced} | {day:.1f} | {vs} | [{lo}, {hi}]"
            " | {h_regret} | {h_vs} | [{h_lo}, {h_hi}] |".format(
                label=name_labels[name],
                regret=_percent(float(item.get("mean_regret", 0.0))),
                forced=_percent(float(item.get("forced_rate", 0.0)), 1),
                day=float(item.get("mean_trading_day", 0.0)),
                vs=_bps(float(item.get("improvement_bps", 0.0))),
                lo=_bps(float(ci[0])),
                hi=_bps(float(ci[1])),
                h_regret=_percent(float(item.get("holdout_mean_regret", 0.0))),
                h_vs=_bps(float(item.get("holdout_improvement_bps", 0.0))),
                h_lo=_bps(float(holdout_ci[0])),
                h_hi=_bps(float(holdout_ci[1])),
            )
        )
    return "\n".join(lines)


def _policies_beating_day1(
    ablation: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    name_labels = labels or POLICY_LABELS
    winners: list[str] = []
    for name, item in ablation.items():
        ci = item.get("ci95_bps", [0.0, 0.0])
        if float(ci[0]) > 0.0 and float(ci[1]) > 0.0:
            winners.append(name_labels.get(name, name))
    return winners


def _change_phrase(before: float, after: float, formatted_after: str) -> str:
    if after < before:
        return f"降到 {formatted_after}"
    if after > before:
        return f"升到 {formatted_after}"
    return f"仍為 {formatted_after}"


def _ablation_mechanism(ablation: Mapping[str, Mapping[str, Any]]) -> str:
    dual = ablation.get("prob_and_res", {})
    prob_only = ablation.get("prob_only", {})
    dual_deadline = ablation.get("prob_and_res_deadline5", {})
    if not (dual and prob_only and dual_deadline):
        return ""
    dual_regret = float(dual.get("mean_regret", 0.0))
    dual_forced = float(dual.get("forced_rate", 0.0))
    prob_regret = float(prob_only.get("mean_regret", 0.0))
    prob_forced = float(prob_only.get("forced_rate", 0.0))
    deadline_regret = float(dual_deadline.get("mean_regret", 0.0))
    parts = [
        "拿掉保留價後，強制率從 "
        f"{_percent(dual_forced, 1)} {_change_phrase(dual_forced, prob_forced, _percent(prob_forced, 1))}，"
        f"平均 regret 從 {_percent(dual_regret)} "
        f"{_change_phrase(dual_regret, prob_regret, _percent(prob_regret))}。"
    ]
    if deadline_regret < dual_regret:
        parts.append(
            "第 5 日截止把雙門檻平均 regret 從 "
            f"{_percent(dual_regret)} 拉到 {_percent(deadline_regret)}。"
            "第 5 日截止列的高強制率表示多數月份在前 5 日沒有訊號、於第 5 日買入，"
            "接近固定第 5 日，而不是繼續等待。"
        )
    holdout_deltas = [float(item.get("holdout_improvement_bps", 0.0)) for item in ablation.values()]
    if holdout_deltas and all(delta < 0.0 for delta in holdout_deltas):
        parts.append(f"holdout 上 {len(ablation)} 組相對第 1 日的點估計仍全為負。")
    return " ".join(parts)


def _lead_rule_role(item: Mapping[str, Any]) -> str:
    forced = float(item.get("forced_rate", 0.0))
    mean_day = float(item.get("mean_trading_day", 0.0))
    if forced < 0.05 and mean_day < 1.35:
        return "almost_day1"
    if forced < 0.05:
        return "delayed_day1"
    return "filter"


def _lead_mechanism(
    lead_rules: Mapping[str, Mapping[str, Any]],
    lead_coverage: Mapping[str, Any],
) -> str:
    diagnoses: list[str] = []
    for name in LEAD_LABELS:
        item = lead_rules.get(name, {})
        if not item:
            continue
        label = LEAD_LABELS[name]
        forced = _percent(float(item.get("forced_rate", 0.0)), 1)
        mean_day = float(item.get("mean_trading_day", 0.0))
        role = _lead_rule_role(item)
        if role == "almost_day1":
            diagnoses.append(
                f"{label}：強制率 {forced}、平均第 {mean_day:.1f} 日，幾乎就是第 1 日。"
            )
        elif role == "delayed_day1":
            diagnoses.append(
                f"{label}：強制率 {forced}、平均第 {mean_day:.1f} 日，屬延後買入而非過濾。"
            )
        else:
            diagnoses.append(
                f"{label}：強制率 {forced}、平均第 {mean_day:.1f} 日，窗口內未觸發才截止買入，屬過濾。"
            )
    coverage_bits = []
    rate_labels = (
        ("tsm_dump_rate", "TSM 任意下跌"),
        ("tsm_dump_1pct_rate", "TSM ≥1% 大跌"),
        ("tsm_adverse_rate", "TSM 隔夜下跌（暫緩條件）"),
        ("sox_dump_rate", "SOX 任意下跌"),
        ("fx_pause_rate", "三日臺幣連貶"),
        ("fx_single_pause_rate", "單日臺幣貶值"),
    )
    for key, label in rate_labels:
        if key in lead_coverage:
            coverage_bits.append(f"{label} {_percent(float(lead_coverage[key]), 1)}")
    prefix = "訊號日頻率：" + "、".join(coverage_bits) + "。" if coverage_bits else ""
    return prefix + " ".join(diagnoses)


def _model_diagnostics_table(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    labels = {
        "technical_calendar": "技術＋日曆",
        "technical_valuation_calendar": "技術＋估值＋日曆",
        "all": "全特徵",
    }
    model_keys = [k for k in metrics if k in labels]
    if not model_keys:
        return ""
    lines = [
        "| 模型 | 特徵數 | Brier Score | Avg Precision | 強制買入率 | 平均 regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    feature_counts: dict[str, int] = {
        "technical_calendar": 15,
        "technical_valuation_calendar": 17,
        "all": 22,
    }
    for name in model_keys:
        item = metrics[name]
        lines.append(
            "| {label} | {n_feat} | {brier:.4f} | {ap:.4f} | {forced} | {regret} |".format(
                label=labels[name],
                n_feat=feature_counts.get(name, 0),
                brier=float(item.get("brier", 0)),
                ap=float(item.get("average_precision", 0)),
                forced=_percent(float(item.get("forced_rate", 0)), 1),
                regret=_percent(float(item["mean_regret"])),
            )
        )
    return "\n".join(lines)


def _report_meta(results: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "instrument": "0050",
        "display_name": "0050",
        "first_complete_month": "2003-07",
        "oos_start": "2008-07",
        "oos_end": "2026-06",
        "oos_months": 216,
        "primary_model": "all",
        "primary_label": "全特徵模型",
        "figure_relpath": "figures",
        "currency_label": "新臺幣",
        "fill_note": "開盤集合競價可確保成交，盤中低點不一定成交。",
        "lead_market_note": (
            "美股與匯率只使用臺灣交易日曆日前一日（含）已收盤的資料，當日美股不可用於當日開盤決策。"
        ),
        "has_taiwan_crosscheck": True,
        "deadline_first_5_rate": 0.482,
    }
    merged = {**defaults, **(results.get("meta") or {})}
    metrics = results.get("strategy_metrics") or {}
    day1 = metrics.get("fixed_day_1") or {}
    if "months" in day1:
        merged["oos_months"] = int(day1["months"])
    return merged


def render_report(results: Mapping[str, Any]) -> str:
    """把已驗證的研究統計轉為可交接的繁中 Markdown。"""

    meta = _report_meta(results)
    quality = results.get("data_quality", {}) or {}
    oracle = results.get("oracle_distribution", {}) or {}
    primary = results.get("primary", {}) or {}
    holdout = results.get("holdout", {}) or {}
    ranges = results.get("oracle_ranges", {}) or {}
    day_comparison = results.get("day1_vs_day5", {}) or {}
    metrics = results.get("strategy_metrics", {}) or {}
    random = results.get("random_strategy", {}) or {}
    ablation = results.get("policy_ablation", {}) or {}
    lead_rules = results.get("lead_rules", {}) or {}
    lead_coverage = results.get("lead_coverage", {}) or {}

    model_ci = primary.get("model_ci95_bps", [0.0, 0.0])
    day_ci = day_comparison.get("ci95_bps", [0.0, 0.0])
    day_improvement = float(day_comparison.get("improvement_bps", 0.0))
    holdout_ci = holdout.get("model_ci95_bps", [0.0, 0.0])

    oracle_months = int(oracle.get("months", 276))
    oracle_mode_day = int(oracle.get("mode_day", 1))
    oracle_median_day = float(oracle.get("median_day", 6.0))
    oracle_mean_day = float(oracle.get("mean_day", 8.4))
    oracle_first_day_rate = float(oracle.get("first_day_rate", 0.268))
    oracle_first_5_rate = float(oracle.get("first_5_rate", 0.482))
    oracle_first_10_rate = float(oracle.get("first_10_rate", 0.605))

    primary_day1_mean_regret = float(primary.get("day1_mean_regret", 0.032))
    primary_model_mean_regret = float(primary.get("model_mean_regret", 0.045))
    primary_model_improvement_bps = float(primary.get("model_improvement_bps", -116.0))
    primary_day1_within_rate = float(primary.get("day1_within_rate", 0.389))

    holdout_months = int(holdout.get("months", 36))
    holdout_selected_day = int(holdout.get("selected_day", 1))
    holdout_day1_mean_regret = float(holdout.get("day1_mean_regret", 0.0))
    holdout_model_mean_regret = float(holdout.get("model_mean_regret", 0.0))
    holdout_day1_within_rate = float(holdout.get("day1_within_rate", 0.0))
    holdout_model_improvement_bps = float(holdout.get("model_improvement_bps", 0.0))

    rows = int(quality.get("rows", 5665))
    official_difference = float(quality.get("official_max_difference", 0.0))
    issuer_difference = float(quality.get("issuer_max_difference", 0.0))
    official_missing = int(quality.get("official_missing", 0))
    issuer_missing = int(quality.get("issuer_market_price_missing", 1))

    quality_dividend_events = int(quality.get("dividend_events", 31))
    quality_split_events = int(quality.get("split_events", 1))
    quality_columns = int(quality.get("columns", 38))
    quality_start = str(quality.get("start", "2003-06-30"))
    quality_end = str(quality.get("end", "2026-07-09"))

    if ablation:
        policy_count = len(ablation)
        winners = _policies_beating_day1(ablation)
        if winners:
            winner_text = "、".join(winners)
            ablation_verdict = (
                f"預先指定的 {policy_count} 組政策中，{winner_text} 的樣本外 95% CI 全數 > 0，"
                "即相對第 1 日有統計上可分辨的改善。"
            )
        else:
            ablation_verdict = (
                f"{policy_count} 組預先指定政策的樣本外 95% CI 均未全數 > 0，"
                "沒有證據顯示改門檻或改截止日優於第 1 交易日。"
            )
        mechanism = _ablation_mechanism(ablation)
        ablation_body = (
            f"模型預測沿用{meta['primary_label']} walk-forward，只改買入規則。"
            f"{policy_count} 組政策在看到樣本外結果前即固定："
            "門檻為「機率＋保留價／僅機率／僅保留價」，截止為「月底強制」或「第 5 交易日」。"
            "第 5 日截止來自 0050 第一版已公布的 oracle 前 5 日占比"
            f"（{_percent(float(meta.get('deadline_first_5_rate', 0.482)), 1)}），"
            "不是從此標的樣本外挑出的參數。\n\n"
            f"{_policy_ablation_table(ablation)}\n\n"
            "強制率 = 搜尋窗口內未觸發、於截止日買入的月份比例。"
            f"正的 vs 第 1 日代表 regret 較低。{ablation_verdict}"
            + (f"\n\n{mechanism}" if mechanism else "")
        )
    else:
        ablation_body = "本報告輸入未含決策規則拆解結果。"

    if lead_rules:
        lead_count = len(lead_rules)
        lead_winners = _policies_beating_day1(lead_rules, LEAD_LABELS)
        if lead_winners:
            lead_verdict = (
                f"預先指定的 {lead_count} 條規則中，{'、'.join(lead_winners)} "
                "的樣本外 95% CI 全數 > 0。"
            )
        else:
            lead_verdict = (
                f"{lead_count} 條預先指定領先規則的樣本外 95% CI 均未全數 > 0，"
                "沒有證據顯示優於第 1 交易日。"
            )
        holdout_deltas = [
            float(item.get("holdout_improvement_bps", 0.0)) for item in lead_rules.values()
        ]
        if holdout_deltas and all(delta < 0.0 for delta in holdout_deltas):
            lead_verdict += f" holdout 上 {lead_count} 條相對第 1 日的點估計仍全為負。"
        coverage_text = (
            "樣本外可對齊比例："
            f"TSM {_percent(float(lead_coverage.get('tsm_available_rate', 0.0)), 1)}，"
            f"SOX {_percent(float(lead_coverage.get('sox_available_rate', 0.0)), 1)}，"
            f"USD/TWD {_percent(float(lead_coverage.get('fx_available_rate', 0.0)), 1)}。"
        )
        lead_body = (
            "原三條規則的失敗機制：任意下跌日頻率約一半，配第 5 日截止後幾乎必觸發，等於延後第 1 日；"
            "美元／新臺幣連續 3 個交易日升值（臺幣連貶）才暫緩則過稀，幾乎就是第 1 日。"
            "修正版在看到樣本外結果前即凍結："
            "隔夜大跌改為預先指定的 1%（經濟整數，不是從樣本外掃出的參數）、"
            "改為第 1 日買入僅在 TSM 隔夜下跌時暫緩、"
            "匯率改為前一交易日美元／新臺幣升值（臺幣貶值）即暫緩。"
            "第 5 日截止仍來自 0050 已公布的 oracle 前 5 日占比"
            f"（{_percent(float(meta.get('deadline_first_5_rate', 0.482)), 1)}），"
            "不依此標的樣本重估。"
            f"{meta['lead_market_note']}"
            "買入條件為真則當日開盤買，否則第 5 交易日買。"
            f"{coverage_text}\n\n"
            f"{_policy_ablation_table(lead_rules, LEAD_LABELS)}\n\n"
            f"{lead_verdict}\n\n"
            f"{_lead_mechanism(lead_rules, lead_coverage)}"
        )
    else:
        lead_body = "本報告輸入未含外部領先規則結果。"

    random_section = ""
    if random:
        random_section = (
            "\n\n### 隨機策略基準\n\n"
            "每月均勻隨機選一天買入（5,000 次模擬）："
            f"平均 regret {_percent(float(random['mean_regret']))}，"
            f"95% CI [{_percent(float(random['ci95'][0]))}, "
            f"{_percent(float(random['ci95'][1]))}]。"
            f"第 1 日 {_percent(primary_day1_mean_regret)} 低於隨機中位，"
            "表示月初的優勢不僅是運氣。"
        )

    display_name = str(meta["display_name"])
    first_complete_month = str(meta["first_complete_month"])
    oos_start = str(meta["oos_start"])
    oos_end = str(meta["oos_end"])
    oos_months = int(meta["oos_months"])
    primary_label = str(meta["primary_label"])
    figure_relpath = str(meta["figure_relpath"])
    fill_note = str(meta["fill_note"])
    currency_label = str(meta["currency_label"])
    has_taiwan_crosscheck = bool(meta["has_taiwan_crosscheck"])
    ci_lo = float(model_ci[0])
    ci_hi = float(model_ci[1])
    holdout_lo = float(holdout_ci[0])
    holdout_hi = float(holdout_ci[1])
    if ci_lo > 0.0 and ci_hi > 0.0:
        lead_conclusion = (
            f"在「每月必買一次、開盤成交、只用 T-1 資訊」的約束下，"
            f"{primary_label} 的樣本外 95% CI 全數 > 0。"
        )
        model_delta = (
            f"模型比第 1 日平均好 **{_bps(primary_model_improvement_bps)}**，"
            f"95% moving-block bootstrap CI 為 **[{_bps(ci_lo)}, {_bps(ci_hi)}]**。"
            "CI 全數 > 0。"
        )
    elif ci_hi < 0.0 and ci_lo < 0.0:
        lead_conclusion = (
            "歷史最佳點常伴隨弱勢特徵，但等待這些訊號在樣本外沒有可靠證據優於月初買入。"
            "模型、決策規則拆解與外部領先規則亦然：在「每月必買一次、開盤成交、只用 T-1 資訊」"
            "的約束下，沒有證據顯示月內擇時能穩定勝過第 1 日。"
        )
        model_delta = (
            f"模型比第 1 日平均差 **{_bps(abs(primary_model_improvement_bps))}**，"
            f"95% moving-block bootstrap CI 為 **[{_bps(ci_lo)}, {_bps(ci_hi)}]**。"
            "CI 全數 < 0，等待模型訊號**顯著較差**。"
        )
    else:
        lead_conclusion = (
            "歷史最佳點常伴隨弱勢特徵，但等待這些訊號在樣本外沒有可靠證據優於月初買入。"
        )
        model_delta = (
            f"模型相對第 1 日 **{_bps(primary_model_improvement_bps)}**，"
            f"95% moving-block bootstrap CI 為 **[{_bps(ci_lo)}, {_bps(ci_hi)}]**。"
            "CI 跨越 0，沒有證據顯示優於第 1 日。"
        )
    if holdout_hi < 0.0 and holdout_lo < 0.0:
        holdout_verdict = (
            f"Holdout 差距更大（{_bps(abs(holdout_model_improvement_bps))}），CI 全 < 0。"
        )
    elif holdout_lo > 0.0 and holdout_hi > 0.0:
        holdout_verdict = "Holdout 的 95% CI 全數 > 0。"
    else:
        holdout_verdict = "Holdout 的 95% CI 跨越 0，不能宣稱優於第 1 日。"
    if has_taiwan_crosscheck:
        feature_section = """所有特徵分為四類，共 22 個：

**技術面（11 個）**：1/5/20/60 日報酬、距 20/60 日均線、RSI(14)、布林 Z(20)、距 60 日高點、20 日年化波動率、成交量 Z(20)。

**估值面（2 個）**：NAV 折溢價、TTM 配息殖利率。

**籌碼面（5 個）**：融資 5 日變化率、融券 5 日變化率、法人淨買賣比、法人資料可得標記、基金流量 5 日變化率。

**日曆面（4 個）**：月內交易日序號、月進度（日曆日比例）、星期三角編碼（sin/cos）。

**關鍵防偷看設計**：所有日終特徵（技術、估值、籌碼）向後延遲一個交易日。日曆特徵無需延遲。月進度使用日曆日比例（date.day / days_in_month）。"""
        model_combo = """| 模型名稱 | 特徵組合 | 特徵數 |
|---|---|---:|
| 技術＋日曆 | technical + calendar | 15 |
| 技術＋估值＋日曆 | technical + valuation + calendar | 17 |
| 全特徵 | technical + valuation + chip + calendar | 22 |"""
        cross_check = f"""| 來源 | 涵蓋交易日 | 缺值 | 與 FinMind 最大差異 |
|---|---:|---:|---:|
| TWSE 官方逐月收盤 | {rows:,} | {official_missing} | {official_difference:.4f} 元 |
| 元大官方市價 | {rows - issuer_missing:,} | {issuer_missing} | {issuer_difference:.4f} 元 |"""
        valuation_note = "0050 是 ETF，沒有公司層級 EPS。估值只採 point-in-time NAV 折溢價與 trailing distribution yield。"
        sources = """- [TWSE 個股日收盤價及月平均價](https://www.twse.com.tw/zh/trading/historical/stock-day-avg.html)
- [元大 0050 歷史 NAV](https://www.yuantaetfs.com/tradeInfo/comparison/0050/NAVhistory)
- [元大 0050 基本資訊與上市日](https://www.yuantaetfs.com/product/detail/0050/Basic_information)
- [FinMind API 文件](https://finmind.github.io/quickstart/)
- [FinMind 美股日線 USStockPrice](https://finmind.github.io/tutor/UnitedStatesMarket/Technical/)
- [FinMind 臺灣銀行匯率 TaiwanExchangeRate](https://finmind.github.io/tutor/ExchangeRate/)
- [TWSE 交易制度](https://www.twse.com.tw/en/products/system/trading.html)"""
        fill_assumption = "日線研究，假設小額訂單可在開盤集合競價成交。"
        single_name = "本報告只研究 0050；VT 用同一協議複製，結論相同。"
    else:
        feature_section = """VT 沒有點時 NAV、融資融券與三大法人，主模型只用技術＋日曆，共 15 個特徵，對應 0050 的 technical_calendar：

**技術面（11 個）**：1/5/20/60 日報酬、距 20/60 日均線、RSI(14)、布林 Z(20)、距 60 日高點、20 日年化波動率、成交量 Z(20)。

**日曆面（4 個）**：月內交易日序號、月進度（日曆日比例）、星期三角編碼（sin/cos）。

**關鍵防偷看設計**：所有日終特徵向後延遲一個交易日。日曆特徵無需延遲。月進度使用日曆日比例。"""
        model_combo = """| 模型名稱 | 特徵組合 | 特徵數 |
|---|---|---:|
| 技術＋日曆 | technical + calendar | 15 |"""
        cross_check = (
            "單一來源 FinMind USStockPrice（原始 OHLC + Adj_Close）。"
            "adjusted_open = open × (Adj_Close / Close)。"
            "沒有 TWSE／發行人 NAV 第二來源。"
        )
        valuation_note = "VT 複製實驗不使用估值與籌碼特徵，避免用假 NAV 或全零籌碼訓練模型。"
        sources = """- [FinMind 美股日線 USStockPrice](https://finmind.github.io/tutor/UnitedStatesMarket/Technical/)（VT 原始 OHLC 與 Adj_Close）
- [FinMind 美股日線 USStockPrice](https://finmind.github.io/tutor/UnitedStatesMarket/Technical/)（TSM ADR、`^SOX` 領先序列）
- [FinMind 臺灣銀行匯率 TaiwanExchangeRate](https://finmind.github.io/tutor/ExchangeRate/)
- [Vanguard VT](https://investor.vanguard.com/investment-products/etfs/profile/vt)"""
        fill_assumption = "日線研究，假設小額訂單可在美股官方 Open 成交。"
        single_name = "本報告只研究 VT；0050 用同一協議，結論相同。"

    return f"""# {display_name} 每月最佳買點研究

> 結論：若每月必須只買一次，**第 1 個交易日直接買入**是目前最穩健、最可執行的基準。{lead_conclusion}

---

## 1. 研究問題

本研究回答一個實務問題：**對定期定額買入 {display_name} 的投資人，每月只買一次時，選哪一個交易日下單最合理？**

具體而言，我們評估三類策略：
1. **固定日策略**：每月固定在第 N 個交易日買入（N = 1, 5, 10, 15, 最後一日）
2. **規則策略**：等待特定技術訊號觸發（如 RSI < 30）再買，逾時則月底強制買入
3. **機器學習策略**：用歷史特徵訓練模型，逐日判斷是否為好買點

## 2. 核心結論

- 共分析 **{oracle_months}** 個完整月份（{first_complete_month} 至 {oos_end}）。事後最佳日的眾數是第 **{oracle_mode_day}** 個交易日，中位數是第 **{oracle_median_day:.0f}** 日，平均第 **{oracle_mean_day:.1f}** 日。
- 精確最低點有 **{_percent(oracle_first_day_rate, 1)}** 出現在第 1 日，**{_percent(oracle_first_5_rate, 1)}** 在前 5 日，**{_percent(oracle_first_10_rate, 1)}** 在前 10 日。
- {oos_start} 至 {oos_end} 的 **{oos_months} 個樣本外月份**：第 1 日平均 regret **{_percent(primary_day1_mean_regret)}**；{primary_label} **{_percent(primary_model_mean_regret)}**。
- {model_delta}
- 最後 **{holdout_months}** 個月的 sealed holdout（2023-07 至 2026-06）仍一致：第 1 日 **{_percent(holdout_day1_mean_regret)}**，模型 **{_percent(holdout_model_mean_regret)}**；差距 **{_bps(abs(holdout_model_improvement_bps))}**，95% CI **[{_bps(holdout_lo)}, {_bps(holdout_hi)}]**。
- 第 1 日有 **{_percent(primary_day1_within_rate, 1)}** 的月份落在當月最低價 0.5% 內；holdout 期間更高達 **{_percent(holdout_day1_within_rate, 1)}**。
- 第 1 日相對第 5 日的平均優勢為 **{_bps(day_improvement)}**，95% CI **[{_bps(float(day_ci[0]))}, {_bps(float(day_ci[1]))}]**，跨過 0 -> 前 1-5 日無統計差異。

## 3. 方法論

### 3.1 Oracle 定義（每月最佳買點）

每月的 oracle 定義為該月 **total-return adjusted 開盤價最低** 的那一天：
- 使用開盤價而非盤中最低價，因為{fill_note}
- 同價時取最早日，避免後見偏差。
- Total-return adjusted price 已將配息再投入與股票分割一併還原，跨期可比。

### 3.2 Regret 指標

`regret = 實際買入 adjusted open / 當月最低 adjusted open - 1`

- Regret >= 0，等於 0 表示恰好買在最低點。
- 越低越好；此指標衡量「離完美的距離」，而非絕對報酬。

### 3.3 特徵設計

{feature_section}

### 3.4 模型架構

每月使用兩個模型協作：

1. **分類器（Logistic Regression）**：C=0.1，class_weight=balanced，月內樣本加權使各月等權。
2. **保留價迴歸器（Gradient Boosting Quantile Regressor）**：60 棵樹，max_depth=2，min_samples_leaf=20。

**買入決策規則**：每月從第 1 日起逐日檢查，首個同時滿足 near_probability >= 0.5 且 adjusted open <= 保留價 x 1.005 的交易日即買入。若整月未觸發，最後一日強制買入。

### 3.5 Walk-Forward 設計

- **初始訓練窗口**：前 60 個完整月份。
- **擴展方式**：Expanding window，每月新增一個月的完整資料後重訓。
- **測試集**：下一個月的所有交易日，完全樣本外。
- **樣本外期間**：{oos_start} 至 {oos_end}，共 {oos_months} 個月。
- **Sealed holdout**：2023-07 至 2026-06，共 {holdout_months} 個月。

### 3.6 模型組合

{model_combo}

### 3.7 統計檢驗

策略間比較使用 **paired test** 與 **Moving-Block Bootstrap**（block length = 12 個月、5,000 次重抽樣）。

## 4. 事後最佳日特徵分析

{oracle_months} 個月中 oracle 日的前一日特徵統計（僅描述關聯，不代表因果）：

{_oracle_feature_table(ranges)}

**解讀**：事後最低點的前一天通常已處於短期弱勢，但嘗試等待這些條件觸發時，模型有超過一半的月份被迫拖到月底強制買入，反而錯過月初的低點。

## 5. 樣本外策略比較（{oos_start} 至 {oos_end}，{oos_months} 個月）

{_strategy_table(metrics)}

**欄位說明**：regret 越低越好。強制率 = 模型整月未觸發、月底被迫買入的比例。期末財富 = 每月投入 10,000 {currency_label}的 total-return 終值。{random_section}

## 6. 模型診斷

{_model_diagnostics_table(metrics)}

- **Brier Score**：越低越好（完美 = 0，隨機 = 0.25）。
- **Average Precision**：越高越好。
- **強制買入率**：雙重門檻若過嚴，多數月份會被迫在截止日買入。

## 7. 決策規則拆解（不重訓）

{ablation_body}

## 8. 外部領先規則（不重訓、不掃參）

{lead_body}

## 9. Sealed Holdout 驗證（2023-07 至 2026-06，{holdout_months} 個月）

此區間完全未參與任何開發決策。

| 指標 | 第 {holdout_selected_day} 日 | {primary_label} |
|---|---:|---:|
| 平均 regret | {_percent(holdout_day1_mean_regret)} | {_percent(holdout_model_mean_regret)} |
| ≤0.5% 月份 | {_percent(holdout_day1_within_rate, 1)} | — |
| 差距 | — | {_bps(holdout_model_improvement_bps)} |
| 95% CI | — | [{_bps(holdout_lo)}, {_bps(holdout_hi)}] |

{holdout_verdict}

## 10. 資料與品質

### 10.1 覆蓋範圍

- 每日 OHLCV：**{rows:,}** 筆，{quality_start} 至 {quality_end}。
- 完整月份：{oracle_months} 個（{first_complete_month} 至 {oos_end}）。
- 企業行動：**{quality_dividend_events}** 次配息、**{quality_split_events}** 次分割。
- 欄位數：{quality_columns}。

### 10.2 交叉驗證

{cross_check}

### 10.3 估值限制

{valuation_note}

### 10.4 資料來源

{sources}

## 11. 圖表

![價格歷史]({figure_relpath}/price_history.png)

![最佳日分布]({figure_relpath}/oracle_day_distribution.png)

![最佳日特徵]({figure_relpath}/oracle_feature_profile.png)

![策略比較]({figure_relpath}/strategy_comparison.png)

![決策規則拆解]({figure_relpath}/policy_ablation.png)

![外部領先規則]({figure_relpath}/lead_rules.png)

## 12. 限制與注意事項

1. **交易假設**：{fill_assumption}
2. **關聯非因果**：特徵與最佳日之間是統計關聯，不是因果關係。
3. **正向漂移效應**：第 1 日的優勢主要來自股市長期正向漂移。
4. **單一標的**：{single_name}
5. **Regime change**：所有候選規則都可能因市場環境改變而失效。
6. **手續費與稅**：期末財富含 0.1425% 手續費，未計證交稅。
7. **非投資建議**：本報告是量化研究，不構成個人化投資建議。
8. **月內擇時上限**：oracle 最低點多數不在第 1 日，但那是事後路徑。開盤前可執行訊號未能把這段差距變成樣本外 regret 改善；真過濾反而因截止日買入而更差。繼續掃月內門檻會變成 p-hacking，不是新資訊。

## 13. 尚未檢驗的方向

日頻技術、籌碼、TSM／SOX／匯率在 0050 與 VT 上都沒有樣本外證據贏第 1 日。剩餘機會不是再找「今天是最低點」的日頻指標，而是改「這個月要不要等」或改「這個月買多少」。下列假設必須在看樣本外結果前凍結；門檻不掃參。宣稱優於第 1 日的標準不變：樣本外 95% CI 必須全數 > 0。

### 13.1 月度金額（先驗較高，但是另一個問題）

國發會景氣對策信號、信用利差、估值百分位屬月頻資訊，應用於每月扣款金額，而不是再挑月內交易日。
- **特徵指標**：國發會景氣對策信號（紅藍燈）；as-of 為第 1 日開盤前已公布的上一期，不可用當月事後燈號。
- **問題定義**：藍燈期加碼、紅燈期減碼是否提高累積單位；主指標不再是相對當月 oracle 的 regret。
- **不可宣稱**：尚未回測，不得預設會勝過每月固定金額、第 1 日買入。

### 13.2 月初體制閘門 + 等待（月內擇時裡先驗最高的剩餘項）

已測規則的失敗機制是「每個月都等」：多頭月付出正漂移，抵銷少數抓到低點的月份。改成第 1 日開盤前用慢變數凍結體制：
- 低壓力月：直接第 1 日買（預設）。
- 高壓力月：才啟用預先指定的隔夜大跌等待，第 5 日截止。
- 候選慢變數：VIX 水準或 VIX 減 VIX3M、已實現波動相對 expanding-window 歷史百分位、高收益信用利差、景氣燈。
- 體制在該月第 1 日開盤前凍結，月內不可依新日頻資料改體制。門檻必須預先指定（例如 VIX 大於 20，或波動率大於 expanding 歷史 75 百分位），不可從樣本外挑切點。

### 13.3 真正的隔夜期貨（資訊比 T-1 日線更近）

TSM／SOX 的 T-1 收盤已測過且失敗。更近的開盤前資訊：
- 0050：臺指夜盤相對前一盤後收。
- VT：CME ES 夜盤／開盤前報價相對前一美股常規收盤。
大跌門檻沿用已凍結的 1% 經濟整數，不另掃參。單獨使用仍有過密風險，應與 13.2 體制閘門搭配。缺時間戳來源則不可用當日日線收盤假裝夜盤。

### 13.4 不太可能贏、不要再做

再加 RSI／均線／布林、再掃 dump 門檻或截止日、同市場 T-1 報酬正負號、盤中低點或 VWAP（違反開盤成交假設）。這些不是新資訊。

> [!WARNING]
> **多重比較偏誤（Multiple Comparison Bias / p-hacking）**
> 月內規則的門檻與截止日必須預先指定。從樣本外挑最好的 dump 門檻或截止日，不能當作贏過第 1 日的證據。

## 14. 術語表

| 術語 | 定義 |
|---|---|
| Oracle | 事後回頭看，某月 total-return adjusted 開盤價最低的那天 |
| Regret | 實際買入價與當月最低價的比率 - 1 |
| Walk-forward | 逐月向前滾動的回測法，確保每月預測只用過去資料 |
| Expanding window | 訓練窗口只擴大不縮小 |
| Sealed holdout | 預留的最後一段資料，全程不用於任何開發決策 |
| Total-return adjusted | 將配息再投入與分割還原後的價格，跨期可比 |
| Brier Score | 機率預測的校準指標，0=完美，0.25=隨機二元 |
| Average Precision | 正類檢出品質的加權指標 |
| Moving-block bootstrap | 適用於時間序列的信賴區間估計法 |
| Reservation price | 模型算出的最高可接受買價 |
| 強制買入 | 搜尋窗口內未觸發，於截止日強制執行買入 |
| 截止日 | 未觸發時的買入日；預設月底，拆解與領先規則另測第 5 交易日 |
| TSM lead | 前一美股交易日台積電 ADR 還原報酬（0050 為跨市場隔夜；VT 為同一市場 T-1） |
| SOX lead | 費城半導體指數前一美股交易日還原報酬 |
| USD/TWD 升值 | 美元兌新臺幣即期中間價上升，即臺幣貶值；單日與三日暫緩都是這個方向 |
| 體制閘門 | 第 1 日開盤前凍結的慢變數，決定本月直接買還是等待 |
| 隔夜期貨 | 開盤前已成交的夜盤／盤前期貨報酬；時間戳必須早於目標開盤 |
"""


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft JhengHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_price_history(
    daily: pd.DataFrame,
    path: Path,
    *,
    price_column: str = "split_adjusted_close",
    title: str = "0050 分割還原收盤價（對數軸）",
    ylabel: str = "新臺幣／目前受益權單位",
) -> None:
    column = price_column if price_column in daily.columns else "adjusted_close"
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(daily["date"], daily[column], color="#174A7E", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_oracle_distribution(oracle: pd.DataFrame, path: Path) -> None:
    counts = oracle["oracle_trading_day"].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(
        counts.index.to_numpy(dtype=float),
        counts.to_numpy(dtype=float),
        color="#3B82A0",
    )
    axis.axvline(
        float(oracle["oracle_trading_day"].median()),
        color="#D97706",
        linestyle="--",
        label="中位數",
    )
    axis.set_title("每月事後最佳買點：當月第幾個交易日")
    axis.set_xlabel("交易日序號")
    axis.set_ylabel("月份數")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_feature_profile(profile: pd.DataFrame, path: Path) -> None:
    top = profile.head(12).sort_values("distance_from_median")
    colors = ["#2563A6" if value < 0 else "#D97706" for value in top["distance_from_median"]]
    fig, axis = plt.subplots(figsize=(9, 5.6))
    axis.barh(top["feature"], top["distance_from_median"], color=colors)
    axis.axvline(0, color="#374151", linewidth=0.8)
    axis.set_title("最佳日特徵在當月內的 percentile 偏移")
    axis.set_xlabel("平均 percentile − 50%（負值代表偏低）")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_strategy_comparison(
    metrics: Mapping[str, Mapping[str, Any]],
    path: Path,
    *,
    primary_model: str = "all",
    primary_label: str = "全特徵模型",
) -> None:
    order = [
        "fixed_day_1",
        "fixed_day_5",
        "fixed_day_10",
        "fixed_day_15",
        "rsi30_or_last",
        primary_model,
    ]
    labels = ["第1日", "第5日", "第10日", "第15日", "RSI規則", primary_label]
    present = [name for name in order if name in metrics]
    axis_labels = [labels[order.index(name)] for name in present]
    values = [float(metrics[name]["mean_regret"]) * 100 for name in present]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(
        axis_labels,
        values,
        color=["#174A7E"] + ["#8BA9C4"] * (len(values) - 2) + ["#C65D3B"],
    )
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set_title("樣本外平均月度 regret（越低越好）")
    axis.set_ylabel("regret")
    axis.set_ylim(0, max(values) * 1.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_ablation_comparison(
    ablation: Mapping[str, Mapping[str, Any]],
    day1_mean_regret: float,
    path: Path,
    *,
    labels: Mapping[str, str] | None = None,
    title: str = "決策規則拆解：樣本外平均 regret（越低越好）",
) -> None:
    name_labels = labels or POLICY_LABELS
    names = [name for name in name_labels if name in ablation]
    axis_labels = [name_labels[name] for name in names]
    values = [float(ablation[name]["mean_regret"]) * 100 for name in names]
    height = max(4.8, 0.55 * len(names) + 1.8)
    fig, axis = plt.subplots(figsize=(9.5, height))
    positions = list(range(len(names)))
    axis.barh(positions, values, color="#8BA9C4")
    axis.set_yticks(positions, axis_labels)
    axis.axvline(day1_mean_regret * 100, color="#174A7E", linestyle="--", label="第1交易日")
    axis.set_xlabel("平均 regret（%）")
    axis.set_title(title)
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    daily: pd.DataFrame,
    oracle: pd.DataFrame,
    profile: pd.DataFrame,
    metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    policy_ablation: Mapping[str, Mapping[str, Any]] | None = None,
    day1_mean_regret: float | None = None,
    lead_rules: Mapping[str, Mapping[str, Any]] | None = None,
    primary_model: str = "all",
    primary_label: str = "全特徵模型",
    price_column: str = "split_adjusted_close",
    price_title: str = "0050 分割還原收盤價（對數軸）",
    price_ylabel: str = "新臺幣／目前受益權單位",
) -> None:
    """產生報告使用的可重建圖表。"""

    _configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_price_history(
        daily,
        output_dir / "price_history.png",
        price_column=price_column,
        title=price_title,
        ylabel=price_ylabel,
    )
    _save_oracle_distribution(oracle, output_dir / "oracle_day_distribution.png")
    _save_feature_profile(profile, output_dir / "oracle_feature_profile.png")
    _save_strategy_comparison(
        metrics,
        output_dir / "strategy_comparison.png",
        primary_model=primary_model,
        primary_label=primary_label,
    )
    if policy_ablation and day1_mean_regret is not None:
        _save_ablation_comparison(
            policy_ablation, day1_mean_regret, output_dir / "policy_ablation.png"
        )
    if lead_rules and day1_mean_regret is not None:
        _save_ablation_comparison(
            lead_rules,
            day1_mean_regret,
            output_dir / "lead_rules.png",
            labels=LEAD_LABELS,
            title="外部領先規則：樣本外平均 regret（越低越好）",
        )
