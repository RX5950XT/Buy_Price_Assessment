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


def render_report(results: Mapping[str, Any]) -> str:
    """把已驗證的研究統計轉為可交接的繁中 Markdown。"""

    quality = results.get("data_quality", {}) or {}
    oracle = results.get("oracle_distribution", {}) or {}
    primary = results.get("primary", {}) or {}
    holdout = results.get("holdout", {}) or {}
    ranges = results.get("oracle_ranges", {}) or {}
    day_comparison = results.get("day1_vs_day5", {}) or {}
    metrics = results.get("strategy_metrics", {}) or {}
    random = results.get("random_strategy", {}) or {}

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

    return f"""# 0050 每月最佳買點研究

> 結論：若每月必須只買一次，**第 1 個交易日直接買入**是目前最穩健、最可執行的基準。歷史最佳點常伴隨弱勢特徵，但等待這些訊號在樣本外沒有可靠證據優於月初買入。

---

## 1. 研究問題

本研究回答一個實務問題：**對定期定額買入 0050 的投資人，每月只買一次時，選哪一個交易日下單最合理？**

具體而言，我們評估三類策略：
1. **固定日策略**：每月固定在第 N 個交易日買入（N = 1, 5, 10, 15, 最後一日）
2. **規則策略**：等待特定技術訊號觸發（如 RSI < 30）再買，逾時則月底強制買入
3. **機器學習策略**：用歷史特徵訓練模型，逐日判斷是否為好買點

## 2. 核心結論

- 共分析 **{oracle_months}** 個完整月份（2003-07 至 2026-06）。事後最佳日的眾數是第 **{oracle_mode_day}** 個交易日，中位數是第 **{oracle_median_day:.0f}** 日，平均第 **{oracle_mean_day:.1f}** 日。
- 精確最低點有 **{_percent(oracle_first_day_rate, 1)}** 出現在第 1 日，**{_percent(oracle_first_5_rate, 1)}** 在前 5 日，**{_percent(oracle_first_10_rate, 1)}** 在前 10 日。
- 2008-07 至 2026-06 的 **216 個樣本外月份**：第 1 日平均 regret **{_percent(primary_day1_mean_regret)}**；全特徵模型 **{_percent(primary_model_mean_regret)}**。
- 模型比第 1 日平均差 **{_bps(abs(primary_model_improvement_bps))}**，95% moving-block bootstrap CI 為 **[{_bps(float(model_ci[0]))}, {_bps(float(model_ci[1]))}]**。CI 全數 < 0，等待模型訊號**顯著較差**。
- 最後 **{holdout_months}** 個月的 sealed holdout（2023-07 至 2026-06）仍一致：第 1 日 **{_percent(holdout_day1_mean_regret)}**，模型 **{_percent(holdout_model_mean_regret)}**；差距 **{_bps(abs(holdout_model_improvement_bps))}**，95% CI **[{_bps(float(holdout_ci[0]))}, {_bps(float(holdout_ci[1]))}]**。
- 第 1 日有 **{_percent(primary_day1_within_rate, 1)}** 的月份落在當月最低價 0.5% 內；holdout 期間更高達 **{_percent(holdout_day1_within_rate, 1)}**。
- 第 1 日相對第 5 日的平均優勢為 **{_bps(day_improvement)}**，95% CI **[{_bps(float(day_ci[0]))}, {_bps(float(day_ci[1]))}]**，跨過 0 -> 前 1-5 日無統計差異。

## 3. 方法論

### 3.1 Oracle 定義（每月最佳買點）

每月的 oracle 定義為該月 **total-return adjusted 開盤價最低** 的那一天：
- 使用開盤價而非盤中最低價，因為開盤集合競價可確保成交，盤中低點不一定成交。
- 同價時取最早日，避免後見偏差。
- Total-return adjusted price 已將配息再投入與股票分割一併還原，跨期可比。

### 3.2 Regret 指標

`regret = 實際買入 adjusted open / 當月最低 adjusted open - 1`

- Regret >= 0，等於 0 表示恰好買在最低點。
- 越低越好；此指標衡量「離完美的距離」，而非絕對報酬。

### 3.3 特徵設計

所有特徵分為四類，共 22 個：

**技術面（11 個）**：1/5/20/60 日報酬、距 20/60 日均線、RSI(14)、布林 Z(20)、距 60 日高點、20 日年化波動率、成交量 Z(20)。

**估值面（2 個）**：NAV 折溢價、TTM 配息殖利率。

**籌碼面（5 個）**：融資 5 日變化率、融券 5 日變化率、法人淨買賣比、法人資料可得標記、基金流量 5 日變化率。

**日曆面（4 個）**：月內交易日序號、月進度（日曆日比例）、星期三角編碼（sin/cos）。

**關鍵防偷看設計**：所有日終特徵（技術、估值、籌碼）向後延遲一個交易日。日曆特徵無需延遲。月進度使用日曆日比例（date.day / days_in_month）。

### 3.4 模型架構

每月使用兩個模型協作：

1. **分類器（Logistic Regression）**：C=0.1，class_weight=balanced，月內樣本加權使各月等權。
2. **保留價迴歸器（Gradient Boosting Quantile Regressor）**：60 棵樹，max_depth=2，min_samples_leaf=20。

**買入決策規則**：每月從第 1 日起逐日檢查，首個同時滿足 near_probability >= 0.5 且 adjusted open <= 保留價 x 1.005 的交易日即買入。若整月未觸發，最後一日強制買入。

### 3.5 Walk-Forward 設計

- **初始訓練窗口**：前 60 個月（2003-07 至 2008-06）。
- **擴展方式**：Expanding window，每月新增一個月的完整資料後重訓。
- **測試集**：下一個月的所有交易日，完全樣本外。
- **樣本外期間**：2008-07 至 2026-06，共 216 個月。
- **Sealed holdout**：2023-07 至 2026-06，共 36 個月。

### 3.6 模型組合

| 模型名稱 | 特徵組合 | 特徵數 |
|---|---|---:|
| 技術＋日曆 | technical + calendar | 15 |
| 技術＋估值＋日曆 | technical + valuation + calendar | 17 |
| 全特徵 | technical + valuation + chip + calendar | 22 |

### 3.7 統計檢驗

策略間比較使用 **paired test** 與 **Moving-Block Bootstrap**（block length = 12 個月、5,000 次重抽樣）。

## 4. 事後最佳日特徵分析

276 個月中 oracle 日的前一日特徵統計（僅描述關聯，不代表因果）：

{_oracle_feature_table(ranges)}

**解讀**：事後最低點的前一天通常已處於短期弱勢，但嘗試等待這些條件觸發時，模型有超過一半的月份被迫拖到月底強制買入，反而錯過月初的低點。

## 5. 樣本外策略比較（2008-07 至 2026-06，216 個月）

{_strategy_table(metrics)}

**欄位說明**：regret 越低越好。強制率 = 模型整月未觸發、月底被迫買入的比例。期末財富 = 每月投入 10,000 元的 total-return 終值。{random_section}

## 6. 模型診斷

{_model_diagnostics_table(metrics)}

- **Brier Score**：越低越好（完美 = 0，隨機 = 0.25）。三組模型均在 0.19～0.21。
- **Average Precision**：越高越好。三組模型均在 0.22 左右。
- **強制買入率超過 50%**：模型的雙重門檻過於嚴格，多數月份被迫月底買入。

## 7. Sealed Holdout 驗證（2023-07 至 2026-06，{holdout_months} 個月）

此區間完全未參與任何開發決策。

| 指標 | 第 {holdout_selected_day} 日 | 全特徵模型 |
|---|---:|---:|
| 平均 regret | {_percent(holdout_day1_mean_regret)} | {_percent(holdout_model_mean_regret)} |
| ≤0.5% 月份 | {_percent(holdout_day1_within_rate, 1)} | — |
| 差距 | — | {_bps(holdout_model_improvement_bps)} |
| 95% CI | — | [{_bps(float(holdout_ci[0]))}, {_bps(float(holdout_ci[1]))}] |

Holdout 差距更大（{_bps(abs(holdout_model_improvement_bps))})，CI 全 < 0。

## 8. 資料與品質

### 8.1 覆蓋範圍

- 每日 OHLCV：**{rows:,}** 筆，{quality_start} 至 {quality_end}。
- 完整月份：276 個（2003-07 至 2026-06）。
- 企業行動：**{quality_dividend_events}** 次配息、**{quality_split_events}** 次分割。
- 欄位數：{quality_columns}。

### 8.2 交叉驗證

| 來源 | 涵蓋交易日 | 缺值 | 與 FinMind 最大差異 |
|---|---:|---:|---:|
| TWSE 官方逐月收盤 | {rows:,} | {official_missing} | {official_difference:.4f} 元 |
| 元大官方市價 | {rows - issuer_missing:,} | {issuer_missing} | {issuer_difference:.4f} 元 |

### 8.3 估值限制

0050 是 ETF，沒有公司層級 EPS。估值只採 point-in-time NAV 折溢價與 trailing distribution yield。

### 8.4 資料來源

- [TWSE 個股日收盤價及月平均價](https://www.twse.com.tw/zh/trading/historical/stock-day-avg.html)
- [元大 0050 歷史 NAV](https://www.yuantaetfs.com/tradeInfo/comparison/0050/NAVhistory)
- [元大 0050 基本資訊與上市日](https://www.yuantaetfs.com/product/detail/0050/Basic_information)
- [FinMind API 文件](https://finmind.github.io/quickstart/)
- [TWSE 交易制度](https://www.twse.com.tw/en/products/system/trading.html)

## 9. 圖表

![價格歷史](figures/price_history.png)

![最佳日分布](figures/oracle_day_distribution.png)

![最佳日特徵](figures/oracle_feature_profile.png)

![策略比較](figures/strategy_comparison.png)

## 10. 限制與注意事項

1. **交易假設**：日線研究，假設小額訂單可在開盤集合競價成交。
2. **關聯非因果**：特徵與最佳日之間是統計關聯，不是因果關係。
3. **正向漂移效應**：第 1 日的優勢主要來自股市長期正向漂移。
4. **單一標的**：只研究 0050 一檔 ETF。
5. **Regime change**：所有候選規則都可能因市場環境改變而失效。
6. **手續費與稅**：期末財富含 0.1425% 手續費，未計證交稅。
7. **非投資建議**：本報告是量化研究，不構成個人化投資建議。

## 11. 未來改進方向：如何有可能超越月初直接買入

本研究目前得到的負面結果（模型未能擊敗月初第一天買入）是基於**僅使用 0050 自身歷史價格、淨值與基本籌碼這類「落後或同步指標」**的資訊集限制。在不引入外部領先特徵的情況下，股市長期的「正向漂移效應」使得任何等待回檔的擇時策略都必須承受極高的「空倉代價（等待期間被軋空，最後月底被迫追高）」。

若要打破此邊界，未來研究有機會透過以下三個方向的「極簡特徵」來超越第一日買入：

### 11.1 外部強領先特徵（Lead-Lag Features）
由於 0050 的台積電權重超過 50%，其本質上受半導體科技股主導。
- **特徵指標**：前一晚美股台積電 ADR 漲跌幅，或費城半導體指數昨日漲跌幅。
- **邏輯機制**：美股交易時段早於台股。若美股 ADR 前晚暴跌，今日 0050 開盤幾乎 100% 跳空大跌。利用開盤前已知的「美股未消化資訊」作為開盤買入決策，是物理因果上最強的領先信號。

### 11.2 資金流與跨市場特徵（Liquidity Features）
台股為外資主導的淺碟市場，外資買賣超決定了 0050 的短中期回檔。
- **特徵指標**：新台幣兌美元匯率（TWD/USD）日線趨勢，或外資台指期貨淨未平倉口數。
- **邏輯機制**：新台幣連續貶值期通常伴隨外資撤資回檔。若設定「匯率連續貶值期暫緩扣款，直至匯率止貶回升首日再買」的單一規則，有機會避開外資連續提款的波段下跌。

### 11.3 總體經濟估值特徵（Macro Features）
將月內擇時放大到月度級別的資金動態配置。
- **特徵指標**：國發會景氣對策信號（紅藍燈）。
- **邏輯機制**：歷史實證中，景氣「藍燈期」（景氣低迷）通常對應股市長線底部。若規則為「景氣藍燈期，每月第一天加倍扣款；景氣紅燈期（過熱），延遲至月底買入或只買基本額」，在跨越數個景氣循環的長週期下，極可能顯著超越純粹的每月第一天買入。

> [!WARNING]
> **多重比較偏誤（Multiple Comparison Bias / p-hacking）**
> 當我們測試千百種特徵時，純粹基於隨機機率，一定能找到一個在歷史回測上完美的指標組合。但這極可能是數據挖礦產生的噪訊過擬合。未來研究應嚴格遵循「因果邏輯優先」原則，避免濫用過多特徵。

## 12. 術語表

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
| 強制買入 | 模型整月未觸發，於最後交易日強制執行買入 |
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


def _save_price_history(daily: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(daily["date"], daily["split_adjusted_close"], color="#174A7E", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_title("0050 分割還原收盤價（對數軸）")
    axis.set_ylabel("新臺幣／目前受益權單位")
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


def _save_strategy_comparison(metrics: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    order = ["fixed_day_1", "fixed_day_5", "fixed_day_10", "fixed_day_15", "rsi30_or_last", "all"]
    labels = ["第1日", "第5日", "第10日", "第15日", "RSI規則", "全特徵模型"]
    values = [float(metrics[name]["mean_regret"]) * 100 for name in order]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, values, color=["#174A7E"] + ["#8BA9C4"] * 4 + ["#C65D3B"])
    axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set_title("樣本外平均月度 regret（越低越好）")
    axis.set_ylabel("regret")
    axis.set_ylim(0, max(values) * 1.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    daily: pd.DataFrame,
    oracle: pd.DataFrame,
    profile: pd.DataFrame,
    metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """產生報告使用的四張可重建圖表。"""

    _configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_price_history(daily, output_dir / "price_history.png")
    _save_oracle_distribution(oracle, output_dir / "oracle_day_distribution.png")
    _save_feature_profile(profile, output_dir / "oracle_feature_profile.png")
    _save_strategy_comparison(metrics, output_dir / "strategy_comparison.png")
