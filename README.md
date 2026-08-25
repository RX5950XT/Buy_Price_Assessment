# 0050 每月最佳買點研究

本專案已完成 0050 自上市以來的每日資料管線與無前視偏誤的滾動窗口（walk-forward）研究，解答：「對定期定額的投資人，每月只買一次時，選哪一天或哪種價格條件買入最合理」。

---

## 1. 研究結論

- **最佳定期定額基準**：若每月必須且只能買一次，**每月第 1 個交易日直接開盤買入**是目前最穩健、最可執行的基準；前 1～5 個交易日前段買入皆合理，但歷史上並不存在統計顯著的神奇單日。
- **事後最低點分佈**：在 276 個完整月份中，事後最低點出現的眾數是第 1 日、中位數是第 6 日；有 48.2% 落在前 5 個交易日，顯示最低點具有偏向月初的特徵。
- **基準與機器學習對比**：在 2008-07～2026-06 共 216 個樣本外月份中，第 1 日平均 regret 爲 **3.20%**；而全特徵機器學習策略平均 regret 爲 **4.36%**。ML 策略平均比第 1 日差 **116 bps**，95% moving-block bootstrap CI 爲 **[-196 bps, -39 bps]**，在統計上**顯著較差**。
- **獨立 Sealed Holdout 驗證**：在完全隔離且未參與任何開發決策的 holdout 期間（2023-07～2026-06，共 36 個月），第 1 日平均 regret 降至 **2.58%**（且高達 **52.8%** 的月份買在最低價 0.5% 內），而 ML 模型平均 regret 爲 **5.79%**（差距 **321 bps**），結論高度一致。
- **隨機對照基準**：每月均勻隨機選一天買入（5,000 次模擬）的平均 regret 爲 **3.77%** (95% CI [3.43%, 4.11%])。第 1 日（3.20%）顯著優於隨機中位，證實月初的領先優勢並非運氣，而是源於股市長期的「正向漂移效應」（資金在市場裡的時間越長，越早享受平均正報酬）。
- **ML 策略落敗主因**：模型基於「近最佳日機率分類器 + 剩餘月內最低價 Quantile 迴歸器」的雙重決策門檻過於嚴苛，導致 **53.2%～56.0% 的月份整月無信號觸發**，被迫於月底最後一日強制買入，因而錯失月初的價格低價波段。
- **決策規則拆解（不重訓）**：拿掉保留價後強制率降至 7.9%、平均 regret 3.33%；第 5 日截止把雙門檻從 4.36% 拉到 3.35%。六組預先指定政策的樣本外 95% CI 均未顯示優於第 1 日（3.20%）；holdout 點估計仍全為負。
- **外部領先規則（不掃參）**：6 條預先指定規則樣本外 CI 均未贏第 1 日（3.20%）；holdout 點估計全為負。TSM ≥1% 大跌屬過濾（3.40%、強制率 26.4%、平均第 3.0 日，−19 bps）。第 1 日除非隔夜下跌、臺幣單日貶值暫緩仍屬延後買入（皆 3.23%、強制率 2.3%、平均第 1.8 日）。美股與匯率對齊只用臺灣日 T-1 已收盤資料；USD/TWD 上升 = 臺幣貶值。
- **月內擇時判斷**：在「每月必須買一次、開盤成交、只用開盤前已知資訊」的約束下，模型、六組決策規則與六條領先規則都沒有樣本外證據優於第 1 日。理論上約 73% 月份最低點不在第 1 日，但可執行訊號抓不到，等待卻付正漂移成本。剩餘未測是月度扣款金額（景氣燈），不是再掃月內門檻。
- **最佳日特徵畫像**：事後最低點的前一日特徵中位數表現為：5 日報酬 -1.59%、低於 20 日均線 1.42%、RSI(14) 41.1、距 60 日高點回檔 5.20%。但嘗試在樣本外「等待」這些特徵，代價是面臨高額的空倉機會成本。當日開盤相對已調整權息／分割的前收中位數為 -0.13%（四分位數區間為 -0.74% 至 +0.09%），跨 23 年不應使用單一固定新臺幣價格作為觸發點。

完整數據、限制與圖表見 [`reports/0050_buy_point_analysis.md`](reports/0050_buy_point_analysis.md)。

---

## 2. 方法摘要

- **Oracle (事後最低點) 定義**：每月的 oracle 定義為該月 total-return adjusted **開盤價最低** 的那一天。使用開盤價而非盤中最低價，因為開盤集合競價可確保成交，同價取最早日以避免後見偏差。
- **Regret (遺憾值) 指標**：
  $$\text{Regret} = \frac{\text{實際買入價 (adjusted open)}}{\text{當月最低價 (oracle adjusted open)}} - 1 \ge 0$$
  Regret 衡量實際成交價與當月完美最低點的百分比差距，越低越好（0 代表買在最低點）。
- **特徵工程與防偷看設計**：
  - 共設計 22 個事前特徵（11 個技術面、2 個估值面、5 個籌碼面、4 個日曆面）。
  - **防前視偏差（Look-ahead Bias）**：所有日終特徵（技術、估值、籌碼）均向後延遲一個交易日（第 N 天決策時只用第 N-1 天收盤後數據）。月進度使用當日已知的日曆日比例，不用事後交易日總數。
- **模型架構與決策規則**：
  - **分類器（Logistic Regression）**：C=0.1，balanced weight，預測「今天是否為近最低點」。
  - **保留價迴歸器（Gradient Boosting Quantile Regressor）**：60 棵樹，max_depth=2，預測今日至月底剩餘最低價的比率中位數。
  - **買入決策**：首個同時滿足 (1) 分類機率 >= 0.5 且 (2) 當日開盤價 <= 保留價 * 1.005 的交易日即買入；若整月未觸發，月底最後一日強制買入。
- **回測與統計檢驗**：
  - 初始訓練 60 個月，後續採用 **Expanding-window 逐月滾動重訓**；回測與評估完全基於樣本外預測。
  - 策略間 regret 差異檢定使用 **Paired Test** 與 **Moving-Block Bootstrap**（區塊長度 12 個月，5,000 次重抽樣），以克服時序 regret 序列的自相關。

---

## 3. 資料與來源

- **覆蓋範圍**：共 5,665 個唯一交易日（2003-06-30 至 2026-07-09），含原始價、分割還原價與 total-return adjusted 還原價。
- **企業行動還原**：還原 31 次現金配息與 2025-06-18 的 1:4 股票分割（同時調整價格、成交量、融資融券與流通單位）。
- **資料交叉驗證**：
  - **TWSE 官方逐月收盤**：涵蓋全部 5,665 日，與 FinMind 逐日收盤最大差異為 0.0000 元。
  - **元大官方市價/NAV**：涵蓋 5,664 日，只缺 2003-12-31，保留缺值且不插值。
  - **估值限制**：0050 無可用歷史 ETF P/E 或 P/B 序列；估值特徵只採 point-in-time NAV 折溢價與 trailing distribution yield，不用現今資料回填歷史。
- **資料來源**：
  - [TWSE 個股日收盤價及月平均價](https://www.twse.com.tw/zh/trading/historical/stock-day-avg.html)
  - [元大 0050 歷史 NAV 數據庫](https://www.yuantaetfs.com/tradeInfo/comparison/0050/NAVhistory)
  - [元大 0050 基本資訊與上市日](https://www.yuantaetfs.com/product/detail/0050/Basic_information)
  - [FinMind API 數據服務](https://finmind.github.io/quickstart/)
  - [FinMind 美股日線 USStockPrice](https://finmind.github.io/tutor/UnitedStatesMarket/Technical/)（TSM ADR、`^SOX`）
  - [FinMind 臺灣銀行匯率 TaiwanExchangeRate](https://finmind.github.io/tutor/ExchangeRate/)（USD/TWD 即期中間價）
  - [TWSE 交易制度與撮合規則](https://www.twse.com.tw/en/products/system/trading.html)

---

## 4. 執行指引

本專案運行環境固定為 Python 3.12，並使用 [uv](https://docs.astral.sh/uv/) 管理依賴。

```powershell
# 1. 安裝與同步依賴庫
uv sync --all-groups

# 2. 執行完整管線（下載、分析、建模、回測與報告生成）
uv run buy-price-assessment all --validate-all-twse-months --force-models

# 3. 執行程式碼格式化、型別檢查與單元測試
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

個別階段執行命令：

```powershell
# 僅下載並交叉驗證數據（TWSE 單線節流與逐月快取；同時抓 TSM／SOX／USD/TWD）
uv run buy-price-assessment fetch --validate-all-twse-months

# 僅進行回測分析與報告渲染（會驗證 cache 指紋，特徵或設定改變時快取會失效）
uv run buy-price-assessment analyze
uv run buy-price-assessment analyze --force-models
```

---

## 5. 主要產物

| 產物路徑 | 內容說明 |
|---|---|
| `data/processed/0050_daily.csv` | 整理後的每日市場數據、NAV、企業行動、還原價及三大法人籌碼 |
| `data/processed/0050_features.csv` | 每日事前特徵值數據 |
| `data/processed/0050_monthly_oracle.csv` | 每月事後最佳買點日期與 adjusted open 最低價 |
| `data/processed/0050_labeled_daily.csv` | 標記 regret、near-optimal 與事前特徵的每日明細數據 |
| `data/processed/oracle_feature_profile.csv` | 事後最低日前一天的特徵值在當月內的分位數偏移統計 |
| `data/processed/walk_forward_predictions.csv` | 完全樣本外滾動回測的每日模型預測值 (觸發機率與預估保留價) |
| `data/processed/walk_forward_purchases.csv` | 三組機器學習模型每月唯一的樣本外買點決策結果 |
| `data/processed/walk_forward_technical_calendar.csv` | 「技術＋日曆」模型每月樣本外決策買點與 regret 明細 |
| `data/processed/walk_forward_technical_valuation_calendar.csv` | 「技術＋估值＋日曆」模型每月樣本外決策買點與 regret 明細 |
| `data/processed/walk_forward_all.csv` | 「全特徵」模型每月樣本外決策買點與 regret 明細 |
| `reports/research_results.json` | 完整統計指標、信賴區間與隨機策略 5,000 次模擬結果的 JSON 快取 |
| `reports/0050_buy_point_analysis.md` | 重建後的繁體中文深度研究報告與 matplotlib 圖表 |
| `reports/figures/policy_ablation.png` | 六組預先指定買入規則相對第 1 日的樣本外平均 regret |
| `data/raw/tsm_us.csv` | 台積電 ADR 還原收盤（FinMind USStockPrice） |
| `data/raw/sox_us.csv` | 費城半導體指數還原收盤（FinMind `^SOX`） |
| `data/raw/usd_twd.csv` | 美元／新臺幣即期中間價（FinMind TaiwanExchangeRate） |
| `reports/figures/lead_rules.png` | 外部領先規則（原三條＋失敗機制修正版）相對第 1 日的樣本外平均 regret |

`data/raw/` 的 TWSE 月檔、TSM／SOX／USD/TWD 可由 `fetch` 重建，不進 git。分析領先規則前需先抓齊這三份外部序列。

---

*量化研究警告：本專案不含 bid-ask quote、高頻分鐘級滑價與市場衝擊成本。本報告所有內容均為量化學術與歷史數據研究，不構成任何個人化投資建議。*
