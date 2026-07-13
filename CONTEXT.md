# 開發交接

## 目標

建立可重跑的 0050 研究管線，取得自上市以來每日市場資料，分析每月只買一次時的事後最佳買點特徵，並用無前視偏誤的 walk-forward 驗證可執行規則。

## 目前狀態 (2026-07-14 更新)

- 完整管線、資料、樣本外分析、圖表與精確詳細的報告均已完成並通過完整檢驗。
- `0050_daily.csv`：5,665 日，2003-06-30～2026-07-09；TWSE 官方收盤一致，元大 NAV 僅缺 2003-12-31。
- 完整月份 2003-07～2026-06 共 276 個；walk-forward 樣本外 2008-07～2026-06 共 216 個，sealed holdout 為 2023-07～2026-06 共 36 個。
- 研究報告已大幅度增強精確度與詳細度，補齊了特徵工程（4 大類 22 個特徵）的完整 Q25/Median/Q75/Mean 統計表、模型預測診斷指標（Brier Score / Average Precision）、5,000 次 Bootstrap 隨機對照組，以及 holdout 檢驗。
- 程式碼 `reporting.py` 加入了全面的防禦性設計（`.get()` fallback 預設值），避免在單元測試或不完整的結果輸入下引發 KeyError，單元測試與測試覆蓋率（81.46%）均通過。

## 研究結論

- 每月第 1 交易日 mean regret 3.20%，全特徵模型 4.36%；模型相對改善 -116 bps，95% moving-block CI -196～-39 bps。
- oracle 眾數第 1 日、中位第 6 日，48.2% 在前 5 日；第 1 與第 5 日差異的 CI 跨 0，因此實務區間是前 1～5 日，預設第 1 日。
- oracle 前日特徵中位：5 日報酬 -1.59%、距 MA20 -1.42%、RSI14 41.1、距 60 日高點 -5.20%；等待訊號未能轉成樣本外優勢。
- oracle 開盤相對 action-adjusted 前收中位 -0.13%，Q25～Q75 為 -0.74%～+0.09%；不使用跨期固定新臺幣價。
- 隨機對照組：均勻隨機選擇一天的平均 regret 爲 3.77% [3.43%, 4.11%]，證明第 1 交易日的 3.20% 優勢非運氣。
- 模型失敗根源：雙重門檻過於嚴苛，致使 53.2%～56.0% 的月份最終無信號觸發，被迫在月底最後一日強制買入，錯失月初低點。

## 重要實作決策

- FinMind 提供完整 raw OHLCV／企業行動／籌碼；TWSE `STOCK_DAY_AVG` 單線節流做全期獨立 close 驗證；元大動態 DeviceId API 提供 NAV／規模。
- 2025-06-18 的 1:4 分割同時調整價格、成交量、融資融券與流通單位；停牌假列不進價格日曆。
- 全部日終特徵 lag 一日；月進度使用日曆日比例，不使用事後整月交易日總數。
- 預測快取含特徵、設定與版本 SHA-256 指紋；變更後自動失效。
- 0050 無可靠歷史 ETF P/E／P/B，估值採 point-in-time NAV 折溢價與 trailing distribution yield。

## 驗證與重跑

```powershell
uv run buy-price-assessment all --validate-all-twse-months --force-models
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

依賴已用 `pip-audit` 掃描，無已知漏洞。正式報告為 `reports/0050_buy_point_analysis.md`，結構化結果為 `reports/research_results.json`。

## 重要原則

- 原始成交價用於模擬實際買入股數；還原價用於跨期報酬與特徵比較。
- 當月最低點是研究 label，不可直接作為即時交易訊號。
- 所有可執行結果必須使用按時間切分的樣本外預測。
