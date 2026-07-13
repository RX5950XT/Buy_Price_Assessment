# 0050 每月最佳買點研究

本專案已完成 0050 自上市以來的每日資料管線與無前視偏誤的 walk-forward 研究，回答「每月只買一次時，哪一天與哪種價格條件較合理」。

## 結論

- 最簡單且樣本外最穩健的規則是：**每月第 1 個交易日買入**；前 1～5 個交易日皆合理，但沒有可靠的神奇單日。
- 276 個完整月份中，事後最佳日的眾數是第 1 日、中位數是第 6 日；48.2% 落在前 5 個交易日。
- 2008-07～2026-06 共 216 個樣本外月份，第 1 日平均離當月最低 adjusted open 3.20%；技術＋估值＋籌碼模型為 4.36%。模型平均差 116 bps，moving-block bootstrap 95% CI 為 -196～-39 bps。
- 歷史最佳日常見特徵是前 5 日報酬 -1.59%、低於 20 日均線 1.42%、RSI(14) 41.1、距 60 日高點 -5.20%；但等待這些訊號在樣本外反而較差。
- 最佳日開盤相對已調整權息／分割的前收，中位數為 -0.13%，四分位區間 -0.74%～+0.09%。跨 23 年不應使用單一固定新臺幣價格。

完整數據、限制與圖表見 [`reports/0050_buy_point_analysis.md`](reports/0050_buy_point_analysis.md)。

## 資料覆蓋

- 5,665 個唯一交易日，2003-06-30～2026-07-09，含原始 OHLCV、分割還原價與 total-return adjusted price。
- TWSE 官方逐月收盤涵蓋全部 5,665 日，與 FinMind 逐日最大差異 0 元。
- 元大官方 NAV／市價涵蓋 5,664 日，只缺 2003-12-31，保留缺值且不插值。
- 2025-06-18 的 1 拆 4、31 次已完成配息，以及融資融券與三大法人資料皆納入。
- 0050 沒有可用的歷史 ETF P/E／P/B 序列；估值只採前日 NAV 折溢價與 trailing distribution yield，不用現今資料回填歷史。

來源：[TWSE 日收盤](https://www.twse.com.tw/zh/trading/historical/stock-day-avg.html)、[元大 0050 歷史 NAV](https://www.yuantaetfs.com/tradeInfo/comparison/0050/NAVhistory)、[元大基本資訊](https://www.yuantaetfs.com/product/detail/0050/Basic_information)、[FinMind API](https://finmind.github.io/quickstart/)。

## 執行

需求為 Python 3.12 與 [uv](https://docs.astral.sh/uv/)。

```powershell
# 安裝與同步依賴
uv sync --all-groups

# 執行完整管線（下載、分析、建模、回測與報告生成）
uv run buy-price-assessment all --validate-all-twse-months --force-models

# 執行靜態檢查與單元測試
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

個別階段：

```powershell
uv run buy-price-assessment fetch --validate-all-twse-months
uv run buy-price-assessment analyze
uv run buy-price-assessment analyze --force-models
```

TWSE 全量驗證採單線節流與逐月快取；後續可續傳。模型預測快取同時驗證日期、特徵值、feature set、模型設定與 cache version，資料或設定變更時會自動失效。

## 主要產物

| 路徑 | 內容 |
|---|---|
| `data/processed/0050_daily.csv` | 每日市場、企業行動、NAV、籌碼與三種價格序列 |
| `data/processed/0050_features.csv` | 每日事前特徵值 |
| `data/processed/0050_monthly_oracle.csv` | 每月事後最佳日期與當時實際原始開盤價 `oracle_open` |
| `data/processed/0050_labeled_daily.csv` | 每日 regret、near-optimal 與事前特徵 |
| `data/processed/oracle_feature_profile.csv` | 事後最佳日的前一日特徵在當月內的偏移統計 |
| `data/processed/walk_forward_predictions.csv` | 完全樣本外的每日模型預測值 (機率與保留價) |
| `data/processed/walk_forward_purchases.csv` | 三組模型每月唯一樣本外買點決策 |
| `reports/research_results.json` | 完整統計、信賴區間與隨機策略模擬結果 |
| `reports/0050_buy_point_analysis.md` | 繁體中文研究報告與圖表 |

## 方法摘要

- 每月 oracle 定義為最低 total-return adjusted **開盤價**；選 open 是因為日內 low 無法保證成交，同價取最早日。
- 所有技術、NAV、籌碼特徵至少 lag 一個交易日；月進度只使用當下已知的日曆日期。
- 初始訓練 60 個月，其後逐月 expanding-window 重訓；正式績效只計樣本外月份，最後 36 個月另作 sealed holdout。
- 基準包含第 1／5／10／15／最後交易日、RSI 規則與隨機日期；主要指標為相對當月最低價的 regret。

本研究不含 bid-ask quote、分鐘 VWAP 與大額滑價，亦非個人化投資建議。
