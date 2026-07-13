# 待辦事項

## 當前任務：增強研究報告精確度與詳細度

- [x] 閱讀現有報告與結構化數據
- [x] 閱讀 reporting.py 報告生成邏輯
- [x] 修改 `render_report()` 增加以下內容：
  - [x] 研究方法論完整說明（label 定義、walk-forward 設計、模型選擇理由）
  - [x] 策略比較表增加 P75/P90/P95 regret、1% 內月份、強制買入率、期末財富
  - [x] oracle 特徵完整表格含 Q25/Q75/mean/n
  - [x] 三組模型分別的 Brier Score 與 Average Precision
  - [x] 隨機策略分佈結果
  - [x] Holdout 區間詳細數據
  - [x] oracle 分佈的 first_10_rate
  - [x] 加入名詞解釋 / 術語表
- [x] 修改 `_strategy_table()` 增加欄位
- [x] 執行 ruff format/check、mypy、pytest 驗證
- [x] 重跑 `uv run buy-price-assessment analyze` 更新報告
- [x] 更新 CONTEXT.md

## 回顧 (Review)

1. **增強報告詳細度**：`reports/0050_buy_point_analysis.md` 已被大幅擴展。新增了 4 大類 X 22 個特徵的詳細定義、雙重門檻模型的決策公式、Expanding window 的 walk-forward 流程、Paired Moving-block bootstrap 統計檢驗設計。
2. **數據完整性**：報告中的統計表現在非常完整，包含：
   - 策略對比：加入了 P75/P90/P95 regret、≤1% regret 月份率、模型觸發強制買入比例、期末財富（ fraction units 累積價值模擬）。
   - 特徵畫像：列出 14 個核心特徵的前一日觀測值之 Q25、中位數、Q75、平均、有效月份。
   - 模型診斷：列出三組模型組合（15, 17, 22 個特徵）在測試集上的 Brier Score（~0.20）與 Average Precision（~0.22），揭示了預測能力接近隨機、強制觸發月底買入（>50%）是 ML 策略落敗的根本原因。
   - 隨機對照：隨機策略的 5,000 次模擬平均 regret 爲 3.77% [3.43%, 4.11%]，顯著差於第 1 交易日的 3.20%，有力支持了月初買入的結構性優勢。
   - 獨立驗證：Holdout 區間（2023-07～2026-06）的 regret 與 CI 對比。
3. **防禦性程式碼與單元測試**：在 `reporting.py` 提取 results 與 metrics 子字典欄位時，全面重構為防呆提取，即使測試數據 mock 沒有填入部分欄位（例如 `mean_day`, `model_ci95_bps`, `dividend_events` 等），也能提供合適的 defaults 並順利渲染。單元測試與覆蓋率（81.46%）均通過。
