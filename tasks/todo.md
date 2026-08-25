# 待辦事項

## 當前任務：VT 複用同一套月內買一次協議

- [x] FinMind USStockPrice 解析完整 OHLC + Adj_Close；adjusted_open = open × (adj_close / close)
- [x] VT 主表組裝（無 NAV／籌碼）；主模型只跑技術＋日曆
- [x] CLI `--symbol VT`：下載與分析不覆寫 0050 產物
- [x] 同一協議：walk-forward 60 月、holdout 2023-07～2026-06、6 組政策、6 條領先規則、第 5 日截止預先指定、T-1 as-of
- [x] 跑分析、寫 `reports/VT_buy_point_analysis.md`，對照 0050 是否同樣未贏第 1 日
- [x] 缺領先序列 fail-closed；第 5 日截止說明用 0050 的 48.2%
- [x] ruff / mypy / pytest，更新 README／規格／CONTEXT

## 上一任務：文件／報告／README 對齊後推送

- [x] 修正匯率表頭：USD/TWD 升值 = 臺幣貶值
- [x] 報告與 README 寫明月內擇時未贏第 1 日
- [x] 整合測試無條件要求 6 條領先規則
- [x] 驗證、commit、push

## 回顧 (Review)

1. **VT 同樣沒贏第 1 日**：樣本外 157 個月，第 1 日 2.75%、模型 3.12%（−37 bps，CI 跨 0）。holdout 模型顯著較差（−95 bps，CI 全 < 0）。
2. **政策與領先規則**：6+6 條樣本外 CI 均未全數 > 0。SOX +6 bps、僅機率 +5 bps 都跨 0，不是贏。
3. **協議對齊**：第 5 日截止說明必須用 0050 已公布的 48.2%，不可改寫成 VT 的 43.3%。TSM／SOX 對 VT 是同一市場 T-1。
4. **缺領先序列不可靜默跳過**。
5. **未做**：景氣燈（月度金額）。
