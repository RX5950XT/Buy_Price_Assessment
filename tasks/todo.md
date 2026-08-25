# 待辦事項

## 當前任務：文件寫入剩餘假設

- [x] 報告產生器寫入凍結的剩餘假設（月度金額、體制閘門、隔夜期貨、不要再做）
- [x] 用 JSON 重渲 0050／VT 報告，避免下次 analyze 蓋掉
- [x] README／規格／CONTEXT／lessons／AGENTS／CLAUDE
- [x] 測試與 ruff／mypy／pytest

## 上一任務：VT 複用同一套月內買一次協議

- [x] FinMind USStockPrice 解析完整 OHLC + Adj_Close；adjusted_open = open × (adj_close / close)
- [x] VT 主表組裝（無 NAV／籌碼）；主模型只跑技術＋日曆
- [x] CLI `--symbol VT`：下載與分析不覆寫 0050 產物
- [x] 同一協議：walk-forward 60 月、holdout 2023-07～2026-06、6 組政策、6 條領先規則、第 5 日截止預先指定、T-1 as-of
- [x] 跑分析、寫 `reports/VT_buy_point_analysis.md`，對照 0050 是否同樣未贏第 1 日
- [x] 缺領先序列 fail-closed；第 5 日截止說明用 0050 的 48.2%
- [x] ruff / mypy / pytest，更新 README／規格／CONTEXT

## 回顧 (Review)

1. **日頻指標幾乎沒機會贏第 1 日**：0050 與 VT 同一失敗機制。剩餘先驗是月度金額（另一個問題）、月初體制閘門、真正隔夜期貨（須搭配閘門）。
2. **不要再做**：RSI／均線、掃 dump 門檻、同市場 T-1 正負號、盤中低點。
3. **未測不得宣稱會贏**。CI 必須全數 > 0。
4. **VT 同樣沒贏第 1 日**：樣本外 157 個月，第 1 日 2.75%、模型 3.12%（−37 bps，CI 跨 0）。holdout 模型顯著較差（−95 bps，CI 全 < 0）。
5. **點估計小幅領先不是贏**：SOX +6 bps、僅機率 +5 bps 都跨 0。
