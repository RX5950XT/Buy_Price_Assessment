# 開發交接

## 目標

建立可重跑的 0050 研究管線，取得自上市以來每日市場資料，分析每月只買一次時的事後最佳買點特徵，並用無前視偏誤的 walk-forward 驗證可執行規則。VT 用同一協議複製，檢驗結論是否為單一標的特例。

## 目前狀態 (2026-08-25 更新)

- 0050 管線完成；VT 複製實驗完成，結論相同：月內擇時沒有證據贏第 1 日。
- CLI：`uv run buy-price-assessment analyze --symbol VT`。產物在 `data/processed/vt/`、`reports/VT_buy_point_analysis.md`，不覆寫 0050。
- 領先資料缺檔必須失敗。VT fetch 會一併抓 TSM／SOX／USD/TWD。`data/raw/` 不進 git。
- 美股／匯率 as-of：目標日 T 只用來源日 `<= T-1`。VT 的 TSM／SOX 是同一市場前一交易日，不是跨市場隔夜。
- 第 5 日截止沿用 0050 已公布的 48.2%，不依 VT 的 43.3% 重估。
- `0050_daily.csv`：5,665 日，2003-06-30～2026-07-09；完整月份 276；樣本外 216；holdout 36。
- `VT_daily.csv`：4,568 日，2008-06-26～2026-08-24；完整月份 217（2008-07～2026-07）；樣本外 157（2013-07～2026-07）；holdout 36。

## 研究結論

- **0050 可執行基準仍是第 1 交易日**，樣本外 mean regret 3.20%；全特徵模型 4.36%，差 116 bps，CI [−196, −39]。
- **VT 同樣是第 1 日**：樣本外 2.75%；技術＋日曆模型 3.12%，−37 bps，CI [−96, 22] 跨 0。holdout 第 1 日 2.24%、模型 3.18%，−95 bps，CI 全 < 0。
- 兩檔的 6 組政策與 6 條領先規則樣本外 CI 均未全數 > 0。VT 的 SOX 規則 +6 bps、僅機率 +5 bps，CI 都跨 0，不是贏。
- 在現有約束下，月內擇時不太可能穩定贏第 1 日；這不是 0050 特例。
- 尚未做：景氣燈（月度金額，不是月內選日；未回測，不得預設會贏）。

## 重要實作決策

- 隔夜大跌門檻預先指定 1%，不是從樣本外 dump 分布挑的。
- 領先特徵不再 `shift(1)`：對齊本身已是開盤前資訊。
- 預測快取含特徵指紋。完整月份截止以資料末日推算。

## 驗證與重跑

```powershell
uv run buy-price-assessment all --validate-all-twse-months --force-models
uv run buy-price-assessment all --symbol VT
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

0050 正式報告 `reports/0050_buy_point_analysis.md`；VT `reports/VT_buy_point_analysis.md`。

## 重要原則

- 原始成交價用於模擬實際買入股數；還原價用於跨期報酬與特徵比較。
- 當月最低點是研究 label，不可直接作為即時交易訊號。
- 所有可執行結果必須使用按時間切分的樣本外預測。
- 完整月份截止以資料末日推算。
- 跨市場序列以目標日 T-1 為 as-of。
- 截止日必須限制搜尋窗口。
