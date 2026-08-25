"""下載、交叉驗證並組裝 0050 每日資料。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from buy_price_assessment.adjustments import build_adjusted_prices
from buy_price_assessment.clients import DataSourceError
from buy_price_assessment.repositories import FinMindRepository, TwseRepository, YuantaRepository


def _distribution_yield(
    daily: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.Series:
    if dividends.empty:
        return pd.Series(0.0, index=daily.index)
    dividend_rows = [
        (pd.Timestamp(values[0]), float(values[1]))
        for values in dividends.loc[:, ["date", "cash_dividend"]].to_numpy().tolist()
    ]
    split_rows = [
        (pd.Timestamp(values[0]), float(values[1]))
        for values in splits.loc[:, ["date", "split_ratio"]].to_numpy().tolist()
    ]
    yields: list[float] = []
    for date_value, close_value in zip(daily["date"], daily["close"], strict=True):
        current_date = pd.Timestamp(date_value)
        current_close = float(close_value)
        cutoff = current_date - pd.Timedelta(days=365)
        total = 0.0
        for dividend_date, cash_dividend in dividend_rows:
            if cutoff < dividend_date <= current_date:
                later_splits = [
                    split_ratio
                    for split_date, split_ratio in split_rows
                    if dividend_date < split_date <= current_date
                ]
                total += cash_dividend / float(np.prod(later_splits) if later_splits else 1.0)
        yields.append(total / current_close)
    return pd.Series(yields, index=daily.index)


def _merge_optional(base: pd.DataFrame, extra: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if extra.empty:
        result = base.copy()
        for column in columns:
            result[column] = np.nan
        return result
    return base.merge(
        extra.loc[:, ["date", *columns]], on="date", how="left", validate="one_to_one"
    )


def _validate_cross_sources(daily: pd.DataFrame, tolerance: float = 0.011) -> None:
    missing_official = int(daily["official_close"].isna().sum())
    if missing_official / len(daily) > 0.001:
        raise ValueError(f"官方收盤價缺少 {missing_official} 個交易日")
    if daily["close_difference"].dropna().abs().max() > tolerance:
        worst = daily.loc[daily["close_difference"].abs().idxmax()]
        raise ValueError(
            f"官方收盤價交叉驗證失敗：{worst['date']:%Y-%m-%d} 差 {worst['close_difference']:.4f}"
        )
    missing_nav = int(daily["nav"].isna().sum())
    if missing_nav / len(daily) > 0.001:
        raise ValueError(f"元大官方 NAV 缺少 {missing_nav} 個交易日")
    issuer_difference = (daily["issuer_market_price"] - daily["close"]).dropna()
    if issuer_difference.abs().max() > tolerance:
        raise ValueError("元大官方市價與每日收盤價不一致")


def assemble_daily_data(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    nav: pd.DataFrame,
    margin: pd.DataFrame,
    institutions: pd.DataFrame,
    official_closes: pd.DataFrame,
) -> pd.DataFrame:
    """合併各來源並 fail-closed 驗證每日資料。"""

    result = build_adjusted_prices(prices, dividends, splits)
    result = result.merge(nav, on="date", how="left", validate="one_to_one")
    result = _merge_optional(result, margin, ["margin_balance", "short_balance"])
    result = _merge_optional(
        result,
        institutions,
        ["institutional_net", "foreign_net", "trust_net", "dealer_net"],
    )
    result = result.merge(official_closes, on="date", how="left", validate="one_to_one")
    result["close_difference"] = result["close"] - result["official_close"]
    result["institutional_available"] = result["institutional_net"].notna()
    for column in ("institutional_net", "foreign_net", "trust_net", "dealer_net"):
        result[column] = result[column].fillna(0.0)
    result["distribution_yield_ttm"] = _distribution_yield(result, dividends, splits)
    result["feature_available_date"] = result["date"].shift(-1)
    _validate_cross_sources(result)
    return result.sort_values("date", ignore_index=True)


def _write_raw_frames(raw_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(raw_dir / f"0050_{name}.csv", index=False, encoding="utf-8-sig")


def _write_lead_frames(raw_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = {"tsm": "tsm_us.csv", "sox": "sox_us.csv", "fx": "usd_twd.csv"}
    for key, filename in names.items():
        frames[key].to_csv(raw_dir / filename, index=False, encoding="utf-8-sig")


def infer_us_actions(close: pd.Series, adj_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """由原始收盤與 vendor Adj_Close 推估除息與分割；無法解釋的大跳動則失敗。"""

    cash = pd.Series(0.0, index=close.index, dtype=float)
    split = pd.Series(1.0, index=close.index, dtype=float)
    close_gross = close.astype(float) / close.shift(1).astype(float)
    adj_gross = adj_close.astype(float) / adj_close.shift(1).astype(float)
    implied = adj_gross / close_gross
    for position in range(1, len(close)):
        value = float(implied.iloc[position])
        if not np.isfinite(value):
            raise ValueError("Adj_Close 與收盤價無法計算企業行動")
        if value >= 1.4:
            nearest = float(round(value))
            if nearest < 2.0:
                raise ValueError(f"無法辨識的分割比率：{value}")
            split.iloc[position] = nearest
        elif value > 1.002:
            previous_close = float(close.iloc[position - 1])
            cash.iloc[position] = previous_close * (1.0 - 1.0 / value)
            if cash.iloc[position] < 0.0:
                raise ValueError("推估配息為負")
        elif value < 0.95:
            raise ValueError(f"無法由 Adj_Close 解釋的價格跳動：implied={value:.4f}")
    return cash, split


def assemble_us_etf_daily(prices: pd.DataFrame) -> pd.DataFrame:
    """用 vendor Adj_Close 當 total-return 還原，組出與特徵管線相容的每日表。"""

    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"美股每日資料缺少欄位：{sorted(missing)}")
    result = prices.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values("date", ignore_index=True)
    if result["date"].duplicated().any():
        raise ValueError("美股每日資料含重複日期")
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("美股 OHLC 與成交量必須是數字")
    result.loc[:, numeric_columns] = numeric
    if (result.loc[:, ["open", "high", "low", "close", "adj_close"]] <= 0.0).any().any():
        raise ValueError("美股價格必須為正")
    if (result["volume"] < 0.0).any():
        raise ValueError("成交量不可為負")
    ohlc = result.loc[:, ["open", "high", "low", "close"]]
    if (ohlc["high"] < ohlc[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("最高價低於其他 OHLC 價格")
    if (ohlc["low"] > ohlc[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("最低價高於其他 OHLC 價格")

    factor = result["adj_close"] / result["close"]
    if (factor <= 0.0).any() or not np.isfinite(factor.to_numpy(dtype=float)).all():
        raise ValueError("還原因子無效")
    result["adjustment_factor"] = factor
    for column in ("open", "high", "low"):
        result[f"adjusted_{column}"] = result[column] * factor
    result["adjusted_close"] = result["adj_close"]
    first_adj = float(result["adj_close"].iloc[0])
    result["total_return_index"] = 100.0 * result["adj_close"] / first_adj
    cash, split = infer_us_actions(result["close"], result["adj_close"])
    result["cash_dividend"] = cash
    result["split_ratio"] = split
    reverse_product = result["split_ratio"].iloc[::-1].cumprod().iloc[::-1]
    future_splits = reverse_product / result["split_ratio"]
    for column in ("open", "high", "low", "close"):
        result[f"split_adjusted_{column}"] = result[column] / future_splits
    result["gross_return"] = result["adj_close"].pct_change().fillna(0.0) + 1.0
    result["trading_value"] = result["volume"] * result["close"]
    result["nav"] = result["close"]
    result["issuer_market_price"] = result["close"]
    result["official_close"] = result["close"]
    result["official_close_source"] = "FinMind USStockPrice"
    result["close_difference"] = 0.0
    result["margin_balance"] = 0.0
    result["short_balance"] = 0.0
    result["institutional_net"] = 0.0
    result["foreign_net"] = 0.0
    result["trust_net"] = 0.0
    result["dealer_net"] = 0.0
    result["institutional_available"] = False
    result["outstanding_units"] = 1.0
    dividends = result.loc[result["cash_dividend"] > 0.0, ["date", "cash_dividend"]]
    splits = result.loc[result["split_ratio"] != 1.0, ["date", "split_ratio"]]
    result["distribution_yield_ttm"] = _distribution_yield(result, dividends, splits)
    result["feature_available_date"] = result["date"].shift(-1)
    return result.sort_values("date", ignore_index=True)


def download_us_etf_data(
    *,
    symbol: str = "VT",
    start: date = date(2008, 6, 24),
    end: date | None = None,
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """下載美股 ETF 日線；目前只支援 VT。"""

    if symbol != "VT":
        raise ValueError(f"尚未支援的美股標的：{symbol}")
    final_date = end or date.today()
    headers = {"User-Agent": "BuyPriceAssessment/0.1 research-contact-none"}
    timeout = httpx.Timeout(45.0, connect=15.0)
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        finmind = FinMindRepository(client)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "prices": executor.submit(finmind.us_ohlc_prices, symbol, start, final_date),
                "tsm": executor.submit(finmind.us_adj_prices, "TSM", start, final_date),
                "sox": executor.submit(finmind.us_adj_prices, "^SOX", start, final_date),
                "fx": executor.submit(finmind.usd_twd, start, final_date),
            }
            frames = {name: future.result() for name, future in futures.items()}
    prices = frames["prices"]
    if prices.empty:
        raise DataSourceError(f"FinMind USStockPrice {symbol} 沒有資料")
    raw_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(raw_dir / "vt_us.csv", index=False, encoding="utf-8-sig")
    _write_lead_frames(
        raw_dir,
        {"tsm": frames["tsm"], "sox": frames["sox"], "fx": frames["fx"]},
    )
    return assemble_us_etf_daily(prices)


def download_daily_data(
    *,
    start: date = date(2003, 6, 30),
    end: date | None = None,
    raw_dir: Path = Path("data/raw"),
    validate_all_twse_months: bool = False,
) -> pd.DataFrame:
    """下載 0050 自上市至指定日的價格、NAV 與籌碼資料。"""

    final_date = end or date.today()
    headers = {"User-Agent": "BuyPriceAssessment/0.1 research-contact-none"}
    timeout = httpx.Timeout(45.0, connect=15.0)
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        finmind = FinMindRepository(client)
        yuanta = YuantaRepository(client)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                "prices": executor.submit(finmind.prices, start, final_date),
                "dividends": executor.submit(finmind.dividends, start, final_date),
                "splits": executor.submit(finmind.splits, start, final_date),
                "margin": executor.submit(finmind.margin, start, final_date),
                "institutions": executor.submit(finmind.institutional, start, final_date),
                "nav": executor.submit(yuanta.nav, start, final_date),
                "tsm": executor.submit(finmind.us_adj_prices, "TSM", start, final_date),
                "sox": executor.submit(finmind.us_adj_prices, "^SOX", start, final_date),
                "fx": executor.submit(finmind.usd_twd, start, final_date),
            }
            frames = {name: future.result() for name, future in futures.items()}
        lead = {name: frames.pop(name) for name in ("tsm", "sox", "fx")}
        _write_lead_frames(raw_dir, lead)
        if validate_all_twse_months:
            twse = TwseRepository(client, cache_dir=raw_dir / "twse_monthly_close")
            frames["official"] = twse.closes(start, final_date + timedelta(days=1))
            official_source = "TWSE_STOCK_DAY_AVG"
        else:
            frames["official"] = (
                frames["nav"]
                .loc[:, ["date", "issuer_market_price"]]
                .rename(columns={"issuer_market_price": "official_close"})
            )
            official_source = "YUANTA_ETF_NAV_HISTORY"
    _write_raw_frames(raw_dir, frames)
    result = assemble_daily_data(
        frames["prices"],
        frames["dividends"],
        frames["splits"],
        frames["nav"],
        frames["margin"],
        frames["institutions"],
        frames["official"],
    )
    result["official_close_source"] = official_source
    return result
