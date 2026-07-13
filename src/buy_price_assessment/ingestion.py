"""下載、交叉驗證並組裝 0050 每日資料。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from buy_price_assessment.adjustments import build_adjusted_prices
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
            }
            frames = {name: future.result() for name, future in futures.items()}
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
