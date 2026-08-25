"""0050 市場資料 Repository。"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from buy_price_assessment.clients import (
    DataSourceError,
    extract_yuanta_device_id,
    parse_roc_date,
    validate_finmind_response,
)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TWSE_MONTHLY_CLOSE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_AVG"
YUANTA_PAGE_URL = "https://www.yuantaetfs.com/tradeInfo/comparison/0050/NAVhistory"
YUANTA_API_URL = "https://etfapi.yuantaetfs.com/ectranslation/api/trans"


def _require_columns(rows: Sequence[Mapping[str, Any]], columns: set[str], source: str) -> None:
    if not rows:
        raise DataSourceError(f"{source} 沒有資料")
    missing = columns.difference(rows[0])
    if missing:
        raise DataSourceError(f"{source} 缺少欄位：{sorted(missing)}")


def _normalize_date(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values("date", ignore_index=True)
    if result["date"].duplicated().any():
        raise DataSourceError(f"{source} 含重複日期")
    return result


def _numeric(frame: pd.DataFrame, columns: Sequence[str], source: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.loc[:, list(columns)].isna().any().any():
        raise DataSourceError(f"{source} 含無效數字")
    return result


def parse_us_prices(rows: Sequence[Mapping[str, Any]], *, source: str) -> pd.DataFrame:
    """正規化美股還原收盤；報酬必須用 adj_close，避免分割被當成暴跌。"""

    _require_columns(rows, {"date", "Adj_Close"}, source)
    frame = pd.DataFrame(rows).rename(columns={"Adj_Close": "adj_close"})
    result = _numeric(frame.loc[:, ["date", "adj_close"]], ["adj_close"], source)
    if (result["adj_close"] <= 0.0).any():
        raise DataSourceError(f"{source} 含非正還原收盤價")
    return _normalize_date(result, source)


def parse_usd_twd(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """臺灣銀行即期中間價；-99 與非正數視為缺值。"""

    source = "FinMind TaiwanExchangeRate USD"
    _require_columns(rows, {"date", "spot_buy", "spot_sell"}, source)
    frame = pd.DataFrame(rows)
    buy = pd.to_numeric(frame["spot_buy"], errors="coerce")
    sell = pd.to_numeric(frame["spot_sell"], errors="coerce")
    mid = (buy + sell) / 2.0
    invalid = buy.isna() | sell.isna() | (buy <= 0.0) | (sell <= 0.0)
    frame["usd_twd"] = mid.mask(invalid)
    result = frame.loc[frame["usd_twd"].notna(), ["date", "usd_twd"]]
    if result.empty:
        raise DataSourceError(f"{source} 沒有有效即期匯率")
    return _normalize_date(result, source)


def parse_finmind_prices(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """將 FinMind OHLCV 欄位正規化。"""

    required = {
        "date",
        "Trading_Volume",
        "Trading_money",
        "open",
        "max",
        "min",
        "close",
        "spread",
        "Trading_turnover",
    }
    _require_columns(rows, required, "FinMind TaiwanStockPrice")
    frame = pd.DataFrame(rows).rename(
        columns={
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
            "Trading_money": "trading_value",
            "spread": "price_change",
            "Trading_turnover": "transactions",
        }
    )
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "price_change",
        "transactions",
    ]
    result = _numeric(frame.loc[:, columns], columns[1:], "FinMind TaiwanStockPrice")
    return _normalize_date(result, "FinMind TaiwanStockPrice")


def parse_finmind_dividends(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """解析實際除息結果；空資料保留穩定 schema。"""

    columns = ["date", "cash_dividend"]
    if not rows:
        return pd.DataFrame(columns=columns)
    _require_columns(rows, {"date", "stock_and_cache_dividend", "stock_or_cache_dividend"}, "配息")
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["stock_or_cache_dividend"].astype(str).str.contains("息")].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["cash_dividend"] = pd.to_numeric(frame["stock_and_cache_dividend"], errors="coerce")
    frame = frame.loc[frame["cash_dividend"] > 0, columns]
    return _normalize_date(frame, "FinMind 配息")


def parse_finmind_splits(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """由分割前後參考價推導受益權單位分割比率。"""

    columns = ["date", "split_ratio"]
    if not rows:
        return pd.DataFrame(columns=columns)
    _require_columns(rows, {"date", "type", "before_price", "after_price"}, "分割")
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["type"].astype(str).str.contains("分割")].copy()
    before = pd.to_numeric(frame["before_price"], errors="coerce")
    after = pd.to_numeric(frame["after_price"], errors="coerce")
    ratio = before / after
    nearest = ratio.round()
    frame["split_ratio"] = ratio.where((ratio - nearest).abs() > 0.02, nearest)
    if frame["split_ratio"].isna().any() or (frame["split_ratio"] <= 0).any():
        raise DataSourceError("FinMind 分割比率無效")
    return _normalize_date(frame.loc[:, columns], "FinMind 分割")


def parse_finmind_margin(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    required = {"date", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"}
    _require_columns(rows, required, "FinMind 融資融券")
    frame = pd.DataFrame(rows).rename(
        columns={
            "MarginPurchaseTodayBalance": "margin_balance",
            "ShortSaleTodayBalance": "short_balance",
        }
    )
    result = _numeric(
        frame.loc[:, ["date", "margin_balance", "short_balance"]],
        ["margin_balance", "short_balance"],
        "FinMind 融資融券",
    )
    return _normalize_date(result, "FinMind 融資融券")


def parse_finmind_institutional(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = ["date", "institutional_net", "foreign_net", "trust_net", "dealer_net"]
    if not rows:
        return pd.DataFrame(columns=columns)
    _require_columns(rows, {"date", "name", "buy", "sell"}, "FinMind 三大法人")
    frame = pd.DataFrame(rows)
    frame["buy"] = pd.to_numeric(frame["buy"], errors="coerce")
    frame["sell"] = pd.to_numeric(frame["sell"], errors="coerce")
    if frame[["buy", "sell"]].isna().any().any():
        raise DataSourceError("FinMind 三大法人含無效數字")
    frame["net"] = frame["buy"] - frame["sell"]
    names = frame["name"].astype(str)
    allowed_names = {
        "Foreign_Investor",
        "Foreign_Dealer_Self",
        "Investment_Trust",
        "Dealer",
        "Dealer_self",
        "Dealer_Hedging",
    }
    unknown_names = set(names.unique()).difference(allowed_names)
    if unknown_names:
        raise DataSourceError(f"FinMind 三大法人含未知類別：{sorted(unknown_names)}")
    frame["foreign_net"] = frame["net"].where(names.eq("Foreign_Investor"), 0.0)
    frame["trust_net"] = frame["net"].where(names.eq("Investment_Trust"), 0.0)
    frame["dealer_net"] = frame["net"].where(
        names.isin({"Dealer", "Dealer_self", "Dealer_Hedging"}), 0.0
    )
    frame["institutional_net"] = frame["foreign_net"] + frame["trust_net"] + frame["dealer_net"]
    result = frame.groupby("date", as_index=False)[columns[1:]].sum()
    return _normalize_date(result, "FinMind 三大法人")


def parse_twse_monthly_closes(payload: Mapping[str, Any]) -> pd.DataFrame:
    """解析 TWSE 個股日收盤價及月平均價回應。"""

    if payload.get("stat") != "OK" or not isinstance(payload.get("data"), list):
        raise DataSourceError(f"TWSE 月收盤資料失敗：{payload.get('stat', '未知錯誤')}")
    parsed: list[dict[str, Any]] = []
    for row in payload["data"]:
        if not isinstance(row, list) or len(row) < 2 or "月平均" in str(row[0]):
            continue
        parsed.append(
            {
                "date": pd.Timestamp(parse_roc_date(str(row[0]))),
                "official_close": float(str(row[1]).replace(",", "")),
            }
        )
    if not parsed:
        raise DataSourceError("TWSE 月收盤資料沒有交易日")
    return _normalize_date(pd.DataFrame(parsed), "TWSE 月收盤")


def parse_yuanta_nav(payload: Mapping[str, Any]) -> pd.DataFrame:
    """解析元大官方歷史 NAV、規模與流通單位。"""

    rows = payload.get("Data")
    if payload.get("ResultCode") != 0 or not isinstance(rows, list):
        raise DataSourceError(f"元大 NAV API 失敗：{payload.get('ResultMsg', '未知錯誤')}")
    required = {"UPDATE_T", "NOW_NAV", "NOW_PRICE", "FUND_SIZE", "OS_UNIT"}
    _require_columns(rows, required, "元大 NAV")
    frame = pd.DataFrame(rows).rename(
        columns={
            "UPDATE_T": "date",
            "NOW_NAV": "nav",
            "NOW_PRICE": "issuer_market_price",
            "FUND_SIZE": "fund_size",
            "OS_UNIT": "outstanding_units",
        }
    )
    columns = ["date", "nav", "issuer_market_price", "fund_size", "outstanding_units"]
    result = _numeric(frame.loc[:, columns], columns[1:], "元大 NAV")
    result = result.loc[(result["nav"] > 0.0) & (result["issuer_market_price"] > 0.0)]
    if result.empty:
        raise DataSourceError("元大 NAV 沒有正值交易資料")
    return _normalize_date(result, "元大 NAV")


def _get_json(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise DataSourceError("API 回應不是 JSON object")
            return payload
        except (httpx.HTTPError, ValueError, DataSourceError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(0.5 * 2**attempt)
    raise DataSourceError(f"API 請求重試 {retries} 次仍失敗") from last_error


@dataclass(frozen=True)
class FinMindRepository:
    client: httpx.Client
    stock_id: str = "0050"

    def _fetch(
        self, dataset: str, start: date, end: date, *, data_id: str | None = None
    ) -> list[dict[str, Any]]:
        payload = _get_json(
            self.client,
            FINMIND_URL,
            params={
                "dataset": dataset,
                "data_id": self.stock_id if data_id is None else data_id,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        )
        return validate_finmind_response(payload, dataset)

    def us_adj_prices(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        return parse_us_prices(
            self._fetch("USStockPrice", start, end, data_id=symbol),
            source=f"FinMind USStockPrice {symbol}",
        )

    def usd_twd(self, start: date, end: date) -> pd.DataFrame:
        return parse_usd_twd(self._fetch("TaiwanExchangeRate", start, end, data_id="USD"))

    def prices(self, start: date, end: date) -> pd.DataFrame:
        return parse_finmind_prices(self._fetch("TaiwanStockPrice", start, end))

    def dividends(self, start: date, end: date) -> pd.DataFrame:
        return parse_finmind_dividends(self._fetch("TaiwanStockDividendResult", start, end))

    def splits(self, start: date, end: date) -> pd.DataFrame:
        return parse_finmind_splits(self._fetch("TaiwanStockSplitPrice", start, end))

    def margin(self, start: date, end: date) -> pd.DataFrame:
        return parse_finmind_margin(self._fetch("TaiwanStockMarginPurchaseShortSale", start, end))

    def institutional(self, start: date, end: date) -> pd.DataFrame:
        rows = self._fetch("TaiwanStockInstitutionalInvestorsBuySell", start, end)
        return parse_finmind_institutional(rows)


@dataclass(frozen=True)
class TwseRepository:
    client: httpx.Client
    cache_dir: Path | None = None
    workers: int = 1
    request_delay_seconds: float = 1.2

    def _cache_path(self, month: pd.Period) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{month.strftime('%Y%m')}.json"

    def _month_payload(self, month: pd.Period) -> dict[str, Any]:
        cache_path = self._cache_path(month)
        cacheable = month < pd.Period(date.today(), freq="M")
        if cacheable and cache_path is not None and cache_path.exists():
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        payload = _get_json(
            self.client,
            TWSE_MONTHLY_CLOSE_URL,
            params={"date": f"{month.strftime('%Y%m')}01", "stockNo": "0050", "response": "json"},
        )
        if cacheable and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        time.sleep(self.request_delay_seconds)
        return payload

    def monthly_closes(self, month: pd.Period) -> pd.DataFrame:
        return parse_twse_monthly_closes(self._month_payload(month))

    def closes(self, start: date, end: date) -> pd.DataFrame:
        months = list(pd.period_range(start, end, freq="M"))
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            frames = list(executor.map(self.monthly_closes, months))
        result = pd.concat(frames, ignore_index=True).sort_values("date", ignore_index=True)
        mask = result["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        result = result.loc[mask].reset_index(drop=True)
        if result["date"].duplicated().any():
            raise DataSourceError("TWSE 全歷史收盤價含重複日期")
        return result


@dataclass(frozen=True)
class YuantaRepository:
    client: httpx.Client

    def nav(self, start: date, end: date) -> pd.DataFrame:
        page = self.client.get(YUANTA_PAGE_URL)
        page.raise_for_status()
        device_id = extract_yuanta_device_id(page.text)
        params = {
            "APIType": "ETFBackstage",
            "CompanyName": "YUANTAFUNDS",
            "PageName": "/tradeInfo/comparison/0050/NAVhistory",
            "DeviceId": device_id,
            "FuncId": "ETFNAV/GetFluctuateValue",
            "AppName": "ETF",
            "Device": "4",
            "Platform": "ETF",
            "stk_cd": "0050",
            "SDATE": start.strftime("%Y%m%d"),
            "EDATE": end.strftime("%Y%m%d"),
        }
        payload = _get_json(
            self.client,
            YUANTA_API_URL,
            params=params,
            headers={"Referer": YUANTA_PAGE_URL},
        )
        return parse_yuanta_nav(payload)
