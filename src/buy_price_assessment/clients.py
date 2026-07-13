"""外部市場資料來源的共用驗證與解析。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import Any


class DataSourceError(RuntimeError):
    """外部資料來源未回傳可安全使用的資料。"""


def parse_roc_date(value: str) -> date:
    """將證交所民國年日期轉為西元日期。"""

    parts = value.strip().split("/")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"無效的民國日期：{value!r}")
    year, month, day = (int(part) for part in parts)
    try:
        return date(year + 1911, month, day)
    except ValueError as error:
        raise ValueError(f"無效的民國日期：{value!r}") from error


def validate_finmind_response(payload: Mapping[str, Any], dataset: str) -> list[dict[str, Any]]:
    """驗證 FinMind 回應，避免把錯誤或 quota 訊息當成空資料。"""

    status = payload.get("status")
    if status != 200:
        message = str(payload.get("msg") or "未知錯誤")
        raise DataSourceError(f"FinMind {dataset} 失敗：{message}")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise DataSourceError(f"FinMind {dataset} 的 data 不是陣列")
    if not all(isinstance(row, dict) for row in rows):
        raise DataSourceError(f"FinMind {dataset} 含非物件資料列")
    return rows


def extract_yuanta_device_id(html: str) -> str:
    """從元大 ETF Nuxt state 擷取公開 API 所需的暫時性 DeviceId。"""

    match = re.search(r'DeviceId\s*:\s*"([0-9a-fA-F-]{36})"', html)
    if match is None:
        raise DataSourceError("元大 ETF 頁面找不到 DeviceId")
    return match.group(1)
