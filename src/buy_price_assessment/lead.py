"""開盤前已知的外部領先序列：美股前一交易日與前一日本地匯率。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEAD_RAW_FILES: dict[str, str] = {
    "tsm": "tsm_us.csv",
    "sox": "sox_us.csv",
    "fx": "usd_twd.csv",
}

# 預先指定的隔夜大跌門檻（經濟整數 1%），不是從樣本外掃出的參數。
OVERNIGHT_DUMP_THRESHOLD = -0.01

LEAD_RULES: tuple[tuple[str, str], ...] = (
    ("tsm_neg_or_day5", "tsm_dump"),
    ("sox_neg_or_day5", "sox_dump"),
    ("fx_pause_or_day5", "fx_not_depreciating"),
    ("tsm_dump1pct_or_day5", "tsm_dump_1pct"),
    ("tsm_buy_unless_adverse", "tsm_not_dump"),
    ("fx_single_pause_or_day5", "fx_not_up"),
)
LEAD_SIGNAL_COLUMNS: tuple[str, ...] = tuple(column for _, column in LEAD_RULES)


def load_lead_data(raw_dir: Path) -> dict[str, pd.DataFrame] | None:
    paths = {key: raw_dir / filename for key, filename in LEAD_RAW_FILES.items()}
    if not all(path.exists() for path in paths.values()):
        return None
    frames: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        frame = pd.read_csv(path, parse_dates=["date"])
        required = {"date", "usd_twd"} if key == "fx" else {"date", "adj_close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} 缺少欄位：{sorted(missing)}")
        frames[key] = frame
    return frames


def align_prior_session(
    source: pd.DataFrame,
    target_dates: pd.Series,
    column: str,
) -> pd.Series:
    """每個目標日 T 取來源日 <= T-1 的最後一筆，避免用到當日尚未收盤的美股。"""

    if source.empty or column not in source.columns or target_dates.empty:
        return pd.Series(np.nan, index=target_dates.index, dtype=float)
    src = source.loc[:, ["date", column]].copy()
    src["date"] = pd.to_datetime(src["date"], errors="raise")
    src[column] = pd.to_numeric(src[column], errors="coerce")
    src = src.dropna().sort_values("date", kind="mergesort")
    if src.empty:
        return pd.Series(np.nan, index=target_dates.index, dtype=float)
    left = pd.DataFrame(
        {
            "asof": pd.to_datetime(target_dates, errors="raise") - pd.Timedelta(days=1),
            "row": np.arange(len(target_dates)),
        }
    )
    ordered = left.sort_values(["asof", "row"], kind="mergesort")
    merged = pd.merge_asof(
        ordered,
        src,
        left_on="asof",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("row", kind="mergesort")[column].set_axis(target_dates.index)


def _session_return(frame: pd.DataFrame, price_column: str) -> pd.DataFrame:
    result = frame.loc[:, ["date", price_column]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["ret"] = pd.to_numeric(result[price_column], errors="coerce").pct_change()
    return result.loc[:, ["date", "ret"]]


def _usd_twd_signals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, ["date", "usd_twd"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    change = pd.to_numeric(result["usd_twd"], errors="coerce").pct_change()
    up = change > 0.0
    result["usd_twd_up"] = up.astype(float)
    result["up_streak3"] = (up & up.shift(1) & up.shift(2)).astype(float)
    return result.loc[:, ["date", "usd_twd_up", "up_streak3"]]


def attach_lead_features(
    frame: pd.DataFrame,
    *,
    tsm: pd.DataFrame,
    sox: pd.DataFrame,
    fx: pd.DataFrame,
) -> pd.DataFrame:
    """把開盤前已知的 ADR／SOX／匯率特徵接到每日列，不再額外 lag。"""

    if "date" not in frame.columns:
        raise ValueError("領先特徵缺少 date")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["tsm_lead_ret"] = align_prior_session(
        _session_return(tsm, "adj_close"), result["date"], "ret"
    )
    result["sox_lead_ret"] = align_prior_session(
        _session_return(sox, "adj_close"), result["date"], "ret"
    )
    fx_signals = _usd_twd_signals(fx)
    result["usd_twd_up"] = align_prior_session(fx_signals, result["date"], "usd_twd_up")
    result["usd_twd_up_streak3"] = align_prior_session(fx_signals, result["date"], "up_streak3")
    result["tsm_dump"] = result["tsm_lead_ret"] < 0.0
    result["sox_dump"] = result["sox_lead_ret"] < 0.0
    result["tsm_dump_1pct"] = result["tsm_lead_ret"] <= OVERNIGHT_DUMP_THRESHOLD
    result["tsm_not_dump"] = result["tsm_lead_ret"] >= 0.0
    result["fx_not_depreciating"] = result["usd_twd_up_streak3"].fillna(1.0) < 0.5
    result["fx_not_up"] = result["usd_twd_up"].fillna(1.0) < 0.5
    return result
