"""研究資料下載與分析 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from buy_price_assessment.analysis import run_analysis
from buy_price_assessment.clients import DataSourceError
from buy_price_assessment.ingestion import download_daily_data

DAILY_PATH = Path("data/processed/0050_daily.csv")


def _add_fetch_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validate-all-twse-months",
        action="store_true",
        help="低速逐月抓 TWSE 收盤價做第二層全量驗證（結果會快取）",
    )


def _add_analysis_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force-models",
        action="store_true",
        help="忽略已驗證的 walk-forward 預測檔並重新逐月訓練",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="0050 每月最佳買點研究")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="下載並驗證自上市以來每日資料")
    _add_fetch_flags(fetch)
    analyze = commands.add_parser("analyze", help="執行特徵、walk-forward 與報告")
    _add_analysis_flags(analyze)
    all_command = commands.add_parser("all", help="依序下載資料並完成分析")
    _add_fetch_flags(all_command)
    _add_analysis_flags(all_command)
    return parser


def _fetch(validate_all_twse_months: bool) -> dict[str, object]:
    daily = download_daily_data(validate_all_twse_months=validate_all_twse_months)
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    return {
        "rows": len(daily),
        "start": str(daily["date"].min().date()),
        "end": str(daily["date"].max().date()),
        "path": str(DAILY_PATH),
    }


def _analyze(force_models: bool) -> dict[str, object]:
    results = run_analysis(reuse_existing_predictions=not force_models)
    return {
        "months": results["oracle_distribution"]["months"],
        "day1_mean_regret": results["primary"]["day1_mean_regret"],
        "model_mean_regret": results["primary"]["model_mean_regret"],
        "report": "reports/0050_buy_point_analysis.md",
    }


def _success(data: MappingLike) -> None:
    print(
        json.dumps(
            {"success": True, "status": "ok", "data": data, "error": None}, ensure_ascii=False
        )
    )


MappingLike = dict[str, object]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            data = _fetch(bool(args.validate_all_twse_months))
        elif args.command == "analyze":
            data = _analyze(bool(args.force_models))
        else:
            fetch_result = _fetch(bool(args.validate_all_twse_months))
            data = {"fetch": fetch_result, "analysis": _analyze(bool(args.force_models))}
    except (DataSourceError, OSError, ValueError) as error:
        payload = {
            "success": False,
            "status": "error",
            "data": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1
    _success(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
