import json

import pandas as pd

from buy_price_assessment import cli
from buy_price_assessment.cli import build_parser
from buy_price_assessment.clients import DataSourceError


def test_cli_exposes_fetch_analyze_and_all_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["fetch"]).command == "fetch"
    assert parser.parse_args(["analyze"]).command == "analyze"
    args = parser.parse_args(["all", "--validate-all-twse-months", "--force-models"])
    assert args.validate_all_twse_months is True
    assert args.force_models is True


def test_cli_fetch_and_analyze_boundaries(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09"]),
            "close": [105.8],
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "download_daily_data", lambda **_: daily)
    fetched = cli._fetch(False)
    assert fetched["rows"] == 1
    assert (tmp_path / "data/processed/0050_daily.csv").exists()

    monkeypatch.setattr(
        cli,
        "run_analysis",
        lambda **_: {
            "oracle_distribution": {"months": 276},
            "primary": {"day1_mean_regret": 0.03, "model_mean_regret": 0.04},
        },
    )
    analyzed = cli._analyze(False)
    assert analyzed["months"] == 276


def test_cli_main_returns_structured_success_and_error(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "_fetch", lambda _: {"rows": 1})
    monkeypatch.setattr(cli, "_analyze", lambda _: {"months": 1})
    assert cli.main(["all"]) == 0
    success = json.loads(capsys.readouterr().out)
    assert success["success"] is True
    assert success["data"]["fetch"]["rows"] == 1

    def fail(_: bool) -> dict[str, object]:
        raise DataSourceError("暫時失敗")

    monkeypatch.setattr(cli, "_fetch", fail)
    assert cli.main(["fetch"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["success"] is False
    assert error["error"]["type"] == "DataSourceError"
