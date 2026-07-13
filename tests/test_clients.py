from datetime import date

import pytest

from buy_price_assessment.clients import (
    DataSourceError,
    extract_yuanta_device_id,
    parse_roc_date,
    validate_finmind_response,
)


def test_parse_roc_date_accepts_twse_format() -> None:
    assert parse_roc_date(" 92/06/30") == date(2003, 6, 30)
    assert parse_roc_date("114/06/18") == date(2025, 6, 18)


def test_parse_roc_date_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="民國日期"):
        parse_roc_date("2025-06-18")


def test_validate_finmind_response_requires_success_and_rows() -> None:
    rows = validate_finmind_response({"status": 200, "data": [{"date": "2025-01-01"}]}, "x")
    assert rows == [{"date": "2025-01-01"}]

    with pytest.raises(DataSourceError, match="quota"):
        validate_finmind_response({"status": 402, "msg": "quota", "data": []}, "x")


def test_extract_yuanta_device_id_from_nuxt_state() -> None:
    html = '<script>DeviceId:"c346ab96-d58b-4014-a495-0c8dcbe4ea8a"</script>'
    assert extract_yuanta_device_id(html) == "c346ab96-d58b-4014-a495-0c8dcbe4ea8a"

    with pytest.raises(DataSourceError, match="DeviceId"):
        extract_yuanta_device_id("<html></html>")
