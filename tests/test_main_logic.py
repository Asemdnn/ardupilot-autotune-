import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
import sys

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main  # noqa: E402


def test_parse_current_params_filters_invalid_tokens():
    raw = "ATC_RAT_RLL_P=0.2, bad, INS_GYRO_FILTER=80;ATC_RAT_RLL_I=abc\nATC_RAT_RLL_D=0.004"  # noqa
    parsed = main.parse_current_params(raw)

    assert parsed["ATC_RAT_RLL_P"] == 0.2
    assert parsed["INS_GYRO_FILTER"] == 80.0
    assert parsed["ATC_RAT_RLL_D"] == 0.004
    assert "ATC_RAT_RLL_I" not in parsed


def test_sanitize_upload_filename_prevents_path_traversal():
    assert main.sanitize_upload_filename("..\\..\\secret.log") == "secret.log"
    assert main.sanitize_upload_filename("../../secret.bin") == "secret.bin"
    assert (
        main.sanitize_upload_filename("my unsafe file.log")
        == "my_unsafe_file.log"
    )


@pytest.mark.asyncio
async def test_recommendations_use_vehicle_current_params():
    metrics = {
        "overshoot_percent": 12.0,
        "vibration_g": 3.1,
        "settling_time_sec": 2.4,
    }
    current_params = {
        "ATC_RAT_RLL_P": 0.2,
        "ATC_RAT_RLL_I": 0.08,
        "INS_HNTC2_ENABLE": 0.0,
        "INS_GYRO_FILTER": 100.0,
    }

    mock_llm_json = [
        {
            "parameter": "ATC_RAT_RLL_P",
            "recommended": 0.15,
            "reason": "Reduce roll overshoot for tighter racing response.",
        },
        {
            "parameter": "INS_HNTC2_ENABLE",
            "recommended": 1.0,
            "reason": "Enable harmonic notch filtering to reduce vibration peaks.",  # noqa
        },
    ]

    from unittest.mock import Mock

    mock_response = Mock()
    mock_response.json.return_value = {"response": json.dumps(mock_llm_json)}
    mock_response.raise_for_status = Mock()

    mock_post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient.post", mock_post):
        recs = await main.generate_recommendations(
            metrics=metrics,
            drone_type="fpv_racing",
            notes=None,
            current_params=current_params,
        )

    by_param = {r["parameter"]: r for r in recs}

    assert by_param["ATC_RAT_RLL_P"]["current"] == "0.2"
    assert by_param["ATC_RAT_RLL_P"]["recommended"] == "0.15"
    assert by_param["INS_HNTC2_ENABLE"]["change"] == "Enable"
