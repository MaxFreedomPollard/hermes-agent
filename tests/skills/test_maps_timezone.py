"""Offline tests for the maps skill's ``timezone`` command.

TimeAPI.io's ``/api/timezone/coordinate`` reports offsets as NodaTime
``Offset`` objects: the whole offset in ``seconds`` (plus the same value in
milliseconds/ticks/nanoseconds), with no hours or minutes fields.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills" / "productivity" / "maps" / "scripts" / "maps_client.py"
)


@pytest.fixture
def maps_client():
    spec = importlib.util.spec_from_file_location("maps_client_tz", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offset(seconds):
    return {
        "seconds": seconds,
        "milliseconds": seconds * 1000,
        "ticks": seconds * 10_000_000,
        "nanoseconds": seconds * 1_000_000_000,
    }


def _run_timezone(maps_client, monkeypatch, capsys, payload, lat="40.7128", lon="-74.0060"):
    def fake_http_get(url, params=None, retries=None, silent=False):
        assert url == maps_client.TIMEAPI_BASE
        assert params == {"latitude": float(lat), "longitude": float(lon)}
        return payload

    monkeypatch.setattr(maps_client, "http_get", fake_http_get)
    maps_client.cmd_timezone(argparse.Namespace(lat=lat, lon=lon))
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("zone", "seconds", "expected"),
    [
        ("America/New_York", -14400, "-04:00"),
        ("Asia/Tokyo", 32400, "+09:00"),
        ("Asia/Kolkata", 19800, "+05:30"),
        ("Asia/Kathmandu", 20700, "+05:45"),
        ("America/St_Johns", -9000, "-02:30"),
        ("UTC", 0, "+00:00"),
    ],
)
def test_utc_offset_comes_from_offset_seconds(maps_client, monkeypatch, capsys, zone, seconds, expected):
    payload = {
        "timeZone": zone,
        "currentLocalTime": "2026-10-26T14:05:12.1234567",
        "currentUtcOffset": _offset(seconds),
        "standardUtcOffset": _offset(seconds),
        "hasDayLightSaving": False,
        "isDayLightSavingActive": False,
        "dstInterval": None,
    }

    result = _run_timezone(maps_client, monkeypatch, capsys, payload)

    assert result["timezone"] == zone
    assert result["utc_offset"] == expected
    assert result["source"] == "timeapi.io"


def test_standard_offset_is_used_when_current_offset_is_missing(maps_client, monkeypatch, capsys):
    payload = {
        "timeZone": "America/New_York",
        "currentLocalTime": "2026-12-01T09:00:00.0000000",
        "standardUtcOffset": _offset(-18000),
    }

    result = _run_timezone(maps_client, monkeypatch, capsys, payload)

    assert result["utc_offset"] == "-05:00"
