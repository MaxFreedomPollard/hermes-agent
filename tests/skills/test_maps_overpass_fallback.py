"""Offline tests for the maps skill's Overpass mirror fallback.

``nearby`` and ``bbox`` print one JSON document on stdout. A mirror that
fails before another one answers must not add a document of its own.
"""

import argparse
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "skills" / "productivity" / "maps" / "scripts" / "maps_client.py"
)

OVERPASS_RESPONSE = {
    "version": 0.6,
    "generator": "Overpass API",
    "elements": [
        {
            "type": "node",
            "id": 101,
            "lat": 40.7585,
            "lon": -73.9850,
            "tags": {"amenity": "cafe", "name": "Corner Cafe"},
        }
    ],
}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def maps_client(monkeypatch):
    spec = importlib.util.spec_from_file_location("maps_client_overpass", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return module


def _serve(maps_client, monkeypatch, failures):
    """Answer Overpass POSTs: mirrors listed in ``failures`` raise that HTTP code."""
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        code = failures.get(req.full_url)
        if code is not None:
            raise urllib.error.HTTPError(req.full_url, code, "mirror failure", None, None)
        return _Response(json.dumps(OVERPASS_RESPONSE).encode("utf-8"))

    monkeypatch.setattr(maps_client.urllib.request, "urlopen", fake_urlopen)
    return calls


def _nearby_args():
    return argparse.Namespace(
        near=None, lat="40.7580", lon="-73.9855", category="cafe",
        category_list=[], radius=500, limit=10,
    )


@pytest.mark.parametrize("code", [429, 400])
def test_nearby_prints_one_document_when_a_later_mirror_answers(maps_client, monkeypatch, capsys, code):
    primary, mirror = maps_client.OVERPASS_URLS[:2]
    calls = _serve(maps_client, monkeypatch, {primary: code})

    maps_client.cmd_nearby(_nearby_args())

    result = json.loads(capsys.readouterr().out)
    assert calls == [primary, mirror]
    assert "error" not in result
    assert [place["name"] for place in result["results"]] == ["Corner Cafe"]


def test_bbox_prints_one_document_when_a_later_mirror_answers(maps_client, monkeypatch, capsys):
    primary = maps_client.OVERPASS_URLS[0]
    _serve(maps_client, monkeypatch, {primary: 504})

    maps_client.cmd_bbox(argparse.Namespace(
        lat1="40.75", lon1="-74.00", lat2="40.77", lon2="-73.98",
        category="cafe", limit=20,
    ))

    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 1


def test_all_mirrors_failing_prints_one_error_document(maps_client, monkeypatch, capsys):
    _serve(maps_client, monkeypatch, {url: 429 for url in maps_client.OVERPASS_URLS})

    with pytest.raises(SystemExit) as excinfo:
        maps_client.cmd_nearby(_nearby_args())

    assert excinfo.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert result["error"].startswith("All Overpass mirrors failed.")
