"""Multiplex secondary-profile scope tests for the Photon adapter + auth module.

__init__'s project_id, check_requirements'/validate_config's node_bin/
project_id, _env_enablement's home_channel, _reactions_enabled's
PHOTON_REACTIONS, __init__'s require_mention, and _standalone_send's
sidecar_port, plus auth.py's load_project_credentials/
load_dashboard_project_id, all previously read raw os.getenv
unconditionally (only PHOTON_PROJECT_SECRET/PHOTON_SIDECAR_TOKEN were
already scoped via _get_scoped_secret). Under gateway.multiplex_profiles,
os.environ holds the DEFAULT profile's YAML-to-env bridge output -- a
secondary profile with its own (different or absent) Photon config could
silently authenticate against the default profile's Spectrum project, or
have its mention-gating/reaction behavior driven by the default profile's
settings.

Notably project_id was a stronger variant of the bug (like the IRC fix in
this series): __init__'s original
`os.getenv("PHOTON_PROJECT_ID") or extra.get("project_id") or stored_id`
ordering let a raw env read override even an explicitly configured
config.yaml extra.

Mirrors the LINE/DingTalk/IRC/Mattermost fix for #98738.

The sidecar child env had the same leak (a residual site listed in #111617):
_start_sidecar copied os.environ, so the settings sidecar/index.mjs reads
(PHOTON_READ_RECEIPTS, PHOTON_TELEMETRY, PHOTON_STREAM_* ...) came from the
default profile.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon import auth as photon_auth
from plugins.platforms.photon.adapter import PhotonAdapter

_PHOTON_ENV = (
    "PHOTON_PROJECT_ID",
    "PHOTON_PROJECT_SECRET",
    "PHOTON_DASHBOARD_PROJECT_ID",
    "PHOTON_REQUIRE_MENTION",
    "PHOTON_REACTIONS",
    "PHOTON_HOME_CHANNEL",
    "PHOTON_HOME_CHANNEL_NAME",
    "PHOTON_SIDECAR_PORT",
)


@pytest.fixture
def tmp_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate from the real ~/.hermes/auth.json fallback in load_project_credentials()."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in _PHOTON_ENV:
        monkeypatch.delenv(key, raising=False)
    yield home
    for key in _PHOTON_ENV:
        os.environ.pop(key, None)


@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("PHOTON_PROJECT_ID", "default-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "default-project-secret")
    monkeypatch.setenv("PHOTON_REQUIRE_MENTION", "true")
    monkeypatch.setenv("PHOTON_REACTIONS", "true")


class TestAuthMultiplexProfileScope:
    """load_project_credentials / load_dashboard_project_id (auth.py)."""

    def test_scoped_miss_does_not_leak_default_project_id(
        self, tmp_hermes_home, multiplex_scope, default_profile_env
    ):
        multiplex_scope({"SOMETHING_ELSE": "x"})
        sid, secret = photon_auth.load_project_credentials()
        assert sid is None
        assert secret is None
        adapter = PhotonAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter._project_id == ""
        assert adapter.require_mention is False
        assert adapter._reactions_enabled() is False

class TestAdapterMultiplexProfileScope:
    """PhotonAdapter.__init__ / _env_enablement / _reactions_enabled (adapter.py)."""

    def test_secondary_extra_wins_over_default_profile_env(
        self, tmp_hermes_home, multiplex_scope, default_profile_env
    ):
        """A secondary profile's own config.yaml extra project_id must be
        authoritative -- not the default profile's bridged env value. The
        pre-fix ordering (raw os.getenv checked BEFORE extra) meant even an
        explicit extra config was silently overridden."""
        multiplex_scope({"PHOTON_PROJECT_SECRET": "profile-secret"})
        cfg = PlatformConfig(
            enabled=True,
            extra={"project_id": "profile-project-id"},
        )
        adapter = PhotonAdapter(cfg)
        assert adapter._project_id == "profile-project-id"


async def _sidecar_spawn_env(monkeypatch: pytest.MonkeyPatch, adapter: PhotonAdapter) -> dict:
    """Run _start_sidecar with Node, npm and the HTTP probe faked; return the env it spawned with."""
    spawned: dict = {}

    async def _noop(*_args, **_kwargs) -> None:
        return None

    for step in ("_ensure_sidecar_deps", "_reap_stale_sidecar", "_apply_spectrum_patch"):
        monkeypatch.setattr(adapter, step, _noop)

    def _fake_popen(cmd, **kwargs):
        spawned.update(kwargs["env"])
        return SimpleNamespace(pid=4242, stdout=None, stdin=None, poll=lambda: None)

    class _HealthyClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc) -> bool:
            return False

        async def post(self, *_args, **_kwargs):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(photon_adapter.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _HealthyClient)
    await adapter._start_sidecar()
    return spawned


class TestSidecarEnvMultiplexProfileScope:
    """_start_sidecar: the settings sidecar/index.mjs reads come from the profile that owns the adapter."""

    @pytest.fixture(autouse=True)
    def _pm_node(self, monkeypatch):
        monkeypatch.setattr(photon_adapter, "find_node_executable", lambda _name: "/pm/node")

    @pytest.fixture
    def launch_sidecar_settings(self, monkeypatch):
        """The default (launch) profile's sidecar settings in os.environ."""
        monkeypatch.setenv("PHOTON_TELEMETRY", "on")
        monkeypatch.setenv("PHOTON_READ_RECEIPTS", "true")
        monkeypatch.setenv("PHOTON_STREAM_SILENCE_PROBE_MS", "0")

    def test_secondary_sidecar_gets_its_own_settings_not_the_launch_profiles(
        self, monkeypatch, tmp_hermes_home, multiplex_scope, default_profile_env, launch_sidecar_settings
    ):
        multiplex_scope({"PHOTON_PROJECT_ID": "profile-project-id", "PHOTON_PROJECT_SECRET": "profile-secret",
                         "PHOTON_READ_RECEIPTS": "false"})
        # asyncio.run copies this context, so the sidecar spawns under the secondary profile's scope.
        env = asyncio.run(_sidecar_spawn_env(monkeypatch, PhotonAdapter(PlatformConfig(enabled=True, extra={}))))
        assert env["PHOTON_READ_RECEIPTS"] == "false"
        # Telemetry is opt-in per profile: the default profile's opt-in must not switch it on here.
        assert "PHOTON_TELEMETRY" not in env
        assert "PHOTON_STREAM_SILENCE_PROBE_MS" not in env
        assert env["PHOTON_PROJECT_ID"] == "profile-project-id"
        assert env["PHOTON_PROJECT_SECRET"] == "profile-secret"

    def test_unscoped_default_profile_sidecar_keeps_its_env_settings(
        self, monkeypatch, tmp_hermes_home, default_profile_env, launch_sidecar_settings
    ):
        env = asyncio.run(_sidecar_spawn_env(monkeypatch, PhotonAdapter(PlatformConfig(enabled=True, extra={}))))
        assert env["PHOTON_TELEMETRY"] == "on"
        assert env["PHOTON_READ_RECEIPTS"] == "true"
        assert env["PHOTON_STREAM_SILENCE_PROBE_MS"] == "0"
        assert env["PHOTON_PROJECT_ID"] == "default-project-id"
