"""Tests for the Vertex AI adapter (agent/vertex_adapter.py).

Vertex uses OAuth2 (short-lived access tokens from a service-account JSON or
ADC), NOT a static API key. These tests mock google-auth entirely — no network
calls — and cover token minting, the config.yaml→env precedence bridge, the
global vs regional base-URL shapes, and the ADC→service-account fallback.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time
import types
from datetime import UTC, datetime, timedelta

import pytest


def _install_fake_google_auth(monkeypatch, *, adc_ok=True, adc_project="adc-project",
                              sa_project="sa-project", token="ya29.FAKE"):
    """Register a fake google-auth tree in sys.modules and return the module set."""
    ga = types.ModuleType("google.auth")
    gt = types.ModuleType("google.auth.transport")
    gtr = types.ModuleType("google.auth.transport.requests")
    go = types.ModuleType("google.oauth2")
    gsa = types.ModuleType("google.oauth2.service_account")
    gp = types.ModuleType("google")

    gtr.Request = type("Request", (), {})

    class _Creds:
        def __init__(self):
            self.token = None
            self.expiry = None
            self.expired = False

        def refresh(self, req):
            self.token = token

    def _default(scopes=None):
        if not adc_ok:
            raise RuntimeError("Could not automatically determine credentials")
        return _Creds(), adc_project

    ga.default = _default
    ga.transport = gt
    gt.requests = gtr

    class _SA:
        @staticmethod
        def from_service_account_file(path, scopes=None):
            c = _Creds()
            c.project_id = sa_project
            return c

        @staticmethod
        def from_service_account_info(info, scopes=None):
            c = _Creds()
            c.project_id = sa_project
            return c

    gsa.Credentials = _SA
    go.service_account = gsa
    gp.auth = ga
    gp.oauth2 = go

    for name, mod in [
        ("google", gp), ("google.auth", ga), ("google.auth.transport", gt),
        ("google.auth.transport.requests", gtr), ("google.oauth2", go),
        ("google.oauth2.service_account", gsa),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return gp


@pytest.fixture
def vertex_adapter(monkeypatch):
    """Fresh vertex_adapter with a fake google-auth and clean caches/env."""
    for var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS",
                "VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    _install_fake_google_auth(monkeypatch)
    import agent.vertex_adapter as va
    va = importlib.reload(va)
    va._creds_cache.clear()
    # Neutralize config.yaml by default; individual tests re-patch _vertex_config.
    monkeypatch.setattr(va, "_vertex_config", dict)
    return va












def test_has_vertex_credentials_via_config_project(vertex_adapter, monkeypatch):
    monkeypatch.setattr(vertex_adapter, "_vertex_config", lambda: {"project_id": "p"})
    assert vertex_adapter.has_vertex_credentials() is True


def test_has_vertex_credentials_false_when_nothing_set(vertex_adapter):
    assert vertex_adapter.has_vertex_credentials() is False




def test_multiplex_scope_takes_precedence_over_raw_environ(vertex_adapter, monkeypatch):
    """In a multiplex gateway, a profile's own secret scope must win over a
    stale value in process os.environ left behind by another profile's
    dotenv load at boot — otherwise Profile B's turn could resolve Profile
    A's Vertex project (or worse, its credentials file path)."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "other-profile-project")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"VERTEX_PROJECT_ID": "this-profile-project"})
    try:
        assert vertex_adapter._resolve_project_override() == "this-profile-project"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_multiplex_unscoped_read_fails_closed(vertex_adapter, monkeypatch):
    """A credential read with no profile scope installed while multiplexing
    is active must raise rather than silently fall back to (possibly another
    profile's) raw os.environ value."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "leaked-project")
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            vertex_adapter._resolve_project_override()
    finally:
        secret_scope.set_multiplex_active(False)


def test_adc_refuses_foreign_profile_google_application_credentials(
    vertex_adapter, monkeypatch, tmp_path
):
    """When this profile's scope defines no Vertex credentials, but os.environ
    still carries a *different* profile's GOOGLE_APPLICATION_CREDENTIALS (left
    there by python-dotenv at gateway boot), ADC must not silently mint a
    token under that foreign service account."""
    from agent import secret_scope

    sa_file = tmp_path / "other_profile_sa.json"
    sa_file.write_text('{"project_id": "other-profile"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_file))

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})  # this profile defines nothing
    try:
        assert vertex_adapter.get_vertex_credentials() == (None, None)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)






# --- Pattern D: credential-file rotation invalidates the cache ---


def test_sa_file_rotation_invalidates_creds_cache(vertex_adapter, monkeypatch, tmp_path):
    """Rotating the service-account file on disk must be picked up on the
    next call — the pre-signature cache served tokens minted from the OLD
    identity for the life of the process (Pattern D: stale cache after an
    out-of-band change)."""
    import os as _os

    sa_file = tmp_path / "sa.json"
    sa_file.write_text('{"project_id": "first-identity"}')
    monkeypatch.setattr(
        vertex_adapter, "_resolve_credentials_path", lambda explicit=None: str(sa_file)
    )

    token1, _project1 = vertex_adapter.get_vertex_credentials()
    assert token1 == "ya29.FAKE"
    assert len(vertex_adapter._creds_cache) == 1

    # Same file untouched: cache hit (same Credentials object).
    (key1,) = vertex_adapter._creds_cache
    creds_obj_1 = vertex_adapter._creds_cache[key1][0]
    vertex_adapter.get_vertex_credentials()
    (key1b,) = vertex_adapter._creds_cache
    assert key1b == key1
    assert vertex_adapter._creds_cache[key1b][0] is creds_obj_1

    # Rotate: rewrite the file with different content and a bumped mtime.
    sa_file.write_text('{"project_id": "second-identity", "rotated": true}')
    st = _os.stat(sa_file)
    _os.utime(sa_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    vertex_adapter.get_vertex_credentials()
    # New signature key replaced the old entry — not appended beside it.
    (key2,) = vertex_adapter._creds_cache
    assert key2 != key1
    assert vertex_adapter._creds_cache[key2][0] is not creds_obj_1






def test_adc_failure_retries_with_late_added_sa_file(vertex_adapter, monkeypatch, tmp_path):
    """ADC failure must fall back to a service-account file that appeared
    after startup. The signature-keyed cache turned keys into tuples and the
    old `cache_key == "__adc__"` string comparison silently disabled this
    retry (caught in review); the guard is now `not resolved_path`."""
    sa_file = tmp_path / "late_sa.json"
    sa_file.write_text('{"project_id": "late-identity"}')

    calls = {"n": 0}

    def _resolve(explicit=None):
        # First resolution (entry): nothing configured -> ADC attempt.
        # Second resolution (retry after ADC failure): the file has appeared.
        calls["n"] += 1
        return None if calls["n"] == 1 else str(sa_file)

    monkeypatch.setattr(vertex_adapter, "_resolve_credentials_path", _resolve)
    # Make the ADC attempt itself raise.
    monkeypatch.setattr(
        vertex_adapter.google.auth, "default",
        lambda scopes=None: (_ for _ in ()).throw(RuntimeError("ADC expired")),
    )

    token, project = vertex_adapter.get_vertex_credentials()
    assert token == "ya29.FAKE"
    assert project == "sa-project"
    assert calls["n"] >= 2


def test_metadata_preserving_rotation_invalidates_creds_cache(vertex_adapter, monkeypatch, tmp_path):
    """Atomic replacement that keeps SIZE and MTIME identical (deployment
    tools that restore metadata; equal-length JSON) must still be picked up.
    Review finding on #97701: a (path, mtime_ns, size) stat signature keyed
    the cache, so this replacement served the old private key; the key is
    now a content digest."""
    import os as _os

    sa_file = tmp_path / "sa.json"
    sa_file.write_text('{"project_id": "AAAA-identity"}')
    monkeypatch.setattr(
        vertex_adapter, "_resolve_credentials_path", lambda explicit=None: str(sa_file)
    )

    vertex_adapter.get_vertex_credentials()
    (key1,) = vertex_adapter._creds_cache
    creds_obj_1 = vertex_adapter._creds_cache[key1][0]

    # Same-length content, mtime restored, atomic replace.
    st = _os.stat(sa_file)
    new = tmp_path / "sa.json.new"
    new.write_text('{"project_id": "BBBB-identity"}')  # equal length
    _os.utime(new, ns=(st.st_atime_ns, st.st_mtime_ns))
    _os.replace(new, sa_file)
    st2 = _os.stat(sa_file)
    assert st2.st_size == st.st_size and st2.st_mtime_ns == st.st_mtime_ns

    vertex_adapter.get_vertex_credentials()
    (key2,) = vertex_adapter._creds_cache
    assert key2 != key1, "content change must produce a new cache key"
    assert vertex_adapter._creds_cache[key2][0] is not creds_obj_1


# --- Token reuse does not depend on the host's timezone ---


def _naive_utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class _ExpiringCreds:
    """google-auth's contract: ``expiry`` is a NAIVE UTC datetime (``_helpers.utcnow()`` plus
    ``expires_in``) and ``expired`` compares it with naive UTC minus a 3m45s skew."""

    project_id = "sa-project"

    def __init__(self):
        self.token, self.expiry, self.refreshes = None, None, 0

    @property
    def expired(self):
        return self.expiry is not None and _naive_utcnow() >= self.expiry - timedelta(minutes=3, seconds=45)

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"ya29.mint{self.refreshes}"
        self.expiry = _naive_utcnow() + timedelta(seconds=3599)


@contextlib.contextmanager
def _process_timezone(tz: str):
    """Switch the process-local timezone (POSIX ``TZ`` string) for the block."""
    original = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


@pytest.fixture
def expiring_creds(vertex_adapter, monkeypatch, tmp_path):
    creds = _ExpiringCreds()
    sa_file = tmp_path / "sa.json"
    sa_file.write_text('{"project_id": "sa-project"}')
    monkeypatch.setattr(vertex_adapter, "_resolve_credentials_path", lambda explicit=None: str(sa_file))
    monkeypatch.setattr(vertex_adapter.service_account.Credentials, "from_service_account_info",
                        staticmethod(lambda info, scopes=None: creds))
    return creds


@pytest.mark.platforms("posix")  # time.tzset() is how the test moves the host timezone
@pytest.mark.parametrize("tz", ["UTC0", "CET-1", "JST-9"])  # UTC, UTC+1, UTC+9
def test_fresh_token_is_reused_in_any_timezone(vertex_adapter, expiring_creds, tz):
    """A minted token is reused until it nears expiry, wherever the host clock is set. Read
    with ``.timestamp()``, the naive-UTC expiry was taken as local time, so east of UTC a
    fresh one-hour token looked expired and every resolution ran a new OAuth exchange."""
    with _process_timezone(tz):
        first, _ = vertex_adapter.get_vertex_credentials()
        second, _ = vertex_adapter.get_vertex_credentials()
    assert expiring_creds.refreshes == 1
    assert second == first


@pytest.mark.platforms("posix")  # time.tzset() is how the test moves the host timezone
@pytest.mark.parametrize("tz", ["UTC0", "EST5", "PST8"])  # UTC, UTC-5, UTC-8
def test_token_inside_the_refresh_window_is_reminted_in_any_timezone(vertex_adapter, expiring_creds, tz):
    """Four minutes left is inside the five-minute refresh window (google-auth's own
    ``expired`` only trips at 3m45s). West of UTC the local-time reading pushed the expiry
    hours out, so the early refresh never fired."""
    with _process_timezone(tz):
        vertex_adapter.get_vertex_credentials()
        expiring_creds.expiry = _naive_utcnow() + timedelta(minutes=4)
        token, _ = vertex_adapter.get_vertex_credentials()
    assert expiring_creds.refreshes == 2
    assert token == "ya29.mint2"
