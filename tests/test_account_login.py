from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from sylliptor_agent_cli import account_login
from sylliptor_agent_cli.config import (
    load_config,
    load_persisted_profile_keys,
    resolve_api_key,
    save_persisted_profile_key,
)
from sylliptor_agent_cli.profile_presets import get_preset, make_profile_from_preset

_ACCESS_KEY = "11111111-2222-3333-4444-555555555555"


class _StubExchangeServer:
    """Stands in for the `cli-auth` edge function: code -> {access_key, email}."""

    def __init__(self, *, access_key: str, email: str | None) -> None:
        self.received: list[dict] = []
        access = access_key
        mail = email
        received = self.received

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.endswith("/cli-auth/exchange"):
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                received.append(body)
                payload = json.dumps({"access_key": access, "email": mail}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SYLLIPTOR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYLLIPTOR_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _callback_browser(code: str, state_override: str | None = None):
    """A fake browser that drives the CLI's localhost callback with a code."""

    def _open(url: str) -> bool:
        query = parse_qs(urlsplit(url).query)
        port = query["port"][0]
        state = state_override if state_override is not None else query["state"][0]
        target = f"http://127.0.0.1:{port}/callback?code={code}&state={state}"
        urllib.request.urlopen(target, timeout=5).read()
        return True

    return _open


def test_sylliptor_preset_offers_mimo_models() -> None:
    preset = get_preset("sylliptor")
    assert preset is not None
    assert preset.api_key_env is None
    # Flagship is the default; flash + omni are the other two trial models.
    assert preset.suggested_models[0] == "mimo-v2.5-pro"
    assert set(preset.suggested_models) == {"mimo-v2.5-pro", "mimo-v2-flash", "mimo-v2.5"}
    profile = make_profile_from_preset(preset, name="sylliptor")
    assert profile.default_model == "mimo-v2.5-pro"
    # The legacy bare "mimo" id canonicalizes to the flagship via the preset alias,
    # so old sessions stop showing a "current model" that no model matches.
    from sylliptor_agent_cli.profile_presets import canonical_model_alias_for_preset

    assert canonical_model_alias_for_preset(preset, "mimo") == "mimo-v2.5-pro"


def test_login_status_defaults_to_logged_out(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    cfg = load_config()
    status = account_login.login_status(cfg)
    assert status.logged_in is False
    assert status.active is False


def test_logout_when_not_logged_in_returns_false(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    cfg = load_config()
    assert account_login.logout(cfg) is False


def test_login_full_flow_wires_access_key_as_bearer(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    stub = _StubExchangeServer(access_key=_ACCESS_KEY, email="u@example.com")
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        cfg = load_config()
        result = account_login.login(
            cfg, browser_opener=_callback_browser("THE-ONE-TIME-CODE"), timeout_s=10
        )

        # The handshake delivered the right code to the exchange endpoint.
        assert stub.received and stub.received[0]["code"] == "THE-ONE-TIME-CODE"

        # Result reflects an active sylliptor profile with MiMo as default.
        assert result.email == "u@example.com"
        assert result.profile_name == "sylliptor"
        assert result.model == "mimo-v2.5-pro"

        # The access_key is persisted as the sylliptor profile key.
        assert load_persisted_profile_keys()["sylliptor"] == _ACCESS_KEY

        # Reloaded config has sylliptor active with MiMo as the default model.
        reloaded = load_config()
        assert reloaded.extra_fields["active_profile"] == "sylliptor"
        assert reloaded.model == "mimo-v2.5-pro"

        # The crucial wiring: resolve_api_key returns the access_key as the
        # Bearer for the sylliptor profile (so requests hit the proxy authed).
        resolution = resolve_api_key(reloaded, profile_name="sylliptor")
        assert resolution.key == _ACCESS_KEY

        status = account_login.login_status(reloaded)
        assert status.logged_in is True
        assert status.active is True
        assert status.base_url.endswith("/functions/v1/llm/v1")
    finally:
        stub.close()


def test_login_rejects_state_mismatch(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    cfg = load_config()
    with pytest.raises(account_login.SylliptorLoginError):
        account_login.login(
            cfg,
            browser_opener=_callback_browser("code", state_override="WRONG-STATE"),
            timeout_s=10,
        )
    # Nothing was persisted on a failed login.
    assert "sylliptor" not in load_persisted_profile_keys()


def test_login_then_logout_clears_key(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    stub = _StubExchangeServer(access_key=_ACCESS_KEY, email=None)
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        cfg = load_config()
        account_login.login(cfg, browser_opener=_callback_browser("code"), timeout_s=10)
        assert load_persisted_profile_keys().get("sylliptor") == _ACCESS_KEY

        reloaded = load_config()
        assert account_login.logout(reloaded) is True
        assert "sylliptor" not in load_persisted_profile_keys()
    finally:
        stub.close()


class _StubStatusServer:
    """Stands in for the proxy's read-only GET .../llm/v1/status route."""

    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        body = json.dumps(payload).encode()

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                if not self.path.endswith("/llm/v1/status"):
                    self.send_error(404)
                    return
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def test_fetch_trial_status_returns_none_when_logged_out(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    cfg = load_config()
    assert account_login.fetch_trial_status(cfg) is None


def test_fetch_trial_status_parses_proxy_response(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    save_persisted_profile_key("sylliptor", _ACCESS_KEY)
    stub = _StubStatusServer(
        {
            "email": "u@example.com",
            "plan": "trial",
            "trial_ends_at": "2099-01-01T00:00:00+00:00",
            "tokens_total": 1_000_000,
            "tokens_used": 275,
            "tokens_remaining": 999_725,
        }
    )
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        status = account_login.fetch_trial_status(load_config())
        assert status is not None
        assert status.plan == "trial"
        assert status.tokens_used == 275
        assert status.tokens_total == 1_000_000
        line = account_login.format_trial_status_line(status)
        assert line is not None
        assert "275 / 1,000,000 tokens used" in line
        assert "ends 2099-01-01" in line
    finally:
        stub.close()


def test_fetch_trial_status_swallows_server_error(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    save_persisted_profile_key("sylliptor", _ACCESS_KEY)
    stub = _StubStatusServer({"error": {"code": "status_failed"}}, status_code=500)
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        assert account_login.fetch_trial_status(load_config()) is None  # graceful on HTTP 500
    finally:
        stub.close()


def test_format_trial_status_line_expired() -> None:
    status = account_login.TrialStatus(
        plan="trial",
        email=None,
        trial_ends_at="2000-01-01T00:00:00+00:00",
        tokens_total=1000,
        tokens_used=1000,
        tokens_remaining=0,
    )
    line = account_login.format_trial_status_line(status)
    assert line is not None
    assert "expired" in line
    assert "1,000 / 1,000 tokens used" in line


def test_format_trial_status_line_empty_returns_none() -> None:
    status = account_login.TrialStatus(None, None, None, None, None, None)
    assert account_login.format_trial_status_line(status) is None


class _StubModelsServer:
    """Stands in for the proxy's GET .../llm/v1/models discovery route."""

    def __init__(self, model_ids: list[str], *, status_code: int = 200) -> None:
        body = json.dumps(
            {"object": "list", "data": [{"id": mid, "object": "model"} for mid in model_ids]}
        ).encode()

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                if not self.path.endswith("/llm/v1/models"):
                    self.send_error(404)
                    return
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def test_list_trial_models_parses_proxy_allowlist(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    stub = _StubModelsServer(["mimo-v2.5-pro", "mimo-v2-flash", "mimo-v2.5"])
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        models = account_login.list_trial_models(load_config())
        assert models == ["mimo-v2.5-pro", "mimo-v2-flash", "mimo-v2.5"]
    finally:
        stub.close()


def test_list_trial_models_empty_when_unreachable(tmp_path: Path, monkeypatch) -> None:
    _config_env(tmp_path, monkeypatch)
    stub = _StubModelsServer(["mimo"])
    base_url = stub.base_url
    stub.close()  # nothing is listening now -> connection refused
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", base_url)
    assert account_login.list_trial_models(load_config()) == []


def test_login_preserves_user_chosen_model_across_relogin(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from sylliptor_agent_cli.config import save_config
    from sylliptor_agent_cli.profiles import add_profile, get_profile

    _config_env(tmp_path, monkeypatch)
    stub = _StubExchangeServer(access_key=_ACCESS_KEY, email=None)
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        # First connect: no profile yet, so it defaults to the flagship MiMo.
        first = account_login.login(
            load_config(), browser_opener=_callback_browser("code"), timeout_s=10
        )
        assert first.model == "mimo-v2.5-pro"

        # Simulate the user picking another model in `/config`.
        cfg = load_config()
        add_profile(cfg, replace(get_profile(cfg, "sylliptor"), default_model="mimo-pro"))
        save_config(cfg)

        # Re-login must keep that choice instead of clobbering it back to MiMo.
        second = account_login.login(
            load_config(), browser_opener=_callback_browser("code"), timeout_s=10
        )
        assert second.model == "mimo-pro"
        assert load_config().model == "mimo-pro"
    finally:
        stub.close()


def test_login_migrates_legacy_bare_mimo_model(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace

    from sylliptor_agent_cli.config import save_config
    from sylliptor_agent_cli.profiles import add_profile, get_profile

    _config_env(tmp_path, monkeypatch)
    stub = _StubExchangeServer(access_key=_ACCESS_KEY, email=None)
    monkeypatch.setenv("SYLLIPTOR_SUPABASE_URL", stub.base_url)
    try:
        # Seed a profile that still carries the legacy bare "mimo" placeholder.
        account_login.login(load_config(), browser_opener=_callback_browser("code"), timeout_s=10)
        cfg = load_config()
        add_profile(cfg, replace(get_profile(cfg, "sylliptor"), default_model="mimo"))
        save_config(cfg)

        # Re-login migrates that placeholder up to the named flagship (not "mimo").
        again = account_login.login(
            load_config(), browser_opener=_callback_browser("code"), timeout_s=10
        )
        assert again.model == "mimo-v2.5-pro"
    finally:
        stub.close()
