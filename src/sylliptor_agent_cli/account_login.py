"""Browser-based login for the hosted Sylliptor MiMo trial (Xiaomi).

`sylliptor login` opens the account website, the user signs in, and the page
hands back a short-lived one-time code over a localhost callback. The CLI swaps
that code (via the `cli-auth` edge function) for the user's long-lived
``access_key`` — never the upstream OpenRouter/Xiaomi key — stores it, and
activates a `sylliptor` provider profile with MiMo as the default model.

Only the disposable code ever travels through a URL; the access_key is returned
in an HTTPS response body and persisted to the local credentials file.
"""

from __future__ import annotations

import html
import json
import math
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

from . import sylliptor_cloud as cloud
from .config import (
    AppConfig,
    clear_persisted_profile_key,
    load_persisted_profile_keys,
    save_config,
    save_persisted_profile_key,
)
from .host_browser import open_url
from .profile_presets import get_preset, make_profile_from_preset
from .profiles import ProfileSpec, add_profile, get_profile, set_active_profile

_CALLBACK_PATH = "/callback"
_DEFAULT_TIMEOUT_S = 180.0
_EXCHANGE_TIMEOUT_S = 15.0
_STATUS_TIMEOUT_S = 10.0
# Kept short: model discovery runs inline while rendering the interactive `/config`
# model picker, so an offline/slow proxy must not stall the menu for long.
_MODELS_TIMEOUT_S = 6.0


class SylliptorLoginError(Exception):
    """Raised when the browser login handshake fails."""


@dataclass(frozen=True)
class _CallbackPayload:
    code: str | None
    state: str | None
    error: str | None = None


@dataclass(frozen=True)
class LoginResult:
    email: str | None
    profile_name: str
    base_url: str
    model: str


@dataclass(frozen=True)
class LoginStatus:
    logged_in: bool
    profile_name: str
    base_url: str
    active: bool
    key_preview: str | None


@dataclass(frozen=True)
class TrialStatus:
    plan: str | None
    email: str | None
    trial_ends_at: str | None
    tokens_total: int | None
    tokens_used: int | None
    tokens_remaining: int | None


# ---------------------------------------------------------------------------
# Localhost callback listener (a minimal, self-contained version of the MCP
# OAuth listener — it only needs to capture ?code=&state= on /callback).
# ---------------------------------------------------------------------------
class _CallbackState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.payload: _CallbackPayload | None = None
        self._lock = threading.Lock()

    def set_payload(self, payload: _CallbackPayload) -> None:
        with self._lock:
            if self.payload is None:
                self.payload = payload
                self.event.set()


class _CallbackHandler(BaseHTTPRequestHandler):
    server_version = "SylliptorCliLogin/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path != _CALLBACK_PATH:
            self.send_error(404)
            return
        query = parse_qs(parsed.query)
        payload = _CallbackPayload(
            code=(query.get("code") or [None])[0],
            state=(query.get("state") or [None])[0],
            error=(query.get("error") or [None])[0],
        )
        self.server.callback_state.set_payload(payload)  # type: ignore[attr-defined]
        message = (
            "Sylliptor login failed. Return to the terminal for details."
            if payload.error or not payload.code
            else "Sylliptor login complete. You can close this window and return to the terminal."
        )
        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>Sylliptor CLI login</title></head>"
            f'<body style="font-family:system-ui;padding:2rem"><p>{html.escape(message)}</p>'
            "</body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True


class _LoopbackListener:
    def __init__(self) -> None:
        self.callback_state = _CallbackState()
        self._server = _LoopbackServer(("127.0.0.1", 0), _CallbackHandler)
        self._server.callback_state = self.callback_state  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._closed = False

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{_CALLBACK_PATH}"

    def wait_for_callback(self, *, timeout_s: float) -> _CallbackPayload:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SylliptorLoginError(
                    f"Timed out after {int(timeout_s)}s waiting for the browser login. "
                    "Run `sylliptor login` again."
                )
            if self.callback_state.event.wait(timeout=min(0.2, remaining)):
                payload = self.callback_state.payload
                if payload is None:  # pragma: no cover - defensive
                    raise SylliptorLoginError("Login callback completed without data.")
                return payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def login(
    cfg: AppConfig,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    browser_opener: Callable[[str], bool] | None = None,
    output_write: Callable[[str], None] | None = None,
) -> LoginResult:
    """Run the browser login handshake and persist the resulting access_key."""
    writer = output_write or (lambda _msg: None)

    # Fail fast on an insecure (non-https) cloud URL before opening a browser,
    # so neither the one-time code nor the access_key could leave over http://.
    try:
        cloud.cli_login_url()
        cloud.token_exchange_url()
        cloud.proxy_base_url()
    except cloud.SylliptorCloudConfigError as exc:
        raise SylliptorLoginError(str(exc)) from exc

    listener = _LoopbackListener()
    try:
        state = secrets.token_hex(32)
        params = {"port": str(listener.port), "state": state}
        login_url = f"{cloud.cli_login_url()}?{urlencode(params)}"

        writer(f"Opening your browser to sign in:\n  {login_url}")
        opener = browser_opener or _open_browser
        if not _safe_open(opener, login_url):
            writer("Could not open a browser automatically. Open the URL above manually.")

        payload = listener.wait_for_callback(timeout_s=timeout_s)
    finally:
        listener.close()

    if payload.error:
        raise SylliptorLoginError(f"Login was rejected: {payload.error}")
    if not payload.code:
        raise SylliptorLoginError("Login did not return a code. Run `sylliptor login` again.")
    # Compare as bytes: payload.state is attacker-controlled and may contain
    # non-ASCII, which secrets.compare_digest rejects with TypeError on str.
    if not payload.state or not secrets.compare_digest(
        payload.state.encode("utf-8", "replace"), state.encode("utf-8")
    ):
        raise SylliptorLoginError("Login state mismatch — aborting for safety. Try again.")

    access_key, email = _exchange_code(payload.code)

    try:
        # Create + activate the profile first (this saves config); only then
        # persist the access_key, so a save failure never leaves a stored key
        # pointing at an unconfigured profile. resolve_api_key then returns the
        # access_key as the Bearer for the `sylliptor` profile (no other wiring).
        result = _activate_sylliptor_profile(cfg, email=email)
        save_persisted_profile_key(cloud.PROFILE_KEY, access_key)
    except OSError as exc:
        raise SylliptorLoginError(
            f"Logged in, but couldn't save your session locally: {exc}"
        ) from exc
    return result


def logout(cfg: AppConfig) -> bool:
    """Forget the stored access_key. Returns True if something was cleared."""
    return clear_persisted_profile_key(cloud.PROFILE_KEY)


def login_status(cfg: AppConfig) -> LoginStatus:
    """Report whether a hosted MiMo session is connected."""
    stored = load_persisted_profile_keys().get(cloud.PROFILE_KEY)
    active = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    profile = get_profile(cfg, cloud.PROFILE_KEY)
    base_url = profile.base_url if profile is not None else cloud.proxy_base_url()
    preview = None
    if stored:
        preview = stored[:8] + "…" if len(stored) > 8 else stored
    return LoginStatus(
        logged_in=bool(stored),
        profile_name=cloud.PROFILE_KEY,
        base_url=base_url,
        active=active == cloud.PROFILE_KEY,
        key_preview=preview,
    )


def fetch_trial_status(cfg: AppConfig) -> TrialStatus | None:
    """Fetch live trial status (days left, token usage) from the proxy.

    Best-effort: returns None on any failure (not logged in, offline, non-200)
    so callers degrade gracefully — `whoami` still shows local state offline.
    """
    access_key = load_persisted_profile_keys().get(cloud.PROFILE_KEY)
    if not access_key:
        return None
    request = urllib.request.Request(
        cloud.status_url(),
        headers={"Authorization": f"Bearer {access_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=_STATUS_TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    return TrialStatus(
        plan=_str_or_none(body.get("plan")),
        email=_str_or_none(body.get("email")),
        trial_ends_at=_str_or_none(body.get("trial_ends_at")),
        tokens_total=_int_or_none(body.get("tokens_total")),
        tokens_used=_int_or_none(body.get("tokens_used")),
        tokens_remaining=_int_or_none(body.get("tokens_remaining")),
    )


def list_trial_models(cfg: AppConfig) -> list[str]:
    """List the model ids the hosted proxy currently allows, via ``/v1/models``.

    The proxy serves a server-side allowlist (just MiMo by default, but the deploy
    can widen it), and exposes it as an OpenAI-shaped ``{"data": [{"id": ...}]}``.
    Surfacing it lets `/config` show the trial's real models instead of pinning a
    single hard-coded id. Best-effort: returns ``[]`` on any failure (offline,
    non-200, malformed) so callers fall back to the static preset model. The
    endpoint is public; the access_key is sent only when present (harmless and
    forward-compatible) and never required.
    """
    try:
        url = cloud.models_url()
    except cloud.SylliptorCloudConfigError:
        return []
    headers = {}
    access_key = load_persisted_profile_keys().get(cloud.PROFILE_KEY)
    if access_key:
        headers["Authorization"] = f"Bearer {access_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_MODELS_TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeDecodeError):
        return []
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if model_id and model_id not in models:
            models.append(model_id)
    return models


def format_trial_status_line(status: TrialStatus) -> str | None:
    """A one-line trial summary for `whoami`, or None if there's nothing to show."""
    parts: list[str] = []
    days = _trial_days_left(status.trial_ends_at)
    if days is not None:
        parts.append("expired" if days <= 0 else f"{days} day{'s' if days != 1 else ''} left")
    ends = _format_date(status.trial_ends_at)
    if ends:
        parts.append(f"ends {ends}")
    tokens = _format_token_usage(status.tokens_used, status.tokens_total)
    if tokens:
        parts.append(tokens)
    return "Trial: " + " · ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _str_or_none(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _trial_days_left(trial_ends_at: str | None) -> int | None:
    ends = _parse_iso(trial_ends_at)
    if ends is None:
        return None
    seconds = (ends - datetime.now(UTC)).total_seconds()
    return 0 if seconds <= 0 else math.ceil(seconds / 86400)


def _format_date(value: str | None) -> str | None:
    dt = _parse_iso(value)
    return dt.date().isoformat() if dt is not None else None


def _format_token_usage(used: int | None, total: int | None) -> str | None:
    if used is None and total is None:
        return None
    if total:
        return f"{used or 0:,} / {total:,} tokens used"
    return f"{used or 0:,} tokens used"


def _exchange_code(code: str) -> tuple[str, str | None]:
    """Swap a one-time code for (access_key, email) via the cli-auth function."""
    request = urllib.request.Request(
        cloud.token_exchange_url(),
        data=json.dumps({"code": code}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_EXCHANGE_TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SylliptorLoginError(_exchange_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise SylliptorLoginError(
            f"Could not reach the Sylliptor login service: {exc.reason}"
        ) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise SylliptorLoginError("Login service returned an invalid response.") from exc
    except (TimeoutError, OSError) as exc:  # noqa: BLE001
        raise SylliptorLoginError(f"Login request failed: {exc}") from exc

    access_key = str((body or {}).get("access_key") or "").strip()
    if not access_key:
        raise SylliptorLoginError("Login service did not return an access key.")
    email = body.get("email") if isinstance(body, dict) else None
    return access_key, (str(email).strip() or None if email else None)


def _exchange_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(exc.read().decode("utf-8"))
        message = detail.get("error", {}).get("message")
        if message:
            return str(message)
    except Exception:  # noqa: BLE001 - best-effort error extraction
        pass
    return f"Login exchange failed (HTTP {exc.code})."


def _activate_sylliptor_profile(cfg: AppConfig, *, email: str | None) -> LoginResult:
    """Create + activate the `sylliptor` profile, keeping any model the user chose.

    We never auto-select a model here. The hosted MiMo trial is no longer
    provisioned, so on first connect the default model is left EMPTY and the user
    picks one (``/model`` in chat or ``sylliptor config set model``); defaulting to a
    MiMo id would point at a model we no longer serve. On later logins we preserve
    whatever model the user has since selected instead of clobbering it, so
    re-logging in never undoes their choice.
    """
    preset = get_preset(cloud.PROFILE_KEY)
    if preset is not None:
        profile = make_profile_from_preset(preset, name=cloud.PROFILE_KEY)
    else:  # pragma: no cover - preset is always present
        profile = ProfileSpec(name=cloud.PROFILE_KEY, protocol="openai_compat")

    existing = get_profile(cfg, cloud.PROFILE_KEY)
    existing_model = str(getattr(existing, "default_model", "") or "").strip() if existing else ""
    # The legacy bare "mimo" placeholder is not a real model id; drop it so the
    # profile no longer carries a name that maps to a model we no longer serve.
    if existing_model.casefold() == "mimo":
        existing_model = ""
    # No default model: the free MiMo trial is gone, so we never auto-select MiMo
    # (or anything). The user chooses a model after login; an unset model surfaces a
    # clear "Model is not set" prompt instead of silently routing to a dead model.
    chosen_model = existing_model

    # Always pin to the live proxy URL (env-overridable for tests); keep the user's
    # chosen model (empty until they pick one — we no longer default to MiMo).
    profile = ProfileSpec(
        name=profile.name,
        protocol=profile.protocol,
        base_url=cloud.proxy_base_url(),
        api_key_env=None,
        extra_headers=dict(profile.extra_headers),
        default_model=chosen_model,
        reasoning_effort=(existing.reasoning_effort if existing is not None else None),
        reasoning_trace_adapter=(
            existing.reasoning_trace_adapter
            if existing is not None
            else profile.reasoning_trace_adapter
        ),
        web_search_adapter=profile.web_search_adapter,
        web_search_model=profile.web_search_model,
        notes=profile.notes,
    )

    add_profile(cfg, profile)
    set_active_profile(cfg, profile.name)
    save_config(cfg)

    return LoginResult(
        email=email,
        profile_name=profile.name,
        base_url=profile.base_url,
        model=profile.default_model,
    )


def _open_browser(url: str) -> bool:
    return open_url(url)


def _safe_open(opener: Callable[[str], bool], url: str) -> bool:
    try:
        return bool(opener(url))
    except Exception:  # noqa: BLE001 - browser launch is best-effort
        return False
