"""Endpoints and identifiers for the hosted Sylliptor MiMo service (Xiaomi trial).

The CLI talks to a Supabase-hosted proxy that holds the OpenRouter/Xiaomi BYOK
key server-side; the CLI only ever holds the user's own ``access_key``. These
values can be overridden via environment variables to point at a different
deployment (e.g. a staging project) during testing.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

# Supabase project hosting the `llm` proxy + `cli-auth` edge functions.
_DEFAULT_SUPABASE_URL = "https://vzigujbcjjmpntxhmyvr.supabase.co"
# Marketing/account site that serves the /cli-login approval page.
_DEFAULT_SITE_URL = "https://sylliptor.alysisai.com"

# The profile/preset key used for the hosted MiMo provider.
PROFILE_KEY = "sylliptor"

# Default model the CLI selects on first login — Xiaomi's flagship reasoning/
# coding/agent model. The proxy honours any model in its server-side allowlist
# (MIMO_ALLOWED_MODELS) and otherwise pins to its canonical MiMo model, so this is
# just the first-connect default; the user can switch in /config or via /model.
SYLLIPTOR_MIMO_MODEL = "mimo-v2.5-pro"

# Default proxy base URL (kept in sync with the `sylliptor` profile preset).
DEFAULT_PROXY_BASE_URL = f"{_DEFAULT_SUPABASE_URL}/functions/v1/llm/v1"


# Loopback hosts may use http:// (local stubs / tests); every other host must
# be https so the one-time code and access_key never travel in cleartext.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class SylliptorCloudConfigError(ValueError):
    """Raised when a configured Sylliptor cloud URL is unsafe (e.g. cleartext http)."""


def _clean(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def _checked(url: str) -> str:
    """Clean a URL and reject cleartext http:// for non-loopback hosts.

    The one-time login code and the long-lived access_key travel to these
    endpoints, so a downgraded (http://) origin from an env override would leak
    them. https is required unless the host is loopback (local stubs / tests) or
    SYLLIPTOR_ALLOW_INSECURE_URLS is explicitly set.
    """
    cleaned = _clean(url)
    if not cleaned:
        return cleaned
    parts = urlsplit(cleaned)
    if parts.scheme.lower() == "https":
        return cleaned
    host = (parts.hostname or "").lower()
    if host in _LOOPBACK_HOSTS or os.environ.get("SYLLIPTOR_ALLOW_INSECURE_URLS"):
        return cleaned
    raise SylliptorCloudConfigError(
        f"Refusing to use insecure Sylliptor URL {cleaned!r}: https is required "
        "(set SYLLIPTOR_ALLOW_INSECURE_URLS=1 only for trusted local testing)."
    )


def supabase_url() -> str:
    return _checked(os.environ.get("SYLLIPTOR_SUPABASE_URL") or _DEFAULT_SUPABASE_URL)


def site_url() -> str:
    return _checked(os.environ.get("SYLLIPTOR_SITE_URL") or _DEFAULT_SITE_URL)


def proxy_base_url() -> str:
    """OpenAI-compatible base URL; the LLM client appends ``/chat/completions``."""
    override = os.environ.get("SYLLIPTOR_PROXY_BASE_URL")
    if override:
        return _checked(override)
    return f"{supabase_url()}/functions/v1/llm/v1"


def cli_login_url() -> str:
    """The website page that approves a CLI login and mints a one-time code."""
    return f"{site_url()}/cli-login"


def token_exchange_url() -> str:
    """The edge-function endpoint that swaps a one-time code for the access_key."""
    return f"{supabase_url()}/functions/v1/cli-auth/exchange"


def status_url() -> str:
    """Read-only endpoint returning the caller's trial status (days + tokens)."""
    return f"{proxy_base_url()}/status"


def models_url() -> str:
    """OpenAI-style discovery endpoint listing the model ids the proxy will serve.

    The proxy returns ``{"object": "list", "data": [{"id": ...}, ...]}`` reflecting
    its server-side allowlist, so the CLI can offer the trial's real models instead
    of guessing. Public (no access_key required).
    """
    return f"{proxy_base_url()}/models"
