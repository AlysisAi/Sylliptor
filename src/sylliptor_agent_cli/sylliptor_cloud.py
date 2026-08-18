"""Endpoints and identifiers for the hosted Sylliptor Pro service.

`sylliptor login` runs an RFC 8628-style device flow against Supabase Edge
Functions (`device-code` / `device-token`); the user approves the code on the
account website's /activate page. The CLI receives a gateway key (``slk_…``)
and talks to the Sylliptor LLM gateway — an OpenAI-compatible proxy that meters
the subscription's credits server-side. The CLI never holds upstream provider
keys for Pro; BYOK profiles are configured separately and take precedence.

Values can be overridden via environment variables to point at a different
deployment (e.g. a staging project or local stubs) during testing.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

# Supabase project hosting the device-login edge functions.
_DEFAULT_SUPABASE_URL = "https://vzigujbcjjmpntxhmyvr.supabase.co"
# Marketing/account site that serves the /activate approval page.
_DEFAULT_SITE_URL = "https://sylliptor.alysisai.com"

# Supabase "anon" key: a PUBLIC client identifier (shipped in the website's
# browser bundle too). It grants nothing by itself — the edge functions are
# either public-by-design (device flow) or JWT/secret-guarded server-side.
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ6aWd1amJjamptcG50eGhteXZyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA5Mzc0NTIsImV4cCI6MjA5NjUxMzQ1Mn0."
    "vLH9q-BNO8IWIZrVlvCw8pZWXdLgmKG4Tl9toTTD3pg"
)

# The profile/preset key used for the hosted Pro provider.
PROFILE_KEY = "sylliptor"

# LEGACY: base URL of the retired MiMo-trial proxy (an OpenRouter-forwarding
# Supabase Edge Function). The service is gone, but URL classifiers (web
# search / provider limits) still recognize it so configs from that era keep
# loading with sensible behavior instead of misclassifying.
DEFAULT_PROXY_BASE_URL = f"{_DEFAULT_SUPABASE_URL}/functions/v1/llm/v1"


# Loopback hosts may use http:// (local stubs / tests); every other host must
# be https so device codes and the gateway key never travel in cleartext.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class SylliptorCloudConfigError(ValueError):
    """Raised when a configured Sylliptor cloud URL is unsafe (e.g. cleartext http)."""


def _clean(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def _checked(url: str) -> str:
    """Clean a URL and reject cleartext http:// for non-loopback hosts.

    Device codes and the long-lived gateway key travel to these endpoints, so a
    downgraded (http://) origin from an env override would leak them. https is
    required unless the host is loopback (local stubs / tests) or
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


def gateway_base_url() -> str:
    """OpenAI-compatible base URL; the LLM client appends ``/chat/completions``.

    The hosted proxy runs as the `llm` Supabase Edge Function (the same shape
    the MiMo-trial proxy used), holding the upstream DeepSeek key server-side
    and metering each account's allowance/credits.
    """
    override = os.environ.get("SYLLIPTOR_GATEWAY_URL")
    if override:
        return _checked(override)
    return f"{supabase_url()}/functions/v1/llm/v1"


def device_code_url() -> str:
    """POST here to start a device login (returns user_code + device_code)."""
    return f"{supabase_url()}/functions/v1/device-code"


def device_token_url() -> str:
    """POST device_code here until the user approves (returns the slk_ key)."""
    return f"{supabase_url()}/functions/v1/device-token"


def activate_url() -> str:
    """The website page where a signed-in user approves a device code."""
    return f"{site_url()}/activate"


def models_url() -> str:
    """The gateway's OpenAI-shaped model listing."""
    return f"{gateway_base_url()}/models"
