from __future__ import annotations

import os
import re
from typing import Any

_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|credential|authorization)",
    re.IGNORECASE,
)
_SECRET_VALUE_MIN_LENGTH = 6
_GENERIC_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"),
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b([A-Z0-9_.-]*(?:api[_-]?key|token|secret|password|credential)"
    r"[A-Z0-9_.-]*\s*[:=]\s*)([^\s,;&]+)"
)
_AUTHORIZATION_VALUE_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[A-Za-z0-9._~+/=-]{8,}"
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_URL_USERINFO_PATTERN = re.compile(
    r"(?i)\b((?:git\+)?https?://)([^/@\s]+)@([A-Za-z0-9.-]+(?::\d+)?)"
)
_PRIVATE_KEY_BLOCK_PATTERNS = (
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^PuTTY-User-Key-File-\d+:[^\r\n]*.*?^Private-MAC:[^\r\n]*",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    ),
)


def _environment_secret_values() -> tuple[str, ...]:
    values = [
        value
        for key, value in os.environ.items()
        if value and len(value) >= _SECRET_VALUE_MIN_LENGTH and _SECRET_KEY_PATTERN.search(key)
    ]
    return tuple(dict.fromkeys(values))


def redact_secrets(value: Any, *, extra_secrets: tuple[str, ...] = ()) -> Any:
    """Redact common credentials from nested values before persistence or display."""

    secrets = tuple(secret for secret in (*_environment_secret_values(), *extra_secrets) if secret)

    def _redact_text(text: str) -> str:
        redacted = text
        for secret in secrets:
            redacted = redacted.replace(secret, "<redacted>")
        redacted = _AUTHORIZATION_VALUE_PATTERN.sub(r"\1<redacted>", redacted)
        redacted = _BEARER_TOKEN_PATTERN.sub("Bearer <redacted>", redacted)
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1<redacted>", redacted)
        redacted = _URL_USERINFO_PATTERN.sub(r"\1<redacted>@\3", redacted)
        for pattern in _PRIVATE_KEY_BLOCK_PATTERNS:
            redacted = pattern.sub("<redacted-private-key>", redacted)
        for pattern in _GENERIC_SECRET_PATTERNS:
            redacted = pattern.sub("<redacted>", redacted)
        return redacted

    def _walk(item: Any, *, key_hint: str = "") -> Any:
        if isinstance(item, str):
            if _SECRET_KEY_PATTERN.search(key_hint):
                return "<redacted>" if item else item
            return _redact_text(item)
        if isinstance(item, list):
            return [_walk(child, key_hint=key_hint) for child in item]
        if isinstance(item, tuple):
            return [_walk(child, key_hint=key_hint) for child in item]
        if isinstance(item, dict):
            return {
                str(key): (
                    "<redacted>"
                    if _SECRET_KEY_PATTERN.search(str(key)) and isinstance(child, str) and child
                    else _walk(child, key_hint=str(key))
                )
                for key, child in item.items()
            }
        return item

    return _walk(value)
