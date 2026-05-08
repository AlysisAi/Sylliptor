from __future__ import annotations

import re

_SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(r"(Authorization\s*:\s*)(.+)", re.IGNORECASE),
]
_DEFAULT_MAX_ERROR_SUMMARY_CHARS = 320


def redact_sensitive_error_text(text: str) -> str:
    out = str(text or "")
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.pattern.lower().startswith("(authorization"):
            out = pattern.sub(r"\1[REDACTED]", out)
            continue
        if pattern.pattern.lower().startswith("(bearer"):
            out = pattern.sub(r"\1[REDACTED]", out)
            continue
        out = pattern.sub("[REDACTED]", out)
    return out


def sanitize_error_summary(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_ERROR_SUMMARY_CHARS,
) -> str:
    clean = redact_sensitive_error_text(" ".join(str(text or "").split())).strip()
    if not clean:
        return "No additional error details."
    if len(clean) <= max_chars:
        return clean
    if max_chars <= 15:
        return clean[:max_chars]
    return clean[: max_chars - 15].rstrip() + "...(truncated)"


def sanitize_optional_error_summary(
    text: str | None,
    *,
    max_chars: int = _DEFAULT_MAX_ERROR_SUMMARY_CHARS,
) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    return sanitize_error_summary(raw, max_chars=max_chars)
