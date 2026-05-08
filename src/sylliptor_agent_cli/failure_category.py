from __future__ import annotations

import re
from enum import StrEnum
from typing import Any


class FailureCategory(StrEnum):
    INFRA_UNAVAILABLE = "infra_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_THROTTLED = "provider_throttled"
    PLANNER_FAILED = "planner_failed"
    IMPLEMENTATION_FAILED = "implementation_failed"
    VERIFICATION_FAILED = "verification_failed"


_THROTTLE_MESSAGE_MARKERS = (
    "too many requests",
    "rate limit",
    "rate-limit",
    "rate_limit",
    "quota",
    "concurrency quota",
    "concurrent request",
    "requests per minute",
    "requests/minute",
    "tokens per minute",
    "tpm",
    "rpm",
)
_QUOTA_CONTEXT_MARKERS = (
    "exceeded",
    "exhausted",
    "insufficient",
    "limit",
    "rate",
    "throttle",
)
_HTTP_429_RE = re.compile(r"\b(?:status[_ -]?code|http|error)?\s*429\b", re.IGNORECASE)
_INFRA_UNAVAILABLE_MARKERS = (
    "docker daemon",
    "docker.sock",
    "/var/run/docker.sock",
    "bwrap",
    "bubblewrap",
    "sandbox unavailable",
    "verify sandbox unavailable",
    "no usable backend",
    "unable to launch verify environment",
)
_PROVIDER_UNAVAILABLE_MESSAGE_MARKERS = (
    "connection aborted",
    "connection refused",
    "connection reset",
    "connect timeout",
    "connecttimeout",
    "dns",
    "name or service not known",
    "name resolution",
    "network is unreachable",
    "nodename nor servname",
    "read operation timed out",
    "read timeout",
    "readtimeout",
    "remote protocol error",
    "server disconnected",
    "temporary failure in name resolution",
    "timed out",
)
_PROVIDER_UNAVAILABLE_STATUS_CODES = frozenset({408, 502, 503, 504})


def failure_category_value(category: FailureCategory | str | None) -> str | None:
    parsed = parse_failure_category(category)
    return parsed.value if parsed is not None else None


def parse_failure_category(category: FailureCategory | str | None) -> FailureCategory | None:
    if isinstance(category, FailureCategory):
        return category
    if category is None:
        return None
    try:
        return FailureCategory(str(category).strip())
    except ValueError:
        return None


def empty_failure_category_counts() -> dict[str, int]:
    return {category.value: 0 for category in FailureCategory}


def increment_failure_category_count(
    counts: dict[str, int],
    category: FailureCategory | str | None,
) -> None:
    parsed = parse_failure_category(category)
    if parsed is None:
        return
    counts[parsed.value] = int(counts.get(parsed.value, 0) or 0) + 1


def is_provider_throttling_error(error: Any) -> bool:
    status_code = _extract_status_code(error)
    if status_code == 429:
        return True

    message = str(error or "").casefold()
    if not message:
        return False
    if _HTTP_429_RE.search(message):
        return True
    if any(marker in message for marker in _THROTTLE_MESSAGE_MARKERS):
        return True
    if "quota" in message and any(marker in message for marker in _QUOTA_CONTEXT_MARKERS):
        return True
    if "concurrency" in message and any(marker in message for marker in _QUOTA_CONTEXT_MARKERS):
        return True
    return False


def is_provider_unavailable_error(error: Any) -> bool:
    status_code = _extract_status_code(error)
    if status_code in _PROVIDER_UNAVAILABLE_STATUS_CODES:
        return True

    message = str(error or "").casefold()
    if not message:
        return False
    return any(marker in message for marker in _PROVIDER_UNAVAILABLE_MESSAGE_MARKERS)


def is_infra_unavailable_error(error: Any) -> bool:
    message = str(error or "").casefold()
    if not message:
        return False
    return any(marker in message for marker in _INFRA_UNAVAILABLE_MARKERS)


def _extract_status_code(error: Any) -> int | None:
    for attr_name in ("status_code", "code"):
        status_code = _coerce_status_code(getattr(error, attr_name, None))
        if status_code is not None:
            return status_code

    response = getattr(error, "response", None)
    if response is not None:
        status_code = _coerce_status_code(getattr(response, "status_code", None))
        if status_code is not None:
            return status_code

    match = re.search(r"\b(?:LLM|Responses)?\s*error\s+(\d{3})\b", str(error or ""), re.IGNORECASE)
    if match is None:
        return None
    return _coerce_status_code(match.group(1))


def _coerce_status_code(value: Any) -> int | None:
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    if status_code <= 0:
        return None
    return status_code
