"""Provider-call helpers for the main agent loop.

These functions are transport/compatibility shims around ``client.chat`` and
LLM error classification. They are used by the agent loop, session runtime,
and the non-repo response path alike, and carry no routing semantics of their
own; they live here so the agent loop does not depend on the router module for
its own provider calls.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..language_policy import DEFAULT_REPLY_LANGUAGE, DEFAULT_REPLY_SCRIPT
from ..llm.base import effective_tools_for_client
from ..llm.types import LLMError
from .prompt_context import _INLINE_CODE_SPAN_RE
from .turn_path import (
    _build_turn_language_system_message,
    _normalize_turn_language_name,
    _normalize_turn_script_name,
)

if TYPE_CHECKING:
    from .tools_assembly import ToolDef


def _llm_error_status_code(err: LLMError) -> int | None:
    match = re.match(r"LLM error (\d{3}):", str(err or "").strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_fatal_non_repo_llm_error(err: LLMError) -> bool:
    status_code = _llm_error_status_code(err)
    # Provider/account and request-shape failures cannot be repaired by the
    # static clarification route. Surface them after the transport's own
    # compatibility retries instead of disguising them as a successful
    # "Could you clarify..." response.
    if status_code in {400, 401, 402, 403, 404, 422, 429}:
        return True

    msg = str(err).lower()
    markers = (
        "invalid_api_key",
        "incorrect api key",
        "api key",
        "authentication",
        "unauthorized",
        "forbidden",
        "permission denied",
        "access denied",
        "credit balance",
        "purchase credits",
        "billing",
        "insufficient_quota",
        "quota exhausted",
        "quota exceeded",
        "rate limit",
    )
    if any(marker in msg for marker in markers):
        return True

    # Recognized Sylliptor MiMo trial proxy errors (trial_expired,
    # quota_exhausted, rate_limit_exceeded, ...) must propagate so the REPL's
    # friendly-error renderer shows an actionable message, instead of being
    # swallowed into the generic "Could you clarify..." fallback. Lazy import
    # avoids an llm -> agent import cycle.
    from ..llm.openai_compat import sylliptor_trial_error_message

    return sylliptor_trial_error_message(err) is not None


def _is_stream_unsupported_error(err: LLMError) -> bool:
    msg = str(err).lower()
    mentions_stream = "stream" in msg or "sse" in msg
    unsupported = "unsupported" in msg or "not support" in msg
    bad_status = "llm error 400" in msg or "llm error 404" in msg or "llm error 422" in msg
    return mentions_stream and (unsupported or bad_status)


def _tool_schema_function_name(tool_schema: dict[str, Any]) -> str:
    if not isinstance(tool_schema, dict):
        return ""
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _registered_tool_schema_list(
    tool_defs: dict[str, ToolDef],
    tool_list: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tool_defs:
        return []
    canonical_by_name = {name: tool.as_openai_tool() for name, tool in tool_defs.items()}
    ordered: list[dict[str, Any]] = []
    added: set[str] = set()
    for tool_schema in tool_list or []:
        name = _tool_schema_function_name(tool_schema)
        if name not in canonical_by_name or name in added:
            continue
        ordered.append(canonical_by_name[name])
        added.add(name)
    for name, tool_schema in canonical_by_name.items():
        if name not in added:
            ordered.append(tool_schema)
    return ordered


def _main_agent_chat(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    stream: bool,
    on_text_delta: Callable[[str], None] | None,
    on_reasoning_delta: Callable[[str], None] | None = None,
    temperature: float | None = None,
    cancellation_token: Any | None = None,
    tool_choice: Any | None = None,
) -> Any:
    tools = effective_tools_for_client(client, tools)
    if tools is None:
        tool_choice = None
    kwargs: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "stream": stream,
        "on_text_delta": on_text_delta,
        "on_reasoning_delta": on_reasoning_delta,
        "temperature": temperature,
    }
    # Pass the token only when present so older/test clients keep working via the
    # TypeError fallbacks below; clients that accept it can abort an in-flight
    # request the instant the user interrupts (even before the first token).
    if cancellation_token is not None:
        kwargs["cancellation_token"] = cancellation_token
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    for compatibility_arg in (
        "cancellation_token",
        "on_reasoning_delta",
        "temperature",
        "tool_choice",
    ):
        try:
            return client.chat(**kwargs)
        except TypeError:
            kwargs.pop(compatibility_arg, None)
    return client.chat(**kwargs)


def _client_supports_tool_calling(client: Any) -> bool:
    return getattr(client, "supports_tool_calling", True) is not False


def _client_reasoning_or_thinking_active(client: Any) -> bool:
    for attr in ("reasoning_active", "thinking_active"):
        value = getattr(client, attr, None)
        if isinstance(value, bool):
            return value
    if getattr(client, "enable_thinking", None) is True:
        return True
    reasoning_effort = str(getattr(client, "reasoning_effort", "") or "").strip().casefold()
    return bool(reasoning_effort and reasoning_effort != "none")


def _client_supports_forced_tool_choice(client: Any) -> bool:
    return (
        getattr(client, "supports_forced_tool_choice", False) is True
        and _client_supports_tool_calling(client)
        and not _client_reasoning_or_thinking_active(client)
    )


def _function_tool_choice(tool_name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": str(tool_name)}}


def _safe_forced_tool_choice_for_recovery(
    *,
    client: Any,
    tools: list[dict[str, Any]] | None,
    preferred_tool_names: tuple[str, ...],
) -> dict[str, Any] | None:
    if not _client_supports_forced_tool_choice(client):
        return None
    available: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = _tool_schema_function_name(tool)
        if name:
            available.add(name)
    for name in preferred_tool_names:
        if name in available:
            return _function_tool_choice(name)
    return None


# ---------------------------------------------------------------------------
# Ephemeral request-message injection
# ---------------------------------------------------------------------------


def _request_messages_with_ephemeral_system_prompts(
    *,
    messages: list[dict[str, Any]],
    insert_index: int,
    prompts: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    cleaned_prompts = [str(prompt or "").strip() for prompt in (prompts or [])]
    cleaned_prompts = [prompt for prompt in cleaned_prompts if prompt]
    if not cleaned_prompts:
        return list(messages)
    bounded_index = max(0, min(len(messages), insert_index))
    injected = [{"role": "system", "content": prompt} for prompt in cleaned_prompts]
    return list(messages[:bounded_index]) + injected + list(messages[bounded_index:])


def _request_messages_with_ephemeral_system_prompt_suffixes(
    *,
    messages: list[dict[str, Any]],
    prompts: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    cleaned_prompts = [str(prompt or "").strip() for prompt in (prompts or [])]
    cleaned_prompts = [prompt for prompt in cleaned_prompts if prompt]
    if not cleaned_prompts:
        return list(messages)
    injected = [{"role": "system", "content": prompt} for prompt in cleaned_prompts]
    return list(messages) + injected


def _request_messages_with_ephemeral_user_messages(
    *,
    messages: list[dict[str, Any]],
    insert_index: int,
    contents: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    cleaned_contents = [str(content or "").strip() for content in (contents or [])]
    cleaned_contents = [content for content in cleaned_contents if content]
    if not cleaned_contents:
        return list(messages)
    bounded_index = max(0, min(len(messages), insert_index))
    injected = [{"role": "user", "content": content} for content in cleaned_contents]
    return list(messages[:bounded_index]) + injected + list(messages[bounded_index:])


# ---------------------------------------------------------------------------
# Final-summary language rewrite
# ---------------------------------------------------------------------------


_FINAL_SUMMARY_REWRITE_SYSTEM_PROMPT = """Rewrite one successful coding-task final summary into the selected reply language/script.

Output only the rewritten summary.
Preserve technical meaning exactly.
Do not add or remove implementation claims, test claims, warnings, or blockers.
Keep file paths, code identifiers, CLI commands, config keys, JSON keys, fenced code blocks, and inline code exactly as written.
If the summary is already in the requested language/script, return it unchanged.
"""


_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


_REWRITE_PROTECTED_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_./:-]*[./_-][A-Za-z0-9_./:-]+\b")


def _non_repo_chat(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
    on_text_delta: Callable[[str], None] | None = None,
    on_reasoning_delta: Callable[[str], None] | None = None,
) -> Any:
    tools = effective_tools_for_client(client, tools)
    base: dict[str, Any] = {"messages": messages, "tools": tools, "stream": stream}
    try:
        return client.chat(
            **base,
            on_text_delta=on_text_delta,
            on_reasoning_delta=on_reasoning_delta,
            temperature=temperature,
        )
    except TypeError:
        pass
    try:
        return client.chat(**base, on_text_delta=on_text_delta, temperature=temperature)
    except TypeError:
        pass
    try:
        return client.chat(**base, temperature=temperature)
    except TypeError:
        return client.chat(**base)


def _extract_rewrite_protected_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    seen: set[str] = set()

    def _add(fragment: str) -> None:
        candidate = str(fragment or "")
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        fragments.append(candidate)

    for fragment in _FENCED_CODE_BLOCK_RE.findall(str(text or "")):
        _add(fragment)
    for fragment in _INLINE_CODE_SPAN_RE.findall(str(text or "")):
        _add(fragment)
    for token in _REWRITE_PROTECTED_TOKEN_RE.findall(str(text or "")):
        _add(token)
    return fragments


def _rewritten_text_preserves_technical_tokens(original: str, rewritten: str) -> bool:
    if not original.strip():
        return True
    if not rewritten.strip():
        return False
    protected = _extract_rewrite_protected_fragments(original)
    return all(fragment in rewritten for fragment in protected)


def _rewrite_final_summary_for_language(
    *,
    client: Any,
    final_text: str,
    language: str = "",
    script: str = "",
    explicit_language_override: bool = False,
    record_usage: Callable[..., Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    clean = str(final_text or "").strip()
    if not clean:
        return clean, None

    resolved_language = _normalize_turn_language_name(language)
    resolved_script = _normalize_turn_script_name(script)
    if not (resolved_language or resolved_script):
        return clean, None
    if (
        not explicit_language_override
        and resolved_language == DEFAULT_REPLY_LANGUAGE
        and resolved_script in {"", DEFAULT_REPLY_SCRIPT}
    ):
        return clean, None

    payload: dict[str, Any] = {
        "language": resolved_language,
        "script": resolved_script,
        "explicit_language_override": explicit_language_override,
    }
    language_directive = _build_turn_language_system_message(
        resolved_language,
        resolved_script,
        explicit_language_override=explicit_language_override,
    )
    if not language_directive:
        payload.update({"status": "skipped", "reason": "no_language_directive"})
        return clean, payload

    protected_fragments = _extract_rewrite_protected_fragments(clean)
    payload["protected_fragment_count"] = len(protected_fragments)
    rewrite_messages = [
        {"role": "system", "content": _FINAL_SUMMARY_REWRITE_SYSTEM_PROMPT},
        {"role": "system", "content": language_directive},
        {
            "role": "user",
            "content": (
                "Rewrite this final assistant summary only.\n\n"
                "<assistant_summary>\n"
                f"{clean}\n"
                "</assistant_summary>"
            ),
        },
    ]
    try:
        response = _non_repo_chat(client=client, messages=rewrite_messages, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        payload.update({"status": "kept_original", "reason": "rewrite_error", "error": str(exc)})
        return clean, payload

    if record_usage is not None:
        record_usage(
            response=response,
            messages=rewrite_messages,
            tool_list=None,
            operation="final_summary_language_rewrite",
        )

    rewritten = str(getattr(response, "content", "") or "").strip()
    if not rewritten:
        payload.update({"status": "kept_original", "reason": "empty_rewrite"})
        return clean, payload
    if not _rewritten_text_preserves_technical_tokens(clean, rewritten):
        payload.update({"status": "kept_original", "reason": "protected_tokens_missing"})
        return clean, payload
    payload.update(
        {
            "status": "applied" if rewritten != clean else "unchanged",
            "reason": "rewrite_succeeded",
        }
    )
    return rewritten, payload
