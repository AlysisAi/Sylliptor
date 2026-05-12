from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .llm.openai_compat import strip_provider_metadata_from_message
from .model_registry import ModelMeta, ModelRegistry
from .request_estimation import RequestTokenBreakdown, estimate_request_token_breakdown
from .session_store import read_session_events
from .token_budget import compute_input_budget, estimate_tokens


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class UsageRecord:
    timestamp: str
    role: str
    requested_model: str
    response_model: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    cost_usd: float | None
    usage_source: str
    cached_prompt_tokens: int | None = None
    uncached_prompt_tokens: int | None = None
    request_token_estimate: RequestTokenBreakdown | None = None
    raw_api_prompt_tokens: int | None = None
    raw_api_completion_tokens: int | None = None
    raw_api_total_tokens: int | None = None
    usage_correction_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_type": "llm_usage",
            "timestamp": self.timestamp,
            "role": self.role,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input_cost_per_token": self.input_cost_per_token,
            "output_cost_per_token": self.output_cost_per_token,
            "cost_usd": self.cost_usd,
            "usage_source": self.usage_source,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "uncached_prompt_tokens": self.uncached_prompt_tokens,
            "request_token_estimate": (
                self.request_token_estimate.to_payload()
                if self.request_token_estimate is not None
                else None
            ),
            "raw_api_prompt_tokens": self.raw_api_prompt_tokens,
            "raw_api_completion_tokens": self.raw_api_completion_tokens,
            "raw_api_total_tokens": self.raw_api_total_tokens,
            "usage_correction_reason": self.usage_correction_reason,
        }


@dataclass(frozen=True)
class ContextLeft:
    model_name: str
    max_input_tokens: int | None
    used_input_tokens: int
    remaining_tokens: int | None
    percent_left: float | None
    source: str
    context_window_tokens: int | None = None
    context_window_remaining_tokens: int | None = None
    context_window_percent_left: float | None = None
    effective_input_budget: int | None = None
    effective_remaining_tokens: int | None = None
    effective_percent_left: float | None = None
    startup_baseline_tokens: int = 0
    dynamic_context_budget_tokens: int | None = None
    dynamic_context_used_tokens: int = 0
    dynamic_context_remaining_tokens: int | None = None
    dynamic_context_percent_left: float | None = None


@dataclass
class _ModelUsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    known_cost_usd: float = 0.0
    known_cost_calls: int = 0
    unknown_cost_count: int = 0
    api_usage_calls: int = 0
    estimate_usage_calls: int = 0
    cached_prompt_tokens: int = 0
    uncached_prompt_tokens: int = 0
    estimated_bootstrap_prompt_tokens: int = 0
    estimated_tool_schema_tokens: int = 0
    estimated_live_conversation_history_tokens: int = 0
    estimated_inline_tool_transcript_tokens: int = 0
    estimated_memory_summary_tokens: int = 0
    estimated_pins_tokens: int = 0
    estimated_total_request_tokens: int = 0
    corrected_usage_calls: int = 0


def _messages_without_provider_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        strip_provider_metadata_from_message(message)
        for message in messages
        if isinstance(message, dict)
    ]


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return 0
    serialized = json.dumps(
        _messages_without_provider_metadata(messages),
        ensure_ascii=False,
        sort_keys=True,
    )
    return estimate_tokens(serialized)


def _completion_payload_for_estimation(content: str, tool_calls: list[Any]) -> str:
    if content.strip():
        base = content
    else:
        base = ""
    if tool_calls:
        base += "\n" + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    return base


def estimate_completion_tokens(content: str, tool_calls: list[Any]) -> int:
    base = _completion_payload_for_estimation(content, tool_calls)
    if not base.strip():
        return 0
    return estimate_tokens(base)


def _prompt_character_count(
    messages: list[dict[str, Any]],
    tool_list: list[dict[str, Any]] | None,
) -> int:
    payload: dict[str, Any] = {"messages": _messages_without_provider_metadata(messages)}
    if tool_list:
        payload["tools"] = tool_list
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _looks_like_character_count(
    *,
    api_count: int | None,
    estimated_tokens: int,
    character_count: int,
) -> bool:
    if api_count is None or api_count <= 0 or estimated_tokens <= 0:
        return False
    if character_count <= 0:
        return api_count >= max(estimated_tokens * 3, estimated_tokens + 256)

    # Some providers put character counts in token fields.
    # Trust plausible API token counts, but replace counts that are much closer
    # to raw text size than to the local tokenizer estimate.
    minimum_delta = 64 if character_count < 2048 else 256
    far_above_tokens = api_count >= max(estimated_tokens * 2.75, estimated_tokens + minimum_delta)
    close_to_chars = abs(api_count - character_count) <= max(16, int(character_count * 0.15))
    return far_above_tokens and close_to_chars


def _safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _resolve_total_tokens(
    *,
    api_total_tokens: int | None,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_character_count: int,
    completion_character_count: int,
) -> tuple[int, str | None]:
    expected_total = prompt_tokens + completion_tokens
    if api_total_tokens is None:
        return expected_total, None
    if api_total_tokens < expected_total:
        return expected_total, "api_usage_total_below_component_sum"
    if _looks_like_character_count(
        api_count=api_total_tokens,
        estimated_tokens=expected_total,
        character_count=prompt_character_count + completion_character_count,
    ):
        return expected_total, "api_usage_looked_like_character_counts:total_tokens"
    return api_total_tokens, None


def build_usage_record(
    *,
    role: str,
    requested_model: str,
    response_model: str | None,
    messages: list[dict[str, Any]],
    response_content: str,
    response_tool_calls: list[Any],
    api_prompt_tokens: int | None,
    api_completion_tokens: int | None,
    api_total_tokens: int | None,
    registry: ModelRegistry,
    api_cached_prompt_tokens: int | None = None,
    tool_list: list[dict[str, Any]] | None = None,
    pinned_prefix_len: int = 0,
) -> UsageRecord:
    request_token_estimate = estimate_request_token_breakdown(
        messages=messages,
        tool_list=tool_list,
        pinned_prefix_len=pinned_prefix_len,
    )
    estimated_prompt_tokens = request_token_estimate.total_tokens or estimate_prompt_tokens(
        messages
    )
    estimated_completion_tokens = estimate_completion_tokens(response_content, response_tool_calls)
    prompt_character_count = _prompt_character_count(messages, tool_list)
    completion_character_count = len(
        _completion_payload_for_estimation(response_content, response_tool_calls)
    )

    raw_api_prompt_tokens = _safe_int(api_prompt_tokens)
    raw_api_completion_tokens = _safe_int(api_completion_tokens)
    raw_api_total_tokens = _safe_int(api_total_tokens)
    prompt_tokens = raw_api_prompt_tokens
    completion_tokens = raw_api_completion_tokens
    total_tokens = raw_api_total_tokens
    usage_source = "api"
    corrected_fields: list[str] = []

    if prompt_tokens is None:
        prompt_tokens = estimated_prompt_tokens
        usage_source = "estimate"
    elif _looks_like_character_count(
        api_count=prompt_tokens,
        estimated_tokens=estimated_prompt_tokens,
        character_count=prompt_character_count,
    ):
        prompt_tokens = estimated_prompt_tokens
        usage_source = "estimate"
        corrected_fields.append("prompt_tokens")
    if completion_tokens is None:
        completion_tokens = estimated_completion_tokens
        usage_source = "estimate"
    elif _looks_like_character_count(
        api_count=completion_tokens,
        estimated_tokens=estimated_completion_tokens,
        character_count=completion_character_count,
    ):
        completion_tokens = estimated_completion_tokens
        usage_source = "estimate"
        corrected_fields.append("completion_tokens")
    total_correction_reason: str | None = None
    if total_tokens is None or usage_source == "estimate":
        total_tokens = prompt_tokens + completion_tokens
        if usage_source != "estimate":
            usage_source = "api"
    else:
        total_tokens, total_correction_reason = _resolve_total_tokens(
            api_total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_character_count=prompt_character_count,
            completion_character_count=completion_character_count,
        )
        if total_correction_reason is not None:
            corrected_fields.append("total_tokens")
    cached_prompt_tokens = _safe_int(api_cached_prompt_tokens)
    if "prompt_tokens" in corrected_fields:
        cached_prompt_tokens = None
    uncached_prompt_tokens: int | None = None
    if prompt_tokens is not None and cached_prompt_tokens is not None:
        uncached_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)
    correction_reasons: list[str] = []
    character_count_corrected_fields = [
        field for field in corrected_fields if field in {"prompt_tokens", "completion_tokens"}
    ]
    if character_count_corrected_fields:
        correction_reasons.append(
            "api_usage_looked_like_character_counts:" + ",".join(character_count_corrected_fields)
        )
    if total_correction_reason is not None:
        correction_reasons.append(total_correction_reason)
    usage_correction_reason = ";".join(correction_reasons) if correction_reasons else None
    requested_meta = registry.get(requested_model)
    response_meta: ModelMeta | None = None
    if response_model:
        response_meta = registry.get(response_model)

    input_rate = requested_meta.input_cost_per_token
    output_rate = requested_meta.output_cost_per_token
    if input_rate is None and response_meta is not None:
        input_rate = response_meta.input_cost_per_token
    if output_rate is None and response_meta is not None:
        output_rate = response_meta.output_cost_per_token

    cost_usd: float | None = None
    if input_rate is not None and output_rate is not None:
        cost_usd = (prompt_tokens * input_rate) + (completion_tokens * output_rate)

    return UsageRecord(
        timestamp=now_iso(),
        role=role,
        requested_model=requested_model,
        response_model=response_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
        cost_usd=cost_usd,
        usage_source=usage_source,
        cached_prompt_tokens=cached_prompt_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
        request_token_estimate=request_token_estimate,
        raw_api_prompt_tokens=raw_api_prompt_tokens if corrected_fields else None,
        raw_api_completion_tokens=raw_api_completion_tokens if corrected_fields else None,
        raw_api_total_tokens=raw_api_total_tokens if corrected_fields else None,
        usage_correction_reason=usage_correction_reason,
    )


class UsageSummary:
    def __init__(self) -> None:
        self._by_model: dict[str, _ModelUsageTotals] = {}
        self._records: list[UsageRecord] = []
        self.calls = 0

    def add_record(self, record: UsageRecord) -> None:
        key = record.requested_model.strip() or "unknown-model"
        totals = self._by_model.setdefault(key, _ModelUsageTotals())
        self._records.append(record)
        totals.prompt_tokens += record.prompt_tokens
        totals.completion_tokens += record.completion_tokens
        totals.total_tokens += record.total_tokens
        if record.cost_usd is None:
            totals.unknown_cost_count += 1
        else:
            totals.known_cost_usd += float(record.cost_usd)
            totals.known_cost_calls += 1
        if record.usage_source == "api":
            totals.api_usage_calls += 1
        else:
            totals.estimate_usage_calls += 1
        if record.usage_correction_reason:
            totals.corrected_usage_calls += 1
        totals.cached_prompt_tokens += max(0, record.cached_prompt_tokens or 0)
        totals.uncached_prompt_tokens += max(0, record.uncached_prompt_tokens or 0)
        breakdown = record.request_token_estimate
        if breakdown is not None:
            totals.estimated_bootstrap_prompt_tokens += breakdown.bootstrap_prompt_tokens
            totals.estimated_tool_schema_tokens += breakdown.tool_schema_tokens
            totals.estimated_live_conversation_history_tokens += (
                breakdown.live_conversation_history_tokens
            )
            totals.estimated_inline_tool_transcript_tokens += (
                breakdown.inline_tool_transcript_tokens
            )
            totals.estimated_memory_summary_tokens += breakdown.memory_summary_tokens
            totals.estimated_pins_tokens += breakdown.pins_tokens
            totals.estimated_total_request_tokens += breakdown.total_tokens
        self.calls += 1

    def records(self) -> list[UsageRecord]:
        return list(self._records)

    def merge_records(self, records: Iterable[UsageRecord]) -> int:
        merged = 0
        for record in records:
            self.add_record(record)
            merged += 1
        return merged

    def add_event_payload(self, payload: dict[str, Any]) -> None:
        if str(payload.get("event_type") or "") != "llm_usage":
            return
        record = UsageRecord(
            timestamp=str(payload.get("timestamp") or ""),
            role=str(payload.get("role") or "main"),
            requested_model=str(payload.get("requested_model") or "unknown-model"),
            response_model=(
                str(payload.get("response_model"))
                if payload.get("response_model") is not None
                else None
            ),
            prompt_tokens=max(0, _safe_int(payload.get("prompt_tokens")) or 0),
            completion_tokens=max(0, _safe_int(payload.get("completion_tokens")) or 0),
            total_tokens=max(0, _safe_int(payload.get("total_tokens")) or 0),
            input_cost_per_token=(
                _safe_float(payload.get("input_cost_per_token"))
                if payload.get("input_cost_per_token") is not None
                else None
            ),
            output_cost_per_token=(
                _safe_float(payload.get("output_cost_per_token"))
                if payload.get("output_cost_per_token") is not None
                else None
            ),
            cost_usd=(
                _safe_float(payload.get("cost_usd"))
                if payload.get("cost_usd") is not None
                else None
            ),
            usage_source=str(payload.get("usage_source") or "estimate"),
            cached_prompt_tokens=_safe_int(payload.get("cached_prompt_tokens")),
            uncached_prompt_tokens=_safe_int(payload.get("uncached_prompt_tokens")),
            request_token_estimate=RequestTokenBreakdown.from_payload(
                payload.get("request_token_estimate")
            ),
            raw_api_prompt_tokens=_safe_int(payload.get("raw_api_prompt_tokens")),
            raw_api_completion_tokens=_safe_int(payload.get("raw_api_completion_tokens")),
            raw_api_total_tokens=_safe_int(payload.get("raw_api_total_tokens")),
            usage_correction_reason=(
                str(payload.get("usage_correction_reason"))
                if payload.get("usage_correction_reason") is not None
                else None
            ),
        )
        self.add_record(record)

    def by_model_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for model_name in sorted(self._by_model):
            totals = self._by_model[model_name]
            rows.append(
                {
                    "model": model_name,
                    "prompt_tokens": totals.prompt_tokens,
                    "completion_tokens": totals.completion_tokens,
                    "total_tokens": totals.total_tokens,
                    "cost_usd": (totals.known_cost_usd if totals.known_cost_calls > 0 else None),
                    "known_cost_calls": totals.known_cost_calls,
                    "unknown_cost_count": totals.unknown_cost_count,
                    "api_usage_calls": totals.api_usage_calls,
                    "estimate_usage_calls": totals.estimate_usage_calls,
                    "corrected_usage_calls": totals.corrected_usage_calls,
                    "cached_prompt_tokens": totals.cached_prompt_tokens,
                    "uncached_prompt_tokens": totals.uncached_prompt_tokens,
                    "request_token_estimate": {
                        "bootstrap_prompt_tokens": totals.estimated_bootstrap_prompt_tokens,
                        "tool_schema_tokens": totals.estimated_tool_schema_tokens,
                        "live_conversation_history_tokens": (
                            totals.estimated_live_conversation_history_tokens
                        ),
                        "inline_tool_transcript_tokens": (
                            totals.estimated_inline_tool_transcript_tokens
                        ),
                        "memory_summary_tokens": totals.estimated_memory_summary_tokens,
                        "pins_tokens": totals.estimated_pins_tokens,
                        "total_tokens": totals.estimated_total_request_tokens,
                    },
                }
            )
        return rows

    def totals(self) -> dict[str, Any]:
        prompt = sum(v.prompt_tokens for v in self._by_model.values())
        completion = sum(v.completion_tokens for v in self._by_model.values())
        total = sum(v.total_tokens for v in self._by_model.values())
        known_cost = sum(v.known_cost_usd for v in self._by_model.values())
        known_cost_calls = sum(v.known_cost_calls for v in self._by_model.values())
        unknown_cost_calls = sum(v.unknown_cost_count for v in self._by_model.values())
        api_usage_calls = sum(v.api_usage_calls for v in self._by_model.values())
        estimate_usage_calls = sum(v.estimate_usage_calls for v in self._by_model.values())
        corrected_usage_calls = sum(v.corrected_usage_calls for v in self._by_model.values())
        cached_prompt_tokens = sum(v.cached_prompt_tokens for v in self._by_model.values())
        uncached_prompt_tokens = sum(v.uncached_prompt_tokens for v in self._by_model.values())
        estimated_bootstrap_prompt_tokens = sum(
            v.estimated_bootstrap_prompt_tokens for v in self._by_model.values()
        )
        estimated_tool_schema_tokens = sum(
            v.estimated_tool_schema_tokens for v in self._by_model.values()
        )
        estimated_live_conversation_history_tokens = sum(
            v.estimated_live_conversation_history_tokens for v in self._by_model.values()
        )
        estimated_inline_tool_transcript_tokens = sum(
            v.estimated_inline_tool_transcript_tokens for v in self._by_model.values()
        )
        estimated_memory_summary_tokens = sum(
            v.estimated_memory_summary_tokens for v in self._by_model.values()
        )
        estimated_pins_tokens = sum(v.estimated_pins_tokens for v in self._by_model.values())
        estimated_total_request_tokens = sum(
            v.estimated_total_request_tokens for v in self._by_model.values()
        )
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cost_usd": known_cost if known_cost_calls > 0 else None,
            "known_cost_usd": known_cost,
            "known_cost_calls": known_cost_calls,
            "unknown_cost_calls": unknown_cost_calls,
            "api_usage_calls": api_usage_calls,
            "estimate_usage_calls": estimate_usage_calls,
            "corrected_usage_calls": corrected_usage_calls,
            "cached_prompt_tokens": cached_prompt_tokens,
            "uncached_prompt_tokens": uncached_prompt_tokens,
            "request_token_estimate": {
                "bootstrap_prompt_tokens": estimated_bootstrap_prompt_tokens,
                "tool_schema_tokens": estimated_tool_schema_tokens,
                "live_conversation_history_tokens": (estimated_live_conversation_history_tokens),
                "inline_tool_transcript_tokens": estimated_inline_tool_transcript_tokens,
                "memory_summary_tokens": estimated_memory_summary_tokens,
                "pins_tokens": estimated_pins_tokens,
                "total_tokens": estimated_total_request_tokens,
            },
            "calls": self.calls,
        }


def aggregate_usage_from_session_logs(paths: list[Path]) -> UsageSummary:
    summary = UsageSummary()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        for event in read_session_events(path):
            if str(event.get("type") or "") != "llm_usage":
                continue
            payload = event.get("payload")
            if isinstance(payload, dict):
                summary.add_event_payload(payload)
    return summary


def compute_context_left(
    *,
    messages: list[dict[str, Any]],
    model_name: str,
    registry: ModelRegistry,
    tool_list: list[dict[str, Any]] | None = None,
    pinned_prefix_len: int = 0,
    safety_margin_tokens: int = 512,
    startup_baseline_tokens: int = 0,
) -> ContextLeft:
    meta = registry.get(model_name)
    context_window = meta.context_window_tokens
    effective_input_budget = compute_input_budget(meta, safety_margin=safety_margin_tokens)
    baseline = max(0, int(startup_baseline_tokens or 0))
    request_estimate = estimate_request_token_breakdown(
        messages=messages,
        tool_list=tool_list,
        pinned_prefix_len=pinned_prefix_len,
    )
    used = request_estimate.total_tokens
    context_remaining = max(0, context_window - used) if context_window > 0 else None
    context_percent = (
        (context_remaining / context_window) * 100.0
        if (context_remaining is not None and context_window > 0)
        else None
    )
    effective_remaining = (
        max(0, effective_input_budget - used) if effective_input_budget > 0 else None
    )
    effective_percent = (
        (effective_remaining / effective_input_budget) * 100.0
        if (effective_remaining is not None and effective_input_budget > 0)
        else None
    )
    dynamic_budget = max(0, effective_input_budget - baseline)
    dynamic_used = max(0, used - baseline)
    dynamic_remaining = max(0, dynamic_budget - dynamic_used)
    dynamic_percent = (dynamic_remaining / dynamic_budget) * 100.0 if dynamic_budget > 0 else 0.0
    return ContextLeft(
        model_name=model_name,
        max_input_tokens=context_window,
        used_input_tokens=used,
        remaining_tokens=context_remaining,
        percent_left=context_percent,
        source=meta.source,
        context_window_tokens=context_window,
        context_window_remaining_tokens=context_remaining,
        context_window_percent_left=context_percent,
        effective_input_budget=effective_input_budget,
        effective_remaining_tokens=effective_remaining,
        effective_percent_left=effective_percent,
        startup_baseline_tokens=baseline,
        dynamic_context_budget_tokens=dynamic_budget,
        dynamic_context_used_tokens=dynamic_used,
        dynamic_context_remaining_tokens=dynamic_remaining,
        dynamic_context_percent_left=dynamic_percent,
    )


def format_usd(cost: float | None, style: str = "table") -> str:
    if cost is None:
        return "n/a"
    decimals = 3 if style == "hud" else 4
    return f"${cost:.{decimals}f}"


def format_cost(cost: float | None) -> str:
    # Backward-compatible alias for existing call sites.
    return format_usd(cost, style="table")


def format_context_percent(ctx: ContextLeft) -> str:
    if ctx.percent_left is None:
        return "n/a"
    return f"{ctx.percent_left:.1f}%"
