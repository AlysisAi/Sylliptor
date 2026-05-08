from __future__ import annotations

import json
from pathlib import Path

from sylliptor_agent_cli.litellm_static_provider import BUNDLED_MODEL_CATALOG_SOURCE
from sylliptor_agent_cli.model_registry import ModelMeta
from sylliptor_agent_cli.request_estimation import estimate_request_token_breakdown
from sylliptor_agent_cli.token_budget import compute_input_budget
from sylliptor_agent_cli.usage_tracker import (
    UsageRecord,
    UsageSummary,
    _looks_like_character_count,
    aggregate_usage_from_session_logs,
    build_usage_record,
    compute_context_left,
    estimate_completion_tokens,
    estimate_prompt_tokens,
    format_usd,
)


class _FakeRegistry:
    def __init__(self, by_model: dict[str, ModelMeta]) -> None:
        self._by_model = by_model

    def get(self, model_name: str) -> ModelMeta:
        return self._by_model[model_name]


def test_build_usage_record_computes_cost_from_registry_rates() -> None:
    registry = _FakeRegistry(
        {
            "gpt-5-nano": ModelMeta(
                model_name="gpt-5-nano",
                context_window_tokens=200000,
                max_output_tokens=8192,
                input_cost_per_token=0.1,
                output_cost_per_token=0.2,
                raw_metadata={},
                source=BUNDLED_MODEL_CATALOG_SOURCE,
            )
        }
    )
    record = build_usage_record(
        role="main",
        requested_model="gpt-5-nano",
        response_model="gpt-5-nano",
        messages=[{"role": "user", "content": "hello"}],
        response_content="world",
        response_tool_calls=[],
        api_prompt_tokens=10,
        api_completion_tokens=5,
        api_total_tokens=15,
        registry=registry,  # type: ignore[arg-type]
    )
    assert record.usage_source == "api"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 5
    assert record.total_tokens == 15
    assert record.cost_usd == 2.0


def test_build_usage_record_replaces_character_like_api_usage_with_token_estimates() -> None:
    registry = _FakeRegistry(
        {
            "qwen3.5-plus": ModelMeta(
                model_name="qwen3.5-plus",
                context_window_tokens=1000000,
                max_output_tokens=8192,
                input_cost_per_token=None,
                output_cost_per_token=None,
                raw_metadata={},
                source=BUNDLED_MODEL_CATALOG_SOURCE,
            )
        }
    )
    messages = [
        {"role": "system", "content": "You are a coding agent.\n" * 200},
        {"role": "user", "content": "hi"},
    ]
    response_content = (
        "I see you have an interesting repository with two projects. "
        "Let me know what you'd like to work on.\n"
        "1 AI News Hub - Web app for AI news.\n"
        "2 Asteroids Game - Pygame implementation.\n"
    )
    api_prompt_char_count = len(
        json.dumps({"messages": messages}, ensure_ascii=False, sort_keys=True)
    )
    api_completion_char_count = len(response_content)

    record = build_usage_record(
        role="main",
        requested_model="qwen3.5-plus",
        response_model="qwen3.5-plus",
        messages=messages,
        response_content=response_content,
        response_tool_calls=[],
        api_prompt_tokens=api_prompt_char_count,
        api_completion_tokens=api_completion_char_count,
        api_total_tokens=api_prompt_char_count + api_completion_char_count,
        api_cached_prompt_tokens=api_prompt_char_count // 2,
        registry=registry,  # type: ignore[arg-type]
    )

    assert record.usage_source == "estimate"
    assert record.prompt_tokens == estimate_prompt_tokens(messages)
    assert record.completion_tokens == estimate_completion_tokens(response_content, [])
    assert record.total_tokens == record.prompt_tokens + record.completion_tokens
    assert record.prompt_tokens < api_prompt_char_count
    assert record.completion_tokens < api_completion_char_count
    assert record.cached_prompt_tokens is None
    assert record.uncached_prompt_tokens is None
    assert record.raw_api_prompt_tokens == api_prompt_char_count
    assert record.raw_api_completion_tokens == api_completion_char_count
    assert record.raw_api_total_tokens == api_prompt_char_count + api_completion_char_count
    assert record.usage_correction_reason == (
        "api_usage_looked_like_character_counts:prompt_tokens,completion_tokens"
    )

    payload = record.to_payload()
    assert payload["raw_api_prompt_tokens"] == api_prompt_char_count
    assert payload["usage_correction_reason"] == record.usage_correction_reason

    summary = UsageSummary()
    summary.add_event_payload(payload)
    assert summary.records()[0].raw_api_prompt_tokens == api_prompt_char_count
    assert summary.totals()["corrected_usage_calls"] == 1
    assert summary.totals()["estimate_usage_calls"] == 1


def test_build_usage_record_replaces_inconsistent_total_with_component_sum() -> None:
    registry = _FakeRegistry(
        {
            "gpt-5-nano": ModelMeta(
                model_name="gpt-5-nano",
                context_window_tokens=200000,
                max_output_tokens=8192,
                input_cost_per_token=0.1,
                output_cost_per_token=0.2,
                raw_metadata={},
                source=BUNDLED_MODEL_CATALOG_SOURCE,
            )
        }
    )

    record = build_usage_record(
        role="main",
        requested_model="gpt-5-nano",
        response_model="gpt-5-nano",
        messages=[{"role": "user", "content": "hello"}],
        response_content="world",
        response_tool_calls=[],
        api_prompt_tokens=10,
        api_completion_tokens=5,
        api_total_tokens=8,
        registry=registry,  # type: ignore[arg-type]
    )

    assert record.usage_source == "api"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 5
    assert record.total_tokens == 15
    assert record.raw_api_total_tokens == 8
    assert record.usage_correction_reason == "api_usage_total_below_component_sum"

    summary = UsageSummary()
    summary.add_event_payload(record.to_payload())
    assert summary.totals()["corrected_usage_calls"] == 1
    assert summary.totals()["total_tokens"] == 15


def test_build_usage_record_replaces_character_like_total_with_component_sum() -> None:
    registry = _FakeRegistry(
        {
            "qwen3.5-plus": ModelMeta(
                model_name="qwen3.5-plus",
                context_window_tokens=1000000,
                max_output_tokens=8192,
                input_cost_per_token=None,
                output_cost_per_token=None,
                raw_metadata={},
                source=BUNDLED_MODEL_CATALOG_SOURCE,
            )
        }
    )
    messages = [
        {"role": "system", "content": "You are a coding agent.\n" * 200},
        {"role": "user", "content": "hi"},
    ]
    response_content = "Short answer."
    api_total_char_count = len(
        json.dumps({"messages": messages}, ensure_ascii=False, sort_keys=True)
    ) + len(response_content)

    record = build_usage_record(
        role="main",
        requested_model="qwen3.5-plus",
        response_model="qwen3.5-plus",
        messages=messages,
        response_content=response_content,
        response_tool_calls=[],
        api_prompt_tokens=estimate_prompt_tokens(messages),
        api_completion_tokens=estimate_completion_tokens(response_content, []),
        api_total_tokens=api_total_char_count,
        registry=registry,  # type: ignore[arg-type]
    )

    assert record.usage_source == "api"
    assert record.total_tokens == record.prompt_tokens + record.completion_tokens
    assert record.raw_api_total_tokens == api_total_char_count
    assert record.usage_correction_reason == "api_usage_looked_like_character_counts:total_tokens"


def test_character_count_detection_does_not_replace_plausible_api_token_counts() -> None:
    assert not _looks_like_character_count(
        api_count=156,
        estimated_tokens=148,
        character_count=160,
    )
    assert _looks_like_character_count(
        api_count=5000,
        estimated_tokens=1200,
        character_count=5080,
    )


def test_build_usage_record_captures_cached_prompt_and_request_breakdown() -> None:
    registry = _FakeRegistry(
        {
            "gpt-5-nano": ModelMeta(
                model_name="gpt-5-nano",
                context_window_tokens=200000,
                max_output_tokens=8192,
                input_cost_per_token=0.1,
                output_cost_per_token=0.2,
                raw_metadata={},
                source=BUNDLED_MODEL_CATALOG_SOURCE,
            )
        }
    )
    messages = [
        {"role": "system", "content": "Core prompt"},
        {"role": "system", "content": "Repo prompt"},
        {"role": "user", "content": "Investigate the failing test."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "fs_read", "arguments": {"path": "app.py"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"path":"app.py","content":"print(1)"}',
        },
        {
            "role": "user",
            "content": '<<<SYLLIPTOR_CONVERSATION_MEMORY_JSON>>>\n{"summary":"keep this"}',
        },
        {
            "role": "user",
            "content": '<<<SYLLIPTOR_CONVERSATION_PINS_JSON>>>\n[{"path":"tests/test_app.py"}]',
        },
    ]
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": "fs_read",
                "description": "Read file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]

    record = build_usage_record(
        role="main",
        requested_model="gpt-5-nano",
        response_model="gpt-5-nano",
        messages=messages,
        response_content="Patched the issue.",
        response_tool_calls=[],
        api_prompt_tokens=120,
        api_completion_tokens=12,
        api_total_tokens=132,
        api_cached_prompt_tokens=48,
        tool_list=tool_list,
        pinned_prefix_len=2,
        registry=registry,  # type: ignore[arg-type]
    )

    assert record.cached_prompt_tokens == 48
    assert record.uncached_prompt_tokens == 72
    assert record.request_token_estimate is not None
    assert record.request_token_estimate.bootstrap_prompt_tokens > 0
    assert record.request_token_estimate.tool_schema_tokens > 0
    assert record.request_token_estimate.live_conversation_history_tokens > 0
    assert record.request_token_estimate.inline_tool_transcript_tokens > 0
    assert record.request_token_estimate.memory_summary_tokens > 0
    assert record.request_token_estimate.pins_tokens > 0


def test_usage_summary_aggregates_multi_model_and_unknown_cost() -> None:
    summary = UsageSummary()
    summary.add_event_payload(
        {
            "event_type": "llm_usage",
            "timestamp": "2026-02-24T00:00:00Z",
            "role": "main",
            "requested_model": "model-a",
            "response_model": "model-a",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "cost_usd": 0.2,
            "usage_source": "api",
        }
    )
    summary.add_event_payload(
        {
            "event_type": "llm_usage",
            "timestamp": "2026-02-24T00:00:01Z",
            "role": "main",
            "requested_model": "model-b",
            "response_model": "model-b",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "input_cost_per_token": None,
            "output_cost_per_token": None,
            "cost_usd": None,
            "usage_source": "estimate",
        }
    )
    rows = summary.by_model_rows()
    assert len(rows) == 2
    totals = summary.totals()
    assert totals["prompt_tokens"] == 110
    assert totals["completion_tokens"] == 55
    assert totals["total_tokens"] == 165
    assert totals["cost_usd"] == 0.2
    assert totals["known_cost_usd"] == 0.2
    assert totals["known_cost_calls"] == 1
    assert totals["unknown_cost_calls"] == 1
    assert totals["api_usage_calls"] == 1
    assert totals["estimate_usage_calls"] == 1


def test_usage_summary_round_trips_cached_prompt_and_request_breakdown() -> None:
    summary = UsageSummary()
    summary.add_event_payload(
        {
            "event_type": "llm_usage",
            "timestamp": "2026-02-24T00:00:00Z",
            "role": "main",
            "requested_model": "model-a",
            "response_model": "model-a",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cached_prompt_tokens": 60,
            "uncached_prompt_tokens": 40,
            "request_token_estimate": {
                "bootstrap_prompt_tokens": 25,
                "tool_schema_tokens": 15,
                "live_conversation_history_tokens": 30,
                "inline_tool_transcript_tokens": 20,
                "memory_summary_tokens": 5,
                "pins_tokens": 3,
                "total_tokens": 98,
            },
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "cost_usd": 0.2,
            "usage_source": "api",
        }
    )

    totals = summary.totals()
    assert totals["cached_prompt_tokens"] == 60
    assert totals["uncached_prompt_tokens"] == 40
    assert totals["request_token_estimate"]["bootstrap_prompt_tokens"] == 25
    assert totals["request_token_estimate"]["tool_schema_tokens"] == 15
    assert totals["request_token_estimate"]["total_tokens"] == 98


def test_usage_summary_exposes_and_merges_raw_records() -> None:
    summary = UsageSummary()
    records = [
        UsageRecord(
            timestamp="2026-03-09T00:00:00Z",
            role="main",
            requested_model="model-a",
            response_model="model-a",
            prompt_tokens=9,
            completion_tokens=4,
            total_tokens=13,
            input_cost_per_token=0.1,
            output_cost_per_token=0.2,
            cost_usd=1.7,
            usage_source="api",
        ),
        UsageRecord(
            timestamp="2026-03-09T00:00:01Z",
            role="main:subagent:reviewer",
            requested_model="model-b",
            response_model="model-b",
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            input_cost_per_token=None,
            output_cost_per_token=None,
            cost_usd=None,
            usage_source="estimate",
        ),
    ]

    merged = summary.merge_records(records)

    assert merged == 2
    assert summary.records() == records
    totals = summary.totals()
    assert totals["prompt_tokens"] == 14
    assert totals["completion_tokens"] == 6
    assert totals["total_tokens"] == 20
    assert totals["calls"] == 2
    assert totals["api_usage_calls"] == 1
    assert totals["estimate_usage_calls"] == 1


def test_compute_context_left_returns_percent() -> None:
    meta = ModelMeta(
        model_name="gpt-5-nano",
        context_window_tokens=1000,
        max_output_tokens=256,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        source="fallback",
    )
    registry = _FakeRegistry(
        {
            "gpt-5-nano": meta,
        }
    )
    ctx = compute_context_left(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain this module."},
        ],
        model_name="gpt-5-nano",
        registry=registry,  # type: ignore[arg-type]
        safety_margin_tokens=100,
    )
    assert ctx.max_input_tokens == meta.context_window_tokens
    assert ctx.context_window_tokens == meta.context_window_tokens
    assert ctx.effective_input_budget == compute_input_budget(meta, safety_margin=100)
    assert ctx.used_input_tokens > 0
    assert ctx.remaining_tokens is not None
    assert ctx.remaining_tokens < ctx.max_input_tokens
    assert ctx.percent_left is not None
    assert 0.0 <= ctx.percent_left <= 100.0


def test_compute_context_left_uses_effective_input_budget() -> None:
    meta = ModelMeta(
        model_name="tiny-budget",
        context_window_tokens=1000,
        max_output_tokens=200,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        source="test",
    )
    registry = _FakeRegistry({"tiny-budget": meta})
    messages = [{"role": "user", "content": "word " * 650}]

    ctx = compute_context_left(
        messages=messages,
        model_name="tiny-budget",
        registry=registry,  # type: ignore[arg-type]
        safety_margin_tokens=100,
    )

    assert ctx.max_input_tokens == meta.context_window_tokens
    assert ctx.percent_left is not None
    assert ctx.percent_left > 25.0
    assert ctx.effective_input_budget == compute_input_budget(meta, safety_margin=100)
    assert ctx.effective_percent_left is not None
    assert ctx.effective_percent_left < 10.0


def test_compute_context_left_starts_dynamic_context_at_100_percent() -> None:
    meta = ModelMeta(
        model_name="fresh-chat",
        context_window_tokens=5000,
        max_output_tokens=500,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        source="test",
    )
    registry = _FakeRegistry({"fresh-chat": meta})
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "<environment_context>\nrepo\n</environment_context>"},
    ]
    tool_list = [{"type": "function", "function": {"name": "read_file"}}]
    baseline = estimate_request_token_breakdown(
        messages=messages,
        tool_list=tool_list,
        pinned_prefix_len=len(messages),
    ).total_tokens

    ctx = compute_context_left(
        messages=messages,
        model_name="fresh-chat",
        registry=registry,  # type: ignore[arg-type]
        tool_list=tool_list,
        pinned_prefix_len=len(messages),
        safety_margin_tokens=100,
        startup_baseline_tokens=baseline,
    )

    assert ctx.used_input_tokens == baseline
    assert ctx.startup_baseline_tokens == baseline
    assert ctx.dynamic_context_used_tokens == 0
    assert ctx.dynamic_context_budget_tokens == ctx.effective_input_budget - baseline
    assert ctx.dynamic_context_percent_left == 100.0


def test_compute_context_left_dynamic_context_tracks_compacted_active_request() -> None:
    meta = ModelMeta(
        model_name="compacted-chat",
        context_window_tokens=6000,
        max_output_tokens=500,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        source="test",
    )
    registry = _FakeRegistry({"compacted-chat": meta})
    baseline_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "<environment_context>\nrepo\n</environment_context>"},
    ]
    baseline = estimate_request_token_breakdown(
        messages=baseline_messages,
        tool_list=None,
        pinned_prefix_len=len(baseline_messages),
    ).total_tokens
    before_compaction = [
        *baseline_messages,
        {"role": "user", "content": "word " * 1200},
        {"role": "assistant", "content": "answer " * 700},
    ]
    after_compaction = [
        *baseline_messages,
        {
            "role": "system",
            "content": "Compacted conversation summary: " + ("summary " * 80),
        },
    ]

    before = compute_context_left(
        messages=before_compaction,
        model_name="compacted-chat",
        registry=registry,  # type: ignore[arg-type]
        pinned_prefix_len=len(baseline_messages),
        safety_margin_tokens=100,
        startup_baseline_tokens=baseline,
    )
    after = compute_context_left(
        messages=after_compaction,
        model_name="compacted-chat",
        registry=registry,  # type: ignore[arg-type]
        pinned_prefix_len=len(baseline_messages),
        safety_margin_tokens=100,
        startup_baseline_tokens=baseline,
    )

    assert before.dynamic_context_used_tokens > after.dynamic_context_used_tokens
    assert after.dynamic_context_used_tokens > 0
    assert before.dynamic_context_percent_left is not None
    assert after.dynamic_context_percent_left is not None
    assert after.dynamic_context_percent_left > before.dynamic_context_percent_left
    assert after.dynamic_context_percent_left < 100.0


def test_compute_context_left_includes_tool_schema_tokens() -> None:
    meta = ModelMeta(
        model_name="tool-heavy",
        context_window_tokens=1000,
        max_output_tokens=200,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        source="test",
    )
    registry = _FakeRegistry({"tool-heavy": meta})
    messages = [{"role": "user", "content": "hello"}]
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": "big_tool",
                "description": "large schema " * 200,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "string",
                            "description": "x " * 1000,
                        },
                    },
                },
            },
        }
    ]

    ctx = compute_context_left(
        messages=messages,
        model_name="tool-heavy",
        registry=registry,  # type: ignore[arg-type]
        tool_list=tool_list,
        safety_margin_tokens=100,
    )
    expected_used = estimate_request_token_breakdown(
        messages=messages,
        tool_list=tool_list,
    ).total_tokens

    assert ctx.used_input_tokens == expected_used
    assert ctx.effective_remaining_tokens == 0
    assert ctx.effective_percent_left == 0.0


def test_compute_context_left_sanitizes_image_urls() -> None:
    meta = ModelMeta(
        model_name="vision-model",
        context_window_tokens=1000,
        max_output_tokens=200,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        source="test",
    )
    registry = _FakeRegistry({"vision-model": meta})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + ("A" * 20_000)},
                },
            ],
        }
    ]

    ctx = compute_context_left(
        messages=messages,
        model_name="vision-model",
        registry=registry,  # type: ignore[arg-type]
        safety_margin_tokens=100,
    )
    expected_used = estimate_request_token_breakdown(
        messages=messages,
        tool_list=None,
    ).total_tokens

    assert estimate_prompt_tokens(messages) > 1000
    assert ctx.used_input_tokens == expected_used
    assert ctx.used_input_tokens < 200


def test_aggregate_usage_from_session_logs_reads_llm_usage_events(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    events = [
        {
            "type": "llm_usage",
            "payload": {
                "event_type": "llm_usage",
                "timestamp": "2026-02-24T00:00:00Z",
                "role": "main",
                "requested_model": "model-a",
                "response_model": "model-a",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 1.0,
                "cost_usd": 5.0,
                "usage_source": "api",
            },
        },
        {"type": "assistant_message", "payload": {"content": "ignored"}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    summary = aggregate_usage_from_session_logs([path])
    totals = summary.totals()
    assert totals["total_tokens"] == 5
    assert totals["cost_usd"] == 5.0


def test_format_usd_hud_rounding() -> None:
    assert format_usd(None, style="hud") == "n/a"
    assert format_usd(0.00004, style="hud") == "$0.000"
    assert format_usd(0.00149, style="hud") == "$0.001"
    assert format_usd(1.239, style="hud") == "$1.239"
