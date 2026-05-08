from __future__ import annotations

from sylliptor_agent_cli.request_estimation import estimate_request_token_breakdown


def test_request_token_breakdown_splits_bootstrap_history_tools_memory_and_pins() -> None:
    messages = [
        {"role": "system", "content": "Core coding rules."},
        {"role": "system", "content": "Repository context."},
        {"role": "user", "content": "Investigate failing parser tests."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "search_rg",
                    "arguments": {"pattern": "ParserError"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"matches":[{"path":"tests/test_parser.py","line":22,"text":"raise ParserError()"}]}',
        },
        {
            "role": "user",
            "content": '<<<SYLLIPTOR_CONVERSATION_MEMORY_JSON>>>\n{"summary":"parser bug affects nested forms"}',
        },
        {
            "role": "user",
            "content": '<<<SYLLIPTOR_CONVERSATION_PINS_JSON>>>\n[{"path":"src/parser.py"}]',
        },
        {"role": "assistant", "content": "The bug is in the nested expression branch."},
    ]
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": "search_rg",
                "description": "Search workspace",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                },
            },
        }
    ]

    breakdown = estimate_request_token_breakdown(
        messages=messages,
        tool_list=tool_list,
        pinned_prefix_len=2,
    )

    assert breakdown.bootstrap_prompt_tokens > 0
    assert breakdown.tool_schema_tokens > 0
    assert breakdown.live_conversation_history_tokens > 0
    assert breakdown.inline_tool_transcript_tokens > 0
    assert breakdown.memory_summary_tokens > 0
    assert breakdown.pins_tokens > 0
    assert breakdown.total_tokens == (
        breakdown.bootstrap_prompt_tokens
        + breakdown.tool_schema_tokens
        + breakdown.live_conversation_history_tokens
        + breakdown.inline_tool_transcript_tokens
        + breakdown.memory_summary_tokens
        + breakdown.pins_tokens
    )
