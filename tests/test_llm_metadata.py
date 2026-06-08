from __future__ import annotations

from sylliptor_agent_cli.llm import metadata, openai_compat
from sylliptor_agent_cli.llm.types import LLMResponse, ToolCall


def test_openai_compat_reexports_shared_provider_metadata_helpers() -> None:
    assert openai_compat.PROVIDER_METADATA_KEY == metadata.PROVIDER_METADATA_KEY
    assert (
        openai_compat.attach_provider_metadata_to_assistant_message
        is metadata.attach_provider_metadata_to_assistant_message
    )
    assert (
        openai_compat.strip_provider_metadata_from_message
        is metadata.strip_provider_metadata_from_message
    )
    assert openai_compat.assistant_message_from_response is metadata.assistant_message_from_response


def test_shared_metadata_helpers_attach_and_strip_provider_state() -> None:
    response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="fs_read",
                arguments={"path": "README.md"},
                provider_metadata={"gemini_generate_content": {"thoughtSignature": "sig_1"}},
            )
        ],
        raw={},
        provider_metadata={"openai_responses": {"response_id": "resp_1"}},
    )

    message = metadata.assistant_message_from_response(response)

    assert message[metadata.PROVIDER_METADATA_KEY] == {
        "openai_responses": {"response_id": "resp_1"},
        "_tool_calls": [
            {
                "id": "call_1",
                "index": 0,
                "metadata": {"gemini_generate_content": {"thoughtSignature": "sig_1"}},
            }
        ],
    }
    assert metadata.strip_provider_metadata_from_message(message) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fs_read", "arguments": '{"path": "README.md"}'},
            }
        ],
    }


def test_shared_metadata_helpers_deep_copy_nested_provider_state() -> None:
    output_items = [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "Sylliptor metadata"},
        }
    ]
    tool_extra_content = {
        "google": {
            "thought_signature": "original-signature",
            "grounding": [{"uri": "https://example.test/original"}],
        }
    }
    response = LLMResponse(
        content="Hosted search answer.",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="fs_read",
                arguments={"path": "README.md"},
                provider_metadata={
                    "gemini": {
                        "extra_content": tool_extra_content,
                    }
                },
            )
        ],
        raw={},
        provider_metadata={
            "openai_responses": {
                "response_id": "resp_1",
                "output_items": output_items,
            }
        },
    )

    message = metadata.assistant_message_from_response(response)

    output_items[0]["id"] = "mutated"
    tool_extra_content["google"]["thought_signature"] = "mutated-signature"
    tool_extra_content["google"]["grounding"][0]["uri"] = "https://example.test/mutated"

    stored_metadata = message[metadata.PROVIDER_METADATA_KEY]
    assert stored_metadata["openai_responses"]["output_items"][0]["id"] == "ws_1"
    stored_tool_metadata = stored_metadata["_tool_calls"][0]["metadata"]
    assert (
        stored_tool_metadata["gemini"]["extra_content"]["google"]["thought_signature"]
        == "original-signature"
    )
    assert (
        stored_tool_metadata["gemini"]["extra_content"]["google"]["grounding"][0]["uri"]
        == "https://example.test/original"
    )
