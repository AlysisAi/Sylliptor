from __future__ import annotations

from sylliptor_agent_cli.agent_loop import (
    _SYSTEM_PROMPT_ONE_SHOT_SECTION,
    _classify_one_shot_repo_turn_intent,
    _completion_gate_nudge_message,
)


def test_one_shot_prompt_forbids_standalone_text_only_plan() -> None:
    assert "Do not emit a standalone text-only plan and wait for the user." in (
        _SYSTEM_PROMPT_ONE_SHOT_SECTION
    )
    assert "Planning may be internal" in _SYSTEM_PROMPT_ONE_SHOT_SECTION


def test_one_shot_visible_plan_must_be_accompanied_by_tool_calls() -> None:
    assert "same assistant response must also include implementation-oriented tool calls." in (
        _SYSTEM_PROMPT_ONE_SHOT_SECTION
    )


def test_one_shot_read_or_explore_only_progress_is_not_final() -> None:
    assert "A progress update is not a final answer." in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    assert "After read/explore-only tool calls" in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    assert "run an implementation-producing command" in _SYSTEM_PROMPT_ONE_SHOT_SECTION


def test_one_shot_prompt_rejects_generic_clarification_bailouts() -> None:
    assert "Do not ask a generic clarification question" in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    assert "safe best effort" in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    assert "destructive alternatives require the user's choice" in _SYSTEM_PROMPT_ONE_SHOT_SECTION


def test_one_shot_prompt_mentions_requirement_review_and_root_fixing() -> None:
    prompt = _SYSTEM_PROMPT_ONE_SHOT_SECTION.casefold()

    assert "re-read" in prompt and "requirement" in prompt
    assert "definition" in prompt and "direct call" in prompt


def test_one_shot_prompt_protects_existing_tests_and_requires_execution_evidence() -> None:
    assert "tracked existing tests as immutable acceptance evidence" in (
        _SYSTEM_PROMPT_ONE_SHOT_SECTION
    )
    assert "New test files are allowed" in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    assert "after the last source edit" in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    assert "observing its output and exit code" in _SYSTEM_PROMPT_ONE_SHOT_SECTION


def test_no_material_edits_nudge_is_implementation_first() -> None:
    message = _completion_gate_nudge_message(["no_material_edits", "verification_not_attempted"])

    assert "No file changes are recorded yet" in message
    assert "Expected verification has not been completed" in message
    assert message.index("No file changes are recorded yet") < message.index(
        "Expected verification has not been completed"
    )
    assert "this checklist is advisory" in message


def test_verification_not_attempted_nudge_is_verification_first() -> None:
    message = _completion_gate_nudge_message(["verification_not_attempted"])

    assert "Expected verification has not been completed" in message
    assert "No file changes are recorded yet" not in message
    assert "this checklist is advisory" in message


def test_plan_and_advice_only_intents_remain_non_execution() -> None:
    assert (
        _classify_one_shot_repo_turn_intent("Plan only: how should we fix the parser?")
        == "plan_or_analysis_only"
    )
    assert (
        _classify_one_shot_repo_turn_intent(
            "Explain how the parser works without modifying anything."
        )
        == "advisory_non_execution"
    )
    assert _classify_one_shot_repo_turn_intent("Fix the parser bug.") == "execute"
