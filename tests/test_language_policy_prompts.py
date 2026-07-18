from __future__ import annotations

from sylliptor_agent_cli.agent_loop import (
    _NON_REPO_RESPONSE_SYSTEM_PROMPT,
    _NON_REPO_TURN_SYSTEM_HINT,
    _ROUTER_SYSTEM_PROMPT,
)
from sylliptor_agent_cli.plan_assistant import PLANNER_SYSTEM_PROMPT
from sylliptor_agent_cli.plan_mode import PLAN_MODE_SYSTEM_PROMPT


def _assert_language_script_policy(prompt: str) -> None:
    assert (
        "Choose the natural response language" in prompt
        or "choose the natural reply language" in prompt
        or "language/script describe the reply language" in prompt
    )
    assert "Latin" in prompt
    assert "explicit" in prompt
    assert "malformed" in prompt or "ambiguous" in prompt or "gibberish" in prompt
    assert (
        "Never translate code identifiers, file paths, CLI commands, config keys, or code blocks"
        in prompt
    )


def _assert_sylliptor_identity_policy(prompt: str) -> None:
    assert "Your name is Sylliptor." in prompt
    assert "Sylliptor is built by Alysis AI." in prompt
    assert "If asked who made, created, or built you" in prompt
    assert "Official Alysis AI website: https://alysisai.com." in prompt
    assert "If asked what Alysis AI is" in prompt
    assert "Official Sylliptor website: https://sylliptor.alysisai.com." in prompt
    assert "canonical source for Sylliptor-specific product information" in prompt
    assert "affordable AI tools and Gen AI services" in prompt
    assert "decentralized compute network" in prompt
    assert (
        "Do not invent team, legal, funding, roadmap, tokenomics, pricing, customer, or launch details."
        in prompt
    )
    assert "Do not claim to be Claude, Anthropic, OpenAI, ChatGPT, Codex" in prompt
    assert "made by Anthropic/OpenAI" in prompt
    assert "If asked about the underlying model/provider" in prompt


def test_non_repo_response_prompt_has_language_script_policy() -> None:
    _assert_language_script_policy(_NON_REPO_RESPONSE_SYSTEM_PROMPT)


def test_non_repo_response_prompt_has_sylliptor_identity_policy() -> None:
    _assert_sylliptor_identity_policy(_NON_REPO_RESPONSE_SYSTEM_PROMPT)


def test_router_prompt_has_language_script_policy_for_fallback_reply() -> None:
    _assert_language_script_policy(_ROUTER_SYSTEM_PROMPT)


def test_router_prompt_has_host_owned_route_context_policy() -> None:
    assert "<<<SYLLIPTOR_ROUTE_CONTEXT_JSON>>>" in _ROUTER_SYSTEM_PROMPT
    assert "host-owned system message" in _ROUTER_SYSTEM_PROMPT
    assert "stable workspace grounding" in _ROUTER_SYSTEM_PROMPT
    assert "active task state" in _ROUTER_SYSTEM_PROMPT
    assert "artifact_capabilities" in _ROUTER_SYSTEM_PROMPT
    assert "does not know the internal tool or subagent name" in _ROUTER_SYSTEM_PROMPT
    assert "Do not downgrade a requested deliverable" in _ROUTER_SYSTEM_PROMPT


def test_router_prompt_requires_execution_posture_schema_and_policy() -> None:
    assert '"execution_posture":"execute|advisory_non_execution|plan_or_analysis_only"' in (
        _ROUTER_SYSTEM_PROMPT
    )
    assert 'execution_posture="execute"' in _ROUTER_SYSTEM_PROMPT
    assert 'execution_posture="advisory_non_execution"' in _ROUTER_SYSTEM_PROMPT
    assert 'execution_posture="plan_or_analysis_only"' in _ROUTER_SYSTEM_PROMPT
    assert "vague or typo-heavy" in _ROUTER_SYSTEM_PROMPT


def test_non_repo_turn_hint_has_language_script_policy() -> None:
    _assert_language_script_policy(_NON_REPO_TURN_SYSTEM_HINT)


def test_non_repo_turn_hint_has_sylliptor_identity_policy() -> None:
    _assert_sylliptor_identity_policy(_NON_REPO_TURN_SYSTEM_HINT)


def test_plan_mode_prompt_has_language_script_policy() -> None:
    _assert_language_script_policy(PLAN_MODE_SYSTEM_PROMPT)


def test_plan_mode_prompt_has_workspace_grounding_policy() -> None:
    assert "Treat host-provided workspace context as the source of truth" in PLAN_MODE_SYSTEM_PROMPT
    assert "describe the area generically instead of inventing details" in PLAN_MODE_SYSTEM_PROMPT


def test_planner_prompt_has_language_script_policy() -> None:
    _assert_language_script_policy(PLANNER_SYSTEM_PROMPT)
