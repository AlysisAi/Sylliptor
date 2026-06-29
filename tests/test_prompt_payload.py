from __future__ import annotations

import json
from pathlib import Path

from sylliptor_agent_cli import agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.skills import SkillBundle, build_explicit_skill_context_message
from sylliptor_agent_cli.skills.prompting import EXPLICIT_SKILL_CONTEXT_TOTAL_MAX_CHARS


def _fake_git_repo(root: Path) -> None:
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text("0" * 40 + "\n", encoding="utf-8")


def _system_prompt(session: object) -> str:
    messages = getattr(session, "messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if str(message.get("role") or "") == "system":
            return str(message.get("content") or "")
    return ""


def _workspace_binding_context(session: object) -> str:
    messages = getattr(session, "messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "")
        if content.lstrip().startswith("<workspace_binding_context>"):
            return content
    return ""


def _estimated_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _write_skill(root: Path, rel_root: str, bundle_name: str) -> None:
    bundle = root / rel_root / bundle_name
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "SKILL.md").write_text(
        (
            "---\n"
            "name: pytest\n"
            "description: Debug pytest failures safely.\n"
            "---\n\n"
            "Read failing tests and fix the root cause.\n"
        ),
        encoding="utf-8",
    )


def test_create_session_adds_write_guidance_only_for_writable_modes(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", web_search_mode="off")

    auto_session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    readonly_session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="readonly",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        auto_prompt = _system_prompt(auto_session)
        readonly_prompt = _system_prompt(readonly_session)

        assert "Editing workflow" in auto_prompt
        assert (
            "Tool descriptions are the canonical source for tool strategy and parameters."
            in auto_prompt
        )
        assert "Editing workflow" not in readonly_prompt
    finally:
        auto_session.close()
        readonly_session.close()


def test_create_session_splits_skill_lifecycle_and_discovery_guidance(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", web_search_mode="off")
    session_without_skills = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    _write_skill(tmp_path, ".sylliptor_skills", "pytest")
    session_with_skills = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        no_skills_prompt = _system_prompt(session_without_skills)
        skills_prompt = _system_prompt(session_with_skills)
        no_skills_tool_names = {
            str(item.get("function", {}).get("name") or "")
            for item in getattr(session_without_skills, "tool_list", [])
            if isinstance(item, dict)
        }
        skills_tool_names = {
            str(item.get("function", {}).get("name") or "")
            for item in getattr(session_with_skills, "tool_list", [])
            if isinstance(item, dict)
        }

        assert "Skills lifecycle" in no_skills_prompt
        assert "sylliptor skill init" in no_skills_prompt
        assert "sylliptor skill create" in no_skills_prompt
        assert "sylliptor skill validate" in no_skills_prompt
        assert "default to the managed project-local scaffold" in no_skills_prompt
        assert "Do not hand-build skill bundles with `fs_mkdir` or `fs_write`" in no_skills_prompt
        assert "Skills and skill_read" not in no_skills_prompt
        assert "skill_read(name)" not in no_skills_prompt
        assert "skill_read" not in no_skills_tool_names

        assert "Skills lifecycle" in skills_prompt
        assert "Skills and skill_read" in skills_prompt
        assert "BEFORE acting on a task that matches a skill's description" in skills_prompt
        assert "Do not invent skill names" in skills_prompt
        assert "Project-local explicit-turn skill context" in skills_prompt
        assert "sylliptor skill init" in skills_prompt
        assert "sylliptor skill validate" in skills_prompt
        assert "sylliptor skill install" in skills_prompt
        assert "skill_read" in skills_tool_names
    finally:
        session_without_skills.close()
        session_with_skills.close()


def test_create_session_respects_explicit_skills_auto_invoke_false_for_discovery_directive(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, ".sylliptor_skills", "pytest")
    session = create_session(
        cfg=AppConfig(
            model="test-model",
            web_search_mode="off",
            skills_auto_invoke=False,
        ),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        joined_messages = "\n".join(
            str(message.get("content") or "") for message in session.messages
        )
        system_prompt = _system_prompt(session)
        tool_names = {
            str(item.get("function", {}).get("name") or "")
            for item in getattr(session, "tool_list", [])
            if isinstance(item, dict)
        }

        assert session.skills_auto_invoke is False
        assert "<skill_context>" in joined_messages
        assert "skill_read" in tool_names
        assert "BEFORE acting on a task that matches a skill's description" not in system_prompt
        assert "<matched_skill_context>" not in joined_messages
    finally:
        session.close()


def test_interactive_bootstrap_payload_stays_bounded(tmp_path: Path) -> None:
    _fake_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", web_search_mode="off")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        messages_json = json.dumps(session.messages, ensure_ascii=True)
        tools_json = json.dumps(session.tool_list, ensure_ascii=True)
        assert _estimated_tokens(messages_json) + _estimated_tokens(tools_json) < 5600
    finally:
        session.close()


def test_create_session_workspace_binding_context_includes_active_workdir(
    tmp_path: Path,
) -> None:
    session = create_session(
        cfg=AppConfig(model="test-model", web_search_mode="off"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        binding_context = _workspace_binding_context(session)

        assert "<workspace_binding_context>" in binding_context
        assert f"workspace_root: {tmp_path.resolve()}" in binding_context
        assert f"focus_dir: {tmp_path.resolve()}" in binding_context
        assert f"active_workdir: {tmp_path.resolve()}" in binding_context
        assert "focus_relpath: ." in binding_context
        assert "active_workdir_relpath: ." in binding_context
    finally:
        session.close()


def test_system_prompt_instructs_model_to_use_session_set_workdir_for_navigation(
    tmp_path: Path,
) -> None:
    session = create_session(
        cfg=AppConfig(model="test-model", web_search_mode="off"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        prompt = _system_prompt(session)

        assert "active_workdir" in prompt
        assert "session_set_workdir" in prompt
        assert "path_base`/`cwd_base` to `workspace_root`" in prompt
        assert "workspace_root" in prompt
        assert "new workspace bind/session is needed" in prompt
    finally:
        session.close()


def test_one_shot_bootstrap_payload_stays_bounded(tmp_path: Path) -> None:
    _fake_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", web_search_mode="off")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        messages_json = json.dumps(session.messages, ensure_ascii=True)
        tools_json = json.dumps(session.tool_list, ensure_ascii=True)
        assert _estimated_tokens(messages_json) + _estimated_tokens(tools_json) < 6400
    finally:
        session.close()


def test_create_session_wires_optional_prompt_cache_knobs(tmp_path: Path) -> None:
    _fake_git_repo(tmp_path)
    cfg = AppConfig(
        model="test-model",
        web_search_mode="off",
        prompt_cache_key="repo-main",
        prompt_cache_retention="24h",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        assert session.client.prompt_cache_key == "repo-main"
        assert session.client.prompt_cache_retention == "24h"
    finally:
        session.close()


def test_route_context_payload_stays_compact_and_structured(tmp_path: Path) -> None:
    _fake_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("notes cli\n", encoding="utf-8")
    session = create_session(
        cfg=AppConfig(model="test-model", web_search_mode="off"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        route_context = agent_loop_mod._turn_route_context(
            session,
            had_active_workspace_task_before_turn=False,
        )
        message = agent_loop_mod._route_context_system_message(route_context)
        assert message is not None
        marker, _newline, payload_raw = message.partition("\n")
        payload = json.loads(payload_raw)
        assert marker == agent_loop_mod._ROUTE_CONTEXT_MARKER
        assert payload["workspace_kind"] == "git_repo"
        assert payload["stable_grounding_available"] is True
        assert payload["workspace_hint"] == "notes cli"
        assert payload["active_workspace_task"] is False
        assert len(message) < 900
    finally:
        session.close()


def test_route_context_payload_keeps_workspace_hint_without_repo_scan(tmp_path: Path) -> None:
    _fake_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("notes cli\n", encoding="utf-8")
    session = create_session(
        cfg=AppConfig(
            model="test-model",
            web_search_mode="off",
            verify_commands=["pytest tests/test_prompt_payload.py -q"],
        ),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
    )
    try:
        route_context = agent_loop_mod._turn_route_context(
            session,
            had_active_workspace_task_before_turn=False,
        )
        assert route_context is not None
        payload = route_context.to_payload()
        assert payload["grounding_source"] == "top_level"
        assert payload["stable_grounding_available"] is True
        assert payload["workspace_hint"] == "notes cli"
    finally:
        session.close()


def test_session_start_logs_workspace_grounding_descriptor(tmp_path: Path) -> None:
    _fake_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("notes cli\n", encoding="utf-8")
    session = create_session(
        cfg=AppConfig(model="test-model", web_search_mode="off"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.close()

    [session_start] = [
        event for event in read_session_events(event_path) if event["type"] == "session_start"
    ]
    payload = session_start["payload"]
    grounding = payload["workspace_grounding"]
    assert grounding["workspace_kind"] == "git_repo"
    assert grounding["stable_grounding_available"] is True
    assert grounding["workspace_hint"] == "notes cli"
    assert "anchor_paths" in grounding


def test_explicit_skill_context_payload_is_turn_scoped_and_argument_bound() -> None:
    skill = SkillBundle(
        name="pytest",
        description="Debug pytest failures safely.",
        instructions='Investigate "$ARGUMENTS" using $1 then $2.',
        bundle_name="pytest",
        bundle_path=Path("/tmp/pytest"),
        entry_path=Path("/tmp/pytest/SKILL.md"),
        source_scope="project",
        source_kind="native",
        source_family=".sylliptor_skills",
        source_path=Path("/tmp/pytest"),
        trust_level="untrusted",
    )

    payload = build_explicit_skill_context_message(
        skill=skill,
        task_text='debug parser "retry bug"',
    )

    assert "This skill is attached only for the current turn." in payload
    assert '- $ARGUMENTS = "debug parser \\"retry bug\\""' in payload
    assert '- $1 = "debug"' in payload
    assert '- $2 = "parser"' in payload
    assert 'Investigate "debug parser "retry bug"" using debug then parser.' in payload


def test_explicit_skill_context_payload_stays_within_total_wrapper_budget() -> None:
    skill = SkillBundle(
        name="oversized",
        description="Large explicit wrapper",
        instructions=("Inspect $ARGUMENTS carefully.\n" * 40) + "Start with $1.\n",
        bundle_name="oversized",
        bundle_path=Path("/tmp/oversized"),
        entry_path=Path("/tmp/oversized/SKILL.md"),
        source_scope="project",
        source_kind="native",
        source_family=".sylliptor_skills",
        source_path=Path("/tmp/oversized"),
        trust_level="untrusted",
    )

    payload = build_explicit_skill_context_message(
        skill=skill,
        task_text="δοκιμή " * 4_000,
    )

    assert len(payload) <= EXPLICIT_SKILL_CONTEXT_TOTAL_MAX_CHARS
    assert "entrypoint_notice:" in payload
    assert "The direct user task remains available in the next user message." in payload
    assert payload.endswith("</explicit_skill_context>\n")


def test_explicit_skill_context_payload_stays_structurally_closed_with_oversized_metadata() -> None:
    skill = SkillBundle(
        name="n" * 2_000,
        description="d" * 3_000,
        instructions='Inspect "$ARGUMENTS" carefully.',
        bundle_name="oversized",
        bundle_path=Path("/tmp/oversized"),
        entry_path=Path("/tmp/oversized/SKILL.md"),
        source_scope="project",
        source_kind="native",
        source_family=".sylliptor_skills",
        source_path=Path("/tmp/" + ("nested/" * 80) + "oversized"),
        trust_level="untrusted",
    )

    payload = build_explicit_skill_context_message(
        skill=skill,
        task_text="run audit",
    )

    assert len(payload) <= EXPLICIT_SKILL_CONTEXT_TOTAL_MAX_CHARS
    assert payload.startswith("<explicit_skill_context>\n")
    assert payload.endswith("</explicit_skill_context>\n")
    assert "\n</skill_instructions>\n</explicit_skill_context>\n" in payload


def test_parse_route_decision_requires_execution_posture() -> None:
    payload = json.dumps(
        {
            "route": "repo",
            "execution_posture": "execute",
            "confidence": 0.9,
            "reply": "",
            "language": "English",
            "script": "Latin",
            "explicit_language_override": False,
        }
    )
    decision = agent_loop_mod._parse_route_decision(payload)
    assert decision is not None
    assert decision.route == "repo"
    assert decision.execution_posture == "execute"
    assert (
        agent_loop_mod._parse_route_decision(
            json.dumps(
                {
                    "route": "repo",
                    "confidence": 0.9,
                    "reply": "",
                    "language": "English",
                    "script": "Latin",
                    "explicit_language_override": False,
                }
            )
        )
        is None
    )
