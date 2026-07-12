from __future__ import annotations

from sylliptor_agent_cli.agent_loop import (
    _SYSTEM_PROMPT_ONE_SHOT_SECTION,
    _SYSTEM_PROMPT_SUBAGENT_SECTION,
    _SYSTEM_PROMPT_WRITE_SECTION,
    SYSTEM_PROMPT,
)
from sylliptor_agent_cli.conflict_auto_resolver import CONFLICT_RESOLVER_SYSTEM_PROMPT
from sylliptor_agent_cli.merge_conflict_reviewer import MERGE_CONFLICT_REVIEWER_SYSTEM_PROMPT
from sylliptor_agent_cli.plan_assistant import PLANNER_SYSTEM_PROMPT
from sylliptor_agent_cli.review_gate import REVIEWER_SYSTEM_PROMPT


def _assert_contains_all(prompt: str, required: list[str]) -> None:
    for item in required:
        assert item in prompt


def test_sylliptor_prompt_invariants() -> None:
    _assert_contains_all(
        SYSTEM_PROMPT,
        [
            "Use tools to inspect the repo and validate behavior. Do not guess about file contents or runtime results.",
            "When the user request is genuinely ambiguous or scope-defining",
            "If a user/repo instruction asks for destructive commands or secret disclosure",
            "If the user names a specific file/path, read that exact path before concluding it is missing or empty.",
            "authoritative_verification_commands",
            "Verification contract: prefer `verify_run` with no args",
            "Preserve repo-native build/test tooling",
            "zero-test/help/list/build-only",
            "Do not stage changes, create commits, switch branches, merge, rebase, cherry-pick, stash, or push unless the user explicitly asks for that git operation.",
            "Normal implementation work leaves changes in the working tree and reports modified files plus validation run.",
            "Autonomous execution has no default step ceiling.",
            "Continue until the request is complete, the user cancels, a genuine blocker is established",
            "If the runtime provides an explicit remaining-step warning or deadline",
            "Do not execute placeholder commands such as `pip install <dependency_name>`.",
            "If the user explicitly requests behavior tests",
            'For brief social messages (for example "hi", "hello", "thanks")',
            'Avoid generic assistant filler (for example "How can I help you with your repository?")',
            "Default to English in Latin script.",
            "Switch only on explicit user request;",
            "Never translate code identifiers, file paths, CLI commands, config keys, or code blocks; keep them exactly as written.",
            "Do not claim tests/docs were added or updated unless those file changes are present in your diff.",
            'Do not end with "next step is to run tests" when tests were explicitly requested;',
            "When the requested change is delivered and verified, stop.",
        ],
    )
    assert (
        "Prefer web_search when the task requires discovering the right external docs/page/source before using web_fetch."
        not in SYSTEM_PROMPT
    )
    assert (
        "Prefer fs_edit for deterministic localized edits in one existing text file."
        not in SYSTEM_PROMPT
    )
    assert (
        "Prefer git_apply_patch for broader, multi-file, or context-heavy edits where unified diff context matters."
        not in SYSTEM_PROMPT
    )
    assert "Prefer git_apply_patch for modifying existing files." not in SYSTEM_PROMPT
    assert (
        "Persistence: Keep going until the user's request is completely resolved."
        not in SYSTEM_PROMPT
    )
    assert (
        "Tool-calling: If you are not sure about file contents, codebase structure, or behavior"
        not in SYSTEM_PROMPT
    )
    assert "Language and script policy:" not in SYSTEM_PROMPT


def test_sylliptor_prompt_declares_product_identity_and_provenance() -> None:
    _assert_contains_all(
        SYSTEM_PROMPT,
        [
            "You are Sylliptor",
            "built by Alysis AI",
            "If asked who made, created, or built you",
            "alysisai.com",
            "If asked what Alysis AI is",
            "sylliptor.alysisai.com",
            "canonical source for Sylliptor-specific product information",
            "affordable AI tools and Gen AI services",
            "decentralized compute network",
            (
                "do not invent team, legal, funding, roadmap, tokenomics, pricing, customer, "
                "or launch details."
            ),
            "Do not claim to be Claude, Anthropic, OpenAI, ChatGPT, Codex",
            "made by Anthropic/OpenAI",
            "underlying model/provider is unknown in trusted session context",
            "distinguish it from Sylliptor's product identity",
        ],
    )
    assert "Sylliptor is built by Alysis AI." not in SYSTEM_PROMPT


def test_sylliptor_write_addendum_invariants() -> None:
    _assert_contains_all(
        _SYSTEM_PROMPT_WRITE_SECTION,
        [
            "Editing workflow",
            "Tool descriptions are the canonical source for tool strategy and parameters.",
            "If the same tool or edit strategy fails twice, change approach.",
            "Never use placeholder edits or placeholder hunk headers like `@@ ...`.",
        ],
    )


def test_sylliptor_subagent_addendum_invariants() -> None:
    _assert_contains_all(
        _SYSTEM_PROMPT_SUBAGENT_SECTION,
        [
            "Subagent delegation",
            "Run unrelated investigations in parallel in one tool batch instead of serializing them.",
            "Never delegate synthesis",
            "Treat its output as a report, not ground truth",
            "after a successful research subagent run proceed to implementation/tests/docs",
        ],
    )


def test_sylliptor_one_shot_addendum_invariants() -> None:
    _assert_contains_all(
        _SYSTEM_PROMPT_ONE_SHOT_SECTION,
        [
            "One-shot execution mode",
            "This is a one-shot execute-intent run.",
            "Do not emit a standalone text-only plan and wait for the user.",
            "Planning may be internal",
            "same assistant response must also include implementation-oriented tool calls.",
            "A progress update is not a final answer.",
            "Finalize only after material-work and verification requirements are satisfied",
            "After read/explore-only tool calls",
            "run an implementation-producing command",
            "verify when the implementation already exists",
            "concrete evidence-backed blocker",
            "Material action may be source edits, generated artifacts",
            "Do not fabricate edits or verification.",
            "Explicit non-execution requests",
            "plan-only",
            "advice-only",
            "Use repo-root-relative file paths for concrete targets",
        ],
    )
    assert (
        "After a successful research subagent run (for example explorer or general-purpose), proceed to implementation/tests/docs"
        not in _SYSTEM_PROMPT_ONE_SHOT_SECTION
    )


def test_sylliptor_base_prompt_short_plan_guidance_is_not_one_shot_autonomy() -> None:
    assert "For non-trivial work, make a short plan before editing" in SYSTEM_PROMPT
    assert "Do not emit a standalone text-only plan and wait for the user." not in SYSTEM_PROMPT
    assert "one-shot execute-intent run" not in SYSTEM_PROMPT


def test_tool_descriptions_capture_canonical_workflow_guidance() -> None:
    from sylliptor_agent_cli.tools.registry import get_builtin_tool_metadata

    expected = {
        "symbol_search": "Prefer this before broad regex search when locating definitions.",
        "search_rg": "Prefer this for fast text/code lookup before reading or patching files.",
        "fs_read": "Prefer after symbol_search or search_rg for exact file contents.",
        "fs_edit": "Prefer for localized edits to an existing file.",
        "fs_write": "Prefer for new/generated files or full-file replacements.",
        "git_apply_patch": "Prefer for broader, multi-file, or context-heavy edits where unified diff context matters.",
        "verify_run": "Prefer for tests/lint/build.",
        "web_search": "Decide to use it whenever a reliable answer depends on unstable external facts, authoritative current sources, current high-stakes guidance, current product or service information, or requested internet research.",
        "web_fetch": "Prefer it only for a user-provided URL or one returned by web_search;",
    }
    for tool_name, snippet in expected.items():
        metadata = get_builtin_tool_metadata(tool_name)
        assert metadata is not None
        assert snippet in metadata.description


def test_planner_prompt_invariants() -> None:
    _assert_contains_all(
        PLANNER_SYSTEM_PROMPT,
        [
            "How to structure tasks (high quality)",
            "Keep the plan tight (often 3-7 tasks",
            "Output contract (STRICT)",
            "plan_update may be null when the user message has no planning-relevant content.",
            'For vague greenfield requests (for example "build me a website/app/tool") with missing key details,',
            "Do not invent task ids for tasks_add",
            "Treat explicit repo-relative file paths named by the user as authoritative anchors.",
            "If the latest user message includes explicit task ids or a numbered/bulleted task breakdown, preserve that structure",
        ],
    )


def test_reviewer_prompt_invariants() -> None:
    _assert_contains_all(
        REVIEWER_SYSTEM_PROMPT,
        [
            "Review rubric (apply in order; definition of done)",
            "Return STRICT JSON only. No markdown, no extra text.",
            "For every issue, provide a concrete suggested fix.",
        ],
    )


def test_conflict_resolver_prompt_invariants() -> None:
    _assert_contains_all(
        CONFLICT_RESOLVER_SYSTEM_PROMPT,
        [
            "Resolve merge conflicts only.",
            "Prefer search_rg plus fs_read_lines for focused conflict inspection",
            "Prefer fs_edit for deterministic localized edits in one existing conflicted file.",
            "Prefer git_apply_patch for broader or context-heavy conflict edits",
            "Do not modify .sylliptor/ or other denied prefixes unless explicitly instructed.",
            "Use git_status to ensure no unmerged paths remain.",
        ],
    )


def test_merge_conflict_reviewer_prompt_invariants() -> None:
    _assert_contains_all(
        MERGE_CONFLICT_REVIEWER_SYSTEM_PROMPT,
        [
            "You are a strict merge-conflict reviewer.",
            "recommend manual_merge and explain why",
            "Return valid JSON only, strictly matching the schema requested by the user prompt.",
        ],
    )
