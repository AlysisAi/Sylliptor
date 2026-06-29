from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.execution_deadline import ExecutionDeadline
from sylliptor_agent_cli.session_store import (
    SessionStore,
    list_sessions,
    make_session_id,
    read_session_events,
)
from sylliptor_agent_cli.web_research import (
    build_web_research_artifact_from_events,
    canonicalize_web_url_input,
    extract_public_web_urls,
    normalize_web_url,
)


def _write_web_artifact_ahead_of_log(
    *,
    sessions_dir: Path,
    session_id: str,
    logged_events: list[dict[str, object]],
    artifact_only_events: list[dict[str, object]],
) -> None:
    (sessions_dir / f"{session_id}.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=True) + "\n" for event in logged_events),
        encoding="utf-8",
    )
    artifact_root = sessions_dir / session_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_payload = build_web_research_artifact_from_events(logged_events + artifact_only_events)
    (artifact_root / "web_research_sources.json").write_text(
        json.dumps(artifact_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def test_session_store_writes_jsonl(tmp_path: Path) -> None:
    sid = make_session_id()
    store = SessionStore(
        enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append("user_message", {"content": "hi"})
    store.append("final", {"content": "bye"})
    store.close()

    p = tmp_path / f"{sid}.jsonl"
    assert p.exists()
    events = list(read_session_events(p))
    assert [e["type"] for e in events] == ["user_message", "final"]
    assert all("cwd" in event for event in events)
    assert all("repo_root" in event for event in events)


def test_session_store_assigns_monotonic_event_ids(tmp_path: Path) -> None:
    store = SessionStore(
        enabled=True,
        sessions_dir=tmp_path,
        session_id="event-id-test",
        cwd=".",
        repo_root=".",
    )
    store.append("user_message", {"content": "hi"})
    store.append("final", {"content": "bye"})
    store.close()

    events = list(read_session_events(tmp_path / "event-id-test.jsonl"))

    assert [event["event_id"] for event in events] == ["event-id-test:1", "event-id-test:2"]


def test_session_store_disables_logging_when_sessions_dir_is_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    sid = make_session_id()
    sessions_dir = tmp_path / "sessions"
    real_mkdir = Path.mkdir

    def fail_sessions_dir_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == sessions_dir:
            raise OSError(errno.EROFS, "Read-only file system", os.fspath(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_sessions_dir_mkdir)

    store = SessionStore(
        enabled=True,
        sessions_dir=sessions_dir,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append("user_message", {"content": "hi"})
    store.close()

    assert store.enabled is False
    assert store.artifact_persistence_enabled is False
    assert not (sessions_dir / f"{sid}.jsonl").exists()


def test_no_log_keeps_session_log_disabled_but_allows_explicit_diagnostics(
    tmp_path: Path,
) -> None:
    diagnostic_log = tmp_path / "diagnostics" / "events.jsonl"
    deadline = ExecutionDeadline.from_absolute(
        started_at_monotonic=10.0,
        deadline_monotonic=10.0,
        configured_duration_seconds=1.0,
        clock=lambda: 10.0,
    )
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        execution_deadline=deadline,
        crash_diagnostic_log_path=diagnostic_log,
    )

    try:
        assert session.run_turn("Do the task.") == 1
        session_log_path = session.store.path
    finally:
        session.close()

    assert session_log_path.exists() is False
    events = [json.loads(line) for line in diagnostic_log.read_text(encoding="utf-8").splitlines()]
    event_types = [event["event_type"] for event in events]
    assert "run_started" in event_types
    assert "turn_started" in event_types
    assert "deadline_exhausted" in event_types
    assert "run_finished" in event_types


def test_normalize_web_url_preserves_valid_parenthesized_path() -> None:
    assert (
        normalize_web_url("https://docs.example.com/Function_(mathematics)")
        == "https://docs.example.com/Function_(mathematics)"
    )


def test_normalize_web_url_preserves_bracket_query_params() -> None:
    assert (
        normalize_web_url("https://docs.example.com/path?foo[bar]=1")
        == "https://docs.example.com/path?foo[bar]=1"
    )
    assert (
        normalize_web_url("https://docs.example.com/path?arr[]=1")
        == "https://docs.example.com/path?arr[]=1"
    )


def test_canonicalize_web_url_input_preserves_legal_trailing_exclamation() -> None:
    assert (
        canonicalize_web_url_input("https://docs.example.com/Yahoo!")
        == "https://docs.example.com/Yahoo!"
    )


def test_extract_public_web_urls_preserves_valid_parenthesized_path() -> None:
    extracted = extract_public_web_urls("Read https://docs.example.com/Function_(mathematics) now.")

    assert extracted == [
        {
            "url": "https://docs.example.com/Function_(mathematics)",
            "normalized_url": "https://docs.example.com/Function_(mathematics)",
            "domain": "docs.example.com",
        }
    ]


def test_extract_public_web_urls_preserves_bracket_query_params() -> None:
    cases = [
        (
            "Please read https://docs.example.com/path?foo[bar]=1 before changes.",
            "https://docs.example.com/path?foo[bar]=1",
        ),
        (
            "See https://docs.example.com/path?arr[]=1 now.",
            "https://docs.example.com/path?arr[]=1",
        ),
    ]

    for text, expected_url in cases:
        assert extract_public_web_urls(text) == [
            {
                "url": expected_url,
                "normalized_url": expected_url,
                "domain": "docs.example.com",
            }
        ]


def test_extract_public_web_urls_preserves_exclamation_and_apostrophes() -> None:
    cases = [
        (
            "Please read https://docs.example.com/Yahoo! before changes.",
            "https://docs.example.com/Yahoo!",
        ),
        (
            "Read https://docs.example.com/O'Connor now.",
            "https://docs.example.com/O'Connor",
        ),
        (
            "Read https://docs.example.com/path?name=O'Connor now.",
            "https://docs.example.com/path?name=O'Connor",
        ),
    ]

    for text, expected_url in cases:
        assert extract_public_web_urls(text) == [
            {
                "url": expected_url,
                "normalized_url": expected_url,
                "domain": "docs.example.com",
            }
        ]


def test_extract_public_web_urls_parses_markdown_link_targets() -> None:
    cases = [
        (
            "Please read ([docs](https://docs.example.com/spec)) before changes.",
            "https://docs.example.com/spec",
        ),
        (
            "[docs](https://docs.example.com/spec)",
            "https://docs.example.com/spec",
        ),
        (
            "[docs](https://docs.example.com/spec).",
            "https://docs.example.com/spec",
        ),
    ]

    for text, expected_url in cases:
        assert extract_public_web_urls(text) == [
            {
                "url": expected_url,
                "normalized_url": expected_url,
                "domain": "docs.example.com",
            }
        ]


def test_extract_public_web_urls_preserves_structured_markdown_target_punctuation() -> None:
    cases = [
        ("[docs](https://docs.example.com/path.)", "https://docs.example.com/path."),
        ("[docs](https://docs.example.com/path:)", "https://docs.example.com/path:"),
        ("[docs](https://docs.example.com/path;)", "https://docs.example.com/path;"),
        ("[docs](https://docs.example.com/path,)", "https://docs.example.com/path,"),
        ("<https://docs.example.com/path.>", "https://docs.example.com/path."),
    ]

    for text, expected_url in cases:
        assert extract_public_web_urls(text) == [
            {
                "url": expected_url,
                "normalized_url": expected_url,
                "domain": "docs.example.com",
            }
        ]


def test_extract_public_web_urls_preserves_markdown_link_title_parenthesized_targets() -> None:
    cases = [
        (
            '[Function](https://docs.example.com/Function_(mathematics) "Function docs")',
            "https://docs.example.com/Function_(mathematics)",
        ),
        (
            '[docs](https://docs.example.com/path_(foo) "Title")',
            "https://docs.example.com/path_(foo)",
        ),
        (
            "[Function](https://docs.example.com/Function_(mathematics))",
            "https://docs.example.com/Function_(mathematics)",
        ),
        (
            '[Function](<https://docs.example.com/Function_(mathematics)> "Function docs")',
            "https://docs.example.com/Function_(mathematics)",
        ),
    ]

    for text, expected_url in cases:
        assert extract_public_web_urls(text) == [
            {
                "url": expected_url,
                "normalized_url": expected_url,
                "domain": "docs.example.com",
            }
        ]


def test_extract_public_web_urls_strips_sentence_punctuation_from_generic_prose() -> None:
    assert extract_public_web_urls(
        "Please read https://docs.example.com/spec. before changes."
    ) == [
        {
            "url": "https://docs.example.com/spec",
            "normalized_url": "https://docs.example.com/spec",
            "domain": "docs.example.com",
        }
    ]


def test_extract_public_web_urls_preserves_order_across_plain_and_markdown_links() -> None:
    extracted = extract_public_web_urls(
        "Read https://docs.example.com/first then [second](https://docs.example.com/second)."
    )

    assert [entry["url"] for entry in extracted] == [
        "https://docs.example.com/first",
        "https://docs.example.com/second",
    ]


def test_extract_public_web_urls_strips_common_markdown_wrappers() -> None:
    cases = [
        "Please read `https://docs.example.com/spec`.",
        "See *https://docs.example.com/spec* now.",
        "See **https://docs.example.com/spec** now.",
        "See _https://docs.example.com/spec_ now.",
        "See __https://docs.example.com/spec__ now.",
    ]

    for text in cases:
        assert extract_public_web_urls(text) == [
            {
                "url": "https://docs.example.com/spec",
                "normalized_url": "https://docs.example.com/spec",
                "domain": "docs.example.com",
            }
        ]


def test_canonicalize_web_url_input_preserves_legal_trailing_parenthesis() -> None:
    assert (
        canonicalize_web_url_input("https://docs.example.com/path?x=a)")
        == "https://docs.example.com/path?x=a)"
    )


def test_extract_public_web_urls_preserves_legal_trailing_parenthesis_in_markdown_link() -> None:
    extracted = extract_public_web_urls(
        "Please read [docs](https://docs.example.com/path?x=a)) before changes."
    )

    assert extracted == [
        {
            "url": "https://docs.example.com/path?x=a)",
            "normalized_url": "https://docs.example.com/path?x=a)",
            "domain": "docs.example.com",
        }
    ]


def test_extract_public_web_urls_preserves_legal_trailing_parenthesis_in_plain_prose() -> None:
    extracted = extract_public_web_urls(
        "Please read https://docs.example.com/path?x=a) before changes."
    )

    assert extracted == [
        {
            "url": "https://docs.example.com/path?x=a)",
            "normalized_url": "https://docs.example.com/path?x=a)",
            "domain": "docs.example.com",
        }
    ]


def test_session_store_writes_workspace_metadata_additively(tmp_path: Path) -> None:
    sid = make_session_id()
    store = SessionStore(
        enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd="/repo/pkg",
        repo_root="/repo/pkg",
        workspace_root="/repo",
        focus_dir="/repo/pkg",
        git_root="/repo",
        workspace_kind="git_repo",
        binding_source="cwd",
        binding_requested_path="/repo/pkg",
        binding_risk_level="healthy",
        binding_created_path=False,
    )
    store.append("session_start", {"mode": "review"})
    store.close()

    [event] = list(read_session_events(tmp_path / f"{sid}.jsonl"))
    assert event["cwd"] == "/repo/pkg"
    assert event["repo_root"] == "/repo/pkg"
    assert event["workspace_root"] == "/repo"
    assert event["focus_dir"] == "/repo/pkg"
    assert event["git_root"] == "/repo"
    assert event["workspace_kind"] == "git_repo"
    assert event["binding_source"] == "cwd"
    assert event["binding_requested_path"] == "/repo/pkg"
    assert event["binding_risk_level"] == "healthy"
    assert event["binding_created_path"] is False


def test_list_sessions(tmp_path: Path) -> None:
    sid = make_session_id()
    (tmp_path / f"{sid}.jsonl").write_text("{}", encoding="utf-8")
    infos = list_sessions(tmp_path)
    assert any(i.session_id == sid for i in infos)


def test_session_store_reopen_preserves_user_provided_web_url_classification(
    tmp_path: Path,
) -> None:
    sid = "resume-user-url"
    store = SessionStore(
        enabled=True,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append(
        "user_message",
        {"content": "Please inspect https://docs.example.com/spec before changing anything."},
    )
    store.close()

    reopened = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )

    assert reopened.classify_web_fetch_url("https://docs.example.com/spec") == "user_provided"


def test_session_store_fetchable_urls_lists_search_sources_then_user_urls(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=True,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="fetchable-urls",
        cwd=".",
        repo_root=".",
    )
    store.append(
        "user_message",
        {"content": "Check https://user.example.com/page later."},
    )
    store.append(
        "tool_result",
        {
            "name": "web_search",
            "step": 1,
            "result": {
                "query": "world cup news",
                "backend": "openrouter_web",
                "sources": [
                    {"title": "FIFA", "url": "https://www.fifa.com/worldcup/news"},
                    {"title": "UEFA", "url": "https://www.uefa.com/news"},
                ],
            },
        },
    )

    fetchable = store.fetchable_web_fetch_urls()

    # web_search sources first (in result order), then the user-provided URL.
    assert fetchable == [
        "https://www.fifa.com/worldcup/news",
        "https://www.uefa.com/news",
        "https://user.example.com/page",
    ]
    # Every listed URL is genuinely authorized for web_fetch.
    for url in fetchable:
        assert store.classify_web_fetch_url(url) is not None
    # The bound is honored.
    assert store.fetchable_web_fetch_urls(limit=1) == ["https://www.fifa.com/worldcup/news"]


def test_session_store_reopen_preserves_search_returned_web_url_classification(
    tmp_path: Path,
) -> None:
    sid = "resume-search-url"
    store = SessionStore(
        enabled=True,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append(
        "tool_result",
        {
            "name": "web_search",
            "step": 1,
            "result": {
                "query": "docs example",
                "backend": "openai_responses",
                "sources": [
                    {
                        "title": "Guide",
                        "url": "https://docs.example.com/guide",
                        "snippet": "Official guide",
                    }
                ],
            },
        },
    )
    store.close()

    reopened = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )

    assert (
        reopened.classify_web_fetch_url("https://docs.example.com/guide")
        == "returned_by_web_search"
    )


def test_session_store_reopen_keeps_cumulative_web_artifact_after_new_events(
    tmp_path: Path,
) -> None:
    sid = "resume-cumulative-web"
    store = SessionStore(
        enabled=True,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append(
        "user_message",
        {"content": "Please inspect https://docs.example.com/spec before changing anything."},
    )
    store.append(
        "tool_result",
        {
            "name": "web_search",
            "step": 1,
            "result": {
                "query": "docs example",
                "backend": "openai_responses",
                "sources": [
                    {
                        "title": "Guide",
                        "url": "https://docs.example.com/guide",
                        "snippet": "Official guide",
                    }
                ],
            },
        },
    )
    store.close()

    reopened = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    reopened.append(
        "tool_call",
        {"name": "web_fetch", "step": 2, "arguments": {"url": "https://docs.example.com/guide"}},
    )
    reopened.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 2,
            "result": {
                "url": "https://docs.example.com/guide",
                "final_url": "https://docs.example.com/final",
                "status_code": 200,
                "content_type": "text/html",
                "title": "Final",
                "backend": "httpx",
            },
        },
    )
    reopened.close()

    payload = json.loads((tmp_path / sid / "web_research_sources.json").read_text(encoding="utf-8"))

    assert payload["deduped_normalized_user_urls"] == ["https://docs.example.com/spec"]
    assert payload["deduped_normalized_search_source_urls"] == ["https://docs.example.com/guide"]
    assert payload["deduped_normalized_fetch_urls"] == ["https://docs.example.com/guide"]
    assert payload["deduped_normalized_final_fetch_urls"] == ["https://docs.example.com/final"]
    assert payload["fetches"][0]["final_url"] == "https://docs.example.com/final"


def test_session_store_web_artifact_dedupes_fetch_url_spelling_variants(
    tmp_path: Path,
) -> None:
    sid = "fetch-url-canonical-dedupe"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append(
        "user_message",
        {"content": "Please inspect https://docs.example.com/spec. before changing anything."},
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": "https://docs.example.com/spec."}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": "https://docs.example.com/spec",
                "final_url": "https://docs.example.com/spec",
                "status_code": 200,
                "content_type": "text/html",
                "title": "Spec",
                "backend": "httpx",
                "raw_input_url": "https://docs.example.com/spec.",
            },
        },
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 2, "arguments": {"url": "https://docs.example.com/spec"}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 2,
            "result": {
                "url": "https://docs.example.com/spec",
                "final_url": "https://docs.example.com/spec",
                "status_code": 200,
                "content_type": "text/html",
                "title": "Spec",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()

    assert payload["deduped_normalized_user_urls"] == ["https://docs.example.com/spec"]
    assert payload["deduped_normalized_fetch_urls"] == ["https://docs.example.com/spec"]
    assert [entry["requested_url"] for entry in payload["fetches"]] == [
        "https://docs.example.com/spec",
        "https://docs.example.com/spec",
    ]
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 1


def test_session_store_resolves_bracket_query_user_url_to_clean_canonical_url(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="bracket-query-user-url",
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/path?foo[bar]=1"
    truncated_url = "https://docs.example.com/path?foo"
    store.append(
        "user_message",
        {"content": f"Please read {clean_url} before changes."},
    )

    assert store.resolve_web_fetch_url(clean_url) == (
        "user_provided",
        clean_url,
    )
    assert store.classify_web_fetch_url(truncated_url) is None


def test_session_store_resolves_exclamation_user_url_to_exact_canonical_url(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="exclamation-user-url",
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/Yahoo!"
    truncated_url = "https://docs.example.com/Yahoo"
    store.append(
        "user_message",
        {"content": f"Please read {clean_url} before changes."},
    )

    assert store.resolve_web_fetch_url(clean_url) == (
        "user_provided",
        clean_url,
    )
    assert store.classify_web_fetch_url(truncated_url) is None


def test_session_store_resolves_parenthesized_markdown_link_to_clean_target_url(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="parenthesized-markdown-link-url",
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/spec"
    corrupted_url = "https://docs.example.com/spec)"
    store.append(
        "user_message",
        {"content": "Please read ([docs](https://docs.example.com/spec)) before changes."},
    )

    assert store.resolve_web_fetch_url(clean_url) == (
        "user_provided",
        clean_url,
    )
    assert store.classify_web_fetch_url(corrupted_url) is None


def test_session_store_resolves_markdown_link_title_parenthesized_target_url(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="markdown-link-title-parenthesized-target",
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/Function_(mathematics)"
    truncated_url = "https://docs.example.com/Function_(mathematics"
    store.append(
        "user_message",
        {
            "content": (
                "Please read [Function](https://docs.example.com/Function_(mathematics) "
                '"Function docs") before changes.'
            )
        },
    )

    assert store.resolve_web_fetch_url(clean_url) == (
        "user_provided",
        clean_url,
    )
    assert store.classify_web_fetch_url(truncated_url) is None

    payload = store.web_research_artifact_payload()
    assert payload["deduped_normalized_user_urls"] == [clean_url]
    assert truncated_url not in payload["deduped_normalized_user_urls"]


def test_session_store_resolves_backtick_wrapped_user_url_to_clean_canonical_url(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="wrapped-user-url",
        cwd=".",
        repo_root=".",
    )
    store.append(
        "user_message",
        {"content": "Please read `https://docs.example.com/spec` and use it."},
    )

    assert store.resolve_web_fetch_url("https://docs.example.com/spec") == (
        "user_provided",
        "https://docs.example.com/spec",
    )


def test_session_store_resolves_markdown_link_url_with_legal_trailing_parenthesis(
    tmp_path: Path,
) -> None:
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id="markdown-link-trailing-parenthesis",
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/path?x=a)"
    store.append(
        "user_message",
        {"content": f"Please read [docs]({clean_url}) before changes."},
    )

    assert store.resolve_web_fetch_url(clean_url) == (
        "user_provided",
        clean_url,
    )
    assert store.classify_web_fetch_url("https://docs.example.com/path?x=a") is None


def test_session_store_web_artifact_preserves_structured_markdown_target_punctuation(
    tmp_path: Path,
) -> None:
    sid = "structured-markdown-target-punctuation"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/path."
    truncated_url = "https://docs.example.com/path"
    store.append(
        "user_message",
        {
            "content": (
                "Please inspect [docs](https://docs.example.com/path.) and "
                "<https://docs.example.com/path.> before changes."
            )
        },
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": clean_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Docs",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()

    assert payload["deduped_normalized_user_urls"] == [clean_url]
    assert payload["deduped_normalized_fetch_urls"] == [clean_url]
    assert truncated_url not in payload["deduped_normalized_user_urls"]
    assert payload["fetches"][0]["requested_url"] == clean_url
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 0


def test_session_store_web_artifact_dedupes_parenthesized_markdown_link_target(
    tmp_path: Path,
) -> None:
    sid = "parenthesized-markdown-link-dedupe"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/spec"
    corrupted_url = "https://docs.example.com/spec)"
    store.append(
        "user_message",
        {
            "content": (
                "Please inspect ([docs](https://docs.example.com/spec)) and "
                "https://docs.example.com/spec before changes."
            )
        },
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": clean_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Spec",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()

    assert payload["deduped_normalized_user_urls"] == [clean_url]
    assert payload["deduped_normalized_fetch_urls"] == [clean_url]
    assert corrupted_url not in payload["deduped_normalized_user_urls"]
    assert payload["fetches"][0]["requested_url"] == clean_url
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 0


def test_session_store_web_artifact_dedupes_markdown_wrapped_fetch_url_variants(
    tmp_path: Path,
) -> None:
    sid = "fetch-url-markdown-wrapper-dedupe"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    raw_wrapped_url = "`https://docs.example.com/spec`"
    clean_url = "https://docs.example.com/spec"
    store.append(
        "user_message",
        {
            "content": (
                "Please inspect `https://docs.example.com/spec`, "
                "*https://docs.example.com/spec*, and _https://docs.example.com/spec_."
            )
        },
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": raw_wrapped_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Spec",
                "backend": "httpx",
                "raw_input_url": raw_wrapped_url,
            },
        },
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 2, "arguments": {"url": clean_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 2,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Spec",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()
    first_fetch = payload["fetches"][0]

    assert payload["deduped_normalized_user_urls"] == [clean_url]
    assert payload["deduped_normalized_fetch_urls"] == [clean_url]
    assert [entry["requested_url"] for entry in payload["fetches"]] == [clean_url, clean_url]
    assert first_fetch["raw_input_url"] == raw_wrapped_url
    assert first_fetch["provenance_classification"] == "user_provided"
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 1


def test_session_store_web_artifact_dedupes_urls_with_legal_trailing_parenthesis(
    tmp_path: Path,
) -> None:
    sid = "fetch-url-trailing-parenthesis-dedupe"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/path?x=a)"
    truncated_url = "https://docs.example.com/path?x=a"
    store.append(
        "user_message",
        {
            "content": (
                f"Please read [docs]({clean_url}) and then {clean_url} again before changes."
            )
        },
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": clean_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Docs",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()

    assert payload["deduped_normalized_user_urls"] == [clean_url]
    assert payload["deduped_normalized_fetch_urls"] == [clean_url]
    assert truncated_url not in payload["deduped_normalized_user_urls"]
    assert payload["fetches"][0]["requested_url"] == clean_url
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 0


def test_session_store_web_artifact_preserves_bracket_query_url_without_truncated_entry(
    tmp_path: Path,
) -> None:
    sid = "fetch-url-bracket-query"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/path?foo[bar]=1"
    truncated_url = "https://docs.example.com/path?foo"
    store.append(
        "user_message",
        {"content": f"Please inspect {clean_url} and `https://docs.example.com/path?arr[]=1`."},
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": clean_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Docs",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()

    assert payload["deduped_normalized_user_urls"] == [
        clean_url,
        "https://docs.example.com/path?arr[]=1",
    ]
    assert payload["deduped_normalized_fetch_urls"] == [clean_url]
    assert truncated_url not in payload["deduped_normalized_user_urls"]
    assert payload["fetches"][0]["requested_url"] == clean_url
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 0


def test_session_store_web_artifact_preserves_exclamation_url_without_truncated_entry(
    tmp_path: Path,
) -> None:
    sid = "fetch-url-exclamation"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    clean_url = "https://docs.example.com/Yahoo!"
    truncated_url = "https://docs.example.com/Yahoo"
    store.append(
        "user_message",
        {"content": f"Please inspect {clean_url} before changes."},
    )
    store.append(
        "tool_call",
        {"name": "web_fetch", "step": 1, "arguments": {"url": clean_url}},
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 1,
            "result": {
                "url": clean_url,
                "final_url": clean_url,
                "status_code": 200,
                "content_type": "text/html",
                "title": "Yahoo",
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    metrics = store.web_research_metrics_payload()

    assert payload["deduped_normalized_user_urls"] == [clean_url]
    assert payload["deduped_normalized_fetch_urls"] == [clean_url]
    assert truncated_url not in payload["deduped_normalized_user_urls"]
    assert payload["fetches"][0]["requested_url"] == clean_url
    assert metrics["unique_web_fetch_urls"] == 1
    assert metrics["duplicate_web_fetches"] == 0


def test_session_store_preserves_parenthesized_web_search_source_url(
    tmp_path: Path,
) -> None:
    sid = "parenthesized-search-source"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    store.append(
        "tool_result",
        {
            "name": "web_search",
            "step": 1,
            "result": {
                "query": "function mathematics docs",
                "backend": "openai_responses",
                "sources": [
                    {
                        "title": "Function (mathematics)",
                        "url": "https://docs.example.com/Function_(mathematics)",
                        "snippet": "Balanced parenthesized path",
                    }
                ],
            },
        },
    )

    payload = store.web_research_artifact_payload()

    assert payload["deduped_normalized_search_source_urls"] == [
        "https://docs.example.com/Function_(mathematics)"
    ]
    assert payload["searches"][0]["returned_sources"][0]["normalized_url"] == (
        "https://docs.example.com/Function_(mathematics)"
    )


def test_session_store_classifies_dashscope_parenthesized_search_url_as_returned_by_web_search(
    tmp_path: Path,
) -> None:
    sid = "dashscope-parenthesized-search-source"
    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    url = "https://docs.example.com/Function_(mathematics)"
    store.append(
        "tool_result",
        {
            "name": "web_search",
            "step": 1,
            "result": {
                "query": "function mathematics docs",
                "backend": "dashscope_chat",
                "sources": [
                    {
                        "title": "Function (mathematics)",
                        "url": url,
                        "snippet": "Balanced parenthesized path",
                    }
                ],
            },
        },
    )
    store.append(
        "tool_call",
        {
            "name": "web_fetch",
            "step": 2,
            "arguments": {"url": url},
        },
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 2,
            "result": {
                "url": url,
                "final_url": url,
                "status_code": 200,
                "backend": "httpx",
            },
        },
    )

    payload = store.web_research_artifact_payload()
    fetch = payload["fetches"][0]

    assert fetch["requested_url"] == url
    assert fetch["normalized_requested_url"] == url
    assert fetch["provenance_classification"] == "returned_by_web_search"
    assert payload["deduped_normalized_search_source_urls"] == [url]


def test_session_store_reopen_merges_newer_web_artifact_ahead_of_log(
    tmp_path: Path,
) -> None:
    sid = "resume-artifact-ahead"
    logged_events: list[dict[str, object]] = [
        {
            "type": "tool_result",
            "session_id": sid,
            "payload": {
                "name": "web_search",
                "step": 1,
                "result": {
                    "query": "docs example start",
                    "backend": "openai_responses",
                    "sources": [
                        {
                            "title": "Start",
                            "url": "https://docs.example.com/start",
                            "snippet": "Initial source",
                        }
                    ],
                },
            },
        }
    ]
    artifact_only_events: list[dict[str, object]] = [
        {
            "type": "tool_result",
            "session_id": sid,
            "payload": {
                "name": "web_search",
                "step": 2,
                "result": {
                    "query": "docs example guide",
                    "backend": "openai_responses",
                    "sources": [
                        {
                            "title": "Guide",
                            "url": "https://docs.example.com/guide",
                            "snippet": "Artifact-only source",
                        }
                    ],
                },
            },
        }
    ]
    _write_web_artifact_ahead_of_log(
        sessions_dir=tmp_path,
        session_id=sid,
        logged_events=logged_events,
        artifact_only_events=artifact_only_events,
    )

    reopened = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )

    assert (
        reopened.classify_web_fetch_url("https://docs.example.com/start")
        == "returned_by_web_search"
    )
    assert (
        reopened.classify_web_fetch_url("https://docs.example.com/guide")
        == "returned_by_web_search"
    )
    payload = reopened.web_research_artifact_payload()
    assert payload["deduped_normalized_search_source_urls"] == [
        "https://docs.example.com/start",
        "https://docs.example.com/guide",
    ]
    assert [entry["normalized_query"] for entry in payload["searches"]] == [
        "docs example start",
        "docs example guide",
    ]


def test_session_store_artifact_only_reopen_preserves_search_url_classification(
    tmp_path: Path,
) -> None:
    sid = "resume-artifact-only"
    artifact_root = tmp_path / sid
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_payload = build_web_research_artifact_from_events(
        [
            {
                "type": "tool_result",
                "session_id": sid,
                "payload": {
                    "name": "web_search",
                    "step": 1,
                    "result": {
                        "query": "docs example",
                        "backend": "openai_responses",
                        "sources": [
                            {
                                "title": "Guide",
                                "url": "https://docs.example.com/guide",
                                "snippet": "Artifact-only source",
                            }
                        ],
                    },
                },
            }
        ]
    )
    (artifact_root / "web_research_sources.json").write_text(
        json.dumps(artifact_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    reopened = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )

    assert (
        reopened.classify_web_fetch_url("https://docs.example.com/guide")
        == "returned_by_web_search"
    )


def test_session_store_reopen_rewrites_cumulative_artifact_when_newer_artifact_preexisted(
    tmp_path: Path,
) -> None:
    sid = "resume-artifact-ahead-append"
    logged_events: list[dict[str, object]] = [
        {
            "type": "user_message",
            "session_id": sid,
            "payload": {"content": "Please inspect https://docs.example.com/spec"},
        }
    ]
    artifact_only_events: list[dict[str, object]] = [
        {
            "type": "tool_result",
            "session_id": sid,
            "payload": {
                "name": "web_search",
                "step": 2,
                "result": {
                    "query": "docs example guide",
                    "backend": "openai_responses",
                    "sources": [
                        {
                            "title": "Guide",
                            "url": "https://docs.example.com/guide",
                            "snippet": "Artifact-only source",
                        }
                    ],
                },
            },
        }
    ]
    _write_web_artifact_ahead_of_log(
        sessions_dir=tmp_path,
        session_id=sid,
        logged_events=logged_events,
        artifact_only_events=artifact_only_events,
    )

    reopened = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=tmp_path,
        session_id=sid,
        cwd=".",
        repo_root=".",
    )
    reopened.append(
        "tool_call",
        {"name": "web_fetch", "step": 3, "arguments": {"url": "https://docs.example.com/guide"}},
    )
    reopened.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 3,
            "result": {
                "url": "https://docs.example.com/guide",
                "final_url": "https://docs.example.com/final",
                "status_code": 200,
                "content_type": "text/html",
                "title": "Final",
                "backend": "httpx",
            },
        },
    )
    reopened.close()

    payload = json.loads((tmp_path / sid / "web_research_sources.json").read_text(encoding="utf-8"))
    assert payload["deduped_normalized_user_urls"] == ["https://docs.example.com/spec"]
    assert payload["deduped_normalized_search_source_urls"] == ["https://docs.example.com/guide"]
    assert payload["deduped_normalized_fetch_urls"] == ["https://docs.example.com/guide"]
    assert payload["deduped_normalized_final_fetch_urls"] == ["https://docs.example.com/final"]
    assert [entry["normalized_query"] for entry in payload["searches"]] == ["docs example guide"]
