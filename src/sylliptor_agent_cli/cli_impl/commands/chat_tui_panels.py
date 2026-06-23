"""TUI-native panel spec builders for chat slash commands.

Each builder mirrors a classic ``_print_*`` / ``_chat_*_panel`` command but
returns a structured *spec* the full-screen TUI renders as a centered popup
panel (via ``_render_kv_panel_rows`` in :mod:`...cli_impl.tui.app`) instead of
dumping flat, colourless captured Rich text into the transcript.

A panel spec is::

    {"title": str, "hint"?: str, "sections": [(section_name, rows)]}

where ``rows`` is a list of ``(key, value, tone)`` tuples. ``tone`` selects the
value colour: ``"accent"`` (on/healthy, green), ``"plain"`` (neutral),
``"warn"`` (amber), ``"err"`` (red), ``"dim"`` (muted).

Unlike the sibling legacy split modules, this one imports every helper it needs
*explicitly* (no ``from .cli_common import *`` global injection), so the builders
are importable and unit-testable in isolation with a plain fake session.
"""

from __future__ import annotations

from typing import Any

# Canonical, self-contained helpers (defined directly in these modules — not
# injected at runtime), so the imports resolve at import time and in tests.
from .startup import (
    _chat_context_percent_value,
    _chat_usage_hud_enabled,
    _format_chat_context_percent,
    _format_exact_token_count,
    _format_usage_cost_for_display,
    _format_usage_source_for_display,
    _known_cost_value,
    _refresh_chat_hud_context_cache,
    _session_skill_listing,
)

PanelSpec = dict[str, Any]


def _format_terminal_runtime_s(runtime_s: float) -> str:
    """Mirror of the classic terminal runtime formatter (kept local so this
    module never imports the injected ``chat/commands.py`` legacy module)."""
    if runtime_s < 0:
        runtime_s = 0.0
    if runtime_s < 60:
        return f"{runtime_s:.1f}s"
    minutes = int(runtime_s // 60)
    seconds = int(runtime_s % 60)
    return f"{minutes}m{seconds:02d}s"


# ------------------------------------------------------------------- /usage
def _chat_usage_panel_spec(*, session: Any) -> PanelSpec:
    """Token count & cost as a panel (mirrors ``_print_chat_usage``)."""
    summary = getattr(session, "usage_summary", None)
    if summary is None:
        return {
            "title": "Usage",
            "sections": [
                ("Usage", [("status", "Usage tracking unavailable for this session.", "plain")])
            ],
        }
    try:
        rows = summary.by_model_rows()
    except Exception:  # noqa: BLE001 - never crash the UI on a panel
        rows = []
    if not rows:
        hud = "on" if _chat_usage_hud_enabled(session) else "off"
        return {
            "title": "Usage",
            "sections": [
                (
                    "Usage",
                    [
                        ("status", "No usage events yet in this session.", "plain"),
                        ("hud", hud, "accent" if hud == "on" else "plain"),
                    ],
                )
            ],
            "hint": "/usage hud on|off to toggle the toolbar HUD · Esc close",
        }

    # Index keys (not the model name) so a long, namespaced model id never blows
    # past the panel's narrow key column — the full name lives in the wrap-safe
    # value column instead.
    model_rows: list[tuple[str, str, str]] = []
    for idx, row in enumerate(rows, start=1):
        unknown = int(row.get("unknown_cost_count") or 0)
        cost = _format_usage_cost_for_display(
            known_cost=_known_cost_value(row), unknown_calls=unknown
        )
        source = _format_usage_source_for_display(row)
        value = (
            f"{row.get('model') or '-'}   "
            f"↓ {_format_exact_token_count(row.get('prompt_tokens'))}  "
            f"↑ {_format_exact_token_count(row.get('completion_tokens'))}  "
            f"Σ {_format_exact_token_count(row.get('total_tokens'))}   "
            f"{cost}  ({source})"
        )
        model_rows.append((str(idx), value, "plain"))

    totals = summary.totals()
    unknown_total = int(totals.get("unknown_cost_calls") or 0)
    total_cost = _format_usage_cost_for_display(
        known_cost=_known_cost_value(totals),
        unknown_calls=unknown_total,
    )
    # Only paint the cost green when it is fully known; an unknown/partial total
    # stays neutral (never reads as healthy) — matching the classic "[yellow]Total
    # cost is partial" warning.
    cost_tone = (
        "accent" if (_known_cost_value(totals) is not None and unknown_total == 0) else "plain"
    )
    total_rows: list[tuple[str, str, str]] = [
        ("tokens", _format_exact_token_count(totals.get("total_tokens")), "accent"),
        ("input", _format_exact_token_count(totals.get("prompt_tokens")), "plain"),
        ("output", _format_exact_token_count(totals.get("completion_tokens")), "plain"),
        ("cost", total_cost, cost_tone),
    ]
    if unknown_total > 0:
        total_rows.append(
            ("note", f"{unknown_total} call(s) unmetered (missing pricing metadata)", "warn")
        )
    corrected_total = int(totals.get("corrected_usage_calls") or 0)
    if corrected_total > 0:
        total_rows.append(
            ("note", f"{corrected_total} provider usage record(s) corrected before display", "warn")
        )
    return {
        "title": "Usage",
        "sections": [("Per model", model_rows), ("Total", total_rows)],
    }


# --------------------------------------------------------------- /ctx /context
def _chat_context_panel_spec(*, session: Any) -> PanelSpec:
    """Context-window usage as a panel (mirrors the primary ``/ctx`` table)."""
    if not callable(getattr(session, "context_left", None)):
        return {
            "title": "Context Window",
            "sections": [
                ("Context", [("status", "Context tracking unavailable for this session.", "plain")])
            ],
        }
    _refresh_chat_hud_context_cache(session)
    ctx = getattr(session, "_hud_context_cache", None)
    if ctx is None:
        return {
            "title": "Context Window",
            "sections": [
                ("Context", [("status", "Context tracking unavailable for this session.", "plain")])
            ],
        }
    context_window = getattr(ctx, "context_window_tokens", getattr(ctx, "max_input_tokens", "n/a"))
    remaining = getattr(
        ctx, "context_window_remaining_tokens", getattr(ctx, "remaining_tokens", "n/a")
    )
    percent = getattr(ctx, "context_window_percent_left", getattr(ctx, "percent_left", None))
    used = getattr(ctx, "used_input_tokens", "n/a")

    # Colour the percent from the SAME metric we display (the context-window
    # percent), not the dynamic-budget percent — otherwise a healthy 60%-left
    # window could be painted red because the baseline-subtracted dynamic value
    # is much lower. Fall back to the session helper only when the window value
    # is missing.
    displayed_percent = percent
    if displayed_percent is None:
        displayed_percent = _chat_context_percent_value(session)
    try:
        percent_float = float(displayed_percent) if displayed_percent is not None else None
    except (TypeError, ValueError):
        percent_float = None
    if percent_float is None:
        pct_tone = "plain"
    elif percent_float < 10.0:
        pct_tone = "err"
    elif percent_float < 25.0:
        pct_tone = "warn"
    else:
        pct_tone = "accent"
    percent_display = _format_chat_context_percent(displayed_percent)

    rows: list[tuple[str, str, str]] = [
        ("model", str(getattr(ctx, "model_name", "-")), "plain"),
        ("source", str(getattr(ctx, "source", "-")), "dim"),
        ("context window", str(context_window), "plain"),
        ("used (est.)", str(used), "plain"),
        ("left tokens", str(remaining), "plain"),
        ("left %", percent_display, pct_tone),
    ]
    return {
        "title": "Context Window",
        "sections": [("Context", rows)],
        "hint": "/compact to reduce context · Esc close",
    }


# ----------------------------------------------------------------- /model-info
def _chat_model_info_panel_spec(*, session: Any, model_ref: str = "") -> PanelSpec:
    """Resolved model metadata as a panel (mirrors ``/model-info``)."""
    model_name = (model_ref or "").strip() or str(
        getattr(getattr(session, "client", None), "model", "") or ""
    ).strip()
    if not model_name:
        return {
            "title": "Model Info",
            "sections": [
                ("Model", [("status", "Model info unavailable (missing model name).", "plain")])
            ],
        }
    registry = getattr(session, "model_registry", None)
    if registry is None:
        return {
            "title": "Model Info",
            "sections": [
                ("Model", [("status", "Model info unavailable for this session.", "plain")])
            ],
        }
    try:
        meta = registry.get(model_name)
    except Exception as exc:  # noqa: BLE001 - surface a resolve failure inline
        return {
            "title": f"Model Info ({model_name})",
            "sections": [("Model", [("error", str(exc), "err")])],
        }

    field_sources = getattr(meta, "field_sources", {}) or {}

    def _src(key: str) -> str:
        return str(field_sources.get(key, "unknown"))

    rows: list[tuple[str, str, str]] = [
        ("resolved", str(getattr(meta, "model_name", model_name)), "accent"),
        ("source", str(getattr(meta, "source", "unknown")), "plain"),
        (
            "context window",
            f"{getattr(meta, 'context_window_tokens', '-')} ({_src('context_window_tokens')})",
            "plain",
        ),
        (
            "max output",
            f"{getattr(meta, 'max_output_tokens', '-')} ({_src('max_output_tokens')})",
            "plain",
        ),
        (
            "vision",
            f"{getattr(meta, 'supports_vision', '-')} ({_src('supports_vision')})",
            "plain",
        ),
        (
            "input $/tok",
            f"{getattr(meta, 'input_cost_per_token', '-')} ({_src('input_cost_per_token')})",
            "plain",
        ),
        (
            "output $/tok",
            f"{getattr(meta, 'output_cost_per_token', '-')} ({_src('output_cost_per_token')})",
            "plain",
        ),
    ]
    last_error = getattr(registry, "last_error", None)
    if last_error:
        rows.append(("registry error", str(last_error), "err"))
    warnings = "; ".join(getattr(meta, "warnings", ())[:5])
    if warnings:
        rows.append(("warnings", warnings, "warn"))
    return {"title": f"Model Info ({model_name})", "sections": [("Model", rows)]}


# -------------------------------------------------------------------- /config
def _chat_config_panel_spec(*, session: Any) -> PanelSpec:
    """Tracked-model config as a panel (mirrors ``_chat_config_panel``).

    The interactive config menu cannot run inside the alt-screen, so the bare
    ``/config`` shows this read-only panel; ``/config set|clear`` still apply via
    the command runner.
    """
    from .chat_status import _chat_used_models  # self-contained getter

    models = _chat_used_models(session)
    registry = getattr(session, "model_registry", None)

    model_rows: list[tuple[str, str, str]] = []
    if models:
        for idx, name in enumerate(models, start=1):
            summary = ""
            if registry is not None:
                try:
                    meta = registry.get(name)
                except Exception as exc:  # noqa: BLE001
                    summary = f"error={exc}"
                else:
                    field_sources = getattr(meta, "field_sources", {}) or {}
                    src = field_sources.get(
                        "context_window_tokens", getattr(meta, "source", "unknown")
                    )
                    summary = (
                        f"ctx={getattr(meta, 'context_window_tokens', '-')}  "
                        f"out={getattr(meta, 'max_output_tokens', '-')}  src={src}"
                    )
            value = f"{name}   {summary}".rstrip() if summary else str(name)
            model_rows.append((str(idx), value, "accent" if idx == 1 else "plain"))
    else:
        model_rows.append(("models", "No tracked models in this session yet.", "plain"))

    usage_rows: list[tuple[str, str, str]] = [
        ("set", "/config set <model|index> <ctx> <max_out> [vision] [in$] [out$]", "dim"),
        ("clear", "/config clear <model|index>", "dim"),
        ("example", "/config set 1 128000 4096 false", "dim"),
    ]
    return {
        "title": "Model Config",
        "sections": [("Tracked models", model_rows), ("Manage", usage_rows)],
    }


# ------------------------------------------------------------------- /toolbar
def _chat_toolbar_panel_spec(*, session: Any) -> PanelSpec:
    """Status-toolbar items as a panel (mirrors bare ``/toolbar``)."""
    from .cli_common import (
        _CHAT_TOOLBAR_ITEM_ORDER,
        _DEFAULT_TOOLBAR_ITEMS,
        _VALID_TOOLBAR_ITEMS,
    )

    cfg_obj = getattr(session, "cfg", None)
    raw_items = getattr(cfg_obj, "toolbar_items", None) if cfg_obj is not None else None
    active: list[str] = []
    seen: set[str] = set()
    for raw_item in list(raw_items or list(_DEFAULT_TOOLBAR_ITEMS)):
        item = str(raw_item).strip().lower()
        if not item or item in seen or item not in _VALID_TOOLBAR_ITEMS:
            continue
        seen.add(item)
        active.append(item)
    available = [item for item in _CHAT_TOOLBAR_ITEM_ORDER if item not in set(active)]

    item_rows: list[tuple[str, str, str]] = [
        ("active", ", ".join(active) if active else "(none)", "accent"),
        ("available", ", ".join(available) if available else "(none)", "plain"),
    ]
    manage_rows: list[tuple[str, str, str]] = [
        ("add", "/toolbar add <item>", "dim"),
        ("remove", "/toolbar remove <item>", "dim"),
        ("reset", "/toolbar reset", "dim"),
        ("save", "/toolbar save", "dim"),
    ]
    return {
        "title": "Toolbar Items",
        "sections": [("Items", item_rows), ("Manage", manage_rows)],
    }


# ----------------------------------------------------------------- /terminals
def _chat_terminals_panel_spec(*, session: Any) -> PanelSpec:
    """Background processes as a panel (mirrors ``/terminals list``)."""
    terminal_manager = getattr(session, "terminal_manager", None)
    if terminal_manager is None:
        return {
            "title": "Background Terminals",
            "sections": [
                (
                    "Terminals",
                    [("status", "Background terminals are unavailable in this session.", "plain")],
                )
            ],
        }
    try:
        summaries = terminal_manager.list()
    except Exception as exc:  # noqa: BLE001
        return {
            "title": "Background Terminals",
            "sections": [("Terminals", [("error", str(exc), "err")])],
        }
    if not summaries:
        process_rows: list[tuple[str, str, str]] = [("status", "No background processes.", "plain")]
    else:
        process_rows = []
        for summary in summaries:
            status = str(getattr(summary, "status", "-"))
            status_lc = status.casefold()
            if status_lc in {"failed", "error"}:
                tone = "err"
            elif status_lc == "running":
                tone = "accent"
            else:
                tone = "plain"
            exit_code = getattr(summary, "exit_code", None)
            runtime = _format_terminal_runtime_s(float(getattr(summary, "runtime_s", 0) or 0))
            value = (
                f"{status}  ·  exit {('-' if exit_code is None else exit_code)}  "
                f"·  {runtime}  ·  {getattr(summary, 'cmd', '')}"
            )
            process_rows.append((str(getattr(summary, "process_id", "-")), value, tone))
    manage_rows: list[tuple[str, str, str]] = [
        ("show", "/terminals show <id>", "dim"),
        ("kill", "/terminals kill <id>", "dim"),
    ]
    return {
        "title": "Background Terminals",
        "sections": [("Processes", process_rows), ("Manage", manage_rows)],
    }


# --------------------------------------------------------------------- /skill
def _chat_skill_listing_panel_spec(*, session: Any) -> PanelSpec:
    """Discovered skills as a panel (mirrors bare ``/skill``)."""
    enabled, ordered, issues = _session_skill_listing(session)
    if not enabled:
        return {
            "title": "Skills",
            "sections": [
                ("Skills", [("status", "Skills are disabled for this session config.", "plain")])
            ],
        }
    if not ordered:
        return {
            "title": "Skills",
            "sections": [
                (
                    "Skills",
                    [("status", "No skills discovered in the supported skill roots.", "plain")],
                )
            ],
        }
    skill_rows = [
        (str(getattr(skill, "name", "")), str(getattr(skill, "description", "") or ""), "plain")
        for skill in ordered
    ]
    sections: list[tuple[str, list[tuple[str, str, str]]]] = [
        (f"Skills ({len(ordered)})", skill_rows)
    ]
    issue_rows: list[tuple[str, str, str]] = []
    for issue in issues:
        source_path = getattr(issue, "source_path", None)
        message = str(getattr(issue, "message", "") or "").strip()
        if source_path is not None and message:
            issue_rows.append((str(source_path), message, "warn"))
    if issue_rows:
        sections.append(("Skipped", issue_rows))
    return {
        "title": "Skills",
        "sections": sections,
        "hint": "/skill <name> for info · /skill <name> <task> to attach · Esc close",
    }


# ------------------------------------------------------------------ /subagent
# Leading boilerplate stripped from a subagent description before summarising
# (longest-first so the most specific prefix wins). The blurbs all open with some
# variant of "Use this when you need to …" / "Catch-all subagent for …", which
# carries no information for a one-line picker label.
_SUBAGENT_DESC_BOILERPLATE = (
    "use this agent when you need to ",
    "use this agent when you need ",
    "use this agent when ",
    "use this when you need to ",
    "use this when you need an ",
    "use this when you need a ",
    "use this when you need ",
    "use this when answering means ",
    "use this when ",
    "use this for ",
    "use for ",
    "catch-all subagent for ",
    "catch-all agent for ",
    "catch-all for ",
)


def _short_subagent_desc(text: str, limit: int = 46) -> str:
    """Condense a (potentially huge) subagent description to a short, scannable
    summary shown to the right of the option in the spawn picker.

    Collapses whitespace, strips the boilerplate "Use this when you need to …"
    lead-in, keeps only the first clause (up to the first ``:`` / ``.`` / dash),
    capitalises it, and hard-truncates with an ellipsis so each option stays a
    single readable line.
    """
    compact = " ".join(str(text or "").split())
    if not compact:
        return ""
    lowered = compact.lower()
    for prefix in _SUBAGENT_DESC_BOILERPLATE:
        if lowered.startswith(prefix):
            compact = compact[len(prefix) :]
            break
    # Keep only the first clause/sentence — the crisp "what it does".
    cut = len(compact)
    for sep in (": ", ". ", "; ", " — ", " - "):
        idx = compact.find(sep)
        if idx != -1:
            cut = min(cut, idx)
    summary = compact[:cut].rstrip(" ,;:.—-")
    if summary:
        summary = summary[0].upper() + summary[1:]
    if len(summary) > limit:
        head = summary[: max(1, limit - 1)]
        # Prefer breaking on a word boundary so we never clip mid-word ("recent c…").
        space = head.rfind(" ")
        if space >= limit // 2:
            head = head[:space]
        summary = head.rstrip(" ,;:.—-") + "…"
    return summary


# --------------------------------------------------------------------- Forge
# Task-status buckets — kept EXACTLY in sync with cli_common._forge_task_status_counts
# (the authority for the done/failed/remaining summary) so a row's tone never
# contradicts the count: done = "done", failure = the same 7 states, obsolete = 2.
_FORGE_FAILURE_STATES = {
    "failed",
    "verify_failed",
    "candidate_rejected",
    "changes_requested",
    "merge_conflict",
    "blocked_integration",
    "blocked",
}
_FORGE_OBSOLETE_STATES = {"superseded", "invalidated"}


def _forge_status_tone(canonical: str) -> str:
    """Map a canonical task status to a panel value tone (accent/err/dim/plain)."""
    if canonical == "done":
        return "accent"
    if canonical in _FORGE_FAILURE_STATES:
        return "err"
    if canonical in _FORGE_OBSOLETE_STATES:
        return "dim"
    return "plain"


# The "how Forge works" guidance shown in the centered popup when the user runs a
# plain ``/forge``. Markdown so the doc panel renders headings/lists; the closing
# call-to-action pairs with the panel's "Enter to begin · Esc to cancel" hint.
_FORGE_INTRO_BODY = """\
# Forge — autonomous build mode

Forge turns a goal into a reviewed plan, then runs a **swarm of agents** to build
it task by task. You stay in control — nothing is built until you say so.

## How it works

1. **Set the goal** — describe what you want built (just type it, or `/goal`).
2. **Shape the plan** — add or refine tasks with `/task`, or let the planner
   assistant draft them for you.
3. **Review** — inspect the plan and task table with `/show` and `/plan`.
4. **Execute** — `/execute plan` runs the swarm to build every task, live.
5. **Finish** — `/done` (or `/back`) returns you to normal chat.

## Handy commands

- `/show` — plan summary: goal, requirements, tasks
- `/plan` — view or edit the plan (tasks · markdown · edit)
- `/assets` — attach and review reference files
- `/execute plan` — run the build
- `/done` · `/back` — leave Forge

Press **Enter** to start a planning session, or **Esc** to stay in chat."""


def _chat_forge_intro_panel_spec() -> PanelSpec:
    """Guidance popup shown for a plain ``/forge`` before entering the session.

    Returns a document-panel spec (rendered through the TUI markdown pipeline)
    carrying a ``confirm`` command: the centered popup explains how Forge works
    and, on Enter, routes ``/forge`` to the command runner to actually enter the
    planning session; Esc cancels. ``/forge resume`` and re-entry while already in
    Forge skip this intro (handled by the caller's gating).
    """
    return {
        "title": "Forge",
        "body": _FORGE_INTRO_BODY,
        "hint": "↵ Enter to begin  ·  Esc to cancel",
        "confirm": "/forge",
    }


def _chat_forge_plan_panel_spec(*, paths: Any, plan: dict[str, Any]) -> PanelSpec:
    """Forge plan summary as a panel — mirrors ``_show_forge_plan_summary``.

    Renders the goal/summary + per-task table that the classic ``/show`` and
    ``/plan tasks`` print, as a centered popup instead of a flat Rich table dump.
    Used in Forge mode for ``/show`` and ``/plan tasks|table|view``.
    """
    from ...forge import requirement_text
    from ...swarm_scheduler import canonical_task_status
    from .cli_common import _forge_task_mcp_summary_label, _forge_task_status_counts
    from .forge_asset_view import forge_asset_view_count

    goal = str(plan.get("project_goal") or "").strip() or "(not set)"
    summary = str(plan.get("summary") or "").strip() or "(not set)"
    tasks_obj = plan.get("tasks") or []
    tasks = tasks_obj if isinstance(tasks_obj, list) else []
    task_count = len(tasks)
    try:
        asset_count = forge_asset_view_count(paths, plan)
    except Exception:  # noqa: BLE001 - never crash the UI on a panel
        asset_count = 0
    try:
        done, failed, remaining = _forge_task_status_counts(plan)
    except Exception:  # noqa: BLE001
        done = failed = remaining = 0

    overview: list[tuple[str, str, str]] = [
        ("run", str(getattr(paths, "run_id", "-")), "plain"),
        ("goal", goal, "plain"),
        ("summary", summary, "plain"),
        (
            "tasks",
            f"{task_count} total · {done} done · {failed} failed · {remaining} remaining",
            "accent" if (task_count and failed == 0) else "plain",
        ),
        ("assets", str(asset_count), "plain"),
    ]
    sections: list[tuple[str, list[tuple[str, str, str]]]] = [("Plan", overview)]

    requirements_obj = plan.get("requirements") or []
    requirements = requirements_obj if isinstance(requirements_obj, list) else []
    if requirements:
        req_rows: list[tuple[str, str, str]] = []
        for idx, req in enumerate(requirements[:12], start=1):
            text = " ".join(str(requirement_text(req) or "").split())
            if not text:
                continue
            req_rows.append((str(idx), text, "plain"))
        if len(requirements) > 12:
            req_rows.append(("…", f"({len(requirements) - 12} more)", "dim"))
        if req_rows:
            sections.append((f"Requirements ({len(requirements)})", req_rows))

    if tasks:
        task_rows: list[tuple[str, str, str]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = str(task.get("id") or "-")
            raw_status = str(task.get("status") or "planned")
            tone = _forge_status_tone(canonical_task_status(raw_status))
            title = str(task.get("title") or "-")
            value = f"{raw_status}  ·  {title}"
            extra: list[str] = []
            deps = task.get("dependencies") or []
            if isinstance(deps, list) and deps:
                extra.append("deps: " + ", ".join(str(dep) for dep in deps))
            try:
                mcp = _forge_task_mcp_summary_label(task)
            except Exception:  # noqa: BLE001
                mcp = "off"
            if mcp and mcp != "off":
                extra.append(f"mcp: {mcp}")
            if extra:
                value += "  ·  " + " · ".join(extra)
            task_rows.append((tid, value, tone))
        sections.append((f"Tasks ({task_count})", task_rows))
    else:
        sections.append(("Tasks", [("-", "(no tasks yet)", "dim")]))

    return {
        "title": f"Forge Plan · {getattr(paths, 'run_id', '')}".rstrip(" ·"),
        "sections": sections,
        "hint": "/execute plan to run · /plan markdown for PLAN.md · Esc close",
    }


def _chat_forge_markdown_panel_spec(*, paths: Any, plan: dict[str, Any]) -> PanelSpec:
    """PLAN.md preview as a scrollable doc panel — replaces the classic pager.

    Saves the in-memory plan first (so PLAN.md reflects edits, like the classic
    ``_show_forge_plan_markdown``), reads PLAN.md, and returns a ``body`` spec the
    TUI renders through its markdown renderer in the centered popup.
    """
    from ...forge import save_plan

    try:
        save_plan(paths, plan)
    except Exception:  # noqa: BLE001 - still try to show whatever is on disk
        pass
    plan_md_path = getattr(paths, "plan_md_path", None)
    body = ""
    if plan_md_path is not None:
        try:
            body = plan_md_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return {
                "title": "PLAN.md",
                "sections": [("PLAN.md", [("error", f"Failed to read PLAN.md: {exc}", "err")])],
            }
    if not body.strip():
        body = "(PLAN.md is empty)"
    return {
        "title": f"PLAN.md · {getattr(paths, 'run_id', '')}".rstrip(" ·"),
        "body": body,
        "hint": "↑↓/PgUp/PgDn scroll · Esc close",
    }


# ---------------------------------------------------------------- Forge assets
def _format_asset_size(size_bytes: int) -> str:
    """Human-readable byte size (mirror of assets_modal._format_size)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _asset_status_tone(status: str) -> str:
    s = str(status or "").strip().lower()
    if s == "ready":
        return "accent"
    if s == "failed":
        return "err"
    if s in {"pending", "running"}:
        return "warn"
    return "plain"


def _chat_assets_picker_spec(*, cfg: Any, paths: Any) -> dict[str, Any] | None:
    """Selectable list of Forge assets — replaces the dead stdin assets modal.

    Returns ``None`` (so the caller falls through) when the asset surface can't be
    built or there are no assets to pick. Choosing a row submits ``/assets <id>``
    to open the detail panel."""
    from ...assets.surface import build_asset_surface
    from ...forge import load_plan

    try:
        load_plan(paths)
    except Exception:  # noqa: BLE001 - listing still works without a loaded plan
        pass
    try:
        surface = build_asset_surface(cfg=cfg, run_paths=paths)
        entries = surface.list_assets()
    except Exception:  # noqa: BLE001
        return None
    if not entries:
        return None
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rec = entry.record
        kind = "img" if str(getattr(rec, "kind", "")) == "image" else "txt"
        size = _format_asset_size(int(getattr(rec, "size_bytes", 0) or 0))
        status = str(getattr(entry, "comprehension_status", "") or "")
        pin = "★ " if getattr(rec, "pinned", False) else ""
        rows.append(
            {
                "label": str(rec.id),
                "description": f"{pin}{rec.title}  ·  {kind} · {status} · {size}",
                "value": str(rec.id),
                "current": False,
            }
        )
    return {
        "title": f"Assets ({len(rows)})",
        "rows": rows,
        "on_select": lambda value: {"submit": f"/assets {value}"},
        "hint": "↑↓ select · Enter to view · Esc cancel",
    }


def _chat_asset_detail_panel_spec(*, cfg: Any, paths: Any, asset_id: str) -> PanelSpec:
    """One asset's metadata + comprehension summary as a panel (replaces the
    interactive modal's detail view)."""
    from ...assets import AssetError
    from ...assets.surface import build_asset_surface

    try:
        surface = build_asset_surface(cfg=cfg, run_paths=paths)
        detail = surface.show_asset(asset_id)
    except AssetError as exc:
        return {
            "title": f"Asset · {asset_id}",
            "sections": [("Asset", [("error", str(exc), "err")])],
        }
    except Exception as exc:  # noqa: BLE001
        return {"title": "Asset", "sections": [("Asset", [("error", str(exc), "err")])]}

    rec = detail.record
    deleted = getattr(rec, "deleted_at", None) is not None
    status = "deleted" if deleted else str(getattr(detail, "comprehension_status", "") or "")
    overview: list[tuple[str, str, str]] = [
        ("id", str(rec.id), "accent"),
        ("title", str(rec.title), "plain"),
        ("kind", str(getattr(rec, "kind", "-")), "plain"),
        ("size", _format_asset_size(int(getattr(rec, "size_bytes", 0) or 0)), "plain"),
        ("mime", str(getattr(rec, "mime", "-")), "plain"),
        (
            "pinned",
            "yes" if getattr(rec, "pinned", False) else "no",
            "accent" if getattr(rec, "pinned", False) else "plain",
        ),
        ("status", status, "err" if deleted else _asset_status_tone(status)),
        ("file", str(getattr(rec, "original_filename", "-")), "dim"),
    ]
    sections: list[tuple[str, list[tuple[str, str, str]]]] = [("Asset", overview)]
    description = str(getattr(rec, "description", "") or "").strip()
    if description:
        sections.append(("Description", [("text", description, "plain")]))
    comp = getattr(detail, "comprehension", None)
    if comp is not None:
        data = getattr(comp, "data", None)
        comp_rows: list[tuple[str, str, str]] = [
            ("summary", str(getattr(data, "semantic_summary", "") or "-"), "plain"),
            ("language", str(getattr(comp, "detected_language", "") or "-"), "dim"),
            ("source", str(getattr(comp, "source", "") or "-"), "dim"),
        ]
        sections.append(("Comprehension", comp_rows))
        facts = list(getattr(data, "stated_facts", None) or [])
        if facts:
            sections.append(
                (
                    "Stated facts",
                    [(str(i + 1), str(fact), "plain") for i, fact in enumerate(facts[:8])],
                )
            )
    return {"title": f"Asset · {rec.id}", "sections": sections, "hint": "Esc close"}


__all__ = [
    "_chat_usage_panel_spec",
    "_chat_context_panel_spec",
    "_chat_model_info_panel_spec",
    "_chat_config_panel_spec",
    "_chat_toolbar_panel_spec",
    "_chat_terminals_panel_spec",
    "_chat_skill_listing_panel_spec",
    "_short_subagent_desc",
    "_forge_status_tone",
    "_chat_forge_intro_panel_spec",
    "_chat_forge_plan_panel_spec",
    "_chat_forge_markdown_panel_spec",
    "_format_asset_size",
    "_asset_status_tone",
    "_chat_assets_picker_spec",
    "_chat_asset_detail_panel_spec",
]
