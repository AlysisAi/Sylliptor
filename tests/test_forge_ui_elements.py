"""Tests for the additive Forge UI elements (readiness chip, next-step hint,
empty-state, phase rule, run-complete card glyphs, and the colorized swarm
trace). These elements are purely additive over data the code already computes;
the pinned output contracts live in test_chat_plan_mode.py / test_cli_ux.py.
"""

from __future__ import annotations

import io

from rich.console import Console

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.surface.rich_surface import RichSurface, _swarm_trace_severity
from sylliptor_agent_cli.swarm_trace import (
    SerializedSwarmTraceSink,
    build_swarm_trace_event,
)


class _FakePaths:
    run_id = "abc123"


class _Cp1252Console:
    """Stands in for a legacy console whose encoding cannot render box glyphs."""

    encoding = "cp1252"


def _render_line(builder, plan: dict, *, console: Console) -> str:
    buffer = io.StringIO()
    out_console = Console(file=buffer, force_terminal=False, width=120)
    out_console.print(builder(console=console, plan=plan), highlight=False)
    return buffer.getvalue()


def _plain_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


# --- _forge_plan_state -------------------------------------------------------


def test_forge_plan_state_classifies_lifecycle() -> None:
    assert cli_mod._forge_plan_state({}) == "empty"
    assert cli_mod._forge_plan_state({"requirements": ["do a thing"]}) == "planning"
    assert cli_mod._forge_plan_state({"tasks": [{"id": "T01", "status": "planned"}]}) == "ready"
    assert (
        cli_mod._forge_plan_state(
            {"tasks": [{"id": "T01", "status": "done"}, {"id": "T02", "status": "done"}]}
        )
        == "done"
    )
    # A failed task keeps the plan actionable (ready), not done.
    assert (
        cli_mod._forge_plan_state(
            {"tasks": [{"id": "T01", "status": "done"}, {"id": "T02", "status": "failed"}]}
        )
        == "ready"
    )


# --- readiness chip ----------------------------------------------------------


def test_forge_readiness_line_states() -> None:
    console = _plain_console()
    empty = _render_line(cli_mod._forge_plan_readiness_line, {}, console=console)
    assert "no tasks yet" in empty

    planning = _render_line(
        cli_mod._forge_plan_readiness_line, {"requirements": ["x", "y"]}, console=console
    )
    assert "2 requirements, 0 tasks" in planning

    ready = _render_line(
        cli_mod._forge_plan_readiness_line,
        {"tasks": [{"id": "T01", "status": "planned"}]},
        console=console,
    )
    assert "ready to execute · 1 task · 0 blocked" in ready

    blocked = _render_line(
        cli_mod._forge_plan_readiness_line,
        {"tasks": [{"id": "T01", "status": "failed"}]},
        console=console,
    )
    assert "needs attention" in blocked and "blocked" in blocked

    done = _render_line(
        cli_mod._forge_plan_readiness_line,
        {"tasks": [{"id": "T01", "status": "done"}]},
        console=console,
    )
    assert "all tasks done" in done


def test_forge_readiness_line_ascii_fallback() -> None:
    # Build with a cp1252 console (cannot encode ●/○) but render anywhere.
    line = cli_mod._forge_plan_readiness_line(console=_Cp1252Console(), plan={})
    buffer = io.StringIO()
    Console(file=buffer, force_terminal=False, width=120).print(line, highlight=False)
    out = buffer.getvalue()
    assert "●" not in out and "○" not in out
    assert "no tasks yet" in out


# --- next-step hint ----------------------------------------------------------


def test_forge_next_step_line_states() -> None:
    console = _plain_console()
    empty = _render_line(cli_mod._forge_next_step_line, {}, console=console)
    assert "Next ·" in empty and "/goal" in empty

    planning = _render_line(cli_mod._forge_next_step_line, {"requirements": ["x"]}, console=console)
    assert "/show" in planning

    ready = _render_line(
        cli_mod._forge_next_step_line,
        {"tasks": [{"id": "T01", "status": "planned"}]},
        console=console,
    )
    assert "/execute plan" in ready

    done = _render_line(
        cli_mod._forge_next_step_line,
        {"tasks": [{"id": "T01", "status": "done"}]},
        console=console,
    )
    assert "/done" in done


# --- glyph capability + phase rule ------------------------------------------


def test_forge_supports_unicode_glyphs_detects_encoding() -> None:
    assert cli_mod._forge_supports_unicode_glyphs(_Cp1252Console()) is False

    class _Utf8Console:
        encoding = "utf-8"

    assert cli_mod._forge_supports_unicode_glyphs(_Utf8Console()) is True


def test_forge_phase_rule_contains_label_and_bar() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=60)
    console.print(cli_mod._forge_phase_rule(console=console, label="DONE"), highlight=False)
    out = buffer.getvalue()
    assert "DONE" in out
    assert "│" in out
    assert "─" in out


def test_forge_phase_rule_ascii_fallback() -> None:
    rule = cli_mod._forge_phase_rule(console=_Cp1252Console(), label="DONE")
    buffer = io.StringIO()
    Console(file=buffer, force_terminal=False, width=60).print(rule, highlight=False)
    out = buffer.getvalue()
    assert "DONE" in out
    assert "─" not in out
    assert "-" in out


# --- /show empty-state + ready-state -----------------------------------------


def test_show_forge_plan_summary_empty_state() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    cli_mod._show_forge_plan_summary(console=console, paths=_FakePaths(), plan={})
    out = buffer.getvalue()
    assert "This plan is empty." in out
    assert "no tasks yet" in out
    assert "Next ·" in out
    assert "Forge Tasks" not in out  # task table is skipped for empty plans


def test_show_forge_plan_summary_ready_state() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    plan = {
        "project_goal": "Ship it",
        "tasks": [{"id": "T01", "status": "planned", "title": "Do X"}],
    }
    cli_mod._show_forge_plan_summary(console=console, paths=_FakePaths(), plan=plan)
    out = buffer.getvalue()
    assert "ready to execute" in out
    assert "Next ·" in out
    assert "/execute plan" in out


# --- swarm trace severity + colorized rendering ------------------------------


def test_swarm_trace_severity_classification() -> None:
    assert _swarm_trace_severity("verify.error", "verification did not run") == "error"
    assert _swarm_trace_severity("review.error", "Changes requested by review gate.") == "error"
    assert _swarm_trace_severity("worker.lifecycle", "Merged successfully (abc).") == "success"
    assert _swarm_trace_severity("worker.lifecycle", "Applied successfully (def).") == "success"
    assert _swarm_trace_severity("swarm.startup", "Heads up warning: be careful") == "warning"
    assert _swarm_trace_severity("worker.tool", "Step 3: read file") == "neutral"


def test_rich_surface_on_swarm_trace_colorizes_outcomes() -> None:
    buffer = io.StringIO()
    surface = RichSurface(console=Console(file=buffer, force_terminal=False, width=120))
    surface.on_swarm_trace("[T01] verification failed", phase="verify.error", task_id="T01")
    surface.on_swarm_trace(
        "[T02] Merged successfully (abc).", phase="worker.lifecycle", task_id="T02"
    )
    surface.on_swarm_trace("[T03] Step 3: read file", phase="worker.tool", task_id="T03")
    out = buffer.getvalue()
    assert "✗" in out and "verification failed" in out
    assert "✓" in out and "Merged successfully" in out
    assert "Step 3: read file" in out  # neutral falls through to the plain bullet line


def test_serialized_swarm_trace_sink_routes_to_on_swarm_trace(tmp_path) -> None:
    calls: list[tuple[str, str, str | None]] = []

    class _CapSurface:
        def on_swarm_trace(self, message: str, *, phase: str = "", task_id=None) -> None:
            calls.append((message, phase, task_id))

    sink = SerializedSwarmTraceSink(
        artifact_path=tmp_path / "trace.jsonl",
        trace_level="compact",
        surface=_CapSurface(),
    )
    sink.emit(
        build_swarm_trace_event(run_id="r1", phase="verify.error", message="boom", task_id="T01")
    )
    sink.close()
    assert calls
    assert calls[0][0] == "[T01] boom"
    assert calls[0][1] == "verify.error"
    assert calls[0][2] == "T01"
