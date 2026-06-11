from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from ... import __version__
from ...config import AppConfig, ConfigError, config_path, load_config
from ...provider_diagnostics import build_provider_diagnostics, validate_active_provider_live
from ...provider_telemetry import (
    diagnostic_bundle_payload,
    last_provider_call_summary,
    last_web_search_summary,
)
from ...sandbox_doctor import diagnose_sandbox
from ...skills import resolve_skills_enabled
from ...tools.availability import get_tool_availability
from ...tools.registry import iter_builtin_tool_metadata
from ...tools.web_search import resolve_web_search_runtime_status
from ..assets_cli import assets_app as forge_assets_app
from . import _patchable
from ._shared import Mode, _console, _Table
from .config import config_app
from .conventions import conventions_app
from .extensions import ext_app
from .forge import forge_app
from .hooks import hooks_app
from .mcp import mcp_app, mcp_auth_app, mcp_prompts_app
from .profile import profile_app
from .report import report_app
from .sandbox import sandbox_app
from .server import server_app
from .sessions import sessions_app
from .skills import skill_app
from .tools import tool_app
from .update import (
    _BACKGROUND_UPDATE_SUBCOMMANDS,
    _cached_update_status_summary,
    _start_background_update_check,
    update_app,
)

if TYPE_CHECKING:
    from rich.table import Table


def _cli_module() -> Any:
    module = sys.modules.get("sylliptor_agent_cli.cli")
    if module is not None:
        return module
    from ... import cli

    return cli


app = typer.Typer(add_completion=False, help="Local CLI coding agent (multi-provider).")
app.add_typer(config_app, name="config")
app.add_typer(profile_app, name="profile")
app.add_typer(update_app, name="update")
app.add_typer(sessions_app, name="sessions")
app.add_typer(forge_app, name="forge")
forge_app.add_typer(forge_assets_app, name="assets")
app.add_typer(server_app, name="server")
app.add_typer(ext_app, name="ext")
app.add_typer(report_app, name="report")
app.add_typer(mcp_app, name="mcp")
app.add_typer(skill_app, name="skill")
app.add_typer(conventions_app, name="conventions")
app.add_typer(tool_app, name="tool")
app.add_typer(hooks_app, name="hooks")
app.add_typer(sandbox_app, name="sandbox")
mcp_app.add_typer(mcp_prompts_app, name="prompts")
mcp_app.add_typer(mcp_auth_app, name="auth")


def _show_version(value: bool) -> None:
    if not value:
        return
    typer.echo(__version__)
    raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_show_version,
        is_eager=True,
        help="Show the installed Sylliptor version and exit.",
    ),
) -> None:
    cli = _cli_module()
    if ctx.invoked_subcommand is not None:
        if ctx.invoked_subcommand in _BACKGROUND_UPDATE_SUBCOMMANDS:
            _start_background_update_check()
        return
    console = cli._console()
    if cli._is_interactive_terminal():
        _start_background_update_check()
    if cli._is_non_interactive_terminal():
        console.print(cli._home_panel())
        return
    if not cli._home_prompt_enabled():
        if not cli._maybe_run_first_run_setup_wizard():
            return
        cli._maybe_run_startup_config_menu()
        cli._run_default_chat_action()
        return
    console.print(cli._home_panel())
    try:
        action = (
            typer.prompt(
                "Action [1=chat|2=run|3=setup|4=doctor|5=plan|6=quit]",
                default="1",
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        console.print("")
        return
    action = cli._HOME_ACTION_ALIASES.get(action, action)

    if action in {"quit", "q", "exit"}:
        return
    if action in {"chat", "c"}:
        cli._maybe_run_startup_config_menu()
        cli._run_default_chat_action()
        return
    if action in {"run", "r"}:
        instruction = typer.prompt("Instruction").strip()
        if instruction:
            cli._run_default_run_action(instruction)
        return
    if action in {"setup", "s"}:
        cli.setup()
        return
    if action in {"doctor", "d"}:
        cli.doctor()
        return
    if action in {"plan", "p"}:
        cli.forge_plan(path=Path("."))
        return
    console.print("[yellow]Unknown action.[/yellow] Run `sylliptor --help`.")


@app.command()
def run(
    instruction: str = typer.Argument(..., help="What you want the agent to do."),
    path: Path = typer.Option(Path("."), "--path", help="Working directory/root."),
    create_path: bool = typer.Option(
        False,
        "--create-path",
        help="Create --path if it does not exist before binding the workspace.",
    ),
    allow_broad_workspace: bool = typer.Option(
        False,
        "--allow-broad-workspace",
        help="Allow guarded broad workspaces in non-interactive startup flows.",
    ),
    image: list[Path] | None = typer.Option(
        None,
        "--image",
        help="Attach image path(s). Repeat --image for multiple files.",
    ),
    mode: Mode | None = typer.Option(None, "--mode", help="Mode override."),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL override."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
    stream: bool | None = typer.Option(
        None,
        "--stream/--no-stream",
        help="Enable streamed assistant output.",
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Max steps override."),
    subagents: bool | None = typer.Option(
        None,
        "--subagents/--no-subagents",
        help="Enable or disable subagent delegation for this session.",
    ),
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
    verify_cmd: list[str] | None = typer.Option(
        None,
        "--verify-cmd",
        help="Override verification command for this run (repeatable).",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help=(
            "Read API key from this environment variable (overrides SYLLIPTOR_API_KEY/OPENAI_API_KEY)."
        ),
    ),
    api_key_stdin: bool = typer.Option(
        False,
        "--api-key-stdin",
        help="Prompt for API key (hidden input). Key is kept in memory for this run only.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help=(
            "UNSAFE: Provide API key via CLI argument (may leak via shell history / process list). "
            "Prefer --api-key-stdin or --api-key-env."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="In auto mode, skip confirmations for sensitive commands (hard blocks still apply).",
    ),
) -> None:
    from ..chat import run_impl

    return run_impl(
        _cli_module(),
        instruction,
        path,
        create_path,
        allow_broad_workspace,
        image,
        mode,
        model,
        base_url,
        temperature,
        stream,
        max_steps,
        subagents,
        no_log,
        verify_cmd,
        api_key_env,
        api_key_stdin,
        api_key,
        yes,
    )


@app.command()
def chat(
    path: Path = typer.Option(Path("."), "--path", help="Working directory/root."),
    create_path: bool = typer.Option(
        False,
        "--create-path",
        help="Create --path if it does not exist before binding the workspace.",
    ),
    allow_broad_workspace: bool = typer.Option(
        False,
        "--allow-broad-workspace",
        help="Allow guarded broad workspaces in non-interactive startup flows.",
    ),
    image: list[Path] | None = typer.Option(
        None,
        "--image",
        help="Queue image path(s) for the next message. Repeat --image for multiple files.",
    ),
    mode: Mode | None = typer.Option(None, "--mode", help="Mode override."),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL override."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
    stream: bool | None = typer.Option(
        None,
        "--stream/--no-stream",
        help="Enable streamed assistant output.",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Max steps override (per user turn).",
    ),
    subagents: bool | None = typer.Option(
        None,
        "--subagents/--no-subagents",
        help="Enable or disable subagent delegation for this session.",
    ),
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
    verify_cmd: list[str] | None = typer.Option(
        None,
        "--verify-cmd",
        help="Override verification command for this chat session (repeatable).",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help=(
            "Read API key from this environment variable (overrides SYLLIPTOR_API_KEY/OPENAI_API_KEY)."
        ),
    ),
    api_key_stdin: bool = typer.Option(
        False,
        "--api-key-stdin",
        help="Prompt for API key (hidden input). Key is kept in memory for this run only.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help=(
            "UNSAFE: Provide API key via CLI argument (may leak via shell history / process list). "
            "Prefer --api-key-stdin or --api-key-env."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="In auto mode, skip confirmations for sensitive commands.",
    ),
) -> None:
    from ..chat import chat_impl

    return chat_impl(
        _cli_module(),
        path,
        create_path,
        allow_broad_workspace,
        image,
        mode,
        model,
        base_url,
        temperature,
        stream,
        max_steps,
        subagents,
        no_log,
        verify_cmd,
        api_key_env,
        api_key_stdin,
        api_key,
        yes,
    )


def _doctor_table(cfg: AppConfig) -> Table:
    table = _Table(title="sylliptor doctor")
    table.add_column("check")
    table.add_column("result")
    api_key = _cli_module()._resolved_api_key_value()

    table.add_row("python", sys.version.split()[0])

    def _cmd_exists(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    table.add_row("git", "ok" if _cmd_exists("git") else "missing")
    table.add_row(
        "rg",
        "ok" if _cmd_exists("rg") else "missing (search fallback will be slower)",
    )
    table.add_row("config_path", os.fspath(config_path()))
    table.add_row(
        "model_set",
        "yes" if bool(cfg.model) else "no (run: sylliptor config set model <MODEL>)",
    )
    table.add_row(
        "api_key_set",
        (
            f"yes ({_cli_module()._api_key_source_label(api_key.source)})"
            if api_key.key
            else "no (run: sylliptor config set-api-key)"
        ),
    )
    table.add_row("base_url", cfg.base_url)
    table.add_row("update", _cached_update_status_summary(cfg))
    web_search_status = resolve_web_search_runtime_status(cfg=cfg, api_key=api_key.key)
    table.add_row("web_search", web_search_status.availability_label)
    table.add_row("web_search_provider", web_search_status.provider or "(none)")
    table.add_row("web_search_setup", web_search_status.setup_hint)
    try:
        sandbox_status = _patchable("diagnose_sandbox", diagnose_sandbox)(
            cfg, include_smoke=False, include_server_image=False
        )
        sandbox_label = (
            "ready" if sandbox_status.ready else "not ready (run: sylliptor doctor sandbox)"
        )
        table.add_row("sandbox", sandbox_label)
        table.add_row("sandbox_backend", sandbox_status.selected_backend or "(none)")
    except ConfigError as exc:
        table.add_row("sandbox", f"config error ({exc})")
    table.add_row("temperature", str(cfg.temperature))
    table.add_row("coding_temperature", str(cfg.coding_temperature))
    table.add_row("chat_temperature", str(cfg.chat_temperature))
    table.add_row("step_budget_policy", str(cfg.step_budget_policy))
    table.add_row("max_steps", str(cfg.max_steps))
    table.add_row("task_max_steps", str(cfg.task_max_steps))
    table.add_row("subagent_max_steps", str(cfg.subagent_max_steps))
    table.add_row("custom_tools_enabled", "yes" if cfg.custom_tools_enabled else "no")
    table.add_row("stream", "yes" if cfg.stream else "no")
    return table


def _provider_doctor_table(cfg: AppConfig) -> Table:
    diagnostics = build_provider_diagnostics(cfg)
    table = _Table(title="sylliptor doctor providers")
    table.add_column("field")
    table.add_column("value")
    for key, value in diagnostics.rows():
        table.add_row(key, value)
    last_call = last_provider_call_summary()
    if last_call:
        table.add_row("last_call_provider", str(last_call.get("provider_key") or "(unknown)"))
        table.add_row("last_call_protocol", str(last_call.get("protocol") or "(unknown)"))
        table.add_row("last_call_status", str(last_call.get("status_category") or "(unknown)"))
        table.add_row("last_call_latency_ms", str(last_call.get("latency_ms") or 0))
        table.add_row("last_call_stream", "yes" if last_call.get("stream") else "no")
        table.add_row(
            "last_call_web_search",
            str((last_call.get("web_search") or {}).get("backend_kind") or "off"),
        )
    last_search = last_web_search_summary()
    if last_search:
        table.add_row("last_web_search_adapter", str(last_search.get("web_search_adapter") or ""))
        table.add_row(
            "last_web_search_hosted",
            "yes" if last_search.get("provider_hosted_search") else "no",
        )
        table.add_row("last_web_search_sources", str(last_search.get("source_count") or 0))
    return table


def _doctor_bundle_payload(cfg: AppConfig) -> dict[str, Any]:
    diagnostics = build_provider_diagnostics(cfg)
    return diagnostic_bundle_payload(
        provider_diagnostics={key: value for key, value in diagnostics.rows()}
    )


def _provider_live_validation_table(cfg: AppConfig, *, timeout_s: float = 15.0) -> Table:
    validation = _patchable("validate_active_provider_live", validate_active_provider_live)(
        cfg,
        timeout_s=timeout_s,
    )
    table = _Table(title="sylliptor doctor providers --live")
    table.add_column("field")
    table.add_column("value")
    for key, value in validation.rows():
        table.add_row(key, value)
    return table


@dataclass(frozen=True)
class _ToolAvailabilityRow:
    name: str
    categories: str
    status: str
    purpose: str
    notes: str


def _default_session_api_key() -> str | None:
    return _cli_module()._resolved_api_key_value().key


def _tool_availability_rows(cfg: AppConfig) -> list[_ToolAvailabilityRow]:
    rows: list[_ToolAvailabilityRow] = []
    main_api_key = _default_session_api_key()

    for spec in iter_builtin_tool_metadata():
        status = "available"
        notes: list[str] = []

        availability = get_tool_availability(spec.name)
        if spec.optional and availability is not None and availability.unavailable_reason:
            status = "optional-unavailable"
            notes.append(f"reason={availability.unavailable_reason}")

        if spec.name == "web_search":
            runtime = resolve_web_search_runtime_status(cfg=cfg, api_key=main_api_key)
            status = runtime.availability_label
            notes.append(f"mode={runtime.mode}")
            notes.append(f"provider={runtime.provider or '(none)'}")
            if runtime.registration_ready:
                notes.append("ready for registration in main agent sessions")
            if runtime.provider == "openai_responses":
                notes.append(
                    "OpenAI Responses readiness is conservative: explicit web_search_base_url or first-party OpenAI base_url"
                )
            elif runtime.provider in {
                "xai_responses",
                "anthropic_messages",
                "gemini_grounding",
                "openrouter_web",
                "perplexity_sonar",
                "groq_compound",
                "mistral_conversations",
                "moonshot_kimi",
                "zhipu_web_search",
                "volcengine_web_search",
            }:
                notes.append(f"available via {runtime.provider} provider adapter")
            elif runtime.provider == "dashscope_chat":
                notes.append("available via DashScope Chat Completions enable_search")
            elif runtime.provider == "tavily":
                notes.append("available via TAVILY_API_KEY")
            notes.extend(runtime.notes)
            if not runtime.registration_ready:
                notes.append(f"setup: {runtime.setup_hint}")
        elif spec.name == "skill_read":
            if not resolve_skills_enabled(cfg):
                status = "optional-disabled"
                notes.append("set skills_enabled=true to advertise skills and register skill_read")
            else:
                status = "contextual"
                notes.append("registered only when discovered skill bundles are available")
        elif spec.name == "subagent_run" and not bool(getattr(cfg, "subagents_enabled", False)):
            status = "optional-disabled"
            notes.append("set subagents_enabled=true or use --subagents for top-level runs")

        if spec.built_in_subagent_exposure.strip().lower() == "hidden":
            notes.append("hidden from built-in readonly subagents")

        rows.append(
            _ToolAvailabilityRow(
                name=spec.name,
                categories=", ".join(spec.categories),
                status=status,
                purpose=spec.description.strip(),
                notes="; ".join(note for note in notes if note.strip()) or "-",
            )
        )
    return rows


def _tools_table(cfg: AppConfig) -> Table:
    table = _Table(title="sylliptor tools")
    table.add_column("tool")
    table.add_column("categories")
    table.add_column("status")
    table.add_column("purpose")
    table.add_column("notes")

    for row in _tool_availability_rows(cfg):
        table.add_row(
            row.name,
            row.categories,
            row.status,
            row.purpose,
            row.notes,
        )
    return table


@app.command()
def setup(
    section: str | None = typer.Argument(
        None,
        help="Optional setup target. Use `sandbox` to prepare the safe command runner.",
    ),
) -> None:
    """Run first-time setup, or prepare a named setup target."""
    if section is not None:
        target = section.strip().lower()
        if target != "sandbox":
            _console().print("[red]Unknown setup target.[/red] Use: sylliptor setup sandbox")
            raise typer.Exit(code=2)
        _cli_module()._run_sandbox_setup_command(pull=True)
        return

    from ..setup_wizard import run_setup_wizard

    if run_setup_wizard():
        console = _console()
        try:
            result = _patchable("diagnose_sandbox", diagnose_sandbox)(
                _patchable("load_config", load_config)(),
                include_smoke=False,
                include_server_image=False,
            )
        except ConfigError:
            return
        if not result.ready:
            console.print()
            console.print(
                "[yellow]Safe runner setup is not complete yet. "
                "Run `sylliptor doctor sandbox` for details or `sylliptor setup sandbox` "
                "after installing/starting Docker.[/yellow]"
            )


@app.command()
def doctor(
    section: str | None = typer.Argument(
        None,
        help="Optional check group. Use `sandbox`, `providers`, or `bundle`.",
    ),
    smoke: bool = typer.Option(
        True,
        "--smoke/--no-smoke",
        help="Run a sandbox smoke command when checking `doctor sandbox`.",
    ),
    env: bool = typer.Option(
        False,
        "--env",
        help="Show sandbox-related environment variable overrides.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Run a minimal live text request for `doctor providers` after confirmation.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm live provider validation without prompting.",
    ),
    live_timeout: float = typer.Option(
        15.0,
        "--live-timeout",
        min=1.0,
        help="Timeout in seconds for `doctor providers --live`.",
    ),
    redacted: bool = typer.Option(
        True,
        "--redacted/--no-redacted",
        help="Emit only redacted diagnostic data for `doctor bundle`.",
    ),
) -> None:
    if section is not None:
        target = section.strip().lower()
        if target == "sandbox":
            _cli_module()._run_sandbox_doctor_command(include_smoke=smoke, include_env=env)
            return
        if target in {"provider", "providers"}:
            cfg = _patchable("load_config", load_config)()
            console = _console()
            console.print(_provider_doctor_table(cfg))
            if live:
                console.print(
                    "[yellow]Live provider validation sends one minimal text request and may "
                    "incur provider cost or rate-limit usage.[/yellow]"
                )
                if not yes and not typer.confirm(
                    "Run live provider validation for the active profile?", default=False
                ):
                    console.print("Live provider validation cancelled.")
                    return
                console.print(_provider_live_validation_table(cfg, timeout_s=live_timeout))
            return
        if target == "bundle":
            if not redacted:
                _console().print("[red]Only redacted doctor bundles are supported.[/red]")
                raise typer.Exit(code=2)
            cfg = _patchable("load_config", load_config)()
            typer.echo(json.dumps(_doctor_bundle_payload(cfg), indent=2, sort_keys=True))
            return
        if target != "sandbox":
            _console().print(
                "[red]Unknown doctor target.[/red] Use: sylliptor doctor sandbox|providers|bundle"
            )
            raise typer.Exit(code=2)
    console = _console()
    cfg = _patchable("load_config", load_config)()
    console.print(_doctor_table(cfg))


@app.command()
def tools() -> None:
    console = _console()
    cfg = _patchable("load_config", load_config)()
    console.print(_tools_table(cfg))
    console.print(
        "[dim]`sylliptor tools` shows the built-in catalog plus config-dependent availability. "
        "Use `/status` inside chat for session-specific details.[/dim]"
    )
    console.print(
        "[dim]`web_search` discovers candidate sources; `web_fetch` retrieves a specific chosen URL.[/dim]"
    )
    console.print(
        "[dim]Top-level readonly/Plan sessions can use ready web tools; nested readonly subagents keep them hidden.[/dim]"
    )
    console.print(
        "[dim]Use `web_search_mode=off|auto|native|external` and optional `web_search_adapter`. "
        "`auto` can use OpenAI Responses, xAI, Anthropic, Gemini, OpenRouter, DashScope "
        "Chat/Qwen, Kimi, Zhipu/GLM, Doubao, Perplexity, Groq, Mistral, or Tavily when "
        "`TAVILY_API_KEY` is set. `native` never uses Tavily; `external` uses only external "
        "search adapters. Legacy `on` and `web_search_enabled` values still load as `auto`.[/dim]"
    )
    console.print(
        "[dim]Custom tools are managed separately via `sylliptor tool list|info|trust|untrust`.[/dim]"
    )


@app.command()
def login() -> None:
    """Connect your Sylliptor account and unlock the free MiMo trial."""
    from ... import account_login

    console = _console()
    cfg = _patchable("load_config", load_config)()
    try:
        result = account_login.login(
            cfg, output_write=lambda message: console.print(message, highlight=False)
        )
    except (account_login.SylliptorLoginError, ConfigError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    who = f" as [bold]{result.email}[/bold]" if result.email else ""
    console.print(f"[green]Logged in{who}.[/green] Your free MiMo trial is ready.")
    console.print(
        f"Active profile: [bold]{result.profile_name}[/bold] · default model: "
        f"[bold]{result.model}[/bold]"
    )
    console.print(
        "[dim]Run `sylliptor chat` to start. Use `sylliptor logout` to disconnect.[/dim]"
    )


@app.command()
def logout() -> None:
    """Disconnect your Sylliptor account (forgets the stored access key)."""
    from ... import account_login

    console = _console()
    cfg = _patchable("load_config", load_config)()
    if account_login.logout(cfg):
        console.print("[green]Logged out.[/green] Your stored MiMo access key was removed.")
    else:
        console.print("You're not logged in to a Sylliptor account.")


@app.command()
def whoami() -> None:
    """Show your Sylliptor login status."""
    from ... import account_login

    console = _console()
    cfg = _patchable("load_config", load_config)()
    status = account_login.login_status(cfg)
    if not status.logged_in:
        console.print("Not logged in. Run `sylliptor login` to start your free MiMo trial.")
        return
    active = "active" if status.active else "not active"
    console.print("[green]Logged in[/green] to the Sylliptor MiMo trial.")
    console.print(
        f"Profile: [bold]{status.profile_name}[/bold] ({active}) · key {status.key_preview}"
    )
    console.print(f"Proxy: {status.base_url}")
    trial = account_login.fetch_trial_status(cfg)
    if trial is not None:
        line = account_login.format_trial_status_line(trial)
        if line:
            console.print(line)
    else:
        console.print("[dim](Could not reach the trial service for live status.)[/dim]")
