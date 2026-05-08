# ruff: noqa: F401,F403,F405,I001
# Legacy split module: dependencies are synced by cli_surface.py.
from __future__ import annotations

from .cli_common import *
from .update import _cached_update_notice


def _is_narrow_terminal() -> bool:
    return _terminal_width() < 88


def _apply_temperature_override(cfg: AppConfig, temperature: float) -> None:
    cfg.temperature = temperature
    cfg.coding_temperature = temperature
    cfg.review_temperature = temperature
    cfg.planner_temperature = temperature
    cfg.conflict_review_temperature = temperature
    cfg.compactor_temperature = temperature
    cfg.chat_temperature = temperature


def _home_panel() -> Panel:
    lines = [
        "Quick actions:",
        "- 1) chat              interactive chat loop",
        "- 2) run               one-shot agent execution (you will be prompted for instruction)",
        "- 3) setup             first-run setup wizard",
        "- 4) doctor            environment checks",
        "- 5) plan              Forge plan command",
        "- 6) quit              exit home prompt",
        "- forge (in chat)    enter planning mode with /forge",
        "- forge show         summarize current plan artifacts",
    ]
    update_notice = _cached_update_notice()
    if update_notice:
        lines.append("")
        lines.append(update_notice)
    content = "\n".join(lines)
    return _Panel(content, title="Sylliptor Home", border_style="bright_black")


def _home_prompt_enabled() -> bool:
    raw = str(env_get("SYLLIPTOR_HOME_PROMPT") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _run_default_chat_action() -> None:
    _patchable("chat", chat)(
        path=Path("."),
        create_path=False,
        allow_broad_workspace=False,
        image=None,
        mode=None,
        model=None,
        base_url=None,
        temperature=None,
        stream=None,
        max_steps=None,
        subagents=None,
        no_log=False,
        verify_cmd=None,
        api_key_env=None,
        api_key_stdin=False,
        api_key=None,
        yes=False,
    )


def _maybe_run_startup_config_menu() -> None:
    cfg = _patchable("load_config", load_config)()
    api_key_present = bool(_patchable("resolve_api_key", resolve_api_key)().key)
    if cfg.model and api_key_present:
        return
    from ..config_menu import run_config_menu

    focus = "api_key" if not api_key_present else "model"
    run_config_menu(cfg=cfg, auto_focus=focus)


def _should_run_first_run_setup_wizard() -> bool:
    cfg_exists = config_path().exists()
    cfg = _patchable("load_config", load_config)()
    api_key_present = bool(_patchable("resolve_api_key", resolve_api_key)().key)
    return not cfg_exists or (not cfg.model and not api_key_present)


def _maybe_run_first_run_setup_wizard() -> bool:
    if not _should_run_first_run_setup_wizard():
        return True
    from ..setup_wizard import run_setup_wizard

    if run_setup_wizard():
        return True
    _console().print("[yellow]Exiting without starting chat.[/yellow]")
    return False


def _run_default_run_action(instruction: str) -> None:
    _patchable("run", run)(
        instruction=instruction,
        path=Path("."),
        create_path=False,
        allow_broad_workspace=False,
        image=None,
        mode=None,
        model=None,
        base_url=None,
        temperature=None,
        stream=None,
        max_steps=None,
        subagents=None,
        no_log=False,
        verify_cmd=None,
        api_key_env=None,
        api_key_stdin=False,
        api_key=None,
        yes=False,
    )


def _current_branch_label(root: Path) -> str:
    if shutil.which("git") is None:
        return "-"
    proc = subprocess.run(
        ["git", "-C", os.fspath(root), "rev-parse", "--abbrev-ref", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "-"
    branch = proc.stdout.strip()
    return branch or "-"


def _is_git_dirty(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    proc = subprocess.run(
        ["git", "-C", os.fspath(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def _is_interactive_terminal() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _is_non_interactive_terminal() -> bool:
    return not _is_interactive_terminal()


def _resolve_api_key_override(
    *,
    api_key: str | None,
    api_key_env: str | None,
    api_key_stdin: bool,
) -> str | None:
    # Per-command overrides stay in-memory; persistent storage is handled only
    # by explicit config credentials commands / setup flow.
    selected = sum(bool(x) for x in [api_key is not None, api_key_env is not None, api_key_stdin])
    if selected > 1:
        raise ConfigError("Use only one of: --api-key, --api-key-env, --api-key-stdin")

    if api_key is not None:
        v = api_key.strip()
        if not v:
            raise ConfigError("API key is empty.")
        return v

    if api_key_env is not None:
        name = api_key_env.strip()
        if not name:
            raise ConfigError("--api-key-env must be non-empty.")
        v = (os.environ.get(name) or "").strip()
        if not v:
            raise ConfigError(f"Environment variable {name} is not set.")
        return v

    if api_key_stdin:
        v = typer.prompt("API key", hide_input=True).strip()
        if not v:
            raise ConfigError("API key is empty.")
        return v

    return None


def _api_key_source_label(source: str) -> str:
    normalized = str(source or "").strip()
    if normalized == "env:SYLLIPTOR_API_KEY":
        return "env (SYLLIPTOR_API_KEY)"
    if normalized == "env:OPENAI_API_KEY":
        return "env (OPENAI_API_KEY)"
    if normalized.startswith("env:"):
        return f"env ({normalized.removeprefix('env:')})"
    if normalized.startswith("stored:profile="):
        return f"stored ({normalized.removeprefix('stored:profile=')})"
    if normalized in {"stored", "stored:legacy"}:
        return "stored"
    return "missing"


def _resolved_api_key_value() -> ApiKeyResolution:
    return resolve_api_key()


def _parse_bool_text(value: str) -> bool | None:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _safe_component(raw: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", str(raw).strip())
    return clean or "x"


def _resolve_usage_hud_default(cfg: AppConfig) -> bool:
    raw_env = env_get("SYLLIPTOR_SHOW_USAGE_HUD")
    if raw_env is not None:
        parsed_env = _parse_bool_text(raw_env)
        if parsed_env is not None:
            return parsed_env
    raw_cfg = cfg.extra_fields.get("show_usage_hud")
    if isinstance(raw_cfg, bool):
        return raw_cfg
    if isinstance(raw_cfg, str):
        parsed_cfg = _parse_bool_text(raw_cfg)
        if parsed_cfg is not None:
            return parsed_cfg
    return True


def _chat_usage_hud_enabled(session: Any) -> bool:
    raw = getattr(session, "_usage_hud_enabled", True)
    return bool(raw)


def _set_chat_usage_hud_enabled(session: Any, enabled: bool) -> None:
    session._usage_hud_enabled = bool(enabled)


def _refresh_chat_hud_context_cache(session: Any) -> None:
    context_fn = getattr(session, "context_left", None)
    if callable(context_fn):
        try:
            session._hud_context_cache = context_fn()
        except Exception:  # noqa: BLE001
            session._hud_context_cache = None


def _chat_context_hud_value(session: Any) -> str:
    value = _chat_context_percent_value(session)
    if value is None:
        return "n/a"
    label = f"{value:.1f}%"
    if value < 10.0:
        return f"{label} !!"
    if value < 25.0:
        return f"{label} !"
    return label


def _format_cost_with_unknown(
    *,
    known_cost: float | None,
    unknown_calls: int,
    style: str,
) -> str:
    base = format_usd(known_cost, style=style)
    if unknown_calls <= 0:
        return base
    return f"{base} (+{unknown_calls} unknown)"


def _known_cost_value(data: dict[str, Any]) -> float | None:
    if int(data.get("known_cost_calls") or 0) <= 0:
        return None
    raw = data.get("cost_usd")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _format_exact_token_count(value: Any) -> str:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        parsed = 0
    return f"{max(0, parsed):,}"


def _format_compact_token_count(value: Any) -> str:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        parsed = 0
    count = max(0, parsed)
    if count < 10_000:
        return f"{count:,}"
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def _chat_context_percent_value(session: Any) -> float | None:
    ctx = getattr(session, "_hud_context_cache", None)
    if ctx is None:
        return None
    percent = getattr(ctx, "dynamic_context_percent_left", None)
    if percent is None:
        percent = getattr(ctx, "context_window_percent_left", None)
    if percent is None:
        percent = getattr(ctx, "percent_left", None)
    if percent is None:
        return None
    try:
        return float(percent)
    except (TypeError, ValueError):
        return None


def _chat_effective_budget_percent_value(session: Any) -> float | None:
    ctx = getattr(session, "_hud_context_cache", None)
    if ctx is None:
        return None
    percent = getattr(ctx, "effective_percent_left", None)
    if percent is None:
        percent = getattr(ctx, "percent_left", None)
    if percent is None:
        return None
    try:
        return float(percent)
    except (TypeError, ValueError):
        return None


def _format_chat_context_percent(percent: float | None) -> str:
    if percent is None:
        return "n/a"
    return f"{percent:.1f}%"


def _chat_usage_warning_line(percent: float | None) -> str | None:
    if percent is None:
        return None
    if percent < 10.0:
        return "\u26a0\ufe0e  Critical input budget — run /compact to reduce context"
    if percent < 20.0:
        return "\u26a0\ufe0e  Low input budget — run /compact to reduce context"
    return None


def _chat_turn_usage_style(session: Any) -> str:
    percent = _chat_effective_budget_percent_value(session)
    if percent is None:
        return "dim"
    if percent < 10.0:
        return "bold red"
    if percent < 20.0:
        return "yellow"
    if percent < 50.0:
        return STYLE_CONTENT
    return "dim"


def _format_usage_cost_for_display(*, known_cost: float | None, unknown_calls: int) -> str:
    if known_cost is None:
        return f"unknown ({unknown_calls} unmetered)" if unknown_calls > 0 else "unknown"
    known = format_usd(known_cost, style="table")
    if unknown_calls > 0:
        return f"{known} ({unknown_calls} unmetered)"
    return known


def _format_usage_source_for_display(row: dict[str, Any]) -> str:
    api_calls = int(row.get("api_usage_calls") or 0)
    est_calls = int(row.get("estimate_usage_calls") or 0)
    corrected_calls = int(row.get("corrected_usage_calls") or 0)
    parts = [f"api {api_calls}", f"est {est_calls}"]
    if corrected_calls > 0:
        parts.append(f"corrected {corrected_calls}")
    return " | ".join(parts)


def _resolve_chat_mode_alias(value: str) -> str | None:
    key = value.strip().lower()
    return _CHAT_MODE_ALIASES.get(key)


def _resolve_trace_level(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in _CHAT_TRACE_LEVELS:
        return normalized
    return None


def _chat_trace_level(session: Any) -> str:
    surface = getattr(session, "surface", None)
    level = getattr(surface, "trace_level", None)
    if isinstance(level, str):
        normalized = level.strip().lower()
        if normalized in _CHAT_TRACE_LEVELS:
            return normalized
    return "compact"


def _session_skill_listing(session: Any) -> tuple[bool, tuple[Any, ...], tuple[Any, ...]]:
    cfg = getattr(session, "cfg", None)
    enabled = resolve_skills_enabled(cfg) if isinstance(cfg, AppConfig) else True
    ordered_obj = getattr(session, "skills_ordered", None)
    ordered = (
        tuple(item for item in ordered_obj if item is not None)
        if isinstance(ordered_obj, tuple)
        else tuple(item for item in (ordered_obj or []) if item is not None)
    )
    registry_obj = getattr(session, "skill_registry", None)
    if ordered or isinstance(registry_obj, dict):
        issues = tuple(getattr(session, "skill_discovery_issues", ()) or ())
        return enabled, ordered, issues
    discovered = _discover_skills_for_path(path=Path(getattr(session, "root", Path("."))))
    return enabled, tuple(discovered.ordered), tuple(discovered.issues)


def _set_chat_trace_level(*, session: Any, level: str) -> str:
    normalized = _resolve_trace_level(level) or "compact"
    surface = getattr(session, "surface", None)
    if surface is None:
        return normalized
    setter = getattr(surface, "set_trace_level", None)
    if callable(setter):
        applied = setter(normalized)
        if isinstance(applied, str):
            applied_normalized = _resolve_trace_level(applied)
            if applied_normalized is not None:
                return applied_normalized
        return normalized
    try:
        surface.trace_level = normalized
    except Exception:  # noqa: BLE001
        return normalized
    return _resolve_trace_level(str(getattr(surface, "trace_level", normalized))) or normalized


def _emit_plan_mode_trace(
    *,
    session: Any,
    message: str,
    full_only: bool = False,
    source: str = "plan_mode",
) -> None:
    clean = message.strip()
    if not clean:
        return
    trace_level = _chat_trace_level(session)
    if trace_level == "off":
        return
    if full_only and trace_level != "full":
        return

    store = getattr(session, "store", None)
    append = getattr(store, "append", None)
    if callable(append):
        try:
            append("progress", {"message": clean, "source": source})
        except Exception:  # noqa: BLE001
            pass

    surface = getattr(session, "surface", None)
    handler = getattr(surface, "on_progress_update", None)
    if callable(handler):
        try:
            handler(clean)
        except Exception:  # noqa: BLE001
            return


def _emit_forge_planner_trace(
    *,
    session: Any,
    message: str,
    full_only: bool = False,
) -> None:
    _emit_plan_mode_trace(
        session=session,
        message=message,
        full_only=full_only,
        source="forge_planner",
    )


def _make_forge_swarm_trace_sink(
    *,
    session: Any,
    paths: Any,
    console: Console,
) -> SerializedSwarmTraceSink:
    surface = getattr(session, "surface", None)
    store = getattr(session, "store", None)
    return SerializedSwarmTraceSink(
        artifact_path=Path(paths.execution_dir) / "trace" / "swarm_trace.jsonl",
        trace_level=_chat_trace_level(session),
        surface=surface,
        session_store=store,
        console=console,
        store_source="forge_swarm",
    )


def _make_plan_mode_delta_trace_callback(*, session: Any) -> Callable[[str], None] | None:
    trace_level = _chat_trace_level(session)
    if trace_level == "off":
        return None

    seen_chars = 0
    started = False
    last_full_bucket = -1

    def _on_text_delta(delta: str) -> None:
        nonlocal seen_chars, started, last_full_bucket
        if not delta:
            return
        seen_chars += len(delta)
        if not started:
            _emit_plan_mode_trace(session=session, message="Receiving planner output...")
            started = True
        if trace_level != "full":
            return
        bucket = seen_chars // 320
        if bucket <= last_full_bucket:
            return
        last_full_bucket = bucket
        _emit_plan_mode_trace(
            session=session,
            message=f"Planner draft progress: ~{seen_chars} chars captured.",
            full_only=True,
        )

    return _on_text_delta


__all__ = [name for name in globals() if (not name.startswith("__") or name == "__version__")]
