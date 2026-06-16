from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..agent import _patchable
from ..background_runner import (
    DisabledBackgroundRunner,
    LazyBackgroundShellRunner,
    build_background_shell_runner_from_settings,
)
from ..branding import env_get
from ..compaction.conversation_compactor import ConversationCompactor
from ..compaction.settings import resolve_compaction_settings
from ..compaction.tool_output_offload import ToolOutputOffloader
from ..config import (
    AppConfig,
    ConfigError,
    get_api_key,
    resolve_api_key,
    resolve_llm_enable_thinking,
    resolve_llm_reasoning_effort,
    resolve_llm_timeout_s,
    resolve_prompt_cache_key,
    resolve_prompt_cache_retention,
    resolve_role_temperature,
)
from ..custom_tools import CustomToolSessionState, build_custom_tool_session_state
from ..extensions.activation import (
    ActivationDecision,
    WorkspaceTrustPromptFn,
    WorkspaceTrustPromptRequest,
)
from ..hooks import (
    HOOK_AUDIT_ARTIFACT_PARTS,
    HookDispatcher,
    HookDispatchResult,
    ResolvedHookConfig,
    load_resolved_hooks_config,
)
from ..llm.base import ChatClient
from ..llm.factory import _resolve_base_url, make_llm_client
from ..llm.metadata import PROVIDER_METADATA_KEY, assistant_message_from_response
from ..llm.openai_compat import OpenAICompatClient as _OpenAICompatClient
from ..llm.protocols import OPENAI_COMPAT_PROTOCOL, get_provider_protocol_capabilities
from ..mcp.config import load_resolved_mcp_config
from ..mcp.manager import ForgeTaskScopedMcpManager, McpManager, create_mcp_manager
from ..model_metadata_policy import ActiveModelRef, evaluate_active_model_metadata_policy
from ..model_registry import ModelRegistry, resolve_model_provider_key
from ..model_router import ROLE_CODING, ROLE_COMPACTOR, ROLE_ROUTER, resolve_model_for_role
from ..profiles import get_active_profile
from ..repo_scan import scan_workspace as scan_workspace
from ..request_estimation import estimate_request_token_breakdown
from ..runtime_context_features import resolve_runtime_context_features
from ..runtime_kind import RuntimeKind, resolve_session_runtime_kind
from ..sandbox_runner import (
    DisabledShellRunner,
    LazyShellRunner,
    build_shell_runner,
    build_shell_runner_from_settings,
)
from ..sandbox_settings import resolve_shell_sandbox_settings
from ..session_store import SessionStore, make_session_id, resolve_sessions_dir
from ..skills import ConventionDocument, SkillBundle, SkillCatalogEntry
from ..step_budget import StepBudgetRuntime
from ..subagents import SubagentDefinition
from ..surface import ApprovalRequest, NoopSurface, StatusEvent
from ..surface.base import Surface
from ..terminal_manager import TerminalManager
from ..tools.registry import iter_builtin_tool_metadata
from ..usage_tracker import ContextLeft, UsageSummary, build_usage_record, compute_context_left
from ..verify_gate import (
    ResolvedVerifyCommands,
    is_authoritative_verify_command_selection,
    verification_selection_payload,
)
from ..workspace_binding import WorkspaceBinding
from .errors import SessionWorkdirError
from .prompt_context import (
    _build_plugin_activation_index,
    _component_plugin_allowed,
    _merge_dropped_counts,
    _normalize_workspace_relpath,
    _PluginActivationIndex,
    _repo_summary_data,
    _resolve_requested_workdir_within_workspace,
    _session_verify_command_selection,
    _workspace_relpath_for_path,
    _WorkspaceGroundingDescriptor,
    prepare_session_prompt_context,
    resolve_session_active_workdir_path,
    resolve_session_active_workdir_relpath,
    resolve_workdir_relpath_within_workspace,
    set_session_active_workdir,
)
from .routing import (
    _ROUTING_MODE_AUTO,
    _emit_assistant_message_events,
    _main_agent_chat,
    _resolve_routing_mode,
    _rewrite_final_summary_for_language,
)
from .tools_assembly import (
    ToolDef,
    _custom_tools_write_scope_restricted,
    _filter_custom_tool_session_state_for_plugins,
    _filter_mcp_config_for_plugins,
    build_tools,
)
from .turn import (
    _FORCED_FINAL_SUMMARY_SYSTEM_PROMPT_TEMPLATE,
    _looks_like_unexecuted_tool_call_markup,
)
from .turn import run_turn as _run_turn

OpenAICompatClient = _OpenAICompatClient
_DEFAULT_CREATE_MCP_MANAGER = create_mcp_manager


def _build_workspace_trust_prompt(
    *,
    surface: Surface,
    non_interactive: bool,
) -> WorkspaceTrustPromptFn | None:
    if non_interactive or env_get("SYLLIPTOR_CI") == "1":
        return None

    def prompt(request: WorkspaceTrustPromptRequest) -> bool:
        preview = (
            f"Workspace: {request.repo_root}\n"
            f"Overrides SHA-256: {request.overrides_sha256}\n"
            f"Project enables: {', '.join(request.plugins_added) or '-'}\n"
            f"Project disables: {', '.join(request.plugins_removed) or '-'}"
        )
        decision = surface.request_approval(
            ApprovalRequest(
                kind="workspace_trust",
                reason="Trust this workspace's plugin enable/disable overrides?",
                preview=preview,
                files=[request.repo_root],
                metadata=request.model_dump(mode="json"),
            )
        )
        return bool(decision.allow)

    return prompt


def _hook_plugin_id(hook_id: str | None, index: _PluginActivationIndex) -> str | None:
    raw = str(hook_id or "").strip()
    if "." not in raw:
        return None
    return index.slug_to_plugin_id.get(raw.split(".", 1)[0])


def _filter_hooks_config_for_plugins(
    *,
    config: ResolvedHookConfig,
    activation_decision: ActivationDecision,
    index: _PluginActivationIndex,
) -> tuple[ResolvedHookConfig, Counter[str]]:
    dropped_counts: Counter[str] = Counter()
    groups_by_event: dict[str, tuple[Any, ...]] = {}
    for event_name, groups in config.groups_by_event.items():
        kept_groups = []
        for group in groups:
            kept_hooks = tuple(
                hook
                for hook in group.hooks
                if _component_plugin_allowed(
                    _hook_plugin_id(hook.id, index),
                    activation_decision,
                    dropped_counts,
                )
            )
            if kept_hooks:
                kept_groups.append(replace(group, hooks=kept_hooks))
        groups_by_event[event_name] = tuple(kept_groups)
    return (
        ResolvedHookConfig(
            groups_by_event=groups_by_event,
            loaded_paths=config.loaded_paths,
            untrusted_project_paths=config.untrusted_project_paths,
        ),
        dropped_counts,
    )


def _make_session_llm_client(
    *,
    cfg: AppConfig,
    api_key: str,
    model: str,
    timeout_s: float | None,
    temperature: float,
    prompt_cache_key: str | None,
    prompt_cache_retention: str | None,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
) -> ChatClient:
    openai_client_cls = _patchable("OpenAICompatClient", OpenAICompatClient)
    if openai_client_cls is _OpenAICompatClient:
        return make_llm_client(
            cfg=cfg,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

    profile = get_active_profile(cfg)
    return openai_client_cls(
        base_url=_resolve_base_url(cfg=cfg, profile=profile),
        api_key=api_key,
        model=model,
        timeout_s=60.0 if timeout_s is None else timeout_s,
        temperature=temperature,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        extra_headers=profile.extra_headers,
    )


def _git_branch(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "-"
    if proc.returncode != 0:
        return "-"
    branch = proc.stdout.strip()
    return branch or "-"


def _git_is_dirty(root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", os.fspath(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def _surface_needs_startup_git_status(surface: Surface) -> bool:
    if isinstance(surface, NoopSurface):
        return False
    # RichSurface stores this flag internally; when hidden, skip expensive startup git probes.
    show_status_line = getattr(surface, "_show_status_line", None)
    if isinstance(show_status_line, bool):
        return show_status_line
    return True


def _meaningful_surface_warning_handler(surface: Surface | object) -> Callable[[str], None] | None:
    surface_cls = getattr(surface, "__class__", None)
    handler = getattr(surface, "emit_warning", None)
    if callable(handler):
        cls_handler = getattr(surface_cls, "emit_warning", None)
        if cls_handler is not getattr(NoopSurface, "emit_warning", None):
            return handler
    return _meaningful_surface_legacy_warning_handler(surface)


def _meaningful_surface_legacy_warning_handler(
    surface: Surface | object,
) -> Callable[[str], None] | None:
    surface_cls = getattr(surface, "__class__", None)
    handler = getattr(surface, "on_warning", None)
    if not callable(handler):
        return None
    cls_handler = getattr(surface_cls, "on_warning", None)
    if cls_handler is getattr(NoopSurface, "on_warning", None):
        return None
    return handler


def _repo_summary(root: Path) -> str:
    return _repo_summary_data(root).text


def _disable_unsupported_native_streaming(
    *,
    cfg: AppConfig,
) -> tuple[AppConfig, str | None]:
    if not bool(getattr(cfg, "stream", False)):
        return cfg, None
    profile = get_active_profile(cfg)
    protocol = str(profile.protocol or OPENAI_COMPAT_PROTOCOL).strip()
    if protocol == OPENAI_COMPAT_PROTOCOL:
        return cfg, None
    base_url = _resolve_base_url(cfg=cfg, profile=profile)
    provider_key = resolve_model_provider_key(
        cfg=cfg,
        model_name=cfg.model,
        base_url=base_url,
        profile_name=profile.name,
    )
    capabilities = get_provider_protocol_capabilities(
        provider_key=provider_key,
        protocol=protocol,
    )
    streaming_supported = (
        capabilities.supports_streaming
        if capabilities is not None
        else protocol == OPENAI_COMPAT_PROTOCOL
    )
    if streaming_supported:
        return cfg, None
    warning = (
        f"Streaming requested but profile {profile.name!r} uses protocol={protocol!r}, "
        "which does not support streaming yet in Sylliptor; streaming is disabled for this run."
    )
    return cfg.model_copy(update={"stream": False}, deep=True), warning


@dataclass
class AgentSession:
    cfg: AppConfig
    root: Path
    mode: str
    yes: bool
    stream: bool
    routing_mode: str
    max_steps: int
    console: Any | None
    surface: Surface
    store: SessionStore
    client: ChatClient
    model_registry: ModelRegistry
    usage_summary: UsageSummary
    usage_role: str
    tool_output_offloader: ToolOutputOffloader | None
    conversation_compactor: ConversationCompactor | None
    tool_output_offload_enabled: bool
    conversation_summarization_enabled: bool
    compaction_profile: str
    tools: dict[str, ToolDef]
    tool_list: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    startup_messages: list[dict[str, Any]] = field(default_factory=list)
    runtime_kind: RuntimeKind = RuntimeKind.INTERACTIVE_CHAT
    mcp_manager: McpManager | ForgeTaskScopedMcpManager | None = None
    terminal_manager: TerminalManager | None = None
    router_client: Any | None = None
    api_key: str = ""
    api_key_source: str = "missing"
    shell_runner: Any | None = None
    no_log: bool = False
    non_interactive: bool = False
    one_shot_execution: bool = False
    enable_chat_turn_step_budget: bool = False
    chat_turn_fixed_override: int | None = None
    verification_enabled: bool = True
    effective_verification_commands: list[str] = field(default_factory=list)
    authoritative_verification_commands: list[str] | None = None
    verification_selection_source: str = ""
    verification_selection_reason: str = ""
    verification_contract_type: str = ""
    verification_authoritative: bool = False
    deny_write_prefixes: list[str] | None = None
    allow_write_globs: list[str] | None = None
    session_log_dir_override: Path | None = None
    skills_enabled: bool = True
    skills_auto_invoke: bool = True
    skill_registry: dict[str, SkillBundle] | None = None
    skills_ordered: tuple[SkillBundle, ...] = ()
    skill_discovery_issues: tuple[Any, ...] = ()
    skill_catalog_entries: tuple[SkillCatalogEntry, ...] = ()
    repo_conventions: tuple[ConventionDocument, ...] = ()
    subagents_enabled: bool = False
    enforce_explicit_subagent_requests: bool = True
    subagent_depth: int = 0
    subagent_registry: dict[str, SubagentDefinition] | None = None
    step_budget_runtime: StepBudgetRuntime | None = None
    planner_workspace_context: dict[str, Any] | None = None
    workspace_grounding: _WorkspaceGroundingDescriptor | None = None
    focus_dir: Path | None = None
    focus_relpath: str = "."
    workspace_kind: str = "plain_dir"
    binding_requested_path: str | None = None
    binding_source: str | None = None
    binding_risk_level: str | None = None
    binding_created_path: bool | None = None
    active_workdir_relpath: str = "."
    session_source: str = "startup"
    session_source_metadata: dict[str, Any] = field(default_factory=dict)
    pinned_prefix_len: int = 0
    startup_context_baseline_tokens: int = 0
    workspace_touched_paths: set[str] = field(default_factory=set)
    custom_tool_session_state: CustomToolSessionState | None = None
    hook_dispatcher: HookDispatcher | None = None

    def close(self, *, reason: str = "session_close") -> None:
        if self.terminal_manager is not None:
            try:
                self.terminal_manager.shutdown_all()
            except Exception as exc:  # noqa: BLE001
                # Session teardown must continue even if terminal shutdown hits an unexpected bug.
                self._hook_warning(
                    f"Terminal manager shutdown failed: {exc}",
                    code="terminal_shutdown_failed",
                )
        try:
            cwd, active_workdir_relpath = self._hook_runtime_context()
            self._safe_dispatch_hooks(
                lambda: self.hook_dispatcher.fire_session_end(
                    cwd=cwd,
                    active_workdir_relpath=active_workdir_relpath,
                    payload={
                        "reason": reason,
                        "mode": self.mode,
                        "runtime_kind": self.runtime_kind.value,
                        "session_source": self.session_source,
                        "session_source_metadata": copy.deepcopy(self.session_source_metadata),
                        "workspace_root": os.fspath(self.root),
                        "focus_dir": os.fspath(self.focus_dir or self.root),
                        "focus_relpath": self.focus_relpath,
                        "active_workdir": os.fspath(cwd),
                        "active_workdir_relpath": active_workdir_relpath,
                        "workspace_kind": self.workspace_kind,
                        "usage_role": self.usage_role,
                        "message_count": len(self.messages),
                        "pinned_prefix_len": self.pinned_prefix_len,
                        "subagent_depth": self.subagent_depth,
                        "skills_enabled": self.skills_enabled,
                        "subagents_enabled": self.subagents_enabled,
                    },
                )
            )
            if self.mcp_manager is not None:
                self.mcp_manager.close()
        finally:
            self.store.close()

    def _hook_warning(self, message: str, *, code: str = "hook_warning") -> None:
        clean = str(message or "").strip()
        if not clean:
            return
        self.store.append("warning", {"warning": code, "message": clean})
        surface_on_warning = _meaningful_surface_warning_handler(self.surface)
        if callable(surface_on_warning):
            surface_on_warning(clean)
        else:
            warnings.warn(clean, stacklevel=2)

    def _safe_dispatch_hooks(
        self,
        dispatcher_call: Callable[[], HookDispatchResult],
    ) -> HookDispatchResult:
        if self.hook_dispatcher is None:
            return HookDispatchResult()
        try:
            result = dispatcher_call()
        except Exception as exc:  # noqa: BLE001
            self._hook_warning(
                f"Lifecycle hook dispatch failed: {exc}",
                code="hook_dispatch_failed",
            )
            return HookDispatchResult()
        for notice in result.system_notices:
            self._hook_notice(notice)
        return result

    def _hook_notice(self, message: str) -> None:
        clean = str(message or "").strip()
        if not clean:
            return
        self.store.append("hook_notice", {"message": clean})
        handler = getattr(self.surface, "on_notice", None)
        if callable(handler):
            handler(clean)
            return
        fallback = _meaningful_surface_legacy_warning_handler(self.surface)
        if callable(fallback):
            fallback(clean)

    def _hook_runtime_context(self) -> tuple[Path, str]:
        return (
            resolve_session_active_workdir_path(self),
            resolve_session_active_workdir_relpath(self),
        )

    def _append_hook_messages(
        self,
        *,
        event_name: str,
        system_messages: Iterable[str] = (),
        user_messages: Iterable[str] = (),
        pinned: bool = False,
    ) -> int:
        appended_count = 0
        for role, messages in (("system", system_messages), ("user", user_messages)):
            for raw_message in messages:
                text = str(raw_message or "").strip()
                if not text:
                    continue
                self.messages.append({"role": role, "content": text})
                appended_count += 1
                self.store.append(
                    "hook_message_added",
                    {
                        "event_name": event_name,
                        "role": role,
                        "chars": len(text),
                        "pinned": pinned,
                    },
                )
        if pinned and appended_count > 0:
            self.pinned_prefix_len += appended_count
        return appended_count

    def context_left(self) -> ContextLeft:
        compaction_settings = resolve_compaction_settings(self.cfg)
        return compute_context_left(
            messages=self.messages,
            model_name=self.client.model,
            registry=self.model_registry,
            tool_list=self.tool_list,
            pinned_prefix_len=self.pinned_prefix_len,
            safety_margin_tokens=compaction_settings.safety_margin_tokens,
            startup_baseline_tokens=self.startup_context_baseline_tokens,
        )

    @staticmethod
    def _normalize_visible_assistant_text(text: str) -> str:
        return str(text or "").strip()

    def _emit_assistant_message_if_changed(
        self,
        *,
        text: str,
        prior_visible_text: str = "",
        extra_payload: dict[str, Any] | None = None,
        streamed_text_emitted: bool = False,
    ) -> str:
        normalized_text = self._normalize_visible_assistant_text(text)
        if not normalized_text:
            if extra_payload:
                payload = {"content": text}
                payload.update(extra_payload)
                self.store.append("assistant_message", payload)
            return self._normalize_visible_assistant_text(prior_visible_text)
        if normalized_text == self._normalize_visible_assistant_text(prior_visible_text):
            if extra_payload:
                payload = {"content": text}
                payload.update(extra_payload)
                self.store.append("assistant_message", payload)
            return normalized_text
        payload = {"content": text}
        if extra_payload:
            payload.update(extra_payload)
        self.store.append("assistant_message", payload)
        _emit_assistant_message_events(
            self.surface,
            text,
            streamed_text_emitted=streamed_text_emitted,
        )
        self.surface.on_assistant_message_done(text)
        return normalized_text

    def _emit_final_assistant_text(
        self,
        *,
        final_text: str,
        assistant_response: Any | None = None,
        language: str = "",
        script: str = "",
        explicit_language_override: bool = False,
        prior_visible_text: str = "",
        streamed_text_emitted: bool = False,
    ) -> str:
        emitted_text = str(final_text or "").strip()
        emitted_text, rewrite_payload = _rewrite_final_summary_for_language(
            client=self.client,
            final_text=emitted_text,
            language=language,
            script=script,
            explicit_language_override=explicit_language_override,
        )
        if rewrite_payload is not None:
            self.store.append("final_summary_rewrite", rewrite_payload)
        assistant_message = None
        if assistant_response is not None:
            candidate_message = assistant_message_from_response(
                assistant_response,
                content=emitted_text,
            )
            if PROVIDER_METADATA_KEY in candidate_message:
                assistant_message = candidate_message
        extra_payload = {"message": assistant_message} if assistant_message is not None else None
        self._emit_assistant_message_if_changed(
            text=emitted_text,
            prior_visible_text=prior_visible_text,
            extra_payload=extra_payload,
            streamed_text_emitted=streamed_text_emitted,
        )
        if assistant_message is not None:
            self.messages.append(assistant_message)
        self.store.append("final", {"content": emitted_text})
        return emitted_text

    def _forced_final_summary_activity_snapshot(self) -> dict[str, Any]:
        tool_calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
        read_paths: list[str] = []
        listed_paths: list[str] = []
        edited_paths: list[str] = []
        verification_commands: list[str] = []
        shell_commands: list[str] = []
        other_actions: list[str] = []
        failed_actions: list[str] = []

        def _append_unique(items: list[str], value: str) -> None:
            clean = str(value or "").strip()
            if clean and clean not in items:
                items.append(clean)

        def _path_arg(args: dict[str, Any]) -> str:
            for key in ("path", "file", "target", "target_path"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        def _command_arg(args: dict[str, Any]) -> str:
            for key in ("command", "cmd"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        for message in self.messages:
            role = str(message.get("role") or "")
            if role == "assistant":
                for raw_call in message.get("tool_calls") or []:
                    if not isinstance(raw_call, dict):
                        continue
                    call_id = str(raw_call.get("id") or "").strip()
                    function = raw_call.get("function")
                    if not call_id or not isinstance(function, dict):
                        continue
                    name = str(function.get("name") or "").strip()
                    raw_args = function.get("arguments")
                    args: dict[str, Any] = {}
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:  # noqa: BLE001
                            parsed_args = None
                        if isinstance(parsed_args, dict):
                            args = parsed_args
                    if name:
                        tool_calls_by_id[call_id] = (name, args)
                continue
            if role != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "").strip()
            name, args = tool_calls_by_id.get(call_id, ("", {}))
            if not name:
                continue
            content = message.get("content")
            result: Any = None
            if isinstance(content, str) and content.strip():
                try:
                    result = json.loads(content)
                except Exception:  # noqa: BLE001
                    result = None
            failed = isinstance(result, dict) and "error" in result
            path = _path_arg(args)
            command = _command_arg(args)
            if failed:
                label = f"{name} {path}".strip() if path else name
                _append_unique(failed_actions, label)
                continue
            if name == "fs_read" and path:
                _append_unique(read_paths, path)
            elif name == "fs_list" and path:
                _append_unique(listed_paths, path)
            elif name in {"fs_write", "fs_edit", "apply_patch"} and path:
                _append_unique(edited_paths, path)
            elif name in {"shell", "shell_command", "shell_run", "verify_run"} and command:
                if name == "verify_run":
                    _append_unique(verification_commands, command)
                else:
                    _append_unique(shell_commands, command)
            else:
                label = f"{name} {path}".strip() if path else name
                _append_unique(other_actions, label)

        return {
            "read_paths": read_paths,
            "listed_paths": listed_paths,
            "edited_paths": edited_paths,
            "verification_commands": verification_commands,
            "shell_commands": shell_commands,
            "other_actions": other_actions,
            "failed_actions": failed_actions,
        }

    def _forced_final_summary_fallback_text(
        self,
        *,
        termination_cause: str,
        max_steps: int,
        fallback_reason: str,
        latest_assistant_text: str = "",
    ) -> str:
        snapshot = self._forced_final_summary_activity_snapshot()

        def _join_limited(items: list[str], *, limit: int = 8) -> str:
            visible = items[:limit]
            text = ", ".join(visible)
            remaining = len(items) - len(visible)
            if remaining > 0:
                text += f", and {remaining} more"
            return text

        completed: list[str] = []
        read_paths = snapshot["read_paths"]
        listed_paths = snapshot["listed_paths"]
        edited_paths = snapshot["edited_paths"]
        verification_commands = snapshot["verification_commands"]
        shell_commands = snapshot["shell_commands"]
        other_actions = snapshot["other_actions"]
        failed_actions = snapshot["failed_actions"]

        if read_paths:
            completed.append(f"- Read files: {_join_limited(read_paths)}.")
        if listed_paths:
            completed.append(f"- Listed directories: {_join_limited(listed_paths)}.")
        if edited_paths:
            completed.append(f"- Edited files: {_join_limited(edited_paths)}.")
        if verification_commands:
            completed.append(
                f"- Ran verification: {_join_limited(verification_commands, limit=4)}."
            )
        if shell_commands:
            completed.append(f"- Ran shell commands: {_join_limited(shell_commands, limit=4)}.")
        if other_actions:
            completed.append(f"- Ran tools: {_join_limited(other_actions)}.")
        latest = str(latest_assistant_text or "").strip()
        if latest:
            latest = " ".join(latest.split())
            if len(latest) > 180:
                latest = latest[:177].rstrip() + "..."
            completed.append(f"- Last assistant progress note: {latest}")
        if not completed:
            completed.append(
                "- No durable repository change was completed before the turn stopped."
            )

        remaining = [
            "- Continue from the recorded tool results instead of restarting from scratch.",
        ]
        if not edited_paths:
            remaining.append(
                "- Implementation has not started yet; identify the smallest safe fix first."
            )
        if edited_paths and not verification_commands:
            remaining.append("- Run focused verification for the edited files before finalizing.")
        if failed_actions:
            remaining.append(
                f"- Resolve failed tool calls: {_join_limited(failed_actions, limit=5)}."
            )
        remaining.append("- Finish the requested implementation or report a concrete blocker.")

        risks = [
            f"- The turn exhausted its {max_steps}-step budget before completion.",
            "- This fallback was generated from runtime state before the turn terminated.",
        ]
        if not verification_commands:
            risks.append("- No verification result was recorded in this turn.")

        return (
            f"The turn stopped before it could finish ({termination_cause}).\n\n"
            "Completed work:\n" + "\n".join(completed) + "\n\n"
            "Remaining work:\n" + "\n".join(remaining) + "\n\n"
            "Known issues or risks:\n" + "\n".join(risks)
        )

    def _emit_forced_final_summary_before_termination(
        self,
        *,
        reason: str,
        termination_cause: str,
        max_steps: int,
        language: str = "",
        script: str = "",
        explicit_language_override: bool = False,
        latest_assistant_text: str = "",
    ) -> str:
        request_messages = list(self.messages)
        latest_assistant_text = str(latest_assistant_text or "").strip()
        if latest_assistant_text:
            request_messages.append({"role": "assistant", "content": latest_assistant_text})
        request_messages.append(
            {
                "role": "system",
                "content": _FORCED_FINAL_SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(
                    termination_cause=termination_cause
                ),
            }
        )
        self.store.append(
            "forced_final_summary_requested",
            {
                "reason": reason,
                "termination_cause": termination_cause,
                "max_steps": max_steps,
            },
        )

        final_text = ""
        fallback_reason: str | None = None
        fallback_error: str | None = None
        try:
            resp = _main_agent_chat(
                client=self.client,
                messages=request_messages,
                tools=None,
                stream=False,
                on_text_delta=None,
            )
        except Exception as exc:  # noqa: BLE001
            fallback_reason = "finalization_error"
            fallback_error = str(exc)
        else:
            usage = resp.usage
            usage_record = build_usage_record(
                role=self.usage_role,
                requested_model=self.client.model,
                response_model=resp.response_model,
                messages=request_messages,
                response_content=resp.content or "",
                response_tool_calls=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ],
                api_prompt_tokens=(usage.prompt_tokens if usage else None),
                api_completion_tokens=(usage.completion_tokens if usage else None),
                api_total_tokens=(usage.total_tokens if usage else None),
                api_cached_prompt_tokens=(usage.cached_prompt_tokens if usage else None),
                pinned_prefix_len=self.pinned_prefix_len,
                registry=self.model_registry,
            )
            self.usage_summary.add_record(usage_record)
            self.store.append("llm_usage", usage_record.to_payload())
            final_text = str(resp.content or "").strip()
            if resp.tool_calls:
                fallback_reason = "tool_call_response"
            elif not final_text:
                fallback_reason = "blank_response"
            elif _looks_like_unexecuted_tool_call_markup(final_text):
                fallback_reason = "tool_call_markup_response"

        if fallback_reason is not None:
            final_text = self._forced_final_summary_fallback_text(
                termination_cause=termination_cause,
                max_steps=max_steps,
                fallback_reason=fallback_reason,
                latest_assistant_text=latest_assistant_text,
            )
            fallback_payload: dict[str, Any] = {
                "reason": reason,
                "termination_cause": termination_cause,
                "max_steps": max_steps,
                "fallback_reason": fallback_reason,
            }
            if fallback_error:
                fallback_payload["error"] = fallback_error
            self.store.append("forced_final_summary_fallback", fallback_payload)

        emitted_text = self._emit_final_assistant_text(
            final_text=final_text,
            assistant_response=resp if fallback_reason is None else None,
            language=language,
            script=script,
            explicit_language_override=explicit_language_override,
        )
        if fallback_reason is None:
            self.store.append(
                "forced_final_summary_completed",
                {
                    "reason": reason,
                    "termination_cause": termination_cause,
                    "max_steps": max_steps,
                    "content_length": len(emitted_text),
                },
            )
        return emitted_text

    def run_turn(
        self,
        instruction: str,
        *,
        image_paths: list[str] | None = None,
        routing_mode_override: str | None = None,
        ephemeral_system_messages: list[str] | tuple[str, ...] | None = None,
        ephemeral_user_messages: list[str] | tuple[str, ...] | None = None,
        cancellation_token: Any | None = None,
    ) -> int:
        return _run_turn(
            self,
            instruction,
            image_paths=image_paths,
            routing_mode_override=routing_mode_override,
            ephemeral_system_messages=ephemeral_system_messages,
            ephemeral_user_messages=ephemeral_user_messages,
            cancellation_token=cancellation_token,
        )


def create_session(
    *,
    cfg: AppConfig,
    root: Path,
    mode: str,
    yes: bool,
    max_steps: int,
    no_log: bool,
    api_key_override: str | None = None,
    console: Any | None = None,
    deny_write_prefixes: list[str] | None = None,
    allow_write_globs: list[str] | None = None,
    non_interactive: bool = False,
    one_shot_execution: bool = False,
    enable_chat_turn_step_budget: bool = False,
    chat_turn_fixed_override: int | None = None,
    session_log_dir_override: Path | None = None,
    session_id_override: str | None = None,
    surface: Surface | None = None,
    usage_role: str = "main",
    trusted_system_prompt_override: str | None = None,
    trusted_system_prompt_append: str | None = None,
    untrusted_prompt_prelude: str | None = None,
    enable_compaction: bool = True,
    enable_tool_output_offload: bool | None = None,
    enable_conversation_summarization: bool | None = None,
    compaction_profile: str = "chat",
    verification_enabled: bool = True,
    authoritative_verification_commands: list[str] | None = None,
    verify_cmd: list[str] | None = None,
    subagents_enabled: bool | None = None,
    enforce_explicit_subagent_requests: bool = True,
    subagent_depth: int = 0,
    subagent_registry: dict[str, SubagentDefinition] | None = None,
    workspace_binding: WorkspaceBinding | None = None,
    active_workdir_relpath_override: str | None = None,
    runtime_kind: RuntimeKind | str | None = None,
    mcp_manager: McpManager | ForgeTaskScopedMcpManager | None = None,
    session_source: str = "startup",
    session_source_metadata: dict[str, Any] | None = None,
) -> AgentSession:
    surface = surface or NoopSurface()
    resolved_runtime_kind = resolve_session_runtime_kind(
        runtime_kind=runtime_kind,
        one_shot_execution=one_shot_execution,
        subagent_depth=subagent_depth,
    )
    workspace_trust_prompt = _build_workspace_trust_prompt(
        surface=surface,
        non_interactive=non_interactive,
    )
    prompt_context = prepare_session_prompt_context(
        cfg=cfg,
        root=root,
        mode=mode,
        yes=yes,
        deny_write_prefixes=deny_write_prefixes,
        allow_write_globs=allow_write_globs,
        non_interactive=non_interactive,
        one_shot_execution=one_shot_execution,
        verification_enabled=verification_enabled,
        authoritative_verification_commands=authoritative_verification_commands,
        verify_cmd=verify_cmd,
        trusted_system_prompt_override=trusted_system_prompt_override,
        trusted_system_prompt_append=trusted_system_prompt_append,
        untrusted_prompt_prelude=untrusted_prompt_prelude,
        subagents_enabled=subagents_enabled,
        subagent_depth=subagent_depth,
        subagent_registry=subagent_registry,
        workspace_binding=workspace_binding,
        workspace_trust_prompt=workspace_trust_prompt,
    )
    root = prompt_context.root
    workspace_context = prompt_context.workspace_context
    binding_requested_path = prompt_context.binding_requested_path
    binding_source = prompt_context.binding_source
    binding_risk_level = prompt_context.binding_risk_level
    binding_created_path = prompt_context.binding_created_path
    authoritative_verify_commands = prompt_context.authoritative_verify_commands
    session_cfg = prompt_context.session_cfg
    session_cfg, native_streaming_warning_message = _disable_unsupported_native_streaming(
        cfg=session_cfg
    )
    normalized_session_source = str(session_source or "startup").strip().lower() or "startup"
    if normalized_session_source not in {"startup", "resume", "fork"}:
        raise ConfigError("session_source must be one of: startup, resume, fork.")
    if session_source_metadata is not None and not isinstance(session_source_metadata, dict):
        raise ConfigError("session_source_metadata must be an object.")
    normalized_session_source_metadata = (
        copy.deepcopy(session_source_metadata) if session_source_metadata is not None else {}
    )
    if active_workdir_relpath_override is None:
        initial_active_workdir_relpath = _normalize_workspace_relpath(
            workspace_context.focus_relpath
        )
    else:
        initial_active_workdir_relpath = _normalize_workspace_relpath(
            active_workdir_relpath_override
        )
    initial_active_workdir = resolve_workdir_relpath_within_workspace(
        workspace_root=workspace_context.workspace_root,
        relpath=initial_active_workdir_relpath,
    )
    if not initial_active_workdir.exists():
        raise SessionWorkdirError(f"Directory does not exist: {initial_active_workdir}")
    if not initial_active_workdir.is_dir():
        raise SessionWorkdirError(f"Path is not a directory: {initial_active_workdir}")

    if api_key_override is None:
        api_key_resolution = resolve_api_key(session_cfg)
        if api_key_resolution.key is None:
            api_key = get_api_key(session_cfg)
            api_key_source = "missing"
        else:
            api_key = api_key_resolution.key
            api_key_source = api_key_resolution.source
    else:
        api_key = api_key_override.strip()
        if not api_key:
            raise ConfigError("API key is empty.")
        api_key_source = "override"
    coding_temperature = resolve_role_temperature(session_cfg, role="coding")
    review_temperature = resolve_role_temperature(session_cfg, role="review")
    planner_temperature = resolve_role_temperature(session_cfg, role="planner")
    conflict_review_temperature = resolve_role_temperature(session_cfg, role="conflict_review")
    compactor_temperature = resolve_role_temperature(session_cfg, role="compactor")
    chat_temperature = resolve_role_temperature(session_cfg, role="chat")
    llm_timeout_s = resolve_llm_timeout_s(session_cfg)
    llm_enable_thinking = resolve_llm_enable_thinking(session_cfg)
    llm_reasoning_effort = resolve_llm_reasoning_effort(session_cfg)
    registry = ModelRegistry(cfg=session_cfg, api_key=api_key)
    routing_mode = _resolve_routing_mode(session_cfg)
    resolved_subagents_enabled = prompt_context.resolved_subagents_enabled
    resolved_skills_enabled = prompt_context.resolved_skills_enabled
    skills_auto_invoke = prompt_context.skills_auto_invoke
    activation_decision = prompt_context.activation_decision
    plugin_activation_index = _build_plugin_activation_index(root)
    plugin_activation_dropped_counts: Counter[str] = Counter(
        prompt_context.plugin_activation_dropped_counts
    )
    discovered_skills = prompt_context.discovered_skills
    repo_conventions = prompt_context.repo_conventions
    effective_one_shot_execution = prompt_context.effective_one_shot_execution
    resolved_subagent_registry = prompt_context.resolved_subagent_registry
    step_budget_runtime = StepBudgetRuntime()

    session_id = session_id_override.strip() if session_id_override else make_session_id()
    if not session_id:
        session_id = make_session_id()
    if mcp_manager is None:
        create_mcp_manager_fn = _patchable("create_mcp_manager", create_mcp_manager)
        if create_mcp_manager_fn is not _DEFAULT_CREATE_MCP_MANAGER:
            mcp_manager = create_mcp_manager_fn(
                workspace_root=workspace_context.workspace_root,
                runtime_kind=resolved_runtime_kind,
                session_id=session_id,
            )
        else:
            resolved_mcp_config = load_resolved_mcp_config(
                workspace_root=workspace_context.workspace_root
            )
            resolved_mcp_config, mcp_dropped_counts = _filter_mcp_config_for_plugins(
                config=resolved_mcp_config,
                activation_decision=activation_decision,
            )
            plugin_activation_dropped_counts.update(mcp_dropped_counts)
            mcp_manager = McpManager(
                resolved_config=resolved_mcp_config,
                workspace_root=workspace_context.workspace_root,
                runtime_kind=resolved_runtime_kind,
                session_id=session_id,
            )
    compaction_settings = resolve_compaction_settings(session_cfg)
    runtime_context_features = resolve_runtime_context_features(
        settings=compaction_settings,
        enable_compaction=enable_compaction,
        enable_tool_output_offload=enable_tool_output_offload,
        enable_conversation_summarization=enable_conversation_summarization,
        logging_enabled=not no_log,
        explicit_session_artifact_root=session_log_dir_override is not None,
    )
    compaction_enabled = runtime_context_features.any_enabled
    tool_output_offload_enabled = runtime_context_features.tool_output_offload_enabled
    conversation_summarization_enabled = runtime_context_features.conversation_summarization_enabled
    compactor_model_name: str | None = None
    if conversation_summarization_enabled:
        compactor_model_name = resolve_model_for_role(
            cfg=session_cfg,
            role=ROLE_COMPACTOR,
            plan=None,
        )
    router_model_name: str | None = None
    if routing_mode == _ROUTING_MODE_AUTO:
        router_model_name = resolve_model_for_role(
            cfg=session_cfg,
            role=ROLE_ROUTER,
            plan=None,
        )
    active_model_refs = [ActiveModelRef(role=ROLE_CODING, model_name=session_cfg.model)]
    if router_model_name:
        active_model_refs.append(ActiveModelRef(role=ROLE_ROUTER, model_name=router_model_name))
    if compactor_model_name:
        active_model_refs.append(
            ActiveModelRef(role=ROLE_COMPACTOR, model_name=compactor_model_name)
        )
    model_metadata_policy_result = evaluate_active_model_metadata_policy(
        cfg=session_cfg,
        registry=registry,
        active_models=active_model_refs,
    )
    client = _make_session_llm_client(
        cfg=session_cfg,
        api_key=api_key,
        model=session_cfg.model,
        timeout_s=llm_timeout_s,
        temperature=coding_temperature,
        prompt_cache_key=resolve_prompt_cache_key(session_cfg),
        prompt_cache_retention=resolve_prompt_cache_retention(session_cfg),
        enable_thinking=llm_enable_thinking,
        reasoning_effort=llm_reasoning_effort,
    )
    router_client: Any | None = None
    if routing_mode == _ROUTING_MODE_AUTO:
        assert router_model_name is not None
        router_client = _make_session_llm_client(
            cfg=session_cfg,
            api_key=api_key,
            model=router_model_name,
            timeout_s=llm_timeout_s,
            temperature=0.0,
            prompt_cache_key=resolve_prompt_cache_key(session_cfg),
            prompt_cache_retention=resolve_prompt_cache_retention(session_cfg),
            enable_thinking=llm_enable_thinking,
            reasoning_effort=llm_reasoning_effort,
        )
    sessions_dir = (
        session_log_dir_override
        if session_log_dir_override is not None
        else resolve_sessions_dir(session_cfg)
    )
    store = SessionStore(
        enabled=not no_log,
        artifact_persistence_enabled=(not no_log) or session_log_dir_override is not None,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cwd=str(initial_active_workdir),
        repo_root=str(root),
        workspace_root=str(workspace_context.workspace_root),
        focus_dir=str(workspace_context.focus_path),
        git_root=(
            str(workspace_context.git_root) if workspace_context.git_root is not None else None
        ),
        workspace_kind=workspace_context.workspace_kind,
        binding_source=binding_source,
        binding_requested_path=binding_requested_path,
        binding_risk_level=binding_risk_level,
        binding_created_path=binding_created_path,
        runtime_kind=resolved_runtime_kind.value,
        active_workdir=str(initial_active_workdir),
        active_workdir_relpath=initial_active_workdir_relpath,
    )
    surface_on_warning = _meaningful_surface_warning_handler(surface)

    def _emit_hook_warning(message: str) -> None:
        clean = str(message or "").strip()
        if not clean:
            return
        store.append("warning", {"warning": "hook_warning", "message": clean})
        if callable(surface_on_warning):
            surface_on_warning(clean)
        else:
            warnings.warn(clean, stacklevel=2)

    tool_output_offloader: ToolOutputOffloader | None = None
    if tool_output_offload_enabled:
        tool_output_offloader = ToolOutputOffloader(
            artifact_layout=store.session_artifact_layout,
            workspace_root=root,
            threshold_chars=compaction_settings.tool_output_offload_threshold_chars,
            preview_chars=compaction_settings.tool_output_preview_chars,
        )
    system_prompt = prompt_context.system_prompt
    system_prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    effective_deny_write_prefixes = prompt_context.effective_deny_write_prefixes
    effective_allow_write_globs = prompt_context.effective_allow_write_globs
    effective_verification_selection = prompt_context.effective_verification_selection
    effective_verification_commands = prompt_context.effective_verification_commands
    recommended_verification_commands = prompt_context.recommended_verification_commands
    verification_selection_metadata = verification_selection_payload(
        effective_verification_selection,
        authoritative=is_authoritative_verify_command_selection(effective_verification_selection),
    )

    usage_summary = UsageSummary()
    custom_tool_session_state = build_custom_tool_session_state(
        workspace_root=root,
        custom_tools_enabled=bool(getattr(session_cfg, "custom_tools_enabled", True)),
        mode=mode,
        runtime_kind=resolved_runtime_kind,
        built_in_tool_names={spec.name.casefold() for spec in iter_builtin_tool_metadata()},
        write_scope_restricted=_custom_tools_write_scope_restricted(
            mode=mode,
            deny_write_prefixes=deny_write_prefixes,
            allow_write_globs=allow_write_globs,
        ),
    )
    (
        custom_tool_session_state,
        custom_tool_dropped_counts,
    ) = _filter_custom_tool_session_state_for_plugins(
        state=custom_tool_session_state,
        activation_decision=activation_decision,
        index=plugin_activation_index,
    )
    plugin_activation_dropped_counts.update(custom_tool_dropped_counts)
    if activation_decision.untrusted_project_plugin_ids:
        ids = ", ".join(sorted(activation_decision.untrusted_project_plugin_ids))
        warning_message = f"Ignoring untrusted project plugin overrides: {ids}"
        store.append(
            "workspace_trust_untrusted_overrides",
            {"plugin_ids": sorted(activation_decision.untrusted_project_plugin_ids)},
        )
        if callable(surface_on_warning):
            surface_on_warning(warning_message)
        else:
            warnings.warn(warning_message, stacklevel=2)

    if native_streaming_warning_message:
        store.append(
            "warning",
            {
                "warning": "native_streaming_disabled",
                "message": native_streaming_warning_message,
            },
        )
        if callable(surface_on_warning):
            surface_on_warning(native_streaming_warning_message)
        else:
            warnings.warn(native_streaming_warning_message, stacklevel=2)

    store.append(
        "session_start",
        {
            "session_source": normalized_session_source,
            "session_source_metadata": normalized_session_source_metadata,
            "mode": mode,
            "runtime_kind": resolved_runtime_kind.value,
            "max_steps": max_steps,
            "step_budget_policy": session_cfg.step_budget_policy,
            "task_max_steps": session_cfg.task_max_steps,
            "subagent_max_steps": session_cfg.subagent_max_steps,
            "model": session_cfg.model,
            "router_model": router_model_name,
            "base_url": session_cfg.base_url,
            "api_key_source": api_key_source,
            "temperature": session_cfg.temperature,
            "coding_temperature": coding_temperature,
            "review_temperature": review_temperature,
            "planner_temperature": planner_temperature,
            "conflict_review_temperature": conflict_review_temperature,
            "compactor_temperature": compactor_temperature,
            "chat_temperature": chat_temperature,
            "llm_enable_thinking": llm_enable_thinking,
            "llm_reasoning_effort": llm_reasoning_effort,
            "stream": session_cfg.stream,
            "routing_mode": routing_mode,
            "subagents_enabled": resolved_subagents_enabled,
            "skills_enabled": resolved_skills_enabled,
            "skills_auto_invoke": skills_auto_invoke,
            "custom_tools_enabled": bool(getattr(session_cfg, "custom_tools_enabled", True)),
            "custom_tool_count": len(custom_tool_session_state.discovery.effective_tools),
            "custom_tool_issue_count": len(custom_tool_session_state.discovery.issues),
            "discovered_skill_count": len(discovered_skills.ordered),
            "repo_convention_count": len(repo_conventions),
            "skill_discovery_issues": [
                {
                    "source_path": issue.source_path.as_posix(),
                    "message": issue.message,
                }
                for issue in discovered_skills.issues
            ],
            "subagent_depth": subagent_depth,
            "subagent_count": len(resolved_subagent_registry),
            "root": str(root),
            "workspace_root": str(workspace_context.workspace_root),
            "focus_dir": str(workspace_context.focus_path),
            "focus_relpath": workspace_context.focus_relpath,
            "active_workdir": str(initial_active_workdir),
            "active_workdir_relpath": initial_active_workdir_relpath,
            "workspace_kind": workspace_context.workspace_kind,
            "git_root": (
                str(workspace_context.git_root) if workspace_context.git_root is not None else None
            ),
            "has_head_commit": workspace_context.has_head_commit,
            "current_branch": workspace_context.current_branch,
            "binding_requested_path": binding_requested_path,
            "binding_source": binding_source,
            "binding_risk_level": binding_risk_level,
            "binding_created_path": binding_created_path,
            "usage_role": usage_role,
            "yes": yes,
            "non_interactive": non_interactive,
            "one_shot_execution": effective_one_shot_execution,
            "enable_chat_turn_step_budget": enable_chat_turn_step_budget,
            "workspace_grounding": prompt_context.workspace_grounding.to_payload(),
            "chat_turn_fixed_override": chat_turn_fixed_override,
            "verification_enabled": verification_enabled,
            "effective_verification_commands": effective_verification_commands,
            **verification_selection_metadata,
            "model_metadata_policy": model_metadata_policy_result.policy,
            "model_metadata_diagnostics": [
                diagnostic.as_payload() for diagnostic in model_metadata_policy_result.diagnostics
            ],
            "deny_write_prefixes": effective_deny_write_prefixes,
            "allow_write_globs": effective_allow_write_globs,
            "recommended_verification_commands": recommended_verification_commands,
            "authoritative_verification_commands": authoritative_verify_commands,
            "system_prompt_sha256": system_prompt_sha256,
            "requested_enable_compaction": runtime_context_features.requested_enable_compaction,
            "requested_tool_output_offload": (
                runtime_context_features.requested_tool_output_offload
            ),
            "requested_conversation_summarization": (
                runtime_context_features.requested_conversation_summarization
            ),
            "logging_enabled": runtime_context_features.logging_enabled,
            "explicit_session_artifact_root": (
                runtime_context_features.explicit_session_artifact_root
            ),
            "tool_output_offload_artifact_persistence_available": (
                runtime_context_features.tool_output_offload_artifact_persistence_available
            ),
            "compaction_enabled": compaction_enabled,
            "compaction_settings_enabled": runtime_context_features.settings_enabled,
            "tool_output_offload_enabled": tool_output_offload_enabled,
            "compaction_settings_offload_tool_outputs": (
                runtime_context_features.settings_offload_tool_outputs
            ),
            "tool_output_offload_threshold_chars": (
                compaction_settings.tool_output_offload_threshold_chars
            ),
            "tool_output_preview_chars": compaction_settings.tool_output_preview_chars,
            "conversation_summarization_enabled": conversation_summarization_enabled,
            "compaction_profile": compaction_profile,
            "compaction_settings_summarize_conversation": (
                runtime_context_features.settings_summarize_conversation
            ),
            "compaction_recent_user_turns_to_keep": (compaction_settings.recent_user_turns_to_keep),
            "compaction_trigger_ratio": compaction_settings.trigger_ratio,
            "compaction_target_ratio": compaction_settings.target_ratio,
            "compaction_max_chunk_messages": compaction_settings.max_chunk_messages,
            "compaction_safety_margin_tokens": compaction_settings.safety_margin_tokens,
            "compactor_model": compactor_model_name,
            "mcp": mcp_manager.startup_metadata(),
        },
    )

    if callable(surface_on_warning):
        for warning_message in model_metadata_policy_result.warning_messages:
            surface_on_warning(warning_message)
    else:
        for warning_message in model_metadata_policy_result.warning_messages:
            warnings.warn(warning_message, stacklevel=2)

    if conversation_summarization_enabled and compactor_model_name == session_cfg.model:
        store.append(
            "warning",
            {
                "warning": "compactor_model_equals_main_model",
                "model": session_cfg.model,
            },
        )
    for issue in discovered_skills.issues:
        store.append(
            "warning",
            {
                "warning": "skill_discovery_issue",
                "source_path": issue.source_path.as_posix(),
                "message": issue.message,
            },
        )
    for issue in custom_tool_session_state.discovery.issues:
        store.append(
            "warning",
            {
                "warning": "custom_tool_discovery_issue",
                "source_scope": issue.source_scope,
                "source_path": issue.source_path.as_posix(),
                "tool_name": issue.tool_name,
                "code": issue.code,
                "message": issue.message,
            },
        )

    resolved_sandbox_settings = resolve_shell_sandbox_settings(session_cfg)

    def _sandbox_warning_callback(message: str) -> None:
        store.append("sandbox_warning", {"message": message})

    def _load_shell_runner_from_resolved_settings() -> Any:
        patched_build_shell_runner = _patchable("build_shell_runner", build_shell_runner)
        if patched_build_shell_runner is not build_shell_runner:
            return patched_build_shell_runner(
                cfg=session_cfg,
                root=root,
                warning_callback=_sandbox_warning_callback,
            )
        return _patchable(
            "build_shell_runner_from_settings",
            build_shell_runner_from_settings,
        )(
            resolved_sandbox_settings,
            root,
            warning_callback=_sandbox_warning_callback,
        )

    if mode == "readonly":
        runner = DisabledShellRunner(reason="shell_run is disabled in readonly mode.")
        bg_runner = DisabledBackgroundRunner(
            reason="Background shell tools are disabled in readonly mode."
        )
    else:
        runner = LazyShellRunner(_load_shell_runner_from_resolved_settings)
        bg_runner = LazyBackgroundShellRunner(
            lambda: _patchable(
                "build_background_shell_runner_from_settings",
                build_background_shell_runner_from_settings,
            )(
                resolved_sandbox_settings,
                root,
                warning_callback=_sandbox_warning_callback,
            )
        )

    terminal_manager = TerminalManager(
        runner=bg_runner,
        settings=resolved_sandbox_settings,
    )

    try:
        active_workdir_state: dict[str, Any] = {
            "relpath": initial_active_workdir_relpath,
            "session": None,
        }

        def _get_active_workdir_relpath() -> str:
            session_obj = active_workdir_state.get("session")
            if session_obj is not None:
                return resolve_session_active_workdir_relpath(session_obj)
            return _normalize_workspace_relpath(active_workdir_state.get("relpath"))

        def _set_active_workdir(raw_path: str, source: str) -> dict[str, Any]:
            session_obj = active_workdir_state.get("session")
            if session_obj is None:
                workspace_root = workspace_context.workspace_root.resolve()
                current_relpath = _normalize_workspace_relpath(active_workdir_state.get("relpath"))
                current_path = resolve_workdir_relpath_within_workspace(
                    workspace_root=workspace_root,
                    relpath=current_relpath,
                )
                next_path = _resolve_requested_workdir_within_workspace(
                    workspace_root=workspace_root,
                    current_workdir=current_path,
                    requested_path=raw_path,
                )
                next_relpath = _workspace_relpath_for_path(
                    workspace_root=workspace_root,
                    path=next_path,
                )
                changed = next_relpath != current_relpath
                active_workdir_state["relpath"] = next_relpath
                store.update_active_workdir(
                    cwd=os.fspath(next_path),
                    active_workdir_relpath=next_relpath,
                )
                payload = {
                    "source": source,
                    "workspace_root": os.fspath(workspace_root),
                    "focus_dir": os.fspath(workspace_context.focus_path),
                    "focus_relpath": workspace_context.focus_relpath,
                    "previous_active_workdir": os.fspath(current_path),
                    "previous_active_workdir_relpath": current_relpath,
                    "active_workdir": os.fspath(next_path),
                    "active_workdir_relpath": next_relpath,
                    "changed": changed,
                }
                if payload["changed"]:
                    store.append("session_workdir_changed", payload)
                return payload
            return set_session_active_workdir(session_obj, raw_path, source=source)

        def _get_verify_command_selection() -> ResolvedVerifyCommands | None:
            session_obj = active_workdir_state.get("session")
            if session_obj is not None:
                return _session_verify_command_selection(session_obj)
            return prompt_context.effective_verification_selection

        tools = build_tools(
            root=root,
            console=console,
            surface=surface,
            store=store,
            mode=mode,
            yes=yes,
            cfg=session_cfg,
            api_key=api_key,
            max_steps=max_steps,
            no_log=no_log,
            usage_role=usage_role,
            usage_summary=usage_summary,
            model_registry=registry,
            deny_write_prefixes=deny_write_prefixes,
            allow_write_globs=allow_write_globs,
            non_interactive=non_interactive,
            shell_runner=runner,
            terminal_manager=terminal_manager,
            verification_enabled=verification_enabled,
            authoritative_verification_commands=authoritative_verify_commands,
            effective_verification_commands=effective_verification_commands,
            verify_command_selection=prompt_context.effective_verification_selection,
            get_verify_command_selection=_get_verify_command_selection,
            one_shot_execution=effective_one_shot_execution,
            skills_enabled=resolved_skills_enabled,
            skill_registry=discovered_skills.skills,
            subagents_enabled=resolved_subagents_enabled,
            subagent_depth=subagent_depth,
            subagent_registry=resolved_subagent_registry,
            session_log_dir_override=session_log_dir_override,
            step_budget_runtime=step_budget_runtime,
            emit_web_search_runtime_diagnostics=(subagent_depth == 0),
            runtime_kind=resolved_runtime_kind,
            mcp_manager=mcp_manager,
            custom_tool_session_state=custom_tool_session_state,
            get_active_workdir_relpath=_get_active_workdir_relpath,
            set_active_workdir_callback=_set_active_workdir,
            create_session_factory=create_session,
        )
        if mcp_manager is not None and mcp_manager.resolved_config.has_any_config:
            store.append("mcp_catalog_snapshot", mcp_manager.catalog_snapshot_metadata())
        tool_list = [t.as_openai_tool() for t in tools.values()]
        messages: list[dict[str, Any]] = list(prompt_context.messages)
        pinned_prefix_len = prompt_context.pinned_prefix_len
        hooks_config = load_resolved_hooks_config(workspace_context.workspace_root)
        hooks_config, hook_dropped_counts = _filter_hooks_config_for_plugins(
            config=hooks_config,
            activation_decision=activation_decision,
            index=plugin_activation_index,
        )
        plugin_activation_dropped_counts.update(hook_dropped_counts)
        dropped_counts_payload = _merge_dropped_counts(plugin_activation_dropped_counts)
        if dropped_counts_payload:
            store.append(
                "plugin_activation_filter",
                {
                    "enabled_plugin_ids": sorted(activation_decision.enabled_plugin_ids),
                    "dropped_component_counts": dropped_counts_payload,
                },
            )
        hook_dispatcher: HookDispatcher | None = None
        if hooks_config.untrusted_project_paths:
            untrusted_paths = [os.fspath(path) for path in hooks_config.untrusted_project_paths]
            store.append("hook_config_untrusted", {"paths": untrusted_paths})
            for path_text in untrusted_paths:
                _emit_hook_warning(
                    "Ignoring untrusted project hooks config: "
                    f"{path_text}. Run `sylliptor hooks trust --path "
                    f"{os.fspath(workspace_context.workspace_root)}` to allow it."
                )
        if hooks_config.has_any_hooks:
            hook_audit_artifact = (
                store.runtime_artifact_path(*HOOK_AUDIT_ARTIFACT_PARTS) if store.enabled else None
            )
            hook_dispatcher = HookDispatcher(
                config=hooks_config,
                workspace_root=workspace_context.workspace_root,
                repo_root=root,
                session_id=session_id,
                mode=mode,
                runtime_kind=resolved_runtime_kind.value,
                warning_callback=_emit_hook_warning,
                log_callback=store.append,
                audit_callback=(
                    (
                        lambda payload: store.append_artifact_jsonl(
                            *HOOK_AUDIT_ARTIFACT_PARTS, payload=payload
                        )
                    )
                    if store.enabled
                    else None
                ),
            )
            store.append(
                "hook_config_loaded",
                {
                    "loaded_paths": [os.fspath(path) for path in hooks_config.loaded_paths],
                    "events": {
                        event_name: len(groups)
                        for event_name, groups in hooks_config.groups_by_event.items()
                    },
                    "hook_audit_artifact": os.fspath(hook_audit_artifact)
                    if hook_audit_artifact is not None
                    else None,
                },
            )
            try:
                session_start_hook_result = hook_dispatcher.fire_session_start(
                    cwd=initial_active_workdir,
                    active_workdir_relpath=initial_active_workdir_relpath,
                    session_source=normalized_session_source,
                    payload={
                        "session_source": normalized_session_source,
                        "session_source_metadata": copy.deepcopy(
                            normalized_session_source_metadata
                        ),
                        "mode": mode,
                        "runtime_kind": resolved_runtime_kind.value,
                        "workspace_root": os.fspath(workspace_context.workspace_root),
                        "focus_dir": os.fspath(workspace_context.focus_path),
                        "focus_relpath": workspace_context.focus_relpath,
                        "active_workdir": os.fspath(initial_active_workdir),
                        "active_workdir_relpath": initial_active_workdir_relpath,
                        "workspace_kind": workspace_context.workspace_kind,
                        "current_branch": workspace_context.current_branch,
                        "max_steps": max_steps,
                        "non_interactive": non_interactive,
                        "one_shot_execution": effective_one_shot_execution,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                _emit_hook_warning(f"Lifecycle hook dispatch failed: {exc}")
                session_start_hook_result = HookDispatchResult()
            for notice in session_start_hook_result.system_notices:
                _emit_hook_warning(notice)
            if session_start_hook_result.blocked:
                blocked_reason = session_start_hook_result.reason or "session start blocked by hook"
                raise ConfigError(f"Session blocked by hook: {blocked_reason}")
            for hook_message in session_start_hook_result.additional_system_messages:
                messages.append({"role": "system", "content": hook_message})
                pinned_prefix_len += 1
                store.append(
                    "hook_message_added",
                    {
                        "event_name": "SessionStart",
                        "role": "system",
                        "chars": len(hook_message),
                        "pinned": True,
                    },
                )
            for hook_message in session_start_hook_result.additional_user_messages:
                messages.append({"role": "user", "content": hook_message})
                pinned_prefix_len += 1
                store.append(
                    "hook_message_added",
                    {
                        "event_name": "SessionStart",
                        "role": "user",
                        "chars": len(hook_message),
                        "pinned": True,
                    },
                )

        startup_messages = copy.deepcopy(messages)

        conversation_compactor: ConversationCompactor | None = None
        if conversation_summarization_enabled and compactor_model_name:
            compactor_client = _make_session_llm_client(
                cfg=session_cfg,
                api_key=api_key,
                model=compactor_model_name,
                timeout_s=llm_timeout_s,
                temperature=compactor_temperature,
                prompt_cache_key=resolve_prompt_cache_key(session_cfg),
                prompt_cache_retention=resolve_prompt_cache_retention(session_cfg),
                enable_thinking=llm_enable_thinking,
                reasoning_effort=llm_reasoning_effort,
            )
            conversation_compactor = ConversationCompactor(
                root=root,
                artifact_layout=store.session_artifact_layout,
                store=store,
                settings=compaction_settings,
                compactor_client=compactor_client,
                model_registry=registry,
                usage_summary=usage_summary,
                usage_role=usage_role,
                pinned_prefix_len=pinned_prefix_len,
                profile=("execution" if compaction_profile == "execution" else "chat"),
            )

        if _surface_needs_startup_git_status(surface):
            git_branch = _patchable("_git_branch", _git_branch)
            git_is_dirty = _patchable("_git_is_dirty", _git_is_dirty)
            startup_branch = git_branch(root)
            startup_dirty = git_is_dirty(root)
        else:
            startup_branch = "-"
            startup_dirty = False

        startup_context_baseline_tokens = estimate_request_token_breakdown(
            messages=messages,
            tool_list=tool_list,
            pinned_prefix_len=pinned_prefix_len,
        ).total_tokens

        surface.on_status_update(
            StatusEvent(
                mode=mode,
                model=session_cfg.model,
                workspace=os.fspath(root),
                session_id=session_id,
                branch=startup_branch,
                dirty=startup_dirty,
                stream=session_cfg.stream,
                task="-",
            )
        )

        session = AgentSession(
            cfg=session_cfg,
            root=root,
            mode=mode,
            yes=yes,
            stream=session_cfg.stream,
            routing_mode=routing_mode,
            max_steps=max_steps,
            api_key=api_key,
            api_key_source=api_key_source,
            no_log=no_log,
            non_interactive=non_interactive,
            one_shot_execution=effective_one_shot_execution,
            enable_chat_turn_step_budget=enable_chat_turn_step_budget,
            chat_turn_fixed_override=chat_turn_fixed_override,
            verification_enabled=verification_enabled,
            effective_verification_commands=list(effective_verification_commands),
            authoritative_verification_commands=(
                list(authoritative_verify_commands)
                if authoritative_verify_commands is not None
                else None
            ),
            verification_selection_source=str(
                verification_selection_metadata.get("verification_selection_source") or ""
            ),
            verification_selection_reason=str(
                verification_selection_metadata.get("verification_selection_reason") or ""
            ),
            verification_contract_type=str(
                verification_selection_metadata.get("verification_contract_type") or ""
            ),
            verification_authoritative=bool(
                verification_selection_metadata.get("verification_authoritative", False)
            ),
            deny_write_prefixes=(
                list(deny_write_prefixes) if deny_write_prefixes is not None else None
            ),
            allow_write_globs=(list(allow_write_globs) if allow_write_globs is not None else None),
            session_log_dir_override=session_log_dir_override,
            skills_enabled=resolved_skills_enabled,
            skills_auto_invoke=skills_auto_invoke,
            skill_registry=dict(discovered_skills.skills),
            skills_ordered=tuple(discovered_skills.ordered),
            skill_discovery_issues=tuple(discovered_skills.issues),
            skill_catalog_entries=tuple(prompt_context.skill_catalog_entries),
            repo_conventions=tuple(repo_conventions),
            console=console,
            surface=surface,
            store=store,
            client=client,
            model_registry=registry,
            usage_summary=usage_summary,
            usage_role=usage_role,
            tool_output_offloader=tool_output_offloader,
            conversation_compactor=conversation_compactor,
            tool_output_offload_enabled=tool_output_offload_enabled,
            conversation_summarization_enabled=conversation_summarization_enabled,
            compaction_profile=compaction_profile,
            tools=tools,
            tool_list=tool_list,
            messages=messages,
            startup_messages=startup_messages,
            runtime_kind=resolved_runtime_kind,
            mcp_manager=mcp_manager,
            terminal_manager=terminal_manager,
            subagents_enabled=resolved_subagents_enabled,
            enforce_explicit_subagent_requests=bool(enforce_explicit_subagent_requests),
            subagent_depth=subagent_depth,
            subagent_registry=resolved_subagent_registry,
            shell_runner=runner,
            step_budget_runtime=step_budget_runtime,
            planner_workspace_context=prompt_context.planner_workspace_context,
            workspace_grounding=prompt_context.workspace_grounding,
            focus_dir=workspace_context.focus_path,
            focus_relpath=workspace_context.focus_relpath,
            workspace_kind=workspace_context.workspace_kind,
            binding_requested_path=binding_requested_path,
            binding_source=binding_source,
            binding_risk_level=binding_risk_level,
            binding_created_path=binding_created_path,
            active_workdir_relpath=initial_active_workdir_relpath,
            session_source=normalized_session_source,
            session_source_metadata=copy.deepcopy(normalized_session_source_metadata),
            pinned_prefix_len=pinned_prefix_len,
            startup_context_baseline_tokens=startup_context_baseline_tokens,
            router_client=router_client,
            custom_tool_session_state=custom_tool_session_state,
            hook_dispatcher=hook_dispatcher,
        )
        active_workdir_state["session"] = session
        return session
    except Exception:
        try:
            terminal_manager.shutdown_all()
        except Exception as exc:  # noqa: BLE001
            # Startup failure cleanup is best-effort; still close MCP/store below.
            store.append(
                "warning",
                {
                    "warning": "terminal_shutdown_failed",
                    "message": f"Terminal manager shutdown failed: {exc}",
                },
            )
        try:
            mcp_manager.close()
        finally:
            store.close()
        raise
