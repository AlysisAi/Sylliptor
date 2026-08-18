"""
Public Harbor installed-agent adapter for Sylliptor.

This adapter installs a prebuilt Sylliptor wheel into each Terminal-Bench task
container, then runs ``sylliptor run`` non-interactively.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists() and os.fspath(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_SRC_ROOT))

from sylliptor_agent_cli.run_outcome import (  # noqa: E402
    AGENT_FAILURE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    extract_process_exit_code,
    run_outcome_metadata,
)

try:
    from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError as exc:
    if exc.name and not exc.name.startswith("harbor"):
        raise

    class AgentContext:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.metadata: dict[str, Any] | None = None

    class BaseEnvironment:  # type: ignore[no-redef]
        pass

    def with_prompt_template(fn: Any) -> Any:  # type: ignore[no-redef]
        return fn

    class BaseInstalledAgent:  # type: ignore[no-redef]
        def __init__(
            self,
            logs_dir: Path | str = Path("."),
            model_name: str | None = None,
            version: str | None = None,
            extra_env: dict[str, str] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            _ = args, kwargs
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self._version = version
            self._extra_env = dict(extra_env or {})

        def version(self) -> str | None:
            return self._version

        def _get_env(self, key: str) -> str | None:
            if key in self._extra_env:
                return self._extra_env[key]
            return os.environ.get(key)


_WHEEL_DIR = "/tmp/sylliptor-agent"
_CONFIG_DIR = "/tmp/sylliptor-cfg"
_ART_DIR = "/logs/artifacts"
_SESSION_DIR = f"{_ART_DIR}/sylliptor-session"
_CRASH_LOG = f"{_ART_DIR}/sylliptor-crash.jsonl"
_SETUP_DIR = "/installed-agent/sylliptor-source/benchmarks/terminal_bench"
_SETUP_SCRIPT = f"{_SETUP_DIR}/setup.sh"
_SETUP_TIMEOUT_SEC = 1800


class SylliptorAgent(BaseInstalledAgent):
    """Installed-agent wrapper around ``sylliptor run``."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "sylliptor"

    def version(self) -> str | None:
        explicit = self._get_env("SYLLIPTOR_BENCH_VERSION")
        if explicit and explicit.strip():
            return explicit.strip()
        inherited = getattr(self, "_version", None)
        if inherited:
            return str(inherited)
        wheel = self._get_env("SYLLIPTOR_WHEEL")
        return Path(wheel).stem if wheel else "local"

    def _host_wheel_path(self) -> str:
        wheel = self._get_env("SYLLIPTOR_WHEEL")
        if not wheel or not wheel.strip():
            raise RuntimeError(
                "SYLLIPTOR_WHEEL is not set. Point it at the Sylliptor wheel built "
                "from the benchmark branch."
            )
        wheel = wheel.strip()
        if not Path(wheel).is_file():
            raise RuntimeError(f"SYLLIPTOR_WHEEL file not found: {wheel}")
        return wheel

    def _host_setup_script_path(self) -> str:
        setup = self._get_env("SYLLIPTOR_TBENCH_SETUP_SH")
        if setup:
            path = Path(setup)
        else:
            path = Path(__file__).with_name("setup.sh")
        if not path.is_file():
            raise RuntimeError(f"Terminal-Bench setup.sh not found: {path}")
        return path.as_posix()

    def _container_wheel_path(self) -> str:
        return f"{_WHEEL_DIR}/{Path(self._host_wheel_path()).name}"

    def _model(self) -> str:
        model = self._get_env("SYLLIPTOR_MODEL")
        if model and model.strip():
            return model.strip()
        mn = getattr(self, "model_name", None)
        if mn and str(mn).strip():
            return str(mn).strip()
        raise RuntimeError("SYLLIPTOR_MODEL is not set and no model_name was provided.")

    def _base_url(self) -> str:
        base_url = self._get_env("SYLLIPTOR_BASE_URL")
        if not base_url or not base_url.strip():
            raise RuntimeError(
                "SYLLIPTOR_BASE_URL is not set. Point it at an OpenAI-compatible endpoint."
            )
        return base_url.strip()

    def _install_env(self) -> dict[str, str]:
        env = {
            "PYTHONUNBUFFERED": "1",
            "SYLLIPTOR_BASE_URL": self._base_url(),
            "SYLLIPTOR_CONFIG_DIR": _CONFIG_DIR,
            "SYLLIPTOR_INSTALL_SPEC": self._get_env("SYLLIPTOR_INSTALL_SPEC")
            or "sylliptor-agent-cli",
            "SYLLIPTOR_MODEL": self._model(),
            "SYLLIPTOR_MODEL_METADATA_POLICY": "warn",
            "SYLLIPTOR_SETUP_ARTIFACT_DIR": f"{_ART_DIR}/setup",
            "SYLLIPTOR_SETUP_LOG_DIR": "/logs/agent/setup",
            "SYLLIPTOR_TBENCH_WEB_SEARCH_MODE": self._get_env("SYLLIPTOR_WEB_SEARCH_MODE") or "off",
            "SYLLIPTOR_VERIFY_SANDBOX_MODE": "off",
            "SYLLIPTOR_WHEEL": self._container_wheel_path(),
        }
        return env

    def _container_env(self) -> dict[str, str]:
        env = {
            "SYLLIPTOR_CONFIG_DIR": _CONFIG_DIR,
            "CI": "1",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "SYLLIPTOR_SHELL_SANDBOX_MODE": self._get_env("SYLLIPTOR_SHELL_SANDBOX_MODE") or "off",
        }
        api_key = self._get_env("SYLLIPTOR_API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError("SYLLIPTOR_API_KEY is not set on the host.")
        env["SYLLIPTOR_API_KEY"] = api_key.strip()
        env["SYLLIPTOR_BASE_URL"] = self._base_url()
        ws_key = self._get_env("SYLLIPTOR_WEB_SEARCH_API_KEY")
        if ws_key:
            env["SYLLIPTOR_WEB_SEARCH_API_KEY"] = ws_key
        llm_timeout = self._get_env("SYLLIPTOR_LLM_TIMEOUT_S")
        if llm_timeout:
            env["SYLLIPTOR_LLM_TIMEOUT_S"] = llm_timeout
        return env

    def _install_command(self) -> str:
        return (
            f"mkdir -p {shlex.quote(_WHEEL_DIR)} {shlex.quote(_SETUP_DIR)} "
            f"{shlex.quote(_ART_DIR)}/setup /logs/agent/setup "
            f"&& chmod 1777 {shlex.quote(_WHEEL_DIR)} "
            f"&& chmod +x {shlex.quote(_SETUP_SCRIPT)} "
            f"&& {shlex.quote(_SETUP_SCRIPT)}"
        )

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(  # type: ignore[attr-defined]
            environment,
            command=(
                f"mkdir -p {shlex.quote(_WHEEL_DIR)} {shlex.quote(_SETUP_DIR)} "
                f"{shlex.quote(_ART_DIR)}/setup /logs/agent/setup "
                f"&& chmod 1777 {shlex.quote(_WHEEL_DIR)}"
            ),
        )
        await environment.upload_file(self._host_setup_script_path(), _SETUP_SCRIPT)
        await environment.upload_file(self._host_wheel_path(), self._container_wheel_path())
        await self.exec_as_root(  # type: ignore[attr-defined]
            environment,
            command=self._install_command(),
            env=self._install_env(),
            timeout_sec=_SETUP_TIMEOUT_SEC,
        )

    def _config_set_cmds(self) -> list[str]:
        cmds: list[str] = []
        base_url = self._get_env("SYLLIPTOR_BASE_URL")
        if base_url:
            cmds.append(f"sylliptor config set base_url {shlex.quote(base_url)}")
        model = self._model()
        if model:
            cmds.append(f"sylliptor config set model {shlex.quote(model)}")
        ws = {
            "web_search_mode": "SYLLIPTOR_WEB_SEARCH_MODE",
            "web_search_adapter": "SYLLIPTOR_WEB_SEARCH_ADAPTER",
            "web_search_base_url": "SYLLIPTOR_WEB_SEARCH_BASE_URL",
            "web_search_model": "SYLLIPTOR_WEB_SEARCH_MODEL",
            "web_search_timeout_s": "SYLLIPTOR_WEB_SEARCH_TIMEOUT_S",
        }
        for cfg_key, env_name in ws.items():
            val = self._get_env(env_name)
            if val:
                cmds.append(f"sylliptor config set {cfg_key} {shlex.quote(val)}")
        steps = self._get_env("SYLLIPTOR_MAX_STEPS")
        if steps:
            for cfg_key in ("max_steps", "task_max_steps", "subagent_max_steps"):
                cmds.append(f"sylliptor config set {cfg_key} {shlex.quote(steps)}")
        cmds.append(f"sylliptor config set session_log_dir {shlex.quote(_SESSION_DIR)}")
        cmds.append(f"sylliptor config set crash_diagnostic_log_path {shlex.quote(_CRASH_LOG)}")
        return cmds

    def _extra_cli_args(self) -> list[str]:
        raw = self._get_env("SYLLIPTOR_EXTRA_ARGS")
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid SYLLIPTOR_EXTRA_ARGS: {exc}") from exc

    def _build_run_command(self, instruction: str, *, model: str, base_url: str) -> str:
        profile = (self._get_env("SYLLIPTOR_RUN_PROFILE") or "auto").strip().lower()
        parts = [
            "sylliptor",
            "run",
            "--path",
            ".",
            "--allow-broad-workspace",
            "--yes",
            "--model",
            shlex.quote(model),
            "--base-url",
            shlex.quote(base_url),
            "--api-key-env",
            "SYLLIPTOR_API_KEY",
        ]
        if profile == "benchmark":
            parts.append("--benchmark")
        else:
            parts += ["--mode", shlex.quote(profile)]

        steps = self._get_env("SYLLIPTOR_MAX_STEPS")
        if steps:
            parts += ["--max-steps", shlex.quote(steps)]
        deadline = self._get_env("SYLLIPTOR_DEADLINE_SECONDS")
        if deadline:
            parts += ["--deadline-seconds", shlex.quote(deadline)]
        parts.extend(shlex.quote(part) for part in self._extra_cli_args())
        parts += ["--", shlex.quote(instruction)]

        return (
            "mkdir -p /logs/agent 2>/dev/null; "
            + " ".join(parts)
            + f" </dev/null 2>&1 | tee {_ART_DIR}/sylliptor.txt /logs/agent/sylliptor.txt"
        )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        _ = context
        env = self._container_env()
        model = self._model()
        base_url = env["SYLLIPTOR_BASE_URL"]
        await self.exec_as_agent(  # type: ignore[attr-defined]
            environment,
            command=f"mkdir -p {shlex.quote(_SESSION_DIR)} /logs/agent 2>/dev/null || true",
            env=env,
        )
        await self.exec_as_agent(  # type: ignore[attr-defined]
            environment,
            command=" && ".join(self._config_set_cmds()),
            env=env,
        )

        cmd = self._build_run_command(instruction, model=model, base_url=base_url)
        try:
            await self.exec_as_agent(  # type: ignore[attr-defined]
                environment,
                command=cmd,
                env=env,
            )
        except Exception as exc:
            exit_code = extract_process_exit_code(exc)
            context.metadata = {
                **(context.metadata or {}),
                **run_outcome_metadata(
                    exit_code if exit_code is not None else AGENT_FAILURE_EXIT_CODE
                ),
            }
            raise
        else:
            context.metadata = {
                **(context.metadata or {}),
                **run_outcome_metadata(SUCCESS_EXIT_CODE),
            }

    def populate_context_post_run(self, context: AgentContext) -> None:
        _ = context
        return
