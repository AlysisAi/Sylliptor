"""Portable mock-provider end-to-end smoke across the execution modes.

A fast (<2 min total), Windows + Linux, default-runnable regression net that drives a
real tiny build -- create a file via the agent, applied to the working tree -- through
the actual CLI (as a subprocess) against the local mock provider. It is the trustworthy
"did I break a real build" signal an autonomous fix-loop needs: the three modes (simple
``run``, ``forge exec``, and ``forge swarm``) must each finish with a
coherent terminal status and the deliverable on disk, with no raw error leaking.

The mock provider (scripts/qa/mock_llm.py) is pattern-driven: an instruction containing
"write file" yields a single ``fs_write`` tool call (pure Python, no shell), so the build
is fully portable. Shell sandboxing is disabled so no Docker/Bubblewrap is required.

Marked ``smoke``; run with ``pytest -m smoke``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.qa.mock_llm import MockLLMServer
from sylliptor_agent_cli.forge import add_task, create_plan_run, load_plan, save_plan
from sylliptor_agent_cli.llm.openai_compat import OpenAICompatClient

QA_MODEL = "qa-mock-model"
# Drives the mock's full-gate path: fs_edit -> verify_run -> git_diff -> done, so the
# build satisfies both the verification and diff-review completion-gate requirements.
_FULL_GATE_BUILD = (
    "raw_proxy_force_diff: implement the smoke change, run the configured verification, "
    "and review the diff before finishing."
)
CLI = (sys.executable, "-m", "sylliptor_agent_cli.cli")
REPO_ROOT = Path(__file__).resolve().parents[1]
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Smoke",
    "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
    "GIT_COMMITTER_NAME": "Smoke",
    "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
}


def _git(repo: Path, args: list[str]) -> None:
    subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_IDENTITY},
    )


def _make_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text("# Smoke\n", encoding="utf-8")
    src = repo / "src"
    src.mkdir(exist_ok=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    _git(repo, ["init", "-q"])
    _git(repo, ["config", "user.name", "Smoke"])
    _git(repo, ["config", "user.email", "smoke@example.invalid"])
    _git(repo, ["checkout", "-B", "main"])
    _git(repo, ["add", "-A"])
    _git(repo, ["commit", "-q", "-m", "seed"])


def _write_config(
    config_dir: Path,
    *,
    base_url: str,
    repo: Path,
    verify_commands: list[str] | None = None,
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "base_url": base_url,
        "model": QA_MODEL,
        "default_mode": "fullaccess",
        "stream": False,
        "routing_mode": "code_only",
        "subagents_enabled": False,
        "skills_enabled": False,
        "max_steps": 8,
        "task_max_steps": 8,
        "subagent_max_steps": 1,
        # A real, assertive, always-passing verification so the completion gate is
        # satisfiable in the smoke (the seeded README always exists). The agent runs it
        # via verify_run on the mock's "run the configured verification" path.
        "verify_commands": verify_commands
        or ["python -c \"import pathlib; assert pathlib.Path('README.md').is_file()\""],
        "profiles": {
            "mock": {
                "name": "mock",
                "protocol": "openai_compat",
                "base_url": base_url,
                "api_key_env": "SYLLIPTOR_API_KEY",
                "default_model": QA_MODEL,
                "extra_headers": {},
                "notes": "mock provider smoke",
            }
        },
        "active_profile": "mock",
        "default_workspace_path": os.fspath(repo.resolve()),
    }
    (config_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _env(work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = os.fspath(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    # Make a bare ``python`` in the configured verify command resolve to the same
    # interpreter running the smoke, on both platforms. Without this, a WSL shell can
    # pick up a non-executable Windows ``python`` stub from PATH interop.
    venv_bin = os.path.dirname(sys.executable)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env.update(
        {
            "SYLLIPTOR_CONFIG_DIR": os.fspath((work_dir / "config").resolve()),
            "SYLLIPTOR_DATA_DIR": os.fspath((work_dir / "data").resolve()),
            "SYLLIPTOR_API_KEY": "mock-key",
            "OPENAI_API_KEY": "mock-key",
            "NO_COLOR": "1",
            "TERM": "xterm-256color",
            "SYLLIPTOR_ROUTING_MODE": "code_only",
            # Portability: no Docker / Bubblewrap required for the smoke.
            "SYLLIPTOR_SHELL_SANDBOX_MODE": "off",
            "SYLLIPTOR_VERIFY_SANDBOX_MODE": "off",
            "SYLLIPTOR_SKILLS_ENABLED": "0",
            "SYLLIPTOR_CONTEXT_WINDOW": "200000",
            "SYLLIPTOR_MAX_OUTPUT_TOKENS": "4096",
        }
    )
    return env


def _run_cli(
    args: list[str], *, cwd: Path, env: dict[str, str], timeout: float = 90.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *args],
        cwd=os.fspath(cwd),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_no_raw_error(result: subprocess.CompletedProcess[str]) -> None:
    combined = f"{result.stdout}\n{result.stderr}"
    assert "LLMError" not in combined, f"raw LLMError leaked:\n{combined}"
    assert "Traceback (most recent call last)" not in result.stderr, (
        f"unhandled traceback leaked:\n{result.stderr}"
    )


def _diagnostic_tool_schema() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "diagnostic_echo",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_mock_provider_tool_rejection_retries_without_tools_and_caches() -> None:
    with MockLLMServer() as server:
        client = OpenAICompatClient(
            base_url=server.base_url,
            api_key="mock-key",
            model=QA_MODEL,
        )
        first = client.chat(
            messages=[{"role": "user", "content": "qa_reject_tools plain answer"}],
            tools=_diagnostic_tool_schema(),
        )
        second = client.chat(
            messages=[{"role": "user", "content": "qa_reject_tools plain answer again"}],
            tools=_diagnostic_tool_schema(),
        )
        requests = list(server.requests)

    assert first.content
    assert second.content
    assert len(requests) == 3
    assert "tools" in requests[0]
    assert "tools" not in requests[1]
    assert "tools" not in requests[2]
    assert first.provider_metadata["transport"] == {
        "tools_omitted": True,
        "tools_omit_reason": "provider_rejected_tool_calling",
        "tools_retry_used": True,
    }
    assert second.provider_metadata["transport"] == {
        "tools_omitted": True,
        "tools_omit_reason": "cached_provider_rejection",
    }


def test_mock_provider_pytest_no_tests_does_not_trip_completion_gate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    with MockLLMServer() as server:
        _write_config(
            tmp_path / "config",
            base_url=server.base_url,
            repo=repo,
            verify_commands=["pytest -q"],
        )
        result = _run_cli(
            [
                "run",
                _FULL_GATE_BUILD,
                "--path",
                os.fspath(repo.resolve()),
                "--allow-broad-workspace",
                "--mode",
                "fullaccess",
                "--base-url",
                server.base_url,
                "--api-key",
                "mock-key",
                "--no-stream",
                "--max-steps",
                "8",
                "--yes",
            ],
            cwd=repo,
            env=_env(tmp_path),
            timeout=120.0,
        )

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, (
        f"pytest no-tests run did not finish cleanly (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "completion_gate_error" not in combined
    assert "forced diff marker" in (repo / "README.md").read_text(encoding="utf-8")
    _assert_no_raw_error(result)


@pytest.mark.smoke
def test_smoke_run_simple_agent_writes_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    with MockLLMServer() as server:
        _write_config(tmp_path / "config", base_url=server.base_url, repo=repo)
        result = _run_cli(
            [
                "run",
                _FULL_GATE_BUILD,
                "--path",
                os.fspath(repo.resolve()),
                "--allow-broad-workspace",
                "--mode",
                "fullaccess",
                "--model",
                QA_MODEL,
                "--base-url",
                server.base_url,
                "--api-key",
                "mock-key",
                "--no-stream",
                "--max-steps",
                "8",
                "--yes",
            ],
            cwd=repo,
            env=_env(tmp_path),
        )

    assert result.returncode == 0, (
        f"simple-agent run did not finish cleanly (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # The build's deliverable (a real, verified, diff-reviewed change) is on disk.
    assert "forced diff marker" in (repo / "README.md").read_text(encoding="utf-8"), (
        "the build's deliverable was not applied to the working tree"
    )
    _assert_no_raw_error(result)


def _prepare_forge_task(repo: Path, *, task_id: str, description: str) -> None:
    paths = create_plan_run(repo)
    plan = load_plan(paths)
    task = add_task(
        plan,
        title=f"Smoke {task_id}",
        description=description,
        estimated_files=["README.md"],
    )
    task["id"] = task_id
    task["write_scope"] = ["src/**", "tests/**", "docs/**", "README.md", "pyproject.toml"]
    save_plan(paths, plan)


def _forge_exec_args(task_id: str, *, repo: Path, base_url: str) -> list[str]:
    return [
        "forge",
        "exec",
        task_id,
        "--path",
        os.fspath(repo.resolve()),
        "--mode",
        "fullaccess",
        "--model",
        QA_MODEL,
        "--base-url",
        base_url,
        "--api-key",
        "mock-key",
        "--no-stream",
        "--max-steps",
        "8",
        "--yes",
    ]


@pytest.mark.smoke
def test_smoke_forge_exec_builds_task(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    with MockLLMServer() as server:
        _write_config(tmp_path / "config", base_url=server.base_url, repo=repo)
        _prepare_forge_task(repo, task_id="T01", description=_FULL_GATE_BUILD)
        result = _run_cli(
            _forge_exec_args("T01", repo=repo, base_url=server.base_url),
            cwd=repo,
            env=_env(tmp_path),
            timeout=120.0,
        )

    assert result.returncode == 0, (
        f"forge exec did not finish cleanly (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    _assert_no_raw_error(result)


@pytest.mark.smoke
def test_smoke_forge_swarm_runs_task(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_repo(repo)
    with MockLLMServer() as server:
        _write_config(tmp_path / "config", base_url=server.base_url, repo=repo)
        _prepare_forge_task(repo, task_id="T01", description=_FULL_GATE_BUILD)
        result = _run_cli(
            [
                "forge",
                "swarm",
                "--path",
                os.fspath(repo.resolve()),
                "--only",
                "T01",
                "--parallel",
                "1",
                "--scope",
                "off",
                "--verify",
                "warn",
                "--mode",
                "auto",
                "--model",
                QA_MODEL,
                "--base-url",
                server.base_url,
                "--api-key",
                "mock-key",
                "--no-stream",
                "--max-steps",
                "8",
                "--yes",
            ],
            cwd=repo,
            env=_env(tmp_path),
            timeout=150.0,
        )

    assert result.returncode == 0, (
        f"forge swarm did not finish cleanly (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    _assert_no_raw_error(result)
