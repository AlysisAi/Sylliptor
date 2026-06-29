from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli.agent_loop import AgentRuntimeError, create_session
from sylliptor_agent_cli.config import AppConfig


def _cfg() -> AppConfig:
    cfg = AppConfig(model="test-model")
    cfg.extra_fields = {"shell_sandbox": {"mode": "off"}}
    return cfg


def _shell_join(args: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(args)
    return shlex.join(args)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_http_ready(port: int) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
        assert response.status == 200


def _create_service_session(root: Path, sessions_dir: Path, *, mode: str = "auto") -> Any:
    return create_session(
        cfg=_cfg(),
        root=root,
        mode=mode,
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        non_interactive=True,
        session_log_dir_override=sessions_dir,
    )


def _wait_loaded_popen(session: Any, service_id: str) -> None:
    manager = session.durable_service_manager
    if manager is None:
        return
    popen = manager._popens.get(service_id)
    if popen is not None:
        popen.wait(timeout=3)


def test_shell_service_survives_close_and_can_stop_from_fresh_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    sessions_dir = tmp_path / "sessions"
    root.mkdir()
    (root / "index.html").write_text("ready\n", encoding="utf-8")
    port = _free_tcp_port()
    session = _create_service_session(root, sessions_dir)
    second_session = None
    service_id = ""
    session_closed = False

    try:
        started = session.tools["shell_service_start"].run(
            {
                "cmd": _shell_join(
                    [
                        sys.executable,
                        "-m",
                        "http.server",
                        str(port),
                        "--bind",
                        "127.0.0.1",
                    ]
                ),
                "readiness": {
                    "type": "tcp",
                    "host": "127.0.0.1",
                    "port": port,
                    "timeout_s": 5,
                },
            }
        )
        service_id = str(started["service_id"])

        assert started["ownership"] == "DURABLE_SERVICE"
        assert started["status"] == "running"
        assert started["readiness"]["status"] == "ready"
        assert Path(str(started["log_paths"]["stdout"])).exists()
        assert Path(str(started["log_paths"]["stderr"])).exists()

        session.close()
        session_closed = True
        _assert_http_ready(port)
        assert any(
            event["type"] == "durable_services_left_active"
            and event["payload"]["services"][0]["service_id"] == service_id
            for event in session.store.events_snapshot()
        )

        second_session = _create_service_session(root, sessions_dir)
        status = second_session.tools["shell_service_status"].run({"service_id": service_id})
        assert status["status"] == "running"
        assert status["readiness"]["status"] == "ready"

        stopped = second_session.tools["shell_service_stop"].run({"service_id": service_id})
        assert stopped["stopped"] is True
        _wait_loaded_popen(session, service_id)
    finally:
        if second_session is not None:
            if service_id:
                second_session.durable_service_manager.stop(service_id)
            second_session.close()
        if service_id and session.durable_service_manager is not None:
            session.durable_service_manager.stop(service_id)
        if not session_closed:
            session.close()


def test_shell_service_tools_hidden_in_readonly_mode(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    sessions_dir = tmp_path / "sessions"
    root.mkdir()
    session = _create_service_session(root, sessions_dir, mode="readonly")
    try:
        assert "shell_service_start" not in session.tools
        assert "shell_service_status" not in session.tools
        assert "shell_service_stop" not in session.tools
    finally:
        session.close()


def test_shell_service_start_uses_shell_policy(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    sessions_dir = tmp_path / "sessions"
    root.mkdir()
    session = _create_service_session(root, sessions_dir)
    try:
        with pytest.raises(AgentRuntimeError, match="Blocked"):
            session.tools["shell_service_start"].run({"cmd": "mkfs.ext4 /dev/sda"})
    finally:
        session.close()


def test_shell_service_command_readiness_uses_shell_policy(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    sessions_dir = tmp_path / "sessions"
    root.mkdir()
    session = _create_service_session(root, sessions_dir)
    try:
        with pytest.raises(AgentRuntimeError, match="Blocked"):
            session.tools["shell_service_start"].run(
                {
                    "cmd": _shell_join([sys.executable, "-c", "import time; time.sleep(30)"]),
                    "readiness": {"type": "command", "command": "mkfs.ext4 /dev/sda"},
                }
            )
    finally:
        session.close()
