from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sylliptor_agent_cli import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _entrypoint_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.pathsep.join([str(SRC_DIR), existing_pythonpath])
        if existing_pythonpath
        else str(SRC_DIR)
    )
    return env


@pytest.mark.parametrize(
    "module_name",
    [
        "sylliptor_agent_cli",
        "sylliptor_agent_cli.cli",
    ],
)
def test_module_entrypoint_help_outputs_cli_usage(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_entrypoint_env(),
    )

    assert proc.returncode == 0
    help_text = proc.stdout + proc.stderr
    assert "Usage:" in help_text
    assert "run" in help_text
    assert "chat" in help_text


@pytest.mark.parametrize(
    "module_name",
    [
        "sylliptor_agent_cli",
        "sylliptor_agent_cli.cli",
    ],
)
def test_module_entrypoint_version_outputs_package_version(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_entrypoint_env(),
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == __version__
    assert proc.stderr.strip() == ""
