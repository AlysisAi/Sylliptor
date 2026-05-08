from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.extensions.models import RegistryEntry, RegistryFile


def test_ext_search_shows_matches(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    registry = RegistryFile(
        extensions=[
            RegistryEntry(
                id="acme.jira",
                name="Acme Jira",
                description="Jira integration",
                repo="https://github.com/acme/jira-ext",
                commit="abc123",
                version="1.2.3",
                tags=["jira", "issues"],
            )
        ]
    )
    monkeypatch.setattr(cli_mod, "load_registry", lambda: registry)

    env = {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path / "config"),
        "SYLLIPTOR_DATA_DIR": os.fspath(tmp_path / "data"),
    }
    result = runner.invoke(sylliptor_app, ["ext", "search", "jira"], env=env)
    assert result.exit_code == 0
    assert "acme.jira" in result.output
    assert "Acme Jira" in result.output


def test_ext_info_returns_exit_code_1_for_unknown_id(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr(cli_mod, "load_registry", lambda: RegistryFile(extensions=[]))

    env = {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path / "config"),
        "SYLLIPTOR_DATA_DIR": os.fspath(tmp_path / "data"),
    }
    result = runner.invoke(sylliptor_app, ["ext", "info", "missing.ext"], env=env)
    assert result.exit_code == 1
    assert "Extension not found" in result.output


def test_ext_list_shows_empty_message_when_no_extensions_installed(tmp_path: Path) -> None:
    runner = CliRunner()
    env = {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path / "config"),
        "SYLLIPTOR_DATA_DIR": os.fspath(tmp_path / "data"),
    }
    result = runner.invoke(sylliptor_app, ["ext", "list"], env=env)
    assert result.exit_code == 0
    assert "No extensions installed." in result.output
