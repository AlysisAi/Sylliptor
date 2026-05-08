from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest

from sylliptor_agent_cli.clipboard import ClipboardError, paste_clipboard_image


def test_paste_clipboard_image_via_powershell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = b"\x89PNG\r\n\x1a\nfake"

    def fake_which(cmd: str) -> str | None:
        if cmd == "powershell.exe":
            return "powershell.exe"
        return None

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[bytes]:
        assert args[0] == "powershell.exe"
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=base64.b64encode(png),
            stderr=b"",
        )

    monkeypatch.setattr("sylliptor_agent_cli.clipboard.shutil.which", fake_which)
    monkeypatch.setattr("sylliptor_agent_cli.clipboard.subprocess.run", fake_run)

    path = paste_clipboard_image(root=tmp_path)
    assert path.exists()
    assert path.read_bytes() == png


def test_paste_clipboard_image_errors_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sylliptor_agent_cli.clipboard.shutil.which", lambda _cmd: None)

    with pytest.raises(ClipboardError, match="Could not read an image from clipboard"):
        paste_clipboard_image(root=tmp_path)
