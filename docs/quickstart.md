# Quickstart

This guide gets Sylliptor running in a local workspace.

## Install

Sylliptor requires Python 3.11 or newer. The recommended install path is `pipx`:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install sylliptor-agent-cli
```

If your default Python is older than 3.11, point `pipx` at a newer interpreter:

```bash
pipx install --python python3.12 sylliptor-agent-cli
```

Virtual-environment installs also work:

```bash
python -m pip install sylliptor-agent-cli
```

## First Run

Start Sylliptor from the project you want to inspect or edit:

```bash
cd /path/to/project
sylliptor
```

On a fresh install, the setup wizard guides you through the connection, model,
and workspace. Later, use `/login` to connect a different account or supported
subscription, and `/config` to change the provider or model.

See [Providers and models](providers.md) for the available connection types.

## Start Working

Describe what you want in normal language. For example:

> Explain this repository and show me the main entry points.

> Reproduce the failing test, fix it, and verify the result.

Sylliptor stays in the same session, so you can inspect its work, refine the
request, or ask a follow-up without starting over.

Useful chat commands:

- `/help`: show commands
- `/status`: show mode, workspace, and active model
- `/pwd`: show workspace root, focus directory, and active workdir
- `/mode`: inspect or change execution mode
- `/config`: change the connection, model, and session settings
- `/login`: connect a Sylliptor account or supported subscription
- `/persona`: switch between Code, Architect, Ask, and Debug
- `/subagent`: delegate a focused task
- `/forge`: start the plan-driven workflow for larger tasks

## Choose A Persona

Open `/persona` whenever you want to change how Sylliptor approaches the work:

- **Code** implements changes.
- **Architect** focuses on plans and design.
- **Ask** answers with read-only inspection.
- **Debug** reproduces a problem before fixing it.

Personas never grant more access than the current execution mode.

## Workspace Binding

Sylliptor binds the current directory as its workspace when the session starts.
Inside a Git repository, it uses the repository root and keeps the starting
subdirectory as the focus directory.

Broad paths such as your home directory require explicit confirmation. The
filesystem root is blocked as a workspace.

In chat, relative file/search/shell paths default to the active workdir. You can move within the
bound workspace with natural-language requests, `/cd`, or tool calls. Sylliptor does not rebind to a
different workspace mid-session.

## Sandbox Setup

Sylliptor can run shell and verification commands through Docker or Bubblewrap. For the simplest
first run on macOS or Windows, install Docker Desktop first, then run:

```bash
sylliptor sandbox setup
sylliptor sandbox doctor --smoke
sylliptor sandbox pull
```

See [Shell sandbox](shell_sandbox.md) for backend selection, production image pinning, and troubleshooting.

## Images And Tools

For a multimodal model, attach an image with `/image <path>` or paste it into
the TUI, then describe what you need.

Web search works through supported provider adapters, configured external
backends, or the keyless DDGS fallback. The active model decides when to use it.

## Updates

Sylliptor checks for newer releases in the background and may prompt before an
interactive launch. It never installs updates silently. To check PyPI immediately:

```bash
sylliptor update check
```

To apply an available update, run:

```bash
sylliptor update
```

The command detects common `pipx`, `uv`, virtualenv, and pip installs, shows the exact upgrade
command, and asks before running it. Source or editable installs are left manual.

## Next Steps

- [Credentials](credentials.md): API key precedence and persisted credentials.
- [Providers and models](providers.md): model access and subscription login.
- [Execution modes](../README.md#execution-modes): readonly, review, auto, and fullaccess.
- [Forge](forge.md): plan, execute, verify, and review larger tasks.
- [MCP](mcp.md): connect external MCP servers.
- [Skills](skills.md): install and use reusable instruction bundles.
