# Quickstart

This guide gets Sylliptor installed, configured, and running against a local workspace.

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

On a fresh install, the setup wizard asks for an API key, default model, and workspace. Re-run it
anytime:

```bash
sylliptor setup
```

For manual configuration:

```bash
export SYLLIPTOR_API_KEY="YOUR_KEY"
sylliptor config set base_url "https://api.openai.com/v1"
sylliptor config set model "gpt-4.1-mini"
```

To avoid storing a key, pass it for one command:

```bash
sylliptor run --api-key-stdin "Explain this repository."
```

To switch providers with a different environment variable:

```bash
export OTHER_API_KEY="YOUR_KEY"
sylliptor run --api-key-env OTHER_API_KEY --base-url "https://example.com/v1" --model "your-model" "Hello"
```

## Run And Chat

Use `run` for one-shot work:

```bash
sylliptor run --mode readonly "Explain this repository and identify the main entrypoints."
sylliptor run --mode review "Fix the failing test and show me the diff."
```

`sylliptor run` is a bounded one-shot flow. It includes guardrails for execution-style prompts, but
it is still best for focused tasks that can be completed from one instruction. For exploratory or
multi-step work, prefer `sylliptor chat` or Forge.

Use `chat` for an interactive session:

```bash
sylliptor chat
```

Useful chat commands:

- `/help`: show commands
- `/status`: show mode, workspace, and active model
- `/pwd`: show workspace root, focus directory, and active workdir
- `/mode`: inspect or change execution mode
- `/config`: open the inline configuration menu
- `/forge`: start the plan-driven workflow for larger tasks

## Workspace Binding

`sylliptor run` and `sylliptor chat` bind a workspace before the session starts. The requested path is
the current directory or `--path`. Inside a Git repository, Sylliptor binds to the repository root
and keeps the starting subdirectory as the focus directory.

Missing paths require `--create-path`. Broad paths such as `~` require an explicit override, and `/`
is blocked as a workspace root.

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

For multimodal-compatible models or providers:

```bash
sylliptor run --image ./screenshot.png "Describe this screenshot."
```

Inspect the built-in tool surface:

```bash
sylliptor tools
```

Web search is optional and configuration-dependent. In this public build, `web_search` is available
through OpenAI Responses, supported DashScope/Qwen endpoints, or Tavily when `TAVILY_API_KEY` is
set. Other provider profiles can still be valid chat providers without being native `web_search`
backends.

Inspect custom tools discovered for the current workspace:

```bash
sylliptor tool list --path .
```

## Updates

Sylliptor checks for newer releases in the background at most once per configured interval, then
shows cached notices in home/status surfaces. It never installs updates silently. To check PyPI for
the latest package immediately:

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
- [Execution modes](../README.md#execution-modes): readonly, review, auto, and fullaccess.
- [Forge](forge.md): plan, execute, verify, and review larger tasks.
- [MCP](mcp.md): connect external MCP servers.
- [Skills](skills.md): install and use reusable instruction bundles.
