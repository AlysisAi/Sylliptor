<p align="center">
  <img src="https://raw.githubusercontent.com/AlysisAi/Sylliptor/main/docs/assets/sylliptor-demo.gif" alt="Sylliptor owl logo" width="192" height="192">
</p>

<h1 align="center">SYLLIPTOR</h1>

<p align="center">
  <strong>Local CLI coding agent that turns plans into reviewed, PR-ready code.</strong>
</p>

<p align="center">
  Bring your own model. Sandboxed by default.
</p>

<p align="center">
  <a href="https://sylliptor.alysisai.com/">Website</a> ·
  <a href="https://github.com/AlysisAi/Sylliptor/tree/main/docs">Docs</a> ·
  <a href="https://github.com/AlysisAi/Sylliptor/blob/main/CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <a href="https://github.com/sponsors/AlysisAi"><img src="https://img.shields.io/github/sponsors/AlysisAi?label=Sponsor&logo=GitHub" alt="GitHub Sponsors"></a>
</p>

<p align="center">
  <a href="https://github.com/AlysisAi/Sylliptor/actions/workflows/ci.yml"><img src="https://github.com/AlysisAi/Sylliptor/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/sylliptor-agent-cli/"><img src="https://img.shields.io/pypi/v/sylliptor-agent-cli.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/sylliptor-agent-cli/"><img src="https://img.shields.io/pypi/pyversions/sylliptor-agent-cli.svg" alt="Python versions"></a>
  <a href="https://github.com/AlysisAi/Sylliptor/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/sylliptor-agent-cli.svg" alt="License"></a>
</p>

---

## ⚡ Free Xiaomi MiMo trial

Run Sylliptor on **[Xiaomi MiMo](https://openrouter.ai/xiaomi)** — **free for 10 days**, no API key, no card. MiMo is the default model.

```bash
pipx install sylliptor-agent-cli
sylliptor login      # connect your Sylliptor account in the browser
sylliptor chat       # start building — MiMo is ready
```

Create your account at **[sylliptor.alysisai.com](https://sylliptor.alysisai.com)**, then `sylliptor login` links the CLI to your trial. Usage is metered server-side and the upstream model key never touches your machine.

---

## Why Sylliptor

- **Forge** — Plan, dispatch parallel workers, verify each task, ship.
- **Cross-run memory** — Failures become structured issues the next run avoids.
- **Bring your own model** — OpenAI, Anthropic, DeepSeek, Qwen, Gemini, Mistral, OpenRouter, xAI.
- **Sandboxed by default** — Docker or Bubblewrap. An always-on denylist refuses `rm -rf /`, `curl | sh`, and `sudo` — even in `fullaccess`.

## How Forge Works

Type `/forge` in chat (or run `sylliptor forge plan`), describe what you want, and Forge:

1. Asks 1–3 clarifying questions if the ask is vague.
2. Writes `plan.json` with explicit tasks and runnable file scope.
3. On `/execute plan`, dispatches a swarm of workers that run tasks in parallel.
4. Verifies each task before marking it done. Failures become `issue` entries the next attempt sees.
5. Merges to `main` when everything passes.

All plans, traces, and per-task artifacts persist under `.sylliptor/runs/<run_id>/`. Resume any time with `/forge resume`.

## Install

Sylliptor requires Python 3.11 or newer.

```bash
pipx install sylliptor-agent-cli
```

If your default `python3` is older than 3.11:

```bash
pipx install --python python3.12 sylliptor-agent-cli
```

`pip` also works inside a virtual environment:

```bash
python -m pip install sylliptor-agent-cli
```

## Quick Start

```bash
pipx install sylliptor-agent-cli
export SYLLIPTOR_API_KEY="YOUR_KEY"
sylliptor config set model "your-model"
sylliptor chat
```

On a fresh install, running `sylliptor` opens a guided setup wizard for provider, API key, default model, optional router model, and workspace. Re-run anytime:

```bash
sylliptor setup
```

Configure a provider endpoint and model:

```bash
sylliptor config set base_url "your-base-url"
sylliptor config set model "your-model"
```

Per-command key, endpoint, and model overrides:

```bash
sylliptor run --api-key-env OTHER_API_KEY --base-url "your-base-url" --model "your-model" "Summarize this project."
```

## Core Commands

| Command | Use |
| --- | --- |
| `sylliptor` | Start setup or interactive chat. |
| `sylliptor setup` | Configure API key, model, and workspace defaults. |
| `sylliptor run "..."` | Run a one-shot task in the current workspace. |
| `sylliptor chat` | Start an interactive coding session. |
| `sylliptor forge plan` | Create or update a Forge plan from the CLI. |
| `sylliptor forge exec` | Execute a Forge task non-interactively. |
| `sylliptor forge swarm` | Run Forge tasks across parallel workers. |
| `sylliptor tools` | Show built-in tools and readiness. |
| `sylliptor sandbox doctor --smoke` | Check sandbox readiness. |

Useful chat commands include `/help`, `/status`, `/mode`, `/config`, `/plan`, `/forge`, `/skill`, `/subagent`, `/terminals`, `/resume`.

## Execution Modes

Choose a mode per command with `--mode`, change it in chat with `/mode`, or set the default with `sylliptor config set default_mode <mode>`.

| Mode | Behavior |
| --- | --- |
| `readonly` | Inspection-only. No file writes, shell, MCP, or subagent delegation. |
| `review` | Default safe mode. Previews and asks before file writes and shell commands. |
| `auto` | Applies changes with fewer prompts. Hard denylist still applies. |
| `fullaccess` | No mode-level approval prompts. Denylist + audit log still active. |

```bash
sylliptor run --mode readonly "Find risky areas in this codebase."
sylliptor run --mode review "Implement the failing test fix."
sylliptor chat --mode auto
```

## Sandbox & Safety

Shell and verification execution run inside a hardened Docker or Bubblewrap sandbox by default.
Shell commands and verification commands default to strict sandboxing. To deliberately disable
verification sandboxing for a trusted local setup, set `verify_sandbox.mode="off"` or
`SYLLIPTOR_VERIFY_SANDBOX_MODE=off`.

```bash
docker pull ghcr.io/alysisai/sylliptor-sandbox:dev
docker pull ghcr.io/alysisai/sylliptor-sandbox:server
```

Prepare or diagnose:

```bash
sylliptor sandbox setup
sylliptor sandbox doctor --smoke
sylliptor sandbox pull
```

The denylist is always-on across every mode. It refuses `rm -rf /`, `curl ... | sh`, `sudo`, force-push to `main` / `master`, raw disk writes, fork-bombs, recursive `chmod 777 /`, and direct `> /dev/sd*` redirects. In `fullaccess`, every successful shell command additionally writes a JSONL audit event.

Outbound HTTP from web tools and MCP OAuth goes through `safe_http_request` with SSRF guards: rejects non-HTTP schemes, loopback / link-local / private / multicast targets across IPv4 and IPv6, validates redirects, and enforces a streamed byte cap.

See [Shell sandbox](docs/sandbox.md) for backend requirements, image cosign signatures, SLSA provenance, and production pinning. See [Security model](docs/security_model.md) for the full threat boundary.

## Extend Sylliptor

Six capability surfaces. Four of them — skills, custom tools, MCP servers, hooks — bundle into a single declarative `.toml` plugin manifest.

- [**MCP**](docs/mcp.md) — connect stdio or Streamable HTTP MCP servers, with OAuth, frozen catalogs, and narrowing-only project overrides.
- [**Custom tools**](docs/custom_tools.md) — drop Python scripts into `.sylliptor/tools/*.py`. AST-only discovery, trust-keyed by file hash.
- [**Skills**](docs/skills.md) — `SKILL.md` instruction bundles. Native + interop roots (`.sylliptor_skills/`, `.agents/skills/`, `.claude/skills/`, `.github/skills/`).
- [**Subagents**](docs/subagents.md) — focused delegation. Drop YAML+markdown into `.sylliptor_agents/*.md` for custom agents. Built-ins: `explorer`, `reviewer`, `test-strategist`.
- [**Hooks**](docs/hooks.md) — lifecycle policy across 11 events (`PreToolUse`, `PostToolUse`, `SessionStart`, ...). Three trust layers.
- [**Plugins**](docs/plugins.md) — declarative bundles of skills + custom tools + MCP servers + hooks. Pinned install (registry id or `git+https://...@<sha40>`).

Run as an HTTP service with [Server mode](docs/server.md) — worker jobs, uploads, queues, and authentication.

**Repo conventions.** Sylliptor reads `AGENTS.md`, `CLAUDE.md`, and `CONVENTIONS.md` from your repo root as read-only project context.

## Configuration & Credentials

API keys can come from per-command options, `SYLLIPTOR_API_KEY` or persisted credentials.

```bash
sylliptor config show
sylliptor config set-api-key
sylliptor config clear-api-key
```

Provider profiles switch between configured endpoints:

```bash
sylliptor profile presets
sylliptor profile use openai
sylliptor profile list
```

See [Credentials](docs/credentials.md) for key resolution and storage details.

## Workspace Behavior

`sylliptor run` and `sylliptor chat` bind a workspace before the session starts. The requested path is `--path` or the current directory. In a git repository, Sylliptor binds to the repository root while preserving the starting directory as the focus directory.

Missing paths require `--create-path`. Broad directories such as `~` require an explicit override. `/` is blocked as a workspace root.

## Project Links

- [Website](https://sylliptor.alysisai.com/)
- [Docs index](docs/README.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [License](LICENSE)

Use Python 3.11 or newer for local development. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and PR expectations. Report vulnerabilities through [SECURITY.md](SECURITY.md), not public GitHub issues.

Sylliptor is distributed under the [Apache License 2.0](LICENSE).
