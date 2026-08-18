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

## Why Sylliptor

- **Forge** — Plan, dispatch parallel workers, verify each task, ship.
- **Personas** — Switch between coding, architecture, questions, and debugging without leaving the session.
- **Cross-run memory** — Failures become structured issues the next run avoids.
- **Flexible model access** — Connect your own provider, a supported subscription, or a Sylliptor account.
- **Sandboxed by default** — Docker or Bubblewrap. An always-on denylist refuses `rm -rf /`, `curl | sh`, and `sudo` — even in `fullaccess`.

## How Forge Works

Type `/forge`, describe what you want, and Forge:

1. Asks 1–3 clarifying questions if the ask is vague.
2. Writes `plan.json` with explicit tasks and runnable file scope.
3. On `/execute plan`, dispatches a swarm of workers that run tasks in parallel.
4. Verifies each task before marking it done. Failures become `issue` entries the next attempt sees.
5. Integrates verified task changes into the branch where Forge started.

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
cd /path/to/project
sylliptor
```

The first launch guides you through connecting a model and choosing a workspace. After that, Sylliptor opens directly into an interactive session.

Use `/login` to connect an account or supported subscription, and `/config` to change the provider, model, or other session defaults.

## Inside Sylliptor

| Command | What it does |
| --- | --- |
| `/login` | Connect a Sylliptor account or supported subscription. |
| `/config` | Change the model and session settings. |
| `/persona` | Switch between Code, Architect, Ask, and Debug. |
| `/mode` | Change the execution mode. |
| `/forge` | Plan and run larger work. |
| `/subagent` | Delegate a focused task. |
| `/status` | Show the current model, mode, and workspace. |
| `/help` | See every available command. |

You can ask naturally for the work you want. Sylliptor inspects the workspace, proposes a plan when needed, makes changes within the selected mode, and verifies the result.

## Personas

Personas change how Sylliptor approaches a task without changing the safety rules underneath it. Open `/persona` to choose:

- **Code** for implementation work.
- **Architect** for plans and design decisions.
- **Ask** for read-only questions and explanations.
- **Debug** for reproduce-first investigation and fixes.

You can switch at any time during a normal session.

## Execution Modes

Open `/mode` to choose how much Sylliptor can do without asking first.

| Mode | Behavior |
| --- | --- |
| `readonly` | Inspection-only. No file writes, shell, MCP, or subagent delegation. |
| `review` | Default safe mode. Previews and asks before file writes and shell commands. |
| `auto` | Applies changes with fewer prompts. Hard denylist still applies. |
| `fullaccess` | No mode-level approval prompts. Denylist + audit log still active. |

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
- [**Subagents**](docs/subagents.md) — focused delegation. Drop YAML+markdown into `.sylliptor_agents/*.md` for custom agents. Built-ins: `explorer`, `implementer`, `frontend-engineer`, `debugger`, `code-reviewer`, `test-strategist`, plus the opt-in `visual-designer` image generator.
- [**Hooks**](docs/hooks.md) — lifecycle policy across 11 events (`PreToolUse`, `PostToolUse`, `SessionStart`, ...). Three trust layers.
- [**Plugins**](docs/plugins.md) — declarative bundles of skills + custom tools + MCP servers + hooks. Pinned install (registry id or `git+https://...@<sha40>`).

Run as an HTTP service with [Server mode](docs/server.md) — worker jobs, uploads, queues, and authentication.

**Repo conventions.** Sylliptor reads `AGENTS.md`, `CLAUDE.md`, and `CONVENTIONS.md` from your repo root as read-only project context.

## Configuration & Credentials

Open `/login` to connect an account or supported subscription. Use `/config` for API-key providers, model selection, and session defaults. Credentials are stored outside the project.

See [Credentials](docs/credentials.md) for key resolution and storage details.
See [Providers and models](docs/providers.md) for additional connection options.

## Workspace Behavior

Sylliptor binds a workspace when the session starts. In a git repository, it uses the repository root while preserving the directory where you launched it as the focus directory.

Broad directories such as your home directory require explicit confirmation. The filesystem root is blocked as a workspace.

## Project Links

- [Website](https://sylliptor.alysisai.com/)
- [Docs index](docs/README.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [License](LICENSE)

Use Python 3.11 or newer for local development. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and PR expectations. Report vulnerabilities through [SECURITY.md](SECURITY.md), not public GitHub issues.

Sylliptor is distributed under the [Apache License 2.0](LICENSE).
