# Reference

This page is a compact reference for the main Sylliptor CLI surface,
configuration model, runtime modes, and extension points. For deeper subsystem
details, follow the linked guides.

## What Sylliptor Does

Sylliptor runs local coding sessions from a terminal. A session binds to a
workspace, sends user turns to the configured model provider, exposes a
controlled set of tools, and stores local logs and artifacts for review.

Core capabilities include:

- interactive chat and one-shot commands
- filesystem, search, git, shell, web, and verification tools
- workspace-aware execution modes
- optional MCP, skills, plugins, hooks, custom tools, and subagents
- Forge planning and execution workflows
- local session logs and feedback bundles

## Commands

Common entrypoints:

```bash
sylliptor
sylliptor chat
sylliptor run "Explain this repository."
sylliptor setup
sylliptor tools
sylliptor auth list
sylliptor update check
```

Workspace selection:

```bash
sylliptor chat --path /path/to/project
sylliptor run --path /path/to/project "Summarize the codebase."
sylliptor run --path ./new-app --create-path "Scaffold a minimal project."
```

Provider and credentials:

```bash
sylliptor config set base_url "https://api.openai.com/v1"
sylliptor config set model "gpt-4.1-mini"
sylliptor config set-api-key
sylliptor run --api-key-stdin "Hello"
sylliptor run --api-key-env OTHER_API_KEY --base-url "https://example.com/v1" --model "your-model" "Hello"
```

Forge:

```bash
sylliptor forge plan --path .
sylliptor forge show --path .
sylliptor forge status --path .
sylliptor forge exec T01 --path . --mode review
sylliptor forge swarm --path . --parallel 3 --mode auto --verify warn
```

## Execution Modes

Execution mode controls the default approval posture.

- `readonly`: inspect-only mode. Write tools, shell commands, and verification
  are not available.
- `review`: preview or ask before writes and shell commands.
- `auto`: allow routine edits with fewer prompts while still blocking dangerous
  operations.
- `fullaccess`: remove mode-level write and shell prompts for trusted
  workspaces.

Set the default mode:

```bash
sylliptor config set default_mode review
```

Override per command:

```bash
sylliptor run --mode readonly "Explain this repository."
sylliptor chat --mode auto
```

The default `autonomous` step-budget policy has no fixed step limit. Set
`--max-steps` for one command, or restore configured limits with:

```bash
sylliptor config set step_budget_policy limited
```

Execution modes, approvals, deadlines, sandbox policy, and provider limits
still apply.

Session policy also records a `runtime_kind` such as `interactive_chat`,
`one_shot`, or `forge_exec`; extension systems use that runtime kind when
deciding which tools or catalogs may be exposed.

## Workspace Binding

`sylliptor chat` and `sylliptor run` bind a workspace before the session starts.
The requested path is the current directory or `--path`.

- Inside a Git repository, Sylliptor binds to the repository root and keeps the
  starting directory as the focus directory.
- Plain directories bind to the requested directory.
- Missing paths require `--create-path`.
- Broad paths such as a home directory require an explicit override.
- The filesystem root is blocked as a workspace root.

In chat, relative file, search, and shell paths default to the active workdir
inside the bound workspace. Use `/pwd` to inspect the current workspace root,
focus directory, and active workdir.

## Chat Slash Commands

Common interactive commands:

- `/help`: show chat commands
- `/status`: show mode, model, workspace, and runtime state
- `/pwd`: show workspace and active workdir
- `/mode`: inspect or change execution mode
- `/config`: open the configuration menu
- `/usage`: show token and cost usage
- `/stream on|off`: toggle streaming
- `/trace off|compact|full`: control reasoning/tool progress detail
- `/image <path>`: attach an image to the next turn
- `/subagent on|off|status`: control subagent availability
- `/skill`: list discovered skills
- `/plan <task>`: draft a plan for review and approval
- `/forge [resume]`: enter or resume Forge for the workspace
- `/report [text]`: create a local feedback bundle
- `/exit`: quit chat

Forge mode has its own command surface for goal, task, plan, review, and
execution actions. See [Forge](forge.md).

## Configuration

Configuration is stored in the platform-specific Sylliptor config directory.
Use `sylliptor config menu` or `/config` in chat for interactive edits.

Common keys:

- `base_url`
- `model`
- `default_mode`
- `max_steps`
- `task_max_steps`
- `step_budget_policy`
- `stream`
- `routing_mode`
- `role_models.router`
- `subagents_enabled`
- `custom_tools_enabled`
- `web_search_mode`
- `web_search_policy`
- `web_search_adapter`
- `web_search_base_url`
- `web_search_model`
- `web_search_timeout_s`
- `session_log_dir`
- `verify_commands`
- `update_check_enabled`
- `update_check_interval_hours`
- `update_check_timeout_s`
- `update_prompt_enabled`
- `prompt_cache_mode`

Useful environment overrides:

- `SYLLIPTOR_API_KEY`
- `SYLLIPTOR_CONFIG_DIR`: overrides the user config directory used for `config.json`, `credentials.json`, and the MCP OAuth token store.
- `SYLLIPTOR_BASE_URL`
- `SYLLIPTOR_MODEL`
- `SYLLIPTOR_MODEL_ROUTER`
- `SYLLIPTOR_LLM_TIMEOUT_S`
- `SYLLIPTOR_ROUTING_MODE`
- `SYLLIPTOR_WEB_SEARCH_API_KEY`
- `SYLLIPTOR_WEB_SEARCH_BASE_URL`
- `SYLLIPTOR_WEB_SEARCH_MODEL`
- `SYLLIPTOR_WEB_SEARCH_TIMEOUT_S`
- `SYLLIPTOR_WEB_SEARCH_KEYLESS`
- `SYLLIPTOR_UPDATE_PROMPT_ENABLED`
- `TAVILY_API_KEY`

`role_models.router` overrides the model used for lightweight routing. Leave it
unset to inherit `model`, or set it to a smaller/cheaper model while keeping the
main coding model stronger.

## Profiles

Profiles group provider settings such as protocol, base URL, API key source,
default model, and provider notes.

Useful commands:

```bash
sylliptor profile presets
sylliptor profile preset <provider-preset>
sylliptor profile set-key <profile> --stdin
sylliptor profile use <profile>
sylliptor profile list
```

OpenAI, Anthropic, and Gemini profiles can use their native API protocols.
Other provider and gateway profiles use the OpenAI-compatible protocol.

Subscription-backed connections are managed separately from static API keys:

```bash
sylliptor auth list
sylliptor auth login openai-codex
sylliptor auth status openai-codex
sylliptor auth logout openai-codex
```

Choose the subscription model in `/config`. See
[Providers and models](providers.md) for details.

Presets are convenience templates, not hard constraints. Custom profiles can
point at model provider endpoints.

## Built-In Tools

The built-in tool surface depends on mode, runtime kind, workspace binding,
sandbox readiness, and configuration.

Common tool families:

- filesystem reads, writes, edits, moves, copies, and deletes
- repository text search and symbol lookup
- compact repository mapping and focused test discovery
- git history inspection
- shell command execution, background terminals, and durable service helpers
- verification command execution
- web fetch and optional web search
- session history and local artifacts
- constrained static workspace previews

Use:

```bash
sylliptor tools
```

for the current built-in tool catalog and configuration-dependent availability.

## Web Access

`web_fetch` retrieves one specific HTTP(S) URL. It is for targeted page or
document retrieval. Use it for URLs explicitly provided by the user, returned
by search, or discovered from trusted fetched or local content.

`web_search` is a discovery tool. In auto mode it can use a supported provider
adapter, configured external backend, or keyless DDGS fallback. Set
`web_search_policy=off` to remove it from the model's tools.

Other chat provider profiles can still be valid model providers without being
native `web_search` backends.

## Verification

Verification commands can be inferred from the workspace, provided by config, or
passed explicitly:

```bash
sylliptor run --verify-cmd "pytest -q" "Fix the failing test."
sylliptor chat --verify-cmd "npm test"
```

When enabled, the `verify_run` tool lets the agent run the selected verification
commands and return a compact result while retaining full output in local
artifacts.

Forge can make verification authoritative for task gates. See [Forge](forge.md).

## Extensions

Sylliptor supports several extension points:

- [MCP](mcp.md): connect external Model Context Protocol servers.
- [Skills](skills.md): use reusable instruction bundles rooted at `SKILL.md`.
- [Skills lifecycle](skills_lifecycle.md): scaffold, validate, install, enable,
  disable, and remove skills.
- [Plugins](plugins.md): package skills, tools, MCP servers, and hooks.
- [Custom tools](custom_tools.md): add trusted Python tools with manifests.
- [Lifecycle hooks](hooks.md): run deterministic command hooks around sessions
  and tool calls.
- [Subagents](subagents.md): delegate focused exploration, review, and testing
  strategy work from normal chat and one-shot flows.

Each extension type has its own trust boundary. Project-local executable
extension points generally require explicit trust before they affect execution.

## One-Shot Runs

`sylliptor run` is optimized for a single bounded instruction. It includes
guardrails for execution-style repository tasks, but it is not meant to replace
interactive refinement.

Use `sylliptor run` for focused tasks such as:

- explain this repository
- summarize a file or module
- make a small targeted change
- run a specific verification command

Prefer `sylliptor chat` or Forge for ambiguous, exploratory, or highly
iterative work.

## Forge

Forge is the plan-driven workflow for larger tasks. It creates a structured
plan, executes scoped tasks, records verification/review evidence, and keeps run
artifacts under the workspace runtime directory.

Use Forge when a change benefits from:

- an explicit plan
- scoped task boundaries
- review or verification gates
- batch execution
- local PR-style task flow

See [Forge](forge.md).

## Sessions And Logs

Sylliptor stores session logs locally as JSONL. Session commands include:

```bash
sylliptor sessions list
sylliptor sessions show <session_id>
sylliptor sessions score <session_id>
sylliptor sessions score --latest 5
```

Session pickers and implicit latest-session operations default to the current
local owner. Show all retained sessions with:

```bash
sylliptor sessions list --all
```

Feedback bundles can be created from retained session artifacts:

```bash
sylliptor report create --path .
sylliptor report create "expected X, got Y" --path . --latest
```

Sylliptor prepares local artifacts for review. It does not submit GitHub issues
or upload archives automatically.

## Updates

Sylliptor checks for newer releases in a non-blocking, cache-backed way when
enabled. It never installs updates silently.

Interactive launches may prompt when a cached check finds a newer release.
Set `update_prompt_enabled=false` to disable only that prompt.

```bash
sylliptor update check
sylliptor update
sylliptor update --dry-run
```

The update command detects common install styles such as `pipx`, `uv`, virtual
environments, and pip installs, then shows the exact upgrade command before
running it.

## Troubleshooting

- If `sylliptor chat` shows a network or model error, verify the API key, base
  URL, model name, and network access.
- If provider setup fails, run `sylliptor doctor providers` for redacted
  provider diagnostics.
- If shell commands cannot run, check the selected execution mode and sandbox
  setup.
- If web search is unavailable, run `sylliptor tools` and check `web_search`
  readiness.
- If the workspace is not what you expected, run `/pwd` in chat or start again
  with `--path`.
- If clipboard image paste does not work, install a supported clipboard backend
  for your platform.

## Detailed Guides

- [Architecture](architecture.md): high-level system structure.
- [Quickstart](quickstart.md): first setup and first run.
- [Providers and models](providers.md): model access and subscription login.
- [Credentials](credentials.md): API key precedence and storage.
- [Security model](security_model.md): trust boundaries and sandboxing.
- [Shell sandbox](shell_sandbox.md): Docker and Bubblewrap setup.
- [Server mode](server.md): HTTP API operation.
- [Forge](forge.md): plan-driven workflows.
- [MCP](mcp.md): external server integration.
- [Skills](skills.md): reusable instruction bundles.
- [Skills lifecycle](skills_lifecycle.md): skill authoring, validation, installation, and removal.
- [Subagents](subagents.md): focused helper agents for exploration, review, and testing strategy.
- [Plugins](plugins.md): trusted extension bundles.
- [Custom tools](custom_tools.md): trusted Python tool authoring.
- [Lifecycle hooks](hooks.md): command-based policy and automation.
