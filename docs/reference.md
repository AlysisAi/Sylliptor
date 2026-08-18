# Reference

This is a compact map of Sylliptor's interactive commands, execution modes,
configuration, and extension points.

## Start A Session

Launch `sylliptor` from the project you want to work on. The first launch opens
setup; later launches open the interactive session directly.

Sylliptor binds the workspace before exposing local tools. In a Git repository,
the repository root becomes the workspace and the launch directory remains the
focus directory. Use `/pwd` to see both.

## Interactive Commands

- `/help`: show every command available in the current view
- `/status`: show the model, mode, workspace, and runtime state
- `/pwd`: show the workspace and active work directory
- `/login`: connect a Sylliptor account or supported subscription
- `/logout`: disconnect the Sylliptor account
- `/config`: change the connection, model, and session settings
- `/persona`: switch between Code, Architect, Ask, and Debug
- `/mode`: change the execution mode
- `/usage`: show token and cost usage
- `/stream on|off`: toggle streaming
- `/trace off|compact|full`: change the amount of progress detail shown
- `/image <path>`: attach an image to the next turn
- `/subagent`: choose or control a focused helper agent
- `/skill`: list discovered skills
- `/plan <task>`: draft a plan for review
- `/forge [resume]`: enter or resume Forge
- `/report [text]`: create a local feedback bundle
- `/exit`: leave the session

Forge has its own on-screen actions for editing a goal and plan, running tasks,
reviewing results, and returning to the normal session.

## Personas

- `code`: implementation work with the tools allowed by the current mode
- `architect`: plans and design, with writes limited to Markdown
- `ask`: read-only questions and explanations
- `debug`: reproduce-first investigation and bug fixing

A persona can narrow access, but it cannot raise the session above its current
execution mode. Project-specific personas can be added under
`.sylliptor_personas/*.md`.

## Execution Modes

- `readonly`: inspect only; write, shell, and verification tools are unavailable
- `review`: ask before writes and shell commands
- `auto`: apply routine changes with fewer prompts while dangerous operations
  remain blocked
- `fullaccess`: remove mode-level prompts in a trusted workspace; the denylist
  and audit log still apply

Use `/mode` for the current session and `/config` for the default. The
`autonomous` step-budget policy has no fixed step limit, but execution modes,
approvals, deadlines, sandbox policy, and provider limits still apply.

Session policy records a `runtime_kind`, such as `interactive_chat`, `one_shot`,
or `forge_exec`. Extension systems use it when deciding which tools and catalogs
can be exposed.

## Connections And Configuration

Use `/login` for account and supported subscription connections. Use `/config`
for API-key providers, model selection, execution defaults, web search, tools,
and subagents.

Common configuration keys include:

- `base_url` and `model`
- `default_mode` and `default_persona`
- `max_steps`, `task_max_steps`, and `step_budget_policy`
- `stream`
- `subagents_enabled` and `subagent_timeout_s`
- `custom_tools_enabled`
- `web_search_policy`, `web_search_adapter`, and related web-search settings
- `verify_commands`
- `session_log_dir`
- `update_check_enabled` and `update_prompt_enabled`
- `prompt_cache_mode`

Common environment overrides include `SYLLIPTOR_API_KEY`,
`SYLLIPTOR_CONFIG_DIR`, `SYLLIPTOR_BASE_URL`, `SYLLIPTOR_MODEL`,
`SYLLIPTOR_LLM_TIMEOUT_S`, and the provider-specific web-search variables.

Profiles group a protocol, endpoint, credential source, and default model.
Native OpenAI, Anthropic, and Gemini protocols are supported; other providers
and gateways can use an OpenAI-compatible profile. Presets are starting points,
not hard constraints. See [Providers and models](providers.md).

## Automation Commands

The interactive workflow is the normal entry point. These commands remain
available for scripts and CI:

| Command | Use |
| --- | --- |
| `sylliptor run` | Run one bounded instruction. |
| `sylliptor forge plan` | Create or update a Forge plan. |
| `sylliptor forge exec` | Run one planned task. |
| `sylliptor forge swarm` | Run ready tasks in parallel. |
| `sylliptor tools` | Show the current tool catalog and readiness. |
| `sylliptor sessions` | Inspect retained session records. |
| `sylliptor update` | Check for or apply an update with confirmation. |

Use each command's `--help` output for flags and machine-oriented options.

## Built-In Tools

The available tool surface depends on the execution mode, `runtime_kind`,
workspace, sandbox readiness, and configuration. It can include:

- filesystem reads and edits
- repository search and symbol lookup
- Git history inspection
- shell commands and background terminals
- verification
- web fetch and optional web search
- session history and local artifacts
- constrained workspace previews

`web_fetch` retrieves a known HTTP(S) page. `web_search` discovers pages through
a supported provider adapter, configured backend, or the public fallback. Web
search can be disabled in `/config`.

## Verification

Sylliptor can infer verification commands from the workspace or use commands
saved in configuration. When verification is enabled, the agent can run the
selected checks and keep the complete output in local artifacts.

Forge can make verification authoritative for task gates. Normal sessions use
it as evidence that the requested work is complete.

## Extensions

- [MCP](mcp.md): connect external Model Context Protocol servers
- [Skills](skills.md): use reusable instruction bundles rooted at `SKILL.md`
- [Plugins](plugins.md): bundle skills, tools, MCP servers, and hooks
- [Custom tools](custom_tools.md): add trusted Python tools with manifests
- [Lifecycle hooks](hooks.md): run deterministic actions around sessions and
  tool calls
- [Subagents](subagents.md): delegate focused exploration, implementation,
  debugging, review, and test planning

Each extension type has its own trust boundary. Project-local executable
extensions generally require explicit trust before they can run.

## Forge

Forge creates a structured plan for larger work, executes scoped tasks, records
verification and review evidence, and keeps run artifacts inside the workspace.
Use it when a change benefits from explicit task boundaries or parallel work.

See [Forge](forge.md).

## Sessions, Feedback, And Updates

Session logs are stored locally as JSONL. Session inspection defaults to the
current local owner. Feedback reports create local bundles for review; they do
not upload an archive or submit a GitHub issue automatically.

Update checks run in the background when enabled. Sylliptor never installs an
update silently and shows the exact upgrade command before asking for approval.

## Troubleshooting

- For a network or model error, check the active connection in `/config`.
- If provider setup fails, use the redacted provider diagnostics from the CLI.
- If shell commands cannot run, check `/mode` and the sandbox setup.
- If web search is unavailable, check the tool status and web-search settings.
- If the workspace is unexpected, use `/pwd` and restart from the intended
  project directory.
- If image paste does not work, install a supported clipboard backend for the
  platform.

## Detailed Guides

- [Architecture](architecture.md): high-level system structure
- [Quickstart](quickstart.md): installation and first use
- [Providers and models](providers.md): model access and login
- [Credentials](credentials.md): API key precedence and storage
- [Security model](security_model.md): trust boundaries and sandboxing
- [Shell sandbox](shell_sandbox.md): Docker and Bubblewrap setup
- [Server mode](server.md): HTTP API operation
- [Forge](forge.md): plan-driven workflows
- [MCP](mcp.md): external server integration
- [Skills](skills.md): reusable instruction bundles
- [Subagents](subagents.md): focused helper agents
- [Plugins](plugins.md): trusted extension bundles
- [Custom tools](custom_tools.md): trusted Python tool authoring
- [Lifecycle hooks](hooks.md): command-based policy and automation
