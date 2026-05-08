# Architecture

Sylliptor is a local CLI coding agent. It binds to a workspace, builds a
runtime session, exposes a controlled set of tools, sends turns to the selected
model provider, and stores local artifacts so work can be inspected after the
session ends.

This page gives a high-level view of the system. Detailed command behavior,
configuration keys, and subsystem contracts live in the feature-specific docs.

## Design Principles

- Local-first operation: source code, runtime state, and logs stay on the local
  machine unless the user configures a model provider, MCP server, web search
  backend, or other networked extension.
- Explicit workspace binding: file, shell, git, and runtime artifacts are scoped
  to a resolved workspace before a session starts.
- Host-owned policy: execution mode, sandbox settings, tool availability,
  approvals, and extension trust are enforced by Sylliptor, not by model prose.
- Progressive context: the model receives bounded workspace context and can ask
  for more through tools instead of receiving the entire repository upfront.
- Inspectable runs: session logs, verification output, Forge artifacts, and
  feedback bundles are written as local records for review and debugging.

## Runtime Flow

The High-level flow is:

1. The CLI resolves configuration, credentials, model/profile settings,
   execution mode, and the requested workspace path.
2. Workspace binding determines the workspace root and focus directory. Git
   repositories bind to their repository root; plain directories bind to the
   requested directory.
   The resulting `workspace_root` and `active_workdir` are carried through the
   session and tool layer.
3. Sylliptor creates a session with prompt context, workspace metadata, allowed
   tools, verification settings, and any enabled extension catalogs.
4. The agent loop sends the current turn to the model provider. If the model
   requests tool calls, Sylliptor validates and runs them through host-owned
   policy checks.
5. Tool results are appended back to the session, and the loop continues until
   the model returns a final answer or the configured step budget is reached.
6. Logs and artifacts remain available locally for status views, resumes,
   feedback exports, and Forge execution reports.

## Main Components

### CLI And Session Runtime

The Typer CLI exposes `sylliptor`, `sylliptor chat`, `sylliptor run`, Forge
commands, setup commands, and supporting inspection commands.

The session runtime owns:

- workspace binding
- prompt assembly
- routing between normal chat, repository work, and tool-assisted turns
- tool assembly
- step limits and final response handling
- session logging and local artifacts

`sylliptor chat` is interactive and supports commands such as `/status`,
`/mode`, `/pwd`, `/plan`, `/subagent`, and `/forge`.

`sylliptor run` is the one-shot entrypoint. It is best for focused tasks that
can be completed from a single instruction. For exploratory or highly iterative
work, interactive chat or Forge is usually a better fit.

### Model Provider Layer

Sylliptor talks to model providers through configured API profiles. The default
transport is OpenAI-compatible chat completions, with provider-specific request
normalization where needed.

Model choice, base URL, API key source, timeout, reasoning options, and role
overrides are resolved before a session starts. Provider credentials are never
embedded in project files by Sylliptor.

### Built-In Tools

Built-in tools are host-owned Python implementations. They cover local
filesystem operations, search, symbol lookup, shell execution, git history, web
fetch/search, session history, verification, and related workflow helpers.

A tool being implemented does not mean it is always visible to the model. The
actual tool surface depends on:

- execution mode
- runtime kind
- workspace binding
- sandbox readiness
- feature configuration
- extension trust and filtering

Web access is optional. `web_fetch` retrieves a specific URL, while
`web_search` is a discovery tool that appears only when a supported search
runtime is configured. See [Reference](reference.md) for the current web-search
scope.

### Workspace And Safety

Sylliptor resolves a workspace before exposing local tools. Relative
file/search/shell paths are interpreted inside that workspace. Broad paths such
as a home directory require explicit confirmation or override, and the
filesystem root is blocked as a workspace root.

Execution modes define the default approval posture:

- `readonly`: inspect only
- `review`: ask before writes and sensitive commands
- `auto`: allow routine edits with fewer prompts
- `fullaccess`: remove mode-level prompts for trusted workspaces

Shell and verification commands can run through sandbox backends such as Docker
or Bubblewrap when configured. Network and URL handling use host-side safety
checks before requests are made.

### Verification

Verification is treated as part of the runtime contract, not as a free-form
model convention. Sylliptor can infer likely commands from the workspace,
accept explicit `--verify-cmd` values, and expose a `verify_run` tool when
verification is enabled for the session.

Forge workflows can make verification authoritative for execution gates. Normal
chat and one-shot sessions use verification as task evidence and completion
support.

### Extensions

Sylliptor has several extension points. They are intentionally separated so
each has a clear trust model.

- MCP connects external Model Context Protocol servers. User configuration is
  higher trust; project configuration can only narrow exposure.
- Skills are local instruction bundles rooted at `SKILL.md`. They provide
  reusable workflow guidance and are loaded progressively.
- Plugins package trusted bundles that may contribute skills, custom tools, MCP
  servers, and hooks.
- Custom tools are trusted Python files with a manifest and `run(args)`
  entrypoint. Project tools require an explicit trust decision.
- Hooks run deterministic command-based policy or automation around lifecycle
  events and require trust for project-local configuration.
- Subagents are focused helper sessions used by normal chat and one-shot flows
  for exploration, review, or testing strategy.

See [MCP](mcp.md), [Skills](skills.md), [Plugins](plugins.md),
[Custom tools](custom_tools.md), [Lifecycle hooks](hooks.md), and
[Subagents](subagents.md) for the user-facing contracts.

### Forge

Forge is the plan-driven workflow for larger tasks. It creates a structured
plan, executes scoped tasks, records verification and review evidence, and keeps
run artifacts under the workspace runtime directory.

Forge is stricter than normal chat. It expects scoped tasks, concrete write
paths, and clear verification commands for strict gates. Use it when a change
is large enough that planning, task boundaries, and review artifacts are useful.

See [Forge](forge.md) for commands and operational guidance.

### Server Mode

Server mode exposes Sylliptor through an HTTP API for managed runs and worker
jobs. It reuses the same workspace binding, execution modes, tool policy, and
artifact model as the CLI.

See [Server mode](server.md) for API and deployment details.

## Local State And Artifacts

Sylliptor writes local state for sessions, logs, verification output, Forge
runs, knowledge artifacts, tool-output artifacts, and feedback exports. Runtime
artifacts are kept separate from source files and are excluded from normal
project work whenever possible.

The important rule is that artifacts are evidence, not hidden authority. The
host decides which artifacts are used for resuming, verification, planning,
Forge execution, and feedback export.

## Trust Boundaries

Sylliptor keeps trust-boundary decisions in the host runtime: external content
may inform a session, but it cannot override system instructions, user
instructions, execution mode, sandbox settings, workspace binding, or host-owned
policy checks.

See [Security model](security_model.md) for the full list of untrusted inputs
and the rules that apply to them.

## Where To Go Next

- [Quickstart](quickstart.md): install and run the first session.
- [Reference](reference.md): detailed runtime behavior and configuration.
- [Security model](security_model.md): trust boundaries and sandboxing.
- [Shell sandbox](shell_sandbox.md): Docker and Bubblewrap setup.
- [Forge](forge.md): plan-driven execution.
- [MCP](mcp.md): external server integration.
- [Skills](skills.md): reusable instruction bundles.
- [Plugins](plugins.md): extension packaging.
- [Custom tools](custom_tools.md): trusted Python tools.
- [Lifecycle hooks](hooks.md): command-based policy and automation.
- [Server mode](server.md): HTTP API operation.
