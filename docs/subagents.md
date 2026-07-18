# Subagents

## Subagents (Non-Swarm)

Subagents are optional focused helper agents that run as nested, isolated sessions from normal
`sylliptor run` / `sylliptor chat` (non-swarm) flows.

Forge execution and swarm workers currently do not expose subagents. Use subagents during
top-level exploration or planning, then execute scoped Forge tasks directly.

Default behavior is ON for top-level chat/run sessions. Use `--no-subagents`, `/subagent off`, or `sylliptor config set subagents_enabled false` to disable it.

Enable/disable options:

- config: `sylliptor config set subagents_enabled true|false`
- per command: `sylliptor run --subagents ...` / `sylliptor run --no-subagents ...`
- per command: `sylliptor chat --subagents` / `sylliptor chat --no-subagents`
- in chat: `/subagent on|off|status`

UX behavior in chat:

- Toolbar includes `subagents=on|off` so the current state is always visible.
- Users can describe the outcome in normal chat; they do not need to select a subagent or name an internal tool. The semantic router receives a capability catalog and sends executable deliverable requests to the repository agent loop, which chooses the appropriate specialist.
- Running `/subagent <name> <task>` auto-enables subagents for the current session if they were off.
- `/subagent` is a one-shot invocation, not a persistent conversation mode. Follow-up chat messages return to the main agent unless another explicit invocation is made.
- A task beginning with another `/subagent` command is rejected with guidance to run the intended inner command directly; subagent commands are never parsed as nested task text.
- `/subagent status` reports callable roles and capability-gated roles with their concrete setup instructions.
- `/subagent` with no args opens an interactive picker when available; otherwise it prints a usage panel with examples and available subagents.
- In the TUI each subagent is identified by its name plus an activity tagline (e.g. `debugger · hunting the root cause`), shown in the picker rows, the spawn line, the live status, and the result attribution. Agents wear no per-agent symbol. Custom subagents get a stable name-derived accent colour and fall back to their description.
- In the TUI, `/subagent <name>` without a task (e.g. the picker prefill submitted early) answers with a one-line hint instead of the full usage panel; the classic CLI keeps the panel.
- While a subagent runs, the TUI footer pins `↪ <name>` in that agent's accent colour, so which agent is working is always visible; the badge clears when the run ends or is interrupted.
- With trace set to `compact` or `full`, subagent runs stream nested live tool progress under the parent session instead of staying silent until completion.

Built-in subagents:

- `explorer` (read-only repository investigation with concise evidence-first findings)
- `implementer` (write-capable implementation of one clearly scoped change, followed by verification)
- `frontend-engineer` (write-capable web UI implementation with responsive, interaction-state, accessibility, and evidence-based visual-QA requirements)
- `debugger` (diagnostic reproduction and root-cause isolation without source edits)
- `code-reviewer` (strict read-only review with verdict + blocking/non-blocking issues)
- `test-strategist` (read-only high-value testing plan focused on regressions; it does not write tests)
- `visual-designer` (opt-in production raster generation with a read-plus-generate sandbox; available only when image generation is enabled)

Built-in prompt behavior:

- gives each role a non-overlapping contract: investigate, implement general code, implement frontend UX, diagnose, review, plan tests, or generate raster assets
- keeps `explorer`, `code-reviewer`, and `test-strategist` strictly read-only
- lets `debugger` run targeted diagnostics and verification while prohibiting repository edits
- lets `implementer` make the smallest scoped change allowed by the parent session
- makes `frontend-engineer` use the repository's existing frontend stack and explicitly cover responsive layout, accessibility, and loading/empty/error/disabled states
- prevents `frontend-engineer` from calling `image_generate`; raster work belongs to `visual-designer`
- prevents `frontend-engineer` from returning a generator prompt as a substitute for an image request
- limits `visual-designer` to read-only repository tools plus `image_generate`, so it cannot edit application source or existing assets
- requires `visual-designer` to generate an actual file for in-scope bitmap requests; users never need to ask it to call a function or tool
- degrades a `visual-designer` run instead of accepting success when `image_generate` is missing from its sandbox or no successful `image_generated` event proves that an artifact was written
- requires both visual specialists to distinguish technical/build validation from visual inspection and to report `Visual QA` as pending when no real browser/vision evidence was inspected
- requires concise evidence-backed handoffs instead of action transcripts
- requires agents to report verification actually performed and remaining uncertainty or blockers

### Image generation setup

Image generation is disabled by default because calls can incur a separate provider charge. The
`visual-designer` agent and `image_generate` tool are omitted from callable roles until enabled.
They remain discoverable through `/subagent status`, direct invocation errors, and the main
agent's capability context, with an actionable reason instead of an "unknown subagent" error.
Mode-gated sessions use the effective tool surface for that context, so a readonly session reports
the concrete mode switch required instead of advertising the visual role as callable.

Configure an OpenAI-compatible image endpoint and credential, then start a new chat session:

```powershell
$env:SYLLIPTOR_IMAGE_API_KEY = "<image-provider-key>"
sylliptor config set image_generation.enabled true
sylliptor config set image_generation.model gpt-image-1
# Optional when the active provider profile is not the image provider:
sylliptor config set image_generation.base_url https://api.openai.com/v1
```

Instead of `SYLLIPTOR_IMAGE_API_KEY`, set
`image_generation.api_key_env` to the name of a provider-specific environment variable. If neither
is configured, Sylliptor reuses the active session credential. Generation accepts PNG, JPEG, and
WebP output paths, creates one to four new files, validates decoded image dimensions and format,
and never overwrites an existing file.

Optional safety and provider limits are configurable through
`image_generation.timeout_s`, `image_generation.max_images_per_call`,
`image_generation.max_image_bytes`, and `image_generation.max_pixels`.

### Test each built-in alone

Use explicit `/subagent` calls in chat to bypass autonomous role selection and exercise one role at
a time:

```text
/subagent frontend-engineer Implement the settings card in src/web/... using the existing design system. Cover mobile/desktop, loading/error/disabled states, keyboard access, and run the focused frontend checks. Report visual QA evidence separately.

/subagent visual-designer Create one transparent product illustration for the empty state. Inspect existing assets under src/web/assets first, write a new PNG under that convention, and report dimensions/hash plus visual-QA status. Do not edit source code.
```

Run `/subagent status` first to confirm the desired role is available. The visual role appears only
as callable in a session started after image generation was enabled. The task itself should state
only the desired creative outcome and output constraints; never instruct the agent to call
`image_generate`.

Discoverability:

- when subagents are enabled, the `subagent_run` tool schema exposes available subagent names in `name.enum`
- enum values are built from the loaded registry, so custom subagents are included automatically
- the main agent also gets a pinned `<subagent_context>` message with available and capability-gated subagents plus delegation guidance, so it can autonomously decide when to delegate or report a grounded blocker
- repo turns also get a bounded turn-scoped `<subagent_turn_context>` when subagents are available; explicit user requests for subagent/delegation behavior are treated as required before finalizing
- if a user explicitly asks for subagents while they are disabled or unavailable, the turn reports that blocker instead of silently continuing without delegation
- interactive repo execution turns that spend multiple read-only steps without subagent delegation receive a runtime nudge to delegate focused exploration or move to implementation/verification
- when enabled, the main system prompt also adds short delegation guidance for autonomous subagent_run use (disabled sessions keep baseline prompt behavior)

Custom subagents can be defined with YAML frontmatter + markdown body in:

- project: `./.sylliptor_agents/*.md`
- user: `~/.config/sylliptor/agents/*.md`

Frontmatter example:

```md
---
name: api-reviewer
description: API-focused reviewer
mode: readonly
allow_tools:
  - fs_read
  - fs_read_lines
  - fs_list
  - symbol_search
  - search_rg
deny_tools:
  - shell_run
# Claude-style aliases are also supported:
# tools: [fs_read, fs_read_lines, fs_list, symbol_search, search_rg]
# disallowedTools: [shell_run]
model_role: review
---
You are a strict API reviewer. Focus on breaking changes and missing tests.
```

Notes:

- Custom markdown bodies are treated as scoped subagent guidance, not as a full system-prompt replacement.
- Base `SYSTEM_PROMPT` is always preserved for subagent sessions.
- Built-in/code-owned subagent prompts may add trusted system-layer guidance; user-defined subagents provide scoped guidance and cannot replace the base system prompt.
- Main agent gets only the final subagent result, not intermediate nested tool outputs.
- Subagents cannot invoke other subagents (no recursion).
- Tool permissions are sandboxed by per-subagent allow/deny lists.
- Each child session receives an exact model-visible catalog of the filtered
  tool names it may call, with required arguments. The catalog is built after
  permission filtering and tells the child not to invent aliases; unknown-tool
  recovery remains active inside the child session.
- Default subagent mode is `readonly` unless explicitly overridden.
- Review-mode subagents suppress nested UI chatter but forward approval prompts through the parent session surface.
- Non-interactive subagent sessions still fail fast when a nested write/shell/verify action would require confirmation.
- Ordinary subagents have no default step ceiling. They continue until their
  delegated task completes, they are cancelled or blocked, or they encounter a
  fatal error. The optional `max_steps` tool argument adds an explicit safety
  limit for that child only.
- Every ordinary subagent has a finite wall-clock ceiling. Its effective
  deadline is the earlier of the active parent deadline and the
  `subagent_timeout_s` fallback (900 seconds by default). An earlier parent
  deadline is reused exactly, so delegation cannot extend the parent run; when
  the parent has no active deadline, the fallback supplies the child ceiling.
  Configure it with, for example,
  `sylliptor config set subagent_timeout_s 600`. The value must be finite and
  greater than zero.
- Child LLM and tool-call timeouts clamp against that resolved child deadline.
  Lifecycle telemetry records `subagent_timeout_s`, `resolved_timeout_s`,
  `resolved_deadline_source`, and the full resolved `deadline` snapshot.
- If too little hard time remains, or the parent has entered the soft
  finalization window, `subagent_run` refuses to launch and returns the usual
  error-shaped result with deadline metadata such as
  `failure_category: "deadline"`, `deadline_prevented_launch`,
  `deadline_start_decision`, and `remaining_seconds`.
- Release changes to subagent deadline propagation should run the focused
  deadline, cancellation, runtime, and TUI suites before the complete test suite.
- For precise inspection, prefer `symbol_search` for Python/JS/TS symbol navigation, `search_rg` for broader text hits, `fs_read_lines` to read the exact surrounding range, and `fs_read` when broader file context is needed.
- For history or regression questions, prefer `git_history` over raw shell commands.
- Subagent execution mode is capped by the parent session mode (no privilege escalation): readonly < review < auto < fullaccess. Built-in definitions additionally restrict their visible tools; for example, `debugger` has diagnostic tools but no direct file-edit tools.
- Subagent token/cost usage is replayed into the parent session one child model call at a time, including failed subagent runs, so `/usage` and the chat HUD preserve call counts and api-vs-estimate attribution.
