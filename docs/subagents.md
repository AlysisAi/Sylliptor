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
- Running `/subagent <name> <task>` auto-enables subagents for the current session if they were off.
- `/subagent` with no args opens an interactive picker when available; otherwise it prints a usage panel with examples and available subagents.
- With trace set to `compact` or `full`, subagent runs stream nested live tool progress under the parent session instead of staying silent until completion.

Built-in subagents:

- `explorer` (read-only repository exploration with concise evidence-first findings)
- `reviewer` (strict read-only review with verdict + blocking/non-blocking issues)
- `test-strategist` (read-only high-value testing plan focused on regressions)

Built-in prompt behavior:

- enforces read-only behavior (no file edits claimed)
- requires concise structured output (not long transcripts)
- always includes docs impact (`README.md` / `docs/`) and test impact (tests + commands)

Discoverability:

- when subagents are enabled, the `subagent_run` tool schema exposes available subagent names in `name.enum`
- enum values are built from the loaded registry, so custom subagents are included automatically
- the main agent also gets a pinned `<subagent_context>` message with available subagents and delegation guidance, so it can autonomously decide when to call `subagent_run`
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
- Default subagent mode is `readonly` unless explicitly overridden.
- Review-mode subagents suppress nested UI chatter but forward approval prompts through the parent session surface.
- Non-interactive subagent sessions still fail fast when a nested write/shell/verify action would require confirmation.
- For precise inspection, prefer `symbol_search` for Python/JS/TS symbol navigation, `search_rg` for broader text hits, `fs_read_lines` to read the exact surrounding range, and `fs_read` when broader file context is needed.
- For history or regression questions, prefer `git_history` over raw shell commands.
- Subagent execution mode is capped by the parent session mode (no privilege escalation): readonly < review < auto < fullaccess.
- Subagent token/cost usage is replayed into the parent session one child model call at a time, including failed subagent runs, so `/usage` and the chat HUD preserve call counts and api-vs-estimate attribution.
