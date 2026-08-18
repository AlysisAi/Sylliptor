# Subagents

Subagents are focused helpers that work on a smaller part of a task and report
back to the main session. Sylliptor can delegate to them when it helps, or you
can choose one directly with `/subagent`.

They are useful when a task benefits from independent investigation, a second
review, or parallel work with clear boundaries.

## Available Subagents

- **Explorer** investigates the repository without changing it.
- **Implementer** makes one clearly scoped change and verifies it.
- **Frontend Engineer** works on web interfaces, including responsive and
  accessible states.
- **Debugger** reproduces and isolates a problem without editing source files.
- **Code Reviewer** reviews a change and separates blocking issues from smaller
  suggestions.
- **Test Strategist** proposes the highest-value regression coverage without
  writing tests.
- **Visual Designer** creates raster assets when image generation is enabled.

Open `/subagent` to see what is available in the current session. You can also
be explicit:

```text
/subagent explorer find where authentication state is stored
```

Use `/subagent on`, `/subagent off`, or `/subagent status` to control delegation
for the session. The setting is also available in `/config`.

## How Delegation Works

Each subagent receives a focused task and only the tools allowed by its role.
Independent subagents may run in parallel. The main session collects their
results and remains responsible for the final answer and any follow-up work.

Subagents cannot create more subagents. Their execution mode can only stay at
or below the parent session's mode, and every run has a finite time limit.
Forge workers do not expose subagents, so use them while exploring or planning
before entering Forge execution.

Treat a subagent report as evidence, not as automatic approval. Review important
findings and verify repository changes before committing them.

## Custom Subagents

Project-specific subagents live in `.sylliptor_agents/*.md`. User-wide ones live
in the Sylliptor config directory under `agents/`.

A definition uses YAML frontmatter followed by a short instruction:

```md
---
name: api-reviewer
description: Reviews API compatibility
mode: readonly
allow_tools:
  - fs_read
  - fs_read_lines
  - fs_list
  - symbol_search
  - search_rg
---
Focus on breaking API changes and missing tests. Report evidence with file paths.
```

Custom instructions extend Sylliptor's normal safety rules; they do not replace
them. Keep each role narrow, give it only the tools it needs, and describe the
result you expect it to return.
