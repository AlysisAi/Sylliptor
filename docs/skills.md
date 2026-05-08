# Skills

Skills are named instruction bundles rooted at a `SKILL.md` file. They let teams package repeatable guidance, references, scripts, and assets without changing Sylliptor itself.

Skills are prompt context, not privileged host policy. Higher-priority system instructions, direct user instructions, execution modes, workspace boundaries, and tool policy still apply.

## Skill Bundle

A skill is a directory with a required `SKILL.md` entrypoint:

```text
<skill_name>/
  SKILL.md
  references/
  scripts/
  assets/
```

Only `SKILL.md` is required. The optional directories are available for supporting material.

`SKILL.md` uses lightweight frontmatter plus a Markdown body:

```markdown
---
name: migrations
description: Help inspect, create, and verify database migrations safely.
---

Use this skill when a task involves migration review, generation, or validation.

Prefer repository-approved migration commands and inspect adjacent migration files before editing.
```

Supported frontmatter:

- `name`
- `description`

Unknown frontmatter keys are ignored for interoperability.

## Discovery Paths

Project-local roots:

- `./.sylliptor_skills/<skill_name>/SKILL.md`
- `./.agents/skills/<skill_name>/SKILL.md`
- `./.claude/skills/<skill_name>/SKILL.md`
- `./.github/skills/<skill_name>/SKILL.md`

User-global roots:

- `~/.config/sylliptor/skills/<skill_name>/SKILL.md`
- `~/.config/agents/skills/<skill_name>/SKILL.md`
- `~/.claude/skills/<skill_name>/SKILL.md`
- `~/.copilot/skills/<skill_name>/SKILL.md`

Sylliptor discovers only approved roots. It does not scan arbitrary `skills/` directories.

When multiple skills have the same name:

1. nearest project ancestor wins
2. native project roots win over interop roots at the same level
3. project-local skills win over user-global skills
4. user-global native roots win over user-global interop roots

Malformed skills are skipped and reported as discovery issues instead of crashing session startup.

## How Skills Are Used

When `skills_enabled=true`, Sylliptor advertises discovered skill names and short descriptions to the session. Full skill bodies are not injected by default.

The model can read a skill on demand with the read-only built-in tool:

```text
skill_read(name)
skill_read(name, path)
```

`path` is relative to the skill bundle and is bounded to that bundle. Typical targets are under `references/`, `scripts/`, and `assets/`.

Skills do not run scripts automatically. If a task calls for a bundled helper script, it must be run through the normal tool surface and execution-mode guardrails.

## Chat Commands

In chat:

```text
/skill
/skill migrations
/skill migrations add an audit-log migration
```

Behavior:

- `/skill` lists discovered skills
- `/skill <name>` shows information for one skill
- `/skill <name> <task...>` attaches that skill to the current turn only

Explicit skill attachment does not persist across later turns.

## CLI

Inspect skills:

```bash
sylliptor skill list
sylliptor skill info migrations
```

Create a skill:

```bash
sylliptor skill init migrations --description "Help inspect and verify DB migrations"
sylliptor skill create docs-consistency --portable
```

Validate skills:

```bash
sylliptor skill validate ./.sylliptor_skills/migrations
sylliptor skill validate --name migrations --path .
sylliptor skill validate --all --path .
```

Install skills:

```bash
sylliptor skill install ./vendor/skills/migrations
sylliptor skill install ./downloads/migrations.zip
sylliptor skill install https://example.com/team/skills.git --subdir skills/migrations
```

Enable, disable, or remove:

```bash
sylliptor skill disable migrations
sylliptor skill enable migrations
sylliptor skill remove migrations
```

Use `--project --path <workspace>` when a lifecycle operation should apply to a project scope instead of the user-global scope.

See [Skills lifecycle](skills_lifecycle.md) for the full authoring, install, validation, and removal workflow.

## Managed And Unmanaged Skills

Managed native roots:

- project-local: `./.sylliptor_skills/`
- user-global: `~/.config/sylliptor/skills/`

Managed state files:

- project: `./.sylliptor/skills.json`
- user: `~/.config/sylliptor/skills.json`

Lifecycle commands create and remove managed native skills. Interop roots remain discoverable, but Sylliptor does not delete arbitrary foreign-root bundles.

## Trust Model

Skills are untrusted instructions. Project-local skills come from the repository. User-global skills come from the user's environment. Neither can override system instructions, direct user requests, execution modes, workspace binding, or tool policy.

Review skills before relying on them, especially when they recommend commands, external services, or broad edits.

## Configuration

```bash
sylliptor config set skills_enabled true
sylliptor config set skills_auto_invoke true
```

Defaults:

- `skills_enabled = true`
- `skills_auto_invoke = true`

Set `skills_enabled=false` to disable skill discovery and `skill_read` registration. Set `skills_auto_invoke=false` for manual discovery and explicit `/skill` usage.

## Limitations

- no marketplace or registry browsing
- no packaged skill update channel
- no dynamic per-skill slash aliases
- no implicit script execution
- no arbitrary-root scanning outside approved paths
- no automatic skill loading inside swarm workers
