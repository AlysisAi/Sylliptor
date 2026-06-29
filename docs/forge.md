# Forge

Forge is Sylliptor's plan-driven workflow for larger coding tasks. It turns a broad request into
an explicit task plan, executes scoped tasks, and keeps verification and review evidence visible.

Use it for multi-file implementation work, staged refactors, release cleanup, or any task where you
want an explicit plan before changes are made.

## Start From Chat

From a repository workspace:

```bash
sylliptor chat
/forge
```

Inside Forge, use the on-screen plan commands to refine the goal, inspect tasks, edit the plan,
and execute approved work. `/back` returns to normal chat while preserving the current run pointer
for the same workspace.

Use `/forge resume` when you want to attach explicitly to the current run pointer instead of
starting a fresh run for the chat session.

## Direct CLI Flow

Create or open a plan:

```bash
sylliptor forge plan --path .
sylliptor forge show --path .
sylliptor forge status --path .
```

Execute one task from the plan:

```bash
sylliptor forge exec T01 --path . --mode review
```

Run a PR-style local flow for one task:

```bash
sylliptor forge exec T01 --path . --pr --verify strict --review
```

Run multiple ready tasks in batches:

```bash
sylliptor forge swarm --path . --parallel 3 --mode auto --verify warn
```

Preview the swarm schedule without executing:

```bash
sylliptor forge swarm --path . --dry-run
```

Review a task result:

```bash
sylliptor forge review T01 --path .
```

## Plan Artifacts

Forge stores run state under the workspace's Sylliptor runtime directory. The important artifacts
are the structured task plan, a human-readable plan summary, per-task execution logs, verification
results, and review outputs.

Tasks should stay small and scoped. A good task has:

- a clear objective
- explicit write paths
- verification commands or acceptance criteria
- dependencies on earlier tasks when needed

## Execution And Review

`forge exec` runs a single task. By default, write-scope enforcement is strict. Use `--scope warn`
or `--scope off` only when a task legitimately needs broader edits.

`--pr` creates a local PR-style flow around the task: branch, execute, commit, verify, review, and
merge back when the gates pass. `--keep-branch` keeps the task branch for debugging.

`--verify` controls verification policy:

- `off`: do not run the verification gate
- `warn`: collect verification output without hard-failing normal execution; in `--pr` mode, failed
  verification still blocks merge
- `strict`: fail the task when verification fails

Repeat `--verify-cmd` to provide explicit verification commands.

## Swarm Runs

`forge swarm` executes multiple ready tasks from the current plan. It respects dependencies and
integrates successful task branches in batches.

Useful controls:

- `--parallel <n>` sets worker concurrency.
- `--max-tasks <n>` limits one run.
- `--only T01,T02` runs selected tasks while still enforcing dependencies.
- `--retry-failed` and `--retry-changes-requested` include tasks that were previously not accepted.
- `--integration-verify` controls the verification gate after a batch is integrated.
- `--replan suggest|apply` enables between-batch replanning.

Start with `--dry-run` when reviewing a plan for the first time.

## Current Scope

Forge is intentionally stricter than normal chat. It expects a concrete workspace, small scoped
tasks, and clear verification commands for strict gates.

- `forge exec --pr` is the strongest acceptance path because it wraps execution in branch, commit,
  verification, review, and merge gates. Plain `forge exec` keeps a simpler local execution flow and
  should be reviewed before you commit or merge results manually.
- A PR-style merge requires verification and review gates to pass. `--verify warn` records a
  non-strict verification failure as evidence, but it does not merge a failing task branch.
- Sequential execution and swarm workers currently run without subagents. Use top-level
  `sylliptor chat` or `sylliptor run` for delegated exploration before starting Forge execution, or
  split the plan into smaller scoped tasks.
- Strict verification needs explicit or inferable commands. If Sylliptor cannot determine what to
  run, provide `--verify-cmd`.
- Forge does not use a persistent partial-success task status. Incomplete work is represented by
  task status, reports, verification output, review results, and execution artifacts.
- Image handling uses conservative budget reserves; Sylliptor does not claim provider-exact vision
  token accounting for Forge execution.

## Modes And Safety

Forge follows the same execution modes as the rest of Sylliptor:

- `readonly` inspects and plans only.
- `review` asks before writes and shell commands.
- `auto` can apply approved changes with fewer prompts.
- `fullaccess` disables mode-level write and shell prompts; use only in trusted workspaces.

For public projects, start with `review` and move to `auto` only after the plan, write scopes, and
verification commands are clear.

## Practical Guidance

- Keep the initial request specific enough to identify target behavior and files.
- Review task scopes before execution.
- Prefer smaller batches for unrelated subsystems.
- Treat failed verification as review evidence, not as noise to hide.
- Commit or merge only after reviewing the final diff and verification output.

See [Execution modes](../README.md#execution-modes), [Shell sandbox](shell_sandbox.md), [Security model](security_model.md),
and [MCP](mcp.md) for the lower-level controls Forge builds on.
