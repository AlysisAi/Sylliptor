# Security Model

Sylliptor is a local coding agent. The host process owns policy decisions, tool registration,
workspace binding, and session logging. Model-authored text can request tools, but host-owned
runtime checks decide which tools exist and what inputs are accepted.

## Trust Boundaries

Treat these inputs as untrusted:

- repository files and project-local configuration
- MCP server output
- web responses
- custom tool output
- image/OCR/asset text
- model responses

Sylliptor keeps those inputs behind host-owned wrappers where possible. These wrappers reduce prompt
confusion, but they are not a substitute for reviewing changes before running them.

## Execution Modes

- `readonly` exposes inspection tools only.
- `review` asks before writes and shell commands.
- `auto` can make approved changes with fewer prompts.
- `fullaccess` removes mode-level write and shell approval prompts.

`fullaccess` is for trusted workspaces only. It is not a sandbox boundary and should not be used for
untrusted repositories, unknown prompts, or commands you would not run directly.

## Shell And Verification Sandboxing

Shell and verification commands can run through Docker or Bubblewrap. Production-style usage should
keep sandbox mode strict, network disabled unless needed, and Docker images pinned by digest.

The sandbox reduces the blast radius of local command execution. It does not make arbitrary code
safe, and it does not replace source review, dependency review, or operating-system permissions.

See [Shell sandbox](shell_sandbox.md) for setup, image pinning, signatures, provenance, and troubleshooting.

## HTTP And SSRF Protection

Sylliptor validates outbound HTTP targets used by web fetch/search helpers and MCP OAuth metadata
fetching before connecting.

The safe HTTP guard rejects:

- unsupported schemes such as `file:`, `data:`, `ftp:`, and `javascript:`
- embedded URL credentials
- loopback, link-local, private, unspecified, and multicast IP ranges
- hostnames that resolve to denied IP ranges
- redirect targets that fail the same validation
- responses larger than the configured byte cap

Where supported, requests connect to a validated resolved address while preserving the original
`Host` header, and redirects are revalidated at each hop.

## MCP Boundaries

Project-local MCP configuration can only narrow or disable exposure relative to user configuration.
Tool, resource, and prompt catalogs are snapshotted per session rather than mutated live in the
model-visible tool surface.

Server-authored MCP tool descriptions are intentionally omitted from model-visible descriptions.
MCP resource text is wrapped as untrusted content and size-limited before prompt inclusion. Task
execution can further restrict MCP access through task-level scope rules.

## Skills, Plugins, Hooks, And Custom Tools

Skills are instruction bundles. They are advertised briefly and read on demand; their text is still
untrusted repo or user content.

Plugins and custom tools require explicit install/trust flows. Project custom tools are trusted by
workspace path and file hash, so edits invalidate trust. Custom tool execution runs in a worker
process with declared capability checks, but it is still trusted code and should be reviewed before
use.

Lifecycle hooks are deterministic policy and automation. Project hook configuration is not trusted
by default and must be explicitly trusted. Hook output can modify prompts or tool inputs, so review
hook configs before trusting them.

## Reporting Security Issues

Report vulnerabilities privately by following the root [Security Policy](../SECURITY.md). Do not
open public GitHub issues for security bugs.
