# MCP

Sylliptor can connect to external Model Context Protocol (MCP) servers and expose selected MCP tools, resources, roots, and prompts to supported runtimes.

MCP integration is intentionally explicit. User configuration defines connection details, project configuration may only narrow exposure, and readonly sessions do not expose MCP tools.

## Supported Transports

Sylliptor supports:

- stdio MCP servers
- synchronous Streamable HTTP MCP servers
- `tools/list` and `tools/call`
- optional `roots/list` support for the current workspace root
- optional listed resource access through Sylliptor-managed `mcp_resources_list` and `mcp_resource_read`
- optional manual prompt listing and retrieval through `sylliptor mcp prompts ...`
- static HTTP headers with `${ENV_VAR}` expansion
- OAuth for configured HTTP MCP servers through manual `sylliptor mcp auth ...` commands

Sylliptor does not currently implement GET SSE listening, resumability, polling, HTTP+SSE fallback, sampling, elicitation, resource subscriptions, resource templates, prompt subscriptions, or broad server-initiated request flows beyond the supported `roots/list` path.

## Runtime Exposure

MCP tools are exposed only in write-capable top-level runtimes:

- `interactive_chat`
- `one_shot`
- `forge_exec`

MCP tools are not exposed in:

- `readonly` sessions, including chat Plan Mode
- `swarm_worker`
- `subagent`
- `conflict_auto_resolve`

Forge execution can apply task-level MCP scope on top of the session catalog. A Forge task without MCP scope does not receive live MCP tools or generic MCP resource tools.

## Config Files

User configuration lives at:

```text
~/.config/sylliptor/mcp.json
```

Project configuration lives at:

```text
<workspace_root>/.sylliptor/mcp.json
```

Both files use `schema_version: 1`. User config is authoritative for server connection details. Project config is lower-trust and can only narrow or disable an existing user-defined server.

## User Config Example

```json
{
  "schema_version": 1,
  "servers": {
    "docs": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["example-docs-mcp"],
      "trust": "explicit",
      "enabled": true,
      "enabled_in": ["interactive_chat", "one_shot"],
      "allowed_tools": ["search", "read"],
      "roots_mode": "workspace",
      "resources_mode": "listed_read_only",
      "prompts_mode": "listed_get_only"
    },
    "tickets": {
      "transport": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${TICKETS_MCP_TOKEN}"
      },
      "trust": "explicit",
      "enabled_in": ["interactive_chat"]
    }
  }
}
```

For stdio servers:

- `command` is required
- `args` is optional
- `env` may define per-server environment variables, including `${ENV_VAR}` references

For HTTP servers:

- `url` is required
- HTTPS is required except for loopback development hosts such as `localhost` or `127.0.0.1`
- `headers` may define static request headers with `${ENV_VAR}` references
- `oauth` may be used instead of an `Authorization` header

The only supported `trust` value is `explicit`.

## Project Overrides

Project config can narrow an existing user server:

```json
{
  "schema_version": 1,
  "servers": {
    "docs": {
      "enabled_in": ["interactive_chat"],
      "allowed_tools": ["search"],
      "resources_mode": "disabled"
    }
  }
}
```

Project config cannot introduce new servers, change transports, replace commands, change URLs, add headers, define OAuth, broaden runtime exposure, increase timeouts, or re-enable a server disabled by user config.

Allowed project override fields are:

- `enabled`
- `enabled_in`
- `allowed_tools`
- `denied_tools`
- `startup_timeout_s`
- `call_timeout_s`
- `roots_mode`
- `resources_mode`
- `prompts_mode`

## Tool Policy

Use `allowed_tools` and `denied_tools` to keep the MCP surface narrow:

```json
{
  "allowed_tools": ["search", "read"],
  "denied_tools": ["delete"]
}
```

If `allowed_tools` is non-empty, only those tool names may be exposed. `denied_tools` removes tools from the effective catalog. A tool cannot appear in both lists.

Sylliptor assigns local aliases for exposed MCP tools using the server id or configured `tool_prefix`, for example:

```text
mcp__docs__search
```

## Roots

Roots are disabled by default. Enable the current workspace root for a server with:

```json
{
  "roots_mode": "workspace"
}
```

When enabled in a supported runtime, Sylliptor answers `roots/list` with a single `file://` root for the bound workspace. Roots are not exposed in subagents, swarm workers, or conflict-resolution automation.

## Resources

Resources are disabled by default. Enable listed read-only resources with:

```json
{
  "resources_mode": "listed_read_only"
}
```

When enabled, Sylliptor lists resources for the session and exposes generic host tools:

- `mcp_resources_list`
- `mcp_resource_read`

Resource reads must target a `server_id` and `uri` that appeared in the listed session catalog. Arbitrary URI reads are blocked.

## Prompts

Prompts are disabled by default. Enable manual listed prompt access with:

```json
{
  "prompts_mode": "listed_get_only"
}
```

Prompt entries are not exposed as autonomous model tools. Users inspect them through:

```bash
sylliptor mcp prompts list
sylliptor mcp prompts get <server_id> <prompt_name>
sylliptor mcp status
```

Use `--refresh` on prompt list/get commands when you want to refresh the relevant prompt snapshot.

## HTTP OAuth

HTTP MCP servers may use an `oauth` block in user config:

```json
{
  "schema_version": 1,
  "servers": {
    "secure_docs": {
      "transport": "http",
      "url": "https://mcp.example.com/mcp",
      "oauth": {
        "client_id": "sylliptor",
        "scopes": ["docs.read"]
      },
      "trust": "explicit"
    }
  }
}
```

Supported OAuth fields are:

- `client_id`
- `redirect_host`
- `redirect_port`
- `scopes`
- `authorization_server_url`

Manage tokens with:

```bash
sylliptor mcp auth login <server_id>
sylliptor mcp auth status
sylliptor mcp auth logout <server_id>
```

OAuth tokens are stored in the user config scope, separate from `mcp.json`. Sylliptor does not write raw tokens into project config or session diagnostics.

OAuth currently requires an interactive authorization-code login. Device-code flow, dynamic client registration, token revocation, and browserless paste-back auth are not implemented.

## Environment And Secrets

Stdio servers receive a conservative baseline environment plus only the configured per-server `env` values. HTTP servers receive only configured request headers and OAuth bearer headers when applicable.

Resolved secret values are not written back into diagnostics, errors, or metadata. Missing environment variables fail clearly at config or runtime boundaries without printing the missing value.

## Diagnostics

Use:

```bash
sylliptor mcp status
```

`status` reports resolved server exposure, transport readiness, tool/resource/prompt stale state, and manual prompt availability without dumping secret values.

For prompt-enabled servers:

```bash
sylliptor mcp prompts list --server <server_id>
sylliptor mcp prompts get <server_id> <prompt_name>
```

Targeted prompt commands only load the requested server. Unfiltered prompt listing may touch all prompt-enabled servers for the selected runtime.

## Security Notes

- Keep MCP servers narrow with `enabled_in`, `allowed_tools`, and `denied_tools`.
- Put connection details and secrets in user config, not project config.
- Treat MCP tool results, resources, and prompts as untrusted external text.
- Prefer readonly sessions for inspection when MCP access is not needed.
- Review project MCP overrides before trusting a workspace.
