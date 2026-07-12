# Maintenance Scripts

This directory is for maintainer scripts that are not imported by the runtime
package during normal Sylliptor execution.

## Catalog Scripts

- `refresh_litellm_model_catalog.py` refreshes the bundled LiteLLM model pricing
  snapshot used by model metadata and packaging checks.
- `refresh_chatgpt_codex_model_catalog.py` refreshes the bundled ChatGPT Codex
  subscription model snapshot. It is a maintainer operation and requires an
  authenticated source configured by the caller.

## Scope

Scripts may use network access or local tools. Review the script before running
it, and prefer explicit environment variables over persistent local state.

User-facing CLI behavior should live under `src/sylliptor_agent_cli/`, not in
this directory.

## See Also

- [Contributing](../CONTRIBUTING.md)
- [Release process](../RELEASING.md)
