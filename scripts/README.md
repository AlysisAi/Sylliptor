# Maintenance Scripts

This directory is for maintainer scripts that are not imported by the runtime
package during normal Sylliptor execution.

## Current Script

- `refresh_litellm_model_catalog.py` refreshes the bundled LiteLLM model pricing
  snapshot used by model metadata and packaging checks.

## Scope

Scripts may use network access or local tools. Review the script before running
it, and prefer explicit environment variables over persistent local state.

User-facing CLI behavior should live under `src/sylliptor_agent_cli/`, not in
this directory.

## See Also

- [Contributing](../CONTRIBUTING.md)
- [Release process](../RELEASING.md)
