# Sandbox Image

This directory contains the Dockerfile used to build Sylliptor sandbox images
for shell execution, verification, and server workers when Docker is selected.

## Contents

- `Dockerfile` builds the supported `base`, `dev`, and `server` variants through
  the `VARIANT` build argument.

## Scope

The image is one execution backend, not the whole security model. Execution
modes, workspace binding, safe HTTP checks, MCP policy, hook trust, and tool
validation still apply outside the container.

Production deployments should pin images by digest and verify signatures and
attestations as described in the sandbox guide.

## Development

Image changes should be checked with `sylliptor sandbox doctor --smoke` against
a locally built image when Docker is available.

## See Also

- [Shell sandbox](../docs/shell_sandbox.md)
- [Server mode](../docs/server.md)
