# Credentials

Sylliptor now supports an explicit persisted API-key store separate from `config.json`.

API key resolution order:

1. Per-command override (`--api-key`, `--api-key-env`, `--api-key-stdin`)
2. `SYLLIPTOR_API_KEY`
3. Persisted credentials (`sylliptor config set-api-key`)
4. `OPENAI_API_KEY`

Notes:

- Persisted credentials are stored in the user config directory at `credentials.json`.
- The main `config.json` still stores non-secret settings such as `model` and `base_url`.
- `sylliptor config show` reports whether an API key is available and which source won.
- `sylliptor setup` can save an API key into the local credentials store.

MCP HTTP OAuth tokens use a separate user-scope store at `mcp_oauth_tokens.json` in the same config
directory. That file is an AES-GCM encrypted envelope containing `version`, `key_source`, `nonce`,
and `ciphertext`; it does not contain plaintext access tokens or refresh tokens. The AES master key
is stored in the OS keychain via `keyring` when possible. On Windows without keyring, Sylliptor uses
DPAPI for the local master key. On macOS and Linux without a working keychain, Sylliptor uses a weak
derived-key fallback with a persistent sibling salt file named `mcp_oauth_tokens.salt`; this fallback
is weak against same-user compromise and exists only as a stopgap for keychain-less environments.

Legacy plaintext `mcp_oauth_tokens.json` files remain readable for one release as a migration bridge.
On first successful read, Sylliptor rewrites the token store as an encrypted envelope. If that
encrypted rewrite fails, the legacy plaintext file is left intact so the user is not locked out.

Useful commands:

```bash
sylliptor config set-api-key
sylliptor config clear-api-key
sylliptor config show
```
