# Credentials

Sylliptor keeps API keys and subscription credentials separate from ordinary
settings in `config.json`.

## API Keys

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

## Account And Subscription Credentials

Account and subscription connections use `sylliptor login` and
`sylliptor auth`. Their credentials are stored separately from non-secret
profile settings. See [Providers and models](providers.md) for setup.

## MCP OAuth Tokens

MCP HTTP OAuth tokens use a separate user-scope store at `mcp_oauth_tokens.json` in the same config
directory. That file is an AES-GCM encrypted envelope containing `version`, `key_source`, `nonce`,
and `ciphertext`; it does not contain plaintext access tokens or refresh tokens. The AES master key
is stored in the OS keychain via `keyring` when possible. On Windows without keyring, Sylliptor uses
DPAPI. On macOS and Linux without a working keychain, the current format uses an atomically created
per-store random key file with restrictive permissions.

Legacy plaintext and older encrypted stores are migrated to the current envelope on successful
read. Migration and key rotation are atomic; failed rewrites do not replace the last readable store.

Useful commands:

```bash
sylliptor config set-api-key
sylliptor config clear-api-key
sylliptor config show
```
