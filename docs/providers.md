# Providers And Models

Sylliptor can use API-key profiles, a Sylliptor account, or a supported AI
subscription. The setup wizard is the simplest place to choose:

```bash
sylliptor setup
```

## API-Key Providers

Start from a preset or configure an OpenAI-compatible endpoint:

```bash
sylliptor profile presets
sylliptor profile preset openai
sylliptor profile use openai
sylliptor config set-api-key
```

```bash
export SYLLIPTOR_API_KEY="YOUR_KEY"
sylliptor config set base_url "https://example.com/v1"
sylliptor config set model "your-model"
```

OpenAI, Anthropic, and Gemini profiles can use native protocols. Other
providers and gateways can use the OpenAI-compatible protocol.

## Sylliptor Account

Use `sylliptor login`, `sylliptor whoami`, and `sylliptor logout` for the hosted
Sylliptor account.

## ChatGPT Codex Subscription

Connect a supported ChatGPT subscription in the browser:

```bash
sylliptor auth login openai-codex
```

Use device-code login when a browser callback is unavailable:

```bash
sylliptor auth login openai-codex --device-code
```

Manage the connection with:

```bash
sylliptor auth list
sylliptor auth status openai-codex
sylliptor auth logout openai-codex
```

After login, open `/config` → **Default Model** to choose a compatible model
and reasoning effort. Sylliptor keeps its own agent, tools, approvals, and TUI;
the subscription supplies model access.

## Runtime Options

Reasoning effort controls the model setting. `/trace off|compact|full` controls
only the safe reasoning summaries and tool progress shown by Sylliptor.

Streaming is enabled by default and can be changed with `/stream` or the
`--stream` and `--no-stream` options.

Prompt caching and web search are provider-aware. Configure them through
`/config`; web search uses keyless DDGS as its public-web fallback. See the
[Reference](reference.md) for the related settings.

## Related Guides

- [Quickstart](quickstart.md)
- [Credentials](credentials.md)
