# Providers And Models

Sylliptor can use API-key profiles, a Sylliptor account, or a supported AI
subscription. Choose a connection during setup, or open `/login` later.

## API-Key Providers

Open `/config`, choose an API-key connection, and enter the endpoint, model,
and key when prompted.

OpenAI, Anthropic, and Gemini profiles can use native protocols. Other
providers and gateways can use the OpenAI-compatible protocol.

## Sylliptor Pro Account

Open `/login`, choose **Sylliptor**, and approve the short one-time code in your
browser. The flow does not need a localhost callback, so it also works over SSH
and in WSL. Use `/logout` to disconnect the account.

## ChatGPT Codex Subscription

Open `/login`, choose the supported ChatGPT connection, and complete the
browser flow. A device-code option is available when a browser callback cannot
reach the terminal.

After login, open `/config` → **Default Model** to choose a compatible model
and reasoning effort. Sylliptor keeps its own agent, tools, approvals, and TUI;
the subscription supplies model access.

## Runtime Options

Reasoning effort controls the model setting. `/trace off|compact|full` controls
only the safe reasoning summaries and tool progress shown by Sylliptor.

Streaming is enabled by default and can be changed with `/stream`.

Prompt caching and web search are provider-aware. Configure them through
`/config`; web search uses keyless DDGS as its public-web fallback. See the
[Reference](reference.md) for the related settings.

## Related Guides

- [Quickstart](quickstart.md)
- [Credentials](credentials.md)
