# Research task: model capability contracts for Sylliptor's runtime layer

Follow-up to `MODEL_CATALOG_REFRESH_2026-07-19.md` (read it first — its per-preset `notes` are your starting leads). The static catalog is now applied. The next engineering work is a **model capability layer**: a real `reasoning_mode` + `context` field on presets, live model discovery, and tier-aware context accounting. Your job is to produce the verified data that layer will be built from. Sylliptor is an agentic coding CLI: every request carries tool definitions, and wrong reasoning flags are now **API errors** on several providers, not just degraded output.

Providers in scope (preset key → endpoint): openai + openai-responses (api.openai.com), anthropic ×3 (api.anthropic.com), gemini ×3 (generativelanguage.googleapis.com), deepseek (api.deepseek.com + /anthropic), qwen-intl/-us/-cn (dashscope-*.aliyuncs.com), zhipu (open.bigmodel.cn + api.z.ai), moonshot/-cn (api.moonshot.ai/.cn), kimi-code (api.kimi.com/coding/v1), minimax (api.minimax.io), bytedance (ark.cn-beijing.volces.com), groq (api.groq.com/openai/v1), cerebras (api.cerebras.ai), mistral (api.mistral.ai), xai (api.x.ai), cohere (api.cohere.ai/compatibility/v1), openrouter, perplexity, together, fireworks.

## Part A — Reasoning parameter contract (the priority)

For every provider surface, document exactly:

1. **Parameter name and shape** for controlling reasoning: `thinking={"type": ...}`, `reasoning_effort`, `reasoning.effort`, `reasoning.mode`, provider-specific, or none.
2. **Allowed values per model family**, and the default when omitted.
3. **Failure mode on an unsupported value**: silently ignored, silently coerced, or HTTP error (record status code + error body shape if documented).
4. Per model in our current catalog: `always-on` | `optional` | `none` — and for always-on models, whether effort is adjustable.
5. **Silent substitutions** (kimi-code routes thinking-off to K2.6 — find any others).
6. **Interaction with tools and streaming**: thinking blocks in SSE, interleaved thinking, whether reasoning output counts against max_output.
7. **Compat-surface passthrough**: does the reasoning param survive OpenAI-compat layers (cohere /compatibility/v1, anthropic-compat, gemini-compat, deepseek /anthropic)? The prior refresh suspects cohere's `thinking` does NOT pass through — settle it if the docs can.

Claims from the last refresh to re-verify and extend (do not trust them, re-source them):
- moonshot: K2.7 models ERROR on `thinking={"type":"disabled"}`; k3 rejects the K2.x `thinking` param entirely and accepts only `reasoning_effort: "max"`; only kimi-k2.6 is toggleable.
- gemini 3.x: effort `{minimal,low,medium,high}` only ({low,medium,high} on Pro), no off switch.
- cerebras gpt-oss-120b: `reasoning_effort` low|medium|high, no "none"; `disable_reasoning` param unsupported since 2026-07-21.
- anthropic: Opus 4.8 / Sonnet 5 / Fable 5 = adaptive thinking, NOT extended thinking; Haiku 4.5 the inverse, no `effort` param.
- together: Kimi-K2.7-Code and MiniMax-M3 reason unconditionally, no toggle — param emission is an unsupported-param error risk.
- minimax: no reasoning control documented on any model.
- openai: `reasoning.mode=pro` replaces a -pro slug on gpt-5.6.

## Part B — Live model discovery

Per provider: the exact listing endpoint, auth requirement, response metadata (ids only vs context/capabilities/pricing), whether the list is account/tier-scoped, and documented rate limits or caching headers. Verdict per preset: `drive-from-live` | `augment-static` | `stay-static`. Prior ranking to verify: openrouter (no auth, `supported_parameters` incl. tools/reasoning) > groq (account-visible, fast churn) > cerebras (tier-scoped) > anthropic (`/v1/models` returns capabilities + token limits) > deepseek/moonshot/mistral/xai/together (ids only). Claimed not viable: kimi-code, zhipu, qwen compat-mode, cohere compat, fireworks (control-plane only), bytedance. For the viable ones, recommend a refresh strategy (on config-open probe vs daily cache vs per-session).

## Part C — Context, max output, and tier gating

Per model in the applied catalog: context window **with tier splits** (cerebras free 65K vs paid 131K; kimi-code k3 256K vs 1M Allegretto+), max output tokens (xai 4.20 family: unpublished — say so), long-context pricing boundaries (minimax M3 bills higher above 512K; qwen coder-plus tiered pricing), and token-counting endpoints (`/v1/messages/count_tokens`; Sonnet 5 emits ~30% more tokens than 4.6 — verify and quantify from official sources). Flag every model where a hardcoded context would over-promise for some accounts.

## Part D (optional, scoping only) — Perplexity Agent API client

Enough detail for a maintainer to scope a Responses-style client: request/response schema of `POST /v1/responses` (the OpenAI-SDK-compatible alias), tool-calling support and shape, streaming format, auth, and how far it diverges from OpenAI's Responses API. No implementation — a one-page divergence summary.

## Rules

1. Official docs are ground truth; registries (models.dev, LiteLLM, OpenRouter) are leads only — the last refresh caught them wrong on kimi-code, minimax, together, and qwen.
2. Every claim carries the URL you actually opened. Conflicts recorded, not resolved by vote.
3. Exact parameter spellings and value strings, copied character-for-character.
4. Where only an authenticated/billed probe can settle a question, say so explicitly and specify the exact probe (method, endpoint, body) so a maintainer can run it in one paste.
5. Deliverable: one YAML block per provider covering A+B+C with a fixed schema you define up front and keep identical across providers; then a summary table; then an **API-error hazard list** (provider, model, wrong emission, resulting error) ranked by how likely Sylliptor is to trigger it today; then a recommended implementation order.

Write the deliverable to `MODEL_CAPABILITY_CONTRACTS_<date>.md` in the repo root.
