# Research task: refresh Sylliptor's provider model catalog

You are researching the **current (July 2026) model lineup of every LLM provider** so we can update the presets in Sylliptor, an AI coding agent. Users pick a provider preset and get a curated list of suggested models. Several of these lists are stale — users of some presets (e.g. Kimi Code) are being offered outdated models. Your job: verify every model id below against live sources, flag what is stale, and propose the corrected lineup per provider.

## How other open-source coding agents solve this (check them first)

Cross-reference what the major open-source coding agents currently ship as their model catalogs — they keep these aggressively up to date:

1. **models.dev** — the open model registry OpenCode uses. Fetch the raw data: `https://models.dev/api.json` (also browsable at models.dev). This is your single best source for exact API model ids, context windows, reasoning support, and deprecation across all providers.
2. **OpenCode** (github.com/sst/opencode) — provider/model config and docs.
3. **Kilo Code** (kilocode.ai/docs and github.com/Kilo-Org/kilocode) — provider docs list supported models per provider.
4. **Roo Code** and **Cline** — provider catalogs in their repos/docs.
5. **LiteLLM** — `model_prices_and_context_window.json` in github.com/BerriAI/litellm (exact ids + pricing + context windows).
6. **OpenRouter live catalog** — `https://openrouter.ai/api/v1/models` (ground truth for `openrouter` preset ids).

Then confirm against **each provider's official docs/pricing page** (platform.openai.com, docs.anthropic.com, ai.google.dev, api-docs.deepseek.com, DashScope/Alibaba Cloud Model Studio, bigmodel.cn, platform.moonshot.ai, platform.minimaxi.com, Volcengine Ark, console.groq.com, inference-docs.cerebras.ai, docs.mistral.ai, docs.x.ai, docs.cohere.com, docs.together.ai, docs.fireworks.ai, docs.perplexity.ai). Official docs win any conflict; note the conflict.

## Current Sylliptor catalog (verify every line)

Format: `model_id :: role - description`. Roles used: default / advanced / fast / economy / coding / reasoning / agentic / fallback.

```
openai + openai-responses (api.openai.com)
    gpt-5.5 :: default | gpt-5.5-pro :: advanced | gpt-5.4-mini :: fast
    gpt-5.4-nano :: economy | gpt-5.4 :: coding
    validation: gpt-5.4-mini · aliases: gpt-5-nano→gpt-5.4-nano

anthropic (+ -compat/-native) (api.anthropic.com)
    claude-sonnet-4-6 :: default | claude-haiku-4-5-20251001 :: fast | claude-opus-4-8 :: advanced
    validation: claude-sonnet-4-6
    aliases: claude-sonnet-4→claude-sonnet-4-6, claude-4-sonnet→claude-sonnet-4-6,
             claude-opus-4.7→claude-opus-4-7, claude-opus-4.8→claude-opus-4-8,
             claude-3-5-haiku-latest→claude-3-5-haiku-20241022

gemini (+ -compat/-native) (generativelanguage.googleapis.com)
    gemini-3.5-flash :: default | gemini-3.1-flash-lite :: fast | gemini-3.1-pro-preview :: advanced
    gemini-2.5-pro :: fallback | gemini-2.5-flash-lite :: economy | gemini-2.5-flash :: fallback
    validation: gemini-2.5-flash · aliases: gemini-flash-latest→gemini-3.5-flash, gemini-pro-latest→gemini-3.1-pro-preview, …

deepseek (api.deepseek.com)
    deepseek-v4-pro :: default | deepseek-v4-flash :: fast

qwen-intl / qwen-us / qwen-cn (dashscope[-intl|-us].aliyuncs.com)
    qwen3.7-plus :: default | qwen3.7-max :: advanced | qwen3.6-flash :: fast | qwen3-coder-plus :: coding

zhipu (open.bigmodel.cn)
    glm-5.1 :: default | glm-5 :: coding | glm-4.6 :: fallback

moonshot / moonshot-cn (api.moonshot.ai / .cn)
    kimi-k3 :: default | kimi-k2.7-code :: coding | kimi-k2.7-code-highspeed :: fast | kimi-k2.6 :: fallback
    aliases: kimi-k2→kimi-k2.6

kimi-code (api.kimi.com — membership/subscription endpoint, ids DIFFER from platform)
    k3 :: default | kimi-for-coding :: coding | kimi-for-coding-highspeed :: fast
    ⚠ KNOWN STALE — verify against current Kimi Code membership docs and what
    OpenCode/Kilo Code/Claude-Code-compatible integrations send to api.kimi.com today.

minimax (api.minimax.io)
    MiniMax-M2.7 :: default | MiniMax-M2.7-highspeed :: fast | MiniMax-M2 :: fallback

bytedance (ark.cn-beijing.volces.com)
    doubao-seed-2-0-pro-260215 :: default | doubao-seed-2-0-lite-260215 :: fast | doubao-seed-1-6-250615 :: fallback

01ai (api.lingyiwanwu.com)        yi-large  ⚠ placeholder — propose real lineup
groq (api.groq.com)               openai/gpt-oss-120b :: default | openai/gpt-oss-20b :: fast | llama-3.3-70b-versatile
cerebras (api.cerebras.ai)        llama3.3-70b  ⚠ placeholder — propose real lineup
mistral (api.mistral.ai)          mistral-medium-3-5 :: default | devstral-2512 :: coding | mistral-small-2603 :: fast
xai (api.x.ai)                    grok-4.3 :: default | grok-4.20-0309-reasoning :: reasoning | grok-code-fast-1 :: coding
cohere (api.cohere.ai)            command-a-plus-05-2026 :: default | command-a-reasoning-08-2025 | command-a-03-2025
openrouter (openrouter.ai)        openai/gpt-5.5 | anthropic/claude-opus-4.8 | google/gemini-3.5-flash | qwen/qwen3.7-plus | deepseek/deepseek-v4-pro
perplexity (api.perplexity.ai)    sonar-pro | sonar  ⚠ thin — propose current lineup
together (api.together.xyz)       zai-org/GLM-5.1 | moonshotai/Kimi-K2.6 | deepseek-ai/DeepSeek-V4-Pro | Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 | openai/gpt-oss-120b
fireworks (api.fireworks.ai)      accounts/fireworks/models/{deepseek-v4-pro, kimi-k2p6, glm-5p1, qwen2p5-coder-32b-instruct}
```

## What to deliver, per provider

Return one YAML block per preset key, exactly this schema:

```yaml
preset: kimi-code
checked: 2026-07-19
sources:
  - https://…            # every URL you actually verified against
verdict: stale | current | partially-stale
remove:                  # ids that are deprecated/renamed/no longer advertised
  - old-id               # + one-line reason (deprecated on <date>, renamed to X, …)
suggested_models:        # the NEW curated lineup, best-first, 3-6 models
  - id: exact-api-model-id
    role: default        # default | advanced | fast | economy | coding | reasoning | agentic | fallback
    desc: "flagship for agentic coding (1M context)"   # ≤60 chars, no marketing fluff
    context: 1000000
    reasoning: always-on | optional | none
add_aliases:             # old-id → new-id mappings so saved configs keep working
  old-id: new-id
validation_model: cheapest-reliable-id   # for API-key validation pings
notes: "anything a maintainer must know (endpoint quirks, tier gating, region)"
```

## Rules

1. **Exact ids only.** Copy model ids character-for-character from official docs or a live API listing (`/models` endpoints, models.dev, LiteLLM json). Watch dots vs dashes (`k2.6` vs `k2p6` on Fireworks) and vendor prefixes on gateways. Never invent or "correct" an id from memory.
2. **Every id needs a source URL** you actually opened. If two sources disagree, say so and pick the official docs.
3. **Deprecations matter as much as additions.** For each removed id, find the announced replacement and add it to `add_aliases` — users have these ids saved in configs.
4. **Match what coding agents actually default to.** If OpenCode/Kilo Code default a provider to a specific model, that is strong evidence for our `default` role. Prefer GA over preview unless the community default IS the preview.
5. **Descriptions ≤60 chars**, lower-case start, no superlatives — style: `"fast - lowest-latency option"`, `"coding - 256K context"`.
6. **Kimi Code is the priority case** (users currently get old models). The membership endpoint api.kimi.com accepts different ids than platform.moonshot.ai — verify both presets independently, including what tier gates apply.
7. If a provider now exposes a **live model-list endpoint** suitable for runtime fetching, note it in `notes` — we may switch that preset from a static list to live discovery.
8. Finish with a **summary table**: preset | verdict | #added | #removed | biggest change.
