# Sylliptor provider catalog refresh — 2026-07-19

Verification of every model id in `src/sylliptor_agent_cli/profile_presets.py` against live
registries and official provider docs.

**Method.** Fetched three live registries first (models.dev `api.json`, the full
`openrouter.ai/api/v1/models` id list, LiteLLM `model_prices_and_context_window.json`), then ran one
research agent per preset against official docs, then an adversarial verifier per preset whose only
job was to find fabricated or misspelled ids, then a correction pass. 18 of 20 presets required
correction after adversarial review. Official docs win every conflict; conflicts are recorded.

**Role normalisation.** Roles below use only Sylliptor's eight: `default | advanced | fast | economy
| coding | reasoning | agentic | fallback`. Where an agent proposed an out-of-set role (`heavy`,
`budget`, `escalation`), it has been mapped into the eight and the nuance moved to `notes`.

**Schema note.** Sylliptor has no structured role field — roles are the prefix of each
`suggested_model_descriptions` value before `" - "` (`profile_presets.py:80-99`). The `role:` key
below maps to that prefix. `context`/`reasoning` have no home in `ProfilePreset` today; they are
included because several fixes below depend on them.

---

## Summary table

| preset | verdict | +added | −removed | biggest change |
|---|---|---|---|---|
| openai + openai-responses | partially-stale | 4 | 2 | whole `gpt-5.6` generation shipped 2026-07-09; `gpt-5.5` default is a generation behind |
| anthropic (+compat/native) | **stale** | 4 | 2 | `claude-sonnet-4-6` default is now in Anthropic's *Legacy* table; `claude-sonnet-5` is newer **and** cheaper |
| gemini (+compat/native) | partially-stale | 1 | 3 | all three `gemini-2.5-*` shut down 2026-10-16; no model in the new lineup can disable thinking |
| deepseek | **current** | 0 | 0 | `deepseek-chat`/`deepseek-reasoner` die **2026-07-24** — aliases are time-critical |
| qwen-intl / -us / -cn | partially-stale | 4 | 0 | `qwen3-coder-next` exists; `qwen-us` has **no** coder models at all |
| zhipu | **stale** | 4 | 2 | `glm-5.2` (1M ctx) shipped 2026-06-13; `glm-5.1` default superseded |
| moonshot / moonshot-cn | partially-stale | 0 | 0 | ids correct, but reasoning contract is wrong — only `kimi-k2.6` accepts thinking-off |
| **kimi-code** | **current** | 0 | 0 | **ids were already right**; real bug is `k3` context — 1M is Allegretto+ only, 256K otherwise |
| minimax | **stale** | 4 | 1 | `MiniMax-M3` (1M ctx) shipped 2026-06-01; entire preset superseded |
| bytedance | partially-stale ⚠ | 3 | 0 | coding model `doubao-seed-2-0-code-preview-260215` missing — **low confidence, see notes** |
| 01ai | **deprecate-preset** | 0 | 1 | provider effectively unmaintained; `yi-large` unconfirmable |
| groq | partially-stale | 3 | 2 | both llama ids shut down **2026-08-16** |
| cerebras | **stale** | 3 | 1 | `llama3.3-70b` was never a valid id (real id `llama-3.3-70b`, deprecated 2026-02-16) |
| mistral | partially-stale | 4 | 1 | `devstral-2512` coding model retires **2026-07-31** |
| xai | **stale** | 4 | 1 | `grok-code-fast-1` retired 2026-05-15; `grok-4.5` shipped 2026-07-08 |
| cohere | **current** | 1 | 0 | base URL was right after all; `north-mini-code-1-0` is Chat-V2-only, cannot ship |
| openrouter | partially-stale | 4 | 2 | `gpt-5.6` tiers + `claude-sonnet-5` now available |
| perplexity | **stale** | 6 | 2 | sonar models **400 on tool definitions** — unusable for agent loops |
| together | partially-stale | 5 | 2 | both `zai-org/GLM-5.1` and the Qwen coder are already retired from serverless |
| fireworks | **stale** | 4 | 3 | `qwen2p5-coder-32b-instruct` is not serverless-capable at all |

Totals: **58 ids added, 25 removed.** Verdicts: 3 current, 7 stale, 9 partially-stale, 1 deprecate.

### Act-now deadlines

| date | what breaks |
|---|---|
| **2026-07-21** | cerebras `disable_reasoning` param unsupported → use `reasoning_effort:"none"` |
| **2026-07-24** | deepseek `deepseek-chat` / `deepseek-reasoner` discontinued (15:59 UTC) |
| **2026-07-31** | mistral `devstral-2512` + magistral family retire |
| **2026-08-15** | xai retired slugs fully shut down |
| **2026-08-16** | groq `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` shut down |
| **2026-08-17** | cerebras `zai-glm-4.7` deprecates |
| **2026-08-31** | moonshot `kimi-k2.5` + `moonshot-v1-*` end; anthropic Sonnet 5 intro pricing reverts |
| **2026-10-16** | gemini `gemini-2.5-pro` / `-flash` / `-flash-lite` shut down |

---

## kimi-code — the priority case

**The brief assumed this preset was stale. It is not — the three ids Sylliptor ships are exactly the
three the official docs list.** models.dev (which OpenCode consumes) carries a fourth id, `k2p7`,
that appears on **no** official Kimi page. The docs explicitly warn that sending anything other than
the listed Model IDs fails. Do not adopt `k2p7`.

The real defects are different, and both are live bugs:

1. **`k3` context is tier-dependent.** 1M is Allegretto+ only; Moderato members get 256K. Any
   hardcoded 1,048,576 over-promises headroom 4× for a paying member.
2. **Turning reasoning off silently swaps the model.** The docs state that disabling thinking routes
   the request to K2.6. On this preset, reasoning-off is not a latency knob — it means the user is no
   longer talking to the model they selected.

```yaml
preset: kimi-code
checked: 2026-07-19
sources:
  - https://www.kimi.com/code/docs/en/kimi-code/models
  - https://www.kimi.com/code/docs/en/third-party-tools/other-coding-agents.html
  - https://www.kimi.com/code/docs/en/
verdict: current
remove: []
suggested_models:
  - id: k3
    role: default
    desc: "default - 256K context, 1M on Allegretto+"
    context: 262144
    context_max_allegretto: 1048576
    reasoning: optional          # effort low|high|max; off => silently routed to K2.6
  - id: kimi-for-coding
    role: coding
    desc: "coding - 256K context, all membership tiers"
    context: 262144
    reasoning: optional
  - id: kimi-for-coding-highspeed
    role: fast
    desc: "fast - 256K context, Allegretto tier or above"
    context: 262144
    reasoning: optional
add_aliases:                     # PRESET-SCOPED ONLY — see notes
  kimi-k3: k3
  kimi-k2.7-code: kimi-for-coding
  kimi-k2.7-code-highspeed: kimi-for-coding-highspeed
validation_model: kimi-for-coding
notes: "Membership/subscription endpoint, NOT platform.moonshot.ai. Dual protocol, both confirmed
  verbatim in the docs index: OpenAI chat-completions at https://api.kimi.com/coding/v1 (Sylliptor's
  current base_url — correct, keep it) and Anthropic Messages at https://api.kimi.com/coding/. Both
  accept the same three ids. Auth is the membership key; a platform.moonshot.ai MOONSHOT_API_KEY does
  not work here and vice versa. TIER GATING: kimi-for-coding = all members; k3 = Moderato+ (256K) and
  Allegretto+ (1M); kimi-for-coding-highspeed = Allegretto+ only — a Moderato key
  gets errors on the fast slot, so degrade rather than hard-fail. REASONING: all three default
  thinking ON. Docs state disabling thinking routes to K2.6, i.e. reasoning-off is a model
  substitution — Sylliptor should surface that, not treat it as a speed toggle. k3 additionally takes
  reasoning_effort low|high|max. Docs warn to send the Model ID and never the display name ('Kimi K3',
  'K2.7 Code' will fail). The k3[1m] bracket form is real but is documented ONLY as a Claude Code
  ANTHROPIC_MODEL env value to force the 1M window — never send it as a request-body model id.
  ALIAS SCOPING IS LOAD-BEARING: kimi-k3 and kimi-k2.7-code are live, valid, DIFFERENT ids on
  platform.moonshot.ai. These aliases are cross-endpoint remaps legal only inside this preset.
  Sylliptor's alias table is per-preset (profile_presets.py model_aliases), so this is safe as-is.
  CONFLICT: models.dev lists `k2p7` under provider kimi-for-coding; no official page documents it.
  Official docs win — excluded. No /models endpoint is documented, so validation costs a real billed
  call against a metered membership key: use a single-token probe, not a startup ping."
```

---

## Core presets

```yaml
preset: openai + openai-responses
checked: 2026-07-19
sources:
  - https://developers.openai.com/api/docs/models
  - https://developers.openai.com/api/docs/deprecations
  - https://developers.openai.com/api/docs/guides/latest-model
verdict: partially-stale
remove:
  - gpt-5.5    # not deprecated; superseded by gpt-5.6-terra at the same $2.50/$15 tier
  - gpt-5.4    # not deprecated; two generations back at the same price tier
suggested_models:
  - id: gpt-5.6-terra
    role: default
    desc: "default - balanced 5.6 tier, 1.05M context"
    context: 1050000
    reasoning: optional
  - id: gpt-5.6-sol
    role: advanced
    desc: "advanced - flagship 5.6 tier, 1.05M context"
    context: 1050000
    reasoning: optional
  - id: gpt-5.6-luna
    role: fast
    desc: "fast - low-cost 5.6 tier, full 1.05M context"
    context: 1050000
    reasoning: optional
  - id: gpt-5.3-codex
    role: coding
    desc: "coding - agentic codex model, 400K context"
    context: 400000
    reasoning: optional
  - id: gpt-5.4-mini
    role: fallback
    desc: "fallback - cheap tier for subagents, 400K"
    context: 400000
    reasoning: optional
  - id: gpt-5.4-nano
    role: economy
    desc: "economy - cheapest live id, 400K context"
    context: 400000
    reasoning: optional
add_aliases:
  gpt-5.6: gpt-5.6-sol
  gpt-5-nano: gpt-5.4-nano
  gpt-5-codex: gpt-5.5
  gpt-5.1-codex: gpt-5.5
  gpt-5.1-codex-max: gpt-5.5
  gpt-5.2-codex: gpt-5.5
  gpt-5.1-codex-mini: gpt-5.4-mini
  gpt-5-chat-latest: gpt-5.5
  gpt-5.1-chat-latest: gpt-5.5
validation_model: gpt-5.4-nano
notes: "Sol/Terra/Luna are durable capability tiers WITHIN generation 5.6, not previews — all
  1,050,000 context / 128,000 max output / cutoff 2026-02-16, all with a reasoning-effort knob and
  full tool support. Prices: luna $1/$6, terra $2.50/$15, sol $5/$30. There is NO gpt-5.6-pro slug:
  the latest-model guide says keep your 5.6 model and set reasoning.mode=pro. gpt-5.5-pro ($30/$180)
  remains the only slug-level 'pro' and must never be auto-selected — it is dropped from the lineup
  for that reason. Every alias above except gpt-5.6 and gpt-5-nano comes from OpenAI's deprecations
  page, which sets a 2026-07-23 shutdown (four days out) for the codex and chat-latest ids listed.
  gpt-5.3-codex is on no deprecation table and stays as the coding model. Removing gpt-5.5/gpt-5.4 is
  an editorial supersession call, not a rename — both remain callable if a user pins them, which is
  why they are alias TARGETS. Live discovery: GET https://api.openai.com/v1/models returns the
  account-visible list but carries no deprecation status."
```

```yaml
preset: anthropic (+ -compat/-native)
checked: 2026-07-19
sources:
  - https://platform.claude.com/docs/en/about-claude/models/overview
  - https://platform.claude.com/docs/en/about-claude/models/migration-guide
  - https://platform.claude.com/docs/en/about-claude/model-deprecations
verdict: stale
remove:
  - claude-sonnet-4-6            # moved to the official "Legacy models" table; sonnet-5 is newer AND cheaper
  - claude-3-5-haiku-20241022    # retired; the existing claude-3-5-haiku-latest alias points at a dead id
suggested_models:
  - id: claude-sonnet-5
    role: default
    desc: "default - 1M context, best speed/intelligence mix"
    context: 1000000
    reasoning: optional
  - id: claude-opus-4-8
    role: advanced
    desc: "advanced - complex agentic coding, 1M context"
    context: 1000000
    reasoning: optional
  - id: claude-fable-5
    role: reasoning
    desc: "reasoning - adaptive thinking always on, 1M ctx"
    context: 1000000
    reasoning: always-on
  - id: claude-haiku-4-5
    role: fast
    desc: "fast - 200K context, lowest cost tier"
    context: 200000
    reasoning: optional
  - id: claude-opus-4-7
    role: fallback
    desc: "fallback - previous-generation opus, 1M context"
    context: 1000000
    reasoning: optional
add_aliases:
  claude-sonnet-4: claude-sonnet-5
  claude-sonnet-4-5: claude-sonnet-5
  claude-sonnet-4-6: claude-sonnet-5
  claude-4-sonnet: claude-sonnet-5
  claude-3-5-haiku-latest: claude-haiku-4-5
  claude-3-5-haiku-20241022: claude-haiku-4-5
  claude-opus-4.8: claude-opus-4-8
  claude-opus-4.7: claude-opus-4-7
  claude-opus-4-1: claude-opus-4-8
  claude-opus-4-6: claude-opus-4-8
validation_model: claude-haiku-4-5
notes: "ROLE-SHAPE CALL: Anthropic's own docs say 'start with Claude Opus 4.8 for complex agentic
  coding', which argues for opus as default. This lineup keeps default=claude-sonnet-5 /
  advanced=claude-opus-4-8 because it preserves Sylliptor's existing role semantics (the old default
  was Sonnet-tier) and Sonnet 5 is $3/$15 vs Opus $5/$25. Flip if you want to follow vendor guidance
  literally. PRICING: Sonnet 5 is $3/$15 list, discounted to $2/$10 through 2026-08-31 — budget for
  the revert; after it, Sonnet 5 prices identically to the sonnet-4-6 it replaces. THINKING SPLIT:
  Opus 4.8 / Sonnet 5 / Fable 5 support *adaptive* thinking and NOT extended thinking; Haiku 4.5 is
  the inverse (extended yes, adaptive no) and is the only lineup model with no `effort` parameter.
  Fable 5 is adaptive-thinking always-on and $10/$50 — above Opus — hence `reasoning`, not `default`.
  ID FORMAT: from the 4.6 generation on, ids are DATELESS pinned snapshots; appending a date (e.g.
  claude-sonnet-4-6-20251114) 404s. Pre-4.6 models keep dated ids with a bare alias, so both
  claude-haiku-4-5 and claude-haiku-4-5-20251001 are valid. TOKENIZER: Sonnet 5 emits ~30% more
  tokens than Sonnet 4.6 for the same text — re-baseline context accounting with
  /v1/messages/count_tokens rather than scaling a stored figure; this moves measured cost even where
  per-token price is flat. DO NOT SHIP claude-mythos-5: it is real and shares Fable 5's specs, but is
  invitation-only under Project Glasswing for defensive-cyber work, with no self-serve signup. Live
  discovery: GET https://api.anthropic.com/v1/models returns max_input_tokens, max_tokens and a
  capabilities object per model — the best runtime source here. Support BOTH auth paths
  (x-api-key and Authorization: Bearer + anthropic-beta: oauth-2025-04-20); hardcoding x-api-key
  breaks OAuth profiles."
```

```yaml
preset: gemini (+ -compat/-native)
checked: 2026-07-19
sources:
  - https://ai.google.dev/gemini-api/docs/models
  - https://ai.google.dev/gemini-api/docs/deprecations
  - https://ai.google.dev/gemini-api/docs/pricing
verdict: partially-stale
remove:
  - gemini-2.5-pro          # shutdown 2026-10-16, replacement gemini-3.1-pro-preview
  - gemini-2.5-flash        # shutdown 2026-10-16, replacement gemini-3.5-flash
  - gemini-2.5-flash-lite   # shutdown 2026-10-16, replacement gemini-3.1-flash-lite
suggested_models:
  - id: gemini-3.5-flash
    role: default
    desc: "default - 1M context, agentic coding driver"
    context: 1048576
    reasoning: always-on      # effort minimal|low|medium|high, cannot be disabled
  - id: gemini-3.1-pro-preview
    role: advanced
    desc: "advanced - hardest tasks, no free tier"
    context: 1048576
    reasoning: always-on      # effort low|medium|high — no minimal
  - id: gemini-3.1-flash-lite
    role: fast
    desc: "fast - lowest-cost tier, 1M context"
    context: 1048576
    reasoning: always-on
  - id: gemini-3-flash-preview
    role: fallback
    desc: "fallback - mid-price flash, 1M context"
    context: 1048576
    reasoning: always-on
add_aliases:
  gemini-2.5-pro: gemini-3.1-pro-preview
  gemini-2.5-flash: gemini-3.5-flash
  gemini-2.5-flash-lite: gemini-3.1-flash-lite
  gemini-3-pro-preview: gemini-3.1-pro-preview
  gemini-3.1-flash-lite-preview: gemini-3.1-flash-lite
  gemini-2.0-flash: gemini-3.5-flash
  gemini-2.0-flash-lite: gemini-3.1-flash-lite
  gemini-flash-latest: gemini-3.5-flash
  gemini-pro-latest: gemini-3.1-pro-preview
validation_model: gemini-3.1-flash-lite
notes: "ACTION FOR SYLLIPTOR: no model in this lineup can turn thinking OFF. The 3.x family exposes
  effort {minimal,low,medium,high} only ({low,medium,high} on Pro — no minimal); the outgoing
  gemini-2.5-flash/-flash-lite DID expose a toggle, so this is a real behavioural regression across
  the 2.5→3.x jump. Two concrete consequences: (1) the `doctor --live` reasoning-off probe cannot
  pass on this preset and should assert effort=minimal rather than off; (2) economy and validation
  calls must pin effort=minimal (low on Pro) or they are billed thinking tokens. NO GA 3.x PRO EXISTS
  as of 2026-07-19 — gemini-3.1-pro-preview is the only 3.x Pro id on the official models page, so
  Sylliptor's use of a -preview id for `advanced` is correct, not stale. It has no free tier and is
  the only lineup entry that hard-fails on a free-tier key. gemini-3-flash-preview is itself
  deprecation-listed (replacement gemini-3.5-flash) but carries NO shutdown date; retained as
  fallback for its distinct $0.5/$3 price point — revisit the moment a date appears. The -latest
  pointers (gemini-flash-latest, gemini-flash-lite-latest, gemini-pro-latest) resolve correctly today
  and are aliased above, but the official models page documents the -latest pattern only in prose and
  lists no -latest model strings, so they are alias targets rather than lineup entries. Note
  gemini-2.0-flash/-flash-lite already shut down 2026-06-01 — those two aliases point at dead ids and
  are remap-only. Live discovery: GET
  https://generativelanguage.googleapis.com/v1beta/models?key=API_KEY (not probed in this pass)."
```

```yaml
preset: deepseek
checked: 2026-07-19
sources:
  - https://api-docs.deepseek.com/quick_start/pricing
  - https://api-docs.deepseek.com/api/list-models
  - https://api-docs.deepseek.com/guides/coding_agents/
verdict: current
remove: []
suggested_models:
  - id: deepseek-v4-pro
    role: default
    desc: "default - flagship coding model, 1M context"
    context: 1000000
    reasoning: optional
  - id: deepseek-v4-flash
    role: fast
    desc: "fast - cheap high-volume work, 1M context"
    context: 1000000
    reasoning: optional
add_aliases:
  deepseek-chat: deepseek-v4-flash
  deepseek-reasoner: deepseek-v4-flash
validation_model: deepseek-v4-flash
notes: "URGENT — deepseek-chat and deepseek-reasoner are discontinued 2026-07-24 15:59 UTC, five days
  from this check. Until then they route to deepseek-v4-flash (non-thinking and thinking mode
  respectively); after, saved configs pinning them break. The aliases above are time-critical, not
  cosmetic. This ANSWERS the open question in the brief: the stable-alias convention is ending —
  deepseek-v4-pro/-flash are the real callable ids, and GET https://api.deepseek.com/models already
  returns exactly those two and no longer lists the legacy aliases. That endpoint is clean and
  suitable for runtime discovery. Reasoning is an explicit toggle: thinking={'type':'enabled'|
  'disabled'} plus optional reasoning_effort. Two surfaces: OpenAI ChatCompletions at
  https://api.deepseek.com and an Anthropic-compatible path at https://api.deepseek.com/anthropic
  (the one DeepSeek documents for Claude Code / OpenCode). Max output 384000 on both. Cache-hit input
  is ~50x cheaper than cache-miss, so prompt-prefix stability matters unusually much here. Only two
  first-party ids exist, so this preset is legitimately below the 3-model floor — nothing was
  invented to pad it."
```

```yaml
preset: qwen-intl / qwen-us / qwen-cn
checked: 2026-07-19
sources:
  - https://www.alibabacloud.com/help/en/model-studio/models
  - https://www.alibabacloud.com/help/en/model-studio/model-pricing
  - https://www.alibabacloud.com/help/en/model-studio/qwen-coder
verdict: partially-stale
remove: []
suggested_models:
  - id: qwen3.7-plus
    role: default
    desc: "default - 1M context, balanced cost"
    context: 1000000
    reasoning: optional
  - id: qwen3.7-max
    role: advanced
    desc: "advanced - flagship, 1M context"
    context: 1000000
    reasoning: optional
  - id: qwen3-coder-plus
    role: coding
    desc: "coding - 1M context, long-repo work"
    context: 1000000
    reasoning: none
  - id: qwen3-coder-next
    role: agentic
    desc: "agentic - newest coder, 256K context"
    context: 262144
    reasoning: none
  - id: qwen3.6-flash
    role: fast
    desc: "fast - lower-latency, 1M context"
    context: 1000000
    reasoning: optional
  - id: qwen-flash
    role: economy
    desc: "economy - cheapest 1M-context option"
    context: 1000000
    reasoning: optional
add_aliases: {}
validation_model: qwen-flash
notes: "REGION SPLIT MATTERS: intl https://dashscope-intl.aliyuncs.com/compatible-mode/v1, US
  (Virginia) https://dashscope-us.aliyuncs.com/compatible-mode/v1, China
  https://dashscope.aliyuncs.com/compatible-mode/v1. Same DASHSCOPE_API_KEY var but keys are
  region-bound — an intl key will not authenticate against .cn and vice versa. CRITICAL FOR qwen-us:
  the model-pricing page lists qwen3-coder-plus / -flash / -next for Singapore, China and Germany
  ONLY, NOT for US (Virginia). qwen-us must therefore ship NO coding slot and fall back to
  qwen3.7-plus for code work — shipping a coder id in the qwen-us preset hands users a model their
  endpoint cannot serve. DO NOT alias coder-plus→coder-next: coder-plus is priced in tiers up to
  256K<Token<=1M while coder-next tops out at 128K<Token<=256K, so the newest coder is NOT a
  drop-in replacement for long-context repo work — that is why both are listed. models.dev files
  coder-next only under provider alibaba-coding-plan (coding-intl.dashscope.aliyuncs.com); the
  official pricing page shows plain pay-as-you-go tiers for it, so treat that as a models.dev
  coverage gap. Reasoning on the 3.x line is the enable_thinking toggle — optional, not always-on.
  Alibaba now also documents workspace-scoped regional hosts
  (https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1); the dashscope-* domains stay
  fully functional per the docs. Docs de-emphasise the coder family and advise preferring the latest
  general-purpose models, which is why qwen3.7-plus is default rather than a coder. No GET /models is
  documented for compatible-mode; /compatible-mode/v1/models may work by OpenAI convention but is
  UNVERIFIED — do not wire runtime discovery on that assumption."
```

```yaml
preset: zhipu
checked: 2026-07-19
sources:
  - https://docs.bigmodel.cn/cn/guide/start/model-overview
  - https://docs.z.ai/guides/overview/pricing
  - https://docs.z.ai/guides/llm/glm-5.2
verdict: stale
remove:
  - glm-5     # not deprecated; glm-5.1/glm-5.2 cover the tier
  - glm-4.6   # not deprecated; same price and 200K class as glm-4.7
suggested_models:
  - id: glm-5.2
    role: default
    desc: "default - 1M context, agentic coding"
    context: 1000000
    reasoning: optional
  - id: glm-5.1
    role: advanced
    desc: "advanced - previous flagship, 200K context"
    context: 200000
    reasoning: optional
  - id: glm-5-turbo
    role: coding
    desc: "coding - 200K context, cheaper than glm-5.1"
    context: 200000
    reasoning: optional
  - id: glm-4.7
    role: fallback
    desc: "fallback - cheap 200K context"
    context: 200000
    reasoning: optional
  - id: glm-4.7-flashx
    role: fast
    desc: "fast - 200K context, no free-tier rate caps"
    context: 200000
    reasoning: optional
  - id: glm-4.7-flash
    role: economy
    desc: "economy - free tier, 200K context, rate limited"
    context: 200000
    reasoning: optional
add_aliases: {}
validation_model: glm-4.7-flash
notes: "NO ALIASES ON PURPOSE: no vendor renames exist. glm-5, glm-4.6, glm-4.5 and glm-4.5-flash all
  remain individually listed AND separately priced on both official pages, so aliasing them would
  silently change what the user is billed (glm-5 is $1.00/M input vs glm-5.1 at $1.40/M). The two
  removals are shortlist curation only — both still work if a user pins them. Two surfaces share the
  same lowercase ids: open.bigmodel.cn/api/paas/v4 (China, ZHIPU_API_KEY) and api.z.ai/api/paas/v4
  (international). A separate GLM Coding Plan subscription surface exists at
  open.bigmodel.cn/api/coding/paas/v4 and api.z.ai/api/coding/paas/v4 — narrower roster, billed by
  plan not per token, so pay-as-you-go pricing does not apply there. Context taken uniformly from the
  CN model-overview page (200K); models.dev's 204800 for glm-4.7 is a rounding difference, not a
  discrepancy. PRICING TRAP: third-party resellers reprice these — OpenRouter lists z-ai/glm-5.1
  BELOW z-ai/glm-5-turbo, inverting the first-party order — so route cost decisions off the vendor
  page, not a gateway. Reasoning is opt-in via thinking={type:enabled}; glm-5.2 additionally accepts
  reasoning_effort. For vision, glm-5v-turbo (200K) is current and glm-4.6v (128K) the cheaper
  fallback; neither belongs in this text lineup. No documented GET /models on either surface — this
  list must stay static."
```

```yaml
preset: moonshot / moonshot-cn
checked: 2026-07-19
sources:
  - https://platform.kimi.ai/docs/models
  - https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
  - https://platform.kimi.ai/docs/api/list-models.md
verdict: partially-stale
remove: []
suggested_models:
  - id: kimi-k2.7-code
    role: default
    desc: "default - 256K context, long-horizon agentic coding"
    context: 262144
    reasoning: always-on     # thinking.type accepts "enabled" only
  - id: kimi-k3
    role: advanced
    desc: "advanced - 1M context, always-thinking at max effort"
    context: 1048576
    reasoning: always-on     # reasoning_effort accepts "max" only
  - id: kimi-k2.7-code-highspeed
    role: fast
    desc: "fast - ~180 tok/s coding variant, 256K context"
    context: 262144
    reasoning: always-on
  - id: kimi-k2.6
    role: fallback
    desc: "fallback - 256K context, thinking toggleable"
    context: 262144
    reasoning: optional
add_aliases:
  kimi-k2: kimi-k2.6
  kimi-k2.5: kimi-k2.6
  kimi-k2-thinking: kimi-k2.6
  kimi-k2-thinking-turbo: kimi-k2.6
  kimi-k2-0905-preview: kimi-k2.6
  kimi-k2-0711-preview: kimi-k2.6
  kimi-k2-turbo-preview: kimi-k2.7-code-highspeed
  kimi-latest: kimi-k2.6
  kimi-thinking-preview: kimi-k2.6
  moonshot-v1-8k: kimi-k2.6
  moonshot-v1-32k: kimi-k2.6
  moonshot-v1-128k: kimi-k2.6
  moonshot-v1-auto: kimi-k2.6
validation_model: kimi-k2.6
notes: "IDS WERE ALREADY CORRECT — the staleness here is the REASONING CONTRACT, and getting it wrong
  produces API errors, not degraded output. kimi-k3 always has thinking enabled and its effort control
  accepts only `max`; the quickstart says explicitly NOT to send the K2.x `thinking` parameter to it.
  Both kimi-k2.7-code models accept thinking.type='enabled' ONLY and ERROR on 'disabled'. kimi-k2.6 is
  the sole toggleable model. Rule: Sylliptor must not emit a disable-reasoning flag to anything on
  this preset except kimi-k2.6. (An earlier draft marked k3 toggleable on the strength of a models.dev
  capability row; official docs contradict it and win.) DEFAULT CHOICE: kimi-k2.7-code over kimi-k3
  deliberately — k3 is always-thinking at pinned max effort and ~3x the input price ($3/M vs $0.95/M),
  so routing routine agentic turns through it buys latency and cost for no reliability gain. Escalate
  to k3 when the task needs the 1M window. On THIS endpoint k3's 1M is flat and NOT plan-gated (that
  gating belongs to the kimi-code preset). DOCS MOVED: platform.moonshot.ai/docs/* now 301s to
  platform.kimi.ai, but platform.moonshot.cn/docs/* 301s to platform.kimi.COM — a different host.
  API base URLs are unchanged (api.moonshot.ai/v1, api.moonshot.cn/v1) and both regions expose an
  identical model set, so one lineup serves both presets. SUNSET WATCH: kimi-k2.5 and the whole
  moonshot-v1-* family end 2026-08-31 and are already closed to new users; the kimi-k2-thinking family
  was deprecated 2026-05-25. Live discovery: GET https://api.moonshot.ai/v1/models is real and
  OpenAI-compatible — but returns ids with no capability metadata, so it can self-heal id drift and
  CANNOT repair a wrong reasoning flag. Do not confuse with the kimi-code preset: api.kimi.com/coding
  is a different endpoint with different ids."
```

---

## Secondary presets

```yaml
preset: minimax
checked: 2026-07-19
sources:
  - https://platform.minimax.io/docs/guides/models-intro
  - https://platform.minimax.io/docs/guides/text-generation
  - https://platform.minimax.io/docs/api-reference/models/openai/list-models.md
verdict: stale
remove:
  - MiniMax-M2   # 2025-10 generation, four releases behind (M2.1, M2.5, M2.7, M3)
suggested_models:
  - id: MiniMax-M3
    role: default
    desc: "default - 1M context, multimodal agentic coding"
    context: 1000000
    reasoning: none          # no documented toggle
  - id: MiniMax-M2.7
    role: coding
    desc: "coding - 200K context, prior flagship"
    context: 204800
    reasoning: none
  - id: MiniMax-M2.7-highspeed
    role: fast
    desc: "fast - same weights as M2.7, latency-tuned"
    context: 204800
    reasoning: none
  - id: MiniMax-M2.5
    role: fallback
    desc: "fallback - stable prior generation"
    context: 204800
    reasoning: none
add_aliases:
  MiniMax-M2: MiniMax-M2.7
validation_model: MiniMax-M2.5
notes: "Verdict is stale rather than partially-stale because the preset's fallback id is its oldest
  model and NOTHING in the shipped lineup is current. Two protocols on one host: OpenAI-compatible at
  https://api.minimax.io/v1/chat/completions and Anthropic-compatible at
  https://api.minimax.io/anthropic/v1/messages. The OpenAI path is the right fit for Sylliptor's
  openai_compat client — note models.dev lists ONLY the /anthropic base for this provider, which is
  what made the endpoint look Anthropic-only. REASONING: no thinking/reasoning toggle is documented
  for ANY MiniMax model. The Anthropic path supports thinking blocks and interleaved thinking but
  exposes no on/off switch, no reasoning_effort and no enable_thinking. Sylliptor should send no
  reasoning-control parameter here and treat emitted thinking as provider-determined. CONTEXT TRAP:
  M3 is documented as up to 1M with a guaranteed minimum of 512K, and input above 512K bills at a
  higher long-context rate — 1M is not uniformly priced. Coding agents commonly cap context at 200K
  by default, which silently truncates M3; set the window explicitly. The -highspeed variants trade
  price for latency on identical weights, not quality. NAMING AMBIGUITY: third-party registries serve
  the M2.1-era fast variant as MiniMax-M2.1-lightning while MiniMax's own table says -highspeed;
  -highspeed is correct on first-party. Region split: api.minimax.io intl, api.minimaxi.com CN, same
  id set. models.dev is deliberately NOT cited for this preset — its minimax entry lists only
  minimax-text-latest / minimax-vision-latest at 245,760 ctx, describing neither M2.x nor M3. Live
  discovery: GET https://api.minimax.io/v1/models (Bearer JWT) — the documented sample response
  contains M3, M2.7 and M2.5, id-confirming three of the four."
```

```yaml
preset: bytedance
checked: 2026-07-19
confidence: LOW — see notes before shipping
sources:
  - https://models.dev/api.json
  - https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
  - https://docs.openclaw.ai/providers/volcengine
verdict: partially-stale
remove: []
suggested_models:
  - id: doubao-seed-2-0-pro-260215
    role: default
    desc: "default - flagship seed 2.0, agentic tasks"
    context: 256000
    reasoning: optional
  - id: doubao-seed-2-0-code-preview-260215
    role: coding
    desc: "coding - 256K context, preview snapshot"
    context: 256000
    reasoning: optional
  - id: doubao-seed-2-0-lite-260215
    role: fast
    desc: "fast - balanced quality and latency"
    context: 256000
    reasoning: optional
  - id: doubao-seed-2-0-mini-260215
    role: economy
    desc: "economy - cheapest seed 2.0, high concurrency"
    context: 256000
    reasoning: optional
add_aliases: {}
validation_model: doubao-seed-2-0-mini-260215
notes: "⚠ LOWEST-CONFIDENCE PRESET IN THIS REFRESH — do not ship without a live-key probe. The
  official Volcengine Ark docs render client-side and could not be read by any fetch in this pass, so
  NOTHING here is confirmed against first-party documentation. Every id above appears verbatim in the
  models.dev registry dump and/or LiteLLM (which stores them volcengine/-prefixed), but registry
  presence is not proof that bare ids are callable on Ark. The base URL
  https://ark.cn-beijing.volces.com/api/v3 (OpenAI-compatible) is likewise unconfirmed here — it is
  Sylliptor's existing value and is left as-is rather than changed on weak evidence. WHAT IS NEW AND
  WORTH ACTING ON: doubao-seed-2-0-code-preview-260215 is a dedicated coding model the current preset
  lacks entirely — that is the one high-value addition. I deliberately chose the -260215 snapshots for
  lite/mini over the -260428 snapshots that a single registry (aihubmix) reports, because -260215 is
  attested by two independent registries; if Ark accepts -260428, prefer those as newer. A GA
  replacement for the code preview probably already exists (zenmux lists volcengine/doubao-seed-2.0-code
  at 2026-03-20) but no dated Ark-form id could be confirmed, so no suffix was guessed. Sylliptor's
  current doubao-seed-1-6-250615 fallback could not be confirmed live either; doubao-seed-1-8-251215
  looks like the successor but registries disagree on its context (128000 / 224000 / 256000) so it is
  not proposed. Ark historically required an endpoint id (ep-xxxx) rather than a model name — whether
  bare names are now callable is UNVERIFIED and is the single most important thing to test. No
  GET /models listing endpoint was found. Do not mix in BytePlus international ids
  (ark.ap-southeast.bytepluses.com uses prefix-free forms like seed-2-0-lite-260228) and never
  transplant a date suffix from one shape onto the other."
```

```yaml
preset: 01ai
checked: 2026-07-19
confidence: low
sources:
  - https://platform.01.ai/                          # intl endpoint: API services temporarily unavailable
  - https://help.aliyun.com/zh/model-studio/yi-api   # Aliyun's hosting of Yi, not the direct endpoint
  - https://www.oschina.net/news/280668
verdict: deprecate-preset
remove: []          # nothing is provably dead; yi-large is UNCONFIRMED, which is different
suggested_models: []
validation_model: yi-lightning
notes: "RECOMMENDATION: deprecate this preset, pending one authenticated GET
  https://api.lingyiwanwu.com/v1/models probe. Grounds that survived adversarial review: (1) the
  shipped id `yi-large` (profile_presets.py:803) cannot be confirmed live on api.lingyiwanwu.com by
  any reachable source; (2) the only id independently corroborated as currently served is
  yi-lightning at 16,384 context — far too small for Sylliptor's system prompt plus tool loop; (3) the
  international endpoint platform.01.ai announces API services temporarily unavailable and redirects
  to the CN platform, and pricing is CNY-denominated with a China-oriented recharge flow; (4) 01.AI is
  not shipped as a first-class provider by OpenCode, Cline or Kilo Code. The route
  GET /v1/models is live (returns HTTP 400 'Required request header Authorization is not present'
  unauthenticated) so ONE keyed call settles this either way — that call would also resolve
  yi-medium-200k (a 200K-window id named in launch coverage, the only member of this family that
  could carry an agent loop) and yi-large-fc (the documented function-calling variant). Claims from an
  earlier draft explicitly WITHDRAWN as unsourced, do not reinstate: 'yi-lightning is a router
  forwarding to DeepSeek-V3', 'yi-large is absent from the official billing table' (Aliyun still
  tabulates yi-large at 32K), and 'no tool-calling support'. Because no id here is confirmed callable,
  suggested_models is intentionally EMPTY rather than populated with guesses — a fabricated lineup
  would be worse than an honest deprecation."
```

```yaml
preset: groq
checked: 2026-07-19
sources:
  - https://console.groq.com/docs/models
  - https://console.groq.com/docs/deprecations
  - https://console.groq.com/docs/coding-with-groq/opencode
verdict: partially-stale
remove:
  - llama-3.3-70b-versatile   # deprecation table: shutdown 2026-08-16
  - llama-3.1-8b-instant      # deprecation table: shutdown 2026-08-16
suggested_models:
  - id: openai/gpt-oss-120b
    role: default
    desc: "default - 131K context, adjustable reasoning"
    context: 131072
    reasoning: optional
  - id: qwen/qwen3.6-27b
    role: coding
    desc: "coding - thinking modes and vision, preview tier"
    context: 131072
    reasoning: optional
  - id: openai/gpt-oss-20b
    role: fast
    desc: "fast - cheapest non-deprecated production id"
    context: 131072
    reasoning: optional
  - id: groq/compound
    role: agentic
    desc: "agentic - server-side web search and code exec"
    context: 131072
    reasoning: none
add_aliases:
  llama-3.3-70b-versatile: openai/gpt-oss-120b
  llama-3.1-8b-instant: openai/gpt-oss-20b
  qwen/qwen3-32b: openai/gpt-oss-120b
  meta-llama/llama-4-scout-17b-16e-instruct: openai/gpt-oss-120b
  meta-llama/llama-4-maverick-17b-128e-instruct: openai/gpt-oss-120b
  moonshotai/kimi-k2-instruct: openai/gpt-oss-120b
  moonshotai/kimi-k2-instruct-0905: openai/gpt-oss-120b
validation_model: openai/gpt-oss-20b
notes: "Base URL is https://api.groq.com/openai/v1, NOT the bare host. Groq splits its catalog three
  ways and the distinction is load-bearing: Production (stable), Production Systems (groq/compound,
  groq/compound-mini — these run server-side built-in tools and do NOT accept client tool_call, so
  never route normal agent tool loops to them), and Preview (may be pulled with little notice). The
  only Preview entry in this lineup is qwen/qwen3.6-27b; give it an explicit fallback to
  openai/gpt-oss-120b so a pulled id degrades instead of breaking the coding role. Both llama ids shut
  down 2026-08-16 — migrate now. DELIBERATELY EXCLUDED: minimaxai/minimax-m2.7 is on Groq but is
  Enterprise-gated (its model page says contact sales), so normal keys get 403/404 — it must not
  occupy a selectable slot; it is also always-on interleaved thinking with no off switch. Also
  excluded: openai/gpt-oss-safeguard-20b is a policy-classification model, not a coding model.
  gpt-oss models support prompt caching at 50% off input. Live discovery: GET
  https://api.groq.com/openai/v1/models returns the ACCOUNT-VISIBLE list — unusually valuable here
  because Groq's catalog churns fast and Preview/gated entries differ per account. This is the
  strongest live-discovery candidate of any preset in this refresh."
```

```yaml
preset: cerebras
checked: 2026-07-19
sources:
  - https://inference-docs.cerebras.ai/models/overview
  - https://inference-docs.cerebras.ai/support/deprecation
  - https://inference-docs.cerebras.ai/capabilities/reasoning
verdict: stale
remove:
  - llama3.3-70b   # never a valid Cerebras id (real id was llama-3.3-70b) AND that model deprecated 2026-02-16
suggested_models:
  - id: gpt-oss-120b
    role: default
    desc: "default - only GA public model, ~3000 tok/s"
    context: 65536              # FREE tier; 131072 on paid
    reasoning: always-on        # effort low|medium|high — no "none"
  - id: zai-glm-4.7
    role: coding
    desc: "coding - strongest here, deprecates 2026-08-17"
    context: 65536              # 131072 on paid
    reasoning: optional
  - id: gemma-4-31b
    role: fallback
    desc: "fallback - only image-input model, preview tier"
    context: 65536              # 131072 on paid
    reasoning: optional
add_aliases:
  llama3.3-70b: gpt-oss-120b
  llama-3.3-70b: gpt-oss-120b
  llama3.1-70b: gpt-oss-120b
  llama3.1-8b: gpt-oss-120b
  qwen-3-32b: gpt-oss-120b
  qwen-3-coder-480b: zai-glm-4.7
  zai-glm-4.6: zai-glm-4.7
  deepseek-r1-distill-llama-70b: gpt-oss-120b
validation_model: gpt-oss-120b
notes: "RESOLVES THE BRIEF'S CONFLICT: models.dev lists no llama for Cerebras while LiteLLM lists
  cerebras/llama-3.3-70b. Official docs settle it — the llama family is GONE from public endpoints
  (deprecated 2026-02-16) and survives only behind Dedicated Endpoints with reserved capacity and
  custom pricing. So Sylliptor's `llama3.3-70b` was doubly wrong: wrong spelling (no dashes) of an id
  that is itself no longer publicly callable. A key without a dedicated endpoint 404s on every
  Llama/Qwen/DeepSeek/Kimi id. CONTEXT AND MAX OUTPUT ARE TIER-DEPENDENT, stated verbatim per model
  page: free tier 64–65K context / 32K output, paid 131K / 40K (zai-glm-4.7 is 40K output on both).
  The `context` values above carry the conservative FREE-tier floor — do not collapse them to 131072
  or free-tier keys get handed headroom they do not have. REASONING IS NOT UNIFORM: gpt-oss-120b
  accepts reasoning_effort low|medium(default)|high with NO 'none', so reasoning cannot be disabled on
  the default model; only zai-glm-4.7 and gemma-4-31b accept 'none' (and it is gemma's default).
  DEADLINES: zai-glm-4.7 deprecates 2026-08-17 — do not make it default, and re-point the coding role
  plus the qwen-3-coder-480b / zai-glm-4.6 aliases before then; its `disable_reasoning` parameter
  loses support 2026-07-21 (two days out), replacement reasoning_effort='none'. Both zai-glm-4.7 and
  gemma-4-31b are Preview/evaluation-only and may be discontinued without notice, so gpt-oss-120b is
  the only genuinely safe default. Live discovery: GET https://api.cerebras.ai/v1/models returns only
  ids the key can actually reach — strongly recommended here because the public catalog is now tiny
  and tier-dependent."
```

```yaml
preset: mistral
checked: 2026-07-19
sources:
  - https://docs.mistral.ai/getting-started/models/models_overview/
  - https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04
  - https://docs.mistral.ai/models/model-cards/codestral-25-08
verdict: partially-stale
remove:
  - devstral-2512   # deprecated 2026-05-22, retires 2026-07-31; replacement "Mistral Medium 3.5"
suggested_models:
  - id: mistral-medium-2604
    role: default
    desc: "default - agentic and coding flagship, 256K"
    context: 262144
    reasoning: optional
  - id: mistral-large-2512
    role: advanced
    desc: "advanced - mistral large 3, 675B MoE, 256K"
    context: 262144
    reasoning: none
  - id: mistral-small-2603
    role: fast
    desc: "fast - mistral small 4, low latency"
    context: 262144
    reasoning: optional
  - id: codestral-2508
    role: coding
    desc: "coding - FIM and completion, 4K max output"
    context: 262144
    reasoning: none
  - id: ministral-8b-2512
    role: economy
    desc: "economy - small tool-capable model"
    context: 262144
    reasoning: none
add_aliases:
  mistral-medium-3-5: mistral-medium-2604
  mistral-medium-3: mistral-medium-2604
  mistral-medium-latest: mistral-medium-2604
  mistral-small-latest: mistral-small-2603
  mistral-large-latest: mistral-large-2512
  codestral-latest: codestral-2508
  devstral-2512: mistral-medium-2604
  devstral-latest: mistral-medium-2604
  devstral-medium-latest: mistral-medium-2604
  devstral-medium-2507: mistral-medium-2604
  devstral-small-2507: mistral-small-2603
  labs-devstral-small-2512: mistral-medium-2604
  magistral-medium-latest: mistral-medium-2604
  magistral-small-latest: mistral-small-2603
  mistral-medium-2508: mistral-medium-2604
  mistral-medium-2505: mistral-medium-2604
  mistral-small-2506: mistral-small-2603
  mistral-large-2411: mistral-medium-2604
  mistral-large-2407: mistral-large-2512
  ministral-8b-latest: ministral-8b-2512
  open-mistral-nemo-2407: ministral-8b-2512
validation_model: ministral-3b-2512
notes: "ANSWERS THE BRIEF: Sylliptor's `mistral-medium-3-5` is a real product NAME but the stable
  callable pin is the dated snapshot mistral-medium-2604 (docs card slug mistral-medium-3-5-26-04;
  models.dev and LiteLLM both carry mistral-medium-2604 with identical metadata). This lineup pins
  dated ids and aliases the name-based forms onto them. CODING SLOT CAVEAT: codestral-2508 is
  FIM/completion-oriented and capped at ~4K max output, so it is unsuitable for large agentic patch
  turns — Mistral itself points agentic coding at Mistral Medium 3.5. Routers should prefer the
  default over the coding slot for multi-file edits; keeping codestral as `coding` is only correct if
  Sylliptor uses it for completions. RETIRING 2026-07-31: devstral-2512, magistral-medium-2509,
  magistral-small-2509, mistral-small-2506, open-mistral-nemo-2407. mistral-medium-2508/2505 survive
  to 2026-08-31. Two non-obvious rows taken from Mistral's own replacement column: labs-devstral-small-2512
  → Medium 3.5 (NOT Small 4), and mistral-large-2411 (Large 2.1) → Medium 3.5 (NOT Large 3), while
  older mistral-large-2407/2402 DO point at Large 3. Context normalised to 262144 throughout ('256k' on
  every card), reconciling models.dev's 256000 vs OpenRouter's 262144 as a rounding artifact. NOTED
  DISAGREEMENT: the verifier argued mistral-large-2512 should be dropped as a downgrade; retained,
  because its card describes a 675B-total/41B-active MoE and the legacy table routes Large 1.0/2.0
  users to it. Real caveat: Large 3 is reasoning=none, so mistral-medium-2604 remains the better
  agentic default. Live discovery: GET https://api.mistral.ai/v1/models (401 unauthenticated) is the
  best runtime source here given the dual dated/-latest scheme."
```

```yaml
preset: xai
checked: 2026-07-19
sources:
  - https://docs.x.ai/docs/models
  - https://docs.x.ai/developers/grok-4-5
  - https://docs.x.ai/developers/migration/may-15-retirement
  - https://x.ai/news/grok-build-0-1
verdict: stale
remove:
  - grok-code-fast-1   # retired 2026-05-15; xAI's migration page points code workloads at grok-build-0.1
suggested_models:
  - id: grok-4.5
    role: default
    desc: "default - flagship coding and agentic work"
    context: 500000
    reasoning: optional      # effort low|medium|high, default high; no off switch
  - id: grok-build-0.1
    role: coding
    desc: "coding - agentic engineering model, 256K"
    context: 256000
    reasoning: always-on
  - id: grok-4.3
    role: advanced
    desc: "advanced - 1M context window"
    context: 1000000
    reasoning: optional
  - id: grok-4.20-0309-reasoning
    role: reasoning
    desc: "reasoning - dedicated snapshot, 1M context"
    context: 1000000
    reasoning: always-on
  - id: grok-4.20-0309-non-reasoning
    role: fast
    desc: "fast - no-reasoning snapshot, 1M context"
    context: 1000000
    reasoning: none
add_aliases:
  grok-code-fast-1: grok-build-0.1
  grok-4-0709: grok-4.3
  grok-4-fast-reasoning: grok-4.3
  grok-4-1-fast-reasoning: grok-4.3
  grok-4-fast-non-reasoning: grok-4.20-0309-non-reasoning
  grok-4-1-fast-non-reasoning: grok-4.20-0309-non-reasoning
  grok-3: grok-4.3
validation_model: grok-4.20-0309-non-reasoning
notes: "Ids use DOTS not dashes (grok-4.5, not grok-4-5). RETIREMENT 2026-05-15, full shutdown
  2026-08-15: eight retired slugs auto-redirect to grok-4.3 and are BILLED AT grok-4.3 RATES
  ($1.25/$2.50 per Mtok) — reasoning slugs land on effort=low, non-reasoning slugs on effort=none.
  Migrate explicitly so cost and effort are intentional; note grok-3 traffic gets silently upgraded to
  a 1M-context tier price. DELIBERATE ALIAS DIVERGENCE: xAI redirects the *-non-reasoning slugs to
  grok-4.3 with effort=none, but Sylliptor's alias table carries no effort field, so those two map to
  grok-4.20-0309-non-reasoning instead — that preserves non-reasoning behaviour by construction rather
  than relying on an effort param the alias cannot express. grok-build-0.1 is available in us-east-1
  and us-west-2 ONLY. CONTESTED FACTS, recorded not resolved: (a) grok-4.20-multi-agent-0309 function
  calling — docs.x.ai says 'Function calling: Yes' and Oracle OCI / Cloudflare agree, while models.dev
  says tool_call=false and OpenRouter omits the tools param; that is a known catalog bug (opencode
  issue #21669), so tool support is real. It is left out of the lineup only because its
  reasoning.effort sets agent count, which is a different cost model. (b) Context for the 4.20 family
  — docs.x.ai says 1M, OpenRouter and llmreference say 2M; official used. (c) Max output is NOT
  published per model; third-party sources conflict (~30K vs 131K), so do NOT hardcode an output
  ceiling — clamp conservatively on long edits. Claims dropped as unsourced: a /v1/language-models
  pricing endpoint, 'grok-4-5 404s', price doubling above 200K, and EU console unavailability. Live
  discovery: GET https://api.x.ai/v1/models."
```

```yaml
preset: cohere
checked: 2026-07-19
sources:
  - https://docs.cohere.com/docs/compatibility-api
  - https://docs.cohere.com/docs/models
  - https://docs.cohere.com/docs/reasoning
  - https://docs.cohere.com/docs/deprecations
verdict: current
remove: []
suggested_models:
  - id: command-a-plus-05-2026
    role: default
    desc: "default - newest command a+, 128K context"
    context: 128000
    reasoning: optional
  - id: command-a-reasoning-08-2025
    role: reasoning
    desc: "reasoning - 256K context, thinking is a toggle"
    context: 256000
    reasoning: optional
  - id: command-a-03-2025
    role: advanced
    desc: "advanced - 256K context, prior flagship"
    context: 256000
    reasoning: none
  - id: command-r7b-12-2024
    role: economy
    desc: "economy - cheapest live chat model, 128K"
    context: 128000
    reasoning: none
add_aliases:
  command: command-a-03-2025
  command-light: command-r-08-2024
  command-r: command-r-08-2024
  command-r-plus: command-r-plus-08-2024
validation_model: command-r7b-12-2024
notes: "BASE URL IS CORRECT AS-IS — do NOT migrate. docs.cohere.com/docs/compatibility-api uses
  https://api.cohere.ai/compatibility/v1 verbatim in every sample, matching profile_presets.py:888.
  An earlier draft of this research claimed the .ai host was undocumented and should move to
  api.cohere.com with v2/chat; that was false and would have broken a working documented endpoint
  under this preset's openai_compat protocol. Retracted and recorded here so it is not 'rediscovered'
  next refresh. NORTH-MINI-CODE-1-0 DELIBERATELY NOT SHIPPED: the id is real (256K context, 64K
  output, Apache-2.0 open weights) and looks like the coding model this preset lacks, but Cohere's own
  file lists its endpoint support as Chat V2 ONLY — it is absent from the chat model table and from
  the compatibility-api docs. Promote it only after a live 200 from /compatibility/v1/chat/completions.
  Note the OpenRouter id cohere/north-mini-code:free is NOT valid against the Cohere API. REASONING
  CAVEAT: command-a-reasoning-08-2025 is a toggle (thinking={'type':'disabled'}, defaults enabled) but
  `thinking` is a native Chat V2 parameter and is NOT documented as passable through
  /compatibility/v1 — over this transport the effective behaviour may be thinking-on with no way to
  disable. Worth a live probe. SEPARATE STALE ITEM, out of scope for the lineup but worth a fix: this
  preset's setup_warning/notes at profile_presets.py:901-908 say the Cohere v1 hosted web-search
  connector 'remains available', but the deprecations page lists /v1/connectors and the /v1/chat
  `connectors` + `search_queries_only` parameters as shut down 2025-09-15 — COHERE_WEB_SEARCH_ADAPTER
  is probably pointed at a dead endpoint. No /models listing on the compatibility API; the native GET
  https://api.cohere.com/v1/models exists but is a different host and protocol — do not wire it into
  the openai_compat client."
```

---

## Gateways and aggregators

```yaml
preset: openrouter
checked: 2026-07-19
sources:
  - https://openrouter.ai/api/v1/models     # live catalog, fetched 2026-07-19
  - https://openrouter.ai/openai/gpt-5.6-terra
  - https://openrouter.ai/anthropic/claude-sonnet-5
  - https://openrouter.ai/z-ai/glm-5.2
verdict: partially-stale
remove:
  - openai/gpt-5.5        # still live; superseded by the gpt-5.6 luna/terra/sol tiers
  - qwen/qwen3.7-plus     # still live; z-ai/glm-5.2 beats it on both price axes at the same context
suggested_models:
  - id: anthropic/claude-sonnet-5
    role: default
    desc: "default - coding and agents, 1M context"
    context: 1000000
    reasoning: optional
  - id: anthropic/claude-opus-4.8
    role: advanced
    desc: "advanced - long-horizon autonomous work"
    context: 1000000
    reasoning: optional
  - id: openai/gpt-5.6-terra
    role: coding
    desc: "coding - balanced gpt-5.6 tier, 1.05M context"
    context: 1050000
    reasoning: optional
  - id: openai/gpt-5.6-luna
    role: fast
    desc: "fast - cost-efficient gpt-5.6 tier"
    context: 1050000
    reasoning: optional
  - id: z-ai/glm-5.2
    role: economy
    desc: "economy - cheap 1M-context tool caller"
    context: 1048576
    reasoning: optional
  - id: deepseek/deepseek-v4-pro
    role: agentic
    desc: "agentic - reasoning MoE, 1M context"
    context: 1048576
    reasoning: optional
add_aliases: {}
validation_model: deepseek/deepseek-v4-flash
notes: "All five ids Sylliptor currently ships DO still resolve — this is curation, not breakage,
  hence no aliases. Vendor prefixes are exact and are the classic error source here: z-ai/ (not zai/
  or zai-org/), x-ai/ (not xai/), moonshotai/ (not moonshot/). Prices above are per 1M in/out as
  returned by the live endpoint on 2026-07-19. google/gemini-3.5-flash was dropped from the curated
  list: at $1.50/$9.00 it is dominated on BOTH axes by openai/gpt-5.6-luna ($1.00/$6.00) at the same
  ~1M context, with no cited evidence of a latency advantage. glm-5.2 ($0.26/$0.81) is chosen for
  capability-per-dollar, NOT for being the price floor — deepseek/deepseek-v4-flash is cheaper at
  $0.09/$0.18 and is used as validation_model. minimax/minimax-m3 is cheaper still but its
  supported_parameters has NO 'reasoning', which is why it lost the economy slot. ROUTING META-MODELS:
  openrouter/auto (routes across candidates, billed at the routed model's rate, no markup) and
  openrouter/auto-beta (2M ctx, tools+reasoning) are usable; openrouter/fusion has NO 'tools' in
  supported_parameters and is unsafe as an agent default; openrouter/pareto-code's tool support is not
  confirmed on its model page — verify supported_parameters before defaulting an agent to either.
  Avoid pinning agents to the '~vendor/...-latest' floating aliases; the underlying model changes
  without notice. :free variants are heavily rate-limited and unsuitable for agent loops. BEST
  LIVE-DISCOVERY TARGET IN THIS REFRESH: GET https://openrouter.ai/api/v1/models needs no auth and
  returns context_length, pricing and supported_parameters per entry — Sylliptor could drive this
  preset entirely from it, checking supported_parameters for 'tools' and 'reasoning'."
```

```yaml
preset: perplexity
checked: 2026-07-19
sources:
  - https://docs.perplexity.ai/docs/agent-api/models.md
  - https://docs.perplexity.ai/docs/agent-api/openai-compatibility.md
  - https://docs.perplexity.ai/api-reference/models-get.md
  - https://docs.perplexity.ai/getting-started/pricing
verdict: stale
remove:
  - sonar-pro   # sonar family returns HTTP 400 on tool definitions — unusable for agent loops
  - sonar       # same; keep wired as a web-search backend only
suggested_models:
  - id: anthropic/claude-sonnet-5
    role: default
    desc: "default - balanced coding workhorse"
    context: null
    reasoning: unknown
  - id: anthropic/claude-opus-4-8
    role: advanced
    desc: "advanced - strongest coding option here"
    context: null
    reasoning: unknown
  - id: openai/gpt-5.6-terra
    role: reasoning
    desc: "reasoning - long-horizon work, tiered at 272K"
    context: null
    reasoning: unknown
  - id: perplexity/kimi-k2.7-code
    role: coding
    desc: "coding - code-tuned, $0.95/$4 per 1M"
    context: null
    reasoning: unknown
  - id: google/gemini-3.1-flash-lite
    role: fast
    desc: "fast - low-latency edits and small tasks"
    context: null
    reasoning: unknown
  - id: nvidia/nemotron-3-super-120b-a12b
    role: economy
    desc: "economy - cheap bulk work, no prompt cache"
    context: null
    reasoning: unknown
add_aliases: {}
validation_model: openai/gpt-5.4-nano
notes: "ENDPOINT CHANGE REQUIRED: base_url must become https://api.perplexity.ai/v1 (currently
  https://api.perplexity.ai). This preset is much bigger than 'sonar' — Perplexity now runs TWO
  surfaces. (1) AGENT API: canonical POST /v1/agent, with POST /v1/responses accepted as an
  OpenAI-SDK-compatible alias. It hosts every vendor-prefixed id above. The docs show NO
  /v1/chat/completions support, so Sylliptor needs a Responses-style client path or this preset stays
  search-only — that is the gating decision for a maintainer. (2) SONAR API: POST /v1/sonar, accepts
  only sonar|sonar-pro|sonar-reasoning-pro|sonar-deep-research, has NO tools parameter and returns
  HTTP 400 when sent tool definitions. Since every Sylliptor agent turn ships tool definitions, the
  sonar ids would 400 on first use — that, not obsolescence, is why they are removed. NO ALIASES ON
  PURPOSE: none of the four sonar ids were renamed; all remain valid on the Sonar surface. This is an
  endpoint migration, not a model rename, and mapping them onto an Agent API id would silently drop
  context (sonar-pro), drop reasoning (sonar-reasoning-pro), or substitute a different product
  (sonar-deep-research). Migrate saved configs deliberately, with a message telling the user those ids
  live on a different, tool-less endpoint. CONTEXT IS null EVERYWHERE ON PURPOSE: models-get.md
  exposes only id/object/created/owned_by, so no per-model context is published — do NOT borrow
  models.dev or OpenRouter numbers, which describe different surfaces. Reasoning is likewise
  undocumented per model; the roster listing xai/grok-4.20-reasoning and -non-reasoning as SEPARATE
  ids suggests reasoning is not a uniform per-request toggle. Anthropic ids here require
  max_output_tokens. sonar-pro-search (200K) is OpenRouter-exclusive and not callable on
  api.perplexity.ai. Live discovery: GET https://api.perplexity.ai/v1/models returns the Agent API
  roster."
```

```yaml
preset: together
checked: 2026-07-19
sources:
  - https://docs.together.ai/docs/serverless-models
  - https://docs.together.ai/docs/deprecations
  - https://www.together.ai/pricing
verdict: partially-stale
remove:
  - zai-org/GLM-5.1                            # retired from serverless, removal date 2026-07-10
  - Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8    # retired from serverless, removal date 2026-06-04
suggested_models:
  - id: zai-org/GLM-5.2
    role: default
    desc: "default - general coding, 256K context"
    context: 262144
    reasoning: optional
  - id: moonshotai/Kimi-K2.7-Code
    role: coding
    desc: "coding - code specialist, 256K context"
    context: 262144
    reasoning: always-on     # no exposed toggle — do NOT emit a reasoning-off param
  - id: deepseek-ai/DeepSeek-V4-Pro
    role: reasoning
    desc: "reasoning - premium tier, 512K context"
    context: 512000
    reasoning: optional
  - id: MiniMaxAI/MiniMax-M3
    role: economy
    desc: "economy - cheapest 512K-context option"
    context: 524288
    reasoning: always-on
  - id: openai/gpt-oss-120b
    role: fast
    desc: "fast - mid-tier, 128K context"
    context: 128000
    reasoning: optional
  - id: openai/gpt-oss-20b
    role: fallback
    desc: "fallback - cheapest tool-capable id"
    context: 128000
    reasoning: optional
add_aliases:                 # FALLBACK POLICY, not vendor renames — see notes
  zai-org/GLM-5.1: zai-org/GLM-5.2
  Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8: moonshotai/Kimi-K2.7-Code
  Qwen/Qwen3-Coder-Next-FP8: moonshotai/Kimi-K2.7-Code
validation_model: openai/gpt-oss-20b
notes: "CORRECTS THE BRIEF'S SUSPICION: yes, both zai-org/GLM-5.1 and
  Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 are gone from serverless — confirmed on the deprecations
  page with removal dates 2026-07-10 and 2026-06-04. openai/gpt-oss-120b, however, IS still live
  (models.dev's togetherai list is simply incomplete). ALIASES ARE NOT RENAMES: Together publishes NO
  successor for any retired id (blank successor column on all three rows), so these are Sylliptor-side
  fallback policy. Two of the three are CROSS-VENDOR substitutions — surface the substitution to the
  user at resolution time rather than swapping silently. Ids are CASE-SENSITIVE and vendor-prefixed;
  watch dots (Kimi-K2.7-Code, GLM-5.2, MiniMax-M3) and the MiniMaxAI/ zai-org/ thinkingmachines/
  prefixes. REASONING IS NOT UNIFORM: Kimi-K2.7-Code and MiniMax-M3 reason unconditionally with no
  exposed toggle, so emitting a reasoning-off or effort field for them sends an unsupported param;
  DeepSeek-V4-Pro and gpt-oss-* expose a toggle. Context for gpt-oss-* differs between the official
  serverless page (128,000) and models.dev (131,072) — the lower value is used. Together retires
  serverless models on a published schedule with no successor mapping, so expect id churn every few
  months; retired models stay reachable only via on-demand or reserved dedicated endpoints. LIVE AND
  WORTH REVISITING but not shipped: thinkingmachines/Inkling (512K, agentic, $1.00/$4.05),
  Qwen/Qwen3.7-Plus (1M ctx, $0.32/$1.28), nvidia/nemotron-3-ultra-550b-a55b (512K reasoning),
  Qwen/Qwen3.7-Max (Together's Qwen flagship — context NOT published on the serverless page, so it was
  left out rather than guessed). Live discovery: GET https://api.together.xyz/v1/models (Bearer key
  required)."
```

```yaml
preset: fireworks
checked: 2026-07-19
sources:
  - https://fireworks.ai/models
  - https://docs.fireworks.ai/serverless/pricing
  - https://docs.fireworks.ai/api-reference/list-models
verdict: stale
remove:
  - accounts/fireworks/models/qwen2p5-coder-32b-instruct   # Qwen2.5 era; serverless "Not supported" — on-demand GPU only
  - accounts/fireworks/models/kimi-k2p6                    # superseded by kimi-k2p7-code, same 262K ctx and input price
  - accounts/fireworks/models/glm-5p1                      # superseded by glm-5p2 (1M ctx vs 203K)
suggested_models:
  - id: accounts/fireworks/models/glm-5p2
    role: default
    desc: "default - general agentic coding, 1M context"
    context: 1048575
    reasoning: optional
  - id: accounts/fireworks/models/kimi-k2p7-code
    role: coding
    desc: "coding - 262K context, tool calling"
    context: 262000
    reasoning: optional
  - id: accounts/fireworks/models/deepseek-v4-pro
    role: reasoning
    desc: "reasoning - 1M context, 384K output"
    context: 1000000
    reasoning: optional
  - id: accounts/fireworks/models/deepseek-v4-flash
    role: fast
    desc: "fast - lowest-cost 1M-context option"
    context: 1000000
    reasoning: optional
  - id: accounts/fireworks/models/minimax-m3
    role: economy
    desc: "economy - 512K context, effort control"
    context: 512000
    reasoning: optional
  - id: accounts/fireworks/models/qwen3p7-plus
    role: fallback
    desc: "fallback - 262K context, standard tier only"
    context: 262144
    reasoning: optional
add_aliases:
  accounts/fireworks/models/qwen2p5-coder-32b-instruct: accounts/fireworks/models/kimi-k2p7-code
  accounts/fireworks/models/kimi-k2p6: accounts/fireworks/models/kimi-k2p7-code
  accounts/fireworks/models/glm-5p1: accounts/fireworks/models/glm-5p2
validation_model: accounts/fireworks/models/deepseek-v4-flash
notes: "The 'p' decimal convention is real and is the trap the brief flagged: 5p2 = 5.2, k2p7 = k2.7.
  WORST BUG HERE: qwen2p5-coder-32b-instruct is not merely old — its model page says serverless 'Not
  supported', so it is on-demand-GPU only and CANNOT work against api.fireworks.ai/inference/v1 at
  all. Catalog membership is NOT proof of serverless availability on this provider; that generalises
  beyond this one id. SECOND NAMESPACE worth exposing separately: accounts/fireworks/routers/*
  (glm-5p2-fast, kimi-k2p7-code-fast, kimi-k2p6-fast, kimi-k2p6-turbo, glm-5p1-fast) are
  latency-optimised routes at roughly 1.5–2x the model price — a distinct opt-in tier, not something
  to mix into the model list. Fireworks also sells Standard vs Priority serverless tiers per model
  (Priority ~1.5x); qwen3p7-plus and gpt-oss-20b are Standard-only, which is why qwen3p7-plus sits at
  fallback. LIVE DISCOVERY IS AWKWARD HERE: there is no plain GET /v1/models; the documented endpoint
  is GET https://api.fireworks.ai/v1/accounts/{account_id}/models (control plane, needs an account
  id), and it still would not tell you which models are serverless-enabled — so keep this preset
  static. Ecosystem signal: Fireworks' own FireConnect CLI defaults OpenCode to the alias
  'glm-fast-latest', supporting GLM as the default role; not proposed as an alias because its
  fully-qualified API id could not be confirmed."
```

---

## Cross-cutting recommendations

1. **Add a `context` / `reasoning_mode` field to `ProfilePreset`.** Six presets have models where
   reasoning cannot be disabled (gemini 3.x, cerebras `gpt-oss-120b`, moonshot K2.7/K3, together
   Kimi/MiniMax) and three where sending a reasoning-off flag is an outright API error. Today
   Sylliptor has nowhere to record that, so the information lives only in prose.

2. **Wire live discovery where it is genuinely good.** Ranked by value:
   `openrouter` (no auth, returns pricing + `supported_parameters`) > `groq` (account-visible, catalog
   churns fast) > `cerebras` (tier-dependent visibility) > `anthropic` (returns a capabilities object)
   > `deepseek`/`moonshot`/`mistral`/`xai`/`together` (ids only). Not viable: `kimi-code`, `zhipu`,
   `qwen`, `cohere` (compat surface), `fireworks`, `bytedance`.

3. **`validation_model` is set on only 9 of 33 presets** (`profile_presets.py:91`). Every preset above
   proposes one. Note validation is a billed `chat()` ping, not a `/models` call
   (`setup_wizard.py:847`) — on `kimi-code` that spends metered membership quota.

4. **Two `role` values in the descriptions are load-bearing but unparsed.** Nothing reads the prefix
   before `" - "` (`model_options_for_preset`, `profile_presets.py:1043`). If roles are ever going to
   drive routing, they need to become a real field first.

### Confidence caveats

- **bytedance** is the weakest result: Ark's docs render client-side and could not be read, so every
  id rests on registry evidence. Probe with a live key before shipping.
- **01ai** deprecation is a low-confidence recommendation resting on absence of evidence; one
  authenticated `GET /v1/models` would settle it.
- **perplexity** context windows are `null` because Perplexity does not publish them per model — that
  is a deliberate refusal to guess, not missing work.
