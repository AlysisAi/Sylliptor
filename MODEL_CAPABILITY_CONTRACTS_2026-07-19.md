# Sylliptor model capability contracts — 2026-07-19

Verified source data for the model capability layer: a real `reasoning_mode` + `context` field on
presets, live model discovery, and tier-aware context accounting. Follow-up to
`MODEL_CATALOG_REFRESH_2026-07-19.md` (the static catalog, already applied). Sylliptor is an agentic
coding CLI — every request carries tool definitions — and on several providers a wrong reasoning
flag is now an HTTP error, not degraded output.

**Method.** One research agent per provider surface (19 surfaces), each re-sourcing every claim from
the prior refresh against official docs with instructions to copy parameter spellings
character-for-character; then one adversarial verifier per provider attacking exact spellings,
claimed failure modes, and registry contamination; then a correction pass wherever the verifier
landed fatal/major challenges (7 providers were corrected: openai, gemini, zhipu, moonshot, groq,
xai, openrouter); then a completeness critic that filed 12 material gaps, each closed by a dedicated
gap-fill agent. 59 agents, ~3.8M tokens, 818 tool calls. Official docs are ground truth throughout;
registries (models.dev, LiteLLM, OpenRouter) appear only as recorded conflicts or clearly-labelled
last-resort values. Where a page would not load, the block says so — and where only a billed call
can settle a question, the block carries a one-paste curl under `probes_needed` instead of a guess.

**How to read the verification tags.** *verified-clean* = the adversarial pass found nothing
serious (minor challenges, if any, are listed under the block). *corrected* = the verifier found
fatal/major problems and a correction pass re-sourced and fixed the block; the challenge count is
shown. Gap-fill addenda are post-verification supplements and, where they conflict with the block
(this happens once, on minimax's `thinking` enum), the addendum is the newer finding — both are
kept, conflict recorded.

## Verdicts on the claims this task asked to re-verify

| prior claim | verdict |
|---|---|
| moonshot: K2.7 models ERROR on `thinking={"type":"disabled"}` | **CONFIRMED** — docs: "Kimi K2.7 Code model will throw an error if the thinking mode is disabled"; Claude Code doc quotes a `400 invalid thinking` error |
| moonshot: k3 rejects the K2.x `thinking` param, accepts only `reasoning_effort: "max"` | **MODIFIED** — k3 does use `reasoning_effort` instead of `thinking` (reject-vs-ignore for a sent `thinking` param is undocumented; probe filed), but the allowed set is `low\|high\|max` with default `max`, not only-`max` |
| only kimi-k2.6 is toggleable (moonshot platform) | **CONFIRMED** |
| gemini 3.x: effort `{minimal,low,medium,high}` only (`{low,medium,high}` on Pro), no off switch | **CONFIRMED** — the wire field is `generationConfig.thinkingConfig.thinkingLevel`; `minimal` is documented "Not supported" on 3.1 Pro; no model in the lineup can disable thinking |
| cerebras gpt-oss-120b: `reasoning_effort` low\|medium\|high, no "none"; `disable_reasoning` unsupported since 2026-07-21 | **CONFIRMED** (both halves; the removal date has now passed — `disable_reasoning` is dead) |
| anthropic: Opus 4.8 / Sonnet 5 / Fable 5 = adaptive thinking NOT extended; Haiku 4.5 the inverse, no effort param | **CONFIRMED** — adaptive = `thinking:{"type":"adaptive"}` with depth via `output_config:{"effort":...}`; the extended shape `{"type":"enabled","budget_tokens":N}` gets a 400 on all four adaptive models; haiku-4-5 is absent from the effort supported-models list |
| together: Kimi-K2.7-Code and MiniMax-M3 reason unconditionally, no toggle | **SPLIT** — Kimi-K2.7-Code confirmed (thinking "cannot be disabled"); MiniMax-M3 **REFUTED**: Together's own model page says thinking is "Toggleable at request time — no separate model version required" (exact wire spelling unconfirmed; probe filed) |
| minimax: no reasoning control documented on any model | **REFUTED for M3** — M3 takes `thinking` with modes `enabled` / `adaptive` / `disabled` (first-party model card + GitHub README; the api-reference enum lists only `disabled\|adaptive` — conflict recorded); M2.x models still expose no control |
| openai: `reasoning.mode=pro` replaces a -pro slug on gpt-5.6 | **CONFIRMED** — Responses API only (`"reasoning": {"mode": "pro"}`); confirmed from the guide AND the official Python SDK type stubs; note the 5.5 generation *does* keep a real `gpt-5.5-pro` slug, so the rule is generation-specific |
| kimi-code: thinking-off silently routes to K2.6 | **CONFIRMED** — a model substitution, not a speed toggle; applies on the coding surface. k3's effort default is `high` on this surface (the platform.kimi.ai `max` default is a surface divergence, resolved — two coding-surface docs agree on `high`) |
| Sonnet 5 emits ~30% more tokens than 4.6 | **CONFIRMED and generalized** — official wording: "The same input text produces approximately 30 percent more tokens than on earlier models"; applies to the new-tokenizer family (fable-5, opus-4-8, opus-4-7, sonnet-5), while haiku-4-5 keeps the previous tokenizer. Never reuse counts across the boundary; `/v1/messages/count_tokens` counts under the tokenizer of the model passed |
| cerebras free 65K vs paid 131K; kimi-code k3 256K vs 1M Allegretto+ | **CONFIRMED** — both tier splits verbatim from official pages (cerebras: 64–65K/32K free vs 131K/40K paid; kimi-code: Moderato 256K, Allegretto+ 1M, Andante no k3 at all) |
| minimax M3 bills higher above 512K; qwen coder-plus tiered pricing | **CONFIRMED** — M3: ≤512K input $0.30/M in, >512K $0.60/M in (2×), only 512K guaranteed; coder-plus: four tiers with boundaries at 32K/128K/256K, topping at $6/$60 per M in the 256K–1M band |
| xai 4.20 family max output unpublished | **CONFIRMED** — no xAI doc publishes a max-output number for any catalog model; third parties conflict (~30K vs 131K); left `unpublished` everywhere |

**Two catalog corrections surfaced in passing** (fix in `profile_presets.py` data): mistral
`codestral-2508` context is **128K**, not the 262144 the applied catalog carries (official model
card); cohere `command-a-plus-05-2026` is 128K input / **64K output** — half the window of the
Command A family it sits beside (catalog context of 128000 is right; do not "fix" it upward).

## The fixed schema

Every provider block below follows this schema exactly. `unknown` / `none` / `unpublished` are
written explicitly — an absent fact is a bug, not an omission.

```yaml
provider: <preset key(s)>
endpoints:
  - url: <base url>
    protocol: <openai-chat | openai-responses | anthropic-messages | native>
checked: 2026-07-19
sources:
  - <url actually opened>   # "(FAILED)" marks pages that would not load or render client-side
reasoning:
  param: <exact request-body spelling(s), copied character-for-character, or none>
  values: <exact allowed value strings, split per model family if they differ>
  default_when_omitted: <behavior when no reasoning param is sent>
  unsupported_value_behavior: <silently-ignored | silently-coerced | http-<code> + error body shape | unknown>
  per_model:
    - id: <model id from the applied catalog>
      mode: <always-on | optional | none>
      effort: <adjustable: [values] | fixed: value | n/a>
  silent_substitutions: <model swaps triggered by reasoning flags, or none>
  tools_and_streaming: <thinking blocks in SSE, interleaved thinking with tools, whether reasoning output counts against max output>
  compat_passthrough: <for each compat surface the provider exposes: passes | stripped | unknown; n/a if single surface>
discovery:
  endpoint: <exact GET url, or none>
  auth: <none | api-key | bearer>
  metadata: <ids-only | ids+context | ids+capabilities | ids+capabilities+pricing>
  account_scoped: <no | yes: what varies per account/tier>
  rate_limits_or_caching: <documented values, or undocumented>
  verdict: <drive-from-live | augment-static | stay-static>
  refresh_strategy: <on-config-open | daily-cache | per-session | n/a>
context:
  per_model:
    - id: <model id>
      context: <tokens, tier splits spelled out explicitly>
      max_output: <tokens | unpublished>
      long_context_pricing: <flat | boundary + both rates>
      overpromise_risk: <no | yes: which accounts get less than the headline number>
  token_counting_endpoint: <url | none>
probes_needed:
  - question: <what only an authenticated/billed probe can settle>
    probe: <one-paste runnable curl>
conflicts:
  - <source A says X; source B says Y — recorded, not resolved>
notes: <anything load-bearing that fits nowhere above>
```

---


## Contents

1. [Summary table](#summary-table)
2. [Provider contracts](#provider-contracts) — one fixed-schema YAML block per preset group, with gap-fill addenda and verifier notes
3. [API-error hazard list](#api-error-hazard-list) — ranked, then the full per-provider inventory
4. [Part D — Perplexity Agent API client scoping](#part-d--perplexity-agent-api-client-scoping)
5. [Recommended implementation order](#recommended-implementation-order)

---

# Summary table

One row per preset group. "No-off models" = models where reasoning cannot be turned off (emitting
any disable flag is at best ignored, at worst a 400 or a silent model swap). Full detail, exact
value sets, and error bodies are in the per-provider blocks.

| preset | reasoning control (wire spelling) | no-off models | wrong-flag failure | discovery | biggest context trap |
|---|---|---|---|---|---|
| openai + openai-responses | chat: `reasoning_effort` (flat string); responses: `reasoning:{effort,mode,summary}` | gpt-5.3-codex | http-400 `unsupported_value`; **tools + effort≠none on /v1/chat/completions = 400 on the 5.6/5.4 families** | augment-static (ids-only) | reasoning tokens count against `max_output_tokens` |
| anthropic ×3 | `thinking:{"type":"adaptive"\|"disabled"}` + depth via `output_config:{"effort":...}`; haiku only: `thinking:{"type":"enabled","budget_tokens":N}` (min 1024) | claude-fable-5 | http-400 on the extended shape to adaptive models; **400 on any non-default temperature/top_p/top_k on all four adaptive models** | **drive-from-live** (`/v1/models` carries context, output caps, thinking flags, effort levels) | haiku 200K beside 1M siblings; new tokenizer ≈30% more tokens (all except haiku) |
| gemini ×3 | `generationConfig.thinkingConfig.thinkingLevel` (REST camelCase) | all four | http-400 on `thinkingLevel`+`thinkingBudget` together; **missing `thoughtSignature` on tool-call replay = 400 at turn 2** | augment-static (`inputTokenLimit`/`outputTokenLimit`/thinking flag live) | 3.1-pro input price doubles past 200K prompt tokens |
| deepseek | `thinking:{"type":"enabled"\|"disabled"}` + `reasoning_effort` `"high"\|"max"` (openai); `output_config:{"effort"}` (anthropic surface) | none | silently-coerced (`low`/`medium`→`high`, `xhigh`→`max`); sampling params silently ignored; unknown model ids on /anthropic silently fall back to v4-flash | augment-static (clean 2-id list) | legacy `deepseek-chat`/`-reasoner` die **2026-07-24**; max_output 384K |
| qwen ×3 regions | `enable_thinking` bool (via `extra_body` in openai SDKs) + `thinking_budget` | none | thinking is ON by default on the 3.7 line (silent cost, `reasoning_content` deltas); coder + `enable_thinking` undocumented | stay-static | coder models absent in US region; coder-plus 4 price tiers (32K/128K/256K); coder-next tops at 256K |
| zhipu | `thinking:{"type":"enabled"\|"disabled"}`; glm-5.2 adds `reasoning_effort` (`low`/`medium`→`high` coerced) | glm-4.7 (thinks compulsorily; toggle documented but forced) | OpenRouter-style `reasoning:{...}` object **silently ignored** (field-confirmed); out-of-enum likely 400 code 1214 | stay-static | **no length-tiered pricing** (flat per model — corrects prior assumption); `max_tokens` hard range 1..131072 |
| moonshot ×2 | K2.x: `thinking:{"type":"enabled"}` only; k3: `reasoning_effort` `low\|high\|max` (no `thinking`) | kimi-k2.7-code, -highspeed, kimi-k3 (only k2.6 toggles) | **http-400 on `thinking:{"type":"disabled"}`** (K2.7); 400 on temperature≠1.0/top_p≠0.95; k2.6 requires `reasoning_content` replay in tool loops | augment-static (`supports_reasoning` + `context_length` live) | k3 Tier0 TPM 500K can never fill the 1M window (429s) |
| kimi-code | `reasoning_effort` `low\|high\|max` on k3 (default `high` on this surface); thinking default ON all three | all three — "off" is not off | **thinking-off silently swaps the model to K2.6**; effort outside the set = 400 | stay-static (no /models; validation is a billed call) | k3 = 256K Moderato / 1M Allegretto+ / unavailable on Andante |
| minimax | M3: `thinking:{"type":"enabled"\|"adaptive"\|"disabled"}` (api-ref enum says `disabled\|adaptive` — conflict recorded); M2.x: none | MiniMax-M2.7, -highspeed, M2.5 (no control) | out-of-enum undocumented (probe); **anthropic surface defaults thinking DISABLED** — silent non-reasoning agent | augment-static | >512K input bills 2×; only 512K of the 1M is guaranteed |
| bytedance | `thinking:{"type":...}` family; seed-2.0 thinking ON by default; `reasoning_effort` with `minimal` = disable (Volcengine doc) | unknown per-model (probe) | http-400 `InvalidParameter` **live-observed** on `reasoning_effort` (coding surface); per-model acceptance unprobed | stay-static (no listing found; SDK ships no models resource) | every number here needs a live-key probe before shipping |
| groq | `reasoning_effort` (gpt-oss `low\|medium\|high`; qwen `none\|default`) + `reasoning_format` + `include_reasoning` | gpt-oss ×2, groq/compound | documented 400s: `raw` format + tools/JSON; `include_reasoning`+`reasoning_format` together; compound rejects client tools | augment-static (`context_window` live, account-visible) | qwen3.6-27b output cap 32K vs gpt-oss 65K; reasoning default budget 1024 if no cap sent |
| cerebras | `reasoning_effort` (gpt-oss `low\|medium\|high`; glm/gemma add `"none"`) + `clear_thinking` (glm only) | gpt-oss-120b | `disable_reasoning` **dead since 2026-07-21**; gpt-oss rejects `tools`+`response_format` together | augment-static (ids the key can reach) | free tier 64–65K ctx / 32K out vs paid 131K/40K |
| mistral | `reasoning_effort` — only `"high"` and `"none"` have documented semantics (API enum advertises more; undefined) | none | 4xx-vs-ignore unknown for mid values (probe); `content` becomes a typed chunk list (`thinking`/`text`) when reasoning — **string-assuming parsers break** | augment-static (capabilities live, no reasoning flag) | **codestral-2508 is 128K, not 262144** — applied catalog over-promises 2× |
| xai | `reasoning_effort` (4.3: default `low`, has off; 4.5: `low\|medium\|high`, default `high`, no off); 4.20 = mode-by-slug | grok-4.5, grok-build-0.1, grok-4.20-…-reasoning | documented error combining `reasoning_effort` with `stop`/`presence_penalty`/`frequency_penalty`; else undocumented | augment-static (`context_length` + long-context threshold live) | max_output unpublished on every model; >200K prompt re-bills the whole request 2× |
| cohere | compat surface: `reasoning_effort` `"none"\|"high"` documented; native V2: `thinking:{"type":"disabled"}` — **not in the compat param list** | none | `medium`/`low` documented "not supported" (failure shape unknown); native `thinking` over compat likely dropped → silently stuck thinking-on (probe) | augment-static (native host listing only) | command-a-plus-05-2026 = 128K in / 64K out — half its siblings |
| openrouter | unified `reasoning:{effort \| max_tokens \| enabled \| exclude}`; live per-model `supported_efforts` + `default_effort` | per-underlying-model (live metadata says which) | `effort`+`max_tokens` mutually exclusive; `mode:"pro"` → **documented silent reroute to the *-pro router model**; effort outside `supported_efforts` unsettled | **drive-from-live** (no auth; context, pricing, supported_parameters, per-route /endpoints) | same slug routes to providers spanning 96K–1M context and 16K output floors |
| perplexity | agent API `reasoning:{"effort":...}` enum `minimal\|low\|medium\|high\|xhigh\|max`; per-model acceptance unpublished | unknown (all six) | **anthropic/* without `max_output_tokens` = 400 every request** (verbatim body documented); flat `reasoning_effort` spelling undocumented | augment-static (roster only — ids-only, no auth) | NO context window published for any model; nothing to account against |
| together | `reasoning:{"enabled":bool}` (sole documented key) + `reasoning_effort` per model (deepseek coerces `low`/`medium`→`high`) | moonshotai/Kimi-K2.7-Code (M3 IS toggleable — corrects prior refresh) | unknown-param reject-vs-drop unsettled (probe); M3 toggle spelling unconfirmed | augment-static (`context_length` live) | M3 served at 524,288 on Together vs 1M native; gpt-oss 128K |
| fireworks | `reasoning_effort` `low\|medium\|high` (guide); API schema advertises `xhigh`/`max`/`none`/ints — undefined | unknown 5 of 6 (probe); minimax-m3 doc-classified adaptive | `thinking` + `reasoning_effort` in one request = documented validation error; else unknown | augment-static (control plane: `supportsServerless`+`supportsTools`+pricing per account) | default `context_length_exceeded_behavior:"truncate"` silently shrinks the window — set `"error"` |

---

# Provider contracts

---

## 1. openai + openai-responses

*Verification: corrected after adversarial review — 7 challenge(s), 1 fatal/major; corrections are applied in the block.*

````yaml
provider: openai, openai-responses
endpoints:
  - url: https://api.openai.com/v1/chat/completions
    protocol: openai-chat
  - url: https://api.openai.com/v1/responses
    protocol: openai-responses
checked: 2026-07-19
sources:
  - https://developers.openai.com/api/docs/models
  - https://developers.openai.com/api/docs/guides/reasoning
  - https://developers.openai.com/api/docs/guides/latest-model
  - https://developers.openai.com/api/docs/models/gpt-5.6-terra
  - https://developers.openai.com/api/docs/models/gpt-5.6-sol
  - https://developers.openai.com/api/docs/models/gpt-5.6-luna
  - https://developers.openai.com/api/docs/models/gpt-5.3-codex
  - https://developers.openai.com/api/docs/models/gpt-5.4-mini
  - https://developers.openai.com/api/docs/models/gpt-5.4-nano
  - https://developers.openai.com/api/docs/models/gpt-5.5-pro (existence + effort set via search snippet of the official model page; page itself not fetched)
  - https://developers.openai.com/api/docs/api-reference/models
  - https://developers.openai.com/api/docs/api-reference/responses (FAILED - page >10MB, could not fetch; responses request-body field list not confirmed from the raw reference page itself)
  - https://developers.openai.com/api/docs/api-reference/responses/create (FAILED - page >10MB)
  - https://developers.openai.com/api/docs/api-reference/chat/create (FAILED - content truncated before the reasoning_effort parameter section; chat-side spelling confirmed from Azure docs + live error text instead)
  - https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create (FAILED - truncated mid parameter list, reasoning_effort section not reachable)
  - https://developers.openai.com/api/reference/resources/responses/streaming-events (partially loaded; truncated at response.reasoning_summary_part.added heading)
  - https://community.openai.com/t/gpt-5-6-chat-completion-reasoning-effort-bug-behavior-change/1386454 (re-opened by correction pass)
  - https://community.openai.com/t/request-for-compatibility-matrix-reasoning-effort-sampling-parameters-across-gpt-5-series/1371738
  - https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
  - https://pi.dev/models/openrouter/openai-gpt-5-6-luna-pro (OpenRouter mirror confirming the router-minted luna-pro slug)
reasoning:
  param: |
    Responses API (/v1/responses): top-level object "reasoning" with nested fields copied exactly: "reasoning": { "effort": "...", "mode": "...", "summary": "..." }. Official example verbatim (reasoning guide, "Using pro reasoning mode" section - NOT the latest-model guide): "reasoning": { "mode": "pro", "effort": "medium" }.
    Chat Completions (/v1/chat/completions): flat string parameter "reasoning_effort" (confirmed via Azure OpenAI reasoning doc code samples and a live OpenAI error message quoting the parameter on /v1/chat/completions; the official chat create reference page itself failed to load - see sources). No "reasoning" object and no mode field on Chat Completions.
  values: |
    Full documented value space for effort (reasoning guide): none, minimal, low, medium, high, xhigh, max - "Some models support only a subset of these values, so check the relevant model page before choosing a setting."
    gpt-5.6-terra / gpt-5.6-sol / gpt-5.6-luna: none, low, medium, high, xhigh, max (latest-model guide verbatim: "GPT-5.6 supports `none`, `low`, `medium`, `high`, `xhigh`, and `max`."; "minimal" NOT in the 5.6 set).
    gpt-5.4-mini / gpt-5.4-nano: "none (default), low, medium, high and xhigh" (model pages verbatim; no minimal, no max).
    gpt-5.3-codex: low, medium, high, xhigh (model page verbatim; no none, no minimal, no max).
    reasoning.mode (Responses API only, GPT-5.6 models only): "standard" and "pro".
    reasoning.summary: auto, concise, detailed (model-dependent).
  default_when_omitted: |
    gpt-5.6 (all three tiers): effort defaults to "medium" - guide verbatim: "If you omit it, GPT-5.6 defaults to `medium` in both standard and pro modes". reasoning.mode defaults to "standard".
    gpt-5.5: medium (reasoning guide).
    gpt-5.4-mini / gpt-5.4-nano: "none" is the default (model pages) - i.e. no reasoning unless requested.
    gpt-5.3-codex: unpublished (model page does not state a default; probe needed).
  unsupported_value_behavior: |
    http-400, error body: { "error": { "message": "Unsupported value: 'reasoning_effort' does not support '<value>' with this model. Supported values are: ...", "type": "invalid_request_error", "param": "reasoning_effort", "code": "unsupported_value" } } (shape observed on prior-generation models in community/GitHub reports; not restated on the current guide pages - the official reasoning guide documents no error semantics).
    Additional documented-in-the-wild 400 (exact message from OpenAI API via community thread): "Function tools with reasoning_effort are not supported for gpt-5.6-sol in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'."
  per_model:
    - id: gpt-5.6-terra
      mode: optional
      effort: "adjustable: [none, low, medium, high, xhigh, max]; plus reasoning.mode [standard|pro] on Responses only"
    - id: gpt-5.6-sol
      mode: optional
      effort: "adjustable: [none, low, medium, high, xhigh, max]; plus reasoning.mode [standard|pro] on Responses only; alias gpt-5.6 routes here"
    - id: gpt-5.6-luna
      mode: optional
      effort: "adjustable: [none, low, medium, high, xhigh, max]; plus reasoning.mode [standard|pro] on Responses only"
    - id: gpt-5.3-codex
      mode: always-on
      effort: "adjustable: [low, medium, high, xhigh] (no 'none' value published - cannot be turned off); default unpublished"
    - id: gpt-5.4-mini
      mode: optional
      effort: "adjustable: [none, low, medium, high, xhigh]; default none; NOTE: raising effort above none while sending function tools on /v1/chat/completions is expected to 400 (restriction started with gpt-5.4 - see tools_and_streaming)"
    - id: gpt-5.4-nano
      mode: optional
      effort: "adjustable: [none, low, medium, high, xhigh]; default none; same chat-completions tools+effort restriction as gpt-5.4-mini"
  silent_substitutions: |
    none triggered by reasoning flags. There is NO separate gpt-5.6 Pro slug on api.openai.com - latest-model guide verbatim: "To use pro mode, keep your selected GPT-5.6 model and set `reasoning.mode` to `pro`" ... "do not switch to a separate Pro model slug." Pro mode is a Responses API execution mode; guide verbatim: "it increases latency and aggregates the tokens from that work in reported usage. Those tokens are billed at the selected model's standard token rates." (Same per-token price, more tokens.)
    Caveat: the no-separate-Pro-slug rule is 5.6-specific - the 5.5 generation DOES have a real separate slug gpt-5.5-pro (official model page exists; Responses API only; effort medium/high (default)/xhigh). Do not generalize either way across generations.
    Alias routing (not reasoning-triggered): latest-model guide verbatim: "The `gpt-5.6` alias routes requests to `gpt-5.6-sol`".
  tools_and_streaming: |
    Reasoning tokens "still occupy space in the model's context window and are billed as output tokens" and count toward max_output_tokens (Responses) - a low max_output_tokens with high/xhigh/max effort can exhaust the budget before any visible output.
    Streaming: reasoning summaries stream as semantic events response.reasoning_summary_text.delta / response.reasoning_summary_text.done and response.reasoning_summary_part.added (partially confirmed - streaming-events reference truncated mid-page). Raw reasoning text as "reasoning_text" content parts is UNCONFIRMED - not mentioned on the reasoning guide, and the streaming-events reference truncated before any such section; probe filed. Do not hardcode the event name.
    Tools: reasoning items are output items interleaved with function calls; guide verbatim: "We highly recommend you pass back any reasoning items returned with the last function call (in addition to the output of your function)." In stateless mode (store: false) reasoning items "include an `encrypted_content` property by default"; include=["reasoning.encrypted_content"] is accepted as legacy but no longer required.
    CRITICAL chat-completions restriction: on gpt-5.4 AND gpt-5.6 reasoning models, combining function tools with reasoning_effort != none on /v1/chat/completions is rejected with a 400 telling you to use /v1/responses. Community thread reply verbatim: "Blocking developer functions on reasoning models started with gpt-5.4." On gpt-5.4-mini/nano the default effort is none, so the 400 only bites when effort is explicitly raised; on gpt-5.6 (default medium) it bites by default. (Correction: an earlier draft scoped this to 5.6 only - that was wrong; both cited sources converge on 5.4 already having the restriction.)
  compat_passthrough: |
    openai-responses (/v1/responses): reasoning object with effort+mode+summary - passes (this is the canonical surface; only surface where mode=pro works).
    openai-chat (/v1/chat/completions): reasoning_effort passes as a flat string for effort only; reasoning.mode does NOT exist here (pro unreachable from chat); tools+reasoning_effort combination rejected on gpt-5.4 and gpt-5.6 reasoning models (http-400; restriction started with gpt-5.4). Sending the Responses-style nested "reasoning" object to chat completions: unknown (expected 400 unrecognized argument; probe).
discovery:
  endpoint: https://api.openai.com/v1/models (GET; also GET /v1/models/{model})
  auth: bearer
  metadata: ids-only (fields per model object: id, object="model", created, owned_by - no context window, no capabilities, no pricing)
  account_scoped: "unknown: docs do not state whether the list varies per account/project; project-level model restrictions exist in the platform so assume yes for availability, but this is undocumented - probe"
  rate_limits_or_caching: undocumented
  verdict: augment-static
  refresh_strategy: on-config-open (list is ids-only, so use it solely to confirm availability / detect new+removed ids; all capability, context, and reasoning metadata must stay static from this research)
context:
  per_model:
    - id: gpt-5.6-terra
      context: 1,050,000 (no tier splits documented)
      max_output: 128,000
      long_context_pricing: "boundary: prompts with >272K input tokens are priced at 2x input and 1.5x output for the entire request; base $2.50/1M in, $0.25/1M cached in, $15.00/1M out"
      overpromise_risk: "no on context size; yes on cost: any account crossing 272K input silently pays 2x/1.5x"
    - id: gpt-5.6-sol
      context: 1,050,000 (no tier splits documented)
      max_output: 128,000
      long_context_pricing: "boundary: >272K input tokens -> 2x input / 1.5x output; base $5.00/1M in, $0.50/1M cached in, $30.00/1M out"
      overpromise_risk: "no on context size; yes on cost past 272K input (same boundary)"
    - id: gpt-5.6-luna
      context: 1,050,000 (no tier splits documented)
      max_output: 128,000
      long_context_pricing: "boundary: >272K input tokens -> 2x input / 1.5x output for the complete request; base $1.00/1M in, $0.10/1M cached in, $6.00/1M out"
      overpromise_risk: "no on context size; yes on cost past 272K input"
    - id: gpt-5.3-codex
      context: 400,000 (no tier splits documented)
      max_output: 128,000
      long_context_pricing: "flat ($1.75/1M in, $0.175/1M cached in, $14.00/1M out; no boundary mentioned)"
      overpromise_risk: no
    - id: gpt-5.4-mini
      context: 400,000 (no tier splits documented)
      max_output: 128,000
      long_context_pricing: "flat ($0.75/1M in, $0.075/1M cached in, $4.50/1M out); regional data-residency endpoints +10% uplift"
      overpromise_risk: no
    - id: gpt-5.4-nano
      context: 400,000 (no tier splits documented)
      max_output: 128,000
      long_context_pricing: "flat ($0.20/1M in, $0.02/1M cached in, $1.25/1M out); regional processing +10% uplift"
      overpromise_risk: no
  token_counting_endpoint: none
probes_needed:
  - question: Does gpt-5.6-terra on /v1/chat/completions reject function tools + reasoning_effort (confirm status code and error body shape for the exact model Sylliptor ships)?
    probe: 'curl https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.6-terra\", \"reasoning_effort\": \"medium\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"noop\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}]}"'
  - question: Does gpt-5.4-mini on /v1/chat/completions reject function tools + reasoning_effort raised above its default of none (docs + thread say the restriction started with 5.4 - confirm the live error shape)?
    probe: 'curl https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.4-mini\", \"reasoning_effort\": \"low\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"noop\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}]}"'
  - question: What is gpt-5.3-codex's default reasoning effort when omitted (unpublished), and is 'none' rejected with code unsupported_value?
    probe: 'curl https://api.openai.com/v1/responses -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.3-codex\", \"reasoning\": {\"effort\": \"none\"}, \"input\": \"hi\"}"'
  - question: Is reasoning.mode rejected (and with what error) on non-5.6 models like gpt-5.4-mini?
    probe: 'curl https://api.openai.com/v1/responses -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.4-mini\", \"reasoning\": {\"mode\": \"pro\"}, \"input\": \"hi\"}"'
  - question: Is 'minimal' rejected on gpt-5.6 with code unsupported_value listing the supported set (settles the exact runtime error shape on current models)?
    probe: 'curl https://api.openai.com/v1/responses -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.6-luna\", \"reasoning\": {\"effort\": \"minimal\"}, \"input\": \"hi\"}"'
  - question: Is GET /v1/models account/project-scoped (do two accounts of different tiers or with project model restrictions see different lists), and does it include gpt-5.6-* and gpt-5.3-codex for this account?
    probe: 'curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"'
  - question: Does the Responses-style nested reasoning object 400 on /v1/chat/completions (compat cross-contamination check)?
    probe: 'curl https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.6-luna\", \"reasoning\": {\"effort\": \"low\"}, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}"'
  - question: Does reasoning.summary still 400 for unverified organizations on gpt-5.6 (verified-org gating)?
    probe: 'curl https://api.openai.com/v1/responses -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.6-luna\", \"reasoning\": {\"effort\": \"low\", \"summary\": \"auto\"}, \"input\": \"hi\"}"'
  - question: What content-part / event type does raw reasoning text stream as on /v1/responses ("reasoning_text" is unconfirmed - the reasoning guide does not name it and the streaming-events reference truncated)?
    probe: 'curl -N https://api.openai.com/v1/responses -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d "{\"model\": \"gpt-5.6-luna\", \"reasoning\": {\"effort\": \"low\"}, \"stream\": true, \"input\": \"hi\"}" | head -100  # inspect event type names for reasoning output items'
conflicts:
  - "openrouter.ai lists router-minted -pro slugs for gpt-5.6: gpt-5.6-terra-pro (openrouter.ai/openai/gpt-5.6-terra-pro) and gpt-5.6-luna-pro (confirmed via pi.dev/models/openrouter/openai-gpt-5-6-luna-pro) are CONFIRMED listings; a sol-pro slug is UNVERIFIED (no listing found - do not assert it exists). Official developers.openai.com/api/docs/guides/latest-model says there is NO separate Pro slug for 5.6 - pro is reasoning.mode: \"pro\" on the base 5.6 model via the Responses API. OpenRouter's -pro ids are router wrappers (same model served with mode pro) and must never be sent to api.openai.com."
  - "learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning (older value matrix) says xhigh only on gpt-5.1-codex-max and minimal only on original GPT-5; developers.openai.com reasoning guide + model pages say xhigh is standard on 5.3/5.4/5.6 and 'max' exists on 5.6. Azure page reflects an older cycle; official OpenAI pages win for api.openai.com."
notes: |
  CORRECTION (was listed as a conflict; it is not one): the tools+reasoning_effort chat-completions restriction is NOT new in gpt-5.6. Re-reading community thread 1386454: the OP only reports hitting the error after upgrading 5.4 -> 5.6 and never claims 5.4 allowed the combination; the first reply states verbatim "Blocking developer functions on reasoning models started with gpt-5.4." This converges with the TradingAgents gpt-5.4 issue rather than conflicting with it. The contract now scopes the restriction to gpt-5.4 AND gpt-5.6; a 5.4 live probe is filed for the exact error shape.
  gpt-5.5 and gpt-5.5-pro both exist as official models: gpt-5.5 defaults to medium effort (reasoning guide); gpt-5.5-pro has its own model page (developers.openai.com/api/docs/models/gpt-5.5-pro - Responses API only, effort medium/high (default)/xhigh, no cached-input discount, +10% regional uplift; details from official-page search snippet, page not fetched). Neither is in Sylliptor's applied lineup. Note the generational asymmetry: 5.5 has a real -pro slug, 5.6 replaces it with reasoning.mode=pro.
  The chat-completions spelling reasoning_effort could not be confirmed from the official chat create reference (page too large / truncated on every attempt - marked FAILED); it is confirmed by (a) the live OpenAI error string naming reasoning_effort on /v1/chat/completions for gpt-5.6-sol, (b) Azure OpenAI reasoning doc code samples, (c) an OpenAI-docs search snippet listing none/low/medium/high/xhigh/max for 5.6 chat.
  Verbosity: the latest-model guide documents `text.verbosity` (low/medium/high) - guide verbatim: "use `text.verbosity` to set the default level of detail". That is the Responses API shape; the chat-completions spelling for this family is unverified. (A verifier challenge claimed no verbosity mention exists in the docs - refuted by direct fetch of the latest-model guide; the reasoning guide indeed does not mention it.)
  The pro-mode JSON example "reasoning": {"mode": "pro", "effort": "medium"} lives on the REASONING guide ("Using pro reasoning mode" section); the latest-model guide has no JSON example (attribution corrected).
  Reasoning guide states reasoning models work better with the Responses API; for Sylliptor (always sends tools) the Responses API is effectively mandatory for any 5.4/5.6 reasoning-enabled call. Pro mode is same per-token price, more tokens. Knowledge cutoff gpt-5.6-luna: 2026-02-16. Snapshot aliases seen: gpt-5.4-mini-2026-03-17, gpt-5.4-nano-2026-03-17; alias gpt-5.6 -> gpt-5.6-sol.
````

### Gap-fill addendum (post-verification)

A1/A7 (Responses request body + reasoning object shape) — CONFIRMED from the official OpenAI Python SDK type stubs, which are the canonical machine-readable schema (the HTML reference page https://developers.openai.com/api/docs/api-reference/responses/create still fails: >10MB, maxContentLength exceeded).

Source A: https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/response_create_params.py — POST /v1/responses top-level fields: model, input, include, reasoning (Optional[Reasoning]; docstring verbatim: "**gpt-5 and o-series models only** Configuration options for reasoning models."), max_output_tokens, background, context_management, conversation, instructions, max_tool_calls, metadata, moderation, parallel_tool_calls, previous_response_id, prompt, prompt_cache_key, prompt_cache_options, prompt_cache_retention, safety_identifier, service_tier, store, stream, stream_options, temperature, text, tool_choice, tools, top_logprobs, top_p, truncation, user.

Source B: https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/reasoning.py — the reasoning object has FIVE fields, not three. Verbatim:
- effort: Optional[ReasoningEffort] — "Constrains effort on reasoning for reasoning models. Currently supported values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`."
- summary: Optional[Literal["auto","concise","detailed"]] — "A summary of the reasoning performed by the model. `concise` is supported for `computer-use-preview` models and all reasoning models after `gpt-5`."
- mode: Union[str, Literal["standard","pro"], None] — "Controls the reasoning execution mode for the request." (confirms mode is a real Responses field; the {standard,pro} enum matches the guide, and the base type is str so it is open-ended.)
- context: Optional[Literal["auto","current_turn","all_turns"]] — "Controls which reasoning items are rendered back to the model on later turns." (NEW — not in current YAML.)
- generate_summary: Optional[Literal["auto","concise","detailed"]] — "Deprecated: use `summary` instead." (NEW — deprecated alias for summary.)
So the guide-derived {effort, mode, summary} shape is correct and is now confirmed against the reference-equivalent schema; it should be extended with `context` and the deprecated `generate_summary` alias.

Source C: https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/reasoning_effort.py — verbatim: ReasoningEffort: TypeAlias = Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]]. Matches the reasoning guide's documented full value space exactly.

A4 (gpt-5.5-pro per_model) — CONFIRMED from https://developers.openai.com/api/docs/models/gpt-5.5-pro (page fetched successfully this pass). Verbatim: "Reasoning.effort supports: medium, high (default) and xhigh." Default is high. No `none` (reasoning is built-in/always-on, cannot be turned off), no low/minimal/max — a distinct effort set from every other current model. Verbatim surface: "GPT-5.5 Pro is available for Responses API requests, including through the Batch API." Snapshot/alias id: gpt-5.5-pro and gpt-5.5-pro-2026-04-23. The model page does not document a reasoning.mode field — pro behavior is baked into the slug (unlike gpt-5.6 where pro is a reasoning.mode value on the shared slug), consistent with the existing silent_substitutions caveat that 5.5 has a real separate gpt-5.5-pro slug.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  param: |
    Responses API (/v1/responses): top-level object "reasoning". Shape now CONFIRMED against the canonical schema (OpenAI Python SDK type stub src/openai/types/shared/reasoning.py; the HTML reference page is unfetchable at >10MB). The object has FIVE fields, not three:
      "reasoning": {
        "effort": "...",           # Optional[none|minimal|low|medium|high|xhigh|max]
        "summary": "...",          # Optional[auto|concise|detailed]
        "mode": "...",             # Union[str, "standard"|"pro"] - "Controls the reasoning execution mode for the request." (base type is str; standard/pro are the documented values)
        "context": "...",          # Optional[auto|current_turn|all_turns] - "Controls which reasoning items are rendered back to the model on later turns."
        "generate_summary": "..."  # DEPRECATED alias for "summary" (auto|concise|detailed); do not emit, use "summary"
      }
    The guide-derived {effort, mode, summary} shape is confirmed correct; add the newly-confirmed "context" field and the deprecated "generate_summary" alias. reasoning field is documented "gpt-5 and o-series models only".
    Chat Completions (/v1/chat/completions): flat string parameter "reasoning_effort" (unchanged; no reasoning object, no mode/summary/context).
  values: |
    effort (canonical, SDK ReasoningEffort type alias verbatim): none, minimal, low, medium, high, xhigh, max. "Some models support only a subset of these values, so check the relevant model page before choosing a setting."
    gpt-5.6-terra / gpt-5.6-sol / gpt-5.6-luna: none, low, medium, high, xhigh, max (no minimal).
    gpt-5.5-pro: medium, high (default), xhigh only (model page verbatim: "Reasoning.effort supports: medium, high (default) and xhigh." - no none/low/minimal/max; reasoning is built-in/always-on).
    gpt-5.4-mini / gpt-5.4-nano: none (default), low, medium, high, xhigh (no minimal, no max).
    gpt-5.3-codex: low, medium, high, xhigh (no none/minimal/max).
    reasoning.mode (Responses API only; base type str, documented values): "standard" and "pro".
    reasoning.summary (and deprecated generate_summary): auto, concise, detailed (model-dependent; "concise" supported on computer-use-preview and all reasoning models after gpt-5).
    reasoning.context (Responses API only, SDK-confirmed): auto, current_turn, all_turns.
  per_model:
    - id: gpt-5.5-pro
      mode: always-on
      effort: "adjustable: [medium, high, xhigh]; default high; no 'none' (reasoning cannot be turned off); Responses API only (incl. Batch API); real separate slug (snapshot gpt-5.5-pro-2026-04-23), NOT a reasoning.mode of another model; no reasoning.mode field documented on the model page"
sources:
  - https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/response_create_params.py (CONFIRMED /v1/responses top-level request body field list incl. reasoning: Optional[Reasoning], via official SDK type stub - substitutes for the unfetchable >10MB HTML reference)
  - https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/reasoning.py (CONFIRMED reasoning object shape: effort, summary, mode[standard|pro], context[auto|current_turn|all_turns], generate_summary[deprecated])
  - https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/shared/reasoning_effort.py (CONFIRMED effort enum: none|minimal|low|medium|high|xhigh|max)
  - https://developers.openai.com/api/docs/models/gpt-5.5-pro (CONFIRMED - fetched this pass; effort medium/high(default)/xhigh, Responses API incl Batch, snapshot gpt-5.5-pro-2026-04-23)
````

---

## 2. anthropic (anthropic / -compat / -native)

*Verification: verified-clean — 5 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: anthropic (covers all 3 anthropic presets; single surface)
endpoints:
  - url: https://api.anthropic.com/v1/messages
    protocol: anthropic-messages
checked: 2026-07-19
sources:
  - https://platform.claude.com/docs/en/about-claude/models/overview.md
  - https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking.md
  - https://platform.claude.com/docs/en/build-with-claude/extended-thinking.md   # loaded OK, but fetch tool returned a synthesized digest, not raw page text; one claim from it conflicted with two raw pages (see conflicts)
  - https://platform.claude.com/docs/en/api/models/list
  - https://platform.claude.com/docs/en/build-with-claude/context-windows.md
  - https://platform.claude.com/docs/en/build-with-claude/token-counting.md
  - https://platform.claude.com/docs/en/about-claude/pricing.md
  - https://platform.claude.com/docs/en/api/rate-limits.md
  - https://platform.claude.com/docs/en/build-with-claude/effort.md
  - https://platform.claude.com/docs/en/api/errors.md
reasoning:
  param: >
    Top-level request field "thinking" (object). Exact shapes, copied character-for-character:
    adaptive: "thinking": {"type": "adaptive"} with optional "display": "summarized" or "omitted";
    manual extended: "thinking": {"type": "enabled", "budget_tokens": N};
    off: "thinking": {"type": "disabled"}.
    Reasoning DEPTH is a separate field: "output_config": {"effort": "..."} — nested inside output_config, NOT top-level, NOT "reasoning_effort", NOT "reasoning.effort". No beta header for either (adaptive-thinking.md: "No beta header is required"; effort.md: "no beta header required").
  values: >
    thinking.type: "adaptive" | "enabled" | "disabled" (availability differs per model — see per_model).
    thinking.display: "summarized" | "omitted" ("display" is invalid together with type "disabled" — adaptive-thinking.md).
    thinking.budget_tokens: integer, must be strictly less than max_tokens (400 invalid_request_error otherwise); exception: with interleaved thinking budget_tokens may exceed max_tokens (extended-thinking digest). Minimum value not confirmed this session (docs digest did not state the 1024 floor — verify before hardcoding).
    output_config.effort: "low" | "medium" | "high" | "xhigh" | "max".
    "max" available on: claude-fable-5, claude-opus-4-8, claude-opus-4-7, claude-sonnet-5 (+ opus-4-6/sonnet-4-6).
    "xhigh" available on: claude-fable-5, claude-opus-4-8, claude-opus-4-7, claude-sonnet-5 ONLY (not opus-4-6/sonnet-4-6, not haiku-4-5).
    effort NOT supported at all on claude-haiku-4-5 (absent from effort.md supported-models list; failure mode unpublished — probe).
    Effort default is "high" everywhere; "Setting effort to \"high\" produces exactly the same behavior as omitting the effort parameter entirely" (effort.md).
  default_when_omitted: >
    Per model when "thinking" field is absent:
    claude-sonnet-5: adaptive thinking ON by default ("adaptive thinking is on by default; pass thinking: {type: \"disabled\"} to turn it off").
    claude-fable-5: adaptive always-on (cannot be turned off).
    claude-opus-4-8: thinking OFF ("Thinking is off unless you explicitly set thinking: {type: \"adaptive\"}").
    claude-opus-4-7: thinking OFF (same wording as 4.8).
    claude-haiku-4-5: no thinking (manual "enabled"+budget_tokens is the only on-mode).
    thinking.display default: "omitted" on fable-5/sonnet-5/opus-4-8/opus-4-7 (thinking blocks arrive with EMPTY thinking text; signature still present); "summarized" on opus-4-6/sonnet-4-6 and earlier Claude 4.
  unsupported_value_behavior: >
    http-400 + body {"type": "error", "error": {"type": "invalid_request_error", "message": "..."}, "request_id": "req_..."} (errors.md canonical shape).
    Documented 400s: thinking {"type": "enabled", "budget_tokens": N} on opus-4-8 / opus-4-7 / sonnet-5 / fable-5 ("rejected with a 400 error", adaptive-thinking.md; effort.md: "not supported and returns a 400 error");
    thinking {"type": "disabled"} on fable-5 "is not supported"/"is rejected" (status code not printed in docs — assume 400, probe for exact body);
    budget_tokens >= max_tokens on haiku-4-5 → 400 invalid_request_error;
    modified/dropped thinking blocks in latest assistant msg → 400, message contains "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. These blocks must remain as they were in the original response." (errors.md verbatim);
    prefilled last assistant message on fable-5/opus-4-8/opus-4-7/opus-4-6/sonnet-4-6 → 400 "Prefilling assistant messages is not supported for this model." (errors.md; NOTE list omits sonnet-5 — see conflicts);
    non-default temperature/top_p/top_k on fable-5/opus-4-8/opus-4-7/sonnet-5 → 400 "regardless of whether thinking is active" (adaptive-thinking.md Validation changes).
    adaptive on haiku-4-5: NOT documented either way — probe.
  per_model:
    - id: claude-sonnet-5
      mode: optional            # adaptive ON by default; "disabled" accepted; manual "enabled" 400s
      effort: "adjustable: [low, medium, high, xhigh, max] (default high)"
    - id: claude-opus-4-8
      mode: optional            # OFF unless adaptive set explicitly; "disabled" accepted; manual "enabled" 400s
      effort: "adjustable: [low, medium, high, xhigh, max] (default high)"
    - id: claude-fable-5
      mode: always-on           # adaptive always on; "disabled" rejected; "enabled"+budget_tokens 400s; omit param or send {"type":"adaptive"}
      effort: "adjustable: [low, medium, high, xhigh, max] (default high) — effort remains fully adjustable despite always-on thinking"
    - id: claude-haiku-4-5
      mode: optional            # via manual {"type":"enabled","budget_tokens":N} ONLY; adaptive NOT supported (overview table: Adaptive=No); no interleaved thinking (beta header "accepted on the Claude API but ignored")
      effort: n/a               # not in effort.md supported list; error behavior unpublished
    - id: claude-opus-4-7
      mode: optional            # OFF unless adaptive set; manual "enabled" 400s
      effort: "adjustable: [low, medium, high, xhigh, max] (default high)"
  silent_substitutions: >
    none — no reasoning flag triggers a model swap on this provider. Related documented silent behavior (speed, not model): requests to claude-opus-4-6 with speed:"fast" "run at standard speed and are billed at standard rates" (no error, silent speed downgrade; pricing.md/rate-limits.md). Fable 5 refusal fallbacks (server-side-fallback-2026-06-01 beta) can serve a response from claude-opus-4-8, but only when explicitly opted in via the "fallbacks" parameter — never triggered by a reasoning flag.
  tools_and_streaming: >
    SSE: thinking arrives as content_block_start (type "thinking") + content_block_delta with delta.type "thinking_delta", then "signature_delta" just before content_block_stop; text uses "text_delta". With display:"omitted", no thinking_delta events are emitted (only signature_delta) — looks like a long pause before first text.
    Interleaved thinking (thinking between tool calls) is AUTOMATIC in adaptive mode on fable-5/opus-4-8/opus-4-7/sonnet-5 (+4.6 family), no beta header; beta header "interleaved-thinking-2025-05-14" is only needed for manual mode on older models (accepted-but-ignored elsewhere; on haiku-4-5 interleaved is not supported at all).
    Tool loops: the thinking block accompanying a tool_use in the LAST assistant message MUST be passed back byte-identical (including empty-thinking blocks and their signature); editing/reordering/filtering → 400 (message quoted above). On fable-5, pass blocks back exactly as received; on model switch other models silently ignore foreign thinking blocks (but they still add input tokens).
    max_tokens: thinking tokens COUNT toward max_tokens ("Thinking tokens count toward max_tokens" adaptive-thinking.md; "The thinking budget tokens are a subset of your max_tokens parameter" context-windows.md); billed as output tokens; usage.output_tokens_details.thinking_tokens gives the reasoning share. High/xhigh/max effort can exhaust max_tokens → stop_reason "max_tokens". Exception: manual+interleaved mode allows budget_tokens > max_tokens.
    Prompt cache: switching between adaptive and enabled/disabled modes breaks message cache breakpoints (system/tools cache survives).
  compat_passthrough: n/a — single surface (anthropic-messages); the three Sylliptor anthropic presets all hit the same /v1/messages endpoint, so there is no compat layer to strip parameters.
discovery:
  endpoint: GET https://api.anthropic.com/v1/models   # per-model retrieve also exists at /v1/models/{model_id} per overview tip
  auth: api-key                  # x-api-key header + anthropic-version: 2023-06-01
  metadata: ids+context+capabilities   # per model: id, display_name, created_at, type, max_input_tokens ("Maximum input context window size in tokens"), max_tokens ("Maximum value for the max_tokens parameter"), capabilities{batch, citations, code_execution, context_management{clear_thinking_20251015, clear_tool_uses_20250919, compact_20260112}, effort{low, medium, high, xhigh, max, supported}, image_input, pdf_input, structured_outputs, thinking{supported, types{adaptive, enabled}}}. NO pricing in the response. Pagination: limit (default 20, 1-1000), after_id, before_id; response has first_id/last_id/has_more. Caveat: the docs example shows "max_input_tokens": 0 (placeholder) and created_at "May be set to an epoch value if the release date is unknown" — treat 0/epoch as unknown, not literal.
  account_scoped: "yes: model roster varies by org entitlement (claude-mythos-5 is Project Glasswing invite-only; claude-fable-5 unavailable to zero-data-retention orgs — 'Neither model is available under zero data retention'), so the list is the authority on what THIS key can call"
  rate_limits_or_caching: undocumented for /v1/models specifically (no dedicated RPM table, no cache headers documented)
  verdict: drive-from-live       # response carries exactly what the capability layer needs: context window, output cap, thinking-mode support (adaptive vs enabled flags), effort levels incl. xhigh — enough to derive reasoning_mode + context per model at runtime; keep only pricing as a static overlay
  refresh_strategy: on-config-open   # plus a daily cache for session start; roster/capabilities change on model launches, not per-request
context:
  per_model:
    - id: claude-sonnet-5
      context: 1000000 (no tier splits; 1M is the default — "you don't need a beta header"; overview table "1M tokens")
      max_output: 128000 (synchronous Messages API; up to 300000 on Message Batches with beta header output-300k-2026-03-24)
      long_context_pricing: flat — "A 900k-token request is billed at the same per-token rate as a 9k-token request"; intro pricing $2/$10 per MTok through 2026-08-31, then $3/$15
      overpromise_risk: no (all tiers get 1M; Start-tier ITPM 2,000,000 >= window)
    - id: claude-opus-4-8
      context: 1000000 (no tier splits, no beta header)
      max_output: 128000 (300000 on Batches via output-300k-2026-03-24)
      long_context_pricing: flat ($5/$25 per MTok across the full window)
      overpromise_risk: no
    - id: claude-fable-5
      context: 1000000 (no tier splits, no beta header; new tokenizer — same text is ~30% more tokens than pre-Opus-4.7 models, so 1M holds ~555k words vs ~750k on old-tokenizer models)
      max_output: 128000
      long_context_pricing: flat ($10/$50 per MTok)
      overpromise_risk: "yes: (a) Start-tier orgs have Fable 5 ITPM of only 500,000 — an uncached prompt over ~500k tokens cannot clear the per-minute rate limiter even though the window is 1M (Build 1.5M, Scale 4M); (b) zero-data-retention orgs cannot use the model at all (400 on every request); a hardcoded '1M, available' overpromises for both groups"
    - id: claude-haiku-4-5
      context: 200000 (no 1M option; no tier splits)
      max_output: 64000 (no Batches 300k extension — not in the output-300k model list)
      long_context_pricing: flat ($1/$5 per MTok)
      overpromise_risk: no (as long as the catalog does not copy the 1M figure from sibling presets)
    - id: claude-opus-4-7
      context: 1000000 (no tier splits, no beta header; new tokenizer as Fable 5/Opus 4.8)
      max_output: 128000 (300000 on Batches via output-300k-2026-03-24)
      long_context_pricing: flat ($5/$25 per MTok); fast mode is separate premium pricing ($30/$150) and is deprecated — removed 2026-07-24
      overpromise_risk: no for context; yes for fast-mode capability flags (deprecated, 5 days from removal)
  token_counting_endpoint: POST https://api.anthropic.com/v1/messages/count_tokens   # free; separate RPM pool (Start 2,000 / Build 4,000 / Scale 8,000); counts under the tokenizer of the model passed; estimate only
probes_needed:
  - question: Exact 400 error body (message text) when budget_tokens is sent to an adaptive-only model — needed so Sylliptor can pattern-match and self-heal
    probe: 'curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" -d "{\"model\":\"claude-sonnet-5\",\"max_tokens\":2048,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: What happens when thinking {"type":"adaptive"} is sent to claude-haiku-4-5 (docs say adaptive=No but never state the failure mode — 400 vs silently ignored)
    probe: 'curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" -d "{\"model\":\"claude-haiku-4-5\",\"max_tokens\":256,\"thinking\":{\"type\":\"adaptive\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Exact status + body for thinking {"type":"disabled"} on claude-fable-5 (docs say "not supported"/"rejected" without printing the code)
    probe: 'curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" -d "{\"model\":\"claude-fable-5\",\"max_tokens\":256,\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Behavior of output_config.effort on claude-haiku-4-5 (model absent from effort supported list; 400 vs silently ignored is unpublished)
    probe: 'curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" -d "{\"model\":\"claude-haiku-4-5\",\"max_tokens\":256,\"output_config\":{\"effort\":\"low\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Real max_input_tokens/max_tokens/capabilities values and org-scoped roster from live discovery (docs example shows placeholder 0s; also confirms whether this key sees fable-5/mythos-5)
    probe: 'curl -s "https://api.anthropic.com/v1/models?limit=1000" -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"'
  - question: Does claude-sonnet-5 reject prefilled last-assistant messages (errors.md prefill-unsupported list omits Sonnet 5 while including Opus 4.6+ and Fable 5)
    probe: 'curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" -d "{\"model\":\"claude-sonnet-5\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Say A or B\"},{\"role\":\"assistant\",\"content\":\"A\"}]}"'
conflicts:
  - "extended-thinking.md (fetch-tool digest) claimed thinking tokens 'do NOT count toward max_tokens'; adaptive-thinking.md ('Thinking tokens count toward max_tokens') and context-windows.md ('The thinking budget tokens are a subset of your max_tokens parameter') say they DO count. Recorded; the two raw pages agree with each other, and the digest likely garbled the narrow interleaved-mode exception (budget_tokens may exceed max_tokens under interleaved thinking)."
  - "errors.md prefill-unsupported list = {Fable 5, Mythos 5, Mythos Preview, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 4.6} — claude-sonnet-5 is absent, while every other current 4.7+/5-gen model is listed; no other opened page states Sonnet 5's prefill behavior either way. Recorded, not resolved (probe above)."
  - "api/models/list example response shows \"max_input_tokens\": 0 and \"max_tokens\": 0 while the field descriptions define them as the real context window / output cap — example placeholders vs described semantics; live values need the discovery probe."
notes: >
  (1) The two reasoning knobs are ORTHOGONAL: thinking (mode) and output_config.effort (depth); effort works even with thinking disabled and controls total spend incl. tool calls. Sylliptor's reasoning_mode should map: fable-5 -> always_on (never emit thinking:disabled, never budget_tokens); sonnet-5 -> default_on (emit disabled to turn off); opus-4-8/4-7 -> default_off (must emit {"type":"adaptive"} to turn on); haiku-4-5 -> manual-only ({"type":"enabled","budget_tokens":N<max_tokens}), no effort, no adaptive, no interleaved.
  (2) Sampling params: fable-5/opus-4-8/opus-4-7/sonnet-5 reject NON-DEFAULT temperature/top_p/top_k with 400 on every request — a generic CLI that always sends temperature will break on all four; haiku-4-5 still accepts them.
  (3) Tokenizer split matters for context accounting: fable-5/opus-4-8(implied 4.7-tokenizer family)/opus-4-7/sonnet-5 use the new tokenizer (~30% more tokens for the same text — official wording: "The same input text produces approximately 30 percent more tokens than on earlier models"); haiku-4-5 uses the previous tokenizer. Never reuse counts across the boundary; count_tokens counts under the tokenizer of the model passed.
  (4) 1M context is standard for every 1M model — no beta header, no tier gating, flat pricing across the window. The legacy beta header "context-1m-2025-08-07" still exists in the header enum but is not needed for any current model.
  (5) Context-overflow semantics: input alone > window -> 400 "prompt is too long" on every model; input+max_tokens > window on 4.5+ models -> request accepted, generation stops with stop_reason "model_context_window_exceeded".
  (6) inference_geo:"us" costs 1.1x on Opus 4.6+/Sonnet 4.6+ models; older models 400 on the parameter.
  (7) usage.output_tokens_details.thinking_tokens exposes billed reasoning tokens; useful for Sylliptor's HUD. Sonnet-5 tier note: haiku-4-5 also has context awareness (API-injected <budget:token_budget> tags) — no client action needed.
  (8) Rate limits: Fable 5 has its OWN (much lower) limit pool (Start 1,000 RPM / 500k ITPM / 100k OTPM); Sonnet 5 is separate from the Sonnet 4.x bucket; all Opus 4.x share one combined bucket. Cache reads don't count toward ITPM.
````

### Gap-fill addendum (post-verification)

Fetched raw markdown via curl (WebFetch's summarizer was dropping both facts). Sources opened raw: https://platform.claude.com/docs/en/build-with-claude/extended-thinking.md , https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking.md , https://platform.claude.com/docs/en/build-with-claude/effort.md , https://platform.claude.com/docs/en/api/errors.md

(1) budget_tokens MINIMUM = 1,024 — CONFIRMED. extended-thinking.md (line 3595, "Budget optimization" bullet), verbatim: "The minimum budget is 1,024 tokens. Start at the minimum and increase the thinking budget incrementally to find the optimal range for your use case." So the 1024 floor is real; safe to hardcode. Note spelling "1,024" (with comma) in the raw page.

(2) claude-haiku-4-5 EFFORT — support status now CONFIRMED UNSUPPORTED (was "probe"). effort.md carries an explicit enumeration (the <Note> after the intro), verbatim: "The effort parameter is supported by Claude Fable 5, Claude Mythos 5, Claude Opus 4.8, Claude Mythos Preview, Claude Opus 4.7, Claude Opus 4.6, Claude Sonnet 5, Claude Sonnet 4.6, and Claude Opus 4.5." Haiku 4.5 is absent. (Tension noted: the prose one line up says effort is "available on all supported models" — the enumerated <Note> is the authoritative narrower list and excludes Haiku.) The explicit "returns a 400 error" language in effort.md applies only to manual budget_tokens on Opus 4.8 / Sonnet 5, NOT to effort-on-haiku. So the exact HTTP failure body when effort IS sent to haiku-4-5 is still unpublished.

(3) claude-haiku-4-5 ADAPTIVE — support status now CONFIRMED UNSUPPORTED (was "not documented either way"). adaptive-thinking.md has a "## Supported models" section headed verbatim "Adaptive thinking is supported on the following models:" enumerating Fable 5, Mythos 5, Mythos Preview, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6 (each with per-model rules). Haiku 4.5 is absent from that list. Corroborated by extended-thinking.md's "condensed comparison" table (line 3518): Claude Haiku 4.5 | budget_tokens: Supported | Thinking output: Summarized | Interleaved thinking: Not supported | Block preservation: Last turn only. Both the top supported-models table and this table list Haiku's only on-mode as manual budget_tokens (Recommended column = N/A). So on haiku-4-5 the sole thinking on-mode is {type:"enabled",budget_tokens:N} (>=1024, <max_tokens); thinking output is Summarized (not omitted-empty like Fable/Opus4.8); interleaved thinking is not supported; only the last assistant turn's thinking is kept in context. The exact rejection body for thinking:{type:"adaptive"} on haiku is still not printed in docs.

Net: both A2/A3 facts (1024 floor; haiku effort+adaptive both UNSUPPORTED) are now doc-confirmed. Only the runtime failure-body shape (400 vs silent-ignore, and exact message string) for effort/adaptive on haiku remains unpublished and needs the probe.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  values: >
    thinking.budget_tokens: integer. MINIMUM = 1024 (extended-thinking.md, "Budget optimization" bullet, verbatim "The minimum budget is 1,024 tokens"). Must also be strictly less than max_tokens (400 invalid_request_error otherwise); exception: with interleaved thinking budget_tokens may exceed max_tokens (limit becomes the full context window). budget_tokens cannot be combined with max_tokens:0 (cache pre-warming) since it must be < max_tokens.
    output_config.effort: "low" | "medium" | "high" | "xhigh" | "max".
    effort supported-models enumeration (effort.md <Note>, verbatim): Claude Fable 5, Claude Mythos 5, Claude Opus 4.8, Claude Mythos Preview, Claude Opus 4.7, Claude Opus 4.6, Claude Sonnet 5, Claude Sonnet 4.6, Claude Opus 4.5. claude-haiku-4-5 is ABSENT — effort is NOT supported on haiku-4-5 (confirmed by omission; exact failure body still unpublished — see unresolved). (Doc tension: intro prose says "available on all supported models" but the enumerated Note is authoritative.)
    Effort default is "high"; setting effort "high" == omitting the parameter.
  per_model:
    - id: claude-haiku-4-5
      mode: optional            # ONLY on-mode is manual {"type":"enabled","budget_tokens":N} with N>=1024 and N<max_tokens. Adaptive thinking CONFIRMED unsupported: haiku-4-5 is absent from adaptive-thinking.md "Adaptive thinking is supported on the following models" enumeration. Thinking output = Summarized (NOT omitted-empty). Interleaved thinking not supported (beta header accepted-but-ignored). Block preservation = last assistant turn only.
      effort: n/a               # NOT in effort.md supported-models enumeration; effort unsupported. Exact failure body (400 vs silent-ignore) unpublished — probe.
  unsupported_value_behavior: >
    Additions/corrections vs prior text:
    thinking.budget_tokens below 1024 → behavior unpublished (docs state 1024 is the minimum but do not print the sub-minimum rejection body — probe if needed).
    effort param sent to claude-haiku-4-5 → unsupported (haiku absent from effort.md enumeration); exact HTTP status + body NOT documented — probe. The effort.md "returns a 400 error" wording applies only to manual budget_tokens on Opus 4.8 / Sonnet 5, not to effort-on-haiku.
    thinking:{"type":"adaptive"} sent to claude-haiku-4-5 → unsupported (haiku absent from adaptive-thinking.md supported-models enumeration); exact rejection body NOT documented — probe.
````

**Still unresolved (needs a live key):**

````
Runtime failure-body for effort/adaptive on claude-haiku-4-5 is unpublished (docs confirm both are UNSUPPORTED by enumeration, but never print the HTTP status/message). No API key in this research env, so a maintainer must run these two isolating probes (do NOT send both params together — that can't tell which one triggered the error):

# Probe A — effort on haiku
curl -sS -w '\nHTTP %{http_code}\n' https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5","max_tokens":64,"output_config":{"effort":"low"},"messages":[{"role":"user","content":"hi"}]}'

# Probe B — adaptive thinking on haiku
curl -sS -w '\nHTTP %{http_code}\n' https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5","max_tokens":64,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"}]}'

For each: record HTTP status and, on 400, the exact error.message string; on 200, whether output_config.effort / thinking was silently ignored (inspect response for thinking blocks / usage). Also optional Probe C to confirm sub-1024 floor rejection: same as manual enabled but "thinking":{"type":"enabled","budget_tokens":512},"max_tokens":1024.
````

### Verifier notes (minor, not applied)

- **discovery.endpoint comment: 'per-model retrieve also exists at /v1/models/{model_id} per overview tip'** — Wrong attribution. The overview.md tip says only that you can 'query model capabilities and token limits programmatically with the Models API' and links the list endpoint; it does not document a /v1/models/{model_id} retrieve path, and the api/models/list page opened this session documents only GET /v1/models. The retrieve endpoint very likely exists, but neither cited source shows it. *Suggested: Cite the Models API retrieve reference page directly (or drop 'per overview tip' and mark the retrieve path as unverified-this-session).*
- **budget_tokens >= max_tokens on haiku-4-5 → '400 invalid_request_error' (unsupported_value_behavior, hazard 8, and claims_reverified item 6 all state the code as documented)** — The constraint ('budget_tokens must be set to a value less than max_tokens') is documented in extended-thinking.md, but the page does not print the status code or error type for violating it. The 400 invalid_request_error is an inference from errors.md's generic mapping, presented as if stated by the source. *Suggested: Mark the status code as inferred-from-errors.md (or fold it into the probe list); keep the documented constraint wording.*
- **fable-5 overpromise_risk: 'zero-data-retention orgs cannot use the model at all (400 on every request)'** — The unavailability is documented ('Neither model is available under zero data retention', adaptive-thinking.md), but no opened page states the failure mode or status code a ZDR org receives. '400 on every request' is an unverified inference stated as fact. *Suggested: Keep the unavailability claim; hedge the '400' as expected-but-unverified or add a probe.*
- **discovery.account_scoped: 'yes: model roster varies by org entitlement ... so the list is the authority on what THIS key can call'** — No cited page explicitly states that GET /v1/models filters its roster by org entitlement. This is an inference stitched from the Mythos-5 invite-only note and the ZDR restriction (both confirmed real). It is a well-grounded inference — the refusals-and-fallback page even says allowed_fallback_models appears 'on the model's entry in the Models API' under a beta header, implying per-org variation — *Suggested: Label account_scoped as 'inferred (confirm via listed discovery probe)' instead of a bare 'yes'.*
- **Overall contract (all parameter spellings, values, defaults, error messages, context/output numbers, tier splits, discovery schema, tokenizer facts, rate limits, pricing, fallback semantics)** — No error found — recorded as a challenge slot only to document verification coverage. Every exact spelling checked character-for-character against raw pages matched: thinking shapes and display values, output_config.effort nesting and per-model xhigh/max availability, effort.md supported list (Haiku 4.5 absent), the verbatim thinking-block-modification and prefill 400 messages (Sonnet 5 genuinely  *Suggested: Add https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback.md to the sources list, since the silent_substitutions fallback claim rests on it.*

---

## 3. gemini (gemini / -compat / -native)

*Verification: corrected after adversarial review — 4 challenge(s), 1 fatal/major; corrections are applied in the block.*

````yaml
provider: gemini (x3 presets)
endpoints:
  - url: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent (+ :streamGenerateContent)
    protocol: native
  - url: https://generativelanguage.googleapis.com/v1beta/interactions
    protocol: native
  - url: https://generativelanguage.googleapis.com/v1beta/openai/
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://ai.google.dev/gemini-api/docs/thinking
  - https://ai.google.dev/gemini-api/docs/models   # top-level lineup only; per-model spec tables not in server-rendered HTML, used per-model pages instead
  - https://ai.google.dev/gemini-api/docs/openai
  - https://ai.google.dev/gemini-api/docs/thought-signatures   # (redirect stub -> /docs/thinking#signatures)
  - https://ai.google.dev/gemini-api/docs/function-calling
  - https://ai.google.dev/gemini-api/docs/pricing
  - https://ai.google.dev/api/generate-content   # ThinkingConfig section not visible in fetched portion; ThinkingLevel enum confirmed from /docs/generate-content/gemini-3 per-model table instead
  - https://ai.google.dev/api/models
  - https://ai.google.dev/api/tokens
  - https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
  - https://ai.google.dev/gemini-api/docs/gemini-3
  - https://ai.google.dev/gemini-api/docs/generate-content/gemini-3   # note page title says "Gemini Generate Content API (Legacy)"
  - https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures
  - https://ai.google.dev/gemini-api/docs/image-generation   # ADDED this correction pass; thinking-process section does NOT state a signature-missing 400, so image-gen enforcement stays unverified
  - https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash
  - https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
  - https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite
  - https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview
reasoning:
  param: >
    THREE surfaces, three spellings — do not mix. (1) generateContent REST JSON:
    "generationConfig": {"thinkingConfig": {"thinkingLevel": "<value>"}} (camelCase; legacy
    "thinkingBudget" integer still accepted for backward compat but MUST NOT be combined with
    thinkingLevel). (2) new Interactions API (POST /v1beta/interactions): "generation_config":
    {"thinking_level": "<value>"} snake_case, plus "thinking_summaries": "auto"|"none".
    (3) OpenAI-compat: "reasoning_effort" top-level, or extra_body {"google": {"thinking_config":
    {"thinking_level": "low", "include_thoughts": true}}}.
  values: >
    Native thinkingLevel/thinking_level: "minimal" | "low" | "medium" | "high".
    gemini-3.5-flash: minimal,low,medium,high. gemini-3-flash-preview: minimal,low,medium,high.
    gemini-3.1-flash-lite: minimal,low,medium,high (attributed to /docs/gemini-3 and
    /docs/generate-content/gemini-3; the /docs/thinking table has NO plain gemini-3.1-flash-lite
    row — it lists only the DISTINCT id gemini-3.1-flash-lite-image with the RESTRICTED set
    {minimal, high}, default minimal; nearby-id trap for anyone extending the catalog).
    gemini-3.1-pro-preview: low,medium,high ONLY — NO CONFLICT. All three official pages agree:
    /docs/generate-content/gemini-3 explicitly marks "minimal" as "Not supported" for 3.1 Pro in
    its per-model column (the generic value enum lists minimal, but the per-model column excludes
    it — the earlier "conflict" was reading the generic enum without the per-model annotation);
    /docs/thinking table row = gemini-3.1-pro-preview | On (high) | low, medium, high;
    OpenAI-compat mapping coerces minimal->low on Pro (consistent with no-minimal).
    OpenAI-compat reasoning_effort: "minimal"|"low"|"medium"|"high"|"none"; mapping: minimal->low on
    3.1 Pro, minimal->minimal on flash family; "none" disables thinking on 2.5 models ONLY (never
    2.5-pro or any 3.x). No value turns thinking off on any 3.x model; docs: "minimal" does not
    guarantee thinking is off.
  default_when_omitted: >
    Thinking is always on; defaults per model: gemini-3.5-flash = "medium" (changed from high at GA),
    gemini-3.1-pro-preview = "high" (dynamic), gemini-3-flash-preview = "high" (dynamic),
    gemini-3.1-flash-lite = "minimal".
  unsupported_value_behavior: >
    Documented: sending both thinking_level and legacy thinking_budget in one request = HTTP 400.
    Legacy thinkingBudget is DOCUMENTED as still supported for backward compat on 3.x
    (whats-new-gemini-3.5 FAQ: "thinking_budget is still supported for backward compatibility...
    Don't use both in the same request") — so budget-alone is not itself an error; only
    thinkingBudget:0-alone (which cannot disable always-on 3.x thinking) is genuinely undocumented.
    Missing thought signature on function-calling replay (3.x) = HTTP 400, message shape:
    "Function call `<Function Call>` in the `<index of contents array>` content block is missing a
    `thought_signature`." Behavior for an out-of-enum thinkingLevel string, thinkingBudget:0 sent
    alone to a 3.x model, or reasoning_effort:"none" against a 3.x model: NOT documented — probes below.
  per_model:
    - id: gemini-3.5-flash
      mode: always-on
      effort: "adjustable: [minimal, low, medium, high] (default medium)"
    - id: gemini-3.1-pro-preview
      mode: always-on
      effort: "adjustable: [low, medium, high] (default high, dynamic); minimal is documented 'Not supported' on 3.1 Pro across all three official pages"
    - id: gemini-3.1-flash-lite
      mode: always-on
      effort: "adjustable: [minimal, low, medium, high] (default minimal; minimal does not guarantee no thinking). NB distinct id gemini-3.1-flash-lite-image supports only {minimal, high}"
    - id: gemini-3-flash-preview
      mode: always-on
      effort: "adjustable: [minimal, low, medium, high] (default high, dynamic)"
  silent_substitutions: none documented (no reasoning-flag-triggered model swaps found on any of the three surfaces)
  tools_and_streaming: >
    Thought signatures are the load-bearing agentic mechanic on ALL 3.x models. generateContent:
    response parts carry "thoughtSignature" (camelCase) on functionCall parts (and sometimes text
    parts); you MUST echo the full unmodified history back. STRICTLY validated for function calling
    (HTTP 400 if missing, exact error string confirmed verbatim on
    /docs/generate-content/thought-signatures). Image generation: that page only CROSS-REFERENCES
    the image-generation guide for enforcement and does NOT state the 400 behavior itself; the
    image-generation guide's thinking-process section likewise does not state a signature-missing
    400 — so strict image-gen enforcement is NOT independently verified this session (treat as
    likely-but-unconfirmed). text/chat signatures not enforced but omission degrades quality.
    Parallel calls: signature attached ONLY to the first functionCall part; response order must be
    FC1+sig, FC2, FR1, FR2 — interleaving FC1+sig, FR1, FC2, FR2 = 400. Validation scope = all
    model functionCall turns after the most recent User message (newest-to-oldest scan). Streaming:
    signature may arrive in a part with an empty text content part. Bypass values for injected/custom
    FCs: "context_engineering_is_the_way_to_go" or "skip_thought_signature_validator". Interactions
    API: stateful mode (store: true + previous_interaction_id) manages signatures server-side;
    stateless mode requires resending all `thought` blocks verbatim; thought summaries via
    generation_config.thinking_summaries: "auto"|"none". gemini-3.5-flash additionally requires `id`
    and matching `name` on all FunctionResponse parts (new at GA); per whats-new, mismatched/omitted
    ids cause the model to return empty responses with finish_reason STOP (not necessarily a hard
    400). Thinking tokens are billed at the output rate ("thinking tokens included" on pricing page)
    and reported separately in usageMetadata.thoughtsTokenCount; whether they consume maxOutputTokens
    is NOT documented — probe. Temperature: keep at default 1.0 on all 3.x; lowering documented to
    cause looping/degradation.
  compat_passthrough: >
    OpenAI-compat (/v1beta/openai/): reasoning_effort PASSES (auto-mapped to thinking_level;
    minimal coerced to low on 3.1 Pro = silent coercion); google-specific extras pass via
    extra_body.google.thinking_config. Thought signatures round-trip as
    extra_content.google.thought_signature nested inside tool_calls — a non-standard extension;
    any OpenAI client/middleware that strips unknown fields silently drops signatures and the NEXT
    request 400s. Interactions API vs generateContent: same thinking levels, different field
    spellings (snake_case vs camelCase thinkingConfig); thinking_summaries exists only on
    Interactions; passthrough between the two surfaces: n/a (separate requests).
discovery:
  endpoint: GET https://generativelanguage.googleapis.com/v1beta/models (also GET /v1beta/models/{model}; OpenAI-compat GET /v1beta/openai/models)
  auth: api-key (?key=$GEMINI_API_KEY query param documented; header alternative not confirmed this session)
  metadata: ids+context+partial-capabilities — name, baseModelId, version, displayName, description, inputTokenLimit, outputTokenLimit, supportedGenerationMethods[], thinking (boolean), temperature/maxTemperature/topP/topK. NO thinking-level lists, NO defaults, NO pricing.
  account_scoped: "no per docs (listing not documented to vary by tier; free vs paid differs in rate limits, not in the model list) — unconfirmed live, see probe"
  rate_limits_or_caching: undocumented (no documented cache headers or listing-specific rate limits; pagination pageSize default 50, max 1000, nextPageToken)
  verdict: augment-static
  refresh_strategy: daily-cache (listing gives real inputTokenLimit/outputTokenLimit + thinking flag to validate the static catalog, but thinking LEVELS/defaults/signature rules must stay static; refresh on config-open acceptable too, do not block session start on it)
context:
  per_model:
    - id: gemini-3.5-flash
      context: 1,048,576 (no tier splits documented; same headline on free tier)
      max_output: 65,536
      long_context_pricing: flat ($1.50/M input, $9.00/M output incl. thinking tokens)
      overpromise_risk: "no (free tier gets same window; rate limits differ, not context)"
    - id: gemini-3.1-pro-preview
      context: 1,048,576 (no tier splits; pricing boundary is per-prompt size, not account tier)
      max_output: 65,536
      long_context_pricing: "boundary at 200k prompt tokens: input $2.00/M <=200k vs $4.00/M >200k; output $12.00/M <=200k vs $18.00/M >200k (thinking included)"
      overpromise_risk: "no on window size; yes on COST if Sylliptor fills the window blindly (>200k prompts double input rate, 1.5x output rate); no free tier for this model"
    - id: gemini-3.1-flash-lite
      context: 1,048,576 (no tier splits documented)
      max_output: 65,536
      long_context_pricing: flat ($0.25/M input text/image/video, $0.50/M audio; $1.50/M output incl. thinking)
      overpromise_risk: no
    - id: gemini-3-flash-preview
      context: 1,048,576 (no tier splits documented)
      max_output: 65,536
      long_context_pricing: flat ($0.50/M input text/image/video, $1.00/M audio; $3.00/M output incl. thinking)
      overpromise_risk: "no on window; note it is the PREVIEW alias listed under the gemini-3.5-flash model page — preview aliases can be re-pointed/retired"
  token_counting_endpoint: POST https://generativelanguage.googleapis.com/v1beta/{model=models/*}:countTokens (accepts contents[] or full generateContentRequest; returns totalTokens, cachedContentTokenCount, promptTokensDetails[])
probes_needed:
  - question: Legacy thinkingBudget is confirmed still supported for backward compat on 3.x (whats-new FAQ); but does thinkingBudget:0 sent ALONE to an always-on 3.x model on generateContent get accepted/ignored, silently clamped, or 400? Only the 0 value is genuinely undocumented (thinking cannot be disabled on 3.x).
    probe: 'curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=$GEMINI_API_KEY" -H "Content-Type: application/json" -d "{\"contents\":[{\"parts\":[{\"text\":\"hi\"}]}],\"generationConfig\":{\"thinkingConfig\":{\"thinkingBudget\":0}}}"'
  - question: What does reasoning_effort:"none" return against a 3.x model on the OpenAI-compat layer (error code + body shape, or silent coercion to minimal/low)? Docs say none only disables thinking on 2.5 models.
    probe: 'curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" -H "Authorization: Bearer $GEMINI_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"gemini-3.5-flash\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Belt-and-braces — confirm thinkingLevel:"minimal" actually 400s on gemini-3.1-pro-preview. Docs are NOT in conflict (all three pages mark minimal "Not supported" on Pro); this only verifies the documented behavior holds live.
    probe: 'curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key=$GEMINI_API_KEY" -H "Content-Type: application/json" -d "{\"contents\":[{\"parts\":[{\"text\":\"hi\"}]}],\"generationConfig\":{\"thinkingConfig\":{\"thinkingLevel\":\"minimal\"}}}"'
  - question: Do thinking tokens consume maxOutputTokens (empty candidate with finishReason MAX_TOKENS while thoughtsTokenCount > 0 would prove yes)?
    probe: 'curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=$GEMINI_API_KEY" -H "Content-Type: application/json" -d "{\"contents\":[{\"parts\":[{\"text\":\"What is the sum of the first 50 primes? Show reasoning.\"}]}],\"generationConfig\":{\"maxOutputTokens\":64,\"thinkingConfig\":{\"thinkingLevel\":\"high\"}}}"'
  - question: Does GET /v1beta/models differ between a free-tier and a billed key (account scoping of discovery)?
    probe: 'curl -s "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key=$GEMINI_API_KEY" | head -c 4000'
conflicts:
  - https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash lists versions as Stable gemini-3.5-flash / Preview gemini-3-flash-preview (a 3.0-named preview under a 3.5 stable) while registry sites (OpenRouter etc.) treat gemini-3-flash-preview as a separate earlier-generation model — a naming/lineage labeling discrepancy, not an API-behavior conflict. Recorded, not resolved.
notes: >
  CORRECTION PASS (2026-07-19): the adversarial verifier was RIGHT on all four challenges; the
  original two "conflicts" about thinking levels did not actually exist and have been removed.
  (1) gemini-3.1-pro-preview minimal: NO conflict. /docs/generate-content/gemini-3 marks "minimal"
  as "Not supported" for 3.1 Pro in its per-model column; the researcher misread the generic value
  enum without the per-model annotation. All three official pages agree Pro = {low, medium, high}.
  Deleted from conflicts; hazard reworded; probe 3 downgraded to a belt-and-braces live check;
  claims_reverified item now a clean confirmation. Operational safe-subset guidance (low|medium|high
  on Pro) was already correct, so no API-breaking output ever shipped from the error.
  (2) legacy thinking_budget on 3.5: NO conflict. whats-new-gemini-3.5 FAQ states verbatim
  "thinking_budget is still supported for backward compatibility... Don't use both in the same
  request." Budget-alone IS documented-supported; only thinkingBudget:0 semantics (0 cannot disable
  always-on 3.x thinking) remain undocumented — probe 1 narrowed to that.
  (3) flash-lite levels are attributed to /docs/gemini-3 and /docs/generate-content/gemini-3, NOT
  the /docs/thinking table, which has NO plain gemini-3.1-flash-lite row — it lists only
  gemini-3.1-flash-lite-image, a distinct id restricted to {minimal, high} (default minimal). Added
  as a nearby-id trap note.
  (4) thought-signature strict enforcement is confirmed for FUNCTION CALLING only; the
  thought-signatures page cross-references the image-generation guide for image-gen enforcement, and
  the image-generation guide's thinking-process section does not state the missing-signature 400 —
  image-gen enforcement softened to unverified.
  UNCHANGED context: Docs reorganized around the Interactions API (POST /v1beta/interactions,
  snake_case generation_config.thinking_level, thinking_summaries, store/previous_interaction_id for
  server-side signature management). generateContent is titled "(Legacy)" but fully supported and is
  what Sylliptor's presets use — keep camelCase thinkingConfig.thinkingLevel there. gemini-3.5-flash
  GA changed default thinking level high->medium and added a REQUIRED id + matching name on every
  FunctionResponse part (mismatch -> empty response, finish_reason STOP). Keep temperature at 1.0 on
  all 3.x. All four catalog models: 1,048,576 in / 65,536 out; knowledge cutoff January 2025.
  usageMetadata.thoughtsTokenCount reports thinking tokens separately; pricing bills them as output.
````

---

## 4. deepseek

*Verification: verified-clean — 1 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: deepseek (models: deepseek-v4-pro, deepseek-v4-flash)
endpoints:
  - url: https://api.deepseek.com
    protocol: openai-chat
  - url: https://api.deepseek.com/anthropic
    protocol: anthropic-messages
checked: 2026-07-19
sources:
  - https://api-docs.deepseek.com
  - https://api-docs.deepseek.com/guides/thinking_mode
  - https://api-docs.deepseek.com/guides/anthropic_api
  - https://api-docs.deepseek.com/api/list-models
  - https://api-docs.deepseek.com/api/create-chat-completion
  - https://api-docs.deepseek.com/quick_start/pricing
  - https://api-docs.deepseek.com/quick_start/error_codes
  - https://api-docs.deepseek.com/quick_start/rate_limit
  - https://api-docs.deepseek.com/quick_start/token_usage
reasoning:
  param: >-
    Two DISTINCT surfaces, different spellings.
    OpenAI-chat surface (https://api.deepseek.com): TWO separate request-body fields --
    (1) thinking = {"type":"enabled"} | {"type":"disabled"}  (in the python SDK this is
    injected via extra_body={"thinking":{"type":"enabled"}}, but it lands as a top-level
    request-body key "thinking");
    (2) reasoning_effort = "high" | "max"  (top-level request-body field, sibling of thinking,
    NOT nested inside thinking).
    Anthropic-messages surface (https://api.deepseek.com/anthropic): toggle via Anthropic-native
    thinking = {"type":"enabled"|"disabled"} (budget_tokens is accepted but IGNORED); effort via
    output_config = {"effort":"high"|"max"}. There is NO reasoning_effort field on the anthropic
    surface.
  values: >-
    thinking.type: exactly "enabled" or "disabled" (both surfaces).
    reasoning_effort (openai surface) / output_config.effort (anthropic surface): canonical set is
    "high" and "max". Compatibility aliases are silently coerced: "low"->"high", "medium"->"high",
    "xhigh"->"max". Same value set applies to both deepseek-v4-pro and deepseek-v4-flash (docs do
    not differentiate the two model families for reasoning values).
  default_when_omitted: >-
    thinking defaults to "enabled" (thinking is ON by default, both models, both surfaces) --
    verified: "the thinking toggle defaults to enabled". Effort default is context-dependent:
    "high" for regular requests; for some complex agent requests (docs name Claude Code, OpenCode)
    effort is AUTOMATICALLY set to "max". So an agentic CLI may be silently bumped to max.
  unsupported_value_behavior: >-
    For effort: out-of-canonical values in the documented compat set (low/medium/xhigh) are
    silently-coerced (mapped to high/max), NOT errored. For the temperature/top_p/presence_penalty/
    frequency_penalty params while thinking is enabled: silently-ignored ("will not trigger an error
    but will also have no effect"). For a genuinely invalid thinking.type or effort string outside
    all documented values: unknown -- docs only publish generic 400 "Invalid Format / Invalid request
    body format" and 422 "Invalid Parameters"; the JSON error body shape (error.message/type/code) is
    NOT documented. See probes_needed.
  per_model:
    - id: deepseek-v4-pro
      mode: optional
      effort: 'adjustable: [high, max] (aliases low/medium->high, xhigh->max); default high, auto->max for complex agent requests'
    - id: deepseek-v4-flash
      mode: optional
      effort: 'adjustable: [high, max] (aliases low/medium->high, xhigh->max); default high, auto->max for complex agent requests'
  silent_substitutions: >-
    On the openai-chat surface, sending model="deepseek-v4-pro"/"deepseek-v4-flash" does NOT swap
    models based on reasoning flags -- thinking just toggles CoT within the chosen model.
    On the ANTHROPIC surface, the MODEL NAME is silently remapped (independent of reasoning flags):
    Claude Opus model names -> deepseek-v4-pro; Claude Haiku/Sonnet model names -> deepseek-v4-flash;
    any UNRECOGNIZED model name -> falls back to deepseek-v4-flash (silent). Legacy deepseek-chat ->
    non-thinking deepseek-v4-flash, deepseek-reasoner -> thinking deepseek-v4-flash (until 2026/07/24).
  tools_and_streaming: >-
    Reasoning is exposed as reasoning_content, a sibling of content. In SSE streaming the delta carries
    chunk.choices[0].delta.reasoning_content (openai surface). Tool calling while thinking is supported,
    BUT when tools are used the prior-turn reasoning_content MUST be passed back in all subsequent
    requests (round-trip requirement); without tool calls, prior reasoning_content need not be
    re-sent. Interleaved thinking-with-tools: supported via that pass-back requirement. Whether
    reasoning tokens count against max output: unpublished (docs do not state; usage.completion_tokens
    from the response is the only authority).
  compat_passthrough: >-
    openai-chat surface: reasoning_effort + thinking{type} pass. anthropic-messages surface:
    thinking{type} passes (budget_tokens stripped/ignored), output_config{effort} passes; the
    openai-style reasoning_effort key is n/a here (unknown field, do not send it to /anthropic).
    Cross-surface hazard: budget_tokens sent to /anthropic is silently ignored (thinking still runs
    at default/auto effort), and a temperature sent with thinking on is silently ignored on both.
discovery:
  endpoint: https://api.deepseek.com/models
  auth: bearer
  metadata: ids-only
  account_scoped: 'no: /models returns a flat id list; response schema is {object:list, data:[{id, object:"model", owned_by}]} with no context/capability/pricing fields'
  rate_limits_or_caching: >-
    /models listing rate limits: undocumented. General API: DeepSeek documents NO RPM/TPM numbers and
    NO x-ratelimit-* headers; it publishes per-model CONCURRENCY limits instead (deepseek-v4-pro ~500
    concurrent, deepseek-v4-flash ~2500; exceeding -> HTTP 429). Concurrency is account/tier scoped
    (expandable via a capacity-expansion request).
  verdict: augment-static
  refresh_strategy: >-
    on-config-open (poll GET /models to detect id availability + catch the 2026/07/24 legacy
    deprecation), but KEEP context/max-output/reasoning-capability as a static hardcoded table since
    /models returns ids-only. daily-cache the id list is acceptable.
context:
  per_model:
    - id: deepseek-v4-pro
      context: 1M (1,000,000 tokens; no tier splits documented -- single flat window)
      max_output: 384K (384,000 tokens maximum)
      long_context_pricing: 'flat (single cache-hit/cache-miss/output rate across the whole 1M window; no context-length boundary documented). Rates per 1M tokens: cache-hit input $0.003625, cache-miss input $0.435, output $0.87'
      overpromise_risk: 'no (headline 1M applies to all accounts; only concurrency, not window size, is tier-scoped)'
    - id: deepseek-v4-flash
      context: 1M (1,000,000 tokens; no tier splits documented -- single flat window)
      max_output: 384K (384,000 tokens maximum)
      long_context_pricing: 'flat (no boundary). Rates per 1M tokens: cache-hit input $0.0028, cache-miss input $0.14, output $0.28'
      overpromise_risk: 'no'
  token_counting_endpoint: none (no /tokens or count_tokens API; DeepSeek ships an offline downloadable tokenizer zip + relies on response usage fields)
probes_needed:
  - question: Exact HTTP status + JSON error body when an invalid reasoning_effort/thinking.type value (outside high/max/enabled/disabled and outside the documented compat aliases) is sent to the openai-chat surface.
    probe: |
      curl -sS -w '\nHTTP %{http_code}\n' https://api.deepseek.com/chat/completions \
        -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"turbo","thinking":{"type":"enabled"}}'
  - question: What error a pinned legacy config gets AFTER 2026/07/24 15:59 UTC (model-not-found vs continued alias). Docs only say deprecated, not the post-deprecation error shape.
    probe: |
      curl -sS -w '\nHTTP %{http_code}\n' https://api.deepseek.com/chat/completions \
        -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}]}'
  - question: Does budget_tokens sent to the /anthropic surface truly no-op (thinking still runs) and does reasoning_content count against usage.completion_tokens / max output?
    probe: |
      curl -sS https://api.deepseek.com/anthropic/v1/messages \
        -H "x-api-key: $DEEPSEEK_API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d '{"model":"deepseek-v4-pro","max_tokens":1000,"thinking":{"type":"enabled","budget_tokens":64},"messages":[{"role":"user","content":"count the Rs in strawberry"}]}'
  - question: Whether GET /models is served on the /anthropic base too, and whether it ever returns legacy ids or account-specific entries.
    probe: |
      curl -sS https://api.deepseek.com/models -H "Authorization: Bearer $DEEPSEEK_API_KEY"
conflicts:
  - 'Default effort is dual-valued by design: thinking_mode guide says default effort is "high" for regular requests BUT is auto-set to "max" for complex agent requests (Claude Code / OpenCode). A generic agentic CLI cannot know which bucket it lands in from the docs alone -- recorded, not resolved.'
  - 'Surface asymmetry (recorded as a cross-surface inconsistency, not a doc contradiction): the openai-chat surface controls effort via top-level reasoning_effort, while the anthropic surface uses output_config.effort and ignores Anthropic budget_tokens -- two different spellings for the same capability on one provider.'
  - 'Registry/secondary sources (evolink.ai, docs.apiyi.com, models.dev-style listings) describe "three reasoning effort modes" for V4; official thinking_mode guide documents only two canonical values (high, max) plus coerced aliases -- official wins; the "three modes" framing likely counts disabled+high+max or the alias set.'
notes: >-
  A WebFetch pass over /api/create-chat-completion initially rendered reasoning_effort as NESTED
  inside thinking ({"thinking":{"type":...,"reasoning_effort":...}}); the VERBATIM code blocks on
  /guides/thinking_mode disprove this -- they are two separate top-level fields (reasoning_effort
  sibling of thinking, thinking injected via extra_body in the SDK). This was a fetch-model misread
  of the same official source, not a genuine source conflict. Runtime layer MUST emit them as
  separate keys on the openai surface. The docs' python multi-turn example even shows a stray missing
  comma between reasoning_effort and extra_body (a docs typo), reinforcing that they are distinct kwargs.
  Thinking mode disables temperature/top_p/presence_penalty/frequency_penalty (silently). max output
  384K coexists with a 1M context, so input+output must fit 1M. Anthropic surface path is
  /anthropic (Messages API mounted under it); model-name remapping there is silent and is the single
  biggest foot-gun.
````

### Verifier notes (minor, not applied)

- **discovery endpoint auth: bearer** — The cited list-models page (https://api-docs.deepseek.com/api/list-models) does not document the authentication method for GET /models; 'bearer' is an inference (consistent with DeepSeek's bearer auth used on the chat endpoints, so not wrong, but not sourced on the cited page). *Suggested: Mark discovery auth as inferred ('bearer, per DeepSeek's standard Authorization: Bearer scheme; not restated on the list-models page') rather than as a documented list-models attribute.*

---

## 5. qwen-intl / qwen-us / qwen-cn

*Verification: verified-clean — 2 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: qwen-intl / qwen-us / qwen-cn
endpoints:
  - url: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    protocol: openai-chat
  - url: https://dashscope-us.aliyuncs.com/compatible-mode/v1
    protocol: openai-chat
  - url: https://dashscope.aliyuncs.com/compatible-mode/v1
    protocol: openai-chat
  - url: https://dashscope-us.aliyuncs.com/apps/anthropic (also {WorkspaceId}.<region>.maas.aliyuncs.com/apps/anthropic for Singapore/Beijing/Hong Kong/Frankfurt/Tokyo)
    protocol: anthropic-messages
checked: 2026-07-19
sources:
  - https://www.alibabacloud.com/help/en/model-studio/models   # (FAILED-partial: page loads but model spec tables render client-side; only ids/regions/API-format flags visible — context/max-output columns NOT retrievable)
  - https://www.alibabacloud.com/help/en/model-studio/deep-thinking
  - https://www.alibabacloud.com/help/en/model-studio/qwen-coder
  - https://www.alibabacloud.com/help/en/model-studio/model-pricing
  - https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
  - https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
  - https://help.aliyun.com/zh/model-studio/models   # (FAILED-partial: same client-side table problem as EN page)
  - https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope
  - https://www.alibabacloud.com/blog/qwen3-7-the-agent-frontier_603154
  - https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages
  - https://www.alibabacloud.com/help/en/model-studio/what-is-qwen-llm   # (FAILED: 301 redirect to modelstudio.console.alibabacloud.com login console)
reasoning:
  param: "enable_thinking" (boolean) on the OpenAI-compatible and DashScope-native surfaces; companion "thinking_budget" (positive integer, caps reasoning tokens); "preserve_thinking" (boolean, carries prior-turn reasoning, specific Qwen3.7/Qwen3.6/Kimi models only). Wire placement — raw HTTP JSON: top-level field "enable_thinking": true; OpenAI Python SDK: must go via extra_body ("Since enable_thinking is not a standard OpenAI parameter, pass it in extra_body"); Node.js SDK: top-level. On the Anthropic-compatible surface (/apps/anthropic) the standard Anthropic thinking object {"type":"enabled","budget_tokens":N} is used instead; enable_thinking is NOT documented there. Note DashScope also documents "reasoning_effort" ("high" | "max") but ONLY for DeepSeek/GLM models hosted on the platform — it is NOT a Qwen parameter.
  values: enable_thinking: true | false. thinking_budget: any positive integer; "The default value is the maximum chain-of-thought length for the model"; when the budget is hit "the model stops reasoning and responds immediately". preserve_thinking: true | false.
  default_when_omitted: family-split — thinking ENABLED by default on qwen3.7-max, qwen3.7-plus, qwen3.6 series (incl. qwen3.6-flash) and Qwen3.5 series; thinking DISABLED by default on Qwen3-generation commercial models qwen-plus, qwen-max, qwen-flash, qwen-turbo. Coder models (qwen3-coder-plus, qwen3-coder-next): no thinking support documented anywhere; omitting the param gives plain output.
  unsupported_value_behavior: unknown — the deep-thinking, OpenAI-compat, and DashScope API reference pages document no behavior for enable_thinking sent to a non-thinking model (no status code, no error-code string). Error body shape on the compatible-mode surface is documented generically as {"error": {"message": "...", "type": "...", "code": "..."}} with 400 for invalid request. Needs probe (see probes_needed). Known documented hard error: some open-source thinking models (e.g. qwen3-235b-a22b, qwen3-32b) are streaming-only and "a non-streaming call returns an error" (code not published); commercial thinking models explicitly DO support non-streaming.
  per_model:
    - id: qwen3.7-plus
      mode: optional   # hybrid; thinking ON by default, enable_thinking=false disables
      effort: "adjustable: thinking_budget (positive integer token cap; default = model max CoT length)"
    - id: qwen3.7-max
      mode: optional   # hybrid; thinking ON by default; also supports preserve_thinking (recommended for agentic tasks)
      effort: "adjustable: thinking_budget (positive integer token cap)"
    - id: qwen3-coder-plus
      mode: none   # not listed on the deep-thinking page; qwen-coder page never mentions thinking
      effort: n/a
    - id: qwen3-coder-next
      mode: none
      effort: n/a
    - id: qwen3.6-flash
      mode: optional   # hybrid; thinking ON by default; preserve_thinking on specific Qwen3.6 models
      effort: "adjustable: thinking_budget (positive integer token cap)"
    - id: qwen-flash
      mode: optional   # hybrid; thinking OFF by default; enable_thinking=true enables
      effort: "adjustable: thinking_budget (positive integer token cap)"
  silent_substitutions: none documented — no model-swap behavior tied to reasoning flags appears in any official page opened.
  tools_and_streaming: In streaming, reasoning arrives incrementally in delta chunks under "reasoning_content"; final answer under "content". Commercial thinking models support both streaming and non-streaming; some open-source thinking models are streaming-only (error otherwise). DashScope-native incremental_output=true is recommended (not documented as required) with thinking. Whether reasoning tokens count against max_tokens on the OpenAI-compatible surface is NOT documented (only "reasoning traces increase latency and token costs; use thinking_budget to cap"); on the Anthropic surface max_tokens is documented as limiting "the final reply" (distinct from thinking budget). Thinking + tool-calling interplay (reasoning_content alongside tool_calls) is undocumented. Legacy compat-page constraint on record: "tools cannot currently be used with stream=True" (Chinese compat page; likely stale but it is what the doc says — probe if load-bearing).
  compat_passthrough: OpenAI compatible-mode /compatible-mode/v1: passes (enable_thinking accepted as top-level body field / extra_body). DashScope native API: passes (parameter is native there). Anthropic /apps/anthropic: enable_thinking undocumented → unknown (surface documents Anthropic thinking object instead; treat enable_thinking as stripped until probed).
discovery:
  endpoint: unknown — GET /compatible-mode/v1/models is NOT documented on either the EN or CN OpenAI-compatibility page (only /chat/completions is specified). The Anthropic surface explicitly has NO list endpoint: "provides only the Messages API (/v1/messages) and does not provide a model list endpoint (/v1/models)", returning HTTP 404. Third-party writeups claim /compatible-mode/v1/models exists and returns ids-only, but that is registry-grade; settle with probe.
  auth: bearer (Authorization: Bearer <DASHSCOPE_API_KEY>) if the endpoint exists
  metadata: ids-only per unofficial reports; unverified
  account_scoped: "yes: model availability differs per region (US region lacks coder models entirely) and per workspace; Coding Plan keys are restricted to interactive coding tools and a plan-specific model set"
  rate_limits_or_caching: undocumented
  verdict: stay-static (until the probe confirms the endpoint exists; even then reported metadata is ids-only, so at best augment-static for existence-checking, never for context/capability data)
  refresh_strategy: on-config-open existence ping IF the probe confirms the endpoint; otherwise n/a — keep the static catalog, region-split
context:
  per_model:
    - id: qwen3.7-plus
      context: "1,000,000 tokens (registry-sourced: OpenRouter/ArtificialAnalysis; official model-list table not retrievable — official pricing page confirms input tiers up to 1M: 0<Token≤256K and 256K<Token≤1M)"
      max_output: "65,536 (registry-sourced; unpublished on retrievable official pages)"
      long_context_pricing: "boundary at 256K input tokens — intl: $0.4/M in, $1.6/M out (0–256K); $1.2/M in, $4.8/M out (256K–1M); thinking-mode output billed same as non-thinking; Beijing: $0.276/$1.101 and $0.826/$3.301"
      overpromise_risk: "no known account tier split; risk only that registry 1M/65,536 figures are unconfirmed by official tables"
    - id: qwen3.7-max
      context: "1,000,000 tokens (official Alibaba Cloud blog qwen3-7-the-agent-frontier)"
      max_output: "65,536 (official Alibaba Cloud blog)"
      long_context_pricing: "flat 0<Token≤1M — intl: $2.5/M in, $7.5/M out (currently 50% promotional discount); Beijing: $1.65/$4.951; context caching supported with separate discount"
      overpromise_risk: no
    - id: qwen3-coder-plus
      context: "1,000,000 tokens implied by official pricing top tier 256K<Token≤1M (official model-table number not retrievable)"
      max_output: unpublished
      long_context_pricing: "4 tiers (intl): 0–32K $1/$5; 32K–128K $1.8/$9; 128K–256K $3/$15; 256K–1M $6/$60 per M in/out; Beijing same boundaries, input $0.574–$2.868 range"
      overpromise_risk: "yes: NOT available in the US (Virginia) region at all — any qwen-us config listing it over-promises; also absent from the qwen-coder doc page (only qwen3-coder-next and qwen-coder-turbo named there), though the pricing page still lists it"
    - id: qwen3-coder-next
      context: "256K (official pricing tops at 128K<Token≤256K tier; Hugging Face model card: 262,144 native)"
      max_output: unpublished
      long_context_pricing: "3 tiers (intl): 0–32K $0.3/$1.5; 32K–128K $0.5/$2.5; 128K–256K $0.8/$4; Beijing same boundaries, input $0.144–$0.359"
      overpromise_risk: "yes: not served in the US region (no coder models there)"
    - id: qwen3.6-flash
      context: "1,000,000 tokens (registry-sourced 1M/65,536; official pricing tiers reach 256K<Token≤1M)"
      max_output: "65,536 (registry-sourced; unpublished on retrievable official pages)"
      long_context_pricing: "boundary at 256K — intl: $0.25/$1.5 (0–256K), $1/$4 (256K–1M); Beijing: $0.165/$0.99 and $0.66/$3.961; batch inference 50% off; context caching supported. NOTE registry (OpenRouter) shows $0.1875/$1.125 — recorded as conflict (likely discount snapshot)"
      overpromise_risk: no
    - id: qwen-flash
      context: "up to 1M by pricing tiers — intl tiers: 0–256K, 256K–1M; Beijing tiers: 0–128K, 128K–256K, 256K–1M (official model-table number not retrievable)"
      max_output: unpublished
      long_context_pricing: "intl: $0.05/$0.4 (0–256K), $0.25/$2 (256K–1M); Beijing: $0.022/$0.216 (0–128K), $0.087/$0.861 (128K–256K), $0.173/$1.721 (256K–1M) — note Beijing has a THIRD boundary at 128K that intl does not"
      overpromise_risk: "yes: qwen-flash did not appear in the US (Virginia) region model listing retrieved from the official models page (US list showed qwen3.7-max, qwen3.7-plus, qwen3.6-flash + third-party) — if the qwen-us preset includes qwen-flash, verify with a live probe"
  token_counting_endpoint: none documented on the compatible-mode or Anthropic surfaces
probes_needed:
  - question: Does GET /compatible-mode/v1/models exist, and what metadata does it return (ids-only vs context/capabilities)? Is it account/region-scoped?
    probe: 'curl -s -X GET "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models" -H "Authorization: Bearer $DASHSCOPE_API_KEY"'
  - question: What happens when enable_thinking=true is sent to a coder model (error code vs silently ignored)?
    probe: 'curl -s -X POST "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"qwen3-coder-plus\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"enable_thinking\":true,\"stream\":true}"'
  - question: Does the compatible-mode surface 400 or silently ignore OpenAI-style reasoning_effort on a Qwen model (it is documented only for DeepSeek/GLM on DashScope)?
    probe: 'curl -s -X POST "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"qwen3.7-max\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"high\"}"'
  - question: Do reasoning tokens count against max_tokens on the OpenAI-compatible surface (truncation semantics with thinking on)?
    probe: 'curl -s -X POST "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"qwen3.7-plus\",\"messages\":[{\"role\":\"user\",\"content\":\"Prove sqrt(2) is irrational\"}],\"enable_thinking\":true,\"max_tokens\":64,\"stream\":true}" | tail -5'
  - question: Is qwen-flash actually callable in the US (Virginia) region (it was absent from the retrieved US model listing)?
    probe: 'curl -s -X POST "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions" -H "Authorization: Bearer $DASHSCOPE_US_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"qwen-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Does enable_thinking=false actually pass through and suppress reasoning_content on default-on models (vs being stripped on some gateway path)?
    probe: 'curl -s -X POST "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"qwen3.6-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"enable_thinking\":false}"'
conflicts:
  - "Official qwen-coder doc page (alibabacloud.com/help/en/model-studio/qwen-coder) names only qwen3-coder-next and qwen-coder-turbo; official model-pricing page still lists qwen3-coder-plus with full tier pricing — recorded, not resolved (coder-plus may be legacy/being phased toward coder-next)"
  - "Applied catalog puts qwen-flash in scope for all three presets, but the official models page US (Virginia) listing retrieved showed only qwen3.7-max, qwen3.7-plus, qwen3.6-flash (+ third-party) — recorded; probe before trusting either (page renders client-side and may have been partial)"
  - "OpenRouter (openrouter.ai/qwen/qwen3.6-flash via search) prices qwen3.6-flash at $0.1875/$1.125 per M; official model-pricing page says $0.25/$1.5 (0–256K tier) — recorded (likely a promotional-discount snapshot)"
  - "Unattributed third-party claim that GET /compatible-mode/v1/models 'returns only ids and names' vs official OpenAI-compat pages (EN and CN) which document no /models endpoint at all — recorded; settle by probe"
  - "Registries (OpenRouter, llm-stats, Requesty) state qwen3.7-plus and qwen3.6-flash have 1M context / 65,536 max output; official retrievable pages publish no context or max-output numbers for these ids (tables render client-side) — registry numbers used as leads only, flagged registry-sourced in the context section"
notes: "Load-bearing operational facts: (1) enable_thinking placement differs by transport — top-level in raw JSON/Node, extra_body in OpenAI Python SDK; Sylliptor emitting raw JSON should place it top-level. (2) Default-ON thinking on qwen3.7-plus/max and qwen3.6-flash means a client that never sends the flag WILL receive reasoning_content deltas — a reasoning_mode:none catalog entry for these models is wrong; the correct off-switch is an explicit enable_thinking:false. (3) The Anthropic-compatible surface is real (all regions incl. US: https://dashscope-us.aliyuncs.com/apps/anthropic), takes the standard Anthropic thinking object, requires max_tokens, temperature range deviates ([0,2) not [0,1]), and 404s on /v1/models. (4) DashScope-native reasoning_effort ('high'|'max') exists but only for DeepSeek/GLM hosted models — never emit it for Qwen ids. (5) qwen3.7-max additionally supports preserve_thinking:true (recommended by Alibaba for agentic tasks) — relevant to an agentic CLI. (6) Beijing qwen-flash has a 128K pricing boundary that intl lacks — region-aware cost estimation needed. (7) Completions (non-chat) API is Beijing-region-only per the qwen-coder page."
````

### Gap-fill addendum (post-verification)

The EN and CN model-spec tables at /model-studio/models still render client-side (context/max-output columns remain non-retrievable via fetch — confirmed again 2026-07-19). BUT the server-rendered model-pricing page (https://www.alibabacloud.com/help/en/model-studio/model-pricing) returns the full per-model input-tier tables in markdown, and each tier's upper bound is the model's officially-published max input/context boundary. That page is the authoritative source for context caps and long-context pricing boundaries; it does not publish max_output (that column lives only in the JS model-list table), so max_output stays registry-sourced.

Exact intl pricing tiers read off the server-rendered pricing page (USD /M tokens, checked 2026-07-19):
- qwen3.7-max: single tier 0<Token≤1M -> $2.5 in / $7.5 out (limited-time 50% off). No sub-256K split. => context 1,000,000; NO long-context boundary (flat to 1M).
- qwen3.7-plus: 0<Token≤256K -> $0.4/$1.6; 256K<Token≤1M -> $1.2/$4.8 (thinking-mode output billed at same $1.6/$4.8). => context 1,000,000; boundary at 256K.
- qwen3.6-flash: 0<Token≤256K -> $0.25/$1.5; 256K<Token≤1M -> $1.0/$4.0. => context 1,000,000; boundary at 256K.
- qwen3-coder-plus: 0<Token≤32K -> $1.0/$5.0; 32K<Token≤128K -> $1.8/$9.0; 128K<Token≤256K -> $3.0/$15.0; 256K<Token≤1M -> $6.0/$60.0. => context 1,000,000; boundaries at 32K / 128K / 256K.
- qwen3-coder-next: 0<Token≤32K -> $0.3/$1.5; 32K<Token≤128K -> $0.5/$2.5; 128K<Token≤256K -> $0.8/$4.0. Top tier ends at 256K (no 256K-1M tier). => context 262,144 (256K); boundaries at 32K / 128K.
- qwen-flash: 0<Token≤256K -> $0.05/$0.4; 256K<Token≤1M -> $0.25/$2.0. => context 1,000,000; boundary at 256K.
- qwen-plus: 0<Token≤256K -> $0.4/$1.2; 256K<Token≤1M -> $1.2/$3.6. => context 1,000,000; boundary at 256K.
- qwen-max: single flat tier -> $1.6/$6.4 (no token-range split published). => context caps at the single-tier ceiling (registry lists 262,144); NO long-context boundary.
- qwen-turbo: single flat tier -> $0.05/$0.2 (non-thinking) / $0.5 (thinking output). => context 1,000,000 per registry; no boundary published on pricing page.

max_output (registry cross-check — Qwen Cloud model card, OpenRouter, ArtificialAnalysis; NOT on any server-retrievable official page):
- qwen3.7-max: 65,536 out; Qwen Cloud card shows 991.80K max input (983K in thinking mode), 65.53K max output within the 1M window.
- qwen3.7-plus: 65,536 out.
- qwen3.6-flash: 65,536 out.
- qwen3-coder-plus: 65,536 out.
- qwen3-coder-next: 262,144 context; registry summarizes output up to 262,144 (OpenRouter model page did not expose a separate max-output figure — treat 262,144 as low-confidence, likely conflated with context).
- qwen-flash: max_output NOT confirmed by any opened source (registry pages silent) — unknown.

URLs opened: https://www.alibabacloud.com/help/en/model-studio/model-pricing (SUCCESS - server-rendered tier tables), https://www.alibabacloud.com/help/en/model-studio/models (FAILED-partial, JS table), https://help.aliyun.com/zh/model-studio/models (FAILED-partial, JS table), https://www.alibabacloud.com/help/en/model-studio/qwen-coder (no spec table), https://www.alibabacloud.com/help/en/model-studio/deep-thinking (defers to JS Model list), https://openrouter.ai/qwen/qwen3.7-max (context 1M, no out), https://openrouter.ai/qwen/qwen3.7-plus (context 1M, no out), https://openrouter.ai/qwen/qwen3-coder-next (context 256K/262K, no out).

Corrected/added YAML lines for the affected keys:

````yaml
context:
  per_model:
    - id: qwen3.7-max
      context: "1,000,000 tokens (official: model-pricing page single tier 0<Token≤1M; Qwen Cloud card 991.80K max input, 983K in thinking mode)"
      max_output: "65,536 (registry-sourced: Qwen Cloud card / OpenRouter / ArtificialAnalysis; not on server-retrievable official page)"
      long_context_pricing: "none — flat single tier to 1M: intl $2.5/M in, $7.5/M out (limited-time 50% off)"
      overpromise_flag: "context/max_output beyond 1M/65,536 = overpromise"
    - id: qwen3.7-plus
      context: "1,000,000 tokens (official: model-pricing tiers cap at 256K<Token≤1M)"
      max_output: "65,536 (registry-sourced; unpublished on retrievable official pages)"
      long_context_pricing: "boundary at 256K input — intl: $0.4/M in, $1.6/M out (0–256K); $1.2/M in, $4.8/M out (256K–1M); thinking-mode output billed same $1.6/$4.8"
      overpromise_flag: "context >1M or max_output >65,536 = overpromise"
    - id: qwen3.6-flash
      context: "1,000,000 tokens (official: model-pricing tiers cap at 256K<Token≤1M)"
      max_output: "65,536 (registry-sourced; unpublished on retrievable official pages)"
      long_context_pricing: "boundary at 256K input — intl: $0.25/M in, $1.5/M out (0–256K); $1.0/M in, $4.0/M out (256K–1M)"
      overpromise_flag: "context >1M or max_output >65,536 = overpromise"
    - id: qwen3-coder-plus
      context: "1,000,000 tokens (official: model-pricing tiers cap at 256K<Token≤1M)"
      max_output: "65,536 (registry-sourced; unpublished on retrievable official pages)"
      long_context_pricing: "3 boundaries at 32K / 128K / 256K input — intl: $1.0/$5.0 (0–32K); $1.8/$9.0 (32K–128K); $3.0/$15.0 (128K–256K); $6.0/$60.0 (256K–1M)"
      overpromise_flag: "context >1M or max_output >65,536 = overpromise"
    - id: qwen3-coder-next
      context: "262,144 tokens / 256K (official: model-pricing top tier ends at 128K<Token≤256K, no 256K–1M tier)"
      max_output: "262,144 claimed by registry but LOW-CONFIDENCE (likely conflated with context; OpenRouter exposes no separate max-output) — treat as unconfirmed, probe"
      long_context_pricing: "2 boundaries at 32K / 128K input — intl: $0.3/$1.5 (0–32K); $0.5/$2.5 (32K–128K); $0.8/$4.0 (128K–256K)"
      overpromise_flag: "context >256K = overpromise; max_output unconfirmed"
    - id: qwen-flash
      context: "1,000,000 tokens (official: model-pricing tiers cap at 256K<Token≤1M)"
      max_output: "unknown — not confirmed by any opened source (registry pages silent); probe before relying on it"
      long_context_pricing: "boundary at 256K input — intl: $0.05/M in, $0.4/M out (0–256K); $0.25/M in, $2.0/M out (256K–1M)"
      overpromise_flag: "context >1M = overpromise; max_output unset (do not promise)"
    - id: qwen-plus
      context: "1,000,000 tokens (official: model-pricing tiers cap at 256K<Token≤1M)"
      max_output: "unknown on retrievable official pages (registry commonly cites 32,768/65,536); probe"
      long_context_pricing: "boundary at 256K input — intl: $0.4/M in, $1.2/M out (0–256K); $1.2/M in, $3.6/M out (256K–1M)"
      overpromise_flag: "context >1M = overpromise"
    - id: qwen-max
      context: "262,144 tokens / 256K per registry (official: model-pricing shows a single flat tier, no token-range split — no 1M evidence)"
      max_output: "unknown on retrievable official pages; probe"
      long_context_pricing: "none — flat single tier: intl $1.6/M in, $6.4/M out"
      overpromise_flag: "context >256K unverified = overpromise"
    - id: qwen-turbo
      context: "1,000,000 tokens per registry (official: model-pricing shows single flat tier, no boundary)"
      max_output: "unknown on retrievable official pages; probe"
      long_context_pricing: "none — flat single tier: intl $0.05/M in, $0.2/M out (non-thinking) / $0.5/M out (thinking)"
      overpromise_flag: "unverified above registry values"
  note: "Context caps are anchored to the SERVER-RENDERED model-pricing page (each input-tier ceiling = official max context boundary); max_output is not published on any server-retrievable official page and remains registry-sourced. Numbers are the intl (Singapore) region; qwen3-coder-* are ABSENT in the US region per account_scoped note."
  sources_add:
    - https://www.alibabacloud.com/help/en/model-studio/model-pricing   # SUCCESS: server-rendered per-model input-tier tables (context boundaries + long-context pricing) retrievable 2026-07-19
````

**Still unresolved (needs a live key):**

````
max_output is not confirmed by any server-rendered official source for ANY Qwen id (qwen3.7-plus/max, qwen3.6-flash, qwen3-coder-plus/next, qwen-flash/plus/max/turbo); qwen3-coder-next's 262,144 max_output and qwen-max's 256K context are especially low-confidence. Authed probe to read the real max output/context ceiling from the 400 error message: curl -sS -X POST "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions" -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"hi"}],"max_tokens":9999999}'  # the InvalidParameter/range_error body states the model's true max output; repeat per id (qwen3.7-max, qwen3.7-plus, qwen3.6-flash, qwen3-coder-plus, qwen-flash, qwen-plus, qwen-max, qwen-turbo). Also probe context ceiling: curl -sS "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models" -H "Authorization: Bearer $DASHSCOPE_API_KEY"  # confirms whether a list endpoint exists and whether it returns max_context_length.
````

### Verifier notes (minor, not applied)

- **'DashScope also documents reasoning_effort ("high" | "max") but ONLY for DeepSeek/GLM models hosted on the platform.'** — The exact accepted value pair "high"|"max" is not confirmed by any source I could open. The deep-thinking page (which I fetched) does not mention reasoning_effort at all, and the DeepSeek/GLM model page carrying it was not among the opened sources. The values are presented as documented but are effectively unsourced/unverifiable via fetch. Non-load-bearing for Qwen safety (the contract correctly s *Suggested: Cite the specific DeepSeek/GLM model page that documents reasoning_effort and its allowed values, or downgrade the value pair from stated fact to 'reported, unverified' and add it to probes_needed.*
- **claims_reverified: 'qwen-us region serves no coder models' = confirmed; and per-model US-region absence of qwen-flash / presence of specific ids.** — These region-membership claims rest entirely on a client-side-rendered listing on the model-studio/models page, which the contract itself elsewhere reports as only partially retrievable. I could not independently reproduce the US (Virginia) region model listing via fetch (spec/listing tables render client-side), so the 'confirmed' verdict on qwen-us coder absence is not fetch-verifiable. Unverifia *Suggested: Downgrade the qwen-us coder-absence verdict from 'confirmed' to 'reported/probe-queued' to match the client-rendered-page caveat already stated elsewhere in the contract, keeping the queued region probe as the settling mechanism.*

---

## 6. zhipu

*Verification: corrected after adversarial review — 4 challenge(s), 1 fatal/major; corrections are applied in the block.*

````yaml
provider: zhipu (zhipu-cn / zhipu-intl / zhipu-coding-plan)
endpoints:
  - url: https://open.bigmodel.cn/api/paas/v4
    protocol: openai-chat
  - url: https://api.z.ai/api/paas/v4
    protocol: openai-chat
  - url: https://open.bigmodel.cn/api/coding/paas/v4
    protocol: openai-chat (Coding Plan keys ONLY; "Coding API endpoint is only for Coding scenarios", not interchangeable with paas/v4)
  - url: https://api.z.ai/api/coding/paas/v4
    protocol: openai-chat (Coding Plan keys ONLY)
  - url: https://api.z.ai/api/anthropic
    protocol: anthropic-messages (Claude Code integration surface; ANTHROPIC_BASE_URL target)
checked: 2026-07-19 (correction pass same day)
sources:
  - https://docs.z.ai/guides/llm/glm-5.2
  - https://docs.z.ai/api-reference
  - https://docs.z.ai/llms.txt
  - https://docs.z.ai/api-reference/llm/chat-completion.md
  - https://docs.z.ai/api-reference/tools/tokenizer.md
  - https://docs.z.ai/guides/capabilities/thinking.md
  - https://docs.z.ai/guides/capabilities/thinking-mode.md
  - https://docs.z.ai/guides/llm/glm-5.1.md
  - https://docs.z.ai/guides/llm/glm-5-turbo.md
  - https://docs.z.ai/guides/llm/glm-4.7.md
  - https://docs.z.ai/guides/overview/pricing.md
  - https://docs.z.ai/devpack/tool/claude
  - https://docs.z.ai/devpack/overview
  - https://docs.z.ai/guides/develop/openai/introduction (FAILED - 404; OpenAI-SDK usage confirmed instead via docs.z.ai/guides/develop/openai/python search snippet)
  - https://docs.bigmodel.cn/cn/guide/start/model-overview
  - https://docs.bigmodel.cn/llms.txt
  - https://docs.bigmodel.cn/api-reference/模型-api/对话补全.md
  - https://docs.bigmodel.cn/api-reference/模型-api/文本分词器.md
  - https://docs.bigmodel.cn/cn/guide/models/text/glm-5.2.md
  - https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash.md
  - https://docs.bigmodel.cn/cn/guide/capabilities/thinking.md
  - https://docs.bigmodel.cn/cn/faq/api-code.md
  - https://docs.bigmodel.cn/cn/coding-plan/overview.md
  - https://docs.bigmodel.cn/cn/guide/start/model-pricing.md (FAILED - 404)
  - https://bigmodel.cn/pricing (FAILED - renders client-side, only loading shell retrieved; CN tiered pricing NOT confirmable)
  - https://api.z.ai/api/paas/v4/models (unauthenticated GET -> HTTP 401, body {"error":{"code":"1001","message":"..."}})
  - https://open.bigmodel.cn/api/paas/v4/models (unauthenticated GET -> HTTP 401, same nested body shape)
reasoning:
  param: |
    "thinking": {"type": "enabled" | "disabled", "clear_thinking": <boolean>}  (top-level request-body object; via OpenAI SDK it must go in extra_body)
    "reasoning_effort": "<string>"  (top-level; documented for GLM-5.2 only)
  values: |
    thinking.type: "enabled" | "disabled" (exact enum, both surfaces). thinking.clear_thinking: true|false (default true = strip reasoning_content from prior turns; note thinking-mode.md says preserved thinking is "enabled by default" on the Coding Plan endpoint but "disabled by default" on the standard API endpoint).
    reasoning_effort (glm-5.2 only): "max" | "xhigh" | "high" | "medium" | "low" | "minimal" | "none". Documented server-side coercion: "none"/"minimal" skip thinking; "low"/"medium" map to "high"; "xhigh" maps to "max".
  default_when_omitted: |
    thinking defaults to {"type":"enabled"} on all catalog models (schema default "enabled", per-model pages state 'default is enabled'). With thinking enabled, glm-5.2 / glm-5.1 / glm-5-turbo AUTO-DECIDE whether to emit reasoning; glm-4.7 uses FORCED thinking ("will think compulsorily when enabled"). glm-4.7-flash / glm-4.7-flashx: thinking supported ("GLM-4.5 and above"), forced-vs-auto not stated. reasoning_effort defaults to "max" on glm-5.2.
  unsupported_value_behavior: |
    Not explicitly documented per-parameter. Platform error catalog: invalid parameter value -> HTTP 400 code 1214 "${field} 参数非法。请检查文档"; general bad params -> HTTP 400 code 1210; conflicting params -> HTTP 400 code 1215; unknown model -> HTTP 400 code 1211. Live-observed error body on BOTH surfaces is nested {"error":{"code":"<string>","message":"<string>"}} (string codes; verified 2026-07-19 via unauthenticated 401 probes). Whether reasoning_effort on non-5.2 models 400s or is silently ignored is UNDOCUMENTED (probe below). Field-report evidence (hermes-agent #16533): unknown top-level keys like OpenRouter-style "reasoning" are silently ignored (no error, no reasoning_content).
  per_model:
    - id: glm-5.2
      mode: optional (default enabled, auto-decide)
      effort: "adjustable: [max, xhigh, high, medium, low, minimal, none], default max (xhigh->max, low/medium->high, minimal/none->no thinking coercion)"
    - id: glm-5.1
      mode: optional (default enabled, auto-decide)
      effort: n/a (reasoning_effort documented as glm-5.2-only; behavior if sent undocumented — probe)
    - id: glm-5-turbo
      mode: optional (default enabled, auto-decide)
      effort: n/a
    - id: glm-4.7
      mode: optional-but-forced (thinking.type can disable; when enabled — the default — it always thinks, no auto-decide)
      effort: n/a
    - id: glm-4.7-flashx
      mode: optional (supported per "GLM-4.5 and above"; forced-vs-auto unspecified)
      effort: n/a
    - id: glm-4.7-flash
      mode: optional (CN model page shows thinking:{type:"enabled"}; forced-vs-auto unspecified)
      effort: n/a
  silent_substitutions: |
    None triggered by reasoning flags. BUT: (1) Coding Plan: "GLM-5.1/GLM-5 automatically redirect to GLM-5.2" (docs.bigmodel.cn/cn/coding-plan/overview.md) — silent model swap on the coding endpoints. (2) Anthropic surface (api.z.ai/api/anthropic): Claude Code opus/sonnet/haiku tiers map to GLM models via ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL, default GLM-4.7 — by-design aliasing.
  tools_and_streaming: |
    Streaming: SSE with reasoning emitted as delta.reasoning_content (separate field from delta.content), stream ends "data: [DONE]". Non-stream: message.reasoning_content field. Tools: tools array (max 128 functions), types "function"|"web_search"|"retrieval", tool_choice only "auto"; tool_stream boolean (default false) enables streamed function-call args on GLM-4.6+. Interleaved thinking with tools: thinking-mode.md instructs that thinking blocks "should be explicitly preserved and returned together with the tool results" (clear_thinking:false retains prior-turn reasoning_content). Whether reasoning tokens count against max_tokens/128K output cap: UNDOCUMENTED (probe below). Coding-plan field reports (Roo Code docs) say useful output may arrive in reasoning_content rather than content on the coding endpoint — clients must render both.
  compat_passthrough: |
    OpenAI SDK on paas/v4: passes — same endpoint; "thinking" and "reasoning_effort" are native body fields sent via extra_body (documented pattern). Coding endpoints (/api/coding/paas/v4): passes — same openai-chat contract, plus preserved-thinking default flips to enabled. Anthropic surface (api.z.ai/api/anthropic): unknown — docs do not state whether anthropic-style "thinking":{"type":"enabled","budget_tokens":N} or the native shape applies; only env-var model mapping is documented.
discovery:
  endpoint: none documented on either surface; unauthenticated GET https://api.z.ai/api/paas/v4/models and https://open.bigmodel.cn/api/paas/v4/models both return HTTP 401 (not 404), so an undocumented authed /models MAY exist — probe below
  auth: unknown (401 suggests bearer if it exists; 401 body is {"error":{"code":"1001","message":"Authentication parameter not received in Header..."}})
  metadata: unknown
  account_scoped: unknown (Coding Plan keys are endpoint-scoped: only valid on /api/coding/paas/v4; general keys only on /api/paas/v4)
  rate_limits_or_caching: undocumented
  verdict: stay-static (upgrade to augment-static only if the authed probe shows a real /models with ids)
  refresh_strategy: n/a (if probe succeeds: on-config-open, ids-only augmentation)
context:
  per_model:
    - id: glm-5.2
      context: 1M (1,000,000-class; both CN and intl docs state "1M", no tier splits documented)
      max_output: 128K (request param max_tokens documented range 1..131072; out-of-range behavior NOT documented — likely 400 code 1214, unverified, probe below)
      long_context_pricing: flat on intl ($1.4/M input, $0.26/M cached input, $4.4/M output; no boundary listed). CN: unverified — bigmodel.cn/pricing renders client-side (FAILED); forum lead claims flat 8/2/28 CNY
      overpromise_risk: "yes: Coding Plan subscribers — plan docs publish prompt quotas but NO context guarantee for the coding endpoints, and coding endpoint silently redirects glm-5.1/glm-5 to glm-5.2; hardcoding 1M is unverified for coding-plan keys"
    - id: glm-5.1
      context: 200K (no tier splits documented)
      max_output: 128K
      long_context_pricing: intl flat ($1.4 / $0.26 / $4.4 per M). CN unverified — bigmodel.cn/pricing FAILED for both researcher and verifier; forum lead claims CN glm-5.1 IS input-length tiered with a [32K+) band — recorded as conflict, not resolved
      overpromise_risk: "yes: on Coding Plan endpoints glm-5.1 is silently served as glm-5.2"
    - id: glm-5-turbo
      context: 200K
      max_output: 128K
      long_context_pricing: intl flat ($1.2 / $0.24 / $4.0 per M); CN unverified
      overpromise_risk: no (per available docs)
    - id: glm-4.7
      context: 200K
      max_output: 128K
      long_context_pricing: intl flat ($0.6 / $0.11 / $2.2 per M); CN unverified
      overpromise_risk: no
    - id: glm-4.7-flashx
      context: 200K
      max_output: 128K
      long_context_pricing: intl flat ($0.07 / $0.01 / $0.4 per M); CN unverified
      overpromise_risk: no
    - id: glm-4.7-flash
      context: 200K
      max_output: 128K
      long_context_pricing: free (input, cached input, output all "Free"); free-tier rate/concurrency limits undocumented
      overpromise_risk: "yes: free model — undocumented rate/concurrency limits may throttle long-context use well below 200K in practice"
  token_counting_endpoint: |
    POST {base}/paas/v4/tokenizer — documented on BOTH surfaces (docs.z.ai/api-reference/tools/tokenizer.md; docs.bigmodel.cn/api-reference/模型-api/文本分词器.md). Request: {model, messages, tools?, request_id?, user_id?}. Response: usage.prompt_tokens / image_tokens / video_tokens / total_tokens (+id, created, request_id). CAVEAT: documented supported-model roster is STALE — intl lists only glm-4.6 (default) / glm-4.6v / glm-4.5; CN lists glm-4.6, glm-4.6v, glm-4.5, glm-4.5-air, glm-4-0520, glm-4-long, glm-4-air, glm-4-flash. NO glm-5.x on either page — whether glm-5.x ids are accepted is unverified (probe below). Do not rely on it for the current catalog without probing.
probes_needed:
  - question: Does an authenticated GET /models exist (both surfaces return 401 not 404 unauthenticated), and how rich is its metadata?
    probe: 'curl -s https://api.z.ai/api/paas/v4/models -H "Authorization: Bearer $ZAI_API_KEY"'
  - question: Is reasoning_effort on a non-glm-5.2 model rejected (400 code 1214), silently ignored, or honored?
    probe: 'curl -s https://api.z.ai/api/paas/v4/chat/completions -H "Authorization: Bearer $ZAI_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.1\",\"reasoning_effort\":\"high\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Does an out-of-enum thinking.type (e.g. "auto") 400 with code 1214 or get silently coerced?
    probe: 'curl -s https://api.z.ai/api/paas/v4/chat/completions -H "Authorization: Bearer $ZAI_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2\",\"thinking\":{\"type\":\"auto\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Does max_tokens above 131072 return HTTP 400 (code 1214) or get silently clamped? (Range 1..131072 is documented; out-of-range behavior is not.)
    probe: 'curl -s https://api.z.ai/api/paas/v4/chat/completions -H "Authorization: Bearer $ZAI_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2\",\"max_tokens\":200000,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Do reasoning tokens count against max_tokens (does a tiny max_tokens starve thinking / return finish_reason length before any content)?
    probe: 'curl -s https://api.z.ai/api/paas/v4/chat/completions -H "Authorization: Bearer $ZAI_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2\",\"thinking\":{\"type\":\"enabled\"},\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Prove sqrt(2) is irrational\"}]}"'
  - question: Does the tokenizer endpoint accept current-catalog ids (glm-5.2, glm-5.1, glm-4.7) or only the stale documented roster (up to glm-4.6)?
    probe: 'curl -s https://api.z.ai/api/paas/v4/tokenizer -H "Authorization: Bearer $ZAI_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: On a Coding Plan key, does requesting glm-5.1 return "model":"glm-5.2" in the response (confirming the silent redirect), and what context length is actually accepted?
    probe: 'curl -s https://api.z.ai/api/coding/paas/v4/chat/completions -H "Authorization: Bearer $ZAI_CODING_PLAN_KEY" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Does the anthropic-messages surface accept anthropic-style thinking ({"type":"enabled","budget_tokens":N}) or reject/ignore it?
    probe: 'curl -s https://api.z.ai/api/anthropic/v1/messages -H "x-api-key: $ZAI_API_KEY" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" -d "{\"model\":\"glm-5.2\",\"max_tokens\":256,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":128},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
conflicts:
  - "docs.z.ai/api-reference/llm/chat-completion.md says reasoning_effort supported on 'GLM-5.2 only'; docs.z.ai/guides/capabilities/thinking.md and docs.bigmodel.cn/cn/guide/capabilities/thinking.md say 'glm-5.2 and above' — recorded, not resolved (identical in practice today since glm-5.2 is the newest, but matters on next release)"
  - "docs.z.ai/guides/overview/pricing.md lists glm-5.1 flat $1.4/$4.4 with no tiers; linux.do forum thread (https://linux.do/t/topic/2418141) claims CN glm-5.1 pricing is input-length tiered with a [32K+) band and glm-5.2 flat at 8/2/28 CNY — official CN pricing page (bigmodel.cn/pricing) FAILED to render for both researcher and verifier, so unresolved (intl rows independently re-confirmed exact)"
  - "docs.z.ai/api-reference/llm/chat-completion.md documents the error response schema as flat {code:int32, message:string}; LIVE 401 responses on BOTH api.z.ai and open.bigmodel.cn return nested {\"error\":{\"code\":\"<string>\",\"message\":\"<string>\"}} with STRING codes (probed 2026-07-19) — parse the nested string-code shape; doc schema recorded as not matching observed behavior"
  - "developersdigest.tech blog claims glm-5.2 has 'two thinking-effort levels'; official docs on both surfaces enumerate seven reasoning_effort values (max/xhigh/high/medium/low/minimal/none) with coercion — official wins, registry/blog recorded as wrong"
  - "docs.z.ai/guides/llm/glm-4.7.md presents thinking as ordinary optional for the 4.7 family; capabilities/thinking docs on both surfaces say GLM-4.7 uses forced thinking when enabled — recorded; treated as forced-when-enabled"
notes: |
  CORRECTION PASS (2026-07-19) applied against the adversarial verifier's four challenges; all four upheld (three fixes, one concurrence):
  (1) TOKENIZER: original "token_counting_endpoint: none" was WRONG. Both llms.txt indexes list a tokenizer API reference; re-fetched both pages and confirmed POST {base}/paas/v4/tokenizer with usage.prompt_tokens/image_tokens/video_tokens/total_tokens. Documented rosters are stale (no glm-5.x anywhere; intl stops at glm-4.6/4.6v/4.5, CN stops at glm-4.6 plus legacy 4.x tail), so glm-5.x acceptance needs the listed probe.
  (2) MAX_TOKENS: the 1..131072 range is documented on both surfaces, but NEITHER documents the out-of-range outcome. "Emitting more is a 400" was an inference from the generic 1214 catalog — now stated as 'likely 400 code 1214, unverified' with a dedicated probe, matching the hedging used for out-of-enum thinking.type.
  (3) ERROR BODY: original claim 'intl error body shown as {code:int, message:string}' was doc-only and does not match reality. Live re-probe of both surfaces (2026-07-19): HTTP 401 with nested {"error":{"code":"1001","message":"..."}} — string code, identical shape CN and intl. The intl doc's flat int32 schema is recorded as a conflict; clients should parse the nested string-code shape.
  (4) CN PRICING: verifier independently hit the same bigmodel.cn/pricing client-side-render failure and the model-pricing.md 404, and independently re-confirmed the intl pricing rows exact (1.4/0.26/4.4, 1.4/0.26/4.4, 1.2/0.24/4.0, 0.6/0.11/2.2, 0.07/0.01/0.4, Flash free). Conflict stands unresolved as originally recorded — no change.
  Everything else from the original contract stands: CN (open.bigmodel.cn) and intl (api.z.ai) parameter contracts are IDENTICAL for thinking/reasoning_effort (same shapes, same enums, same defaults, same per-model behavior lists); model enums differ slightly in legacy tail (CN keeps glm-4-flash-250414 etc.). All six catalog models appear in both enums. reasoning_effort's documented coercion means Sylliptor can safely offer the full 7-value knob on glm-5.2 but should surface that low/medium silently become high and xhigh becomes max (billing/latency surprise, not an error). thinking.clear_thinking default differs by surface class (Coding Plan endpoints preserve prior-turn reasoning by default; standard API strips it) — relevant to context accounting since retained reasoning_content re-enters the prompt. Coding Plan keys and general keys are NOT interchangeable across paas/v4 vs coding/paas/v4. GLM-4.7-Flash is fully free; GLM-4.7 docs pitch the $10/month Coding Plan. Error catalog codes (1210/1211/1214/1215/1261) are CN-documented; error BODY shape is the nested string-code form on both surfaces per live probes.
````

### Gap-fill addendum (post-verification)

Two-part gap resolved except the A6 max_tokens-vs-reasoning relationship, which needs a live key.

PART 1 — CN "C long-context pricing boundary" is moot: there is NO length-tiered pricing for the current GLM lineup on either surface. Every source opened shows one flat input/output/cache rate per model, independent of context length used.
- docs.z.ai/guides/overview/pricing.md (server-rendered md, flat table, per 1M tokens): GLM-5.2 $1.4 in / $0.26 cache / $4.4 out; GLM-5.1 $1.4/$0.26/$4.4; GLM-5 $1.0/$0.2/$3.2; GLM-5-Turbo $1.2/$0.24/$4.0; GLM-4.7 $0.6/$0.11/$2.2.
- vibecoding.app/blog/zhipu-ai-glm-pricing-2026 explicitly: "Zhipu does not use tiered pricing based on input context length... flat rates per model regardless of context window size."
- avenchat.com/blog/glm-5.2-pricing and datalearner.com (glm-5-2, glm-5-1): same flat USD rates, no 32K/128K/200K breakpoints.
- CN RMB rates (bigmodel.cn/pricing is client-side, so from mirrors): hvoy.ai/models/glm-5-1 gives GLM-5.1 = ¥6 in / ¥24 out / ¥1.3 cache per 1M; GLM-5.2 ~¥8/¥28/¥2 (single-source, uncorroborated — flag). No RMB tier boundaries anywhere.
- Caps confirmed on docs.bigmodel.cn/cn/guide/start/model-overview + CN model pages: GLM-5.2 = 1M context / 128K max output; GLM-5.1 = 200K context / 128K max output; GLM-5 and GLM-4.7 = 128K max output. (Note GLM-5.1 CN context is 200K, not 1M — only GLM-5.2 is 1M.)

PART 2 — A6 reasoning-token accounting: docs.z.ai/guides/capabilities/thinking.md and docs.bigmodel.cn/cn/guide/capabilities/thinking both state only "Thinking process will consume extra tokens / 思考过程会消耗额外的 Token, please plan usage." The usage schema in the chat-completion API reference (docs.z.ai/api-reference/llm/chat-completion.md and docs.bigmodel.cn/api-reference/模型-api/对话补全.md) documents ONLY prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details.cached_tokens — there is NO reasoning_tokens and NO completion_tokens_details.reasoning_tokens field. That absence implies reasoning_content is folded into completion_tokens (not billed/reported separately). Whether max_tokens / the 128K output cap counts reasoning+visible-content together or visible-content only is NOT documented on any page (one CN-doc summary claimed "excludes thinking" but produced no supporting quote — treat as unconfirmed). Requires a live-key probe.

Corrected/added YAML lines for the affected keys:

````yaml
pricing:
  model: flat per-token; NO input-length tiers on EITHER surface (verified 2026-07-19). The "C long-context boundary" does not exist for the current GLM lineup — one uniform input/cache/output rate per model regardless of context length used.
  intl_usd_per_1M:  # docs.z.ai/guides/overview/pricing.md (server-rendered)
    glm-5.2: {input: 1.4, cached_input: 0.26, output: 4.4}
    glm-5.1: {input: 1.4, cached_input: 0.26, output: 4.4}
    glm-5: {input: 1.0, cached_input: 0.2, output: 3.2}
    glm-5-turbo: {input: 1.2, cached_input: 0.24, output: 4.0}
    glm-4.7: {input: 0.6, cached_input: 0.11, output: 2.2}
  cn_rmb_per_1M:  # bigmodel.cn/pricing is client-side; from mirrors, NOT official server-render
    glm-5.1: {input: 6, output: 24, cached_input: 1.3}   # hvoy.ai/models/glm-5-1
    glm-5.2: {input: 8, output: 28, cached_input: 2}     # SINGLE-SOURCE, uncorroborated — verify
    note: no RMB length-tier boundaries found on any source
  caps:  # docs.bigmodel.cn/cn/guide/start/model-overview + CN model pages
    glm-5.2: {context: 1M, max_output: 128K}
    glm-5.1: {context: 200K, max_output: 128K}   # NOTE 200K, not 1M
    glm-5: {max_output: 128K}
    glm-4.7: {max_output: 128K}
reasoning:
  token_accounting: |
    Docs (thinking.md CN+intl) state only "Thinking process will consume extra tokens / 思考过程会消耗额外的 Token, please plan usage." The usage schema in the chat-completion reference (both surfaces) documents ONLY {prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details.cached_tokens} — NO reasoning_tokens and NO completion_tokens_details breakdown. Absence implies reasoning_content is folded into completion_tokens (billed as output, not separately reported). Whether max_tokens / the 128K output cap counts reasoning+content together or visible-content only is UNDOCUMENTED on every page checked (one CN-doc summary claimed "excludes thinking" but gave no supporting quote — UNCONFIRMED). Needs live-key probe.
````

**Still unresolved (needs a live key):**

````
A6 max_tokens-vs-reasoning relationship still needs a live GLM-5.2 probe (no API key available here). Run with a valid key and inspect finish_reason + usage:
curl -sS https://open.bigmodel.cn/api/paas/v4/chat/completions -H "Authorization: Bearer $ZHIPU_API_KEY" -H "Content-Type: application/json" -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Think step by step in detail, then answer: what is 17*23?"}],"thinking":{"type":"enabled"},"reasoning_effort":"high","max_tokens":64,"stream":false}'
Interpretation: (a) if finish_reason=="length" AND message.content is empty/short while message.reasoning_content is populated => reasoning tokens DO count against max_tokens (max_tokens caps reasoning+content together, o1-style). (b) if usage.completion_tokens >> 64 and content is complete => reasoning is NOT capped by max_tokens (Anthropic-style separate budget). Also record whether usage exposes any reasoning_tokens / completion_tokens_details field not present in the docs. Repeat against https://api.z.ai/api/paas/v4/chat/completions to confirm parity across surfaces.
````

---

## 7. moonshot / moonshot-cn

*Verification: corrected after adversarial review — 6 challenge(s), 1 fatal/major; corrections are applied in the block.*

````yaml
provider: moonshot / moonshot-cn
endpoints:
  - url: https://api.moonshot.ai/v1
    protocol: openai-chat
  - url: https://api.moonshot.cn/v1
    protocol: openai-chat
  - url: https://api.moonshot.ai/anthropic
    protocol: anthropic-messages
checked: 2026-07-19
sources:
  - https://platform.kimi.ai/docs/models
  - https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
  - https://platform.kimi.ai/docs/api/chat
  - https://platform.kimi.ai/docs/api/list-models.md
  - https://platform.kimi.ai/docs/api/errors.md
  - https://platform.kimi.ai/docs/api/models-overview.md
  - https://platform.kimi.ai/docs/api/estimate.md
  - https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart
  - https://platform.kimi.ai/docs/guide/claude-code-kimi.md
  - https://platform.kimi.ai/docs/pricing/chat
  - https://platform.kimi.ai/docs/pricing/chat-k3
  - https://platform.kimi.ai/docs/pricing/chat-k27-code
  - https://platform.kimi.ai/docs/pricing/chat-k26
  - https://platform.kimi.ai/docs/pricing/limits.md
  - https://platform.kimi.ai/docs/llms.txt
  - https://platform.kimi.com/docs/api/chat   # .cn docs; platform.moonshot.cn/docs/api/chat 301-redirects here
  - https://platform.kimi.ai/docs/api/error-codes (FAILED)   # resolved to quickstart content; real page is /docs/api/errors.md
  - https://platform.kimi.ai/docs/api/token-count (FAILED)   # resolved to getting-started content; real page is /docs/api/estimate.md
reasoning:
  param: "thinking" (object; kimi-k2.6, kimi-k2.7-code, kimi-k2.7-code-highspeed, kimi-k2.5) AND "reasoning_effort" (top-level string; kimi-k3 only). Sub-fields of thinking: "type", "keep".
  values: >
    kimi-k3: reasoning_effort one of "low" | "high" | "max".
    kimi-k2.6: thinking {"type":"enabled"} | {"type":"disabled"} | {"type":"enabled","keep":"all"} (keep default null = historical thinking NOT preserved).
    kimi-k2.7-code / kimi-k2.7-code-highspeed: thinking type "enabled" ONLY (disabled errors); keep — passing "all", passing null, or omitting it all behave identically and are treated as "all" on the server; "passing any other invalid value returns an error".
  default_when_omitted: >
    kimi-k3: thinking always on, reasoning_effort defaults to "max" ("Thinking effort supports low, high, and max, with max as the default.").
    kimi-k2.7-code(-highspeed): thinking defaults to {"type":"enabled"} ("Default to be {\"type\": \"enabled\"}"), keep omission coerced to "all" server-side — omitting the whole object or the keep field is safe.
    kimi-k2.6: defaults to enabled (thinking ON by default; keep default null — historical reasoning_content NOT preserved unless keep:"all" is sent).
  unsupported_value_behavior: >
    http-400 + body {"error":{"type":"invalid_request_error","message":"..."}} (errors.md: 400 invalid_request_error = "Request format error, missing required parameter, or invalid parameter type").
    k2.7-code with thinking disabled: "Kimi K2.7 Code model will throw an error if the thinking mode is disabled"; Claude Code doc quotes it as a "400 invalid thinking" error. Exact message string on the /v1 surface unprobed.
    k2.7-code keep: only values OTHER than "all"/null/omitted error; null and omission are coerced to "all" (a silent semantics flip vs k2.6, not a 400).
    Sending K2.x-style "thinking" to kimi-k3 (which "uses reasoning_effort instead"): rejected-vs-ignored NOT documented — probe needed.
  per_model:
    - id: kimi-k2.7-code
      mode: always-on
      effort: "n/a (binary; {\"type\":\"enabled\"} with keep omitted, null, or \"all\" all accepted — all treated as keep:\"all\"; thinking cannot be disabled)"
    - id: kimi-k3
      mode: always-on
      effort: "adjustable: [\"low\", \"high\", \"max\"] via reasoning_effort (default \"max\")"
    - id: kimi-k2.7-code-highspeed
      mode: always-on
      effort: "n/a (same contract as kimi-k2.7-code)"
    - id: kimi-k2.6
      mode: optional
      effort: "n/a (toggle only: thinking.type enabled|disabled; keep \"all\"|null — null means historical thinking NOT preserved, \"all\" enables Preserved Thinking)"
  silent_substitutions: >
    none documented (no auto-routing aliases in the current catalog). moonshot-v1-* and kimi-k2.5 are NOT yet sunset: "no longer available to newly registered users (full platform sunset on August 31)" — still served to existing accounts until 2026-08-31; docs recommend explicit migration to kimi-k3.
    Semantics caveat, not substitution: templating kimi-k2.6's keep:null onto kimi-k2.7-code silently flips behavior from not-preserved to Preserved-Thinking-ON (server coerces null to "all").
  tools_and_streaming: >
    Streaming: "Streaming responses provide separate reasoning_content and final-answer content deltas" — read delta.reasoning_content alongside delta.content. Non-streaming: choices[0].message.reasoning_content "returned only when thinking mode is enabled".
    Tools: k2.7-code — "During multi-step tool calling, you must keep the reasoning_content from the assistant message in the current turn's tool call within the context, otherwise an error will be thrown"; with k2.6 keep:"all" and k2.7-code (always) you must "keep the reasoning_content from every historical assistant message in messages as-is". k2.7-code: tool_choice "can only be set to 'auto' or 'none'".
    Whether reasoning tokens count against max_completion_tokens: not documented (unknown).
    Constraints while thinking: kimi-k2.7-code(-highspeed): temperature fixed 1.0, top_p fixed 0.95, n fixed 1, penalties fixed 0.0 — each documented per-param with "Any other value will result in an error". kimi-k3: same fixed values but docs only say they "are fixed; omit them from requests" — error-on-send NOT documented for k3 (probe queued; omit regardless). kimi-k2.6: temperature "1.0 thinking / 0.6 non-thinking". Partial Mode must not be mixed with response_format={"type":"json_object"}.
  compat_passthrough: >
    openai-chat /v1 (.ai and .cn): native surface — passes; .cn docs (platform.kimi.com) document identical model ids and identical thinking/reasoning_effort contracts.
    anthropic-messages https://api.moonshot.ai/anthropic: thinking behavior passes through; kimi-k3 "thinks by default and works out of the box"; kimi-k2.7-code requests WITHOUT thinking enabled "are rejected with a '400 invalid thinking' error" (stricter than /v1, where omission defaults to enabled). Mapping of k3 reasoning_effort on this surface: unknown (docs set CLAUDE_CODE_EFFORT_LEVEL="max" client-side; exact wire field unverified). Existence of api.moonshot.cn/anthropic: unverified.
discovery:
  endpoint: GET https://api.moonshot.ai/v1/models (and GET https://api.moonshot.cn/v1/models)
  auth: bearer (API key)
  metadata: ids+capabilities — per model: id, object, created, owned_by, context_length (tokens), supports_image_in, supports_video_in, supports_reasoning. No pricing, no reasoning-param spelling.
  account_scoped: "inferred, not documented for the listing itself: the 404 resource_not_found_error ('model does not exist or your account does not have access') is documented for chat-completion CREATION, not GET /v1/models; whether the listing filters by account access is unprobed. .ai vs .cn rosters documented identical but served per-endpoint."
  rate_limits_or_caching: undocumented for this endpoint (account tiers Tier0-Tier5 gate RPM/TPM/TPD globally); docs note "the model list and capability flags may change as the platform is updated" and recommend querying before creating completions
  verdict: augment-static
  refresh_strategy: on-config-open (context_length + supports_reasoning are live and authoritative; reasoning param SPELLING per family — thinking vs reasoning_effort — is not in the listing and must stay static)
context:
  per_model:
    - id: kimi-k2.7-code
      context: 262,144 (marketed "256K"); no tier splits
      max_output: default 32768 ("Default to be 32k aka 32768"); documented maximum unpublished
      long_context_pricing: flat ($0.95/M input cache-miss, $0.19/M cache-hit, $4.00/M output; no boundary)
      overpromise_risk: no
    - id: kimi-k3
      context: 1,048,576 flat on this platform — no plan gating; pricing page states "1,048,576 tokens", single tier
      max_output: "max_completion_tokens defaults to 131072 and can be set up to 1048576"
      long_context_pricing: flat ($3.00/M input cache-miss, $0.30/M cache-hit, $15.00/M output; no boundary)
      overpromise_risk: "yes: Tier0 accounts (< $10 cumulative recharge) have TPM 500,000 and TPD 1,500,000 — a full 1M-token request cannot clear rate limiting, so headline 1M is unusable until Tier1"
    - id: kimi-k2.7-code-highspeed
      context: 262,144 (same as kimi-k2.7-code); no tier splits
      max_output: default 32768 (quickstart covers both k2.7-code models); documented maximum unpublished
      long_context_pricing: flat ($1.90/M input cache-miss, $0.38/M cache-hit, $8.00/M output — 2x standard; no boundary)
      overpromise_risk: "no on context; throughput marketed 'approximately 180 Tokens/s and up to 260 Tokens/s in short context scenarios' — treat as marketing figure, no SLA stated; a previously cited 'experience may slightly fluctuate' resource-limited caveat could not be found on any loadable page and was dropped"
    - id: kimi-k2.6
      context: 262,144; no tier splits
      max_output: unpublished
      long_context_pricing: flat ($0.95/M input cache-miss, $0.16/M cache-hit, $4.00/M output; no boundary)
      overpromise_risk: no
  token_counting_endpoint: POST https://api.moonshot.ai/v1/tokenizers/estimate-token-count (bearer auth; body {model, messages}; returns {"data":{"total_tokens":N}}); .cn equivalent assumed at api.moonshot.cn — unverified
probes_needed:
  - question: Exact status + error body when thinking is disabled on kimi-k2.7-code on the /v1 surface (docs confirm error, Claude Code page says "400 invalid thinking", but exact message/type string unprobed)
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/v1/chat/completions -H 'Authorization: Bearer $MOONSHOT_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k2.7-code\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking\":{\"type\":\"disabled\"}}'"
  - question: Does kimi-k3 REJECT a K2.x-style thinking object (http-400) or silently ignore it? Docs only say k3 "uses reasoning_effort instead"
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/v1/chat/completions -H 'Authorization: Bearer $MOONSHOT_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"thinking\":{\"type\":\"enabled\"}}'"
  - question: Does kimi-k3 reject the OpenAI-conventional reasoning_effort value "medium" (not in low|high|max) with 400, or coerce it?
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/v1/chat/completions -H 'Authorization: Bearer $MOONSHOT_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"medium\"}'"
  - question: Does kimi-k3 ERROR on a non-default temperature/top_p, or coerce/ignore it? (k3 quickstart only says the values "are fixed; omit them from requests" — the "Any other value will result in an error" sentence is documented for kimi-k2.7-code only)
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/v1/chat/completions -H 'Authorization: Bearer $MOONSHOT_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0.7}'"
  - question: Is reasoning_effort rejected or ignored when sent to a K2.x model (kimi-k2.6 / kimi-k2.7-code)?
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/v1/chat/completions -H 'Authorization: Bearer $MOONSHOT_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k2.6\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"high\"}'"
  - question: Is legacy max_tokens accepted on kimi-k3, or only max_completion_tokens (k3 quickstart documents only the latter)?
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/v1/chat/completions -H 'Authorization: Bearer $MOONSHOT_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":128}'"
  - question: Live roster + context_length parity between .ai and .cn, whether the listing varies by account tier, and whether sunset-bucket models (kimi-k2.5, moonshot-v1-*) still appear for existing accounts pre-2026-08-31
    probe: "curl -s https://api.moonshot.ai/v1/models -H 'Authorization: Bearer $MOONSHOT_API_KEY'; curl -s https://api.moonshot.cn/v1/models -H 'Authorization: Bearer $MOONSHOT_CN_API_KEY'"
  - question: On the anthropic-messages surface, does an omitted thinking block on kimi-k2.7-code really 400 (stricter than /v1 default-enabled), and how does k3 effort map?
    probe: "curl -s -w '\\n%{http_code}\\n' https://api.moonshot.ai/anthropic/v1/messages -H 'x-api-key: $MOONSHOT_API_KEY' -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d '{\"model\":\"kimi-k2.7-code\",\"max_tokens\":128,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
conflicts:
  - "Prior-refresh claim says kimi-k3 accepts only reasoning_effort:\"max\"; https://platform.kimi.ai/docs/guide/kimi-k3-quickstart and /docs/api/chat both say 'Thinking effort supports low, high, and max, with max as the default' — recorded, prior claim was too narrow."
  - "Prior-refresh claim says the k3 quickstart 'explicitly says not to send thinking to it'; the quickstart as fetched 2026-07-19 contains no such sentence (no mention of a thinking parameter at all) — recorded."
  - "https://platform.kimi.ai/docs/guide/claude-code-kimi.md says kimi-k2.7-code 'requires requests to explicitly enable thinking' (omission rejected 400 on the anthropic surface); https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart says on /v1 thinking is 'Default to be {\"type\": \"enabled\"}' (omission safe) — surface-dependent behavior, recorded not resolved."
  - "https://platform.kimi.ai/docs/models table shows k2.7-code context as '256k' marketing figure; pricing pages give exact 262,144 — consistent rounding, recorded for exactness."
notes: >
  CORRECTION PASS 2026-07-19: all six verifier challenges checked against the live docs and all six upheld; none rejected.
  (1) MAJOR fix: kimi-k2.7-code keep is COERCED, not rejected — /docs/api/chat: passing "all", passing null, or omitting it "all behave identically and are treated as 'all' on the server"; only "any other invalid value returns an error". The earlier 'only keep:"all" accepted' line and the matching 400 hazard were wrong and contradicted the contract's own default_when_omitted. The real trap is semantic, not HTTP: k2.6's keep:null means NOT preserved, k2.7-code's null is silently promoted to Preserved-Thinking-ON. Docs explicitly cover null, so no null-vs-omission probe is needed.
  (2) The "Any other value will result in an error" sentence for temperature/top_p/n/penalties is documented ONLY in the kimi-k2-7-code quickstart (verbatim, per-param). The k3 quickstart says only that the values "are fixed; omit them from requests" — k3's on-send behavior is unprobed (new probe added). Emitter guidance unchanged: OMIT sampling params for k3 and k2.7-code(-highspeed); kimi-k2.6 temperature is mode-coupled (1.0 thinking / 0.6 non-thinking).
  (3) moonshot-v1-* and kimi-k2.5 are NOT sunset yet: closed to newly registered users now, full platform sunset 2026-08-31, still served to existing accounts until then (/docs/models verbatim).
  (4) The high-speed "experience may slightly fluctuate" resource-limited quote could not be found on /docs/models, /docs/pricing/chat-k27-code, or the .cn chat docs — dropped; the sourced figure is "approximately 180 Tokens/s and up to 260 Tokens/s in short context scenarios".
  (5) errors.md 400 invalid_request_error verbatim is "Request format error, missing required parameter, or invalid parameter type. Check the request body against the API documentation." — earlier quoted paraphrase replaced.
  (6) account_scoped for GET /v1/models softened to inferred: list-models.md attaches the 404 access sentence to chat-completion creation, not the listing.
  .cn docs live at platform.kimi.com (platform.moonshot.cn 301s there); model ids and reasoning contracts verified identical to .ai. Do NOT conflate with api.kimi.com/coding/v1 (kimi-code membership preset, model "k3" — separate surface, out of scope here).
  Clients must round-trip reasoning_content on assistant tool-call messages (k2.7-code always; k2.6 when keep:"all") or the next request errors — OpenAI-SDK message-pruning breaks this.
  kimi-k2.5 exists (toggleable thinking, type field only, no keep) but is in the closed-to-new-users / 2026-08-31-sunset bucket and is not in the applied catalog. Error body shape platform-wide: {"error":{"type":...,"message":...}}; estimate endpoint errors add "code".
````

### Gap-fill addendum (post-verification)

A7 (wire field for kimi-k3 effort on the anthropic-messages surface) is now RESOLVED at the protocol level by cross-referencing two official sources; only the byte-for-byte confirmation on api.moonshot.ai/anthropic itself is left to a probe.

1) Moonshot's own Claude Code guide (https://platform.kimi.ai/docs/guide/claude-code-kimi.md, base URL ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic") does NOT document a wire field. It only sets three client-side env vars: ANTHROPIC_MODEL="kimi-k3[1m]", CLAUDE_CODE_EFFORT_LEVEL="max" ("enables thorough reasoning"), CLAUDE_CODE_AUTO_COMPACT_WINDOW="1048576". Model-behavior note verbatim: "kimi-k3: Thinking enabled by default; no additional configuration needed." The prior "truncates mid-sentence" impression is because the page genuinely stops there — there is no wire-field prose on this page.

2) The wire field is defined by the Anthropic Messages protocol, which Claude Code speaks to this endpoint. Per the official Anthropic effort spec (https://platform.claude.com/docs/en/build-with-claude/effort), effort is sent as top-level "output_config":{"effort":"..."} (NOT a "thinking" sub-field, NOT top-level reasoning_effort), values low|medium|high|xhigh|max, API default high; paired with adaptive thinking (thinking:{type:"adaptive"}). CLAUDE_CODE_EFFORT_LEVEL is the client default that Claude Code serializes into output_config.effort; /effort changes it at runtime. So on api.moonshot.ai/anthropic the request carries output_config.effort, and Moonshot's proxy must translate that 5-level scale down to kimi-k3's native 3-level reasoning_effort (which on /v1 is a top-level string low|high|max, default max — confirmed at https://platform.kimi.ai/docs/guide/kimi-k3-quickstart and https://platform.kimi.ai/docs/guide/use-thinking-effort.md, example {"model":"kimi-k3",...,"reasoning_effort":"high"}).

3) The exact translation table is documented on the sibling Kimi Code Anthropic endpoint doc (https://www.kimi.com/code/docs/en/third-party-tools/other-coding-agents.html, base URL api.kimi.com/coding/): Claude Code low->K3 low, medium->high, high->high, xhigh->max, max->max, not-set->high. Consequence: via the Anthropic/Claude-Code surface the EFFECTIVE default is "high", not kimi-k3's native /v1 default of "max" — which is exactly why the Moonshot guide tells you to set CLAUDE_CODE_EFFORT_LEVEL="max" explicitly. The same page also documents a silent route not in the current contract: "Disabling thinking routes both K3 and K2.7 Code to K2.6." (Caveat: that mapping/route is documented for api.kimi.com/coding; whether api.moonshot.ai/anthropic uses an identical table and identical disable-routing is not separately documented — hence the probe.)

A6 (reasoning tokens vs max_completion_tokens): still NOT explicitly documented by Moonshot. Positive evidence gathered: the Kimi chat usage object (https://platform.kimi.ai/docs/api/chat) has only {prompt_tokens, completion_tokens, total_tokens, cached_tokens} — there is NO separate reasoning_tokens bucket and NO completion_tokens_details. Since kimi-k3 "always enables thinking with Preserved Thinking" (reasoning_effort enum [low,high,max] default max, per the chat API model reference) yet exposes no separate reasoning bucket, reasoning_content is folded into completion_tokens, which strongly implies thinking tokens DO count against max_completion_tokens/max_tokens. Anthropic's own effort doc corroborates the pattern for the protocol ("max_tokens ... is a hard limit on total output, thinking plus response text"). This remains inference, not an explicit Moonshot statement, and truncation-under-small-max_tokens is unprobed.

Sources opened: https://platform.kimi.ai/docs/guide/claude-code-kimi.md ; https://platform.claude.com/docs/en/build-with-claude/effort ; https://platform.kimi.ai/docs/guide/kimi-k3-quickstart ; https://platform.kimi.ai/docs/guide/use-thinking-effort.md ; https://platform.kimi.ai/docs/api/chat ; https://www.kimi.com/code/docs/en/third-party-tools/other-coding-agents.html

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  tools_and_streaming: >
    Whether reasoning tokens count against max_completion_tokens: NOT explicitly documented, but strongly implied to count. The Kimi usage object (docs/api/chat) exposes only {prompt_tokens, completion_tokens, total_tokens, cached_tokens} — no separate reasoning_tokens bucket and no completion_tokens_details. kimi-k3 "always enables thinking with Preserved Thinking" yet returns no separate reasoning bucket, so reasoning_content is folded into completion_tokens, implying thinking tokens ARE billed within and capped by max_completion_tokens/max_tokens. Anthropic's own effort spec corroborates the pattern ("max_tokens ... is a hard limit on total output, thinking plus response text"). Truncation-under-small-max_tokens unprobed.
  compat_passthrough: >
    anthropic-messages https://api.moonshot.ai/anthropic: kimi-k3 effort is NOT carried as top-level reasoning_effort or as a thinking sub-field on this surface. Claude Code speaks the Anthropic Messages protocol, so effort travels as the standard Anthropic wire field output_config.effort (5-level scale low|medium|high|xhigh|max, protocol default high), alongside thinking:{type:"adaptive"}; CLAUDE_CODE_EFFORT_LEVEL is only the client default that Claude Code serializes into output_config.effort (/effort changes it at runtime). Moonshot's proxy translates output_config.effort down to kimi-k3's native 3-level reasoning_effort (low|high|max). Documented mapping (on the sibling api.kimi.com/coding endpoint, docs/en/third-party-tools/other-coding-agents): low->low, medium->high, high->high, xhigh->max, max->max, not-set->high. NET EFFECT: via the Anthropic/Claude-Code surface the effective K3 default is "high", NOT kimi-k3's native /v1 default of "max" — set CLAUDE_CODE_EFFORT_LEVEL="max" explicitly to get max. Whether api.moonshot.ai/anthropic uses this identical table is not separately documented (verified only for api.kimi.com/coding) — probe. Same coding-endpoint doc adds a silent route not previously in this contract: "Disabling thinking routes both K3 and K2.7 Code to K2.6" — i.e. on the coding surface a thinking-disabled request is silently re-routed to kimi-k2.6, contrasting the platform.kimi.ai claim that k2.7-code without thinking is rejected 400; surface-specific, unverified for api.moonshot.ai/anthropic.
````

**Still unresolved (needs a live key):**

````
Byte-level confirmation on api.moonshot.ai/anthropic is unprobed (no API key available to this run). Maintainer probe A (accepted wire field + effort mapping + usage accounting):

curl -sD - https://api.moonshot.ai/anthropic/v1/messages \
  -H "x-api-key: $MOONSHOT_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"kimi-k3[1m]","max_tokens":64,"thinking":{"type":"adaptive"},"output_config":{"effort":"low"},"messages":[{"role":"user","content":"Derive the closed form for 1,4,9,25,64,... and show your reasoning."}]}'

Expect 200. In the response inspect: (i) usage.* fields — confirm whether any reasoning/thinking bucket appears or whether output_tokens includes thinking (A6); (ii) that low effort actually shortens thinking vs a second call with output_config.effort:"max". Probe B — confirm output_config.effort is the honored field (not a passthrough): repeat with the K2-native form to see if it is accepted, ignored, or 400'd on this surface: replace the effort/thinking keys with top-level "reasoning_effort":"low". Probe C — A6 truncation: set "max_tokens":8 with output_config.effort:"max" and check whether the response is stop_reason:"max_tokens" mid-thinking (proves thinking counts against the output cap). Probe D — disable-routing: send "thinking":{"type":"disabled"} with model kimi-k3[1m] and observe whether it 400s or is silently served/routed to kimi-k2.6 (check the response's model field).
````

---

## 8. kimi-code

*Verification: verified-clean — 4 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: kimi-code
endpoints:
  - url: https://api.kimi.com/coding/v1
    protocol: openai-chat
  - url: https://api.kimi.com/coding/
    protocol: anthropic-messages
checked: 2026-07-19
sources:
  - https://www.kimi.com/code/docs/en/kimi-code/models
  - https://www.kimi.com/code/docs/en/kimi-code/models.html
  - https://www.kimi.com/code/docs/en/
  - https://www.kimi.com/code/docs/en/third-party-tools/other-coding-agents.html
  - https://www.kimi.com/code/docs/en/kimi-code/membership.html
  - https://www.kimi.com/code/docs/en/kimi-code/whats-new.html
  - https://platform.kimi.ai/docs/guide/use-thinking-effort
  - https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model
  - https://www.kimi.com/code/docs/en/third-party-tools/claude-code.html (FAILED)
reasoning:
  param: "reasoning_effort" (top-level, k3 only, openai-chat surface); "thinking" object {"type": "enabled"|"disabled", "keep": null|"all"} (K2.x family semantics per platform docs; the coding-surface kimi-for-coding ids are K2.7 Code); anthropic-messages surface control is NOT documented as a request param (Claude Code controls it via the client-side /effort command) — unknown, probe needed
  values: "k3: reasoning_effort one of \"low\" / \"high\" / \"max\" (exact strings). kimi-for-coding & kimi-for-coding-highspeed: no reasoning_effort documented; thinking.type \"enabled\"|\"disabled\", thinking.keep only \"all\" valid (K2.7 Code treats keep as \"all\" always)"
  default_when_omitted: "k3 on kimi-code surface: reasoning_effort defaults to \"high\" (kimi.com/code models page) — CONFLICT: platform.kimi.ai says kimi-k3 default is \"max\"; kimi-for-coding / -highspeed: Thinking:ON by default; k3 thinking itself is always on (\"K3 always has thinking and Preserved Thinking enabled\")"
  unsupported_value_behavior: "http-400 for unknown reasoning_effort value on k3 (\"any other unknown -> HTTP 400 error\"; error body shape undocumented); invalid thinking.keep value \"errors\" (code/body undocumented); thinking.type \"disabled\" on the coding surface is NOT an error — it silently reroutes (see silent_substitutions); on platform.kimi.ai, kimi-k2.7-code with thinking disabled ERRORS instead (surface divergence, recorded in conflicts)"
  per_model:
    - id: k3
      mode: always-on
      effort: "adjustable: [\"low\", \"high\", \"max\"] via top-level reasoning_effort (openai-chat surface); default high per kimi-code docs (platform says max — conflict)"
    - id: kimi-for-coding
      mode: always-on
      effort: "fixed: Thinking:ON, no effort knob documented; sending thinking disabled is a silent model swap, not a speed toggle"
    - id: kimi-for-coding-highspeed
      mode: always-on
      effort: "fixed: same Thinking:ON semantics as kimi-for-coding; ~5-6x output speed; Allegretto+ only"
  silent_substitutions: "Disabling thinking routes the request to K2.6 — applies to BOTH k3 and kimi-for-coding (exact doc wording: \"keep Thinking on to use K3 or K2.7 Code; disabling thinking routes the request to K2.6\" and \"Disabling thinking routes both K3 and K2.7 Code to K2.6\"). This is a model substitution, not degraded output. Also: \"k3[1m]\" is a Claude Code ANTHROPIC_MODEL env-var-only spelling (\"exclusively in Claude Code environment variables\"); in request bodies always use plain k3 — k3[1m] in a body is not a valid model id"
  tools_and_streaming: "Reasoning arrives in reasoning_content (openai-chat surface); in streaming it always appears before content in the delta and must be read via hasattr/getattr (not a declared SDK attribute). Reasoning tokens COUNT against max_tokens: sum of reasoning_content + content tokens must be <= max_tokens (platform doc; coding surface presumed same — not separately restated). Tool loops: the complete assistant message must be passed back to messages as-is including reasoning_content and tool_calls (Preserved Thinking); K2.7 Code and K3 keep reasoning across turns always (keep treated as \"all\"). Anthropic surface supports thinking content blocks alongside tool_use/tool_result per protocol; exact SSE block naming on /coding/v1/messages undocumented"
  compat_passthrough: "openai-chat (/coding/v1/chat/completions): reasoning_effort and thinking object are the documented native spellings — passes. anthropic-messages (/coding/v1/messages, base ANTHROPIC_BASE_URL=https://api.kimi.com/coding/): no request-body reasoning param documented at all; whether Anthropic's thinking: {type, budget_tokens} maps, is stripped, or errors is unknown — probe needed. Claude Code effort is set client-side via /effort (values low|medium|high|xhigh|max) and mapped to K3 levels by the client, not a documented server param"
discovery:
  endpoint: none
  auth: "n/a (no listing endpoint documented; API keys minted in Kimi Code Console; Anthropic surface authenticates via ANTHROPIC_API_KEY)"
  metadata: "n/a — no /models documented anywhere in the kimi-code docs; existence can only be settled by an authenticated probe"
  account_scoped: "yes: model availability is membership-tier scoped — Andante: kimi-for-coding only; Moderato: k3 + kimi-for-coding; Allegretto+: all three and the k3 1M context unlock"
  rate_limits_or_caching: "roughly 300-1,200 requests per rolling 5-hour window depending on tier, up to 30 concurrent requests; usage quota refreshes every 7 days; Extra Usage top-up balance (min CNY 25, max CNY 10,000); no caching headers documented"
  verdict: stay-static
  refresh_strategy: "n/a (stay-static). If the /models probe finds a live endpoint, upgrade to augment-static with on-config-open refresh; until then validate a chosen id lazily on first real call (a dedicated validation call is billed against membership quota)"
context:
  per_model:
    - id: k3
      context: "tier split: Moderato = 262144 (256K); Allegretto and above = 1048576 (1M). Andante keys cannot use k3 at all"
      max_output: unpublished
      long_context_pricing: "flat within membership quota (subscription model, no per-token boundary pricing published); overage via Extra Usage balance"
      overpromise_risk: "yes: hardcoding 1048576 over-promises for Moderato keys (256K actual) and k3 is entirely unavailable to Andante keys"
    - id: kimi-for-coding
      context: "262144 (256K), all tiers (Andante and above)"
      max_output: unpublished
      long_context_pricing: flat (membership quota)
      overpromise_risk: no
    - id: kimi-for-coding-highspeed
      context: "262144 (256K); model itself is Allegretto+ only"
      max_output: unpublished
      long_context_pricing: flat (membership quota)
      overpromise_risk: "yes: Andante/Moderato keys get an API error (shape undocumented), not a smaller context — listing it for those accounts over-promises the model's existence"
  token_counting_endpoint: none
probes_needed:
  - question: "Does GET /models exist on the coding surface (settles discovery verdict and free validation)?"
    probe: "curl -s -X GET https://api.kimi.com/coding/v1/models -H \"Authorization: Bearer $KIMI_CODE_API_KEY\""
  - question: "Exact 400 error body shape for an unsupported reasoning_effort value on k3 (docs say HTTP 400, body undocumented)"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/chat/completions -H \"Authorization: Bearer $KIMI_CODE_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"k3\",\"reasoning_effort\":\"medium\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'"
  - question: "Is reasoning_effort on kimi-for-coding silently ignored or a 400 (undocumented)?"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/chat/completions -H \"Authorization: Bearer $KIMI_CODE_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"kimi-for-coding\",\"reasoning_effort\":\"high\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'"
  - question: "Does thinking:{\"type\":\"disabled\"} on the coding surface really return a K2.6-served response (check the response model field to observe the silent swap), and does the same happen on the anthropic surface?"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/chat/completions -H \"Authorization: Bearer $KIMI_CODE_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"kimi-for-coding\",\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'"
  - question: "Does the anthropic-messages surface accept Anthropic's thinking param (and what happens with type:disabled — strip, error, or K2.6 route)?"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/messages -H \"x-api-key: $KIMI_CODE_API_KEY\" -H \"anthropic-version: 2023-06-01\" -H \"Content-Type: application/json\" -d '{\"model\":\"k3\",\"max_tokens\":64,\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Exact error (status + body) a Moderato key gets calling kimi-for-coding-highspeed (tier gate shape undocumented)"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/chat/completions -H \"Authorization: Bearer $MODERATO_TIER_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"kimi-for-coding-highspeed\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'"
  - question: "How the 256K tier cap surfaces for a Moderato key sending >262144 tokens to k3 (hard error vs truncation), needed for honest context accounting"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/chat/completions -H \"Authorization: Bearer $MODERATO_TIER_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"k3\",\"messages\":[{\"role\":\"user\",\"content\":\"<~300k-token payload>\"}],\"max_tokens\":16}'"
  - question: "Does a token-counting endpoint exist on the anthropic surface (count_tokens is standard Anthropic protocol)?"
    probe: "curl -s -X POST https://api.kimi.com/coding/v1/messages/count_tokens -H \"x-api-key: $KIMI_CODE_API_KEY\" -H \"anthropic-version: 2023-06-01\" -H \"Content-Type: application/json\" -d '{\"model\":\"k3\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
conflicts:
  - "https://www.kimi.com/code/docs/en/kimi-code/models.html says k3 reasoning_effort default is \"high\"; https://platform.kimi.ai/docs/guide/use-thinking-effort says kimi-k3 default is \"max\" — both official, different surfaces (kimi-code vs Moonshot platform); recorded, not resolved"
  - "https://www.kimi.com/code/docs/en/kimi-code/models.html says disabling thinking routes K2.7 Code (and K3) to K2.6 (silent substitution); https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model says on the platform surface kimi-k2.7-code with thinking \"disabled\" ERRORS (\"Passing \\\"disabled\\\" errors\") — surface-divergent behavior; recorded, not resolved"
  - "models.dev previously listed a k2p7 id for this surface — known-wrong per prior verification; official coding ids remain k3 / kimi-for-coding / kimi-for-coding-highspeed; not resurrected"
notes: "Membership keys minted in the Kimi Code Console are valid ONLY on api.kimi.com/coding endpoints, not platform.moonshot/platform.kimi.ai. The Anthropic-compatible base for clients is https://api.kimi.com/coding/ with clients appending v1/messages (full path https://api.kimi.com/coding/v1/messages). \"k3[1m]\" (quoted, brackets) is exclusively a Claude Code environment-variable spelling to request 1M context; every other context (request bodies, third-party tool model fields) must use plain k3. Claude Code env recipe: ANTHROPIC_BASE_URL=https://api.kimi.com/coding/, ANTHROPIC_API_KEY=<console key>, ANTHROPIC_MODEL=<id>, CLAUDE_CODE_EFFORT_LEVEL=high, plus CLAUDE_CODE_AUTO_COMPACT_WINDOW / CLAUDE_CODE_MAX_CONTEXT_TOKENS matched to the tier's real context. Kimi Code CLI keeps reasoning across turns by default since v0.23.0 ([thinking] keep = \"off\" to opt out — CLI config, not an API param). The dedicated claude-code.html doc page 404s; Claude Code facts above come from other-coding-agents.html. Timeline: K2.7 Code 2026-06-12, HighSpeed 2026-07-09, K3 2026-07-16."
````

### Gap-fill addendum (post-verification)

Live docs settle A2 and mostly settle A7; one raw-body probe remains for A7.

A2 (default): The "high" vs "max" conflict is a SURFACE DIVERGENCE, not a genuine conflict — resolved to "high" for the kimi-code coding surface. Two coding-surface docs agree:
- https://www.kimi.com/code/docs/en/kimi-code/models : k3 supports reasoning_effort "low"/"high"/"max" with (default "high").
- https://www.kimi.com/code/docs/en/third-party-tools/other-coding-agents.html : the Claude-Code→K3 effort mapping table ends with "Not set (default) → high", and documents env var CLAUDE_CODE_EFFORT_LEVEL as "Default thinking effort level; set to high".
The "max" default is stated only at https://platform.kimi.ai/docs/guide/use-thinking-effort ("supports low, high, and max, with max as the default") — that is platform.kimi.ai's separate general API surface (NOT api.kimi.com/coding). So for provider kimi-code the default_when_omitted is "high"; the "max" default does not apply to this provider's endpoints.

A7 (Anthropic-surface control): No request-body reasoning parameter is documented on the anthropic-messages surface (base ANTHROPIC_BASE_URL=https://api.kimi.com/coding/). Effort there is controlled ENTIRELY CLIENT-SIDE, and this is now explicitly documented at other-coding-agents.html:
- env var CLAUDE_CODE_EFFORT_LEVEL (values low/high/max, default high), shown in the exact setup block: export ANTHROPIC_BASE_URL=https://api.kimi.com/coding/ ; export ANTHROPIC_MODEL="k3[1m]" ; export CLAUDE_CODE_EFFORT_LEVEL=high  (note: this corroborates the k3[1m] env-var-only model spelling already in the contract).
- in-session command "/effort" ("type /effort in a session to switch thinking effort levels — no environment variable needed").
- The client maps Claude-Code levels to K3 levels: low→low, medium→high, high→high, xhigh→max, max→max, unset→high.
- Thinking-disable behavior restated verbatim: "Disabling thinking routes both K3 and K2.7 Code to K2.6. Keep Thinking on to use K3 / K2.7."
What remains UNKNOWN from docs: whether a RAW Anthropic thinking:{type,budget_tokens} object placed directly in a /coding/v1/messages request body maps, is silently stripped, or errors. The docs only describe the client-injected control path, never the raw-body contract — so this specific question still needs the probe below.
Note: https://www.kimi.com/code/docs/en/third-party-tools/claude-code.html now returns HTTP 404 (content folded into other-coding-agents.html).

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  default_when_omitted: "k3 on kimi-code coding surface: reasoning_effort defaults to \"high\" (CONFIRMED by two coding-surface docs: kimi.com/code models page \"(default \\\"high\\\")\" and other-coding-agents.html mapping \"Not set (default) → high\" + CLAUDE_CODE_EFFORT_LEVEL \"set to high\"). The \"max\" default is platform.kimi.ai's SEPARATE general API surface only, NOT api.kimi.com/coding — no longer a conflict for this provider. kimi-for-coding / -highspeed: Thinking:ON by default; k3 thinking itself is always on"
  per_model:
    - id: k3
      mode: always-on
      effort: "adjustable: [\"low\", \"high\", \"max\"] via top-level reasoning_effort (openai-chat surface); default \"high\" on the kimi-code coding surface (settled — platform.kimi.ai's \"max\" default is a different surface, not this provider)"
  compat_passthrough: "openai-chat (/coding/v1/chat/completions): reasoning_effort and thinking object are the documented native spellings — passes. anthropic-messages (/coding/v1/messages, base ANTHROPIC_BASE_URL=https://api.kimi.com/coding/): NO request-body reasoning param documented. Effort is controlled entirely CLIENT-SIDE and is now explicitly documented: env var CLAUDE_CODE_EFFORT_LEVEL (low|high|max, default high) plus in-session /effort. Client maps Claude-Code effort→K3: low→low, medium→high, high→high, xhigh→max, max→max, unset→high. Whether a RAW Anthropic thinking:{type,budget_tokens} object in a /coding/v1/messages body maps/strips/errors is still undocumented — probe needed"
sources:
  - https://www.kimi.com/code/docs/en/kimi-code/models
  - https://www.kimi.com/code/docs/en/third-party-tools/other-coding-agents.html
  - https://platform.kimi.ai/docs/guide/use-thinking-effort
  - https://www.kimi.com/code/docs/en/third-party-tools/claude-code.html (now HTTP 404 — content merged into other-coding-agents.html)
checked: 2026-07-19
````

**Still unresolved (needs a live key):**

````
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://api.kimi.com/coding/v1/messages -H "Authorization: Bearer $KIMI_CODE_API_KEY" -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" -d '{"model":"k3","max_tokens":64,"thinking":{"type":"enabled","budget_tokens":32},"messages":[{"role":"user","content":"hi"}]}'  # then re-run with -o - (full body) to observe: does a RAW Anthropic thinking object MAP (response contains a thinking content block) / STRIP (200 OK, no thinking block, field ignored) / ERROR (HTTP 400)? Docs only cover the client-side CLAUDE_CODE_EFFORT_LEVEL + /effort path, never the raw request-body contract. A2 default ("high" on the coding surface) is settled from docs and needs no probe.
````

### Verifier notes (minor, not applied)

- **discovery.rate_limits_or_caching: "roughly 300-1,200 requests per rolling 5-hour window depending on tier, up to 30 concurrent requests"** — The '300-1,200 requests' band and the '30 concurrent requests' figure could not be verified. membership.html confirms only the 'rolling 5-hour rate window' phrasing and the 7-day refresh, and explicitly defers all per-tier request numbers to the membership pricing page (https://www.kimi.com/membership/pricing), which did not render these figures on fetch. Unverifiable via fetch — presented as roug *Suggested: Attribute the 300-1,200 / 30-concurrent numbers to the pricing page explicitly, or mark them unverified/probe-needed rather than stating them as roughly-fact.*
- **endpoints: openai-chat surface url https://api.kimi.com/coding/v1 (chat completions at /coding/v1/chat/completions)** — The Anthropic base https://api.kimi.com/coding/ is directly confirmed in other-coding-agents.html (export ANTHROPIC_BASE_URL=https://api.kimi.com/coding/), but the OpenAI-compatible base URL and /v1/chat/completions path are not stated in any doc that loaded for me. The value is internally consistent and matches the standard Kimi convention, but is inferred rather than doc-confirmed. Unverifiable  *Suggested: Cite the specific doc that gives the OpenAI base path, or flag the openai-chat endpoint URL as inferred/probe-needed.*
- **notes: Extra Usage top-up balance (min CNY 25, max CNY 10,000)** — Numbers are correct but incomplete. membership.html states 'minimum ¥25 per top-up, up to 10 times and ¥3,000 per day, with a balance cap of ¥10,000' — the daily ceiling (¥3,000/day, max 10 top-ups/day) is omitted. Not wrong, just partial. *Suggested: Add the ¥3,000/day and 10-top-ups/day daily limits alongside the min ¥25 / cap ¥10,000.*
- **notes: "Kimi Code CLI keeps reasoning across turns by default since v0.23.0 ([thinking] keep = \"off\" to opt out — CLI config, not an API param)"** — The v0.23.0 version boundary and the '[thinking] keep = off' CLI-config opt-out were not present in any doc that loaded for me (models.html, other-coding-agents.html, membership.html, whats-new.html). Unverifiable via fetch. *Suggested: Cite the CLI doc/changelog that states the v0.23.0 default and the keep=off opt-out, or mark it unverified.*

---

## 9. minimax

*Verification: verified-clean — 1 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: minimax (MiniMax-M3, MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.5)
endpoints:
  - url: https://api.minimax.io/v1
    protocol: openai-chat
  - url: https://api.minimax.io/anthropic
    protocol: anthropic-messages
  - url: https://api.minimaxi.com/v1 (CN mirror; anthropic mirror https://api.minimaxi.com/anthropic)
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://platform.minimax.io/docs/guides/models-intro
  - https://platform.minimax.io/docs/guides/text-generation
  - https://platform.minimax.io/docs/guides/text-generation.md
  - https://platform.minimax.io/docs/api-reference
  - https://platform.minimax.io/docs/api-reference/text-chat-openai.md
  - https://platform.minimax.io/docs/api-reference/text-chat-anthropic.md
  - https://platform.minimax.io/docs/guides/text-m3-function-call.md
  - https://platform.minimax.io/docs/api-reference/models/openai/list-models.md
  - https://platform.minimax.io/docs/api-reference/models/anthropic/list-models.md
  - https://platform.minimax.io/docs/guides/pricing-paygo.md
  - https://platform.minimax.io/docs/llms.txt
  - https://www.minimax.io/models/text/m3
  - https://www.minimax.io/blog/minimax-m3
reasoning:
  param: >-
    thinking (object) on BOTH the openai-chat and anthropic-messages surfaces —
    shape { "type": "<value>" }. Plus openai-chat-only reasoning_split (boolean):
    it does NOT toggle reasoning on/off, it only changes how thinking is
    surfaced (True -> separated into reasoning_content / reasoning_details
    fields; False -> embedded as <think>...</think> tags inside content).
  values: >-
    thinking.type per official api-reference: "disabled" | "adaptive" (M3 only).
    "disabled" answers directly / suppresses thinking; "adaptive" lets M3 decide
    when to reason. NOTE a third value "enabled" (always-on) is claimed by
    third-party/registry summaries but is NOT in the official enum — see
    conflicts. reasoning_split accepts true | false.
  default_when_omitted: >-
    SURFACE-DEPENDENT (this is the load-bearing gotcha). openai-chat (/v1):
    omitting thinking => adaptive (thinking ON by default). anthropic-messages
    (/anthropic): omitting thinking => disabled (thinking OFF by default).
    For M2.x models thinking is always active regardless of the param.
  unsupported_value_behavior: >-
    unknown / undocumented. Docs do not state the HTTP code or error body when
    an out-of-enum thinking.type (e.g. "enabled", "high") or an Anthropic-style
    budget_tokens is sent, nor when thinking is sent to an M2.x model (M2.x
    docs merely say thinking "remains enabled regardless", implying the value is
    ignored rather than erroring). Needs an authed probe (see probes_needed).
  per_model:
    - id: MiniMax-M3
      mode: optional
      effort: "adjustable: [disabled, adaptive] via thinking.type (no numeric/effort ladder; adaptive = model self-decides depth)"
    - id: MiniMax-M2.7
      mode: always-on
      effort: n/a (thinking always active; thinking param ignored)
    - id: MiniMax-M2.7-highspeed
      mode: always-on
      effort: n/a (thinking always active; latency-optimized variant, same reasoning behavior as M2.7)
    - id: MiniMax-M2.5
      mode: always-on
      effort: n/a (thinking always active; thinking param ignored)
  silent_substitutions: >-
    none documented. Reasoning flags do not swap you to a different model id;
    the -highspeed suffix is a separate explicit model id, not a flag-triggered
    swap.
  tools_and_streaming: >-
    Interleaved thinking with tools is a headline M3 feature on the
    anthropic-messages surface: the model reflects before each Tool Use, thinking
    appears as content blocks type:"thinking" with a signature field, and the
    ENTIRE assistant response (including thinking blocks) MUST be preserved and
    passed back in history to keep reasoning continuity across tool turns.
    Thinking blocks are emitted during streaming (Anthropic SSE thinking/
    signature deltas; on openai-chat they arrive as reasoning_content /
    reasoning_details when reasoning_split=true, else inline <think> tags).
    Whether thinking tokens count against the max output budget is NOT documented
    for M3 (unpublished); note the sibling M2 spec explicitly states its 128k max
    output is "including CoT", so treat M3 reasoning as counting against
    max_completion_tokens / max_tokens until confirmed.
  compat_passthrough: >-
    openai-chat (/v1): thinking passes (default adaptive), reasoning_split passes.
    anthropic-messages (/anthropic): thinking passes (default disabled),
    reasoning_split is an OpenAI-only field -> treat as stripped/ignored on this
    surface. A third Responses API surface exists
    (/docs/api-reference/responses-create) but is outside the two cataloged
    endpoints: unknown. CN mirror (api.minimaxi.com): assumed same param shape as
    api.minimax.io but not separately verified this session -> unknown.
discovery:
  endpoint: >-
    openai-chat: GET https://api.minimax.io/v1/models .
    anthropic-messages: GET https://api.minimax.io/anthropic/v1/models
  auth: >-
    openai surface: bearer (HTTP Bearer Auth, JWT API key). anthropic surface:
    api-key header named X-Api-Key.
  metadata: >-
    ids-only. openai /v1/models returns {id, object:"model", created (unix int),
    owned_by}. anthropic /anthropic/v1/models returns {id, display_name,
    created_at (ISO 8601), type:"model"} with first_id/last_id/has_more paging.
    NEITHER returns context window, max output, capabilities, or pricing.
  account_scoped: >-
    unknown — docs show a static sample list and do not state whether the
    returned model set varies by account tier/entitlement. Assume the list can
    be entitlement-filtered until confirmed.
  rate_limits_or_caching: undocumented (no rate-limit or cache headers specified for the models endpoint)
  verdict: augment-static
  refresh_strategy: >-
    on-config-open (or daily-cache) for model-id presence/availability only;
    keep context windows, max-output ceilings, reasoning modes and pricing as
    curated static facts because the discovery endpoint carries none of them.
context:
  per_model:
    - id: MiniMax-M3
      context: >-
        up to 1,000,000 tokens with a guaranteed minimum of 512K (524,288).
        Tier split is a BILLING boundary on INPUT tokens: <=512K input billed at
        standard rate, >512K input billed at higher long-context rate.
      max_output: >-
        max_completion_tokens (openai) / max_tokens (anthropic): maximum 524,288
        (512K), recommended 131,072 (128K). (openai legacy max_tokens deprecated
        in favor of max_completion_tokens.)
      long_context_pricing: >-
        boundary at 512K input tokens. Standard: <=512K input $0.30 /M, output
        $1.20 /M (cache read $0.06 /M); >512K input $0.60 /M, output $2.40 /M
        (cache read $0.12 /M). Priority (service_tier=priority) = 1.5x:
        <=512K input $0.45 /M, output $1.80 /M; >512K input $0.90 /M, output
        $3.60 /M. (page states M3 rates reflect a permanent 50% discount.)
      overpromise_risk: >-
        yes — the headline "1M" is only a guaranteed-minimum-512K contract; docs
        do not confirm every account/request between 512K and 1M is accepted
        (the guarantee is 512K, 1M is "up to"). Hardcoding 1M as usable context
        may over-promise; also >512K silently costs 2x. Treat 512K as the safe
        floor.
    - id: MiniMax-M2.7
      context: "204,800 tokens (no tier split)"
      max_output: "max 204,800 (200K), recommended 65,536 (64K)"
      long_context_pricing: "flat: input $0.3 /M, output $1.2 /M (cache read $0.06 /M, cache write $0.375 /M)"
      overpromise_risk: no
    - id: MiniMax-M2.7-highspeed
      context: "204,800 tokens (no tier split)"
      max_output: "max 204,800 (200K), recommended 65,536 (64K)"
      long_context_pricing: "flat: input $0.6 /M, output $2.4 /M (cache read $0.06 /M, cache write $0.375 /M) — 2x the base M2.7 rate for the low-latency variant"
      overpromise_risk: no
    - id: MiniMax-M2.5
      context: "204,800 tokens (no tier split)"
      max_output: "max 204,800 (200K), recommended 65,536 (64K)"
      long_context_pricing: "unpublished — M2.5 (legacy) not itemized on the pay-as-you-go page fetched this session"
      overpromise_risk: no
  token_counting_endpoint: none (no dedicated count-tokens endpoint documented for either surface)
probes_needed:
  - question: >-
      What is the failure mode of an out-of-enum thinking.type value on each
      surface (does "enabled" or an Anthropic budget_tokens object 400, or get
      coerced/ignored)? This decides whether a wrong reasoning flag is a hard
      production error.
    probe: >-
      curl -X POST https://api.minimax.io/v1/chat/completions
      -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json"
      -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"},"max_completion_tokens":64}'
  - question: >-
      Does the anthropic surface reject a Claude-style extended-thinking object
      (type:"enabled" + budget_tokens), and does omitting thinking really leave
      reasoning OFF for M3?
    probe: >-
      curl -X POST https://api.minimax.io/anthropic/v1/messages
      -H "X-Api-Key: $MINIMAX_API_KEY" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json"
      -d '{"model":"MiniMax-M3","max_tokens":128,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":1024}}'
  - question: >-
      Is a request with 512K<input<=1M ever rejected on some accounts (testing
      the "guaranteed minimum 512K vs up-to-1M" gap), and is it silently billed
      at the 2x long-context rate?
    probe: >-
      curl -X POST https://api.minimax.io/v1/chat/completions
      -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json"
      -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"<~600K-token padded input>"}],"max_completion_tokens":64}'
  - question: >-
      Does sending an OpenAI o-series reasoning_effort field to /v1 error (400)
      or get silently ignored (MiniMax uses thinking, not reasoning_effort)?
    probe: >-
      curl -X POST https://api.minimax.io/v1/chat/completions
      -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json"
      -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high","max_completion_tokens":64}'
  - question: >-
      Do the discovery endpoints return an entitlement-filtered list (account-
      scoped) or the full catalog?
    probe: >-
      curl https://api.minimax.io/v1/models -H "Authorization: Bearer $MINIMAX_API_KEY"
conflicts:
  - >-
    Official api-reference (text-chat-openai.md / text-chat-anthropic.md) lists
    thinking.type enum as ONLY {disabled, adaptive}; third-party/registry-style
    summaries surfaced via web search (minimax-ai.chat, aggregator pages) claim
    THREE modes {enabled, adaptive, disabled}. Official wins; "enabled" treated
    as unverified. Recorded, not resolved.
  - >-
    Default-when-omitted differs by surface per official reference (openai =>
    adaptive/ON, anthropic => disabled/OFF). Some third-party writeups describe
    a single "off by default" behavior; the split is the authoritative fact.
  - >-
    models-intro marketing page leaves M3 max output "Not specified" while
    text-chat-openai.md gives max 524,288 (recommended 131,072). Reference
    number used.
notes: >-
  (1) The prior-refresh claim "NO reasoning/thinking toggle on ANY MiniMax
  model" is REFUTED: M3 exposes thinking.type={disabled|adaptive} on both
  surfaces plus openai-only reasoning_split. It is still TRUE for M2.x
  (M2.7/M2.7-highspeed/M2.5), which are always-on with no exposed switch.
  (2) Biggest runtime trap for a generic agentic CLI: the surface-dependent
  default. A Claude-shaped client hitting /anthropic and omitting thinking gets
  M3 reasoning OFF (silent quality regression), while the same omission on /v1
  gets adaptive/ON. Sylliptor should set thinking explicitly per intent, never
  rely on the default.
  (3) openai route param is max_completion_tokens (max_tokens deprecated);
  anthropic route param is max_tokens. Emitting the wrong one per surface risks
  an ignored/clamped ceiling.
  (4) reasoning_split is an OUTPUT-FORMAT flag, not an on/off; do not treat it
  as a reasoning enable.
  (5) service_tier=priority is a 1.5x cost multiplier, default standard.
````

### Gap-fill addendum (post-verification)

The A2/A3 enum conflict is settled on the documentation side, and it inverts the current YAML's assumption. `enabled` is NOT a third-party invention — it is documented by MiniMax's OWN first-party model sources:

1. HuggingFace model card https://huggingface.co/MiniMaxAI/MiniMax-M3 and GitHub README https://github.com/MiniMax-AI/MiniMax-M3 both state M3 "supports three reasoning modes through the `thinking` parameter": `enabled` — "Reasoning is always enabled."; `adaptive` — "M3 automatically determines when additional reasoning is beneficial."; `disabled` — "Reasoning is disabled to minimize latency and maximize throughput." (verbatim from the raw README at https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/README.md).
2. The platform api-reference pages https://platform.minimax.io/docs/api-reference/text-chat-openai.md and .../text-chat-anthropic.md list only `disabled` | `adaptive` in the parameter enum — they simply omit `enabled`. Both confirm the surface-split default: openai-chat omits => adaptive ON; anthropic-messages "When omitted, thinking is disabled by default." The anthropic page also confirms `budget_tokens` is NOT part of the shape (only `type`).
3. The launch blog https://www.minimax.io/blog/minimax-m3 uses a coarser on/off framing ("M3 supports toggling thinking on or off"), consistent with `enabled`/`disabled` being real and `adaptive` being the self-decide middle.

So: treat `enabled` (always-on) as a VALID value alongside `disabled` and `adaptive`; the platform api-reference table under-documents it.

STILL UNSETTLED from docs (genuinely undocumented, need an authed probe — no API key in this session so I could not run it):
- unsupported_value_behavior: the api-reference (text-chat-openai.md, text-chat-anthropic.md, /docs/api-reference index) contains NO error schema for out-of-enum values. There is no documented HTTP status, no `base_resp`/`status_code`/`status_msg` or `error.type`/`error.message` body shape for a bad thinking.type (e.g. "high"), for Anthropic-style `budget_tokens` on /anthropic, or for `thinking` sent to an M2.x model. (M2.x guides only imply the param is ignored, never confirm error-vs-ignore.)
- A6 reasoning-token accounting for M3: unpublished. Only the M2 spec on https://platform.minimax.io/docs/guides/models-intro states "Maximum Output: 128k tokens (including CoT)". No M3 statement on whether thinking tokens count against max_completion_tokens or are broken out in `usage`; the openai schema exposes only `total_tokens` with no reasoning breakdown documented.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  values: >-
    thinking.type: "disabled" | "adaptive" | "enabled" (M3). "disabled" answers
    directly / suppresses thinking; "adaptive" lets M3 self-decide when to reason;
    "enabled" forces reasoning always-on. All three are FIRST-PARTY documented:
    MiniMax's own model card (huggingface.co/MiniMaxAI/MiniMax-M3) and GitHub
    README (github.com/MiniMax-AI/MiniMax-M3) list all three ("three reasoning
    modes through the thinking parameter"). The platform api-reference tables
    (text-chat-openai.md, text-chat-anthropic.md) list only "disabled"|"adaptive"
    — they OMIT "enabled"; this is an incomplete doc table, not evidence "enabled"
    is invalid. reasoning_split accepts true | false (openai-chat only; not a
    thinking toggle — only changes surfacing). budget_tokens is NOT part of the
    thinking object on the anthropic surface (only `type`), per text-chat-anthropic.md.
  unsupported_value_behavior: >-
    Partly settled: "enabled" is a valid always-on value (see values), so it is
    NOT out-of-enum. For a genuinely out-of-enum thinking.type (e.g. "high"), an
    Anthropic-style budget_tokens on /anthropic, or thinking sent to an M2.x
    model, the HTTP code and error body remain UNDOCUMENTED — no error schema
    (base_resp/status_code/status_msg or error.type/error.message) appears in the
    api-reference. M2.x guides imply the param is ignored (thinking "remains
    enabled regardless") rather than erroring, but do not confirm. Needs authed
    probe.
  per_model:
    - id: MiniMax-M3
      mode: optional
      effort: "adjustable: [disabled, adaptive, enabled] via thinking.type (three first-party-documented modes; enabled=always-on, adaptive=self-decide depth, disabled=off; no numeric/effort ladder)"
  tools_and_streaming: >-
    (accounting note) M3 reasoning-token-vs-max-output accounting is UNPUBLISHED:
    no M3 statement on whether thinking tokens count against max_completion_tokens
    / max_tokens or are broken out in usage (openai usage exposes only
    total_tokens, no reasoning breakdown documented). Only the sibling M2 spec
    (models-intro) states its 128k max output is "including CoT"; treat M3
    reasoning as counting against the output budget until an authed usage probe
    confirms otherwise.
````

**Still unresolved (needs a live key):**

````
Run against a live M3 key (JWT). Probe 1 — confirm `enabled` is accepted and capture out-of-enum 400 body on openai-chat:
curl -sS -D - https://api.minimax.io/v1/chat/completions -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"2+2? one word"}],"max_completion_tokens":64,"thinking":{"type":"enabled"}}'
# then swap thinking.type to a bogus value to capture the error shape:
curl -sS -D - https://api.minimax.io/v1/chat/completions -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16,"thinking":{"type":"high"}}'
# Probe 2 — Anthropic-style budget_tokens + thinking to an M2.x model (expect error-or-ignore):
curl -sS -D - https://api.minimax.io/anthropic/v1/messages -H "X-Api-Key: $MINIMAX_API_KEY" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" -d '{"model":"MiniMax-M3","max_tokens":64,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"hi"}]}'
curl -sS -D - https://api.minimax.io/v1/chat/completions -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMax-M2.7","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16,"thinking":{"type":"disabled"}}'
# Probe 3 — A6 accounting: force long reasoning and inspect usage for a reasoning/CoT breakdown and whether it counts toward max_completion_tokens:
curl -sS https://api.minimax.io/v1/chat/completions -H "Authorization: Bearer $MINIMAX_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMax-M3","messages":[{"role":"user","content":"Think step by step, then give final answer: what is 17*23?"}],"max_completion_tokens":128,"thinking":{"type":"enabled"},"reasoning_split":true}' | python -c "import sys,json;d=json.load(sys.stdin);print(json.dumps(d.get('usage'),indent=2));print('finish:',d['choices'][0].get('finish_reason'))"
# Record: HTTP status + full JSON error body (base_resp/status_code/status_msg or error.type/error.message) for each rejection, and whether usage exposes reasoning/CoT tokens separately from completion_tokens.
````

### Verifier notes (minor, not applied)

- **context.per_model MiniMax-M2.5 long_context_pricing: "unpublished — M2.5 (legacy) not itemized on the pay-as-you-go page fetched this session"** — The current pricing-paygo.md page DOES itemize MiniMax-M2.5: input $0.30/M, output $1.20/M, cache read $0.03/M, cache write $0.375/M. The 'unpublished/not itemized' claim is contradicted by the cited source. Note the M2.5 cache-read rate ($0.03) differs from the M2.7 family's $0.06, so it is not a copy of the M2.7 line. This does not cause an API error or a wrong-model swap — it marks a known fact *Suggested: Replace with: MiniMax-M2.5 (legacy) flat: input $0.30/M, output $1.20/M (cache read $0.03/M, cache write $0.375/M) per pricing-paygo.md.*

---

## 10. bytedance

*Verification: verified-clean — 1 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: bytedance
endpoints:
  - url: https://ark.cn-beijing.volces.com/api/v3
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/_constants.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/types/chat/completion_create_params.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/resources/chat/completions.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/types/shared/reasoning_effort.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/types/chat/chat_completion_chunk.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/types/chat/chat_completion_message.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/_client.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/volcenginesdkarkruntime/resources/tokenization.py
  - https://raw.githubusercontent.com/volcengine/volcengine-python-sdk/master/README.md
  - https://api.github.com/repos/volcengine/volcengine-python-sdk/contents/volcenginesdkarkruntime/types/chat
  - https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkexamples/volcenginesdkarkruntime/completions.py
  - https://docs.byteplus.com/en/docs/Byteplus_LAS/Multimodal-Deep-Thinking-Doubao-Seed-2-0
  - https://github.com/YishenTu/claudian/discussions/529
  - https://github.com/QuantumNous/new-api/issues/4073
  - https://docs.byteplus.com/en/docs/ModelArk/ (nav index only; article bodies client-rendered)
  - https://docs.byteplus.com/en/docs/ModelArk/1449737 (FAILED)
  - https://docs.byteplus.com/en/docs/ModelArk/1494384 (FAILED)
  - https://docs.byteplus.com/api/docs/ModelArk/1494384 (FAILED)
  - https://docs.byteplus.com/api/docs/ModelArk/1449737 (FAILED)
  - https://docs.byteplus.com/en/docs/ModelArk/1330626 (FAILED)
  - https://docs.volcengine.com/docs/82379/1298459?lang=zh (FAILED)
  - https://docs.volcengine.com/docs/82379/2121998 (FAILED)
reasoning:
  param: |
    Two independent params, both top-level in the JSON body (first-party: volcengine-python-sdk resources/chat/completions.py sends both):
    1. "thinking" — object: {"type": "<value>"}. SDK TypedDict verbatim: class Thinking(TypedDict, total=False): type: Literal["enabled", "disabled", "auto"]
    2. "reasoning_effort" — string. SDK verbatim: ReasoningEffort: TypeAlias = Optional[Literal["minimal", "low", "medium", "high"]]
    When calling through the plain OpenAI SDK, official docs (ModelArk 1330626, via search snippet; page itself FAILED) show "thinking" passed via extra_body — over raw HTTP it is simply a top-level body field.
  values: |
    thinking.type: "enabled" | "disabled" | "auto" (SDK-wide union; which of the three each seed-2.0 model accepts is NOT first-party-confirmed — older models like deepseek-v3.1 on Ark accept only enabled/disabled per official doc snippets)
    reasoning_effort: "minimal" | "low" | "medium" | "high" (registry/blog claims say "minimal" = no thinking on seed-2.0; unconfirmed first-party)
  default_when_omitted: |
    CONFLICTING, unresolved: official BytePlus LAS Doubao-Seed-2.0 page (loaded, but it is the Lake AI Service platform, not Ark): "Deep thinking mode is enabled by default and can be manually turned off." A ModelArk doc snippet for other models says thinking "defaults to Off state". Per-model default on Ark v3 for the four catalog ids: unknown — probe required. Do NOT assume omitted = no reasoning for seed-2.0.
  unsupported_value_behavior: |
    http-400 + OpenAI-style error body. Live-observed (claudian discussion #529, doubao-seed-2.0-code via Ark):
    400 {"error":{"code":"InvalidParameter","message":"Unsupported reasoning_effort type. Request id: ...","param":"","type":"BadRequest"}}
    Same InvalidParameter/BadRequest shape reported for other unsupported params (response_format.type=json_schema). Behavior for thinking:{"type":"auto"} on a model without auto support: unknown, probe.
  per_model:
    - id: doubao-seed-2-0-pro-260215
      mode: optional (thinking on by default per LAS sibling doc; registry says controllable via thinking.type + reasoning_effort) — first-party unconfirmed
      effort: "adjustable: [minimal, low, medium, high] (registry claim, probe to confirm)"
    - id: doubao-seed-2-0-code-preview-260215
      mode: optional — unconfirmed; NOTE reasoning_effort was live-rejected on a doubao-seed-2.0-code call via the coding-plan surface (claudian #529), so effort support may differ by surface
      effort: "unknown — probe (rejection evidence exists on at least one surface)"
    - id: doubao-seed-2-0-lite-260215
      mode: optional — unconfirmed first-party
      effort: "adjustable: [minimal, low, medium, high] (registry claim, probe)"
    - id: doubao-seed-2-0-mini-260215
      mode: optional — unconfirmed; registry copy says mini supports "four-level thinking"
      effort: "adjustable: [minimal, low, medium, high] (registry claim, probe)"
  silent_substitutions: none documented anywhere found; Ark routes Model IDs to preset endpoints automatically but no evidence of reasoning-flag-triggered model swaps
  tools_and_streaming: |
    First-party (SDK types): assistant message and streaming delta both carry reasoning_content: Optional[str] ("The reasoning content of the message") and encrypted_content: Optional[str] ("The encrypted reasoning content of the message") — so thinking arrives as delta.reasoning_content in SSE chunks, and encrypted thinking blocks exist for multi-turn replay. Usage only in final chunk when stream_options: {"include_usage": true}. Official seed-1.6 doc snippet: models support "managing thinking blocks and tool call results in context" (interleaved thinking with tools). Token accounting (LAS platform doc, may not transfer to Ark v3): max_tokens EXCLUDES chain-of-thought; max_completion_tokens INCLUDES reasoning tokens; the two cannot be set simultaneously. Both max_tokens and max_completion_tokens exist on the Ark v3 SDK create() signature.
  compat_passthrough: |
    Surfaces: (1) native OpenAI-chat /api/v3/chat/completions — thinking + reasoning_effort are first-class body params: passes. (2) Responses API (/responses; SDK has responses + input_items resources): unknown — reasoning param mapping not sourced. (3) Anthropic-compat coding-plan surface (used by Claude-Code-style clients): reasoning_effort observed REJECTED with 400 InvalidParameter (claudian #529) — an OpenAI-ism does not pass there.
discovery:
  endpoint: none confirmed. The official SDK client attaches NO models resource (first-party negative evidence from _client.py); GET https://ark.cn-beijing.volces.com/api/v3/models is rumored by third parties only — probe.
  auth: api-key (Authorization Bearer) if the endpoint exists
  metadata: unknown
  account_scoped: "yes (if it exists): Ark model access requires per-model activation (开通) in the console; a listing would reflect account-activated models"
  rate_limits_or_caching: undocumented
  verdict: stay-static
  refresh_strategy: n/a (re-evaluate to augment-static if the /models probe succeeds)
context:
  per_model:
    - id: doubao-seed-2-0-pro-260215
      context: "unpublished first-party this session (Ark model-list docs client-rendered); registry consensus 256K — treat as registry-only"
      max_output: "unpublished first-party; registry claims up to 128K, default 4K"
      long_context_pricing: "unknown; Ark historically prices doubao tiers by input-length bands (e.g. <=32k / 32k-128k / >128k) — bands for seed-2.0 not sourced"
      overpromise_risk: "yes: no first-party number; also LAS platform caps max_completion_tokens at 64k for seed-2.0, so hardcoding 128K output could over-promise"
    - id: doubao-seed-2-0-code-preview-260215
      context: "unpublished first-party; registry: 256K"
      max_output: "unpublished first-party; registry: 128K"
      long_context_pricing: unknown
      overpromise_risk: "yes: registry-only numbers; preview models can also change limits without notice"
    - id: doubao-seed-2-0-lite-260215
      context: unpublished (no first-party or consistent registry figure found)
      max_output: unpublished
      long_context_pricing: unknown
      overpromise_risk: "yes: nothing verifiable — do not hardcode"
    - id: doubao-seed-2-0-mini-260215
      context: unpublished (no first-party or consistent registry figure found)
      max_output: unpublished
      long_context_pricing: unknown
      overpromise_risk: "yes: nothing verifiable — do not hardcode"
  token_counting_endpoint: "POST https://ark.cn-beijing.volces.com/api/v3/tokenization (first-party: SDK resources/tokenization.py; body: {\"model\": str, \"text\": str|[str], \"user\": optional str})"
probes_needed:
  - question: Does GET /api/v3/models exist, what auth, and what metadata does it return (ids only vs context/capabilities)?
    probe: 'curl -sS -X GET "https://ark.cn-beijing.volces.com/api/v3/models" -H "Authorization: Bearer $ARK_API_KEY"'
  - question: Are all four catalog ids live as direct Model IDs on cn Ark (vs requiring an ep- endpoint or console activation)?
    probe: 'for m in doubao-seed-2-0-pro-260215 doubao-seed-2-0-code-preview-260215 doubao-seed-2-0-lite-260215 doubao-seed-2-0-mini-260215; do curl -sS -X POST "https://ark.cn-beijing.volces.com/api/v3/chat/completions" -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" -d "{\"model\": \"$m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}"; echo; done'
  - question: Per model — is thinking on by default when omitted, and is "auto" accepted? (compare reasoning_content presence with param omitted vs {"type":"disabled"} vs {"type":"auto"})
    probe: 'curl -sS -X POST "https://ark.cn-beijing.volces.com/api/v3/chat/completions" -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" -d ''{"model": "doubao-seed-2-0-pro-260215", "messages": [{"role": "user", "content": "What is 17*23?"}], "thinking": {"type": "auto"}}'''
  - question: Which models accept reasoning_effort, and does "minimal" fully suppress reasoning_content? (repeat per model, per value)
    probe: 'curl -sS -X POST "https://ark.cn-beijing.volces.com/api/v3/chat/completions" -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" -d ''{"model": "doubao-seed-2-0-mini-260215", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "minimal"}'''
  - question: True context window and max output per model — Ark 400 errors state the allowed range in the message, so an over-limit request reveals the real ceiling
    probe: 'curl -sS -X POST "https://ark.cn-beijing.volces.com/api/v3/chat/completions" -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" -d ''{"model": "doubao-seed-2-0-pro-260215", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 999999}'''
  - question: Is setting max_tokens AND max_completion_tokens together a 400 on Ark v3 (LAS doc says they are mutually exclusive)?
    probe: 'curl -sS -X POST "https://ark.cn-beijing.volces.com/api/v3/chat/completions" -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" -d ''{"model": "doubao-seed-2-0-pro-260215", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100, "max_completion_tokens": 100}'''
  - question: Does the tokenization endpoint accept the seed-2.0 model ids (usable for pre-flight context accounting)?
    probe: 'curl -sS -X POST "https://ark.cn-beijing.volces.com/api/v3/tokenization" -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" -d ''{"model": "doubao-seed-2-0-pro-260215", "text": "hello world"}'''
conflicts:
  - "BytePlus (intl) LAS/ModelArk docs list seed-2.0 ids WITHOUT the doubao- prefix and with different version dates (seed-2-0-pro-260328, seed-2-0-lite-260228, seed-2-0-mini-260215, seed-2-0-mini-260428, seed-2-0-lite-260428) — https://docs.byteplus.com/en/docs/Byteplus_LAS/Multimodal-Deep-Thinking-Doubao-Seed-2-0 ; cn Ark ids per QuantumNous/new-api#4073 and CometAPI/api.airforce are doubao-seed-2-0-*-260215. Two platforms, two id namespaces + version streams — recorded, not resolved."
  - "Default thinking state: BytePlus LAS Seed-2.0 page says deep thinking 'enabled by default'; a ModelArk doc snippet (other models) says thinking 'defaults to Off state' — recorded, probe decides per Ark model."
  - "Max output: registries (cloudprice.net, atlascloud.ai, howaiworks.ai) say up to 128K output for pro/code; official LAS doc caps max_completion_tokens at [0, 64k] for seed-2.0 — different platforms, recorded."
notes: |
  Doc situation: every ModelArk/volcengine Ark doc article (docs.byteplus.com/en/docs/ModelArk/*, docs.volcengine.com/docs/82379/*) is client-side rendered — direct fetches return navigation only. Everything labeled first-party here comes from the official volcengine-python-sdk source on GitHub (which encodes the wire format exactly) or from search-engine extraction of the official pages (used only as corroboration, flagged as such). This preset still needs the live-key probe list above more than any other.
  Load-bearing positives: base URL ark.cn-beijing.volces.com/api/v3 is the SDK default; model field accepts either a Model ID (auto-associated preset endpoint) or an ep-xxxx endpoint id (official 'Get Model ID' doc snippet + Chat API snippet using bare "seed-1-6-250915"); repetition_penalty is an Ark-specific extra param; encrypted_content exists for encrypted thinking replay in multi-turn tool loops.
  For Sylliptor's reasoning_mode layer: emit "thinking": {"type": ...} as the primary control, treat reasoning_effort as secondary and per-model-gated, and never emit OpenAI's nested "reasoning": {"effort": ...} shape — that spelling has no basis on Ark.
````

### Gap-fill addendum (post-verification)

The A2 default conflict is RESOLVED: for the seed-2.0 series, deep thinking is ON by default. First-party, loadable evidence: the BytePlus LAS "Multimodal-Deep-Thinking-Doubao-Seed-2-0" page states verbatim "Deep thinking mode is enabled by default and can be manually turned off" (https://docs.byteplus.com/en/docs/Byteplus_LAS/Multimodal-Deep-Thinking-Doubao-Seed-2-0). The Volcengine Ark 深度思考 doc (https://www.volcengine.com/docs/82379/1449737 -> redirects to docs.volcengine.com/docs/82379/1449737) independently indexes the same: thinking defaults ON with reasoning_effort "medium", and "minimal" disables thinking. The earlier "defaults to Off" snippet was for OTHER/older Ark models (deepseek-v3.1-style), NOT seed-2.0 — no genuine contradiction remains.

A4 value sets: (1) thinking.type accepts "enabled" | "disabled" | "auto" — the LAS page describes the thinking parameter as controlling deep-thinking mode "(enabled / disabled / auto)", matching the SDK Literal union verbatim. (2) reasoning_effort accepts "minimal" | "low" | "medium" | "high", DEFAULT "medium"; "minimal" = 关闭思考/直接作答 (turns thinking off, answers directly), "low" = lightweight/speed, "medium" = balanced (default), "high" = deep reasoning. Corroborated by the Ark 深度思考 doc body, by CometAPI's doubao-seed-2-0 model page (https://www.cometapi.com/models/doubao/doubao-seed-2-0/ shows extra_body={"reasoning_effort":"medium"} and references "four-level thinking"), and by the huasheng Ark survey (https://www.huasheng.ai/insights/volcengine-ark-api-guide/ shows thinking passed as extra_body={"thinking":{"type":"enabled",...}}). The doc treats the seed-2.0 SERIES uniformly — no per-id (pro/code/lite/mini) divergence is documented for native Ark v3.

Surface caveat resolves the claudian #529 rejection: reasoning_effort was rejected (400 InvalidParameter) on the Anthropic-compat coding-plan surface, NOT on native /api/v3/chat/completions. On native v3 the OpenAI-ism reasoning_effort is a first-class body field per the series docs. So per-model "mode: optional, thinking on by default" now has first-party backing; only a native-v3 per-id acceptance TABLE could not be extracted (Volcengine's own pages are client-rendered, nav-only). Note a LAS-platform naming variant exists in examples (model "seed-2-0-pro-260328", no doubao- prefix) distinct from the Ark catalog ids (doubao-seed-2-0-pro-260215).

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  values: |
    thinking.type: "enabled" | "disabled" | "auto" (SDK Literal union; first-party LAS Doubao-Seed-2.0 page describes the thinking parameter as controlling deep-thinking mode "(enabled / disabled / auto)" — all three apply to the seed-2.0 series)
    reasoning_effort: "minimal" | "low" | "medium" | "high"; DEFAULT "medium". "minimal" = thinking off / direct answer (关闭思考,直接作答); "low" = lightweight, speed-first; "medium" = balanced (default); "high" = deep reasoning. (Ark 深度思考 doc 82379/1449737; CometAPI doubao-seed-2-0 page demonstrates reasoning_effort:"medium" + "four-level thinking")
  default_when_omitted: |
    RESOLVED — thinking is ON by default for the seed-2.0 series (equivalent to reasoning_effort "medium"). First-party: BytePlus LAS Doubao-Seed-2.0 page (loaded) "Deep thinking mode is enabled by default and can be manually turned off"; Ark 深度思考 doc 82379/1449737 concurs (default effort "medium", "minimal" disables). The prior "defaults to Off" snippet applied to other/older Ark models (deepseek-v3.1 etc.), NOT seed-2.0. Omitting thinking => reasoning runs; send reasoning_effort:"minimal" (or thinking:{"type":"disabled"}) to suppress it.
  per_model:
    - id: doubao-seed-2-0-pro-260215
      mode: optional, thinking ON by default (first-party: seed-2.0 series enabled-by-default; controllable via thinking.type or reasoning_effort)
      effort: "[minimal, low, medium, high], default medium (series-level doc; native /api/v3)"
    - id: doubao-seed-2-0-code-preview-260215
      mode: optional, thinking ON by default on native /api/v3. NOTE the reasoning_effort 400-rejection in claudian #529 was on the Anthropic-compat CODING-PLAN surface, not native v3 — an OpenAI-ism does not pass there; on native v3 it is a first-class field.
      effort: "[minimal, low, medium, high] on native /api/v3 (series-level); reasoning_effort NOT accepted on coding-plan/anthropic-compat surface"
    - id: doubao-seed-2-0-lite-260215
      mode: optional, thinking ON by default (series-level)
      effort: "[minimal, low, medium, high], default medium (series-level)"
    - id: doubao-seed-2-0-mini-260215
      mode: optional, thinking ON by default (series-level; mini documented as supporting four-level thinking)
      effort: "[minimal, low, medium, high], default medium (series-level)"
  compat_passthrough: |
    (1) native OpenAI-chat /api/v3/chat/completions — thinking + reasoning_effort are first-class body params: passes; seed-2.0 defaults thinking ON. (2) Responses API (/responses): reasoning mapping still unsourced. (3) Anthropic-compat coding-plan surface: reasoning_effort REJECTED 400 InvalidParameter (claudian #529) — this is surface-specific, NOT a per-model limitation; the same doubao-seed-2.0-code accepts reasoning_effort on native /api/v3.
````

**Still unresolved (needs a live key):**

````
Optional confirmation only — the seed-2.0 series docs already resolve default+values, but Volcengine's per-id pages are client-rendered so a per-catalog-id native-v3 acceptance table could not be extracted. To confirm each of the four Ark ids honors reasoning_effort + thinking.type on native /api/v3 (run per id, e.g. doubao-seed-2-0-pro-260215 / -code-preview- / -lite- / -mini-):

curl -sS https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Authorization: Bearer $ARK_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"doubao-seed-2-0-code-preview-260215","messages":[{"role":"user","content":"2+2?"}],"reasoning_effort":"high","thinking":{"type":"enabled"},"stream":true,"stream_options":{"include_usage":true}}'

Expect 200 with delta.reasoning_content in SSE chunks (thinking ON). Then repeat with reasoning_effort:"minimal" (expect no reasoning_content => thinking off) and thinking:{"type":"auto"} (confirm 200 vs 400 InvalidParameter) to pin each id's accepted thinking.type subset.
````

### Verifier notes (minor, not applied)

- **unsupported_value_behavior: 'http-400 + OpenAI-style error body. Live-observed (claudian discussion #529, doubao-seed-2.0-code via Ark)'** — The cited claudian #529 rejection was observed on the coding-plan surface (endpoint https://ark.cn-beijing.volces.com/api/coding), not the /api/v3 chat/completions surface this contract's reasoning section otherwise describes. The reasoning.unsupported_value_behavior block attributes it generically to 'via Ark' without noting the sub-surface, while compat_passthrough correctly states reasoning_eff *Suggested: In the unsupported_value_behavior block, tag the observation as coding-plan-surface (/api/coding) specifically, and note the native /api/v3 error-shape claim (400 + InvalidParameter/BadRequest body) is a plausible generalization of the gateway shape rather than a v3-observed datapoint (probe listed)*

---

## 11. groq

*Verification: corrected after adversarial review — 5 challenge(s), 1 fatal/major; corrections are applied in the block.*

````yaml
provider: groq
endpoints:
  - url: https://api.groq.com/openai/v1
    protocol: openai-chat
  - url: https://api.groq.com/openai/v1/responses
    protocol: openai-responses
checked: 2026-07-19
sources:
  - https://console.groq.com/docs/reasoning
  - https://console.groq.com/docs/models
  - https://console.groq.com/docs/api-reference
  - https://console.groq.com/docs/compound
  - https://console.groq.com/docs/compound/systems/compound
  - https://console.groq.com/docs/model/openai/gpt-oss-120b
  - https://console.groq.com/docs/model/openai/gpt-oss-20b
  - https://console.groq.com/docs/model/qwen/qwen3.6-27b
  - https://console.groq.com/docs/model/groq/compound (FAILED)   # 404; compound specs taken from /docs/models and /docs/compound/systems/compound instead
  - https://console.groq.com/docs/rate-limits
  - https://console.groq.com/docs/prompt-caching
  - https://console.groq.com/docs/openai
  - https://console.groq.com/docs/responses-api
reasoning:
  param: "chat-completions surface: reasoning_effort, reasoning_format, include_reasoning (reasoning_format and include_reasoning are mutually exclusive). responses surface: reasoning object with effort key, i.e. \"reasoning\": {\"effort\": \"low\"}"
  values: "reasoning_effort full enum per api-reference: none, default, low, medium, high — but split per family: gpt-oss (openai/gpt-oss-120b, openai/gpt-oss-20b) accept low | medium | high; qwen/qwen3.6-27b accepts none | default. reasoning_format: hidden | raw | parsed (NOT supported on gpt-oss models — use include_reasoning there). include_reasoning: true | false"
  default_when_omitted: "gpt-oss: reasoning always on; default reasoning_effort is \"medium\" — DOCUMENTED (api-reference reasoning_effort description: \"'medium' is the default value\"). qwen3.6-27b: reasoning_effort default \"default\" (thinking mode on; docs: set 'none' to disable, 'default' or null to let Qwen reason). reasoning_format default \"raw\", auto-switched to \"parsed\" when JSON mode or tool use is enabled (reasoning docs: defaults to raw, or parsed when JSON mode/tool use enabled; \"hidden\" is a valid explicit choice in that situation but is NOT part of the automatic default switch). include_reasoning default true (hides reasoning field when false; does not claim to disable thinking)"
  unsupported_value_behavior: "documented http-400 for reasoning_format:\"raw\" combined with JSON mode or tool use. reasoning_format on gpt-oss documented as 'not supported' but error code/body unpublished. Invalid enum cross-family (e.g. \"medium\" to qwen, \"none\" to gpt-oss) undocumented — unknown, probe needed. logprobs/logit_bias/top_logprobs/messages[].name: documented http-400 (openai-compat page). n>1: documented http-400 (api-reference n param: 'only n=1 is supported. Other values will result in a 400 response'; the openai-compat page states the constraint without an error code). frequency_penalty/presence_penalty: api-reference marks both 'not yet supported by any of our models' but neither appears in the openai-compat page's explicit 400 list — reject-vs-silently-ignore undocumented, probe queued. Error body shapes not documented"
  per_model:
    - id: openai/gpt-oss-120b
      mode: always-on
      effort: "adjustable: [low, medium, high]; default medium (documented); no documented off switch (include_reasoning:false only hides output)"
    - id: qwen/qwen3.6-27b
      mode: optional
      effort: "adjustable: [none, default] (binary thinking toggle, no low/medium/high)"
    - id: openai/gpt-oss-20b
      mode: always-on
      effort: "adjustable: [low, medium, high]; default medium (documented)"
    - id: groq/compound
      mode: always-on
      effort: "n/a — agentic system over Llama 4 Scout + GPT-OSS 120B; reasoning-param support undocumented, do not send reasoning params"
  silent_substitutions: none documented; groq/compound internally routes across Llama 4 Scout and GPT-OSS 120B by design (system, not a substitution triggered by reasoning flags)
  tools_and_streaming: "reasoning_format must be parsed or hidden when using tool calling or JSON mode; explicit raw + tools/JSON mode = 400 (implicit default auto-switches to parsed). parsed puts reasoning in a separate message reasoning field; hidden returns only the final answer. SSE delta field shape for reasoning is undocumented on the reasoning page. Whether reasoning tokens count against max_completion_tokens is unpublished (probe). Default max_completion_tokens noted as 1024 in reasoning docs — far too low for agentic use, always set explicitly"
  compat_passthrough: "openai-chat surface: native — reasoning_effort/reasoning_format/include_reasoning are Groq's own spellings, pass. openai-responses surface: uses OpenAI's reasoning.effort object spelling (documented, passes); reasoning_format/include_reasoning on the responses surface: unknown. OpenAI value \"minimal\" for reasoning_effort is NOT in Groq's enum — expect rejection (unknown exact behavior)"
discovery:
  endpoint: https://api.groq.com/openai/v1/models
  auth: bearer (api key in Authorization header, required)
  metadata: "ids+context — LIST response model objects documented with id, object, created, owned_by, active, context_window, public_apps; max_completion_tokens is documented on the RETRIEVE-model (/models/{model}) response (in practice list entries carry it too, but only the retrieve schema documents it — treat list-derived max_completion_tokens as best-effort); no capability flags (reasoning support not exposed), no pricing"
  account_scoped: "unknown — docs do not state whether the listing varies by account/tier; rate limits (not the listing) are per-org and visible on the console limits page. Probe with two keys to confirm"
  rate_limits_or_caching: "undocumented for /models itself; chat endpoints return retry-after (on 429), x-ratelimit-limit-requests (RPD), x-ratelimit-limit-tokens (TPM), x-ratelimit-remaining-requests, x-ratelimit-remaining-tokens, x-ratelimit-reset-requests, x-ratelimit-reset-tokens; limits are per-model and per-plan (free vs developer vs enterprise)"
  verdict: augment-static
  refresh_strategy: "on-config-open (cheap authenticated GET; use live context_window/active to correct static entries and drop inactive models; max_completion_tokens authoritative via per-model retrieve; keep reasoning capability map static since listing has no capability metadata)"
context:
  per_model:
    - id: openai/gpt-oss-120b
      context: "131,072 (no tier splits documented — same for all plans)"
      max_output: "65,536"
      long_context_pricing: "flat: $0.15/M input, $0.60/M output; cached input $0.075/M (50% discount, automatic, gpt-oss models only); no long-context boundary"
      overpromise_risk: "no for context; note free-plan TPM limits can make full-context requests fail with 429 long before 131K"
    - id: qwen/qwen3.6-27b
      context: "131,072"
      max_output: "32,768"
      long_context_pricing: "flat: $0.60/M input, $3.00/M output (models page); no caching, no boundary"
      overpromise_risk: "no for context; PREVIEW model — 'evaluation purposes only', may be deprecated/removed without notice"
    - id: openai/gpt-oss-20b
      context: "131,072"
      max_output: "65,536"
      long_context_pricing: "flat: $0.075/M input, $0.30/M output; cached input ~$0.037/M (50% discount, automatic); no boundary"
      overpromise_risk: "no for context; free-plan TPM caveat as above"
    - id: groq/compound
      context: "131,072"
      max_output: "8,192"
      long_context_pricing: "token pricing at underlying-model rates (GPT-OSS 120B $0.15/$0.60 per M) PLUS per-tool fees: basic web search $5/1000 req, advanced web search $8/1000 req, visit website $1/1000 req, code execution $0.18/hour"
      overpromise_risk: "yes: max_output is only 8,192 (vs 65,536 on plain gpt-oss-120b) — a hardcoded generic 'gpt-oss-class' output cap over-promises here; also not usable with client tools, regional/sovereign endpoints, or PHI"
  token_counting_endpoint: none
probes_needed:
  - question: "Exact status/error body when a cross-family reasoning_effort value is sent (generic CLI sends its one global knob, e.g. medium, to qwen)"
    probe: "curl -s https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"qwen/qwen3.6-27b\",\"reasoning_effort\":\"medium\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Exact status/error body when reasoning_effort:none is sent to gpt-oss (docs list no 'none' for that family)"
    probe: "curl -s https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"openai/gpt-oss-120b\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Exact status/error body when reasoning_format is sent to a gpt-oss model (documented unsupported, error unpublished)"
    probe: "curl -s https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"openai/gpt-oss-120b\",\"reasoning_format\":\"parsed\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Exact status/error body when client tool definitions are sent to groq/compound (docs say custom tools unsupported but do not name the error) — required to build the never-route-agent-loops guard"
    probe: "curl -s https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"groq/compound\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}'"
  - question: "Whether reasoning tokens count against max_completion_tokens on gpt-oss (unpublished; default effort itself is documented as medium, no probe needed for that)"
    probe: "curl -s https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"openai/gpt-oss-120b\",\"max_completion_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Prove sqrt(2) is irrational.\"}]}' # inspect usage + finish_reason; repeat with reasoning_effort high/low and compare usage"
  - question: "Whether frequency_penalty / presence_penalty (api-reference: 'not yet supported by any of our models', absent from the openai-compat 400 list) are rejected or silently ignored — many OpenAI-compat clients emit them by default"
    probe: "curl -s https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"openai/gpt-oss-20b\",\"frequency_penalty\":0.5,\"presence_penalty\":0.5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Whether GET /models output differs per account/tier (claimed 'account-visible' in prior refresh; docs silent)"
    probe: "curl -s https://api.groq.com/openai/v1/models -H \"Authorization: Bearer $GROQ_API_KEY\" # run with a free-plan key and a developer-plan key, diff the id/active sets; also confirm whether list entries carry max_completion_tokens (documented only on retrieve)"
  - question: "SSE delta field name carrying reasoning with reasoning_format:parsed (undocumented; needed for stream parser)"
    probe: "curl -N https://api.groq.com/openai/v1/chat/completions -H \"Authorization: Bearer $GROQ_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"qwen/qwen3.6-27b\",\"stream\":true,\"reasoning_format\":\"parsed\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23?\"}]}'"
conflicts:
  - "Per-model pages (console.groq.com/docs/model/openai/gpt-oss-20b, /model/qwen/qwen3.6-27b) render pricing as tokens-per-dollar figures ('$0.075 per 13M tokens', '$0.60 per 1.7M tokens') while console.groq.com/docs/models states per-1M rates ($0.075/$0.30 and $0.60/$3.00 per 1M) — recorded; almost certainly the same rates displayed two ways, but the per-model page rendering is ambiguous"
  - "Prior-refresh claim said GET /models is 'account-visible'; no opened doc page states account scoping of the listing — recorded as unverified, probe added"
  - "frequency_penalty/presence_penalty: api-reference marks both 'not yet supported by any of our models', but the openai-compat page's explicit will-400 list (logprobs, logit_bias, top_logprobs, messages[].name) omits them — sources leave reject-vs-ignore ambiguous; probe queued"
notes: "CORRECTION PASS 2026-07-19: (1) gpt-oss default reasoning_effort IS published — api-reference: \"'medium' is the default value\"; contract updated, probe narrowed to reasoning-token accounting only. (2) discovery.metadata corrected: public_apps added; max_completion_tokens is documented on the retrieve-model response, not the list-response schema. (3) Verifier's n>1 challenge is itself only half right: the openai-compat page indeed gives no error code, but the api-reference n-param description explicitly documents 'Other values will result in a 400 response' — the 400 stands as documented, with corrected source attribution. (4) reasoning_format auto-switch corrected to 'parsed' only (reasoning docs: defaults to raw, or parsed when JSON mode/tool use enabled); hidden is a valid explicit choice, not part of the auto-switch. (5) frequency_penalty AND presence_penalty confirmed 'not yet supported by any of our models' on api-reference; absent from the 400 list — hazard + probe added. — Original notes: Three request surfaces total: chat completions (primary), Responses API at /openai/v1/responses (fully OpenAI-compatible; reasoning via reasoning.effort object; previous_response_id, store, truncation, include, safety_identifier, prompt_cache_key, reusable prompt NOT supported), and batch. groq/compound also has groq/compound-mini (single tool call/request, ~3x lower latency) — not in catalog. compound built-in tools: web search, visit website, code execution, Wolfram Alpha; versioned system (default 2025-08-16); no PHI, no regional/sovereign endpoints. temperature 0 silently converted to 1e-8 (docs: float32, >0 and <=2). Prompt caching automatic, gpt-oss family only (incl. gpt-oss-safeguard-20b), 50% cached-input discount, min cacheable prefix 128-1024 tokens, does not stack with batch's 50% discount. Reasoning-docs default max_completion_tokens of 1024 will truncate agentic outputs if Sylliptor doesn't set it. qwen/qwen3.6-27b also takes image input (max 3 images, 20MB) and supports reasoning_format hidden/parsed."
````

---

## 12. cerebras

*Verification: verified-clean — 3 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: cerebras
endpoints:
  - url: https://api.cerebras.ai/v1
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://inference-docs.cerebras.ai/capabilities/reasoning
  - https://inference-docs.cerebras.ai/models/overview
  - https://inference-docs.cerebras.ai/models/openai-oss
  - https://inference-docs.cerebras.ai/models/zai-glm-47
  - https://inference-docs.cerebras.ai/models/gemma-4-31b
  - https://inference-docs.cerebras.ai/support/deprecation
  - https://inference-docs.cerebras.ai/api-reference/chat-completions
  - https://inference-docs.cerebras.ai/api-reference/models
  - https://inference-docs.cerebras.ai/api-reference/completions
  - https://inference-docs.cerebras.ai/support/rate-limits
  - https://inference-docs.cerebras.ai/resources/openai
  - https://inference-docs.cerebras.ai/support/error-codes (FAILED)
reasoning:
  param: reasoning_effort (string, nullable); also reasoning_format (string); disable_reasoning (boolean, zai-glm-4.7 only, DEPRECATED - removed after 2026-07-21); clear_thinking (boolean, zai-glm-4.7 only, default true)
  values: >
    reasoning_effort — gpt-oss-120b: "low" | "medium" | "high" (NO "none");
    zai-glm-4.7: "none" disables reasoning (whether "low"/"medium"/"high" are accepted as graduated levels is ambiguous across doc pages — see conflicts);
    gemma-4-31b: "none" | "low" | "medium" | "high", with docs stating the three active values are currently equivalent (on/off, no graduated control).
    reasoning_format — "parsed" | "raw" | "hidden" | "none"; docs also cite a default of "text_parsed" for GLM and GPT-OSS (spelling discrepancy recorded in conflicts). Gemma 4 31B does not support "raw" or "hidden".
  default_when_omitted: gpt-oss-120b reasons at "medium"; zai-glm-4.7 reasoning enabled (always-on unless "none"); gemma-4-31b reasoning disabled ("none" is the default)
  unsupported_value_behavior: unknown — docs do not publish the status code or error body for an out-of-range reasoning_effort (e.g. "none" on gpt-oss-120b); the /support/error-codes page 404s; needs live probe
  per_model:
    - id: gpt-oss-120b
      mode: always-on
      effort: "adjustable: [low, medium, high]; cannot be disabled — no \"none\" value"
    - id: zai-glm-4.7
      mode: optional
      effort: "adjustable at minimum on/off via \"none\"; graduated low/medium/high acceptance unconfirmed (conflict); enabled by default"
    - id: gemma-4-31b
      mode: optional
      effort: "adjustable: [none, low, medium, high] but low/medium/high are documented as currently equivalent (binary enable); default none"
  silent_substitutions: none documented (no model swaps triggered by reasoning flags); note reasoning_format is silently coerced raw->hidden when combined with json_object/json_schema on models that default to raw
  tools_and_streaming: >
    Streaming: "reasoning tokens are delivered in the reasoning field of the delta" (SSE delta.reasoning).
    Reasoning tokens are "generated and counted toward total completion tokens" even with reasoning_format "hidden",
    and max_completion_tokens is documented as "including reasoning tokens".
    Interleaved thinking with tool calls: not documented.
    Reasoning does NOT persist across turns; to carry it forward you must manually re-include it (GLM: <think>...</think> tags in assistant content; GPT-OSS: direct prepend); zai-glm-4.7 clear_thinking=false keeps prior-turn thinking in context for agentic workflows.
    gpt-oss-120b REJECTS requests containing both tools and response_format; other models accept both but prioritize tool calling.
  compat_passthrough: >
    openai-chat /v1/chat/completions: passes (reasoning_effort is a first-class documented param; non-standard params like clear_thinking require extra_body in the OpenAI SDK).
    legacy /v1/completions: no reasoning params documented at all (accepts prompt/model/stream/return_raw_tokens/max_tokens/min_tokens/grammar_root/seed/stop/temperature/top_p/echo/user/prompt_cache_key/logprobs) — whether a sent reasoning_effort errors or is ignored: unknown.
    No Responses API surface documented.
discovery:
  endpoint: GET https://api.cerebras.ai/v1/models
  auth: bearer (Authorization: Bearer $CEREBRAS_API_KEY)
  metadata: ids-only (id, object:"model", created, owned_by — no context, capabilities, or pricing)
  account_scoped: "unknown — docs do not state whether the list is filtered to key-reachable models; prior-refresh claim unverifiable from docs (probe listed)"
  rate_limits_or_caching: undocumented (general 429 Too Many Requests on limit breach; no listing-specific limits or cache headers published)
  verdict: augment-static
  refresh_strategy: on-config-open (use it only for availability/liveness of ids; all capability and context data must stay static since the endpoint returns ids-only)
context:
  per_model:
    - id: gpt-oss-120b
      context: "free tier: 65k tokens; paid tiers: 131k tokens"
      max_output: "free tier: 32k tokens; paid tiers: 40k tokens"
      long_context_pricing: flat ($0.35/M input, $0.75/M output; no boundary pricing)
      overpromise_risk: "yes: free-trial keys get 65k context and 32k output, not the 131k/40k headline"
    - id: zai-glm-4.7
      context: "free tier: 64k tokens; paid tiers: 131k tokens"
      max_output: "40k tokens on BOTH tiers"
      long_context_pricing: flat ($2.25/M input, $2.75/M output; no boundary pricing)
      overpromise_risk: "yes: free-trial keys get 64k context, not 131k (output is 40k on both). Also model is scheduled for deprecation on 2026-08-17"
    - id: gemma-4-31b
      context: "free tier: 65k tokens; paid tiers: 131k tokens"
      max_output: "free tier: 32k tokens; paid tiers: 40k tokens"
      long_context_pricing: flat ($0.99/M input, $1.49/M output; no boundary pricing)
      overpromise_risk: "yes: free-trial keys get 65k context and 32k output, not 131k/40k"
  token_counting_endpoint: none
probes_needed:
  - question: Does gpt-oss-120b return an HTTP error (and what body shape) for reasoning_effort:"none", or silently ignore/coerce it?
    probe: 'curl -s -w "\n%{http_code}\n" -X POST https://api.cerebras.ai/v1/chat/completions -H "Authorization: Bearer $CEREBRAS_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"gpt-oss-120b\",\"reasoning_effort\":\"none\",\"max_completion_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: What exactly does disable_reasoning return on zai-glm-4.7 after its 2026-07-21 removal (400? ignored? which error body)? Run on/after 2026-07-22.
    probe: 'curl -s -w "\n%{http_code}\n" -X POST https://api.cerebras.ai/v1/chat/completions -H "Authorization: Bearer $CEREBRAS_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"zai-glm-4.7\",\"disable_reasoning\":true,\"max_completion_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Does zai-glm-4.7 accept graduated reasoning_effort values ("low"/"high") or only "none" (conflict between api-reference and reasoning guide)?
    probe: 'curl -s -w "\n%{http_code}\n" -X POST https://api.cerebras.ai/v1/chat/completions -H "Authorization: Bearer $CEREBRAS_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"zai-glm-4.7\",\"reasoning_effort\":\"low\",\"max_completion_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"'
  - question: Is GET /v1/models filtered to key-reachable models and does richness differ on free vs paid keys (compare a free-trial key vs a Developer-tier key)?
    probe: 'curl -s https://api.cerebras.ai/v1/models -H "Authorization: Bearer $CEREBRAS_API_KEY"'
  - question: Does the free tier hard-cap context at 64-65k with an HTTP error (which code/body) when a longer prompt is sent, or silently truncate?
    probe: 'python -c "print(chr(34)+ (chr(120)*300000) +chr(34))" > /tmp/longmsg.json; curl -s -w "\n%{http_code}\n" -X POST https://api.cerebras.ai/v1/chat/completions -H "Authorization: Bearer $CEREBRAS_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"gpt-oss-120b\",\"max_completion_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":$(cat /tmp/longmsg.json)}]}"'
conflicts:
  - "https://inference-docs.cerebras.ai/api-reference/chat-completions lists zai-glm-4.7 reasoning_effort support as only \"none\" (disables reasoning); https://inference-docs.cerebras.ai/capabilities/reasoning reads as \"none\" disables while other values (low/medium/high) enable it — whether graduated effort exists on GLM is recorded, not resolved (probe listed)"
  - "https://inference-docs.cerebras.ai/capabilities/reasoning lists reasoning_format allowed values as parsed|raw|hidden|none but states the GLM/GPT-OSS default as \"text_parsed\" — a value string not in the allowed list; exact default spelling recorded, not resolved"
notes: >
  DEADLINE-CRITICAL: disable_reasoning on zai-glm-4.7 is removed after 2026-07-21 (2 days after checked date); Sylliptor must emit reasoning_effort:"none" instead. zai-glm-4.7 itself is scheduled for deprecation 2026-08-17 (per model page) — catalog should carry a sunset flag. gpt-oss-120b is the only Production model; gemma-4-31b and zai-glm-4.7 are Preview. Chat completions uses max_completion_tokens (documented as including reasoning tokens); legacy /v1/completions uses max_tokens and has no reasoning params. gpt-oss-120b rejects tools+response_format in the same request — a generic agentic CLI combining tool defs with structured output will hard-fail on this model. reasoning_format "raw" is incompatible with json_object/json_schema (auto-coerced to hidden on raw-default models); gemma-4-31b does not support "raw" or "hidden" at all. stream is incompatible with json_object response format. gpt-oss-120b may invoke tools beyond those specified (docs advise reprompting), and min_tokens can cause parser failures on it. Rate limits: free trial all models 5 RPM / 30K TPM / 1M TPH / 1M TPD; Developer tier per-model (gpt-oss-120b 1M TPM/1K RPM, zai-glm-4.7 500K TPM/500 RPM, gemma-4-31b 500K TPM/300 RPM); dual-bucket uncached vs total TPM (total defaults to 3x uncached), 429 message names which bucket. The models/overview page fetch rendered the GPT-OSS id as "gpt-120b", but the per-model page, rate-limits page, and API reference all consistently give "gpt-oss-120b" — treated as extraction noise, official id is gpt-oss-120b. /support/error-codes returned 404, so no error-body schema could be confirmed anywhere; all unsupported-value behaviors remain probe-gated.
````

### Verifier notes (minor, not applied)

- **Gemma 4 31B does not support "raw" or "hidden" (reasoning_format)** — Incomplete: the reasoning capabilities page lists FOUR unsupported items on Gemma 4 31B — verbatim "raw, hidden, clear_thinking, and preserve_thinking reasoning formats are not supported". The contract omits clear_thinking (so its per-model hazard coverage for clear_thinking-on-gemma rests only on the 'GLM-only param' framing) and omits preserve_thinking entirely, a token that appears nowhere else *Suggested: Change the Gemma line to: does not support "raw", "hidden", "clear_thinking", or "preserve_thinking"; note preserve_thinking as an otherwise-undocumented token.*
- **other models accept both [tools and response_format] but prioritize tool calling** — Two problems: (1) the cited source actually hedges — verbatim "Other models may accept the combination but prioritize tool calling, so response_format should not be relied upon"; the contract drops both "may" and the do-not-rely warning, upgrading a hedge to a fact. (2) An unrecorded cross-page conflict: https://inference-docs.cerebras.ai/capabilities/structured-outputs states universally that "to *Suggested: Restore the hedge ("may accept ... response_format should not be relied upon") and add a third conflicts entry for structured-outputs' universal prohibition vs resources/openai's model-dependent statement; add a probe (tools+response_format on zai-glm-4.7).*
- **sources list includes https://inference-docs.cerebras.ai/capabilities/tool-use implications via notes ("gpt-oss-120b may invoke tools beyond those specified")** — Attribution nit only: the tool-hallucination and min_tokens-parser-failure statements live on the model page (models/openai-oss), which IS in the sources list, and both were verified verbatim there — but the capabilities/tool-use page contains no tools+response_format interaction text at all, so nothing in the contract may lean on it for that claim. No factual error found; recording so the provena *Suggested: None required; optionally annotate that the tools+response_format claim is sourced solely from resources/openai.*

---

## 13. mistral

*Verification: verified-clean — 1 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: mistral
endpoints:
  - url: https://api.mistral.ai/v1
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://docs.mistral.ai/capabilities/reasoning
  - https://docs.mistral.ai/getting-started/models/models_overview
  - https://docs.mistral.ai/models/overview
  - https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04
  - https://docs.mistral.ai/models/model-cards/mistral-large-3-25-12
  - https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03
  - https://docs.mistral.ai/models/model-cards/codestral-25-08
  - https://docs.mistral.ai/models/model-cards/ministral-3-8b-25-12
  - https://docs.mistral.ai/api/
  - https://docs.mistral.ai/api/endpoint/chat
  - https://docs.mistral.ai/api/endpoint/models
  - https://docs.mistral.ai/resources/changelogs
reasoning:
  param: reasoning_effort (top-level request-body string). Legacy magistral param prompt_mode still appears in the API reference (enum value "reasoning") but is absent from the current reasoning capability docs; magistral family is retiring.
  values: "Capability docs (https://docs.mistral.ai/capabilities/reasoning) document exactly two values: \"high\" and \"none\". API reference (https://docs.mistral.ai/api/endpoint/chat) shows a wider enum: \"none\"|\"minimal\"|\"low\"|\"medium\"|\"high\"|\"xhigh\". Recorded as a conflict; only \"high\" and \"none\" have documented semantics. prompt_mode enum: \"reasoning\" only."
  default_when_omitted: unpublished. The capability docs frame "none" as the opt-out ("model thinks minimally and the thinking chunk is omitted"), which implies hybrid models (Medium 3.5, Small 4) think by default when the param is omitted — but no page states the default explicitly. Probe needed.
  unsupported_value_behavior: unknown — no error codes or error body shapes documented for invalid reasoning_effort values or for sending it to non-reasoning models. Probe needed.
  per_model:
    - id: mistral-medium-2604
      mode: optional
      effort: "adjustable: [\"high\", \"none\"] documented; API-ref enum also lists minimal/low/medium/xhigh (semantics unpublished). Docs recommend \"high\" for agentic/code tasks."
    - id: mistral-large-2512
      mode: none
      effort: n/a (no reasoning mention on card; not listed on the reasoning capability page)
    - id: mistral-small-2603
      mode: optional
      effort: "adjustable: [\"high\", \"none\"] — reasoning page lists mistral-small-latest; models overview calls Small 4 a hybrid instruct/reasoning/coding model"
    - id: codestral-2508
      mode: none
      effort: n/a
    - id: ministral-8b-2512
      mode: none
      effort: n/a
  silent_substitutions: none triggered by reasoning flags. Related but distinct hazard — "-latest" aliases re-point across families over time (per search leads, magistral-small-latest now resolves to Mistral Small 4, a hybrid model, after magistral's retirement); this is alias drift, not a reasoning-flag-triggered swap.
  tools_and_streaming: "With reasoning_effort=\"high\", message.content becomes a LIST of chunks: ThinkChunk (type: \"thinking\") then TextChunk (type: \"text\"); with \"none\" it is a plain string. In SSE streaming, delta.content changes shape mid-stream: a list containing ThinkChunk during thinking, then plain string for the answer phase. Docs REQUIRE replaying the full assistant message including ThinkChunk back into message history for multi-turn reasoning. Whether thinking tokens count against max_tokens: unpublished (max_tokens doc only says prompt + max_tokens must not exceed context length). Interleaved thinking with tool calls: undocumented."
  compat_passthrough: "n/a — single surface (native /v1/chat/completions, OpenAI-chat-shaped). Caveat: the surface is only OpenAI-LIKE; the chunked thinking content shape (list of typed chunks in content/delta.content) breaks strict OpenAI-chat clients that assume content is always a string."
discovery:
  endpoint: GET https://api.mistral.ai/v1/models (retrieve: GET /v1/models/{model_id})
  auth: bearer (Authorization Bearer API key)
  metadata: "ids+capabilities+context: per model: id, capabilities object (completion_chat, completion_fim, function_calling, fine_tuning, vision, classification — NO reasoning capability flag), max_context_length, aliases, created, owned_by, root/job/archived on fine-tuned variants. No pricing. Deprecation/retirement dates NOT shown in the documented response schema (probe: the live response may still carry a deprecation field; docs page examples appear to show the fine-tuned variant schema)."
  account_scoped: "yes: fine-tuned models owned by the account appear in the listing; whether the base-model roster varies by tier is undocumented (assume uniform)."
  rate_limits_or_caching: undocumented
  verdict: augment-static
  refresh_strategy: daily-cache. Use live max_context_length + capabilities to correct static entries and detect retired ids (404/missing); reasoning support must stay static because the capabilities object has no reasoning flag.
context:
  per_model:
    - id: mistral-medium-2604
      context: "256k per official card (registries render this as 262144; exact token integer not printed on card — confirm via /v1/models max_context_length). No tier splits documented."
      max_output: unpublished
      long_context_pricing: flat ($1.5/M in, $7.5/M out; no boundary documented)
      overpromise_risk: no (no documented tier variance)
    - id: mistral-large-2512
      context: "256k per official card; no tier splits documented"
      max_output: unpublished
      long_context_pricing: flat ($0.5/M in, $1.5/M out)
      overpromise_risk: no
    - id: mistral-small-2603
      context: "256k per official card; no tier splits documented"
      max_output: unpublished
      long_context_pricing: flat ($0.15/M in, $0.6/M out)
      overpromise_risk: no
    - id: codestral-2508
      context: "128k per official card — NOT 256k/262144; a catalog carrying 262144 here over-promises by 2x for everyone"
      max_output: "unpublished — the card does not state any output cap; the prior ~4K claim is not on the card"
      long_context_pricing: flat ($0.3/M in, $0.9/M out)
      overpromise_risk: "yes: any account, if the catalog copied the 262144 figure from the other cards; official card says 128k"
    - id: ministral-8b-2512
      context: "256k per official card; no tier splits documented"
      max_output: unpublished
      long_context_pricing: flat ($0.15/M in and out)
      overpromise_risk: no
  token_counting_endpoint: none (no count-tokens API documented; Mistral publishes tokenizers only)
probes_needed:
  - question: "Default reasoning behavior when reasoning_effort is omitted on mistral-medium-2604 — does the response contain a thinking chunk list or a plain string?"
    probe: "curl -s https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23?\"}],\"max_tokens\":512}'"
  - question: "Are the undocumented enum values (minimal/low/medium/xhigh) actually accepted, and what is the error status+body for a bogus value?"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"medium\",\"max_tokens\":64}' ; curl -s -w '\\nHTTP %{http_code}\\n' https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"maximum\",\"max_tokens\":64}'"
  - question: "Behavior of reasoning_effort on non-reasoning models (mistral-large-2512, codestral-2508): silently ignored or HTTP error, and what shape?"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-large-2512\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"high\",\"max_tokens\":64}' ; curl -s -w '\\nHTTP %{http_code}\\n' https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"codestral-2508\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"high\",\"max_tokens\":64}'"
  - question: "Does Mistral reject unknown/foreign reasoning params (OpenAI-style nested reasoning object, Anthropic-style thinking) with 422 'Extra inputs are not permitted', or ignore them? Also: does legacy prompt_mode:\"reasoning\" still work on mistral-medium-2604?"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning\":{\"effort\":\"high\"},\"max_tokens\":64}' ; curl -s -w '\\nHTTP %{http_code}\\n' https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"prompt_mode\":\"reasoning\",\"max_tokens\":64}'"
  - question: "Does the live GET /v1/models response include deprecation/retirement dates and a description per base model (docs page examples only show the fine-tuned-card field set), and what exact max_context_length integers are returned for the 5 catalog models?"
    probe: "curl -s https://api.mistral.ai/v1/models -H 'Authorization: Bearer $MISTRAL_API_KEY' | python -c \"import json,sys; d=json.load(sys.stdin); [print(json.dumps(m,indent=1)) for m in d['data'] if m['id'] in ('mistral-medium-2604','mistral-large-2512','mistral-small-2603','codestral-2508','ministral-8b-2512')]\""
  - question: "Do thinking tokens count against max_tokens (i.e. can a small max_tokens starve the final answer when reasoning_effort=high)?"
    probe: "curl -s https://api.mistral.ai/v1/chat/completions -H 'Authorization: Bearer $MISTRAL_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"Prove there are infinitely many primes.\"}],\"reasoning_effort\":\"high\",\"max_tokens\":100}'"
conflicts:
  - "https://docs.mistral.ai/capabilities/reasoning documents only two reasoning_effort values (\"high\", \"none\") with semantics; https://docs.mistral.ai/api/endpoint/chat shows enum \"none\"|\"minimal\"|\"low\"|\"medium\"|\"high\"|\"xhigh\" — both official, recorded not resolved."
  - "https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04 page header/slug presents the id as mistral-medium-3-5-26-04, while https://docs.mistral.ai/models/overview lists the API id as mistral-medium-2604 (card slugs are dotted long-forms; the versioned API ids are the YYMM short forms) — same pattern on all cards."
  - "https://docs.mistral.ai/api/endpoint/chat still lists prompt_mode (enum \"reasoning\") while https://docs.mistral.ai/capabilities/reasoning never mentions it and only documents reasoning_effort, with a note about deprecated native reasoning (magistral) models."
  - "Search-result leads (blog.xentoo.info retirement post, not opened directly) claim a summer-2026 retirement wave (magistral-small-2509 retiring 2026-07-31 etc.); https://docs.mistral.ai/resources/changelogs as fetched only confirmed Leanstral 1.5 retirement on September 30, 2026 — retirement wave unconfirmed from an opened official page."
notes: "KEY FINDING vs prior refresh: 'optional' reasoning on Medium 3.5 / Small 4 means the top-level reasoning_effort string param (documented values \"high\"/\"none\"), introduced with Mistral Medium 3.5 on April 28, 2026 per the changelog — NOT magistral's prompt_mode. These are hybrid models: one set of weights, thinking toggled per-request. The content shape mutates with reasoning (typed chunk list vs plain string, including mid-stream shape change in delta.content), which is the biggest client-breakage surface for an openai-chat adapter. Docs mandate replaying ThinkChunk in history for multi-turn. No reasoning capability flag exists in /v1/models, so Sylliptor's reasoning_mode map must stay static per model id. Codestral is the context outlier (128k vs 256k for the rest). Max output is unpublished for every catalog model. All prices flat; no long-context boundary pricing anywhere. mistral-small-2603 pricing $0.15/$0.6 per M; ministral-8b-2512 $0.15/$0.15. Card pages render context as '256k' — get exact integers from /v1/models max_context_length rather than hardcoding 262144."
````

### Gap-fill addendum (post-verification)

A2 conflict CONFIRMED against live docs (checked 2026-07-19), and it is a docs-vs-OpenAPI split, not a docs-internal disagreement. Four independent human-facing surfaces document exactly two reasoning_effort values and nothing between them: (1) https://docs.mistral.ai/capabilities/reasoning — "high" = "The response includes a full thinking chunk before the final answer, at the cost of increased token usage."; "none" = "The model thinks minimally and the thinking chunk is omitted from the response." (2) https://docs.mistral.ai/studio-api/conversations/reasoning — identical two-value wording; also notes reasoning_effort is settable on Agents/Conversations inside the completion_args field. (3) https://mistral.ai/news/mistral-small-4/ — introduces the param with only "none" (fast) and "high" (deep). (4) https://docs.mistral.ai/capabilities/reasoning/adjustable — now HTTP 404 (removed; the studio-api page is the live successor). The ONLY source listing minimal/low/medium/xhigh is the raw OpenAPI schema at https://docs.mistral.ai/api/endpoint/chat (enum: "none"|"minimal"|"low"|"medium"|"high"|"xhigh"; prompt_mode enum: "reasoning"), with NO semantics for the four intermediate values. Conclusion: emitting minimal/low/medium/xhigh is unverified — no doc states whether they are accepted, silently coerced to the nearest tier, or 400'd. Supported model ids per docs: mistral-small-latest (=mistral-small-2603, Small 4) and mistral-medium-3-5 (=mistral-medium-2604); "high" recommended for agentic/code. max_output CONFIRMED unpublished: Medium 3.5 card (https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04) and models overview print only 256k context, no output cap; API ref only states prompt+max_tokens must not exceed context length; GET /v1/models exposes max_context_length but no output field. No page states a default when reasoning_effort is omitted. Live accept/coerce/400 behavior and true max output could NOT be probed: no MISTRAL_API_KEY in this environment.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  values: "CONFLICT CONFIRMED (docs vs OpenAPI). Four human-facing sources document exactly two values with semantics: \"high\" (full thinking chunk before answer, higher token cost) and \"none\" (minimal thinking, thinking chunk omitted) — https://docs.mistral.ai/capabilities/reasoning, https://docs.mistral.ai/studio-api/conversations/reasoning, https://mistral.ai/news/mistral-small-4/. The former https://docs.mistral.ai/capabilities/reasoning/adjustable now 404s (studio-api page is the live successor). Only the OpenAPI schema (https://docs.mistral.ai/api/endpoint/chat) lists the wider enum \"none\"|\"minimal\"|\"low\"|\"medium\"|\"high\"|\"xhigh\"; minimal/low/medium/xhigh have NO documented semantics anywhere. Treat only \"high\" and \"none\" as verified; the four intermediate values are unverified — do not emit them without a live accept/coerce/400 probe. prompt_mode enum: \"reasoning\" only."
  default_when_omitted: "unpublished — no page states a default. Probe-only (needs API key)."
  unsupported_value_behavior: "unpublished AND unprobed (no MISTRAL_API_KEY available this run). Whether minimal/low/medium/xhigh are accepted, coerced to a neighboring tier, or rejected with 400 is unknown; likewise the error shape for a bogus value and the behavior of reasoning_effort on non-reasoning ids. See unresolved for the exact curls."
  per_model:
    - id: mistral-medium-2604
      mode: optional
      effort: "verified adjustable values: [\"high\", \"none\"] (docs recommend \"high\" for agentic/code). OpenAPI-only extras minimal/low/medium/xhigh are UNVERIFIED (no documented semantics; not probed)."
    - id: mistral-small-2603
      mode: optional
      effort: "verified adjustable values: [\"high\", \"none\"] (mistral-small-latest / Mistral Small 4; \"none\" == fast Small-3.2-style chat). OpenAPI-only extras minimal/low/medium/xhigh are UNVERIFIED."
context:
  per_model:
    - id: mistral-medium-2604
      max_output: "unpublished — CONFIRMED: model card + models overview show only 256k context, no output cap; /v1/models has max_context_length but no output field. Probe-only."
    - id: mistral-large-2512
      max_output: "unpublished — CONFIRMED (no output cap on card). Probe-only."
    - id: mistral-small-2603
      max_output: "unpublished — CONFIRMED (no output cap on card). Probe-only."
    - id: codestral-2508
      max_output: "unpublished — CONFIRMED (card states no output cap; prior ~4K claim not on card). Probe-only."
    - id: ministral-8b-2512
      max_output: "unpublished — CONFIRMED (no output cap on card). Probe-only."
````

**Still unresolved (needs a live key):**

````
No MISTRAL_API_KEY in this environment, so runtime behavior is unprobed. A maintainer with a key must run (bash; expects $MISTRAL_API_KEY):
# 1. Intermediate reasoning_effort values — accepted / coerced / 400? (repeat for each: minimal low medium xhigh, and a bogus "maximum")
for v in minimal low medium xhigh maximum; do echo "== $v =="; curl -s -w '\nHTTP %{http_code}\n' https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $MISTRAL_API_KEY" -H 'Content-Type: application/json' -d "{\"model\":\"mistral-medium-2604\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":\"$v\",\"max_tokens\":64}"; done
# repeat the loop with "model":"mistral-small-2603"
# 2. Omit-default response shape (thinking-chunk list vs plain string) on the two hybrid ids:
curl -s https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $MISTRAL_API_KEY" -H 'Content-Type: application/json' -d '{"model":"mistral-medium-2604","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}' | python -c 'import sys,json;print(type(json.load(sys.stdin)["choices"][0]["message"]["content"]))'
# 3. reasoning_effort on a non-reasoning id — ignored vs 400 + shape:
curl -s -w '\nHTTP %{http_code}\n' https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $MISTRAL_API_KEY" -H 'Content-Type: application/json' -d '{"model":"mistral-large-2512","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high","max_tokens":64}'
curl -s -w '\nHTTP %{http_code}\n' https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $MISTRAL_API_KEY" -H 'Content-Type: application/json' -d '{"model":"codestral-2508","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high","max_tokens":64}'
# 4. Max-output boundary (per id): request an absurd max_tokens with tiny prompt and read the 400 message for any output ceiling distinct from context length:
curl -s -w '\nHTTP %{http_code}\n' https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $MISTRAL_API_KEY" -H 'Content-Type: application/json' -d '{"model":"mistral-medium-2604","messages":[{"role":"user","content":"hi"}],"max_tokens":999999}'
# also GET https://api.mistral.ai/v1/models/mistral-medium-2604 and inspect for any max_output/max_completion field beyond max_context_length.
````

### Verifier notes (minor, not applied)

- **mistral-medium-2604 long_context_pricing: flat ($1.5/M in, $7.5/M out)** — Unverifiable via fetch and anomalous. The official pricing pages (mistral.ai/pricing, /pricing/api) did not render per-model modern figures for me, and the only external figure surfaced for the Medium line is Mistral Medium 3.1 at $0.40/M in, $2.00/M out (cloudzero/pricepertoken). $1.5/$7.5 is ~3.75x higher than the known Medium-line price, so it is either a version-specific increase for 3.5 or a  *Suggested: Verify mistral-medium-2604 input/output price against the live /pricing/api page or /v1/models before shipping; if it is actually ~$0.40/$2.00, the $1.5/$7.5 figure is wrong. Mark as probe-confirmed rather than asserted.*

---

## 14. xai

*Verification: corrected after adversarial review — 6 challenge(s), 2 fatal/major; corrections are applied in the block.*

````yaml
provider: xai
endpoints:
  - url: https://api.x.ai/v1/chat/completions
    protocol: openai-chat
  - url: https://api.x.ai/v1/responses
    protocol: openai-responses
checked: 2026-07-19
sources:
  - https://docs.x.ai/docs/models
  - https://docs.x.ai/docs/guides/reasoning
  - https://docs.x.ai/developers/model-capabilities/text/reasoning   # reasoning capability page re-opened in correction pass; event names + penalty/stop error statement live here
  - https://docs.x.ai/developers/grok-4-5
  - https://docs.x.ai/developers/migration/may-15-retirement
  - https://docs.x.ai/developers/models/grok-4.5
  - https://docs.x.ai/developers/models/grok-4.3
  - https://docs.x.ai/developers/models/grok-build-0.1
  - https://docs.x.ai/developers/models/grok-4.20-0309-reasoning
  - https://docs.x.ai/developers/models/grok-4.20-0309-non-reasoning
  - https://docs.x.ai/developers/api-reference   # loaded but only a category overview, no parameter detail on this page itself
  - https://docs.x.ai/developers/rest-api-reference/inference/chat   # documents BOTH /v1/chat/completions and /v1/responses request bodies
  - https://docs.x.ai/developers/rest-api-reference/inference/models
  - https://models.dev/xai   # registry, leads only — used solely to record max-output conflicts
reasoning:
  param: "Chat Completions: top-level `reasoning_effort` (string|null) only — no nested `reasoning` object is documented for /v1/chat/completions. Responses API: BOTH shapes are documented — nested `reasoning` object with `effort` (primary), AND top-level `reasoning_effort` as an explicitly documented compatibility fallback: 'reasoning_effort alternative to reasoning configuration. This is a non-standard field meant to ease user experience. We only look at this if the reasoning field is unset.' Precedence on /v1/responses: `reasoning` wins; `reasoning_effort` is consulted only when `reasoning` is unset. The only undocumented cross-surface emission is nested `reasoning:{effort}` sent to /v1/chat/completions."
  values: "grok-4.3: `none` | `low` | `medium` | `high` (chat API reference; `none` disables reasoning completely — same value list is annotated on the Responses nested reasoning.effort field). grok-4.5: `low` | `medium` | `high` (reasoning guide + /developers/grok-4-5; NO `none`, 'Reasoning cannot be disabled'). grok-4.20-multi-agent: `low` | `medium` | `high` | `xhigh` (controls agent count, 4 or 16 agents — not in Sylliptor catalog). grok-4.20-0309-reasoning / -non-reasoning snapshots: no documented values (mode is chosen by model id). grok-build-0.1: no documented values."
  default_when_omitted: "grok-4.3: `low` ('this is the default if not specified' — chat API reference). grok-4.5: `high` (reasoning guide). Others: unpublished."
  unsupported_value_behavior: "unknown — the REST reference documents no error responses, status codes, or error body shapes for invalid reasoning_effort values or unsupported models. Documented error case: reasoning_effort combined with `presencePenalty`, `frequencyPenalty`, or `stop` 'returns an error' (guide uses SDK camelCase spellings; wire spellings presence_penalty/frequency_penalty/stop — status code and body shape unpublished). See probes_needed."
  per_model:
    - id: grok-4.5
      mode: always-on
      effort: "adjustable: [low, medium, high], default high; cannot be disabled (no `none`)"
    - id: grok-build-0.1
      mode: always-on
      effort: "n/a — model page says 'Reasoning: Yes' but no reasoning_effort values documented; chat API reference says reasoning_effort is 'Only supported by grok-4.3' (probe needed)"
    - id: grok-4.3
      mode: optional
      effort: "adjustable: [none, low, medium, high], default low; `none` disables reasoning completely"
    - id: grok-4.20-0309-reasoning
      mode: always-on
      effort: "n/a — mode fixed by model id; no effort values documented for the snapshot; whether reasoning_effort is accepted at all is unknown"
    - id: grok-4.20-0309-non-reasoning
      mode: none
      effort: "n/a — page states 'Reasoning: No'; behavior when reasoning_effort is sent is undocumented (probe needed)"
  silent_substitutions: "Three classes. (1) Retirement redirects after 2026-05-15 12:00 PM PT: grok-4-1-fast-reasoning / grok-4-fast-reasoning / grok-4-0709 → grok-4.3 with `low` reasoning effort; grok-4-1-fast-non-reasoning / grok-4-fast-non-reasoning / grok-3 → grok-4.3 with `none`; grok-code-fast-1 → grok-build-0.1; grok-imagine-image-pro → grok-imagine-image-quality. Redirected requests billed at target-model rates ($1.25/$2.50 for grok-4.3). These are triggered by model slug, not by reasoning flags. (2) Alias trap: `grok-build-latest` is documented as an alias of grok-4.5 (NOT grok-build-0.1) — resolves to a different, 2x-priced model than the name suggests. `grok-latest` and `grok-4.3-latest` alias grok-4.3; grok-build-0.1 carries aliases grok-code-fast-1, grok-code-fast, grok-code-fast-1-0825. (3) Responses-surface precedence: if a request to /v1/responses carries BOTH `reasoning` and `reasoning_effort`, the top-level `reasoning_effort` is silently ignored ('We only look at this if the reasoning field is unset') — not an error, just a dropped knob."
  tools_and_streaming: "Chat Completions response/SSE chunks carry `reasoning_content` (string|null, 'The reasoning trace generated by the model'; `chunk.reasoning_content` when streaming). Responses API streams reasoning via event types `response.reasoning_text.delta` / `response.reasoning_summary_text.delta` (confirmed verbatim on the reasoning capability page); raw traces retrievable via `include: [\"reasoning.encrypted_content\"]`. NOTE: output items of type \"reasoning\" are NOT documented on any xAI page opened — that shape is expected per OpenAI Responses conventions only, do not rely on it. Output-cap accounting DIFFERS per surface: Chat `max_completion_tokens` 'only applies to visible output tokens (i.e. does not apply to tokens used for reasoning or function calls)'; Responses `max_output_tokens` 'includes both output and reasoning tokens'. `max_tokens` is [DEPRECATED] in favor of max_completion_tokens. Reasoning tokens are billed as part of total consumption. Function calling + structured outputs supported on all five catalog models. Interleaved thinking with tool calls: not explicitly documented. Caching: set the `prompt_cache_key` REQUEST-BODY parameter on EITHER surface — it is documented on both: /v1/chat/completions ('A stable cache key for best-effort sticky routing / prompt-cache hits across requests sharing a prompt prefix. Plumbed to x-grok-conv-id, same as on /v1/responses') and /v1/responses ('Plumbed to x-grok-conv-id for Open Responses compatibility, used for routing'). `x-grok-conv-id` is the INTERNAL routing target prompt_cache_key is plumbed to; it is NOT documented as a client-settable header — do not set it."
  compat_passthrough: "openai-chat (/v1/chat/completions): top-level `reasoning_effort` passes (documented first-class); nested `reasoning:{effort}` is NOT documented on this surface — the one real cross-surface hazard. openai-responses (/v1/responses): nested `reasoning:{\"effort\": ...}` passes (primary, documented first-class) AND top-level `reasoning_effort` ALSO passes as a documented non-standard fallback honored only when `reasoning` is unset — so a chat-shaped payload's reasoning knob survives on the Responses surface. No Anthropic-compatible surface documented on the pages opened. Sylliptor targets the openai-chat surface: emit top-level `reasoning_effort` there; that same spelling also works on Responses via the fallback, but prefer nested `reasoning.effort` on Responses and never send both (reasoning silently wins)."
discovery:
  endpoint: "GET https://api.x.ai/v1/models (also GET /v1/models/{model_id}); richer per-family listings exist: GET /v1/language-models (+ /{model_id}), GET /v1/image-generation-models, GET /v1/video-generation-models"
  auth: api-key
  metadata: "ids+context+pricing on /v1/models: id, aliases, context_length, created, object, owned_by, prompt_text_token_price, cached_prompt_text_token_price, prompt_image_token_price, completion_text_token_price, prompt_text_token_price_long_context, completion_text_token_price_long_context, cached_prompt_text_token_price_long_context, long_context_threshold, image_price. /v1/language-models adds fingerprint, version, input_modalities, output_modalities, search_price (but its documented field list does NOT include context_length, and neither listing exposes reasoning capability/effort values or max output)."
  account_scoped: "yes: /v1/models 'lists all models available to authenticated API key' — roster varies per key; regional gating exists (grok-4.5 'isn't available in the API console for EU users yet')"
  rate_limits_or_caching: undocumented
  verdict: augment-static
  refresh_strategy: "on-config-open with a daily cache fallback: drive roster, context_length, aliases, and long_context_threshold/pricing from GET /v1/models (it is the only listing documented to carry context_length); keep reasoning modes/effort values static from this research, since no discovery endpoint exposes them"
context:
  per_model:
    - id: grok-4.5
      context: "500,000 tokens (no tier splits documented)"
      max_output: unpublished
      long_context_pricing: "boundary at 200k prompt tokens; <200k: $2.00 in / $6.00 out / $0.30 cached; >=200k: $4.00 in / $12.00 out / $0.60 cached; 'Requests whose prompt reaches 200k tokens are billed at the higher rate for all tokens in the request'"
      overpromise_risk: "no (context); but regional availability gating exists (EU console) — roster, not size"
    - id: grok-build-0.1
      context: "256,000 tokens"
      max_output: unpublished
      long_context_pricing: "boundary at 200k prompt tokens; <200k: $1.00 in / $2.00 out / $0.20 cached; >=200k: $2.00 in / $4.00 out / $0.40 cached (cached long-context rate confirmed on both the model page and the docs/models pricing table)"
      overpromise_risk: no
    - id: grok-4.3
      context: "1,000,000 tokens"
      max_output: "unpublished (models.dev claims 30,000 — third-party, unconfirmed)"
      long_context_pricing: "boundary at 200k prompt tokens; <200k: $1.25 in / $2.50 out / $0.20 cached; >=200k: $2.50 in / $5.00 out / $0.40 cached; whole request billed at higher rate once boundary reached"
      overpromise_risk: no
    - id: grok-4.20-0309-reasoning
      context: "1,000,000 tokens (docs; third-party 2M claims from prior pass NOT found in official docs — see conflicts)"
      max_output: "unpublished (third parties conflict ~30K vs 131K; docs publish no number)"
      long_context_pricing: "boundary at 200k prompt tokens; <200k: $1.25 in / $2.50 out / $0.20 cached; >=200k: $2.50 in / $5.00 out / $0.40 cached"
      overpromise_risk: no
    - id: grok-4.20-0309-non-reasoning
      context: "1,000,000 tokens"
      max_output: "unpublished (same third-party conflict as reasoning twin)"
      long_context_pricing: "boundary at 200k prompt tokens; <200k: $1.25 in / $2.50 out / $0.20 cached; >=200k: $2.50 in / $5.00 out / $0.40 cached"
      overpromise_risk: no
  token_counting_endpoint: "none documented in the REST reference models section ('No dedicated tokenization or token-counting endpoint is documented'); note pricing-tier detection therefore needs client-side estimation against long_context_threshold"
probes_needed:
  - question: "What HTTP status + error body does grok-4.5 return for reasoning_effort:\"none\" (a value valid on grok-4.3 but not 4.5)?"
    probe: "curl -s -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $XAI_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"grok-4.5\",\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Does grok-4.20-0309-non-reasoning reject, ignore, or coerce reasoning_effort? (status code + body shape)"
    probe: "curl -s -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $XAI_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"grok-4.20-0309-non-reasoning\",\"reasoning_effort\":\"high\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Does grok-build-0.1 accept reasoning_effort at all (chat reference says only grok-4.3 supports it; guide lists grok-4.5 too — grok-build is documented nowhere)?"
    probe: "curl -s -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $XAI_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"grok-build-0.1\",\"reasoning_effort\":\"low\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Exact status/body when reasoning_effort is combined with stop sequences (guide says it 'returns an error' but publishes no code/shape) — agentic CLIs routinely send stop"
    probe: "curl -s -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $XAI_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"grok-4.3\",\"reasoning_effort\":\"low\",\"stop\":[\"\\n\\n\"],\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "What does /v1/chat/completions do with a nested reasoning:{effort} object (undocumented on that surface): reject as unknown field, or ignore?"
    probe: "curl -s -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $XAI_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"grok-4.3\",\"reasoning\":{\"effort\":\"high\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Does the live /v1/models payload for this account include context_length for all five catalog models, and does /v1/language-models really omit context_length? Also confirms account-scoped roster."
    probe: "curl -s https://api.x.ai/v1/models -H \"Authorization: Bearer $XAI_API_KEY\" ; curl -s https://api.x.ai/v1/language-models -H \"Authorization: Bearer $XAI_API_KEY\""
  - question: "Are max output tokens observable live (finish via length)? Docs publish no max_output for any catalog model; third parties conflict (30K vs 131K for 4.20 family)."
    probe: "curl -s -X POST https://api.x.ai/v1/chat/completions -H \"Authorization: Bearer $XAI_API_KEY\" -H \"Content-Type: application/json\" -d '{\"model\":\"grok-4.20-0309-reasoning\",\"max_completion_tokens\":200000,\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 upward, one number per line, do not stop.\"}]}'"
conflicts:
  - "https://docs.x.ai/developers/rest-api-reference/inference/chat annotates reasoning_effort as 'Only supported by grok-4.3' (values none|low|medium|high, default low) on BOTH surfaces — the /v1/chat/completions reasoning_effort parameter AND the /v1/responses nested reasoning.effort field carry the identical sentence; meanwhile https://docs.x.ai/developers/model-capabilities/text/reasoning and https://docs.x.ai/developers/grok-4-5 document reasoning_effort on grok-4.5 (low|medium|high, default high) and grok-4.20-multi-agent (adds xhigh). Recorded, not resolved — the reference's 'only' claim looks stale on both surfaces, but per rule it is not voted away."
  - "Original pass reported the https://docs.x.ai/docs/models overview table rendering grok-4.5 with Reasoning: No; a 2026-07-19 re-fetch found the table has NO Reasoning column at all (Model/Context/Input/Cached/Output only), so the earlier reading was almost certainly a scrape/rendering artifact. The authoritative model page https://docs.x.ai/developers/models/grok-4.5 says Reasoning: Yes. Kept only as a provenance note."
  - "Max output: official pages publish no max output for any catalog model; https://models.dev/xai claims 30,000 for grok-4.3 and both grok-4.20 snapshots, 500,000 for grok-4.5, 256,000 for grok-build-0.1. Registry vs official-silence — treat as unpublished."
  - "grok-4.20 context: prior-pass third parties claimed 2M; https://docs.x.ai/docs/models and both 4.20 model pages say 1,000,000; no 2M source located this session. Docs say 1M."
notes: "CORRECTION PASS 2026-07-19: verifier challenges accepted on all six points. Two API surfaces with DIFFERENT primary reasoning spellings and DIFFERENT output-cap semantics: chat top-level `reasoning_effort` + `max_completion_tokens` (excludes reasoning tokens) vs responses nested `reasoning.effort` + `max_output_tokens` (includes reasoning tokens) — Sylliptor's tier-aware accounting must branch on surface. HOWEVER the spellings are NOT strictly non-interchangeable: /v1/responses also documents top-level `reasoning_effort` as a non-standard compatibility fallback honored only when `reasoning` is unset (so the only undocumented cross-surface emission is nested reasoning → /v1/chat/completions, and sending BOTH fields to /v1/responses silently drops reasoning_effort). Caching: use the `prompt_cache_key` body parameter on either surface; `x-grok-conv-id` is internal plumbing, not a client header — the original contract had this inverted. Effort value sets are per-family, not global: `none` exists ONLY on grok-4.3; `xhigh` ONLY on grok-4.20-multi-agent. The 4.20 catalog snapshots select mode by model id, so Sylliptor's reasoning_mode toggle for that family should swap the model id, not send a param. `grok-build-latest` aliases grok-4.5 (2x price), not grok-build-0.1 — never derive it. Long-context billing is cliff-edge: reaching 200k prompt tokens bills the ENTIRE request at the higher rate on all five models (grok-build-0.1's >=200k cached rate is $0.40, previously omitted). No token-counting endpoint. Responses 'reasoning' output item type is NOT in xAI docs — treat as OpenAI-convention conjecture. grok-4.20-multi-agent-0309 exists in docs but is absent from the applied catalog."
````

---

## 15. cohere

*Verification: verified-clean — 5 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: cohere
endpoints:
  - url: https://api.cohere.ai/compatibility/v1
    protocol: openai-chat
  - url: https://api.cohere.com/v2/chat
    protocol: native
checked: 2026-07-19
sources:
  - https://docs.cohere.com/docs/compatibility-api
  - https://raw.githubusercontent.com/cohere-ai/cohere-developer-experience/main/fern/pages/v2/text-generation/compatibility-api.mdx
  - https://docs.cohere.com/docs/reasoning
  - https://docs.cohere.com/docs/reasoning.md
  - https://docs.cohere.com/reference/chat
  - https://docs.cohere.com/docs/models
  - https://docs.cohere.com/reference/models (FAILED)   # 404; real page is /reference/list-models
  - https://docs.cohere.com/reference/list-models
  - https://docs.cohere.com/docs/command-a-plus
  - https://raw.githubusercontent.com/cohere-ai/cohere-developer-experience/main/fern/pages/models/the-command-family-of-models/command-a-plus.mdx
  - https://docs.cohere.com/docs/command-a-reasoning
  - https://github.com/cohere-ai/cohere-developer-experience/blob/main/fern/pages/models/the-command-family-of-models/command-a-reasoning.mdx?plain=1
  - https://docs.cohere.com/changelog/command-a-plus-05-2026
  - https://docs.cohere.com/docs/rate-limits
  - https://docs.cohere.com/docs/streaming
  - https://api.github.com/repos/cohere-ai/cohere-developer-experience/contents/fern/pages/v2/text-generation
  - https://api.github.com/repos/cohere-ai/cohere-developer-experience/contents/fern/pages/models/the-command-family-of-models
reasoning:
  param: >-
    Compat surface (/compatibility/v1/chat/completions): "reasoning_effort" (top-level string).
    Native Chat V2 (/v2/chat): "thinking" (object) with fields "type" and "token_budget";
    exact doc snippets: thinking={ "type": "disabled" } and thinking = { "token_budget": 500 }.
    "thinking" is NOT listed among the compat surface's supported parameters; "reasoning_effort" is not a native V2 parameter.
  values: >-
    reasoning_effort (compat): only "none" and "high" — verbatim: "Currently, only `none` and `high` are supported for
    `reasoning_effort`. These correspond to enabling or disabling `thinking` in the Cohere Chat API. Passing `medium` or
    `low` is not supported." Applies to reasoning-capable models (command-a-reasoning-08-2025, command-a-plus-05-2026).
    thinking.type (native): "enabled" | "disabled"; thinking.token_budget (native): positive integer, guide recommends
    max ~31K on command-a-reasoning (leave >=1K tokens for the response).
  default_when_omitted: >-
    Native V2: "Reasoning is enabled by default for models that support it, but can be turned off by setting
    `"type": "disabled"`" (reference/chat). Reasoning guide: "Thinking is enabled by default". Non-reasoning models
    (command-a-03-2025, command-r7b-12-2024) have no thinking. Compat surface: default when reasoning_effort omitted is
    NOT documented; native default (thinking on) presumably applies to reasoning models — unconfirmed, see probes.
  unsupported_value_behavior: >-
    unknown — docs state medium/low are "not supported" for reasoning_effort but do not document whether that is an
    HTTP error, silent ignore, or coercion (no status code or error body given). Native V2 errors use body
    {"message": "...", "id": "..."} across 400/401/403/404/422/429/498/499/500/501/503/504. Behavior of thinking sent
    to a non-reasoning model: undocumented. See probes.
  per_model:
    - id: command-a-plus-05-2026
      mode: optional   # thinking on by default, disable-able; docs/models lists it as a reasoning model and its model page defers thinking control to the Reasoning guide
      effort: "adjustable: compat reasoning_effort [none, high]; native thinking.type [enabled, disabled] + thinking.token_budget (integer, native only)"
    - id: command-a-reasoning-08-2025
      mode: optional   # "Thinking is enabled by default"; disable via thinking={"type":"disabled"} (native) or reasoning_effort:"none" (compat)
      effort: "adjustable: compat reasoning_effort [none, high]; native thinking.type [enabled, disabled] + thinking.token_budget (recommended <=31K, leave >=1K for response)"
    - id: command-a-03-2025
      mode: none
      effort: n/a
    - id: command-r7b-12-2024
      mode: none
      effort: n/a
  silent_substitutions: none documented anywhere; no aliasing of model ids triggered by reasoning flags found.
  tools_and_streaming: >-
    Native V2 SSE event types: message-start, content-start, content-delta, content-end, citation-start, citation-end,
    tool-plan-delta, tool-call-start, tool-call-delta, tool-call-end, message-end. There is NO dedicated thinking event:
    thinking streams inside content-delta and is read as event.delta.message.content.thinking (reasoning guide code);
    final text arrives in the same event type's text field. Interleaved thinking with tool calls: not documented.
    Whether thinking tokens count against max_tokens is not stated explicitly, but the guide's arithmetic implies they
    share the output limit: on a 32K-max-output model it recommends a 31K token_budget while telling you to "leave at
    least 1K tokens for the response"; "When the budget is exceeded, the model will immediately proceed with the final
    response". max_tokens defaults to the model's maximum output token limit when unset. How thinking content is
    surfaced in COMPAT responses (e.g. a reasoning_content field) is completely undocumented — see probes.
  compat_passthrough: >-
    /compatibility/v1: reasoning_effort = passes (documented; mapped to native thinking on/off). thinking = unknown
    (not in the supported-parameter list; docs do not say whether unknown params are dropped or rejected — probe).
    token_budget = no compat exposure at all (cannot set a thinking budget over the compat surface). Explicitly
    unsupported compat chat params (documented): store, metadata, logit_bias, top_logprobs, n, modalities, prediction,
    audio, service_tier, parallel_tool_calls — failure mode for sending them undocumented.
discovery:
  endpoint: "GET https://api.cohere.com/v1/models (native only; NO /models endpoint documented on the /compatibility/v1 surface)"
  auth: bearer
  metadata: >-
    ids+context+capabilities: per model — name, is_deprecated, endpoints, finetuned, context_length, tokenizer_url,
    default_endpoints, features, sampling_defaults {temperature, k, p, frequency_penalty, presence_penalty,
    max_tokens_per_doc}; pagination via next_page_token; query params page_size (1-1000, default 20), page_token,
    endpoint (chat|embed|classify|summarize|rerank|rate|generate), default_only. No pricing, no max_output tokens,
    and it is unverified whether the features array flags reasoning/thinking support (probe).
  account_scoped: "yes: finetuned models appear per account (finetuned boolean); no documented tier-based differences in context_length for base models"
  rate_limits_or_caching: >-
    undocumented for /v1/models specifically; general limits page documents Chat at 20 req/min (trial) / 500 req/min
    (production, older models) and notes newer models (Command A+, Command A Reasoning, Command A Translate, Command A
    Vision) require contacting sales for production limits; Tokenize 100/min trial, 2000/min production.
  verdict: augment-static
  refresh_strategy: daily-cache   # use live context_length + is_deprecated + endpoints to correct/prune the static catalog; reasoning capability and max_output must stay static (not in the listing) — and the listing is on the native host, not the compat base URL Sylliptor ships
context:
  per_model:
    - id: command-a-plus-05-2026
      context: "128,000 (changelog phrasing: '128K input, 64K output'); no tier splits documented"
      max_output: "64,000"
      long_context_pricing: flat (no long-context price boundary documented on any page opened)
      overpromise_risk: "no (no documented tier splits), but note the 128K figure is INPUT context — smaller than the 256K of command-a-reasoning/command-a-03-2025; do not copy 256K onto this model"
    - id: command-a-reasoning-08-2025
      context: "256,000; no tier splits documented"
      max_output: "32,000"
      long_context_pricing: flat (no boundary documented)
      overpromise_risk: "no per docs; but with thinking enabled (the default) the implied shared 32K output budget means effective answer length can shrink to near zero if token_budget is unset/high — account for thinking in output headroom"
    - id: command-a-03-2025
      context: "256,000; no tier splits documented"
      max_output: "8,000"
      long_context_pricing: flat (no boundary documented)
      overpromise_risk: no
    - id: command-r7b-12-2024
      context: "128,000; no tier splits documented"
      max_output: "4,000"
      long_context_pricing: flat (no boundary documented)
      overpromise_risk: no
  token_counting_endpoint: "POST https://api.cohere.com/v1/tokenize (native; rate limited 100 req/min trial, 2,000 req/min production; no counting endpoint on the compat surface)"
probes_needed:
  - question: Does the compat surface reject or silently ignore reasoning_effort values "medium"/"low" (the OpenAI default a generic CLI sends), and what is the error body?
    probe: |
      curl -sS -X POST https://api.cohere.ai/compatibility/v1/chat/completions -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-reasoning-08-2025","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"medium","max_tokens":32}'
  - question: Is the native `thinking` object dropped, forwarded, or rejected when sent through /compatibility/v1?
    probe: |
      curl -sS -X POST https://api.cohere.ai/compatibility/v1/chat/completions -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-reasoning-08-2025","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"max_tokens":32}'
  - question: Confirm no /models listing exists on the compat surface (404 vs 401 vs a working list).
    probe: |
      curl -sS -i https://api.cohere.ai/compatibility/v1/models -H "Authorization: Bearer $COHERE_API_KEY"
  - question: On a reasoning model over compat with reasoning_effort omitted, is thinking on (native default) and where does thinking text land in the OpenAI-shape response (reasoning_content? prefixed into content?)?
    probe: |
      curl -sS -X POST https://api.cohere.ai/compatibility/v1/chat/completions -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-reasoning-08-2025","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":2048}'
  - question: Does native V2 accept thinking={"type":"enabled"} WITHOUT token_budget (reference rendering implies token_budget is required when enabled; the guide's default-on behavior implies it is not)?
    probe: |
      curl -sS -X POST https://api.cohere.com/v2/chat -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-reasoning-08-2025","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"},"max_tokens":256}'
  - question: What happens when thinking is sent to a non-reasoning model on native V2 (HTTP code + {"message","id"} body vs silent ignore)?
    probe: |
      curl -sS -X POST https://api.cohere.com/v2/chat -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-03-2025","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","token_budget":500},"max_tokens":64}'
  - question: What happens when reasoning_effort (any value) is sent to a non-reasoning model over compat (command-a-03-2025 / command-r7b-12-2024)?
    probe: |
      curl -sS -X POST https://api.cohere.ai/compatibility/v1/chat/completions -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-03-2025","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high","max_tokens":32}'
  - question: Does the documented-unsupported parallel_tool_calls (sent by many OpenAI-SDK agents) error or get ignored on compat?
    probe: |
      curl -sS -X POST https://api.cohere.ai/compatibility/v1/chat/completions -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-03-2025","messages":[{"role":"user","content":"list files"}],"tools":[{"type":"function","function":{"name":"ls","description":"list","parameters":{"type":"object","properties":{}}}}],"parallel_tool_calls":false,"max_tokens":64}'
  - question: Do reasoning tokens consume max_tokens (send tiny max_tokens with thinking on and check finish reason / empty content), and does the v1/models features array flag reasoning support?
    probe: |
      curl -sS -X POST https://api.cohere.com/v2/chat -H "Authorization: Bearer $COHERE_API_KEY" -H "Content-Type: application/json" -d '{"model":"command-a-reasoning-08-2025","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":64}' ; curl -sS "https://api.cohere.com/v1/models?endpoint=chat&page_size=100" -H "Authorization: Bearer $COHERE_API_KEY"
conflicts:
  - https://docs.cohere.com/reference/chat (rendered summary) presents thinking.token_budget as required when type is "enabled"; https://docs.cohere.com/docs/reasoning shows thinking enabled by default with no budget set and passes token_budget without type — whether enabled-without-budget is valid is unresolved (probe queued).
  - https://docs.cohere.com/docs/models capability table marks command-a-plus-05-2026 as a reasoning model (and its model page defers to the Reasoning guide for thinking control), but https://docs.cohere.com/docs/reasoning names only command-a-reasoning-08-2025 — doc drift on which models accept thinking; recorded, not resolved.
  - Prior-refresh registry-derived assumption said reasoning is uncontrollable over the compat surface; official https://docs.cohere.com/docs/compatibility-api documents reasoning_effort (none|high) on that surface — official wins.
notes: >-
  Load-bearing for Sylliptor: (1) over the shipped compat surface, reasoning on/off is expressed ONLY as
  reasoning_effort "none"|"high" — never emit "medium"/"low" (docs: "not supported", failure mode undocumented) and
  never emit the native thinking object there; (2) thinking token budgets are native-V2-only, so tier-aware output
  accounting over compat must assume default budget behavior; (3) the compat base URL is api.cohere.ai while native is
  api.cohere.com — discovery/tokenize live on the native host only; (4) compat surface supported chat params are
  exactly: model, messages, stream, reasoning_effort, response_format, tools, temperature, max_tokens, stop, seed,
  top_p, frequency_penalty, presence_penalty; (5) production rate limits for command-a-plus-05-2026 and
  command-a-reasoning-08-2025 are sales-gated (older models get 500 req/min), trial keys 20 req/min on all chat
  models; (6) reasoning guide budget guidance (31K rec / >=1K reserve on a 32K-max-output model) implies thinking and
  answer share the output limit — treat max_output as inclusive of thinking until a probe proves otherwise.
````

### Verifier notes (minor, not applied)

- **Verbatim quote in reasoning.values: "Passing `medium` or `low` is not supported."** — Presented as verbatim but truncated: the compat doc (docs page and fern source compatibility-api.mdx) actually ends the sentence "...is not supported at this time." The dropped qualifier signals Cohere may add medium/low later, which slightly changes how durable the none|high constraint should be treated in a static contract. *Suggested: Restore the full sentence including "at this time" and note the constraint may be version-dependent.*
- **Reasoning guide quote: "Thinking is enabled by default"** — Quote-precision: the guide's actual sentence is "For reasoning models, `thinking` is enabled by default." The contract's shortened form drops the model-scoping clause, though the surrounding contract text preserves the meaning. *Suggested: Quote the full sentence with the "For reasoning models" scope.*
- **conflicts: "docs.cohere.com/docs/models capability table marks command-a-plus-05-2026 as a reasoning model" (echoed in per_model comment "docs/models lists it as a reasoning model")** — Overstatement: the models table describes Command A+ as having "agentic, reasoning, and world-class translation capabilities" but does not brand it "a reasoning model" (that label is reserved for Command A Reasoning: "Cohere's first reasoning model"). The per_model mode: optional verdict still stands on stronger evidence — the command-a-plus model page explicitly defers thinking control to the Rea *Suggested: Cite the command-a-plus model page's thinking deferral and the compat doc's examples as the basis for reasoning capability; soften the docs/models table claim to "describes reasoning capabilities".*
- **conflicts: "docs.cohere.com/reference/chat (rendered) presents thinking.token_budget as required when thinking.type is 'enabled'"** — The reference page as fetched today reads the opposite: within the thinking object, `type` is the required field and `token_budget` is optional ("When type is 'enabled', the token_budget field is optional, not required"). The alleged docs conflict may not exist — either the page changed or the researcher misread the rendered schema. Harmless because the contract records it only as an unresolved co *Suggested: Re-read the reference schema; likely resolve the conflict as "token_budget optional when enabled" and demote the probe to confirmation-only.*
- **reasoning.values: reasoning_effort "Applies to reasoning-capable models (command-a-reasoning-08-2025, command-a-plus-05-2026)"** — The compat doc never states which models reasoning_effort applies to — it lists the parameter with no model-specific callout. The model applicability is an inference (reasonable, since the page's examples use command-a-plus-05-2026 and the semantics map to thinking), presented as if documented. *Suggested: Mark the model applicability as inferred from the Reasoning guide + page examples, not stated by the compat doc.*

---

## 16. openrouter

*Verification: corrected after adversarial review — 2 challenge(s), 1 fatal/major; corrections are applied in the block.*

````yaml
provider: openrouter
endpoints:
  - url: https://openrouter.ai/api/v1
    protocol: openai-chat
  - url: https://openrouter.ai/api/v1/responses
    protocol: openai-responses
checked: 2026-07-19
sources:
  - https://openrouter.ai/docs/use-cases/reasoning-tokens
  - https://openrouter.ai/docs/api-reference/overview
  - https://openrouter.ai/docs/api-reference/list-available-models (FAILED)
  - https://openrouter.ai/docs/api-reference/list-endpoints-for-a-model/~explorer (FAILED)
  - https://openrouter.ai/docs/guides/overview/models
  - https://openrouter.ai/docs/features/provider-routing
  - https://openrouter.ai/docs/api-reference/errors
  - https://openrouter.ai/docs/api_reference/responses/overview
  - https://openrouter.ai/docs/api_reference/responses/reasoning
  - https://openrouter.ai/api/v1/models (live GET, HTTP 200, no auth)
  - https://openrouter.ai/api/v1/model/z-ai/glm-5.2 (live GET, HTTP 200, no auth — pricing re-verified)
  - https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints (live GET, HTTP 200, no auth)
  - https://openrouter.ai/api/v1/models/anthropic/claude-sonnet-5/endpoints (live GET, HTTP 200, no auth)
  - https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-pro/endpoints (live GET, HTTP 200, no auth)
  - https://openrouter.ai/api/v1/models/openai/gpt-5.6-terra/endpoints (live GET, HTTP 200, no auth)
  - https://openrouter.ai/api/v1/models/user (live GET, HTTP 401 without key — auth required)
reasoning:
  param: 'reasoning (object) with subfields copied exactly: "effort", "max_tokens", "exclude", "enabled", "context" (GPT-5.6+ only), "mode" (GPT-5.6+ only). Legacy top-level "include_reasoning" (boolean) is deprecated: include_reasoning:true == reasoning:{}, include_reasoning:false == reasoning:{exclude:true}. On the beta Responses surface (/api/v1/responses) the shape is reasoning:{effort} ONLY.'
  values: 'effort: "max" | "xhigh" | "high" | "medium" | "low" | "minimal" | "none" (full set; per-model subset in live metadata reasoning.supported_efforts — anthropic/claude-sonnet-5 + claude-opus-4.8: [max, xhigh, high, medium, low]; openai/gpt-5.6-terra + gpt-5.6-luna: [max, xhigh, high, medium, low, none]; z-ai/glm-5.2 + deepseek/deepseek-v4-pro: [xhigh, high] only). max_tokens: integer (Anthropic thinking budget: min 1024, max 128000). exclude: boolean (default false). enabled: true == reasoning at "medium" effort. context: "auto" | "all_turns" | "current_turn" (GPT-5.6+). mode: "standard" | "pro" (GPT-5.6+; "pro" REROUTES to the matching *-pro model). effort and max_tokens are mutually exclusive — the docs read "One of the following (not both): effort or max_tokens" (re-verified 2026-07-19). Anthropic effort->budget formula: budget_tokens = max(min(max_tokens * ratio, 128000), 1024) with ratios 0.95 (max/xhigh), 0.8 (high), 0.5 (medium), 0.2 (low), 0.1 (minimal). Gemini: effort maps to thinkingLevel, max_tokens converts to thinkingBudget; Qwen: max_tokens maps to thinking_budget.'
  default_when_omitted: 'Docs (verbatim, re-verified 2026-07-19): "Reasoning tokens are included in the response by default if the model decides to output them." Live metadata refines this per model: reasoning.default_enabled:true + default_effort "medium" for gpt-5.6-terra/luna, default_effort "high" for glm-5.2; claude-sonnet-5/opus-4.8 and deepseek-v4-pro carry default_effort "medium"/"high" but NO default_enabled flag (treat as opt-in). Ambiguity recorded in conflicts.'
  unsupported_value_behavior: 'Per-provider-route, not per-slug: each endpoint has its own supported_parameters. Default (require_parameters:false): providers "that don''t support all the LLM parameters specified in your request can still receive the request, but will ignore unknown parameters" — silently-ignored. With provider.require_parameters:true the request is not routed to non-supporting providers; if none qualify -> http-503, body {"error":{"code":503,"message":"There is no available model provider that meets your routing requirements","metadata":{...}}}. Behavior for a supported param with an OUT-OF-RANGE value (e.g. effort "medium" to a [xhigh,high]-only model) is NOT documented — unknown (see probes). General error shape: {"error":{"code":number,"message":string,"metadata":{...}}}; mid-stream errors arrive as a chunk with top-level "error" and choices[0].finish_reason:"error", then the stream terminates.'
  per_model:
    - id: anthropic/claude-sonnet-5
      mode: optional
      effort: 'adjustable: [max, xhigh, high, medium, low] (default_effort medium; NO "none"/"minimal" — disable by omitting reasoning, not effort:"none"); or reasoning.max_tokens 1024-128000'
    - id: anthropic/claude-opus-4.8
      mode: optional
      effort: 'adjustable: [max, xhigh, high, medium, low] (default_effort medium; NO "none"/"minimal"); or reasoning.max_tokens 1024-128000'
    - id: openai/gpt-5.6-terra
      mode: optional
      effort: 'adjustable: [max, xhigh, high, medium, low, none] (default_enabled:true, default_effort medium — reasoning ON unless effort:"none"); also accepts reasoning.context and reasoning.mode ("pro" reroutes to the *-pro sibling)'
    - id: openai/gpt-5.6-luna
      mode: optional
      effort: 'adjustable: [max, xhigh, high, medium, low, none] (default_enabled:true, default_effort medium); openai/gpt-5.6-luna-pro exists as a separate listed model — mode:"pro" reroutes to it'
    - id: z-ai/glm-5.2
      mode: optional
      effort: 'adjustable: [xhigh, high] ONLY (default_enabled:true, default_effort high; no "none" listed — disable path undocumented, see probes)'
    - id: deepseek/deepseek-v4-pro
      mode: optional
      effort: 'adjustable: [xhigh, high] ONLY (default_effort high, no default_enabled flag)'
  silent_substitutions: 'Two documented: (1) GPT-5.6+: reasoning.mode:"pro" — "OpenRouter reroutes the request to the matching *-pro model" (different pricing); (2) the ":thinking" variant suffix "is no longer supported for Anthropic models. Use the reasoning parameter instead" — old anthropic slugs with :thinking now fail rather than substitute. No effort-level-triggered swaps documented.'
  tools_and_streaming: 'Streaming: reasoning arrives as choices[].delta.reasoning_details — array of objects with type, content, format, id (a plain delta.reasoning string is not shown in the current streaming examples). Tools: preserve reasoning across tool-call turns by echoing back message.reasoning (plaintext) or message.reasoning_details (full structured array — REQUIRED for encrypted/summarized reasoning, i.e. Anthropic signed thinking blocks); docs: "preserving reasoning blocks is useful specifically for tool calling". Output accounting: "Reasoning tokens are considered output tokens and charged accordingly"; on Anthropic "max_tokens must be strictly higher than the reasoning budget to ensure there are tokens available for the final response after thinking" — i.e. reasoning consumes the output budget. Responses surface streams reasoning as event "response.reasoning.delta" with a "delta" field; reasoning appears as an output item {"type":"reasoning", "encrypted_content", "summary":[...]}.'
  compat_passthrough: 'openai-chat /api/v1/chat/completions: passes (native surface for the unified reasoning object). openai-responses /api/v1/responses (beta, stateless — store:true and previous_response_id are rejected with 400): PARTIAL — only reasoning.effort with values "minimal"|"low"|"medium"|"high"; max_tokens/exclude/enabled are NOT part of this surface (docs: "does not use max_tokens, exclude, or enabled fields") and "max"/"xhigh"/"none" are not listed — rejection-vs-strip behavior unknown (probe). No Anthropic-messages compat surface.'
discovery:
  endpoint: 'GET https://openrouter.ai/api/v1/models (query params: output_modalities, supported_parameters, sort, offset, limit [max 1000, default 500]); per-model detail: GET https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints; single model: GET /api/v1/model/{author}/{slug}; account-scoped: GET https://openrouter.ai/api/v1/models/user (401 without key)'
  auth: 'none for /models, /model/{author}/{slug}, and /models/{author}/{slug}/endpoints (verified live HTTP 200 with no key); bearer for /models/user'
  metadata: 'ids+capabilities+pricing — per model: id, canonical_slug, name, created, description, context_length, architecture{input_modalities, output_modalities, tokenizer, instruct_type}, pricing{prompt, completion, request, image, web_search, internal_reasoning, input_cache_read, input_cache_write, input_cache_write_1h, overrides[{min_prompt_tokens, ...both rates}]}, top_provider{context_length, max_completion_tokens, is_moderated}, supported_parameters[], default_parameters, reasoning{mandatory, default_enabled?, supported_efforts[], default_effort}, per_request_limits, knowledge_cutoff, expiration_date, benchmarks, hugging_face_id. NOTE: pricing fields are per-token decimal STRINGS (e.g. glm-5.2 prompt "0.0000009786" == $0.9786/M) and many models omit input_cache_write entirely. /endpoints route adds per-provider: provider_name, context_length, max_completion_tokens, max_prompt_tokens, quantization, status, uptime_last_*, throughput/latency_last_30m, supported_parameters, pricing, supports_implicit_caching.'
  account_scoped: 'no for the base /models list; yes via /models/user — filters by the user''s provider preferences, privacy/ZDR settings, and guardrails, so EFFECTIVE model/provider availability (and therefore effective context/max_output) varies per account even though the public list does not'
  rate_limits_or_caching: 'undocumented in docs; observed live response headers on /models: Cache-Control: public, max-age=300, s-maxage=300, stale-while-revalidate=3600, stale-if-error=3600 (CF-Cache-Status: HIT) — i.e. a 5-minute CDN cache'
  verdict: drive-from-live
  refresh_strategy: 'on-config-open fetch of /api/v1/models (no key needed) with a per-session cache honoring the 300s Cache-Control TTL; lazily fetch /models/{author}/{slug}/endpoints for the selected model to compute worst-case context/max_output across providers; when an API key is configured, prefer /models/user so the roster matches the account''s provider/ZDR filters. The reasoning.supported_efforts + default_effort + mandatory metadata should directly drive the per-model reasoning_mode UI instead of any static table. Pricing must be read from live pricing fields (per-token decimal strings) rather than transcribed — transcription introduced a glm-5.2 error corrected in this pass.'
context:
  per_model:
    - id: anthropic/claude-sonnet-5
      context: '1,000,000 (uniform across all 7 live endpoints: Anthropic, Azure, Google, Amazon Bedrock)'
      max_output: 128000
      long_context_pricing: 'flat as listed on OpenRouter: $2.00/M prompt, $10.00/M completion; cache read $0.20/M, cache write $2.50/M (input_cache_write_1h $4.00/M); no pricing.overrides array'
      overpromise_risk: 'no (all providers list 1M / 128k)'
    - id: anthropic/claude-opus-4.8
      context: '1,000,000 (uniform across live endpoints)'
      max_output: 128000
      long_context_pricing: 'flat as listed: $5.00/M prompt, $25.00/M completion; cache read $0.50/M, cache write $6.25/M (1h $10.00/M); no overrides'
      overpromise_risk: no
    - id: openai/gpt-5.6-terra
      context: '1,050,000 (uniform: OpenAI + Azure endpoints)'
      max_output: 128000
      long_context_pricing: 'boundary at min_prompt_tokens 272,000 (pricing.overrides): <=272k $2.50/M prompt + $15.00/M completion (cache read $0.25/M); >272k $5.00/M prompt + $22.50/M completion (cache read $0.50/M)'
      overpromise_risk: 'no on context; note the price DOUBLES past 272k prompt tokens'
    - id: openai/gpt-5.6-luna
      context: '1,050,000 (uniform)'
      max_output: 128000
      long_context_pricing: 'boundary at min_prompt_tokens 272,000: <=272k $1.00/M prompt + $6.00/M completion; >272k $2.00/M prompt + $9.00/M completion'
      overpromise_risk: no
    - id: z-ai/glm-5.2
      context: 'headline 1,048,576 but PER-PROVIDER: 96,890 (AkashML), 101,376 (Ambient), 256,000 (BaseTen), 262,144 (DigitalOcean, Cloudflare, WandB, Parasail, Together, Io Net), 1,000,000 (Venice), 1,024,000 (StreamLake), 1,048,576 (Z.AI, DeepInfra, Novita, Fireworks, others)'
      max_output: 'top_provider 131,072; per-provider ranges 32,768 (DeepInfra) / 65,535 (Chutes) / 65,536 (Io Net) up to 1,048,576 (Inceptron, Morph, Decart, Friendli); several unpublished (null)'
      long_context_pricing: 'flat as listed (re-verified live 2026-07-19 via /api/v1/model/z-ai/glm-5.2): $0.9786/M prompt, $3.0756/M completion, cache read $0.18174/M; no input_cache_write listed; no pricing.overrides (per-provider prices vary on /endpoints). [CORRECTED — prior contract figures $1.12/$3.52/$0.208 were ~1.144x too high, a transcription/wrong-provider error.]'
      overpromise_risk: 'yes: any account whose routing lands on (or is restricted by provider prefs/ZDR to) AkashML/Ambient/BaseTen/262k-class providers gets 96k-262k, not the 1,048,576 headline; hardcoding 1M context WILL over-promise'
    - id: deepseek/deepseek-v4-pro
      context: 'headline 1,048,576 but PER-PROVIDER: 262,144 (DigitalOcean, BaseTen), 512,000 (Together), 1,000,000 (Alibaba, Venice), 1,024,000 (StreamLake), 1,048,576 (DeepSeek first-party, Baidu, Novita, SiliconFlow, Fireworks, others)'
      max_output: 'top_provider 384,000; per-provider ranges 16,384 (DeepInfra!) / 32,768 (Venice) / 262,144 (BaseTen) / 384,000-393,216 (most) up to 1,048,576 (Parasail, WandB); several unpublished'
      long_context_pricing: 'flat as listed: $0.435/M prompt, $0.87/M completion, cache read ~$0.0036/M; no overrides'
      overpromise_risk: 'yes: same slug can serve 262k context or 16,384 max output depending on routed provider; ZDR/data_collection:deny accounts see a reduced provider set'
  token_counting_endpoint: none
probes_needed:
  - question: 'Exact gateway behavior when reasoning:{effort} is sent to a model whose endpoint supported_parameters lacks "reasoning" under default routing — silently stripped by OpenRouter, forwarded-and-ignored by the provider, or 4xx? (Docs only cover the generic unknown-parameter case.)'
    probe: 'curl -s https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" -d ''{"model":"<any model whose /endpoints entries lack \"reasoning\" in supported_parameters>","messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"high"}}'''
  - question: 'Out-of-range effort on a restricted-effort model: does reasoning:{effort:"medium"} to z-ai/glm-5.2 (supported_efforts [xhigh,high] only) get coerced to a supported level, silently ignored, or return 400?'
    probe: 'curl -s https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" -d ''{"model":"z-ai/glm-5.2","messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"medium"}}'''
  - question: 'Does reasoning:{effort:"none"} on anthropic/claude-sonnet-5 (no "none" in supported_efforts) disable thinking, get ignored, or 400? (Needed to pick the correct disable path: omit vs enabled:false vs effort:none.)'
    probe: 'curl -s https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" -d ''{"model":"anthropic/claude-sonnet-5","messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"none"}}'''
  - question: 'Sending BOTH reasoning.effort and reasoning.max_tokens (docs say mutually exclusive — "One of the following (not both)") — 400 or silent precedence, and which wins?'
    probe: 'curl -s https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" -d ''{"model":"anthropic/claude-opus-4.8","messages":[{"role":"user","content":"hi"}],"max_tokens":8000,"reasoning":{"effort":"high","max_tokens":2000}}'''
  - question: 'reasoning.mode:"pro" reroute observability on gpt-5.6-terra/luna: does the response "model" field (and billing) show the *-pro sibling, so Sylliptor can detect the substitution?'
    probe: 'curl -s https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer $OPENROUTER_API_KEY" -H "Content-Type: application/json" -d ''{"model":"openai/gpt-5.6-luna","messages":[{"role":"user","content":"hi"}],"reasoning":{"mode":"pro"},"usage":{"include":true}}'''
  - question: 'Shape of the account-scoped listing (/models/user): does it return the same Model objects (incl. reasoning metadata and top_provider) filtered by this account''s provider prefs/ZDR, and does context_length shrink when preferred providers are excluded?'
    probe: 'curl -s https://openrouter.ai/api/v1/models/user -H "Authorization: Bearer $OPENROUTER_API_KEY"'
conflicts:
  - 'https://openrouter.ai/docs/api-reference/overview parameter table does not list "reasoning"/"include_reasoning" at all, while https://openrouter.ai/docs/use-cases/reasoning-tokens fully specifies them and the live /api/v1/models supported_parameters includes both — intra-doc drift, recorded.'
  - 'https://openrouter.ai/docs/use-cases/reasoning-tokens says reasoning tokens are "included in the response by default if the model decides to output them", but live /api/v1/models metadata sets default_enabled:true only on some reasoning-capable models (gpt-5.6-terra/luna, glm-5.2) and omits it on others (claude-sonnet-5/opus-4.8, deepseek-v4-pro) — default-on vs opt-in ambiguity recorded, not resolved.'
  - 'Reasoning-tokens doc says Gemini "maps xhigh down" to its supported levels (coercion) but documents no coercion rule for other restricted-effort families (glm-5.2/deepseek-v4-pro list only [xhigh,high]) — per-family coercion policy inconsistent/unspecified, recorded.'
  - 'Starting URL https://openrouter.ai/docs/api-reference/list-available-models (from the prior refresh) is now 404; the models API is documented at https://openrouter.ai/docs/guides/overview/models and /docs/api/api-reference/models/* — prior-refresh URL stale, recorded.'
notes: 'CORRECTION PASS 2026-07-19. Two challenges re-verified against live sources and both upheld: (1) z-ai/glm-5.2 pricing — live GET /api/v1/model/z-ai/glm-5.2 returns pricing.prompt "0.0000009786" ($0.9786/M), pricing.completion "0.0000030756" ($3.0756/M), pricing.input_cache_read "0.00000018174" ($0.18174/M), NO input_cache_write; the prior contract''s $1.12/$3.52/$0.208 was a ~1.144x transcription/wrong-provider error and has been replaced with the live listed values. deepseek-v4-pro and gpt-5.6 override pricing were spot-re-checked and remain correct. (2) The default-behavior sentence was mis-quoted; corrected to the verbatim doc text "Reasoning tokens are included in the response by default if the model decides to output them." Verifier was RIGHT on both. Separately re-confirmed the verifier''s aside was WRONG to imply effort/max_tokens are not mutually exclusive: the doc reads "One of the following (not both): effort or max_tokens", so the original mutual-exclusivity claim stands unchanged. The live /api/v1/models reasoning metadata object ({mandatory, default_enabled, supported_efforts, default_effort}, present on 207 of 338 models) is exactly the shape Sylliptor''s reasoning_mode field needs — drive it from live, not a static table; likewise drive PRICING from the live per-token decimal strings rather than transcribing. Errors: HTTP status matches error.code; otherwise 200 OK with the error embedded in the body or SSE; mid-stream errors terminate the stream with finish_reason:"error". 503 = "There is no available model provider that meets your routing requirements" (the require_parameters:true dead-end). provider.require_parameters:true is the safe way to guarantee the reasoning param is honored, at the cost of shrinking the provider pool (and possibly 503). The default OpenRouter list returned 338 models with links.next:null (no pagination needed below limit=500). Chat-completions probes above are cheap but billed — they need a funded key (402 = insufficient credits).'
````

---

## 17. perplexity

*Verification: verified-clean — 1 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: perplexity
endpoints:
  - url: https://api.perplexity.ai/v1/agent
    protocol: native
  - url: https://api.perplexity.ai/v1/responses
    protocol: openai-responses
  - url: https://api.perplexity.ai/v1/sonar
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://docs.perplexity.ai/docs/agent-api/models.md
  - https://docs.perplexity.ai/docs/agent-api/openai-compatibility.md
  - https://docs.perplexity.ai/llms.txt
  - https://docs.perplexity.ai/api-reference/models-get.md
  - https://docs.perplexity.ai/docs/agent-api/presets.md
  - https://docs.perplexity.ai/api-reference/agent-post.md
  - https://docs.perplexity.ai/docs/agent-api/model-fallback.md
  - https://docs.perplexity.ai/docs/sonar/openai-compatibility.md
  - https://docs.perplexity.ai/docs/agent-api/output-control.md
  - https://docs.perplexity.ai/docs/sonar/pro-search/tools.md
  - https://docs.perplexity.ai/docs/admin/rate-limits-usage-tiers.md
  - https://docs.perplexity.ai/docs/sdk/error-handling.md
  - https://docs.perplexity.ai/docs/resources/faq.md
  - https://api.perplexity.ai/v1/models
reasoning:
  param: "reasoning" (object, top-level, containing field "effort") on POST /v1/agent and its alias /v1/responses; also "preset" (top-level string) which bundles a reasoning effort level internally; no reasoning param documented on the Sonar surface (/v1/sonar, /chat/completions alias)
  values: reasoning.effort accepts exactly "minimal", "low", "medium", "high", "xhigh", "max" (api-reference/agent-post.md); documented request-wide, NOT split per model family — per-family acceptance is unpublished. xai additionally bakes reasoning into ids (xai/grok-4.20-reasoning vs xai/grok-4.20-non-reasoning, confirmed in live GET /v1/models), but no such -reasoning id variants exist for any of the six catalog models
  default_when_omitted: unpublished. When "preset" is used it supplies the effort level internally; when neither preset nor reasoning is sent, docs do not state the effective effort
  unsupported_value_behavior: unknown — SDK docs define a ValidationError class and generic 400 error body {"error": {"message", "type", "code"}}, but no doc states whether an invalid effort value (or effort sent to a non-reasoning model) is 400, silently-ignored, or silently-coerced; see probes_needed
  per_model:
    - id: anthropic/claude-sonnet-5
      mode: unknown (request schema accepts reasoning.effort for any model; per-model behavior unpublished — probe)
      effort: adjustable per request schema: [minimal, low, medium, high, xhigh, max] (acceptance on this id unverified)
    - id: anthropic/claude-opus-4-8
      mode: unknown (same — unpublished; probe)
      effort: adjustable per request schema: [minimal, low, medium, high, xhigh, max] (acceptance unverified)
    - id: openai/gpt-5.6-terra
      mode: unknown (same — unpublished; probe)
      effort: adjustable per request schema: [minimal, low, medium, high, xhigh, max] (acceptance unverified)
    - id: perplexity/kimi-k2.7-code
      mode: unknown (same — unpublished; probe)
      effort: adjustable per request schema: [minimal, low, medium, high, xhigh, max] (acceptance unverified)
    - id: google/gemini-3.1-flash-lite
      mode: unknown (same — unpublished; "flash-lite" naming suggests none, but that is inference, not doc)
      effort: adjustable per request schema: [minimal, low, medium, high, xhigh, max] (acceptance unverified)
    - id: nvidia/nemotron-3-super-120b-a12b
      mode: unknown (same — unpublished; probe)
      effort: adjustable per request schema: [minimal, low, medium, high, xhigh, max] (acceptance unverified)
  silent_substitutions: none triggered by reasoning flags specifically. Two adjacent mechanisms cause model drift: (1) "models" array (max 5) activates automatic mandatory fallback — "tries each model in order until one succeeds"; substitution IS visible in response "model" field, billed at the serving model; (2) "preset" pins models internally (e.g. preset "low" uses google/gemini-3-flash-preview) and "calling a preset by name" opts into auto-updating configuration — the served model can change over time without a client change
  tools_and_streaming: reasoning surfaces in SSE as agentic research events, not raw thinking-token blocks — events "response.reasoning.started", ".search_queries", ".search_results", ".fetch_url_queries", ".fetch_url_results", ".stopped", alongside "response.output_text.delta"/".done" and "response.output_item.added"/".done" (agent-post.md). Interleaving with tool activity is inherent (reasoning events wrap search/fetch tool phases). Whether reasoning tokens count against max_output_tokens is explicitly NOT stated in the docs — unknown; see probes_needed
  compat_passthrough: /v1/responses alias — passes: docs state "both endpoints are treated identically" to /v1/agent, so the "reasoning" object goes through; note the shape coincides with the OpenAI Responses SDK's reasoning={"effort": ...}. Flat chat-completions spelling "reasoning_effort" — unknown on every surface (never documented; probe). Sonar /chat/completions alias — reasoning params not documented at all: unknown (supported OpenAI params listed are only model, messages, max_tokens, stream, temperature, top_p, response_format)
discovery:
  endpoint: GET https://api.perplexity.ai/v1/models
  auth: none (docs: "security: []"; confirmed live this session without a key)
  metadata: ids-only — each object has exactly id, object ("model"), created (always 0), owned_by; no context, no capabilities, no pricing (confirmed live: 31 models, all 6 catalog ids present character-for-character)
  account_scoped: no (endpoint is unauthenticated, so the roster cannot vary per account); model ACCESS after auth could still vary by usage tier but this is not documented
  rate_limits_or_caching: undocumented for /v1/models specifically; Agent API POST limits are tier-scoped (Tier 0: 1 QPS / 50 RPM up to Tier 4-5: 33 QPS / 2,000 RPM, leaky-bucket, 429 on excess); no caching headers documented
  verdict: augment-static — drive roster presence/absence (and new-id detection) from live since the call is free and unauthenticated, but ALL capability/context/reasoning metadata must stay in Sylliptor's static layer because the endpoint publishes none
  refresh_strategy: on-config-open (plus a daily cache for startup roster validation); unauthenticated GET is safe to call eagerly
context:
  per_model:
    - id: anthropic/claude-sonnet-5
      context: unpublished — Perplexity publishes NO context windows for Agent API models; FAQ punts to "the linked provider documentation". Do not borrow Anthropic-native numbers (200k/1M-beta): the proxy's effective limit is unverified
      max_output_tokens: unpublished (but max_output_tokens is a REQUIRED request field for anthropic/* — HTTP 400 if omitted)
      long_context_pricing: flat ($2/1M in, $10/1M out, $0.20/1M cache)
      overpromise_risk: "yes: any hardcoded number is a guess; no account tier is documented to get the vendor-native headline window"
    - id: anthropic/claude-opus-4-8
      context: unpublished (same)
      max_output_tokens: unpublished (required request field, 400 if omitted)
      long_context_pricing: flat ($5/1M in, $25/1M out, $0.50/1M cache)
      overpromise_risk: "yes: same — unverified proxy ceiling"
    - id: openai/gpt-5.6-terra
      context: unpublished; pricing tierThreshold of 272000 input tokens implies the model accepts >272k input, but this is a pricing boundary, NOT a context window
      max_output_tokens: unpublished
      long_context_pricing: "boundary at 272,000 input tokens: input $2.50/1M below, $5/1M above; output $15/1M below, $22.50/1M above; 90% cache discount"
      overpromise_risk: "yes: context unknown above the 272k floor; hardcoding OpenAI-native numbers is unverified here"
    - id: perplexity/kimi-k2.7-code
      context: unpublished
      max_output_tokens: unpublished
      long_context_pricing: flat ($0.95/1M in, $4.00/1M out)
      overpromise_risk: "yes: do not borrow Moonshot-native 256k — nothing published for the Perplexity-hosted id"
    - id: google/gemini-3.1-flash-lite
      context: unpublished
      max_output_tokens: unpublished
      long_context_pricing: flat ($0.25/1M in, $1.50/1M out) — the Google 200,000-token tier boundary applies to tiered Gemini models (e.g. gemini-3.1-pro-preview), not to flash-lite's flat rate
      overpromise_risk: "yes: unpublished; Google-native numbers unverified on this proxy"
    - id: nvidia/nemotron-3-super-120b-a12b
      context: unpublished
      max_output_tokens: unpublished
      long_context_pricing: flat ($0.25/1M in, $2.50/1M out)
      overpromise_risk: "yes: unpublished"
  token_counting_endpoint: none
probes_needed:
  - question: Does reasoning.effort error, get ignored, or get coerced on a model with no known reasoning mode (per-model applicability is unpublished)?
    probe: "curl -s -X POST https://api.perplexity.ai/v1/agent -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"google/gemini-3.1-flash-lite\",\"input\":\"ping\",\"reasoning\":{\"effort\":\"high\"}}'"
  - question: Exact status code and error body for an invalid reasoning.effort value string (silently-ignored vs 400 ValidationError)?
    probe: "curl -s -X POST https://api.perplexity.ai/v1/agent -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"anthropic/claude-sonnet-5\",\"max_output_tokens\":256,\"input\":\"ping\",\"reasoning\":{\"effort\":\"turbo\"}}'"
  - question: Does the flat chat-completions spelling reasoning_effort pass, error, or get silently dropped on /v1/responses?
    probe: "curl -s -X POST https://api.perplexity.ai/v1/responses -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"openai/gpt-5.6-terra\",\"input\":\"ping\",\"reasoning_effort\":\"high\"}'"
  - question: Does the Sonar surface 400 on a tools array (prior-refresh claim; docs say custom tools cannot be registered but give no status code)?
    probe: "curl -s -X POST https://api.perplexity.ai/v1/sonar -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"sonar-pro\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"noop\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}'"
  - question: Do reasoning tokens count against max_output_tokens (docs explicitly silent) — does high effort + tiny cap yield status "incomplete" with near-zero visible text?
    probe: "curl -s -X POST https://api.perplexity.ai/v1/agent -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"anthropic/claude-opus-4-8\",\"max_output_tokens\":64,\"input\":\"Prove there are infinitely many primes.\",\"reasoning\":{\"effort\":\"high\"}}'"
  - question: When both preset and model are sent, which wins (docs only state models[] > model; preset precedence undocumented) — is the explicit model silently replaced by the preset's pinned model?
    probe: "curl -s -X POST https://api.perplexity.ai/v1/agent -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"anthropic/claude-sonnet-5\",\"preset\":\"low\",\"max_output_tokens\":256,\"input\":\"ping\"}' # inspect response .model"
  - question: What is the effective context ceiling per model (nothing published anywhere) — binary-search input size until 400/413 per catalog id?
    probe: "python -c \"import json;print(json.dumps({'model':'openai/gpt-5.6-terra','input':'x '*300000,'max_output_tokens':16}))\" > /tmp/big.json && curl -s -X POST https://api.perplexity.ai/v1/agent -H 'Authorization: Bearer $PPLX_API_KEY' -H 'Content-Type: application/json' -d @/tmp/big.json"
conflicts:
  - https://docs.perplexity.ai/docs/agent-api/presets.md documents six preset values including "wide-research"; https://docs.perplexity.ai/api-reference/agent-post.md lists only five ("fast", "low", "medium", "high", "xhigh") with no "wide-research" — recorded, not resolved
  - https://docs.perplexity.ai/docs/agent-api/models.md publishes tierThreshold values (272000 OpenAI / 200000 Google) that look like context windows but are pricing boundaries, while https://docs.perplexity.ai/docs/resources/faq.md says to consult "the linked provider documentation" for model specs — Perplexity itself publishes no window; do not conflate the two numbers
notes: The Agent API is a research-agent surface, not a raw chat proxy — reasoning manifests as search/fetch phases in SSE, and Perplexity's own web_search/fetch tools plus caller function tools coexist on /v1/agent, while the Sonar legacy surface allows no custom tools at all ("custom tools cannot be registered"). Billing usage block includes a per-request cost object (usage.cost with input_cost etc.) — useful for Sylliptor's accounting since context/pricing metadata is otherwise absent. models[] fallback is billed at the serving model. GET /v1/models being unauthenticated makes roster drift detection free. Prior refresh's "no reasoning control on the Agent API" is WRONG — the reasoning.effort object exists; only its per-model semantics are unpublished.
````

### Gap-fill addendum (post-verification)

Both halves of the gap are now resolved as far as public docs allow; one authenticated probe remains for the reasoning half.

A4 (reasoning mode per id): Perplexity itself publishes nothing per-model. Re-verified today:
- https://docs.perplexity.ai/docs/agent-api/models.md — publishes NO context windows and NO per-model reasoning info; only pricing + one warning: "Requests that use an `anthropic/*` model must include `max_output_tokens`. If omitted, the API returns HTTP 400 with `validation failed: max_output_tokens is required when using Anthropic models`."
- https://docs.perplexity.ai/api-reference/agent-post.md — reasoning.effort enum is exactly [minimal, low, medium, high, xhigh, max]; effort field description "How much effort the model should spend on reasoning"; silent on invalid-value behavior and on non-reasoning models.
- https://docs.perplexity.ai/docs/agent-api/openai-compatibility.md — does NOT mention `reasoning_effort` (flat) nor the nested reasoning object at all; so the flat chat-completions spelling stays unverified.
CRUCIAL new lever: the models.md page DOES hyperlink each model to its vendor card, and the FAQ (https://docs.perplexity.ai/docs/resources/faq.md) says verbatim: for third-party models "see the [Agent API models page](/docs/agent-api/models) and the linked provider documentation." So Perplexity explicitly delegates capability/context authority to the vendor cards it links. I opened those cards and read the underlying reasoning nature:
- anthropic/claude-sonnet-5 & anthropic/claude-opus-4-8 (https://platform.claude.com/docs/en/about-claude/models/overview): both list Adaptive thinking = Yes; a Note states "On Claude Opus 4.8, the `effort` parameter defaults to `high` ... On Claude Sonnet 5, it defaults to `high` on the Claude API and Claude Code." => reasoning models, vendor-confirmed.
- openai/gpt-5.6-terra (https://developers.openai.com/api/docs/models/gpt-5.6-terra): "Reasoning token support." => reasoning model.
- perplexity/kimi-k2.7-code (https://openrouter.ai/moonshotai/kimi-k2.7-code): "it always operates in a thinking mode, preserving full reasoning content across multi-turn conversations." => always-on reasoning (an effort knob may be a no-op).
- google/gemini-3.1-flash-lite (https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/): card does NOT confirm a dedicated thinking mode for this variant (Gemini 3 series are reasoning models generally) => genuinely unknown.
- nvidia/nemotron-3-super-120b-a12b (https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/): "agentic reasoning" but no documented reasoning toggle on the card => unknown.
IMPORTANT caveat kept: knowing the underlying model reasons natively does NOT prove Perplexity's proxy forwards reasoning.effort for that id (Perplexity's 6-level scale is its own abstraction). Per-id acceptance on /v1/agent is still only settleable by an authenticated probe (I cannot run a billed authenticated POST here).

C (context windows): sourced from the exact vendor cards Perplexity links. Verified numbers:
- claude-sonnet-5: context 1M tokens, max output 128k (Anthropic overview table).
- claude-opus-4-8: context 1M tokens, max output 128k (same table).
- gpt-5.6-terra: "1,050,000 context window" and "128,000 max output tokens" (OpenAI dev card).
- kimi-k2.7-code: "262K-token context window" (262,144); max output not published.
- gemini-3.1-flash-lite: input context "up to 1M" tokens, "64K token output" (DeepMind card).
- nemotron-3-super-120b-a12b: "native 1M-token context window"; max output not published (NVIDIA blog/card).
Overpromise note: these are vendor-native ceilings; Perplexity restates none and does not separately guarantee the proxy preserves the full window — but its own docs name these cards as the authority, so this is the best documented source rather than a guess.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  # openai-compatibility.md re-read today: no `reasoning_effort` (flat) and no nested
  # reasoning object documented on the OpenAI-compat surface -> flat spelling stays a probe.
  per_model:
    - id: anthropic/claude-sonnet-5
      mode: reasoning
      mode_source: "vendor card (Adaptive thinking = Yes; effort defaults to high) — https://platform.claude.com/docs/en/about-claude/models/overview; Perplexity proxy acceptance of reasoning.effort on this id still unverified (probe)"
      effort: adjustable per request schema [minimal, low, medium, high, xhigh, max]; vendor-native effort default is high (proxy default via /v1/agent unpublished)
    - id: anthropic/claude-opus-4-8
      mode: reasoning
      mode_source: "vendor card (Adaptive thinking = Yes; effort defaults to high) — https://platform.claude.com/docs/en/about-claude/models/overview; Perplexity proxy acceptance unverified (probe)"
      effort: adjustable per request schema [minimal, low, medium, high, xhigh, max]; vendor-native effort default is high (proxy default unpublished)
    - id: openai/gpt-5.6-terra
      mode: reasoning
      mode_source: "vendor card ('Reasoning token support') — https://developers.openai.com/api/docs/models/gpt-5.6-terra; Perplexity proxy acceptance unverified (probe)"
      effort: adjustable per request schema [minimal, low, medium, high, xhigh, max] (proxy acceptance unverified)
    - id: perplexity/kimi-k2.7-code
      mode: reasoning-always-on
      mode_source: "vendor doc ('always operates in a thinking mode') — https://openrouter.ai/moonshotai/kimi-k2.7-code; because thinking is always on, reasoning.effort may be a no-op on this id — probe"
      effort: adjustable per request schema [minimal, low, medium, high, xhigh, max] (effect on an always-thinking model unverified)
    - id: google/gemini-3.1-flash-lite
      mode: unknown
      mode_source: "vendor card does NOT confirm a dedicated thinking mode for this variant — https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/ (Gemini 3 series are reasoning models generally, but flash-lite is not confirmed); probe"
      effort: adjustable per request schema [minimal, low, medium, high, xhigh, max] (acceptance unverified)
    - id: nvidia/nemotron-3-super-120b-a12b
      mode: unknown
      mode_source: "vendor blog describes 'agentic reasoning' but documents no reasoning toggle — https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/; probe"
      effort: adjustable per request schema [minimal, low, medium, high, xhigh, max] (acceptance unverified)
context:
  # Perplexity publishes no windows; models.md hyperlinks each id to its vendor card and the
  # FAQ says to use "the linked provider documentation". Numbers below are read from those
  # exact linked cards. They are vendor-native ceilings; Perplexity does not separately
  # confirm the proxy preserves the full window.
  per_model:
    - id: anthropic/claude-sonnet-5
      context: 1000000  # "1M tokens" — Anthropic overview table (the card Perplexity links)
      max_output_tokens: 128000  # "128k tokens" (sync Messages API); required request field for anthropic/* (HTTP 400 if omitted)
      context_source: https://platform.claude.com/docs/en/about-claude/models/overview
      overpromise_risk: "low: vendor-native window via the card Perplexity links; proxy passthrough of full 1M not separately restated by Perplexity"
    - id: anthropic/claude-opus-4-8
      context: 1000000  # "1M tokens" — Anthropic overview table
      max_output_tokens: 128000  # "128k tokens"; required request field for anthropic/* (HTTP 400 if omitted)
      context_source: https://platform.claude.com/docs/en/about-claude/models/overview
      overpromise_risk: "low: vendor-native window via linked card; proxy passthrough not separately restated"
    - id: openai/gpt-5.6-terra
      context: 1050000  # "1,050,000 context window" — OpenAI dev card
      max_output_tokens: 128000  # "128,000 max output tokens" — OpenAI dev card
      context_source: https://developers.openai.com/api/docs/models/gpt-5.6-terra
      overpromise_risk: "low: exceeds the 272k pricing boundary; window is the vendor number, proxy ceiling not separately restated"
    - id: perplexity/kimi-k2.7-code
      context: 262144  # "262K-token context window" (256K) — Moonshot/OpenRouter card
      max_output_tokens: unpublished
      context_source: https://openrouter.ai/moonshotai/kimi-k2.7-code
      overpromise_risk: "low-medium: 256K is the vendor window; max_output unpublished; proxy ceiling not restated by Perplexity"
    - id: google/gemini-3.1-flash-lite
      context: 1000000  # "up to 1M" input tokens — Google DeepMind card
      max_output_tokens: 65536  # "64K token output" — DeepMind card
      context_source: https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/
      overpromise_risk: "low: vendor-native window; proxy passthrough not separately restated"
    - id: nvidia/nemotron-3-super-120b-a12b
      context: 1000000  # "native 1M-token context window" — NVIDIA blog/card
      max_output_tokens: unpublished
      context_source: https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/
      overpromise_risk: "low-medium: 1M is the vendor window; max_output unpublished; proxy ceiling not restated"
sources:
  # add the vendor cards Perplexity's own models.md links (context + reasoning authority):
  - https://platform.claude.com/docs/en/about-claude/models/overview
  - https://developers.openai.com/api/docs/models/gpt-5.6-terra
  - https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/
  - https://openrouter.ai/moonshotai/kimi-k2.7-code
  - https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/
````

**Still unresolved (needs a live key):**

````
Reasoning half only: whether Perplexity's proxy actually accepts/forwards reasoning.effort per id (vs silently ignoring/coercing) is not documented — it needs an authenticated, billed POST. Context half is settled from the vendor cards Perplexity links. Probe (replace $PPLX_API_KEY; repeat the model value for each of the six ids):

# 1) valid effort — does the id accept reasoning.effort and echo a served model?
curl -sS https://api.perplexity.ai/v1/agent \
  -H "Authorization: Bearer $PPLX_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"anthropic/claude-sonnet-5","max_output_tokens":64,"reasoning":{"effort":"high"},"input":"Reply with the single word: ok"}'

# 2) invalid effort — 400 vs silent-coerce?
curl -sS https://api.perplexity.ai/v1/agent \
  -H "Authorization: Bearer $PPLX_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"google/gemini-3.1-flash-lite","max_output_tokens":64,"reasoning":{"effort":"ultra"},"input":"ok"}'

# 3) reasoning sent to a possibly-non-reasoning id — 400, ignored, or honored?
curl -sS https://api.perplexity.ai/v1/agent \
  -H "Authorization: Bearer $PPLX_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-3-super-120b-a12b","max_output_tokens":64,"reasoning":{"effort":"minimal"},"input":"ok"}'

# 4) flat OpenAI-style spelling on /v1/responses — accepted at all?
curl -sS https://api.perplexity.ai/v1/responses \
  -H "Authorization: Bearer $PPLX_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-5.6-terra","max_output_tokens":64,"reasoning_effort":"high","input":"ok"}'
````

### Verifier notes (minor, not applied)

- **compat_passthrough: /v1/responses alias — passes: docs state "both endpoints are treated identically" to /v1/agent** — The phrase "both endpoints are treated identically" is presented in quotation marks as a doc quote, but openai-compatibility.md does not contain that string. The actual wording is that /v1/responses is "accepted as an alias" and that the OpenAI SDK route to it is "handled seamlessly." The substance (identical handling, reasoning object passes through) is accurate and independently confirmed, so th *Suggested: Drop the quotation marks or replace with the actual doc language: /v1/responses is documented as an alias for /v1/agent and handled seamlessly, so the reasoning object passes through.*

---

## 18. together

*Verification: verified-clean — adversarial pass found no challenges.*

````yaml
provider: together
endpoints:
  - url: https://api.together.xyz/v1
    protocol: openai-chat
checked: 2026-07-19
sources:
  - https://docs.together.ai/docs/serverless-models
  - https://docs.together.ai/reference/chat-completions-1
  - https://docs.together.ai/docs/function-calling
  - https://docs.together.ai/docs/reasoning-overview
  - https://docs.together.ai/reference/models-1
  - https://docs.together.ai/docs/openai-api-compatibility
  - https://docs.together.ai/docs/gpt-oss
  - https://docs.together.ai/docs/glm-5.2-quickstart
  - https://docs.together.ai/docs/deepseek-v4-quickstart
  - https://docs.together.ai/docs/minimax-m3-quickstart (FAILED)   # 404; M3 toggle param spelling could not be confirmed from a quickstart
  - https://www.together.ai/models/glm-52
  - https://www.together.ai/models/glm-5-2 (FAILED)   # 404; correct slug is /models/glm-52
  - https://www.together.ai/models/kimi-k27-code
  - https://www.together.ai/models/minimax-m3
  - https://www.together.ai/models/gpt-oss-120b
  - https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets   # infra only, no API params
reasoning:
  param: >
    Three Together-native spellings, all top-level unless noted:
    (1) "reasoning" — object, {"enabled": true|false} (hybrid models);
    (2) "reasoning_effort" — string enum (adjustable-effort models);
    (3) "chat_template_kwargs" — object escape hatch, documented keys "thinking": true|false, "enable_thinking": true, and "clear_thinking": false (GLM-5.2 preserved thinking).
  values: >
    reasoning.enabled: true | false.
    reasoning_effort global enum in the chat-completions reference: "low" | "medium" | "high".
    Per family: gpt-oss-120b/20b accept "low" | "medium" | "high";
    deepseek-ai/DeepSeek-V4-Pro accepts "high" | "max" ONLY, and Together documents automatic normalization: "low" and "medium" map to "high", "xhigh" maps to "max";
    zai-org/GLM-5.2 accepts "high" | "max".
    Note "max" is accepted on DeepSeek-V4-Pro and GLM-5.2 despite NOT being in the reference enum — the reference enum lags the model quickstarts.
  default_when_omitted: >
    Hybrid models (GLM-5.2, DeepSeek-V4-Pro, and per Together's M3 page MiniMax-M3): reasoning ON by default.
    GLM-5.2 default reasoning_effort is "max"; DeepSeek-V4-Pro default is "high".
    gpt-oss-*: reasoning always runs; default effort "medium" (documented as "recommended default").
    Kimi-K2.7-Code: thinking always on ("Thinking and preserve_thinking are always enabled — thinking mode cannot be disabled").
  unsupported_value_behavior: >
    Partially documented. DeepSeek-V4-Pro out-of-range efforts are silently-coerced (low/medium→high, xhigh→max) — documented.
    For truly unknown params/values Together docs do NOT state 400-vs-drop; compat page lists a specific allowlist of "accepted but ignored" OpenAI params (service_tier, store, metadata, prediction) which implies other unknowns are NOT guaranteed ignored.
    Error shape when a 400 does occur: OpenAI-shaped {"error": {"message", "type", "code"}} with Together-specific type/code values; docs advise matching on HTTP status. unknown for the specific reasoning-param-to-wrong-model case — see probes_needed.
  per_model:
    - id: zai-org/GLM-5.2
      mode: optional   # hybrid, on by default; reasoning={"enabled": false} disables
      effort: 'adjustable: ["high", "max"] (default "max"); also chat_template_kwargs {"clear_thinking": false} to preserve reasoning across turns'
    - id: moonshotai/Kimi-K2.7-Code
      mode: always-on   # "thinking mode cannot be disabled"; preserve_thinking also always on
      effort: n/a   # no reasoning_effort or reasoning toggle documented for this model
    - id: deepseek-ai/DeepSeek-V4-Pro
      mode: optional   # hybrid, on by default; reasoning={"enabled": false} disables
      effort: 'adjustable: ["high", "max"] (default "high"); other values silently coerced (low/medium→high, xhigh→max)'
    - id: MiniMaxAI/MiniMax-M3
      mode: optional   # Together model page: "Toggleable at request time"; exact param spelling for M3 not shown in any doc I could open — presumed reasoning={"enabled": ...} per hybrid convention; PROBE
      effort: n/a   # no effort control documented
    - id: openai/gpt-oss-120b
      mode: always-on   # adjustable-effort class; no documented off switch
      effort: 'adjustable: ["low", "medium", "high"] (default "medium")'
    - id: openai/gpt-oss-20b
      mode: always-on
      effort: 'adjustable: ["low", "medium", "high"] (default "medium")'
  silent_substitutions: none documented — no model swaps triggered by reasoning flags anywhere in Together docs
  tools_and_streaming: >
    Streaming: reasoning arrives in ChatCompletionChunk delta.reasoning (string, nullable), separate from delta.content; non-streamed message has a top-level "reasoning" field. Exception noted in docs: legacy DeepSeek-R1 embeds <think> tags inside content instead.
    Token accounting: reasoning is billed as completion tokens, reported under usage.completion_tokens_details.reasoning_tokens — i.e. it consumes the max_tokens budget (Together tells gpt-oss users to set max_tokens ~30,000 at high effort; DeepSeek-V4-Pro "max" mode wants max_tokens set generously with ≥384K context headroom).
    Tools: GLM-5.2 does interleaved thinking between tool calls automatically; tool-call arguments stream incrementally (concatenate arguments fragments from deltas). DeepSeek-V4-Pro multi-turn tool use requires passing back the assistant message's content, reasoning_content, AND tool_calls to preserve reasoning. Stale doc note still present: "For DeepSeek V3.1, function calling only works in non-reasoning mode" (V3.1-era; no such restriction documented for V4-Pro).
  compat_passthrough: >
    Single surface (OpenAI-compatible chat completions) — n/a for cross-surface stripping.
    Within the OpenAI compat story: reasoning_effort passes through and "works on GPT-OSS models" per the compat page; the "reasoning" object and chat_template_kwargs are Together-native extensions (an OpenAI SDK must send them via extra_body). OpenAI params service_tier, store, metadata, prediction are accepted-but-ignored; vision "detail" ignored. No Responses API or Anthropic-messages surface documented on Together.
discovery:
  endpoint: GET https://api.together.ai/v1/models   # docs' canonical host; the catalog's api.together.xyz alias serves the same API — see notes
  auth: bearer   # Authorization: Bearer $TOGETHER_API_KEY
  metadata: ids+context+pricing   # per object: id, object, created, type (chat|language|code|image|embedding|moderation|rerank), display_name, organization, link, license, context_length (optional int), pricing {base, hourly, input, output, cached_input (optional), finetune}. NO reasoning/capability flags, NO max_output.
  account_scoped: "no documented per-tier variance in the list itself; optional query param dedicated=true filters to dedicated-endpoint models (account-specific set)"
  rate_limits_or_caching: undocumented — reference only lists 429/504 response codes, no numeric limits or cache headers
  verdict: augment-static
  refresh_strategy: on-config-open   # context_length + pricing are worth pulling live to catch serving-window changes (e.g. M3 524288 vs 1M); reasoning capability map must stay static since /v1/models carries no reasoning metadata
context:
  per_model:
    - id: zai-org/GLM-5.2
      context: 262144 (serverless table "262,144"; quickstart "262K"; model page phrases it "usable 256K" — same number; no tier splits)
      max_output: 131072 (model page; quickstart says "up to 128K")
      long_context_pricing: flat ($1.40 in / $4.40 out per 1M; $0.26 cached input; no boundary pricing)
      overpromise_risk: "yes: Z.ai markets GLM-5.2 as a 1M-context model — Together serves 262,144; a catalog copying upstream 1M over-promises 4x for every Together account"
    - id: moonshotai/Kimi-K2.7-Code
      context: 262144 (serverless table "262,144"; model page "256K")
      max_output: unpublished
      long_context_pricing: flat ($0.95 in / $4.00 out per 1M; $0.19 cached input)
      overpromise_risk: no
    - id: deepseek-ai/DeepSeek-V4-Pro
      context: 512000 (serverless table "512,000"; model page "512K"; note it is 512,000 not 524,288 — do not round to a power of two)
      max_output: unpublished (quickstart shows max_tokens=384000 in "max"-mode examples and says allocate generously)
      long_context_pricing: flat ($1.74 in / $3.48 out per 1M; no boundary)
      overpromise_risk: no
    - id: MiniMaxAI/MiniMax-M3
      context: "CONFLICTED: serverless table says 524,288; Together's own model page says 1M; upstream MiniMax says 1M with 'guaranteed minimum of 512K'. Treat 524,288 as the safe served window until /v1/models context_length is probed"
      max_output: unpublished
      long_context_pricing: flat ($0.30 in / $1.20 out per 1M; $0.06 cached input; no long-context boundary documented on Together)
      overpromise_risk: "yes: hardcoding the marketed 1M over-promises ~2x if Together's served window is 524,288 (the serverless table's number)"
    - id: openai/gpt-oss-120b
      context: 128000 (serverless table "128,000"; docs consistently say 128K; registries claiming 131072 conflict — probe /v1/models for the exact int)
      max_output: unpublished (docs recommend max_tokens ~30,000 at high effort)
      long_context_pricing: flat ($0.15 in / $0.60 out per 1M)
      overpromise_risk: "yes-if-131072-is-hardcoded: Together's documented number is 128,000, 1,072 tokens less than the registry figure"
    - id: openai/gpt-oss-20b
      context: 128000 (same basis as 120b)
      max_output: unpublished
      long_context_pricing: flat ($0.05 in / $0.20 out per 1M)
      overpromise_risk: "yes-if-131072-is-hardcoded: same 128,000 vs 131,072 delta"
  token_counting_endpoint: none   # no /tokenize or count-tokens endpoint in the Together API reference
probes_needed:
  - question: Does Together 400 on a truly unknown top-level param or unknown reasoning_effort value, or silently drop it? (Decides whether always-on models are an error hazard or a no-op hazard.)
    probe: 'curl -s -w "\nHTTP %{http_code}\n" -X POST https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d ''{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "reasoning_effort": "banana", "frobnicate": true}'''
  - question: Does reasoning={"enabled": false} sent to always-on moonshotai/Kimi-K2.7-Code return an error, or get ignored (response still contains a reasoning field)?
    probe: 'curl -s -w "\nHTTP %{http_code}\n" -X POST https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d ''{"model": "moonshotai/Kimi-K2.7-Code", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32, "reasoning": {"enabled": false}}'''
  - question: Does reasoning_effort (any value) sent to moonshotai/Kimi-K2.7-Code error or no-op?
    probe: 'curl -s -w "\nHTTP %{http_code}\n" -X POST https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d ''{"model": "moonshotai/Kimi-K2.7-Code", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32, "reasoning_effort": "high"}'''
  - question: What is the exact param that toggles MiniMax-M3 thinking on Together (reasoning={"enabled": false}? chat_template_kwargs?), and does disabling actually remove the reasoning field from the response?
    probe: 'curl -s -w "\nHTTP %{http_code}\n" -X POST https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d ''{"model": "MiniMaxAI/MiniMax-M3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32, "reasoning": {"enabled": false}}'''
  - question: Exact context_length ints from live discovery — settles gpt-oss 128000 vs 131072 and MiniMax-M3 524288 vs 1000000, and whether cached_input pricing appears per model.
    probe: 'curl -s https://api.together.xyz/v1/models -H "Authorization: Bearer $TOGETHER_API_KEY" | python -c "import json,sys; ms={m[chr(39)+chr(39).join([])or chr(105)+chr(100)]: m for m in json.load(sys.stdin)}; print(ms)" # simpler: pipe to jq: jq ''[.[] | select(.id as $i | ["zai-org/GLM-5.2","moonshotai/Kimi-K2.7-Code","deepseek-ai/DeepSeek-V4-Pro","MiniMaxAI/MiniMax-M3","openai/gpt-oss-120b","openai/gpt-oss-20b"] | index($i)) | {id, context_length, pricing}]'''
  - question: Does gpt-oss on Together accept reasoning={"enabled": false} (i.e. can reasoning be fully disabled on the adjustable-effort class), or is that an error/no-op?
    probe: 'curl -s -w "\nHTTP %{http_code}\n" -X POST https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d ''{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32, "reasoning": {"enabled": false}}'''
conflicts:
  - "https://docs.together.ai/docs/serverless-models says MiniMaxAI/MiniMax-M3 context is 524,288; https://www.together.ai/models/minimax-m3 (Together's own model page) says 1M; upstream https://www.minimax.io/blog/minimax-m3 says 1M with guaranteed minimum 512K — recorded, not resolved"
  - "https://docs.together.ai/docs/serverless-models and https://docs.together.ai/docs/gpt-oss say gpt-oss context 128,000 / '128k'; prior-refresh registry data said 131072 — Together's published number is 128,000; exact served int needs the /v1/models probe"
  - "https://docs.together.ai/docs/reasoning-overview's supported-models table lists only the PREVIOUS generation (MiniMaxAI/MiniMax-M2.7 reasoning-only, zai-org/GLM-5, moonshotai/Kimi-K2.6 hybrid) and omits GLM-5.2, Kimi-K2.7-Code, MiniMax-M3 entirely, while the per-model quickstarts and model pages document them — the overview page lags the catalog; do not classify M3 as reasoning-only by analogy to M2.7"
  - "https://docs.together.ai/reference/chat-completions-1 declares reasoning_effort enum as low|medium|high only, but https://docs.together.ai/docs/deepseek-v4-quickstart and https://docs.together.ai/docs/glm-5.2-quickstart document 'max' as a valid (and for GLM-5.2 default) value — the reference schema lags the quickstarts"
  - "Z.ai marketing (via interconnects.ai and Z.ai's own copy) calls GLM-5.2 a 1M-context model; Together serves 262,144 — recorded, not resolved"
notes: >
  Base-URL: current docs uniformly show https://api.together.ai/v1 (compat page and API reference); the catalog's https://api.together.xyz/v1 is the long-standing alias and still what most SDK examples in the wild use — keep .xyz but know docs have moved to .ai.
  The three reasoning spellings are NOT interchangeable per model class: hybrid models take the "reasoning" object, adjustable-effort models take "reasoning_effort", and chat_template_kwargs is the raw escape hatch — a capability layer should store which spelling each model takes, not emit all three.
  DeepSeek-V4-Pro's documented effort normalization (low/medium→high, xhigh→max) means a generic CLI sending "low" to save money silently gets full "high" reasoning — a cost surprise, not an error.
  Kimi-K2.7-Code also always preserves thinking across turns (preserve_thinking) — clients must tolerate reasoning content on every turn of an agentic session; docs claim ~30% fewer thinking tokens than K2.6.
  Recommended sampling defaults differ per model and matter for agentic use: GLM-5.2 and Kimi-K2.7-Code temperature=1.0/top_p=0.95; DeepSeek-V4-Pro temperature=1.0/top_p=1.0.
  usage.completion_tokens_details.reasoning_tokens is the field for reasoning-token accounting; reasoning output consumes max_tokens, so tight max_tokens on a "max"-effort call can truncate before the answer starts.
````

### Gap-fill addendum (post-verification)

The M3 reasoning-toggle spelling remains UNCONFIRMED from Together's own docs — but I narrowed it and corrected a stale assumption in the YAML.

What the docs actually say (URLs opened):
1. https://www.together.ai/models/minimax-m3 — Confirms M3 has a toggleable thinking mode ("Toggle thinking mode at request time — no separate model version required"; "Toggleable at request time — enabled for complex reasoning and long-horizon agentic tasks, disabled for fast responses in latency-sensitive scenarios"). But the cURL/Python/TS request examples on this page show ONLY model + messages — NO reasoning, reasoning_effort, chat_template_kwargs, enable_thinking, or thinking param is demonstrated. The exact toggle field is not on the model page.
2. The old https://docs.together.ai/docs/minimax-m3-quickstart still 404s (confirmed again). There is NO M3-specific quickstart page; a site search of docs.together.ai returns only the generic /docs/quickstart, never an M3 page.
3. The reasoning doc has MOVED: the canonical URL is now https://docs.together.ai/docs/inference/chat/reasoning (the old /docs/reasoning-overview redirects there). Its reasoning-models table lists MiniMaxAI/MiniMax-M2.7 as type "reasoning_only" — "always produces reasoning tokens, cannot be toggled off" — and does NOT list MiniMax-M3 at all. So the YAML's per-model note deriving M3 from "hybrid convention" is an inference, not something the table states.
4. That same reasoning doc enumerates Together's three reasoning-control surfaces: (a) reasoning object {"enabled": true|false} for hybrid models; (b) reasoning_effort string (GPT-OSS); (c) chat_template_kwargs escape hatch with documented keys thinking, enable_thinking, clear_thinking, and now also medium_effort (Nemotron 3 Ultra). M3 is not tied to any of these by name.
5. Upstream MiniMax M3 docs (platform.minimax.io) diverge from the Together convention: M3's native toggle in its OpenAI-compat surface is NOT reasoning={"enabled":...}; native M3 uses reasoning_split (output-format flag only, not an on/off toggle) and, in its Anthropic-compat surface, thinking={"type":"adaptive"} / thinking={"type":"disabled"}. This means the underlying model's real switch is thinking-type/enable_thinking-style, so chat_template_kwargs={"enable_thinking": false} is at least as likely to be M3's actual Together toggle as reasoning={"enabled": false}.

Net: two plausible spellings (reasoning={"enabled": false} per Together's hybrid front-door, OR chat_template_kwargs={"enable_thinking": false} per the model's template) and zero Together documentation disambiguating them for M3. This must be settled by a live probe.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  per_model:
    - id: MiniMaxAI/MiniMax-M3
      mode: optional   # model page: toggleable thinking at request time. EXACT param UNCONFIRMED — Together's reasoning doc (docs/inference/chat/reasoning) lists only MiniMax-M2.7 (reasoning_only, cannot toggle) and does NOT list M3; the M3 model page shows no reasoning param in its examples; no M3 quickstart exists (still 404). Two candidate spellings, both Together-native, neither doc-confirmed for M3: reasoning={"enabled": false} (hybrid front-door) OR chat_template_kwargs={"enable_thinking": false} (model-template escape hatch). Upstream M3 uses enable_thinking / Anthropic thinking={"type":"disabled"} natively, so the chat_template_kwargs form is a live possibility. PROBE before wiring.
      effort: n/a   # no effort control documented
sources:
  # reasoning doc moved: old /docs/reasoning-overview -> canonical below
  - https://docs.together.ai/docs/inference/chat/reasoning   # reasoning-models table lists MiniMax-M2.7 (reasoning_only) but NOT MiniMax-M3
  - https://docs.together.ai/docs/minimax-m3-quickstart (FAILED)   # 404 confirmed 2026-07-19; no M3 quickstart exists anywhere on docs.together.ai
  - https://www.together.ai/models/minimax-m3   # confirms toggleable thinking; request examples show NO reasoning/thinking param
  - https://platform.minimax.io/docs/guides/text-m3-function-call   # upstream M3: native toggle is enable_thinking / Anthropic thinking={"type":"disabled"}; reasoning_split is output-format only, not an on/off switch
````

**Still unresolved (needs a live key):**

````
Run all three variants against the live endpoint and compare whether a `reasoning`/`reasoning_content` field (or <think> tags) appears in the response — the variant that SUPPRESSES reasoning is the correct off-switch:

# Variant A — Together hybrid front-door
curl -s https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMaxAI/MiniMax-M3","messages":[{"role":"user","content":"What is 17*24? Answer only."}],"reasoning":{"enabled":false},"max_tokens":64}'

# Variant B — model-template escape hatch
curl -s https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMaxAI/MiniMax-M3","messages":[{"role":"user","content":"What is 17*24? Answer only."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64}'

# Baseline — no toggle (confirm reasoning is ON by default)
curl -s https://api.together.xyz/v1/chat/completions -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" -d '{"model":"MiniMaxAI/MiniMax-M3","messages":[{"role":"user","content":"What is 17*24? Answer only."}],"max_tokens":256}'

The variant whose response drops usage.completion_tokens_details.reasoning_tokens to ~0 (and returns no top-level `reasoning` field) is M3's real off-switch. If BOTH A and B are silently ignored (reasoning still present), M3's reasoning is effectively always-on on Together despite the model-page marketing — record that instead. Also note whether the losing variant returns a 400 (rejected) vs. is silently dropped, to fill unsupported_value_behavior for this model.
````

---

## 19. fireworks

*Verification: verified-clean — 3 minor challenge(s) from the adversarial pass, listed below the block.*

````yaml
provider: fireworks
endpoints:
  - url: https://api.fireworks.ai/inference/v1
    protocol: openai-chat
  - url: https://api.fireworks.ai/inference/v1/messages
    protocol: anthropic-messages
  - url: https://api.fireworks.ai/inference/v1/responses
    protocol: openai-responses
checked: 2026-07-19
sources:
  - https://docs.fireworks.ai/guides/querying-text-models
  - https://docs.fireworks.ai/api-reference/post-chatcompletions
  - https://docs.fireworks.ai/llms.txt
  - https://docs.fireworks.ai/guides/reasoning.md
  - https://docs.fireworks.ai/guides/reasoning
  - https://docs.fireworks.ai/api-reference/list-models
  - https://docs.fireworks.ai/tools-sdks/openai-compatibility.md
  - https://docs.fireworks.ai/tools-sdks/anthropic-compatibility.md
  - https://docs.fireworks.ai/guides/response-api.md
  - https://fireworks.ai/models/fireworks/glm-5p2
  - https://fireworks.ai/models/fireworks/kimi-k2p7-code
  - https://fireworks.ai/models/fireworks/deepseek-v4-pro
  - https://fireworks.ai/models/fireworks/deepseek-v4-flash
  - https://fireworks.ai/models/fireworks/minimax-m3
  - https://fireworks.ai/models/fireworks/qwen3p7-plus
  - https://api.fireworks.ai/inference/v1/models   # live unauthenticated probe: HTTP 401 (route exists)
  - https://api.fireworks.ai/inference/v1/definitely-not-a-real-endpoint-xyz   # control probe: HTTP 404
reasoning:
  param: "reasoning_effort (top-level string; API schema also admits integer|boolean|null); thinking (object: {\"type\": \"enabled\"} | {\"type\": \"enabled\", \"budget_tokens\": <int, min 1024>} | {\"type\": \"disabled\"} | {\"type\": \"adaptive\"}); reasoning_history (string). thinking and reasoning_effort are mutually exclusive — sending both raises a validation error (documented)."
  values: "reasoning_effort: reasoning guide documents only 'low' | 'medium' | 'high'; the chat-completions API reference schema additionally lists 'xhigh', 'max', 'none', 'adaptive', positive integers, and booleans — CONFLICT recorded, do not emit the extra values without the probe below. thinking.type: 'enabled' | 'disabled' | 'adaptive' (note: 'adaptive' is explicitly UNSUPPORTED on the anthropic-messages surface). reasoning_history: 'disabled' | 'interleaved' | 'preserved'. No per-model value split is published — Fireworks documents these as platform-level params, not per-family."
  default_when_omitted: "Not explicitly documented. Reasoning-capable models emit reasoning per their template default; reasoning_history defaults to 'model/template default'. context: prior refresh's 'all optional' claim is NOT supported by any doc — probe required."
  unsupported_value_behavior: "unknown for out-of-range reasoning_effort values and for reasoning params sent to non-reasoning models (docs silent; only the thinking+reasoning_effort combination is documented to raise a validation error, status code and body shape unpublished). budget_tokens < 1024 violates the documented minimum; error shape unpublished. Probes below."
  per_model:
    - id: accounts/fireworks/models/glm-5p2
      mode: "unknown (model page shows no reasoning badge; GLM family is reasoning-capable but the reasoning guide only names glm-4p7 — probe required)"
      effort: "unknown; if supported, adjustable: ['low','medium','high']"
    - id: accounts/fireworks/models/kimi-k2p7-code
      mode: "always-on presumed (model page describes it as an agentic model using thinking tokens, '~30% fewer thinking tokens' vs prior version); not confirmed whether it can be disabled — probe required"
      effort: "unknown whether reasoning_effort adjusts it — probe required"
    - id: accounts/fireworks/models/deepseek-v4-pro
      mode: "unknown ('designed for frontier reasoning' marketing copy, no thinking-mode contract on the page — probe required)"
      effort: "unknown"
    - id: accounts/fireworks/models/deepseek-v4-flash
      mode: "unknown, likely none or integrated (page: 'near-Pro reasoning quality', no thinking mode referenced — probe required)"
      effort: "unknown"
    - id: accounts/fireworks/models/minimax-m3
      mode: "unknown (page shows NO reasoning capability; prior refresh's 'effort control' claim is unsourced; reasoning guide names minimax-m2, not m3 — probe required)"
      effort: "unknown"
    - id: accounts/fireworks/models/qwen3p7-plus
      mode: "unknown (page shows no reasoning capability; Qwen family is hybrid-thinking upstream — probe required)"
      effort: "unknown"
  silent_substitutions: none documented (no doc describes swapping model ids based on reasoning flags)
  tools_and_streaming: "Reasoning is returned in a dedicated reasoning_content field: choice.message.reasoning_content (non-streaming) and choice.delta.reasoning_content per SSE chunk (accumulate across chunks). Interleaved thinking with tool use is supported: a last message with role 'tool' triggers interleaved thinking, and the client MUST pass reasoning_content back on subsequent turns (pass complete message objects or manually include the reasoning_content field) or reasoning context is silently lost. Whether reasoning tokens count against max_tokens is UNDOCUMENTED — probe below. Streaming usage stats arrive in the final chunk (Fireworks-specific vs OpenAI)."
  compat_passthrough: "openai-chat (/inference/v1/chat/completions): native surface — reasoning_effort and thinking both accepted; passes. anthropic-messages (/inference/v1/messages): thinking supported, and thinking with output_config.effort maps to Fireworks reasoning_effort — passes (mapped); thinking type 'adaptive', output_config.speed, inference_geo, and server-side tools are unsupported on this surface; anthropic-version header ignored; reasoning returned as thinking content blocks in the content array, not reasoning_content. openai-responses (/inference/v1/responses): reasoning params not documented at all (no reasoning_effort, no reasoning.effort) — unknown; do not route reasoning traffic here without a probe."
discovery:
  endpoint: "Control plane: GET https://api.fireworks.ai/v1/accounts/{account_id}/models (documented; pageSize max 200, default 50, pageToken, AIP-160 filter, orderBy, readMask). ALSO: GET https://api.fireworks.ai/inference/v1/models EXISTS on the inference host — live probe returned 401 (auth required) while a bogus path returned 404, so the route is real; its payload shape is unverified (probe below). The prior 'no /v1/models on inference host' claim is REFUTED at the routing level."
  auth: bearer (Fireworks API key) — both surfaces
  metadata: "control-plane listing: ids+capabilities+pricing — contextLength, supportsImageInput, supportsTools, supportsServerless ('if true, the model has a serverless deployment'), serverlessModes[] with skuInfos (pricing), deployedModelRefs[] with state, state (UPLOADING/READY). This directly solves the qwen2p5 trap: supportsServerless distinguishes catalog membership from serverless availability. inference-host /v1/models metadata: unknown."
  account_scoped: "yes: {account_id} path segment scopes the listing; a user's own account lists their models/deployments; the public catalog lives under the 'fireworks' account (whether an arbitrary user key can list accounts/fireworks/models needs the probe below). deployedModelRefs and serverless availability vary per account."
  rate_limits_or_caching: undocumented for the listing endpoints (no documented rate limits or caching headers found)
  verdict: "augment-static — OVERTURNS the prior 'likely stay-static' lean: contextLength + supportsServerless + supportsTools + pricing skuInfos are exactly the fields Sylliptor needs and are served per-account, so use live data to validate/annotate the static catalog (especially serverless availability); keep reasoning-mode facts static since the listing carries no reasoning metadata."
  refresh_strategy: on-config-open (with a daily cache fallback; never per-request)
context:
  per_model:
    - id: accounts/fireworks/models/glm-5p2
      context: "1,040K tokens as displayed on the model page (page renders '1,040k'; whether that is exactly 1,040,000 or a rounded 2^20-based figure is not machine-readable from the page — confirm via control-plane contextLength). No tier splits published."
      max_output: unpublished
      long_context_pricing: "flat per model page: $1.40/M input, $0.14/M cached input, $4.40/M output; no boundary published"
      overpromise_risk: "yes (unverified): serverless deployments have historically capped below headline context on some Fireworks models; confirm live contextLength before advertising 1.04M"
    - id: accounts/fireworks/models/kimi-k2p7-code
      context: "262K tokens (model page: 262,000). No tier splits."
      max_output: unpublished
      long_context_pricing: "flat: $0.95/M input, $0.19/M cached input, $4.00/M output; no boundary"
      overpromise_risk: no (page and prior claim agree at 262K)
    - id: accounts/fireworks/models/deepseek-v4-pro
      context: "1,040K tokens (page renders '1,040k'; same rounding caveat as glm-5p2). No tier splits."
      max_output: unpublished
      long_context_pricing: "flat: $1.74/M input, $0.14/M cached input, $3.48/M output; no boundary"
      overpromise_risk: "yes (unverified): confirm serverless contextLength via control plane before advertising 1.04M"
    - id: accounts/fireworks/models/deepseek-v4-flash
      context: "1,040,000 tokens per model page. No tier splits."
      max_output: unpublished
      long_context_pricing: "flat: $0.14/M input, $0.03/M cached input, $0.28/M output; no boundary"
      overpromise_risk: "yes (unverified): same 1M-class serverless-cap concern"
    - id: accounts/fireworks/models/minimax-m3
      context: "512K tokens. No tier splits."
      max_output: unpublished
      long_context_pricing: "flat: $0.30/M input, $0.06/M cached input, $1.20/M output; no boundary"
      overpromise_risk: no (page matches prior claim)
    - id: accounts/fireworks/models/qwen3p7-plus
      context: "262K tokens. No tier splits."
      max_output: unpublished
      long_context_pricing: "flat: $0.40/M input, $0.08/M cached input, $1.60/M output; no boundary"
      overpromise_risk: no (page matches prior claim)
  token_counting_endpoint: none (no tokenization/token-counting page exists in the docs index; only an image-token billing FAQ)
probes_needed:
  - question: "Per-model reasoning contract: which of the 6 catalog models accept reasoning_effort, which return reasoning_content by default, and which error (repeat per model id)"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.fireworks.ai/inference/v1/chat/completions -H 'Authorization: Bearer $FIREWORKS_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"accounts/fireworks/models/minimax-m3\",\"max_tokens\":64,\"reasoning_effort\":\"low\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2?\"}]}'"
  - question: "Unsupported/extended reasoning_effort values ('xhigh','none','adaptive', integers): accepted per the API-reference union, or 400 per the guide's low/medium/high-only list — and the exact error body shape"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.fireworks.ai/inference/v1/chat/completions -H 'Authorization: Bearer $FIREWORKS_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"accounts/fireworks/models/kimi-k2p7-code\",\"max_tokens\":64,\"reasoning_effort\":\"xhigh\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Whether kimi-k2p7-code thinking can be disabled ({\"thinking\":{\"type\":\"disabled\"}}) or is hard always-on"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.fireworks.ai/inference/v1/chat/completions -H 'Authorization: Bearer $FIREWORKS_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"accounts/fireworks/models/kimi-k2p7-code\",\"max_tokens\":64,\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
  - question: "Do reasoning tokens count against max_tokens (send tiny max_tokens to a reasoning model and inspect usage + finish_reason + whether content is empty while reasoning_content is populated)"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.fireworks.ai/inference/v1/chat/completions -H 'Authorization: Bearer $FIREWORKS_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"accounts/fireworks/models/kimi-k2p7-code\",\"max_tokens\":16,\"reasoning_effort\":\"high\",\"messages\":[{\"role\":\"user\",\"content\":\"Prove sqrt(2) is irrational.\"}]}'"
  - question: "What GET /inference/v1/models returns when authenticated (it exists — 401 unauth vs 404 for bogus paths): ids-only OpenAI list, or richer; whether it is account-scoped"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.fireworks.ai/inference/v1/models -H 'Authorization: Bearer $FIREWORKS_API_KEY'"
  - question: "Whether a normal user key can list the public catalog with serverless flags and real contextLength (settles both the qwen2p5 trap and the 1M-class serverless-cap question)"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' 'https://api.fireworks.ai/v1/accounts/fireworks/models?pageSize=200&readMask=name,contextLength,supportsServerless,supportsTools,supportsImageInput,state' -H 'Authorization: Bearer $FIREWORKS_API_KEY'"
  - question: "Responses API surface: is reasoning_effort (or reasoning.effort) accepted at /inference/v1/responses or rejected"
    probe: "curl -s -w '\\nHTTP %{http_code}\\n' https://api.fireworks.ai/inference/v1/responses -H 'Authorization: Bearer $FIREWORKS_API_KEY' -H 'Content-Type: application/json' -d '{\"model\":\"accounts/fireworks/models/kimi-k2p7-code\",\"input\":\"hi\",\"reasoning_effort\":\"low\",\"store\":false}'"
conflicts:
  - "https://docs.fireworks.ai/api-reference/post-chatcompletions.md schema allows reasoning_effort in {'low','medium','high','xhigh','max','none','adaptive', positive integer, boolean}; https://docs.fireworks.ai/guides/reasoning documents only 'low'|'medium'|'high' — recorded, not resolved (probe queued)."
  - "Prior-refresh claim (registry-derived) said there is NO plain GET /v1/models on the inference host; live probe of https://api.fireworks.ai/inference/v1/models returned 401 while a bogus path returned 404 — the route exists; payload unverified."
  - "Prior refresh marked minimax-m3 'effort control' and all six models reasoning:optional; https://fireworks.ai/models/fireworks/minimax-m3 shows no reasoning capability at all and https://docs.fireworks.ai/guides/reasoning names only kimi-k2-thinking / minimax-m2 / glm-4p7 (none of the six catalog models) — recorded, not resolved."
  - "Model pages for glm-5p2 / deepseek-v4-pro / qwen3p7-plus render 'not a reasoning model' while these upstream families ship thinking modes elsewhere; Fireworks pages may under-render capabilities client-side — recorded, probe queued."
notes: "Load-bearing for Sylliptor's runtime layer: (1) reasoning output lives in reasoning_content (message and streaming delta), and in multi-turn tool loops the client must send reasoning_content back or interleaved-thinking context is silently dropped — a generic OpenAI-shaped message replay that strips unknown fields will degrade quality with no error. (2) thinking and reasoning_effort are mutually exclusive (documented validation error). (3) context_length_exceeded_behavior defaults to 'truncate' — Fireworks silently truncates max_tokens/prompt fit instead of erroring, which can mask context-accounting bugs; set 'error' if Sylliptor wants honest overflow signals. (4) The anthropic-messages surface returns thinking as content blocks (not reasoning_content) and rejects/omits thinking type 'adaptive'. (5) max output tokens are unpublished for all six models. (6) Discovery should use control-plane list-models readMask to pull contextLength + supportsServerless per account (augment-static)."
````

### Gap-fill addendum (post-verification)

Verified 2026-07-19 against live official docs. A2 (reasoning_effort value conflict) is RECONCILED at the doc level; A4 (per-model modes) remains probe-required for 5 of 6 ids, with one new doc signal for minimax-m3.

reasoning_effort values: the prose guide at https://docs.fireworks.ai/guides/reasoning and its markdown twin https://docs.fireworks.ai/guides/reasoning.md document ONLY the string values "low", "medium", "high" (exact quote: 'string values like "low", "medium", or "high"'), and explicitly state neither integers nor booleans are the primary documented values. The JSON request schema at https://docs.fireworks.ai/api-reference/post-chatcompletions is the authoritative superset and additionally admits the string enum "xhigh", "max", "none", "adaptive" plus positive integer, boolean, and null. These are not contradictory: the guide is a documented prose subset, the schema is the full platform contract. The schema is platform-level (not per-family/per-model), so which extra values a given catalog id accepts-vs-rejects is NOT published and needs the runtime probe.

thinking object (from schema): {"type":"enabled"} | {"type":"enabled","budget_tokens":<int>=>1024} | {"type":"enabled","keep":"all"} (preserve historical reasoning) | {"type":"disabled"} | {"type":"adaptive"} | null. Critical new per-model signal: the schema annotates {"type":"adaptive"} as "MiniMax M3 only" — this is the ONLY per-model reasoning fact any live doc carries for the current catalog, and it ties accounts/fireworks/models/minimax-m3 to reasoning-capable / optional (adaptive). thinking and reasoning_effort remain mutually exclusive (validation error if both).

reasoning_history (from schema): "disabled" (strip reasoning from all messages) | "interleaved" (strip up to last user message) | "preserved" (keep across conversation) | null (model/template default).

Per-model A4 status: all six model pages (https://fireworks.ai/models/fireworks/{glm-5p2,kimi-k2p7-code,deepseek-v4-pro,deepseek-v4-flash,minimax-m3,qwen3p7-plus}) were fetched and carry NO reasoning/thinking contract or badge — only marketing prose ("frontier reasoning", "near-Pro reasoning quality", kimi "reducing thinking-token usage by ~30% vs Kimi K2.6"). The reasoning guide names ONLY prior-generation examples (accounts/fireworks/models/kimi-k2-thinking, minimax-m2, glm-4p7), none of which are in the current catalog. Therefore docs classify NONE of glm-5p2 / kimi-k2p7-code / deepseek-v4-pro / deepseek-v4-flash / qwen3p7-plus as always-on|optional|none; only minimax-m3 gets a doc-level classification (optional/adaptive) via the schema annotation. Default-when-omitted is not documented for any id. Runtime probe with an API key is the only way to settle the remaining five modes and the accepted reasoning_effort value set per id.

Corrected/added YAML lines for the affected keys:

````yaml
reasoning:
  values: "reasoning_effort: DOC-LEVEL RECONCILED (was CONFLICT). Prose guide (/guides/reasoning[.md]) documents only 'low'|'medium'|'high' and says integers/booleans are not the primary documented values; the /api-reference/post-chatcompletions JSON schema is the authoritative SUPERSET and additionally admits string 'xhigh'|'max'|'none'|'adaptive', positive integer, boolean, and null. Not contradictory (prose subset vs full platform contract). Schema is platform-level, NOT per-model — which extra values any given id accepts vs rejects is unpublished; per-model runtime probe still required. thinking.type: 'enabled'|'disabled'|'adaptive'|null; schema annotates {\"type\":\"adaptive\"} as 'MiniMax M3 only'; thinking also supports {\"type\":\"enabled\",\"keep\":\"all\"} (preserve historical reasoning) and budget_tokens>=1024. reasoning_history: 'disabled'|'interleaved'|'preserved'|null(model/template default)."
  per_model:
    - id: accounts/fireworks/models/glm-5p2
      mode: "unknown — probe required (model page has no reasoning contract; guide names glm-4p7 not glm-5p2)"
      effort: "unknown; schema-admitted set is low|medium|high|xhigh|max|none|adaptive|int|bool|null but per-id acceptance unpublished — probe required"
    - id: accounts/fireworks/models/kimi-k2p7-code
      mode: "unknown — probe required (page mentions 'thinking-token usage ~30% lower vs Kimi K2.6', an efficiency note, NOT a mode contract; cannot confirm always-on vs disable-able)"
      effort: "unknown — probe required"
    - id: accounts/fireworks/models/deepseek-v4-pro
      mode: "unknown — probe required ('frontier reasoning' is marketing, no thinking-mode contract)"
      effort: "unknown — probe required"
    - id: accounts/fireworks/models/deepseek-v4-flash
      mode: "unknown — probe required ('near-Pro reasoning quality' is marketing, no thinking-mode contract)"
      effort: "unknown — probe required"
    - id: accounts/fireworks/models/minimax-m3
      mode: "optional/adaptive (DOC-CLASSIFIED): schema ties thinking {\"type\":\"adaptive\"} to 'MiniMax M3 only'; reasoning-capable with an adaptive (model-decides) mode. Whether it can be fully disabled via {\"type\":\"disabled\"} vs reasoning_effort='none' still needs confirmation — probe recommended"
      effort: "adaptive supported via thinking; reasoning_effort acceptance unpublished — probe recommended"
    - id: accounts/fireworks/models/qwen3p7-plus
      mode: "unknown — probe required (page shows no reasoning capability; guide does not name it)"
      effort: "unknown — probe required"
````

**Still unresolved (needs a live key):**

````
Authenticated per-model probe (no doc classifies glm-5p2/kimi-k2p7-code/deepseek-v4-pro/deepseek-v4-flash/qwen3p7-plus; minimax-m3 is doc-classified adaptive but disable-path unconfirmed). Run for EACH id in {glm-5p2,kimi-k2p7-code,deepseek-v4-pro,deepseek-v4-flash,minimax-m3,qwen3p7-plus}, sweeping reasoning_effort across the full schema set, and observe (a) HTTP 200 vs 400 per value and (b) whether choices[0].message.reasoning_content is non-empty:

for M in glm-5p2 kimi-k2p7-code deepseek-v4-pro deepseek-v4-flash minimax-m3 qwen3p7-plus; do for E in low medium high xhigh max none adaptive 2048 true false; do echo "== $M effort=$E =="; curl -s -o /dev/null -w "%{http_code}\n" https://api.fireworks.ai/inference/v1/chat/completions -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" -d "{\"model\":\"accounts/fireworks/models/$M\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2? think first\"}],\"max_tokens\":64,\"reasoning_effort\":\"$E\"}"; done; done

Then confirm the omit-param default and the thinking/adaptive disable path (esp. minimax-m3):
curl -s https://api.fireworks.ai/inference/v1/chat/completions -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" -d '{"model":"accounts/fireworks/models/minimax-m3","messages":[{"role":"user","content":"hi"}],"max_tokens":64}' | python -c "import sys,json;d=json.load(sys.stdin);print('reasoning_content' in d['choices'][0]['message'])'
curl -s -o /dev/null -w "%{http_code}\n" https://api.fireworks.ai/inference/v1/chat/completions -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" -d '{"model":"accounts/fireworks/models/minimax-m3","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"thinking":{"type":"disabled"}}'

Classify each id: reasoning_content present when omitted + rejects 'none'/disabled => always-on; togglable => optional; never present + rejects reasoning params => none. Record the accepted reasoning_effort value set per id (esp. whether xhigh/max/int/bool are honored or 400).
````

### Verifier notes (minor, not applied)

- **discovery.metadata lists 'deployedModelRefs[] with state' as a field returned by the control-plane listing (GET /v1/accounts/{account_id}/models).** — The list-models doc explicitly states deployedModelRefs is 'Populated from GetModel API call only', i.e. it is NOT returned by ListModels. Presenting it as part of the listing metadata is inaccurate. *Suggested: Move deployedModelRefs out of the listing-fields set and note it is only populated on GetModel, not ListModels. The augment-static verdict is unaffected (it rests on contextLength/supportsServerless/supportsTools/pricing, all confirmed listing fields).*
- **compat_passthrough / notes assert that on the anthropic-messages surface reasoning is 'returned as thinking content blocks in the content array, not reasoning_content'.** — The anthropic-compatibility doc does not explicitly document how reasoning/thinking is returned in the response content blocks. This is a reasonable inference from Anthropic's native shape but is stated as documented fact. *Suggested: Mark the 'thinking content blocks vs reasoning_content' response-shape claim as inferred/probe-required rather than documented.*
- **discovery.endpoint describes the control-plane filter as an 'AIP-160 filter'.** — The list-models doc describes the filter param only as 'Only model satisfying the provided filter (if specified) will be returned' and does not name the AIP-160 standard. The specific 'AIP-160' attribution is unsourced from the cited page. *Suggested: Drop the 'AIP-160' qualifier or mark it as an inference about the filter syntax.*

---

# API-error hazard list

Ranked by how likely Sylliptor is to trigger each **today**, given how it actually behaves: every
request carries tool definitions, most presets speak openai-chat, the TUI exposes a reasoning
on/off toggle, clients commonly set temperature, and context headroom is computed from hardcoded
preset numbers. Rank 1 fires on default settings; the bottom of the table needs a specific (but
plausible) config.

## Top-ranked hazards

| # | provider · model(s) | wrong emission | result | why Sylliptor hits it |
|---|---|---|---|---|
| 1 | openai · gpt-5.6-* and gpt-5.4-* on /v1/chat/completions | `tools:[...]` + `reasoning_effort` ≠ `none` | http-400: "Function tools with reasoning_effort are not supported for <model> in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'." | **Fires on the very first default call**: the `openai` preset is chat-protocol, Sylliptor always sends tools, and gpt-5.6's server default effort is `medium`. The chat preset is broken out of the box unless effort is pinned to `none` or traffic moves to `openai-responses`. |
| 2 | anthropic · sonnet-5, opus-4-8, opus-4-7, fable-5 | extended-thinking shape `thinking:{"type":"enabled","budget_tokens":N}` | http-400 invalid_request_error — manual extended thinking is rejected on all four | Any client still speaking the pre-adaptive shape breaks the moment reasoning is toggled on. The correct emission is `thinking:{"type":"adaptive"}` (+ `output_config:{"effort":...}`). |
| 3 | anthropic · same four models | any non-default `temperature` / `top_p` / `top_k` | http-400 on **every** request, thinking or not | A generic CLI that always sends temperature breaks on all four; haiku-4-5 still accepts sampling params. Guard at the client layer. |
| 4 | gemini · all 3.x | replaying tool-call history without `thoughtSignature` on each functionCall part (or letting an OpenAI-compat layer strip it) | http-400: "Function call ... is missing a `thought_signature`" | Fails at **turn 2 of every tool loop** — invisible in single-shot testing. The OpenAI-compat path drops it unless `extra_content.google.thought_signature` is round-tripped. |
| 5 | moonshot · kimi-k2.7-code, -highspeed | `thinking:{"type":"disabled"}` | http-400 "invalid thinking" | This is exactly what a reasoning-off toggle emits. Only kimi-k2.6 may receive a disable flag on this preset. |
| 6 | moonshot · kimi-k2.7-code (k3 likewise fixed) | `temperature` ≠ 1.0 or `top_p` ≠ 0.95 | http-400 — "Any other value will result in an error" | Same temperature habit as #3, different provider. Omit sampling params entirely on this preset. |
| 7 | perplexity · anthropic/claude-sonnet-5, anthropic/claude-opus-4-8 | omitting `max_output_tokens` | http-400 `{"error":{"message":"validation failed: max_output_tokens is required when using Anthropic models"}}` | Generic OpenAI-style clients rarely send it; fails on every request to the two anthropic ids. |
| 8 | bytedance · seed-2.0 family (coding/anthropic-compat surface) | `reasoning_effort: "high"` (auto-attached by agentic CLIs) | http-400 `{"error":{"code":"InvalidParameter","message":"Unsupported reasoning_effort type..."}}` — live-observed | The one bytedance failure that is field-confirmed, and it is triggered by the most common effort spelling in the ecosystem. |
| 9 | cerebras · zai-glm-4.7 / gpt-oss-120b | `disable_reasoning: true` (pre-2026-07-21 spelling), or `reasoning_effort:"none"` on gpt-oss | param removed (400 or silent-ignore, undocumented); `"none"` not in gpt-oss's set | The refresh's own act-now deadline has passed — any config still carrying `disable_reasoning` is emitting a dead parameter. And a uniform "reasoning off" path breaks on the default model. |
| 10 | kimi-code · all three ids | any thinking-disable spelling | **no error — silent swap to K2.6**, a different model | Worse than a 400: the reasoning toggle quietly changes which model answers. Sylliptor must surface the substitution, never treat it as a speed knob. |
| 11 | deepseek · pinned `deepseek-chat` / `deepseek-reasoner` | any request after 2026-07-24 15:59 UTC | model-not-found-class failure (exact body undocumented; probe filed) | Five days out at check time. Saved user configs pinning the legacy aliases break; preset-level aliases cover new sessions only. |
| 12 | openrouter · z-ai/glm-5.2, deepseek/deepseek-v4-pro | `reasoning:{"effort":"medium"}` (the generic default) | live `supported_efforts` is `["xhigh","high"]` only — coercion, ignore, or 400 depending on gateway/provider (unsettled) | Sylliptor's economy/agentic OpenRouter slots would receive the most common effort value in existence. Read `supported_efforts` live instead of assuming the OpenAI set. |
| 13 | groq · groq/compound | standard `tools:[{type:"function",...}]` | rejected — "Custom user-provided tools are not supported at this time" | The preset's agentic slot cannot run Sylliptor's agent loop at all; compound runs only its own server-side tools. |
| 14 | cerebras · gpt-oss-120b | `tools` + `response_format` in one request | documented hard rejection | Agentic CLIs that attach tools everywhere and request JSON output hit this on the default model. |
| 15 | mistral · mistral-medium-2604, mistral-small-2603 | parsing `message.content` / `delta.content` as a string while reasoning is active | not an HTTP error — content is a **list of typed chunks** (`thinking`, then `text`); string-assuming parsers break or lose output | Breaks response handling, not the request. Also: replaying history without the ThinkChunks loses reasoning continuity. |
| 16 | openai · any 5.x carrying old configs | `reasoning_effort: "minimal"` (valid on the original gpt-5 family) | http-400 `unsupported_value` — `minimal` exists on **no** current catalog model | Carried-forward configs and generic CLIs still emit it; the 5.6 set is `none\|low\|medium\|high\|xhigh\|max`. |
| 17 | xai · grok-4.5 / grok-4.3 | `reasoning_effort` together with `stop`, `presence_penalty`, or `frequency_penalty` | documented "returns an error" (shape unpublished) | Stop sequences are a common agent-CLI emission; on xai they conflict with the effort param itself. |
| 18 | zhipu · glm-5.2 (all models) | OpenRouter-style `reasoning:{"effort":"high"}` instead of native `thinking`/`reasoning_effort` | **silently ignored** — request succeeds, reasoning is not controlled | A silent no-op: the user's chosen reasoning mode simply never applies. Field-confirmed. |
| 19 | anthropic/moonshot/openrouter/fireworks · reasoning models in tool loops | stripping `thinking` blocks / `reasoning_content` / `reasoning_details` when replaying assistant tool-call turns | anthropic + moonshot: http-400; openrouter/fireworks/mistral: silent reasoning-continuity loss | The naive openai-chat replay shape (content + tool_calls only) is wrong on five surfaces. Preserve provider-returned reasoning fields verbatim. |
| 20 | qwen · qwen3.7 line | omitting `enable_thinking`, assuming thinking is off | no error — thinking runs by default; unparsed `reasoning_content` deltas + billed thinking tokens | Silent cost/latency inflation and possible parser confusion on a preset default model. |

## Context over-promise cluster (not request errors — accounting lies)

These are the models where a hardcoded context makes Sylliptor's headroom gauge wrong for some
accounts. Each is flagged `overpromise_risk: yes` in its block:

- **kimi-code k3** — 1M is Allegretto+ only; Moderato keys get 256K (4× over-promise), Andante keys cannot use k3 at all.
- **cerebras all three** — free-trial keys: 64–65K context / 32K output vs the 131K/40K headline.
- **minimax M3** — only 512K guaranteed; >512K input may be rejected for some accounts and bills 2×.
- **moonshot k3** — window is a flat 1M, but Tier0 accounts (TPM 500K) can never fit a large prompt through rate limiting.
- **mistral codestral-2508** — 128K real vs 262144 in the applied catalog.
- **cohere command-a-plus-05-2026** — 128K in / 64K out, half the Command-A family.
- **anthropic claude-haiku-4-5** — 200K beside 1M siblings; plus the ~30% tokenizer shift on the other four models moves *measured* headroom even where the window is right.
- **openrouter open-weight slugs** — per-route context spans 96,890–1,048,576 under one id; compute worst-case across `/endpoints` for the routes actually allowed.
- **perplexity all six** — no context published at all; any number Sylliptor displays is invented.
- **together MiniMax-M3** — served at 524,288, not the 1M the same model has elsewhere; same-name-different-window is a trap when copying context across presets.
- **fireworks all six** — default `context_length_exceeded_behavior: "truncate"` silently shrinks the effective window; set `"error"` so accounting failures are visible.
- **qwen qwen3-coder-next** — 256K, sitting in a lineup of 1M models; qwen-us serves no coder ids at all.

---

## Full hazard inventory (all 179, grouped by likelihood)

The complete list as filed per provider, verbatim. "Likelihood" is the researching
agent's estimate of a generic agentic CLI triggering the emission, before the
Sylliptor-specific ranking above.

### high (69)

| provider | model | wrong emission | result |
|---|---|---|---|
| openai + openai-responses | gpt-5.6-terra / gpt-5.6-sol / gpt-5.6-luna / gpt-5.4-mini / gpt-5.4-nano | POST /v1/chat/completions with tools:[...] AND reasoning_effort set to any value other than 'none' (on 5.6 this happens by default since default effort is medium; on 5.4-mini/nano only when effort is explicitly raised ab… | HTTP 400: "Function tools with reasoning_effort are not supported for <model> in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'." (message observed live for gpt-5.6-sol; community reply confirms the blocking st… |
| openai + openai-responses | gpt-5.6-* / gpt-5.4-* / gpt-5.3-codex | reasoning effort value 'minimal' (valid on original gpt-5 family, carried forward by generic CLIs) | HTTP 400 invalid_request_error, code 'unsupported_value', param reasoning_effort: "Unsupported value: 'reasoning_effort' does not support 'minimal' with this model..." |
| openai + openai-responses | gpt-5.3-codex | reasoning effort 'none' (to disable reasoning, as works on 5.4/5.6) | HTTP 400 unsupported_value (model publishes only low/medium/high/xhigh; reasoning cannot be turned off) |
| openai + openai-responses | any catalog model on /v1/chat/completions | nested Responses-style body field reasoning: {"effort": ...} instead of flat reasoning_effort | expected HTTP 400 unrecognized/unknown parameter (unconfirmed - probe filed); the two surfaces spell the parameter differently and neither accepts the other's shape per docs |
| openai + openai-responses | gpt-5.6 family at high/xhigh/max effort or mode=pro | small max_output_tokens (e.g. 4096) carried over from non-reasoning presets | not an HTTP error: reasoning tokens are billed as output and count against max_output_tokens, so the run ends with reasoning exhausted and empty/truncated visible output (status incomplete) |
| anthropic (x3 presets, one surface: api.… | claude-opus-4-8, claude-opus-4-7, claude-sonnet-5, claude-fable-5 | "thinking": {"type": "enabled", "budget_tokens": 8000} | HTTP 400 invalid_request_error — manual extended thinking is rejected on all four ('rejected with a 400 error') |
| anthropic (x3 presets, one surface: api.… | claude-fable-5 | "thinking": {"type": "disabled"} | Request rejected (docs: 'is not supported'/'is rejected'; status code unprinted, expected 400) — thinking is always on; the only safe emissions are omitting thinking or {"type": "adaptive"} |
| anthropic (x3 presets, one surface: api.… | claude-fable-5, claude-opus-4-8, claude-opus-4-7, claude-sonnet-5 | "temperature": 0.7 (or any non-default temperature/top_p/top_k) | HTTP 400 — 'reject non-default temperature, top_p, and top_k values with a 400 error. This applies to every request on these models, regardless of whether thinking is active' |
| anthropic (x3 presets, one surface: api.… | claude-opus-4-8, claude-opus-4-7 | omitting the thinking parameter entirely while expecting reasoning | No error — request silently runs WITHOUT thinking ('Thinking is off unless you explicitly set thinking: {type: "adaptive"}'); degraded quality on hard tasks with no signal |
| anthropic (x3 presets, one surface: api.… | claude-haiku-4-5 | "thinking": {"type": "adaptive"} | Undocumented (overview table says Adaptive=No); expected 400 invalid_request_error but unverified — probe listed |
| anthropic (x3 presets, one surface: api.… | all five (agentic tool loops) | rebuilding the last assistant message and dropping/re-serializing its thinking or redacted_thinking blocks before sending tool_result | HTTP 400 invalid_request_error: '`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. These blocks must remain as they were in the original response.' |
| gemini (x3 presets) | gemini-3.5-flash / all 3.x | Replaying function-call history without the thoughtSignature field on the first functionCall part of each step (or after middleware strips unknown fields) | HTTP 400: "Function call `<Function Call>` in the `<index of contents array>` content block is missing a `thought_signature`" |
| gemini (x3 presets) | all 3.x | Interleaving parallel function calls/results as FC1+sig, FR1, FC2, FR2 instead of FC1+sig, FC2, FR1, FR2 | HTTP 400 validation error on the replay turn |
| gemini (x3 presets) | all 3.x via OpenAI-compat | OpenAI client normalizes assistant tool_calls and drops extra_content.google.thought_signature | Next request fails HTTP 400 missing thought_signature; silent until turn 2 of any tool loop |
| gemini (x3 presets) | all 3.x | temperature: 0 or 0.2 (standard agentic-CLI habit) | No HTTP error but documented looping/degraded reasoning — silent quality failure |
| deepseek | deepseek-chat / deepseek-reasoner (pinned legacy config) | model="deepseek-chat" (or deepseek-reasoner) requested after 2026/07/24 15:59 UTC | Hard failure once the alias is removed -- expected 400 "Invalid Format" / model-not-found (exact post-deprecation body undocumented; see probe). Until 2026/07/24 it silently maps to deepseek-v4-flash (non-thinking / thinking). |
| deepseek | deepseek-v4-pro / deepseek-v4-flash (anthropic surface) | model set to an unrecognized/Claude name, or Anthropic thinking={"type":"enabled","budget_tokens":1024} on https://api.deepseek.com/anthropic | SILENT model swap: unknown model name falls back to deepseek-v4-flash (Opus->pro, Haiku/Sonnet->flash); budget_tokens is silently ignored so thinking runs at default/auto effort instead of the requested budget. No error surfaced -- wrong model or unbounded rea… |
| qwen-intl / qwen-us / qwen-cn | qwen3.7-plus / qwen3.7-max / qwen3.6-flash | No reasoning parameter at all (client assumes thinking is off by default) | No HTTP error, but thinking runs by default: responses carry reasoning_content delta chunks the client may not parse, latency and billed output tokens inflate, and any context-accounting that ignores reasoning tokens drifts |
| qwen-intl / qwen-us / qwen-cn | any qwen model via OpenAI Python SDK path | enable_thinking passed as a named SDK kwarg instead of extra_body (or dropped by an SDK that validates kwargs) | OpenAI SDK raises TypeError client-side or strips the field, so the toggle silently never reaches the API — default thinking behavior persists |
| qwen-intl / qwen-us / qwen-cn | qwen3.7-max (and any qwen id) | "reasoning_effort": "high" (OpenAI-style) on /compatible-mode/v1/chat/completions | Undocumented: DashScope documents reasoning_effort only for DeepSeek/GLM; on Qwen it is either 400 InvalidParameter or silently ignored — probe queued |
| qwen-intl / qwen-us / qwen-cn | qwen3-coder-plus / qwen3-coder-next on qwen-us | Any request routed to https://dashscope-us.aliyuncs.com with a coder model id | Model-not-found class 400/404 — coder models are not served in the US region per the official models page |
| zhipu | glm-5.2 (all models) | OpenRouter/OpenAI-Responses-style reasoning param: "reasoning": {"effort": "high"} instead of native "thinking"/"reasoning_effort" | Silently ignored — request succeeds but no reasoning_content is produced/controlled (field-confirmed in NousResearch/hermes-agent #16533); degraded output with no error signal |
| zhipu | glm-5.1, glm-5-turbo, glm-4.7, glm-4.7-flashx, glm-4.7-flash | reasoning_effort: "high" (any value) copied from the glm-5.2 config onto sibling models | Undocumented: either HTTP 400 code 1214 (invalid param) or silent ignore; docs scope reasoning_effort to glm-5.2 only |
| zhipu | glm-5.1, glm-5 (Coding Plan endpoints) | model: "glm-5.1" on /api/coding/paas/v4 | Silent model swap: request is served by glm-5.2 per documented redirect — responses attributed to the wrong model in logs/accounting |
| zhipu | any (Coding Plan key) | Coding Plan key sent to https://api.z.ai/api/paas/v4 (or general key to /api/coding/paas/v4) | Auth/quota rejection — endpoints documented as not interchangeable; plan quota unusable outside designated tools |
| moonshot / moonshot-cn | kimi-k2.7-code / kimi-k2.7-code-highspeed | "thinking": {"type": "disabled"} (CLI exposes a reasoning-off toggle) | HTTP 400 invalid_request_error — docs: 'Kimi K2.7 Code model will throw an error if the thinking mode is disabled'; Claude Code doc quotes '400 invalid thinking' |
| moonshot / moonshot-cn | kimi-k2.7-code(-highspeed) (documented) / kimi-k3 (fixed, error unprobed) | temperature: 0.7 (or any value != 1.0), or top_p != 0.95 | k2.7-code: HTTP 400 — 'Any other value will result in an error' (documented per-param in the k2.7-code quickstart). kimi-k3: the same values 'are fixed; omit them from requests' — error-vs-coerce on send is NOT documented (probe queued). Either way the params … |
| moonshot / moonshot-cn | kimi-k3 | reasoning_effort: "medium" (OpenAI-convention value) | Not in allowed set low\|high\|max — expected HTTP 400 invalid_request_error (coercion not documented; probe queued) |
| moonshot / moonshot-cn | kimi-k2.7-code (and kimi-k2.6 with keep:"all") | Resending conversation history with reasoning_content stripped from prior assistant tool-call messages | HTTP error on the follow-up request — 'you must keep the reasoning_content from the assistant message in the current turn's tool call within the context, otherwise an error will be thrown' |
| moonshot / moonshot-cn | kimi-k3 | "thinking": {"type": "enabled"} (K2.x-style param sent to k3) | Docs say k3 'uses reasoning_effort instead' of thinking; reject-vs-ignore undocumented — plausibly HTTP 400 (probe queued) |
| kimi-code | k3 | reasoning_effort: "medium" (or "minimal"/"xhigh" — any generic OpenAI-style effort value outside low\|high\|max) | HTTP 400 ("any other unknown -> HTTP 400 error"); error body shape undocumented |
| kimi-code | k3 | context accounting hardcoded to 1048576 for a Moderato key | Requests between 256K and 1M tokens fail (overflow behavior undocumented); headroom gauge lies until then |
| kimi-code | all (k3, kimi-for-coding, kimi-for-coding-highspeed) | membership key sent to api.moonshot.ai / platform endpoints, or a platform key sent to api.kimi.com/coding | Authentication failure — membership keys are valid only on api.kimi.com/coding endpoints |
| kimi-code | k3, kimi-for-coding | max_tokens sized for visible output only (e.g. 512-1024) with thinking always on | Truncated or empty content — reasoning_content tokens count against max_tokens (sum must be <= max_tokens) |
| minimax | MiniMax-M3 | Calling the /anthropic (anthropic-messages) surface and OMITTING the thinking param, assuming Claude-style default-on reasoning | Silent behavior change, not an error: M3 defaults to thinking DISABLED on the anthropic surface, so the agent gets non-reasoning answers and degraded coding/agentic quality with no signal |
| bytedance | doubao-seed-2-0-code-preview-260215 (and any seed-2.0 model on the coding-plan/Anthropic-c… | "reasoning_effort": "high" (OpenAI-style top-level param, auto-attached by many agentic CLIs) | HTTP 400 {"error":{"code":"InvalidParameter","message":"Unsupported reasoning_effort type. Request id: ...","param":"","type":"BadRequest"}} — live-observed on a doubao-seed-2.0-code call |
| bytedance | all four catalog models | BytePlus-intl id (e.g. seed-2-0-lite-260228 or seed-2-0-pro-260328) sent to ark.cn-beijing.volces.com, or the doubao- prefixed cn id sent to BytePlus | model-not-found class error (Model.InvalidName / InvalidParameter); the two platforms have disjoint id namespaces AND different version dates |
| bytedance | any Ark model | stream without "stream_options": {"include_usage": true} | no error, but usage never arrives in SSE — token accounting silently reads zero |
| groq | qwen/qwen3.6-27b | reasoning_effort: "medium" (or "low"/"high") — a single global effort knob applied across the catalog | value outside this family's documented enum (none \| default); behavior undocumented — expected 400 invalid_request_error, possibly silent coercion; probe queued |
| groq | groq/compound | tools: [ {type: function, ...} ] — standard agent-loop tool definitions | docs: 'Custom user-provided tools are not supported at this time'; error shape unpublished (probe queued). Never route agent loops here — compound runs its own server-side tools only |
| groq | openai/gpt-oss-120b | reasoning_format: "parsed" (or any value) sent to a gpt-oss model | documented 'not supported' on gpt-oss; error code unpublished — expected 400; must use include_reasoning on this family instead |
| cerebras | zai-glm-4.7 | "disable_reasoning": true (the pre-2026-07-21 disable spelling) | Parameter is removed after 2026-07-21; post-removal behavior undocumented (likely 400 invalid_request or silent ignore leaving reasoning ON). Either way the CLI's disable-reasoning intent breaks. |
| cerebras | gpt-oss-120b | "reasoning_effort": "none" (generic 'reasoning off' path applied uniformly across models) | "none" is not in gpt-oss-120b's allowed set (low\|medium\|high only); docs do not publish the failure mode — either HTTP 400 or silently ignored with reasoning still on at medium. Reasoning cannot be disabled on this model. |
| cerebras | gpt-oss-120b | request containing both "tools" and "response_format" | Documented hard rejection: "gpt-oss-120b rejects requests containing both fields". Agentic CLIs that attach tool definitions to every request and also request structured output will get an API error on this model only. |
| cerebras | all, free-trial keys | prompt sized to the 131k headline context (or max_completion_tokens 40000) on a free-tier key | Free tier is capped at 64-65k context / 32k output; over-length requests fail or truncate (exact behavior undocumented). Hardcoding the paid-tier numbers over-promises for free accounts. |
| mistral | mistral-medium-2604 / mistral-small-2603 | Client assumes message.content / delta.content is always a string while reasoning is active (reasoning_effort omitted or "high") | Not an HTTP error — response parsing breaks or thinking text is lost: content is a list of typed chunks (type "thinking" then "text"), and in SSE delta.content switches shape mid-stream from chunk list to plain string |
| mistral | mistral-medium-2604 | OpenAI-Responses-style nested object reasoning: {"effort": "high"} or Anthropic-style thinking: {"type": "enabled"} instead of top-level reasoning_effort string | Likely HTTP 422 validation error (Mistral's API validates request bodies; exact behavior unconfirmed — probe queued); at best silently ignored, losing reasoning control |
| mistral | mistral-medium-2604 / mistral-small-2603 | Multi-turn history replayed without the ThinkChunk entries (chunks stripped to plain text) | No HTTP error, but docs state reasoning continuity is lost; docs explicitly require replaying the full assistant message including ThinkChunk |
| xai | grok-4.5 | reasoning_effort: "none" (offering an 'off' switch, valid on grok-4.3) | undocumented — grok-4.5 'Reasoning cannot be disabled' and none is not in its value set; expected HTTP 4xx, shape unpublished (probe queued) |
| xai | grok-4.3 / grok-4.5 (any reasoning-effort request) | reasoning_effort combined with stop, presence_penalty, or frequency_penalty | documented as 'returns an error' (code/body unpublished) — request fails outright |
| xai | grok-4.20-0309-non-reasoning | reasoning_effort: "low" (or any value) sent to the non-reasoning snapshot | undocumented; page says Reasoning: No — likely HTTP 4xx, possibly silently ignored (probe queued) |
| xai | all five catalog models | letting prompt assembly cross 200,000 prompt tokens by a few tokens | not an error: entire request re-billed at the doubled long-context rate ('billed at the higher rate for all tokens in the request') |
| cohere | command-a-reasoning-08-2025 (and command-a-plus-05-2026) | reasoning_effort: "medium" or "low" on /compatibility/v1/chat/completions | Docs state only "none" and "high" are supported and medium/low are "not supported", but do not document whether the result is an HTTP 4xx or a silent ignore/coercion; either way the requested effort level is not honored |
| cohere | all (compat surface) | parallel_tool_calls (also n, logit_bias, top_logprobs, service_tier, store, metadata) | explicitly listed as unsupported on the compat surface; failure mode (reject vs ignore) undocumented |
| cohere | command-a-reasoning-08-2025 | small max_tokens (e.g. 512-1024) with thinking left at its enabled default | thinking appears to share the output limit (guide: 31K budget rec + reserve >=1K for the response on a 32K model); "When the budget is exceeded, the model will immediately proceed with the final response" — a tight max_tokens can yield truncated or near-empty … |
| openrouter | z-ai/glm-5.2 and deepseek/deepseek-v4-pro | reasoning: {"effort": "medium"} (or "low"/"minimal") — the generic default effort most CLIs emit | Undocumented: live metadata lists supported_efforts ["xhigh","high"] only; outcome is coercion, silent ignore, or 400 {"error":{"code":400,...}} depending on gateway policy (probe filed). Either wrong reasoning level or a hard request failure. |
| openrouter | z-ai/glm-5.2, deepseek/deepseek-v4-pro (any multi-provider open-weight slug) | Hardcoding the headline context_length 1,048,576 (or top_provider.max_completion_tokens) into context accounting | Routed provider may only offer 96,890-262,144 context (AkashML/Ambient/DigitalOcean/BaseTen/Together) or 16,384 max output (DeepInfra on deepseek-v4-pro) — 400 context-length-exceeded or silent truncation on long agentic transcripts |
| openrouter | anthropic/claude-sonnet-5, anthropic/claude-opus-4.8 | reasoning: {"max_tokens": N} with N >= the request max_tokens (e.g. small max_tokens for a quick tool turn) | Docs: max_tokens must be strictly higher than the reasoning budget; violating this yields an Anthropic-side 400 surfaced through OpenRouter's error envelope |
| openrouter | any tool-using reasoning model (esp. anthropic/*) | Dropping message.reasoning_details when echoing assistant tool-call turns back (keeping only content+tool_calls, the naive openai-chat pattern) | Loses encrypted/signed thinking blocks that must be passed back verbatim for reasoning continuity — degraded multi-step tool runs or provider-side 400 on strict validation of the resumed thinking chain |
| perplexity | anthropic/claude-sonnet-5 and anthropic/claude-opus-4-8 | POST /v1/agent or /v1/responses without max_output_tokens (a generic OpenAI-style client rarely sends it) | HTTP 400, body {"error":{"message":"validation failed: max_output_tokens is required when using Anthropic models",...}} — hard request failure, not degraded output |
| perplexity | all six catalog models | flat "reasoning_effort": "high" (chat-completions spelling) instead of the object "reasoning": {"effort": "high"} | undocumented — either 400 ValidationError or silent drop leaving reasoning at unknown default; either way the intended reasoning_mode is not applied |
| perplexity | all six catalog models (context accounting) | hardcoding a context window borrowed from vendor-native docs (e.g. Anthropic 200k, Moonshot 256k) or misreading pricing tierThreshold 272000/200000 as a context window | context-gauge overpromise and mid-conversation hard failures at the real (unpublished) proxy ceiling; no token-counting endpoint exists to self-correct |
| together | moonshotai/Kimi-K2.7-Code | "reasoning": {"enabled": false} (a generic reasoning-off toggle) | Unknown — thinking "cannot be disabled" per Together's model page, but docs never say whether the param 400s or is silently ignored (model reasons anyway). Either way the CLI's reasoning-off intent is not honored; if it 400s it is a hard production error. |
| together | deepseek-ai/DeepSeek-V4-Pro | "reasoning_effort": "low" or "medium" (to reduce cost/latency) | Silently coerced to "high" (documented normalization). No error, but the user gets full-depth reasoning tokens billed as completion tokens — a silent cost/latency surprise, and the CLI's telemetry will misreport the effective effort. |
| together | MiniMaxAI/MiniMax-M3 | Treating M3 as always-on (never sending a toggle) OR hardcoding context 1000000 | No API error, but two silent wrongs: (1) prior catalog said no toggle exists — Together's page says thinking IS toggleable at request time, so the CLI can't offer reasoning-off even though the provider supports it; (2) a 1M hardcoded context over-promises ~2x … |
| together | zai-org/GLM-5.2 | Tight "max_tokens" (e.g. 4096) combined with default reasoning_effort "max" | No HTTP error, but reasoning tokens consume max_tokens (billed as completion tokens), so the chain of thought exhausts the budget and the content field arrives empty/truncated — presents as "model returned no answer" in an agentic loop. |
| fireworks | non-reasoning models in the lineup (e.g. deepseek-v4-flash if it has no thinking mode) | reasoning_effort: "high" sent uniformly to every model in the preset | Unknown: silently-ignored vs 4xx is undocumented; prior refresh assumed 'optional everywhere' without a source |
| fireworks | all six (context accounting rather than reasoning) | Omitting context_length_exceeded_behavior while sending near-limit prompts | No error at all: default 'truncate' silently shrinks the effective completion/prompt window, corrupting Sylliptor's tier-aware context accounting; set 'error' to surface overflows |
| fireworks | reasoning models in multi-turn tool loops | Replaying assistant messages WITHOUT the reasoning_content field (standard OpenAI message shape) | No HTTP error — interleaved-thinking context silently lost, degraded agentic quality |

### medium (91)

| provider | model | wrong emission | result |
|---|---|---|---|
| openai + openai-responses | gpt-5.4-mini / gpt-5.4-nano / gpt-5.3-codex | reasoning effort 'max' (valid only on gpt-5.6 family) | HTTP 400 unsupported_value |
| openai + openai-responses | gpt-5.4-mini (or any non-5.6 model) | reasoning: {"mode": "pro"} on /v1/responses | expected HTTP 400 (mode documented for GPT-5.6 models only; exact error unconfirmed - probe filed) |
| openai + openai-responses | any gpt-5.6 tier | model slug 'gpt-5.6-pro', 'gpt-5.6-terra-pro', or 'gpt-5.6-luna-pro' (OpenRouter-style router-minted ids) sent to api.openai.com | HTTP 404/400 model_not_found - no such slug exists on the OpenAI API; pro is reasoning.mode:'pro' on the base slug (5.6-specific rule; 5.5 does have a real gpt-5.5-pro slug, so a generation-blind 'strip -pro' rewrite is also wrong) |
| openai + openai-responses | any reasoning model, unverified organization | reasoning: {"summary": "auto"} added by default (as LiteLLM did) | HTTP 400 for unverified orgs (org-verification-gated feature per BerriAI/litellm#16032) |
| anthropic (x3 presets, one surface: api.… | claude-sonnet-5 | omitting thinking + a max_tokens tuned for a non-thinking model (e.g. 1024) | No error — adaptive thinking runs by default, thinking tokens count toward max_tokens, response truncates with stop_reason "max_tokens" (possibly all-thinking, empty text) |
| anthropic (x3 presets, one surface: api.… | claude-haiku-4-5 | "output_config": {"effort": "low"} | Undocumented — haiku-4-5 is absent from the effort supported-models list; 400 vs silently-ignored unknown (probe listed) |
| anthropic (x3 presets, one surface: api.… | claude-haiku-4-5 | "thinking": {"type": "enabled", "budget_tokens": 10000} with "max_tokens": 8000 | HTTP 400 invalid_request_error — budget_tokens must be strictly less than max_tokens |
| anthropic (x3 presets, one surface: api.… | claude-fable-5, claude-opus-4-8, claude-opus-4-7 (+4.6 family) | prefilled last assistant message (e.g. forcing JSON with a trailing assistant turn) | HTTP 400: 'Prefilling assistant messages is not supported for this model.' (Sonnet 5 absent from the documented list — probe) |
| anthropic (x3 presets, one surface: api.… | claude-fable-5, claude-opus-4-8, claude-opus-4-7, claude-sonnet-5 | top-level "effort": "high" or "reasoning_effort": "high" instead of "output_config": {"effort": ...} | Unknown top-level parameter — expected 400 invalid_request_error (unrecognized field); at minimum effort is not applied |
| anthropic (x3 presets, one surface: api.… | claude-haiku-4-5 | catalog context field copied from sibling presets as 1000000 | 400 'prompt is too long' the moment a conversation exceeds 200k input tokens; tier-aware accounting overpromises 5x |
| gemini (x3 presets) | all 3.x (generateContent) | Emitting both thinkingConfig.thinkingLevel and legacy thinkingConfig.thinkingBudget in one request (e.g. merged config layers) | HTTP 400 (documented; note budget-alone is fine, it is the COMBINATION that 400s) |
| gemini (x3 presets) | gemini-3.1-pro-preview | thinkingLevel: "minimal" (valid on the flash family; documented 'Not supported' on Pro) | HTTP 400 INVALID_ARGUMENT expected — all three official pages agree minimal is 'Not supported' on 3.1 Pro (no doc conflict); live confirmation filed as belt-and-braces probe |
| gemini (x3 presets) | all 3.x via OpenAI-compat | reasoning_effort: "none" (works on 2.5-era models, cannot disable thinking on 3.x) | Undocumented: error or silent coercion; either way thinking is NOT disabled — user pays thinking tokens they asked to turn off |
| gemini (x3 presets) | gemini-3.5-flash | FunctionResponse parts without `id` and matching `name` (fine on 3-flash-preview, required at 3.5 GA) | Function-calling failure — per whats-new migration guide the model returns an empty response with finish_reason STOP (in most cases), not necessarily a hard 400 |
| gemini (x3 presets) | all 3.x | Sending snake_case generation_config.thinking_level in a generateContent body (Interactions-API spelling on the wrong surface) | Field unrecognized — at best ignored (model runs at its default level, silently wrong effort), at worst 400 depending on proto strictness |
| gemini (x3 presets) | gemini-3.1-pro-preview | Hardcoding flat pricing while filling the 1M window | No API error; input rate doubles ($2->$4/M) and output rises ($12->$18/M) past 200k prompt tokens — cost surprise, not a swap |
| deepseek | deepseek-v4-pro / deepseek-v4-flash (openai surface) | reasoning_effort="medium" (or "low"/"xhigh") -- a generic OpenAI-style effort value | Silently coerced: low/medium->high, xhigh->max. No error, but the agent's intended effort tier is not honored; behavior/cost differs from expectation. |
| deepseek | deepseek-v4-pro / deepseek-v4-flash (openai surface) | temperature / top_p / presence_penalty / frequency_penalty sent together with thinking enabled (default) | Silently ignored (no error, no effect). An agent relying on temperature=0 for determinism gets default sampling and non-deterministic output. |
| deepseek | deepseek-v4-pro / deepseek-v4-flash (openai surface) | reasoning_effort nested inside thinking, e.g. thinking={"type":"enabled","reasoning_effort":"max"} (the WebFetch-misread shape) | Nested reasoning_effort is an unknown sub-field -> effort silently defaults (high/auto) instead of max; likely no error but the effort request is dropped. If a strict validator rejects unknown body keys: possible 400/422. |
| qwen-intl / qwen-us / qwen-cn | qwen3-coder-plus / qwen3-coder-next | "enable_thinking": true | Undocumented — plausibly HTTP 400 {error:{code,message,type}} or silent ignore; coder models are absent from the thinking-models list |
| qwen-intl / qwen-us / qwen-cn | qwen-flash on qwen-us | model=qwen-flash against dashscope-us | Possible model-not-found: qwen-flash was absent from the retrieved US-region listing (unconfirmed, page partial) — probe queued |
| qwen-intl / qwen-us / qwen-cn | qwen3.7-* via Anthropic surface /apps/anthropic | "enable_thinking": true instead of thinking:{type:"enabled",budget_tokens:N} | Undocumented on that surface — likely stripped (thinking stays at model default) or 400; also omitting required max_tokens on this surface is a hard error |
| qwen-intl / qwen-us / qwen-cn | any thinking-enabled model | Small max_tokens combined with enable_thinking:true | Unknown truncation semantics on the OpenAI surface (docs do not say whether reasoning counts against max_tokens); on the Anthropic surface max_tokens limits only the final reply — divergent behavior between surfaces can truncate answers unexpectedly |
| zhipu | all | thinking: {"type": "auto"} or "thinking": true (boolean) — plausible normalizations of the enabled/disabled enum | Out-of-enum/of-type value: likely HTTP 400 code 1214 "参数非法" (invalid parameter); not explicitly documented |
| zhipu | all | max_tokens above 131072 (e.g. hardcoding 200000 because 'max output 128K' rounds up) | Documented parameter range is 1..131072; out-of-range outcome is NOT documented — likely HTTP 400 code 1214 but could be silent clamping (unverified, probe listed) |
| zhipu | glm-4.7 | Omitting thinking entirely while expecting a plain non-reasoning completion | Not an error, but forced thinking runs by default (thinking defaults to enabled and 4.7 thinks compulsorily) — latency and output-token cost surprise; clients that drop delta.reasoning_content may show long silent stalls |
| zhipu | glm-5.2 (xhigh/low/medium effort) | reasoning_effort: "medium" expecting mid-tier reasoning cost | Silent coercion: low/medium -> "high", xhigh -> "max" — more reasoning tokens billed than the requested tier implies |
| zhipu | glm-5.2, glm-5.1, glm-5-turbo, glm-4.7 family (tokenizer endpoint) | POST /paas/v4/tokenizer with a current-catalog model id (e.g. "glm-5.2") for pre-flight token counting | Documented supported roster stops at glm-4.6/4.6v/4.5 (intl) — a glm-5.x id may 400 (code 1211 unknown model) or be counted with a mismatched tokenizer; unverified either way |
| moonshot / moonshot-cn | kimi-k2.7-code | tool_choice: "required" (or a forced {type:function} choice) | HTTP 400 — 'tool_choice can only be set to 'auto' or 'none'' |
| moonshot / moonshot-cn | kimi-k2.6 | temperature: 0.6 while thinking enabled (or 1.0 while disabled) | Temperature is mode-coupled ('1.0 thinking / 0.6 non-thinking'); wrong pairing may error like other fixed-param violations (not explicitly documented) |
| moonshot / moonshot-cn | kimi-k3 (Tier0 accounts) | A near-full-context request based on the hardcoded 1,048,576 window | HTTP 429 rate_limit_reached_error — Tier0 TPM is 500,000, so the headline context can never fit through rate limiting until Tier1 ($10 recharge) |
| moonshot / moonshot-cn | any (anthropic compat surface) | kimi-k2.7-code request on https://api.moonshot.ai/anthropic without an explicit thinking block | 'rejected with a 400 invalid thinking error' — stricter than /v1 where omission defaults to enabled |
| kimi-code | kimi-for-coding | thinking: {"type": "disabled"} (attempting a speed/cost toggle) | No error — request is silently routed to K2.6, a different model; user believes they are on K2.7 Code |
| kimi-code | k3 | thinking disabled (any surface's disable spelling) | Silent routing to K2.6 — the flagship model is swapped out, not slowed down |
| kimi-code | k3 | model: "k3[1m]" in the request body (copying the Claude Code env spelling) | Invalid-model API error — the bracket form is exclusively a Claude Code ANTHROPIC_MODEL env value |
| kimi-code | kimi-for-coding-highspeed | any request from an Andante or Moderato membership key | Tier-gate API error (status/body undocumented — probe queued); model is Allegretto+ only |
| kimi-code | k3 (tool loops) | stripping reasoning_content from prior assistant messages before resending during a tool-call loop | Degraded/incorrect multi-step behavior; docs REQUIRE the complete assistant message passed back as-is including reasoning_content and tool_calls (error vs silent degradation undocumented) |
| minimax | MiniMax-M3 | thinking: {"type": "enabled"} or Claude-style thinking: {"type": "enabled", "budget_tokens": N} | Out-of-enum value (official enum is disabled\|adaptive). Undocumented failure mode: likely HTTP 4xx invalid-parameter or silent coercion; budget_tokens is not a MiniMax field and would be ignored or rejected |
| minimax | MiniMax-M3 | reasoning_effort: "high" (OpenAI o-series style) on the /v1 chat completions surface | MiniMax has no reasoning_effort field (it uses thinking); either silently ignored (no effort control applied) or rejected as unknown parameter — unconfirmed |
| minimax | MiniMax-M3 | max_tokens instead of max_completion_tokens on /v1, or max_completion_tokens > 524288 | max_tokens is deprecated on the openai route (may be ignored, leaving no output cap); a value above the 524,288 ceiling risks a 4xx or silent clamp |
| minimax | MiniMax-M3 | Assuming the full 1,000,000-token context is usable for every account/request and packing >512K input | Only 512K is guaranteed; 512K-1M is 'up to' and may be rejected on some accounts, and any >512K input is silently billed at 2x the standard input/output rate |
| bytedance | all four catalog models | "reasoning": {"effort": "high"} (OpenAI Responses-style nested object) or "thinking": {"type": "enable"} (typo'd value) | 400 InvalidParameter/BadRequest body shape (same family as observed errors); no silent coercion documented |
| bytedance | doubao-seed-2-0-* models that lack "auto" support (per-model support matrix unverified) | "thinking": {"type": "auto"} | 400 InvalidParameter on models accepting only enabled/disabled (older Ark models like deepseek-v3.1 documented as enabled/disabled only); probe required per seed-2.0 model |
| bytedance | all four catalog models | both "max_tokens" and "max_completion_tokens" in one request | documented (LAS platform) as mutually exclusive — expected 400 InvalidParameter on Ark; unprobed |
| bytedance | doubao-seed-2-0-pro-260215 (and siblings) with thinking omitted | omitting thinking + setting a small max_tokens/max_completion_tokens assuming no reasoning happens | silent quality failure not HTTP error: deep thinking is enabled by default (per official Seed-2.0 LAS doc), reasoning tokens consume the max_completion_tokens budget, yielding truncated or empty final answers with finish_reason=length |
| groq | openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.6-27b | reasoning_format: "raw" explicitly set while also sending tools or response_format json_object | documented 400 error (raw is incompatible with tool use / JSON mode; must be parsed or hidden). Note: the implicit default auto-switches to parsed (not parsed/hidden) under tools/JSON mode, so only an explicit raw triggers this |
| groq | any (all four catalog models) | include_reasoning and reasoning_format in the same request | documented mutually exclusive — request rejected |
| groq | any (all four catalog models) | messages[].name, logprobs, logit_bias, top_logprobs, or n > 1 (standard OpenAI-compat fields) | documented 400: logprobs/logit_bias/top_logprobs/messages[].name per the openai-compat page's will-400 list; n>1 per the api-reference n-param description ('Other values will result in a 400 response') |
| groq | any (all four catalog models) | frequency_penalty or presence_penalty (OpenAI-compat client defaults, often emitted as 0 or small non-zero values) | api-reference: 'not yet supported by any of our models' for both; NOT in the openai-compat page's 400 list — reject-vs-silently-ignore undocumented, probe queued |
| groq | openai/gpt-oss-120b | reasoning_effort: "none" (attempting to disable reasoning on gpt-oss) | 'none' not in the gpt-oss enum (low\|medium\|high); behavior undocumented — expected 400; gpt-oss reasoning has no documented off switch (default is medium when omitted) |
| groq | qwen/qwen3.6-27b | max_completion_tokens: 65536 (copying the gpt-oss cap) | exceeds this model's 32,768 max output; rejection vs clamping undocumented |
| groq | openai/gpt-oss-120b (any reasoning model) | no max_completion_tokens at all on the reasoning path | not an error — silent truncation risk: reasoning docs note a 1,024-token default completion budget, far below agentic needs (and default effort medium spends reasoning tokens against it if accounting is shared — probe) |
| cerebras | zai-glm-4.7 | "reasoning_effort": "low" or "high" (graduated effort assumed) | API reference lists only "none" as supported on GLM; if graduated values are unsupported the request may 400 or be silently ignored — undocumented (probe listed). |
| cerebras | gemma-4-31b | "reasoning_format": "raw" or "hidden" | Docs: "The raw and hidden reasoning formats are not supported" on Gemma 4 31B — likely HTTP error or ignored; undocumented shape. |
| cerebras | zai-glm-4.7 (clear_thinking on other models) | "clear_thinking": false sent to gpt-oss-120b or gemma-4-31b | Param is documented for zai-glm-4.7 only; behavior on other models undocumented (400 or ignored). Also requires extra_body in the OpenAI SDK — sending it top-level via a strict OpenAI client fails client-side. |
| mistral | mistral-medium-2604 / mistral-small-2603 | reasoning_effort: "medium" or "low" (OpenAI-style effort values) | Undefined: only "high" and "none" have documented semantics; API-ref enum lists minimal/low/medium/xhigh but no page defines what they do — could be accepted with unknown behavior or rejected |
| mistral | mistral-large-2512, codestral-2508, ministral-8b-2512 | reasoning_effort: "high" sent to a non-reasoning model | Unknown: silently ignored vs HTTP 4xx undocumented (probe queued); prior refresh marked these reasoning:none so a blanket per-provider flag would emit this |
| mistral | mistral-medium-2604 | prompt_mode: "reasoning" (carryover from the retired magistral family; still present in the API reference) | Unknown on hybrid models — magistral-era code paths that kept prompt_mode will emit it; behavior on Medium 3.5 undocumented (probe queued) |
| mistral | codestral-2508 | Prompt sized against a hardcoded 262144-token context (copied from sibling cards) | Over-length request error: official card says 128k, half the assumed window |
| mistral | any -latest alias (e.g. mistral-small-latest, magistral-small-latest) | Pinning behavior expectations to a -latest alias | Silent model swap over time: aliases re-point across families (search lead: magistral-small-latest now resolves to Mistral Small 4); reasoning/context assumptions silently invalidated |
| xai | grok-4.5 or grok-4.3 | reasoning_effort: "xhigh" (valid only on grok-4.20-multi-agent) | undocumented rejection expected; xhigh is documented for the multi-agent model only |
| xai | grok-4.5 (via alias) | model: "grok-build-latest" intending grok-build-0.1 | silent model swap to grok-4.5 at ~2x the price (docs list grok-build-latest as a grok-4.5 alias) |
| xai | grok-4.3 / grok-build-0.1 (via retired slugs) | model: grok-4-fast-reasoning, grok-3, grok-code-fast-1, etc. after 2026-05-15 12:00 PM PT | silent redirect: reasoning slugs → grok-4.3 effort low, non-reasoning → grok-4.3 effort none, grok-code-fast-1 → grok-build-0.1; billed at target-model rates |
| xai | any (Chat Completions surface) | nested reasoning:{"effort": ...} object sent to /v1/chat/completions | unknown — this shape is documented only on /v1/responses; chat/completions documents no nested reasoning object (probe queued). NOTE the reverse direction is NOT a hazard: /v1/responses documents top-level reasoning_effort as a first-class compatibility fallba… |
| xai | grok-4.20-0309-reasoning (any reasoning model on Responses surface) | treating max_output_tokens as visible-output-only (chat semantics) | not an API error: reasoning tokens consume the cap, so long thinking silently truncates visible output (finish via length) |
| xai | grok-build-0.1 | reasoning_effort: "low" | unknown — no docs; chat reference's 'only grok-4.3' phrasing suggests rejection (probe queued) |
| xai | any (caching path) | setting an x-grok-conv-id request header instead of the prompt_cache_key body parameter (the original contract's advice) | not an API error: header is undocumented as client-settable — expected silent cache misses / no sticky routing while the documented body param goes unsent |
| cohere | command-a-reasoning-08-2025 | native thinking object ({"type":"disabled"} or with token_budget) sent to /compatibility/v1 | thinking is not in the compat supported-parameter list; behavior (dropped vs 400 {"message","id"}) undocumented — if dropped, the model silently stays thinking-on and burns output budget |
| cohere | command-a-03-2025, command-r7b-12-2024 | reasoning_effort (any value) over compat, or thinking over native V2, sent to these non-reasoning models | undocumented; native V2 errors return {"message":"...","id":"..."} with 400/422 if rejected — possibly a hard error rather than degradation |
| cohere | command-a-plus-05-2026 | hardcoding 256K context copied from the Command A family | context overflow errors on long agentic sessions — command-a-plus-05-2026 is 128K input / 64K output, HALF the context of command-a-reasoning/command-a-03-2025 |
| cohere | all | GET /compatibility/v1/models for live discovery | no /models endpoint is documented on the compat surface; listing lives only at GET https://api.cohere.com/v1/models on the native host with bearer auth |
| openrouter | z-ai/glm-5.2 (and any model priced by transcription rather than live read) | Cost accounting from a transcribed rate (the prior contract's $1.12/$3.52/M) instead of the live listed pricing string ($0.9786/$3.0756/M) | Silent cost-accounting error: ~14.4% over-estimate on glm-5.2 spend (and, more generally, any per-token rate that drifts from the live pricing field); not an API failure but wrong billing/quota math on long agentic transcripts. Mitigation: read pricing from li… |
| openrouter | openai/gpt-5.6-terra, openai/gpt-5.6-luna | reasoning: {"mode": "pro"} copied from another preset/config | Documented silent substitution: OpenRouter reroutes the request to the matching *-pro model (e.g. gpt-5.6-luna-pro) — different model and higher price billed under a request that named the base slug |
| openrouter | anthropic/claude-sonnet-5, anthropic/claude-opus-4.8 | reasoning: {"effort": "none"} to disable thinking (valid on the GPT-5.6 family, absent from Anthropic supported_efforts) | Not a supported effort for these models — likely 400 or ignored (probe filed); the reliable disable path is omitting reasoning entirely, and reasoning-tokens docs warn never to send effort:"none" to models with mandatory:true |
| openrouter | all catalog models | reasoning: {"effort": "high", "max_tokens": 2000} (both subfields together) | Docs state effort and max_tokens are mutually exclusive ("One of the following (not both)") — 400 or undocumented precedence; either way the requested reasoning level is not what runs |
| openrouter | any model via /api/v1/responses (beta surface) | The chat-completions unified object — reasoning.max_tokens / exclude / enabled, or effort values "max"/"xhigh"/"none" | Responses API accepts only reasoning:{effort} with "minimal"\|"low"\|"medium"\|"high"; other fields/values are not part of the surface — likely 400 (surface also hard-rejects store:true and previous_response_id with 400) |
| openrouter | anthropic/* (older configs) | Model slug with the ":thinking" variant suffix (e.g. anthropic/claude-sonnet-5:thinking) | ":thinking" is no longer supported for Anthropic models — 404 "model does not exist" instead of a reasoning-enabled request |
| openrouter | openai/gpt-5.6-terra, openai/gpt-5.6-luna | Treating pricing as flat when prompts exceed 272,000 tokens | Not an API error but a billing surprise: pricing.overrides doubles prompt cost ($2.50->$5.00 terra, $1->$2 luna) and raises completion cost past min_prompt_tokens 272,000 — cost accounting silently wrong on long agentic contexts |
| perplexity | google/gemini-3.1-flash-lite, nvidia/nemotron-3-super-120b-a12b, perplexity/kimi-k2.7-code | reasoning: {"effort": ...} sent to a model whose reasoning support is unpublished | unknown: possible 400, possible silent ignore (user believes reasoning is on when it is not) |
| perplexity | any | invalid effort value string (e.g. "auto", "none", "maximal") — note valid set is minimal\|low\|medium\|high\|xhigh\|max, which does NOT match other providers' sets | unknown, likely 400 ValidationError with {"error":{message,type,code}} body |
| perplexity | any (via preset) | sending "preset" (or preset alongside model) expecting the named model to serve | silent model selection drift: presets pin models internally (preset "low" = google/gemini-3-flash-preview) and auto-update over time; precedence over an explicit model field is undocumented |
| perplexity | any (via models[] array) | passing a models fallback array assuming it is advisory | automatic mandatory failover to a different model — no opt-out; substitution only visible post-hoc in response.model |
| perplexity | anthropic/* under agentic use | small max_output_tokens (sent only to satisfy the anthropic requirement) combined with high reasoning effort | if reasoning tokens count against max_output_tokens (undocumented), responses come back status "incomplete" with truncated/empty visible text |
| together | moonshotai/Kimi-K2.7-Code | "reasoning_effort": "high" (or any effort value) | Unknown — no effort control documented for this model; either 400 with OpenAI-shaped error body or silent no-op. Probe specced. |
| together | openai/gpt-oss-120b | "reasoning_effort": "max" or "xhigh" (valid on DeepSeek-V4-Pro/GLM-5.2, so plausible to generalize) | Not in the documented low\|medium\|high set for gpt-oss; reference schema declares a closed enum, so a 400 invalid_request-style error is plausible; silent-drop also possible. Unknown pending probe. |
| together | openai/gpt-oss-20b | "reasoning": {"enabled": false} (attempting to disable reasoning entirely) | Unknown — gpt-oss is adjustable-effort, not hybrid; no off switch documented. Either 400 or ignored-with-reasoning-anyway. |
| together | any (provider-wide) | Anthropic-style "thinking": {"type": "enabled", "budget_tokens": N} or OpenAI Responses-style "reasoning": {"effort": "high"} | Together's "reasoning" object has exactly one documented key, "enabled" (boolean) — a nested {"effort": ...} shape or a top-level "thinking" object is an unknown shape; behavior undocumented (400 vs drop), and if dropped the intended effort silently does not a… |
| fireworks | any reasoning-capable catalog model (e.g. accounts/fireworks/models/kimi-k2p7-code) | Request body containing BOTH "thinking": {...} and "reasoning_effort": "..." | Documented validation error (request rejected); exact status/body unpublished |
| fireworks | all six catalog models | reasoning_effort values beyond low/medium/high (e.g. "xhigh", "minimal", "none", integer budgets) — plausible because the API-reference schema advertises them but the reasoning guide does not | Unknown: either accepted per schema or 4xx validation error per guide; divergence itself is the hazard (works on one model, 400s on another) |
| fireworks | any model via the anthropic-messages surface (/inference/v1/messages) | thinking: {"type": "adaptive"} or output_config.speed or inference_geo (valid on real Anthropic API) | Unsupported on Fireworks' Anthropic-compat surface — request error or dropped feature; docs list these as explicitly unsupported |
| fireworks | any model via /inference/v1/responses | reasoning: {"effort": "low"} (OpenAI Responses spelling) or reasoning_effort on the Responses surface | Unknown — reasoning params are entirely undocumented on this surface; may 400 or be silently ignored (silent ignore = paying for hidden default reasoning) |

### low (19)

| provider | model | wrong emission | result |
|---|---|---|---|
| openai + openai-responses | any reasoning model, streaming consumer | stream parser hardcoded to a 'reasoning_text' content-part/event name for raw reasoning text | silent: reasoning deltas dropped or parser falls into an unknown-event path if the real event name differs (name unconfirmed - reasoning guide never states it and the streaming-events reference truncated; probe filed) |
| openai + openai-responses | gpt-5.6 alias 'gpt-5.6' | emitting alias gpt-5.6 while cost-accounting as terra or luna | no error - alias silently routes to gpt-5.6-sol (2x-5x the per-token price of terra/luna) |
| anthropic (x3 presets, one surface: api.… | claude-fable-5 | "thinking": {"type": "disabled", "display": "omitted"} or display combined with disabled on any model | Invalid — 'display is invalid with thinking.type: "disabled"' (400) |
| anthropic (x3 presets, one surface: api.… | claude-opus-4-7 | "speed": "fast" (after 2026-07-24) | Fast mode on Opus 4.7 is deprecated, 'will be removed on July 24, 2026' — 5 days from checked date; on claude-opus-4-6 the same flag already silently runs at standard speed (no error, no speedup) |
| gemini (x3 presets) | gemini-3.1-flash-lite-image (nearby-id trap) | Reusing the plain gemini-3.1-flash-lite level set {minimal,low,medium,high} for the -image id, which only supports {minimal, high} | thinkingLevel low/medium on gemini-3.1-flash-lite-image likely 400 INVALID_ARGUMENT — the -image variant is a distinct id with a restricted level set |
| deepseek | deepseek-v4-pro / deepseek-v4-flash (openai surface) | OpenAI Responses-style reasoning={"effort":"high"} object instead of top-level reasoning_effort | Unknown field; effort request dropped (silent) or 400 Invalid Format if unknown keys are rejected (undocumented -- see probe). |
| qwen-intl / qwen-us / qwen-cn | open-source thinking models if ever added (qwen3-235b-a22b, qwen3-32b) | Non-streaming call with thinking enabled | HTTP error (code unpublished): these models are streaming-only per the deep-thinking page |
| moonshot / moonshot-cn | kimi-k2.7-code(-highspeed) | "thinking": {"type": "enabled", "keep": null} templated from kimi-k2.6's default (or keep omitted) | NO HTTP error — CORRECTED: null and omission are coerced to "all" server-side ('passing "all", passing null, or omitting it all behave identically'). The failure is semantic: k2.6's null means historical thinking NOT preserved, k2.7-code silently flips it to P… |
| moonshot / moonshot-cn | kimi-k2.7-code (documented) / kimi-k3 (fixed, error unprobed) | n: 2 (multi-sample) or presence_penalty/frequency_penalty != 0 | k2.7-code: HTTP 400 — n fixed at 1, penalties fixed at 0.0, 'Any other value will result in an error'. k3: fixed per quickstart, on-send behavior undocumented — omit. |
| kimi-code | kimi-for-coding | thinking: {"keep": "none"} or any keep value other than "all" | API error ("any other invalid value errors"); K2.7 Code treats keep as always "all" |
| minimax | MiniMax-M2.7 / MiniMax-M2.7-highspeed / MiniMax-M2.5 | thinking: {"type": "disabled"} sent to an M2.x model to save latency/cost | No effect (thinking is always-on for M2.x regardless of the param) — the agent believes reasoning is off and mis-budgets output tokens/latency; no error surfaced |
| minimax | MiniMax-M3 | reasoning_split (openai-only output-format flag) sent on the /anthropic messages surface | Not a valid anthropic-surface field; ignored or rejected — and if the client then looks for reasoning_content it will not find the thinking, which lives in anthropic-style thinking content blocks instead |
| groq | any via Responses API surface | previous_response_id / store / truncation / include / prompt_cache_key on /openai/v1/responses | documented as not supported by Groq's Responses API (exact error shape unpublished) |
| cerebras | all (gpt-oss-120b, zai-glm-4.7, gemma-4-31b) | "reasoning_format": "raw" combined with response_format json_object/json_schema | Documented incompatibility; models defaulting to raw are auto-coerced to hidden, but an explicit raw + JSON mode request is an error/undefined. |
| cerebras | gpt-oss-120b | "min_tokens" set on requests | Documented: "When min_tokens is set, the model may generate EOS tokens which may cause parser failures" — degraded/broken parsing, not an HTTP error. |
| xai | any (Responses API surface) | sending BOTH reasoning:{effort} and top-level reasoning_effort to /v1/responses with different values | not an error: reasoning_effort is silently ignored ('We only look at this if the reasoning field is unset') — the effective effort is the nested one, no warning |
| openrouter | any reasoning model with provider.require_parameters:true | reasoning param + restrictive provider prefs (only/ignore/zdr/data_collection:deny) that exclude all reasoning-supporting endpoints | http-503 {"error":{"code":503,"message":"There is no available model provider that meets your routing requirements"}} |
| perplexity | sonar surface models (if Sylliptor ever routes there) | tool definitions array on POST /v1/sonar or its /chat/completions alias | custom tools cannot be registered per docs; exact rejection behavior (400 vs silent ignore) undocumented — probe #4 |
| fireworks | accounts/fireworks/models/kimi-k2p7-code (and any always-on thinker) | thinking: {"type": "enabled", "budget_tokens": 512} (below documented minimum 1024) | Validation error (budget_tokens minimum 1024); body shape unpublished |

---

# Part D — Perplexity Agent API client scoping

## Perplexity Agent API vs OpenAI Responses — divergence summary (verified 2026-07-19)

**Verdict for Sylliptor:** fork the OpenAI client, don't reuse it blind. The wire shape is Responses-flavored (flat function tools, `function_call`/`function_call_output`, `previous_response_id`, `response.output_text.delta`), but structured output uses Chat-Completions-style `response_format`, `tool_choice`/`parallel_tool_calls` are absent from the request schema, and `anthropic/*` models hard-require `max_output_tokens` (HTTP 400 otherwise).

### 1. Endpoint, auth, versioning
- Base `https://api.perplexity.ai/v1`; canonical `POST /v1/agent`; `POST /v1/responses` is an accepted alias so OpenAI SDKs work by swapping base_url + key ([openai-compatibility](https://docs.perplexity.ai/docs/agent-api/openai-compatibility)).
- `Authorization: Bearer $PERPLEXITY_API_KEY`; no versioning header documented ([quickstart](https://docs.perplexity.ai/docs/agent-api/quickstart)). Retrieval is `GET /v1/agent/{id}`; stream resume `GET /v1/agent/{id}?stream=true&starting_after=N` ([output-control](https://docs.perplexity.ai/docs/agent-api/output-control), [conversation-state](https://docs.perplexity.ai/docs/agent-api/conversation-state)).

### 2. Request schema ([agent-post](https://docs.perplexity.ai/api-reference/agent-post))
Present & OpenAI-shaped: `input` (string | item array), `model` (**must be `provider/model`**, e.g. `openai/gpt-5.1`, `anthropic/claude-sonnet-5`), `instructions`, `max_output_tokens`, `temperature` (0–2), `top_p`, `reasoning` (effort: `minimal|low|medium|high|xhigh|max`), `tools`, `stream`, `background`, `previous_response_id`, `store`.
Perplexity-only extras: `models` (fallback chain, max 5), `preset` (`fast|low|medium|high|xhigh` → curated model+tools; passable via OpenAI SDK `extra_body`), `max_steps` (1–100 research-loop cap), `skills` (max 16), `language_preference` (ISO 639-1).
Renamed: OpenAI's `text.format` → top-level `response_format: {type:"json_schema", json_schema:{name, schema}}` (name 1–64 alnum; first use of a new schema adds 10–30 s prep).
Absent (no docs): `tool_choice`, `parallel_tool_calls`, `include`, `truncation`, `metadata`, `service_tier`, `top_logprobs`, `stream_options`, `user`/`safety_identifier`, `prompt_cache_key`, `conversation`.

### 3. Response schema
`{id, object:"response", created_at, status: completed|failed|incomplete|in_progress|queued|cancelled, model, output[], usage, error}`. `output` item types: `message` (role assistant, content[]), `function_call` `{id, name, call_id, arguments:string, status}`, plus Perplexity-only `search_results`, `fetch_url_results`, `finance_results`, `people_search_results`, `sandbox_results`, `mcp_list_tools`, `mcp_call`. **No `reasoning` output item documented.** Usage: `input_tokens/output_tokens/total_tokens` match OpenAI, but details are Anthropic-style (`cache_creation_input_tokens`, `cache_read_input_tokens`) plus a `cost` object (`total_cost`, `currency:"USD"`, per-bucket costs) and `tool_calls_details` ([agent-post](https://docs.perplexity.ai/api-reference/agent-post)).

### 4. Tools ([tools](https://docs.perplexity.ai/docs/agent-api/tools), [function-calling](https://docs.perplexity.ai/docs/agent-api/tools/function-calling))
- Function defs are **flat, Responses-style**: `{type:"function", name, description, parameters}` (no nested `function` object; no `strict` documented). `arguments` is a JSON string.
- Follow-up: append `{type:"function_call_output", call_id, output}` to `input` along with prior items — docs show **full-history resend**; combining with `previous_response_id` is not addressed.
- Parallel: model may emit several `function_call` items in one response; execute all, return all outputs together. No `parallel_tool_calls` toggle.
- Built-ins (unique surface): `web_search` (with `search_context_size`, `filters` incl. `search_domain_filter` ≤20, recency/date filters, `user_location`, `max_results` 1–50; $5/1k calls), `fetch_url`, `finance_search`, `people_search`, `sandbox` (code exec), `mcp`.

### 5. Streaming (SSE, `text/event-stream`)
Shared with OpenAI: `response.created|in_progress|completed|failed`, `response.output_item.added|done`, `response.output_text.delta|done`; every event carries `sequence_number`. Perplexity-only: `response.reasoning.started|search_queries|search_results|fetch_url_queries|fetch_url_results|stopped`. **Not documented:** `response.content_part.*`, `response.function_call_arguments.delta`, annotation events, `[DONE]` sentinel ([agent-post](https://docs.perplexity.ai/api-reference/agent-post), [output-control](https://docs.perplexity.ai/docs/agent-api/output-control)).

### 6. Client special cases
- **`anthropic/*` ⇒ `max_output_tokens` required or 400** ([models](https://docs.perplexity.ai/docs/agent-api/models)).
- `store:false` hides from `GET /v1/agent/{id}` (404) but the id **still works** as `previous_response_id`; `previous_response_id` needs status `completed` else 400 ([conversation-state](https://docs.perplexity.ai/docs/agent-api/conversation-state)).
- `background:true` → status `queued`, poll retrieve; resume streams via `starting_after`.
- Output parser must tolerate interleaved non-message items (`search_results` etc.) even when the client sent no tools (presets inject them).

### Divergence table
| OpenAI Responses feature | Perplexity status |
|---|---|
| `POST /v1/responses` | same (alias of `/v1/agent`) |
| `input` / `instructions` / `previous_response_id` / `store` / `stream` / `background` | same (store semantics differ slightly) |
| `model` ids | renamed (`provider/model`) |
| `max_output_tokens` | same, but **required** for `anthropic/*` |
| `text.format` (structured output) | renamed → `response_format.json_schema` |
| `reasoning.effort` | same field, extra values (`xhigh`, `max`) |
| Flat function tools, `function_call(_output)`, `call_id` | same |
| `tool_choice`, `parallel_tool_calls`, `include`, `truncation`, `metadata`, `stream_options`, `top_logprobs` | absent |
| `web_search` built-in | extra/divergent config (`filters`, `max_results`, pricing) |
| `fetch_url`, `finance_search`, `people_search`, `sandbox`, `mcp` built-ins; `preset`, `models` fallback, `max_steps`, `skills` | extra |
| `response.output_text.delta` etc. | same |
| `response.function_call_arguments.delta`, `content_part.*` | absent/undocumented |
| `response.reasoning.*` SSE family; `usage.cost` | extra |
| `GET /v1/responses/{id}` | unverified (documented as `GET /v1/agent/{id}`) |

### Sources
- https://docs.perplexity.ai/docs/agent-api/openai-compatibility
- https://docs.perplexity.ai/api-reference/agent-post
- https://docs.perplexity.ai/docs/agent-api/models
- https://docs.perplexity.ai/docs/agent-api/tools
- https://docs.perplexity.ai/docs/agent-api/quickstart
- https://docs.perplexity.ai/docs/agent-api/tools/function-calling
- https://docs.perplexity.ai/docs/agent-api/conversation-state
- https://docs.perplexity.ai/docs/agent-api/output-control

### Probes
- Is tool_choice accepted (auto/none/required/named function), rejected with 400, or silently ignored? It is absent from the documented request schema but a cookbook mentions an 'auto' default.
  `curl -s https://api.perplexity.ai/v1/responses -H "Authorization: Bearer $PERPLEXITY_API_KEY" -H 'Content-Type: application/json' -d '{"model":"openai/gpt-5-mini","input":"What is 2+2? Use the calc tool.","tools":[{"type":"function","name":"calc","description":"evaluate math","parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}}],"tool_choice":{"type":"function","name":"calc"}}'`
- Do streamed function calls emit response.function_call_arguments.delta (OpenAI-style) or only arrive whole inside response.output_item.added/done?
  `curl -sN https://api.perplexity.ai/v1/responses -H "Authorization: Bearer $PERPLEXITY_API_KEY" -H 'Content-Type: application/json' -d '{"model":"openai/gpt-5-mini","input":"Call calc on 17*23","stream":true,"tools":[{"type":"function","name":"calc","parameters":{"type":"object","properties":{"expr":{"type":"string"}}}}]}' | grep -E '"type"|^event|DONE' | sort -u`
- Can previous_response_id be combined with input containing only function_call_output items (OpenAI supports this; Perplexity docs show full-history resend only)?
  `curl -s https://api.perplexity.ai/v1/responses -H "Authorization: Bearer $PERPLEXITY_API_KEY" -H 'Content-Type: application/json' -d '{"model":"openai/gpt-5-mini","previous_response_id":"<RESP_ID_WITH_PENDING_FUNCTION_CALL>","input":[{"type":"function_call_output","call_id":"<CALL_ID>","output":"{\"result\":391}"}]}'`
- Does the OpenAI-alias retrieval path GET /v1/responses/{id} exist, or only GET /v1/agent/{id} (matters for openai-sdk responses.retrieve/cancel)? Also confirm the cancel route hinted in llms-full.txt.
  `RID=$(curl -s https://api.perplexity.ai/v1/responses -H "Authorization: Bearer $PERPLEXITY_API_KEY" -H 'Content-Type: application/json' -d '{"model":"openai/gpt-5-mini","input":"hi","background":true}' | jq -r .id); curl -s -o /dev/null -w 'responses/{id}: %{http_code}\n' https://api.perplexity.ai/v1/responses/$RID -H "Authorization: Bearer $PERPLEXITY_API_KEY"; curl -s -o /dev/null -w 'agent/{id}/cancel: %{http_code}\n' -X POST https://api.perplexity.ai/v1/agent/$RID/cancel -H "Authorization: Bearer $PERPLEXITY_API_KEY"`
- Does the SSE stream terminate with a data: [DONE] sentinel after response.completed (OpenAI Responses does not use [DONE]; Chat Completions does — parsers must know which)?
  `curl -sN https://api.perplexity.ai/v1/responses -H "Authorization: Bearer $PERPLEXITY_API_KEY" -H 'Content-Type: application/json' -d '{"model":"openai/gpt-5-mini","input":"say hi","stream":true}' | tail -5`
- Are unknown OpenAI params (metadata, truncation, include, parallel_tool_calls) rejected with 400 or silently dropped? Determines whether the client must strip them before proxying.
  `curl -s https://api.perplexity.ai/v1/responses -H "Authorization: Bearer $PERPLEXITY_API_KEY" -H 'Content-Type: application/json' -d '{"model":"openai/gpt-5-mini","input":"hi","metadata":{"k":"v"},"truncation":"auto","parallel_tool_calls":false,"include":["output[*].results"]}'`

---

# Recommended implementation order

The ordering principle: fix what errors on default settings first, then what silently lies, then
what improves over time. Every step is model/provider-agnostic in mechanism (per the standing
mandate) — the *data* is per-provider, the *code paths* are not.

**1. Ship the `reasoning_mode` field + wire-shape table (Part A data) as static preset data.**
Four values per model: `always-on | optional | none | unknown`, plus three facts the mode alone
cannot carry: the wire spelling (`reasoning_effort` vs `thinking:{type}` vs `thinkingConfig.thinkingLevel`
vs `reasoning:{...}` vs `output_config:{effort}`), the exact allowed value strings, and the
off-path semantics (`omit-param | explicit-disable | impossible | swaps-model`). Nothing at runtime
can discover these (OpenRouter excepted) — this table IS the product of this research. Emission
must become **allowlist-based**: Sylliptor sends a reasoning parameter only when the table says the
model accepts that exact spelling and value; `unknown` means send nothing. That single rule
neutralizes hazard ranks 5, 8, 9, 12, 16, 18 at once.

**2. Fix the three protocol-level breakages that fire on defaults.**
(a) openai chat preset: tools + effort≠none is a 400 — either pin `reasoning_effort:"none"` on the
chat preset or make `openai-responses` the tool-loop default and demote chat to fallback.
(b) anthropic client: emit `thinking:{"type":"adaptive"}` + `output_config:{"effort":...}` for the
adaptive four, keep `{"type":"enabled","budget_tokens":≥1024}` only for haiku-4-5; stop sending
temperature/top_p/top_k to the adaptive four.
(c) reasoning-continuity replay: preserve provider-returned reasoning artifacts verbatim when
rebuilding history — gemini `thoughtSignature` (including through the OpenAI-compat path),
moonshot/kimi/fireworks `reasoning_content`, anthropic `thinking`/`redacted_thinking` blocks,
mistral ThinkChunks, openrouter `reasoning_details`. One generic rule: *never re-serialize an
assistant message; echo it back byte-identical plus the tool result.*

**3. Context layer with a tier dimension (Part C data).**
`context` must be `(model, account-tier) → tokens`, not `model → tokens`, with an
`overpromise_risk` bit that the HUD surfaces as a range or conservative floor when the tier is
unknown. Apply the two catalog corrections (codestral 128K; cohere plus-64K-output). Wire the three
real token-counting endpoints (anthropic `count_tokens`, gemini `countTokens`, moonshot
`estimate-token-count`) and re-baseline anthropic accounting for the ~30% tokenizer shift instead
of scaling stored figures. Set fireworks `context_length_exceeded_behavior:"error"`.

**4. Live discovery, two waves (Part B data).**
Wave 1 — the two providers where live data is authoritative: **openrouter** (no-auth `/api/v1/models`:
context, pricing, `supported_parameters`, `supported_efforts`/`default_effort`; honor the 300s
Cache-Control; lazily hit `/models/{slug}/endpoints` for worst-case context across routes) and
**anthropic** (`/v1/models` carries token limits + thinking/effort capability flags). Wave 2 —
augment-static existence/context checks on config-open with a daily cache: groq, cerebras, mistral,
together, xai, moonshot, gemini, minimax, deepseek, fireworks (control-plane, per-account
serverless flags), openai + perplexity (ids-only roster pings). Never: kimi-code, zhipu, qwen,
bytedance, cohere-compat — their blocks say why, and each carries the probe that could upgrade it.

**5. Run the probe backlog with real keys.**
Every `unknown` above is backed by a one-paste curl in its provider block. Highest-value first:
bytedance per-model thinking acceptance (the whole preset is probe-gated), fireworks per-model A4
classification (5 of 6 unknown), together M3 toggle spelling, minimax `thinking` enum + 400 body,
mistral mid-effort values, kimi-code anthropic-surface mapping, anthropic haiku effort failure
body. A maintainer afternoon with six keys converts most `unknown`s into table rows.

**6. Scope the Perplexity Responses client (Part D).**
The divergence summary below is the input. Decision gate: without a Responses-style client the
perplexity preset stays search-only; with one, six vendor-prefixed models open up — but context
accounting stays blind there (nothing published), so ship it with explicit "context unknown"
handling or don't ship it.

**7. Fold `reasoning_mode` into doctor --live.**
The reasoning-off probe must assert per the table (`impossible` models assert effort-floor instead
of off; `swaps-model` models assert the substitution warning fires), not a uniform "reasoning off
works" check — that assumption is exactly what this research falsified.
