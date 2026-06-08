#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+'
    printf ' %s' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run "$PYTHON_BIN" -m pytest \
  tests/test_llm_protocols.py \
  tests/test_llm_factory_profiles.py \
  tests/test_profile_presets.py \
  tests/test_profiles.py \
  tests/test_openai_responses.py \
  tests/test_anthropic_messages.py \
  tests/test_gemini_generate_content.py \
  tests/test_native_provider_conformance.py \
  tests/test_provider_diagnostics.py \
  -q "$@"
