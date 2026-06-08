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
  tests/test_web_search_tool.py \
  tests/test_web_search_tavily.py \
  tests/test_web_search_dashscope.py \
  tests/test_provider_diagnostics.py \
  -q "$@"
