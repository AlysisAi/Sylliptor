#!/bin/sh

# Install Sylliptor inside an isolated Terminal-Bench or Harbor task container.
# Provider credentials are intentionally not read or written during setup.

set -eu

SETUP_LOG_DIR="${SYLLIPTOR_SETUP_LOG_DIR:-/logs/agent/setup}"
SETUP_ARTIFACT_DIR="${SYLLIPTOR_SETUP_ARTIFACT_DIR:-/logs/artifacts/setup}"
SETUP_LOG="$SETUP_LOG_DIR/install.log"

mkdir -p "$SETUP_LOG_DIR" "$SETUP_ARTIFACT_DIR"

if [ "${SYLLIPTOR_SETUP_LOG_ACTIVE:-0}" != "1" ]; then
  export SYLLIPTOR_SETUP_LOG_ACTIVE=1
  status_file="$(mktemp "${TMPDIR:-/tmp}/sylliptor-setup-status.XXXXXX")"
  set +e
  (
    "$0" "$@"
    status="$?"
    printf '%s\n' "$status" >"$status_file"
    exit "$status"
  ) 2>&1 | tee -a "$SETUP_LOG" "$SETUP_ARTIFACT_DIR/install.log"
  if [ -s "$status_file" ]; then
    status="$(cat "$status_file")"
    rm -f "$status_file"
  else
    status=1
  fi
  exit "$status"
fi

export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:/opt/sylliptor-venv/bin:$PATH"

retry() {
  label="$1"
  shift
  for attempt in 1 2 3; do
    echo "setup_step label=$label attempt=$attempt started_at=$(date -u +%FT%TZ)"
    if "$@"; then
      echo "setup_step label=$label attempt=$attempt status=ok"
      return 0
    else
      status="$?"
    fi
    echo "setup_step label=$label attempt=$attempt status=failed exit_code=$status"
    if [ "$attempt" -ge 3 ]; then
      return "$status"
    fi
    sleep $((attempt * 10))
  done
}

bootstrap_system_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    retry apt-get-update apt-get update
    retry apt-get-install apt-get install -y --no-install-recommends \
      ca-certificates curl git python3 python3-pip python3-venv
  elif command -v apk >/dev/null 2>&1; then
    retry apk-add apk add --no-cache bash ca-certificates curl git python3 py3-pip
  elif command -v dnf >/dev/null 2>&1; then
    retry dnf-install dnf install -y ca-certificates curl git python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    retry yum-install yum install -y ca-certificates curl git python3 python3-pip
  else
    echo "setup_warning no_supported_package_manager_found"
  fi
}

python_is_supported() {
  python3 - <<'PY'
import sys

raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

install_target() {
  if [ -n "${SYLLIPTOR_WHEEL:-}" ] && [ -f "$SYLLIPTOR_WHEEL" ]; then
    printf '%s\n' "$SYLLIPTOR_WHEEL"
  elif [ -f /installed-agent/sylliptor-source/pyproject.toml ]; then
    printf '%s\n' /installed-agent/sylliptor-source
  else
    printf '%s\n' "${SYLLIPTOR_INSTALL_SPEC:-sylliptor-agent-cli}"
  fi
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  installer_path="$(mktemp "${TMPDIR:-/tmp}/sylliptor-uv-installer.XXXXXX")"
  if ! retry uv-download \
    curl -LsSf https://astral.sh/uv/0.11.23/install.sh -o "$installer_path"; then
    rm -f "$installer_path"
    return 1
  fi
  if ! retry uv-installer sh "$installer_path"; then
    rm -f "$installer_path"
    return 1
  fi
  rm -f "$installer_path"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

install_with_python_venv() {
  target="$1"
  rm -rf /opt/sylliptor-venv
  retry python-venv python3 -m venv /opt/sylliptor-venv || return 1
  retry pip-upgrade \
    /opt/sylliptor-venv/bin/python -m pip install --no-input --upgrade pip || return 1
  retry pip-install-sylliptor \
    /opt/sylliptor-venv/bin/python -m pip install --no-input "$target"
}

install_with_uv_python() {
  target="$1"
  install_uv
  retry uv-python-install uv python install 3.12
  rm -rf /opt/sylliptor-venv
  retry uv-venv uv venv --python 3.12 /opt/sylliptor-venv
  retry uv-pip-install uv pip install --python /opt/sylliptor-venv/bin/python "$target"
}

main() {
  echo "sylliptor_setup started_at=$(date -u +%FT%TZ)"
  bootstrap_system_packages
  target="$(install_target)"
  echo "sylliptor_setup install_target_selected"
  if python_is_supported; then
    if ! install_with_python_venv "$target"; then
      echo "setup_warning python_venv_path_failed_falling_back_to_uv"
      install_with_uv_python "$target"
    fi
  else
    install_with_uv_python "$target"
  fi
  ln -sf /opt/sylliptor-venv/bin/sylliptor /usr/local/bin/sylliptor
  sylliptor --version
  echo "sylliptor_setup finished_at=$(date -u +%FT%TZ)"
}

main "$@"
