#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$PROJECT_ROOT/scripts/with_docker.sh"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PROXY_CHECKER="$PROJECT_ROOT/scripts/configure_kind_proxy.py"
LOG_DIR="$PROJECT_ROOT/.local"
INSTALL_LOG="$LOG_DIR/airbyte-install.log"
STATUS_FILE="$LOG_DIR/kind-proxy-status.json"
AIRBYTE_NODE="airbyte-abctl-control-plane"
INSTALL_PID=""
STATUS_TEMP=""

cleanup() {
  if [[ -n "$STATUS_TEMP" ]]; then
    rm -f "$STATUS_TEMP"
    STATUS_TEMP=""
  fi
  if [[ -n "$INSTALL_PID" ]]; then
    local child_pid="$INSTALL_PID"
    INSTALL_PID=""
    if kill -0 "$child_pid" 2>/dev/null; then
      kill -TERM "$child_pid" 2>/dev/null || true
      local attempt
      for attempt in 1 2 3 4 5; do
        if ! kill -0 "$child_pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "$child_pid" 2>/dev/null; then
        kill -KILL "$child_pid" 2>/dev/null || true
      fi
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
}

fail() {
  local code="$1"
  printf 'Airbyte setup failed (exit %s). Private diagnostic log: %s\n' "$code" "$INSTALL_LOG" >&2
  exit "$code"
}

trap cleanup EXIT
trap 'fail 130' INT
trap 'fail 143' TERM

mkdir -p "$LOG_DIR"
touch "$INSTALL_LOG"
chmod 600 "$INSTALL_LOG"
if [[ ! -x "$PYTHON" || ! -f "$WRAPPER" || ! -f "$PROXY_CHECKER" ]]; then
  printf '%s\n' 'Project Python, Docker wrapper, or Kind proxy checker is missing.' >> "$INSTALL_LOG"
  fail 2
fi
cd "$PROJECT_ROOT"

printf '\n[%s] Installing Airbyte chart 2.4.0\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$INSTALL_LOG"
DO_NOT_TRACK=1 bash "$WRAPPER" abctl local install \
  --chart-version 2.4.0 --host localhost --port 8000 --no-browser \
  >> "$INSTALL_LOG" 2>&1 &
INSTALL_PID=$!

while kill -0 "$INSTALL_PID" 2>/dev/null; do
  STATUS_TEMP="$(mktemp "$LOG_DIR/kind-proxy-status.XXXXXX")"
  if "$PYTHON" "$PROXY_CHECKER" > "$STATUS_TEMP" 2>> "$INSTALL_LOG"; then
    checker_code=0
  else
    checker_code=$?
  fi
  mv -f "$STATUS_TEMP" "$STATUS_FILE"
  STATUS_TEMP=""
  if [[ "$checker_code" -ne 0 ]]; then
    fail "$checker_code"
  fi
  if decision="$("$PYTHON" - "$STATUS_FILE" 2>> "$INSTALL_LOG" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text())
status, reason = value.get("status"), value.get("reason")
if status in {"configured", "ready"}:
    print("done")
elif status == "skipped" and reason in {"node_unavailable", "node_not_running"}:
    print("wait")
elif status == "skipped" and reason in {
    "no_server_proxy", "original_proxy_not_loopback",
    "original_proxy_matches_daemon", "environment_files_present",
}:
    print("done")
else:
    raise SystemExit(2)
PY
  )"; then
    if [[ "$decision" == "done" ]]; then
      break
    fi
  else
    fail "$?"
  fi
  sleep 10
done

# Airbyte 2.4.0's PostgreSQL pod uses UID/GID 70 on this one static PV.
# Check every path component before mkdir; never follow a symlink or recurse.
if node_running="$(bash "$WRAPPER" docker inspect --format '{{.State.Running}}' "$AIRBYTE_NODE" 2>> "$INSTALL_LOG")" \
    && [[ "$node_running" == "true" ]]; then
  if bash "$WRAPPER" docker exec "$AIRBYTE_NODE" sh -c '
    set -eu
    for component in / /var /var/local-path-provisioner /var/local-path-provisioner/airbyte-volume-db; do
      test ! -L "$component"
    done
    mkdir -p /var/local-path-provisioner/airbyte-volume-db
    test ! -L /var/local-path-provisioner/airbyte-volume-db
    chown 70:70 /var/local-path-provisioner/airbyte-volume-db
    chmod 700 /var/local-path-provisioner/airbyte-volume-db
  ' >> "$INSTALL_LOG" 2>&1; then
    :
  else
    fail "$?"
  fi
fi

if wait "$INSTALL_PID"; then
  install_code=0
else
  install_code=$?
fi
INSTALL_PID=""
if [[ "$install_code" -ne 0 ]]; then
  fail "$install_code"
fi
printf 'Airbyte install completed. Private diagnostic log: %s\n' "$INSTALL_LOG"
