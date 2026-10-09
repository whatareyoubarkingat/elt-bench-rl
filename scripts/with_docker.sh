#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_CONTEXT="${ELT_BENCH_RL_DOCKER_CONTEXT:-colima-elt-bench-rl}"
PROJECT_DOCKER_CONFIG="$PROJECT_ROOT/.local/docker"
mkdir -p "$PROJECT_DOCKER_CONFIG"
if [[ ! -f "$PROJECT_DOCKER_CONFIG/config.json" ]]; then
  cat > "$PROJECT_DOCKER_CONFIG/config.json" <<'JSON'
{"cliPluginsExtraDirs":["/opt/homebrew/lib/docker/cli-plugins"]}
JSON
fi
PROJECT_DOCKER_ENDPOINT="$(docker context inspect "$PROJECT_CONTEXT" --format '{{.Endpoints.docker.Host}}')"
# Use a project-specific daemon/config without switching the user's default context.
unset DOCKER_CONTEXT
export DOCKER_HOST="$PROJECT_DOCKER_ENDPOINT"
export DOCKER_CONFIG="$PROJECT_DOCKER_CONFIG"
export PATH="/opt/homebrew/opt/libpq/bin:$PROJECT_ROOT/.venv/bin:$PATH"
if [[ $# -eq 0 ]]; then
  printf '%s\n' 'Usage: scripts/with_docker.sh <command> [args...]' >&2
  exit 2
fi
exec "$@"
