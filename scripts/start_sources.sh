#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRBYTE_NODE="airbyte-abctl-control-plane"
KIND_NETWORK="kind"

project_docker() {
  bash "$PROJECT_ROOT/scripts/with_docker.sh" docker "$@"
}

cd "$PROJECT_ROOT"
if [[ -z "${ELT_BENCH_RL_CONTAINER_PROXY+x}" ]]; then
  ELT_BENCH_RL_CONTAINER_PROXY="$(project_docker info --format '{{.HTTPSProxy}}')"
  export ELT_BENCH_RL_CONTAINER_PROXY
fi
project_docker compose -f "$PROJECT_ROOT/configs/sources.compose.yml" up -d --build

if ! node_running="$(project_docker inspect --format '{{.State.Running}}' "$AIRBYTE_NODE" 2>/dev/null)" \
    || [[ "$node_running" != "true" ]]; then
  printf '%s\n' \
    'Source service requested. Airbyte Kind node is absent or stopped.' \
    'Install/start Airbyte first, then rerun this script to connect the source API to its Kind network:' \
    "  bash \"$PROJECT_ROOT/scripts/install_airbyte.sh\""
  exit 0
fi

if ! project_docker network inspect "$KIND_NETWORK" >/dev/null 2>&1; then
  printf '%s\n' 'Kind network is unavailable. Install/start Airbyte, then rerun this script.'
  exit 0
fi

api_container_id="$(project_docker compose -f "$PROJECT_ROOT/configs/sources.compose.yml" ps -q elt-api)"
if [[ -z "$api_container_id" ]]; then
  printf '%s\n' 'The source API container is unavailable after compose up.' >&2
  exit 1
fi
api_connected="$(project_docker inspect \
  --format "{{if index .NetworkSettings.Networks \"$KIND_NETWORK\"}}true{{else}}false{{end}}" \
  "$api_container_id")"
if [[ "$api_connected" == "true" ]]; then
  printf 'Source API is already connected to %s.\n' "$KIND_NETWORK"
else
  project_docker network connect --alias elt-api "$KIND_NETWORK" "$api_container_id"
  printf 'Connected source API to %s as elt-api.\n' "$KIND_NETWORK"
fi
