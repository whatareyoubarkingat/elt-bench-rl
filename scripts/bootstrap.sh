#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
command -v uv >/dev/null || { printf '%s\n' 'Install uv first (see docs/setup.md).' >&2; exit 1; }
git submodule update --init --recursive
# Default dependencies support DuckDB, CSV/HTTP loading, dbt and Tinker.
# Optional cloud adapters require an explicit `uv sync --locked --extra cloud`.
uv sync --locked --python "${ELT_BENCH_RL_PYTHON:-3.11}"
.venv/bin/python scripts/fetch_trains_data.py
.venv/bin/python scripts/fetch_tokenizer.py
.venv/bin/python scripts/check_tokenizer.py
.venv/bin/python scripts/check_prereqs.py
