# Configuration

- `.env.example`: Tinker configuration and commented optional cloud fields;
  never commit a filled copy. Local execution needs no credentials.
- `train.yaml`: bounded one-step Qwen3.5-4B defaults.
- `upstream-revisions.json`: benchmark/Cookbook/dataset/tokenizer pins.
- `airbyte-trains.yaml`: optional public declarative `cars` stream, using `elt-api`
  inside Airbyte's Kind network.
- `sources.compose.yml`: optional cloud source service with source-only mounts,
  no verifier or secrets.

The default route uses DuckDB, `LocalIngestor` for CSV/HTTP extraction and
loading, and dbt-duckdb transformations. The CLI starts an in-process local
HTTP source server for DuckDB execution commands; Docker and Airbyte are unnecessary.
`bash scripts/bootstrap.sh` installs only the default route. Prerequisites can
be checked with `.venv/bin/python scripts/check_prereqs.py --json`.

Cloud Python packages are available explicitly through
`uv sync --locked --extra cloud`. To inspect those optional packages and tools,
use `scripts/check_prereqs.py --check-cloud`; their absence never fails the
default local readiness check. Cloud credentials are only reported as presence
booleans, and no authentication or paid request is performed by that check.

Model-visible loading choices contain source names and mode, not credentials
or connection IDs. Offline CSV ingestion uses pinned bytes with provenance
in `data/manifest.json`; HTTP ingestion makes an actual request. The local
loader never claims to perform an Airbyte sync.
