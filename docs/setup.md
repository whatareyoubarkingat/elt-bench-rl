# Local setup on macOS

The default route is DuckDB + Python CSV/HTTP extraction + dbt-duckdb.
Tinker hosts model sampling/training remotely. No local GPU, model weights,
Docker, Airbyte, Databricks or Snowflake account is required for local execution.
Real Tinker calls still require an API key and an agreed spending limit.

## Install and check

Install `uv` and Git, then run from the repository root:

```bash
bash scripts/bootstrap.sh
uv sync --locked --group dev --python 3.11
uv run --locked pytest -q tests
uv run --locked elt-rl local-smoke --output runs/local-smoke
```

Bootstrap initializes pinned submodules, installs the locked Python 3.11
runtime, fetches selected public source/reference data and the tokenizer, and
checks local dependencies. It makes no paid Tinker calls. `Brewfile` lists
optional legacy service tools; `brew bundle` is not a default prerequisite.

```bash
.venv/bin/python scripts/check_prereqs.py
.venv/bin/python scripts/check_prereqs.py --json
uv run --locked elt-rl preflight --destination duckdb --online
```

The checker imports the local/training code, executes CPU Torch and DuckDB
operations, and verifies upstream revisions and asset hashes. Missing Tinker
credentials are reported separately and do not fail credential-free local
readiness. It does not contact Tinker or inspect Docker/Airbyte by default.
`--check-cloud` enables optional legacy checks; these cannot fail local readiness.
The CLI's online DuckDB preflight opens and cleans a local database only.

DuckDB rollout, training and replay commands start their own ephemeral Python
HTTP source server and close it on exit. CSV/HTTP loading is real execution,
not simulated Airbyte synchronization. Use `--source-api URL` only when an
existing source server is wanted.

## Pinned official assets

| Component | Fixed revision |
| --- | --- |
| ELT-Bench | `fcf3129df49ce5ce63ff46b2bbbbba1eaf9ec055` |
| Tinker Cookbook | `cd008aacedbf0e6f7452b3ec106a345d29482ea5` |
| `tttjjj/elt_bench` data | `52a48541547a4340e19e54386350740acca80146` |
| Qwen3.5-4B tokenizer | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

| Local asset | Rows |
| --- | ---: |
| `data/trains/train.csv` | 20 |
| `data/trains/cars.csv` | 63 |
| `ground_truth/gt_databricks/trains/trains.csv` | 20 |

`scripts/fetch_trains_data.py` verifies SHA-256 hashes and row counts and records
provenance in `data/manifest.json`. Ground truth stays private to the verifier.
The `databricks` task/reference folder identifies the upstream variant, not a
runtime warehouse requirement. The SQL dialect is adapted to DuckDB; the
source data, target reference and pinned official comparison rules are retained.
This is not an upstream-native DuckDB destination.

The tokenizer cache contains config/template/tokenizer files (about 23 MB),
not model weights. `scripts/check_tokenizer.py` tests Cookbook formatting
without Tinker requests. Downloads, `.env`, databases, logs and checkpoints
are ignored by Git.

## Tinker credentials

```bash
cp -n configs/.env.example .env
# Fill TINKER_API_KEY locally; keep an existing .env intact.
```

Only `TINKER_API_KEY` is needed for the default real model path. Key presence
alone does not prove account access, balance or successful training. See
[running.md](running.md) for model rollout and one-step RL commands and
[acceptance.md](acceptance.md) for the remaining grading boundaries.

## Optional cloud extension

Cloud adapters and service scripts remain available for future warehouse
extensions. Install their drivers explicitly with:

```bash
uv sync --locked --extra cloud
.venv/bin/python scripts/check_prereqs.py --check-cloud
```

They are not used by the default route. Existing Colima/Airbyte containers and
volumes have been preserved; this migration does not stop or delete them.
Historical service repair/probe outcomes are recorded in
[acceptance.md](acceptance.md) and [airbyte-seed-repair.md](airbyte-seed-repair.md).
