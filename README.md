# ELT-Bench RLVR

Stateful execution-graded ELT tasks with real CSV/HTTP extraction, dbt SQL,
clean pipeline replay, and a [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
RL recipe. Local execution uses DuckDB; the credentialed path uses Databricks
and Airbyte. Upstream revisions and Python dependencies are pinned.

## Run without credentials

```bash
git clone --recurse-submodules https://github.com/whatareyoubarkingat/elt-bench-rl.git
cd elt-bench-rl
uv sync --locked --group dev --python 3.11
uv run --locked pytest -q tests
uv run --locked python scripts/fetch_trains_data.py
uv run --locked elt-rl local-smoke --output runs/local-smoke
```

Tests use committed synthetic fixtures and start their own source HTTP server.
The smoke command uses pinned official `trains` source bytes and Databricks
reference data, repairs a real dbt error, submits a frozen pipeline, and
replays it in another empty DuckDB database. Neither command calls Tinker or
requires Docker. The scripted demonstration is **not** a model-generated
credentialed official rollout.

## Acceptance status

| Requirement | Status |
| --- | --- |
| Credential-free integration: extraction, dbt, feedback and replay | Passed locally |
| Different execution rewards for correct and incorrect pipelines | Passed: 1.0 vs 0.628571 |
| Cookbook environment, reward and optimizer integration | Offline tests passed; injected clients are labeled |
| Model-generated official Databricks rollout with actual Airbyte sync | Not run: needs Databricks credentials and a cloud budget |
| Real Tinker optimization with nonzero advantages and model change | Not run: paid calls deliberately deferred |

See [design](docs/design.md), [run commands](docs/running.md),
[setup](docs/setup.md), and [validation boundaries](docs/acceptance.md).

## Environment

Tools: `read_task`, `inspect_source`, `configure_load`, `execute_load`,
`list_tables`, `query`, `write_model`, `run_dbt`, `submit_pipeline`.

Each attempt starts empty. Tools execute sequentially. Raw data is writable only
by the loader. Queries access only this episode; model files accept SQL and
controlled literal `source`/`ref` references. Submission freezes logical loads
and SQL, excludes credentials/runtime IDs, then executes from scratch in a new
namespace. Every exit cleans owned warehouse resources.

Terminal reward: `0.2 × loading + 0.5 × target + 0.3 × strict_replay`.
Missing submissions receive zero. Loading checks receipts, schema and content;
target reward requires every source to load correctly. Key-aligned cell scores
penalize missing/extra rows/columns and duplicate keys. Fresh official Stage-1/2
metrics are reported separately from the stronger reward rules.

## Cloud commands (after choosing a budget)

```bash
cp -n configs/.env.example .env
# Fill Databricks OAuth M2M, Airbyte and Tinker fields locally.
bash scripts/start_sources.sh
uv run --locked python scripts/fetch_tokenizer.py
uv run --locked elt-rl preflight --destination databricks --online --allow-paid
uv run --locked elt-rl rollout --destination databricks --allow-paid --output runs/official
uv run --locked elt-rl train --destination duckdb --config configs/train.yaml --allow-paid --output runs/rl-one-step
```

The recipe reuses Cookbook rollouts, trajectory assembly, assistant token masks,
reward-minus-group-mean advantages, `importance_sampling`, optimizer steps and
checkpoint helpers. Defaults: Qwen3.5-4B, rank 16, four independent attempts,
serial execution, one update. Constant groups are resampled at most twice;
zero effective advantage means no update. Fixed action-token logprobs must
change after optimization before success is reported. No local GPU or model
weights are required.

## Layout

```text
src/elt_bench_rl/   Tasks, destinations, loaders, dbt, verifier, replay, RL recipe, CLI
tests/             Offline fixtures, integration, safety, mock cloud/optimizer tests
examples/          Human-written trains SQL demonstration
configs/           Credential templates, training defaults, services, upstream pins
scripts/           Locked installation and public data/tokenizer setup
docs/              Implementation, operations and acceptance evidence
third_party/       Pinned ELT-Bench and Tinker Cookbook submodules
```

Secrets, downloads, databases, tokenizers, logs and checkpoints are ignored.
The [assignment](https://docs.google.com/document/d/1Rr9Y7wG93Fw-vEHgGMuALsIKZMyXO5zRPhlo4NtvrzQ/edit)
requires a real credentialed rollout and a real RL update; these remain explicit
future acceptance checks, not claims based on mocks.
