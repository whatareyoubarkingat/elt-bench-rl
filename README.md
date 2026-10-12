# ELT-Bench RLVR

Stateful execution-graded ELT tasks with real CSV/HTTP extraction, dbt SQL,
clean pipeline replay, and a [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
RL recipe. The default path uses local DuckDB, a Python CSV/HTTP loader and
dbt-duckdb. Tinker provides remote model sampling and training. No Databricks,
Airbyte, Docker, cloud warehouse account or local GPU is needed for this path.
Upstream revisions and Python dependencies are pinned.

## Run without credentials

```bash
git clone https://github.com/whatareyoubarkingat/elt-bench-rl.git
cd elt-bench-rl
python3 scripts/fetch_upstreams.py
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
| Real Tinker credential check | Passed: `KEY_OK`, 18 input tokens and 3 output tokens; access only |
| Model-generated rollout on the official `trains` task adapted to DuckDB | Two real training rollouts earned 1.0 with strict fresh replay |
| Real Tinker optimization with nonzero advantages and model change | Passed: one actual optimizer step, fresh checkpoint restore and changed fixed-token logprobs |

The first three standalone rollouts scored 0, 0 and 0.6285714286; the training
group scored [0.6285714286, 1, 0, 1]. Training completed its optimizer step,
then exited during same-client `LoadWeights` verification at sequence 7.
Separate real-API recovery verified the saved update through a fresh model
with optimizer-state restoration, without repeating optimization. On 814 fixed
action tokens, the maximum logprob change was 0.19415616989135742, above 1e-8.
The original failed run remains recorded separately. The sole post-update
evaluation submitted at 13 turns/2079 generated tokens with reward
0.6285714286 (L=1, T=6/7, R=0); general quality improvement is not established.
See [sanitized real-experiment evidence](docs/evidence/2026-10-12-real-experiment).
Final validation: 262 full-suite tests (47.36s), 35 focused training tests
(3.91s), and 39 files checked by Ruff/format. Test counts overlap.

See the [concise English report](docs/report.md), [design](docs/design.md), [run commands](docs/running.md),
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

## Model rollout and training (after choosing a Tinker budget)

```bash
cp -n configs/.env.example .env
# Fill only TINKER_API_KEY locally.
uv run --locked python scripts/fetch_tokenizer.py
uv run --locked elt-rl preflight --destination duckdb --online
RUN_ROOT=runs/real-experiment-20261012
uv run --locked elt-rl rollout --destination duckdb --allow-paid \
  --budget-ledger "$RUN_ROOT/budget.json" --max-cost-usd 10 \
  --output "$RUN_ROOT/rollout"
uv run --locked elt-rl train --destination duckdb --config configs/train.yaml \
  --allow-paid --budget-ledger "$RUN_ROOT/budget.json" --max-cost-usd 10 \
  --output "$RUN_ROOT/train"
```

Share the same ledger across rollout, training and recovery commands. It reserves
each remote request before dispatch and keeps uncertain requests charged against
the estimate. The $10 limit includes a $0.50 storage buffer and $0.000008955
estimated prior key-check cost. These are uncached rate-card estimates, not an
invoice or an account-level spending cap. Checkpoints currently have no TTL;
their ongoing provider storage is billed separately from token reservations.
Recorded checkpoint metadata lists four files totaling 1,167,002,514 bytes with
no TTL, approximately $0.1167/month at the published $0.10/GB/month rate.
The final uncached token estimate is $0.448166396; including the fixed storage
buffer, the ledger accounts for $0.948166396. These are not invoice amounts;
ongoing storage billing remains separate.
See [budget and checkpoint details](docs/running.md#shared-budget-and-checkpoint-storage).

The recipe reuses Cookbook rollouts, trajectory assembly, assistant token masks,
reward-minus-group-mean advantages, `importance_sampling`, optimizer steps and
checkpoint helpers. Defaults: Qwen3.5-4B, rank 16, four independent attempts,
serial execution, one update, 24 turns and 8192 sampled tokens per attempt.
Constant groups are resampled at most twice;
zero effective advantage means no update. Fixed action-token logprobs must
change after optimization before success is reported. No local GPU or model
weights are required.

For continuous local supervision, wrap each command in
`scripts/monitor_experiment.py`, with matching watchdog `--run-dir` and CLI
`--output`. It writes private PID/status evidence, stdout logs and an
auto-refresh dashboard; default heartbeats are 15 seconds, stall timeout
900 seconds and overall timeout 3600 seconds. Its own heartbeat files never
count as experiment progress. [Monitoring commands](docs/running.md#watchdog-and-progress-evidence)
include the shorter rollout timeout and HTTP proxy fallback.

DuckDB commands start and stop their own local HTTP source automatically.
The assignment explicitly requires Tinker, but does not mandate Databricks or
Airbyte. DuckDB uses the pinned official task inputs and reference data with
an adapted SQL dialect; it is not an upstream-supported warehouse destination.
The interpretation of the assignment's credentialed warehouse rollout should
be confirmed with the grader before claiming that gate passed. Existing cloud
adapters remain optional extensions (`uv sync --locked --extra cloud`).

## Layout

```text
src/elt_bench_rl/   Tasks, destinations, loaders, dbt, verifier, replay, RL recipe, CLI
tests/             Offline fixtures, integration, safety, mock cloud/optimizer tests
examples/          Human-written trains SQL demonstration
configs/           Credential templates, training defaults, services, upstream pins
scripts/           Locked installation and public data/tokenizer setup
docs/              Implementation, operations and acceptance evidence
External cache     Pinned upstream checkouts, outside the project repository
```

Secrets, downloads, databases, tokenizers, logs and checkpoints are ignored.
Project code, comments, prompts and documentation use English. Fixed upstream
source checkouts live under `~/.cache/elt-bench-rl/upstreams` (override with
`ELT_BENCH_RL_UPSTREAM_CACHE`); their multilingual examples are outside this
repository. Cookbook is installed from its exact Git revision through `uv`.
The [assignment](https://docs.google.com/document/d/1Rr9Y7wG93Fw-vEHgGMuALsIKZMyXO5zRPhlo4NtvrzQ/edit)
requires a real credentialed rollout and a real RL update. The successful key
check establishes access, and two model-generated pipelines now pass strict
execution/replay checks. One real RL update is verified. The single post-update
evaluation was partially correct; this does not establish general improvement.
Grader acceptance of the adapted warehouse remains unconfirmed, so the entire
assignment is not declared accepted.
