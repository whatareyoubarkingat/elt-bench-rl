# Running and collecting evidence

Use the locked environment from the repository root. Commands load `.env`
without printing it. Keep secrets out of CLI arguments and public logs.

## Local execution

```bash
python3 scripts/fetch_upstreams.py
uv sync --locked --group dev --python 3.11
uv run --locked pytest -q tests
uv run --locked python scripts/fetch_trains_data.py --destination databricks
uv run --locked elt-rl local-smoke --output runs/local-smoke
uv run --locked elt-rl replay --bundle runs/local-smoke/episode_<id>/pipeline.json --output runs/replay
```

All DuckDB execution commands start and close an ephemeral Python HTTP source
server by default. No separate service or Docker deployment is needed. Set
`--source-api` only to use an existing server. Each episode retains tool trace, dbt feedback,
frozen pipeline/digest and original/replay grades. Warehouse files are removed.
`examples/trains.sql` is human-written and never injected into model prompts.

## Model-generated official-task rollout on DuckDB

Set `TINKER_API_KEY` in `.env` and choose a Tinker spending limit first:

```bash
uv run --locked python scripts/fetch_tokenizer.py
uv run --locked elt-rl preflight --destination duckdb --online
RUN_ROOT=runs/real-experiment-20261012
uv run --locked elt-rl rollout --destination duckdb --allow-paid \
  --budget-ledger "$RUN_ROOT/budget.json" --max-cost-usd 10 \
  --output "$RUN_ROOT/rollout"
```

The minimal real Tinker credential check returned `KEY_OK` with 18 input tokens
and 3 output tokens. It confirms sampling access only. Formal rollout and
optimization results are pending. DuckDB `preflight --online` checks the local
warehouse and credential presence; it does not itself authenticate with Tinker.

Sampling uses real Tinker credentials; extraction/loading, dbt execution and
clean replay run locally. Preserve model tool trajectories, load receipts,
original/replay grades and the bundle. The official `trains` inputs and target
reference are unchanged, but DuckDB is an adapted destination. The assignment
does not mandate Databricks or Airbyte; acceptance of a Tinker-credentialed
local warehouse rollout should be confirmed with the grader.

## Optional Databricks and Airbyte extension

These services are not needed for the default route. Install optional drivers
with `uv sync --locked --extra cloud` before using this extension.

Use a SQL Warehouse and Unity Catalog. The OAuth M2M principal needs catalog
use, episode schema creation/use, tables and the selected Airbyte connector's
staging volume permissions. Restrict it to a training catalog. Free Edition
compatibility is not assumed.

Fill `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_CATALOG`,
`DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `AIRBYTE_URL`,
`AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, `AIRBYTE_WORKSPACE_ID` in `.env`.
Destination configuration is connector-version-specific: validate against
the actual Airbyte definition/spec before cloud sync.

```bash
bash scripts/install_airbyte.sh  # only if not installed
bash scripts/start_sources.sh
uv run --locked elt-rl preflight --destination databricks
# The next commands may start billed warehouse / Tinker sessions.
uv run --locked elt-rl preflight --destination databricks --online --allow-paid
uv run --locked python scripts/fetch_tokenizer.py
uv run --locked elt-rl rollout --destination databricks --allow-paid \
  --budget-ledger runs/optional-cloud/budget.json --max-cost-usd 10 \
  --output runs/optional-cloud/rollout
```

Credential presence is not authenticated access. This optional cloud validation requires
a model-generated trajectory, actual successful Airbyte jobs for both sources,
Databricks execution, fresh two-stage metrics, a bundle and clean replay.
Health/check/discovery alone is insufficient. Preserve job IDs in trusted
receipts, not bundles. Do not publish secrets in connector manifests.

## Real one-step RL

Choose a spending limit and set `TINKER_API_KEY`, then:

```bash
uv run --locked elt-rl train --destination duckdb \
  --config configs/train.yaml --allow-paid \
  --budget-ledger runs/real-experiment-20261012/budget.json --max-cost-usd 10 \
  --output runs/real-experiment-20261012/train
```

The SDK performs inference/training on remote GPUs. The Mac hosts ELT execution
and tokenizers; model weights remain remote. Defaults are Qwen/Qwen3.5-4B, one
optimization step, four serial attempts, 24 turns and 8192 sampled tokens per
attempt. Constant groups can require twelve attempts (three groups of four),
checkpoints, two logprob probes, and one optimizer step. The shared dollar
ledger bounds request estimates as described below; token limits also bound
generation.

Required evidence: generated trajectories/logprobs, execution rewards,
nonzero advantages, unmasked effective action tokens, awaited optimizer
responses, state/sampler checkpoints and a changed fixed action-token logprob
probe. Injected runtimes are explicitly labeled. `no_learning_signal` and
`update_unverified` do not pass effective-training acceptance.

Use the sampler checkpoint with a Tinker sampling client; load the state
checkpoint into a compatible LoRA training client to restore optimizer state.
Keep paths and `training_result.json` with the revision/recipe manifest. Verify
model availability and account access when executing with the real account.

To resume, add `load_checkpoint_path: tinker://...` to a training YAML. To
evaluate a trained policy, add `sampler_checkpoint_path: tinker://...` to a
rollout YAML and pass `--config`; state and sampler paths have different roles.

## Shared budget and checkpoint storage

Pass the same `--budget-ledger` path and `--max-cost-usd 10` to every sampling,
training and recovery command belonging to this experiment. Without an explicit
path, each command uses its own output directory's `budget.json`, which would
not provide a shared experiment limit. Do not replace or reset the ledger to
retry a failed request. Existing ledger settings are immutable.

Before dispatch, the locked ledger reserves input tokens and maximum output
tokens, or sequence tokens for training. Successful requests settle to observed
token counts. Exceptions, timeouts and uncertain remote outcomes keep their
reservation; terminating a local client does not prove that server work was
unbilled. Pending reservations also survive process failure. The ledger refuses
new requests that would exceed its limit.

The $10 estimate includes a $0.50 storage buffer and $0.000008955 for the earlier
18-input/3-output key check. The encoded Qwen3.5-4B rates are $0.33 input,
$1.005 output and $0.737 training per million tokens. Cached-prefill discounts
are not assumed. The ledger is an uncached estimate, not an invoice, provider
account balance, or a cap on other account activity. Rates and billed usage
should be checked against [Tinker's pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).
Optional warehouse charges are outside this Tinker token ledger.

The recipe saves initial and updated state/sampler checkpoints without a TTL.
Tinker retains these until deletion and bills their actual storage over time;
the published storage rate is $0.10/GB/month, checked on 2026-10-12.
The fixed $0.50 buffer is not a storage measurement or a limit on ongoing
charges. Local process exit and warehouse cleanup do not delete remote
checkpoints. Review actual storage and retention separately, keeping the
evaluation checkpoints needed for evidence. The recipe applies no automatic
TTL or checkpoint deletion. See [Tinker checkpoint management](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/).

## Watchdog and progress evidence

Run from the repository root and give the watchdog and experiment the same
output directory. This example supervises a rollout; use `train`,
`--config configs/train.yaml` and `--timeout 3600` for the optimization phase.
Both phases must reuse the same ledger.

```bash
RUN_ROOT=runs/real-experiment-20261012
.venv/bin/python scripts/monitor_experiment.py \
  --run-dir "$RUN_ROOT/rollout" --heartbeat 15 \
  --stall-timeout 900 --timeout 1200 --grace 10 -- \
  .venv/bin/python -m elt_bench_rl.cli rollout --destination duckdb \
  --allow-paid --budget-ledger "$RUN_ROOT/budget.json" --max-cost-usd 10 \
  --output "$RUN_ROOT/rollout"
```

The run directory is mode 0700; watchdog logs, JSON status/history and dashboard
are mode 0600. `stdout.log` merges child stdout/stderr.
`monitor-status.json` records child PID, owned process-group ID, start time,
elapsed time, artifact metadata, exit code and termination reason.
`monitor-heartbeats.jsonl` retains the status history; `monitor.html` refreshes
every 15 seconds for local screenshots. Command arguments and environment
values are omitted from watchdog metadata. Treat raw logs, checkpoint paths
and model traces as private until sanitized for publication.

The watcher observes experiment log changes, `progress.jsonl`, manifests,
metrics, tool/dbt/grade artifacts and child CPU time. Its own heartbeat/status
files and dashboard never count as progress. Running PID, CPU activity or an
API wait event establishes activity, not a completed ELT task or RL update;
inspect durable turn events, execution grades and final results for that claim.
The recipe flushes turn/phase evidence and bounds remote waits, normally
300 seconds per remote call. Local supervision has a 900-second stall limit
and separate phase wall limit. On timeout/interruption it sends TERM only to
its newly created process group, then KILL after the grace period if needed.
It preserves artifacts; it does not stop other processes or cancel account-wide
server work. Exit codes are the child code, 124 for overall timeout, 125 for
stall, 130 for interruption and 127 for launch failure.

HTTP proxy fallback is confined to the experiment process. When both
`HTTP_PROXY` and `HTTPS_PROXY` use HTTP(S) routes, the CLI removes inherited
SOCKS `ALL_PROXY`/`all_proxy` values. The watchdog removes those two generic
proxy variables before launch while preserving HTTP(S) proxy and `NO_PROXY`
settings. It does not change system proxy configuration. Local source HTTP
requests bypass inherited proxies. Keep proxy credentials in local configuration,
not command arguments or screenshots.

Offline tests use synthetic sources and injected model/optimizer clients. They
make no paid Tinker calls and do not establish credentialed experiment results.

## Failures and cleanup

Infrastructure errors stop unusable samples; invalid SQL returns repairable
feedback. Episode/group cleanup runs on failure. Incomplete cloud cleanup writes
a private mode-0600 `cleanup_pending.json` containing owned namespace/schema/
actor/job identities for recovery after process exit; it excludes connection
configurations and credentials. Successful same-process retry removes the pending
record. Never delete unrelated schemas/jobs. Airbyte jobs/connections are cleaned
before warehouse schemas.
